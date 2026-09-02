#!/usr/bin/env python3
"""addon.stack-bench local-service entry (http-json on 127.0.0.1:4889).

ResonantOS add-on contract: protocol http-json, healthCommand stackbench.status.
Wraps the FROZEN vendored stack_bench module in-process (no subprocess, no
shell, no secrets on argv). API key is read from the STACKBENCH_API_KEY
environment variable only. Benchmark target must be loopback unless the
operator explicitly sets STACKBENCH_ALLOW_REMOTE=1.

All persisted output is home-path-redacted before acknowledgement (real
llama-server /props often reports model_path under $HOME; redaction, not
failure — review finding C1).

Exit codes: 0 normal stop; 78 port bind failure.
"""

import json
import os
import re
import socket
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import stack_bench  # noqa: E402  (vendored, byte-identical, hash-pinned by tests)

PORT = int(os.environ.get("STACKBENCH_PORT", "4889"))  # dev override; manifest port 4889 is the contract
ALLOW_REMOTE = os.environ.get("STACKBENCH_ALLOW_REMOTE", "") == "1"
API_KEY_ENV = "STACKBENCH_API_KEY"
MAX_BODY = 64 * 1024
MAX_STR = 2048
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")  # ASCII, matches the manifest pattern

ADDON_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(ADDON_ROOT, "var")

_state = {
    "busy": False,
    "last_run_id": None,
    "runs": {},  # run_id -> {"state": ..., "summary": ..., "summary_path": ...}
}
_lock = threading.Lock()


def _validate_run_params(params):
    if not isinstance(params, dict):
        return None, "params must be an object"
    url = params.get("endpoint_url")
    if not isinstance(url, str) or not (0 < len(url) <= MAX_STR):
        return None, "endpoint_url must be a non-empty string"
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):  # control chars never valid here (ultraQA finding)
        return None, "endpoint_url contains control characters"
    try:
        parts = urlsplit(url)
    except ValueError:
        return None, "endpoint_url is not a valid URL"
    if parts.scheme not in ("http", "https"):
        return None, "endpoint_url must be http(s)"
    host = parts.hostname or ""
    if host not in LOOPBACK_HOSTS and not ALLOW_REMOTE:
        return None, "endpoint_url must be loopback (127.0.0.1/localhost); set STACKBENCH_ALLOW_REMOTE=1 to override"
    run_id = params.get("run_id")
    if run_id is None:
        run_id = "run-" + uuid.uuid4().hex[:12]
    if not isinstance(run_id, str) or len(run_id) > 64 or not RUN_ID_RE.match(run_id) or run_id.startswith("."):
        return None, "run_id may only contain ASCII letters, digits, dot, underscore, hyphen"
    targets = params.get("prompt_tokens")
    if targets is not None:
        if not isinstance(targets, list) or len(targets) > 8 or len(targets) == 0:
            return None, "prompt_tokens must be a non-empty array of at most 8 integers"
        if any(not isinstance(t, int) or isinstance(t, bool) or not (1 <= t <= 200000) for t in targets):
            return None, "prompt_tokens values must be integers in 1..200000"
    rounds = params.get("rounds")
    if rounds is not None and (not isinstance(rounds, int) or isinstance(rounds, bool) or not (1 <= rounds <= 10)):
        return None, "rounds must be an integer in 1..10"
    for key in params:
        if key not in ("endpoint_url", "run_id", "prompt_tokens", "rounds", "api_key"):
            return None, f"unknown field: {key}"
    if "api_key" in params:
        return None, "api_key is never accepted as a request field; set " + API_KEY_ENV
    return {"url": url, "run_id": run_id, "targets": targets, "reps": rounds or 3}, None


def _redact_text(text):
    home = os.path.expanduser("~")
    return text.replace(home, "~") if home and home != "~" else text


def _redact_obj(obj):
    if isinstance(obj, str):
        return _redact_text(obj)
    if isinstance(obj, list):
        return [_redact_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _redact_obj(value) for key, value in obj.items()}
    return obj


def _redact_persisted(out_dir):
    """Redact home paths in the files run_all already wrote (review C1)."""
    summary = None
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path) as f:
        summary = _redact_obj(json.load(f))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    rows_path = os.path.join(out_dir, "rows.jsonl")
    if os.path.exists(rows_path):
        with open(rows_path) as f:
            lines = f.readlines()
        with open(rows_path, "w") as f:
            for line in lines:
                f.write(_redact_text(line.rstrip("\n")) + "\n")
    return summary


def _execute_run(job):
    url, run_id, targets, reps = job["url"], job["run_id"], job["targets"], job["reps"]
    try:
        # upstream records sys.argv verbatim into summary.env — sanitize around
        # the call; safe because single-flight guarantees one _execute_run
        real_argv = sys.argv
        try:
            sys.argv = ["stackbench", run_id]
            transport = stack_bench.UrllibTransport(url, api_key=os.environ.get(API_KEY_ENV))
            summary, out, rc = stack_bench.run_all(
                url, run_id, targets, reps, OUT_BASE,
                {"tokenize": 600, "completion": 1800},
                transport=transport,
                log=lambda *a: None,  # suppress home-anchored out-path printing
            )
        finally:
            sys.argv = real_argv
        summary = _redact_persisted(out)
        with _lock:
            _state["runs"][run_id] = {
                "state": "done" if rc == 0 else "failed",
                "summary": summary,
                "summary_path": os.path.relpath(os.path.join(out, "summary.json"), ADDON_ROOT),
            }
    except Exception as exc:  # bench errors surface as job failure, never server crash
        with _lock:
            _state["runs"][run_id] = {"state": "failed", "summary": None, "summary_path": None, "error": str(exc)[:300]}
    finally:
        with _lock:
            _state["busy"] = False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30  # ultraQA finding: a lying Content-Length must not pin a thread forever

    def _reply(self, code, payload, close=False):
        if close:
            self.close_connection = True  # review I1: never leave undrained bodies on a keep-alive connection
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if close:
            self.send_header("Connection", "close")  # advertise what the socket is about to do
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._reply(200, self._status())
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/":
            self._reply(404, {"error": "not found"}, close=True)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reply(400, {"error": "bad content-length"}, close=True)
            return
        if length <= 0 or length > MAX_BODY:
            self._reply(413 if length > MAX_BODY else 400, {"error": "body must be 1..65536 bytes"}, close=True)
            return
        try:
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except (TimeoutError, socket.timeout, OSError):
            self._reply(408, {"error": "request body incomplete (timeout)"}, close=True)
            return
        except (ValueError, UnicodeDecodeError):
            self._reply(400, {"error": "body must be valid JSON"}, close=True)
            return
        if not isinstance(req, dict):
            self._reply(400, {"error": "body must be a JSON object"}, close=True)
            return
        method = req.get("method")
        params = req.get("params", {})
        for key in req:
            if key not in ("method", "params"):
                self._reply(400, {"error": f"unknown field: {key}"}, close=True)
                return
        if method == "stackbench.status":
            self._reply(200, self._status())
        elif method == "stackbench.run":
            self._run(params)
        elif method == "stackbench.results":
            self._results(params)
        else:
            self._reply(404, {"error": f"unknown method: {method}"})

    def _status(self):
        with _lock:
            return {
                "ok": True,
                "version": stack_bench.TOOL_VERSION,
                "busy": _state["busy"],
                "last_run_id": _state["last_run_id"],
            }

    def _run(self, params):
        job, err = _validate_run_params(params)
        if err:
            self._reply(400, {"error": err})
            return
        with _lock:
            if _state["busy"]:
                self._reply(409, {"error": "a run is already in progress", "run_id": _state["last_run_id"]})
                return
            _state["busy"] = True
            # review I2: publish the run record in the same critical section, so
            # an immediate stackbench.results poll can never 404
            _state["runs"][job["run_id"]] = {"state": "running", "summary": None, "summary_path": None}
            _state["last_run_id"] = job["run_id"]
        try:
            threading.Thread(target=_execute_run, args=(job,), daemon=True).start()
        except Exception:  # review: never strand busy=True if the thread cannot start
            with _lock:
                _state["busy"] = False
                _state["runs"][job["run_id"]] = {"state": "failed", "summary": None, "summary_path": None, "error": "worker thread failed to start"}
            self._reply(500, {"error": "worker thread failed to start"})
            return
        self._reply(200, {"run_id": job["run_id"], "state": "running"})

    def _results(self, params):
        if not isinstance(params, dict):
            self._reply(400, {"error": "params must be an object"})
            return
        run_id = params.get("run_id")
        if run_id is None:
            with _lock:
                run_id = _state["last_run_id"]
        if not isinstance(run_id, str) or len(run_id) > 64:
            self._reply(400, {"error": "run_id must be a string of at most 64 characters"})
            return
        with _lock:
            record = _state["runs"].get(run_id)
        if record is None:
            self._reply(404, {"error": "unknown run_id"})
            return
        payload = {"run_id": run_id, "state": record["state"], "summary_path": record.get("summary_path")}
        if record.get("error"):
            payload["error"] = record["error"]
        if record.get("summary") is not None:
            payload["summary"] = record["summary"]
        self._reply(200, payload)

    def log_message(self, fmt, *args):  # keep service logs quiet and content-free
        sys.stderr.write("stackbench-service: " + (fmt % args) + "\n")


def main():
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        sys.stderr.write(f"stackbench-service: cannot bind 127.0.0.1:{PORT} ({exc}); manifest entrypoint expects this port\n")
        return 78
    sys.stderr.write(f"stackbench-service: listening on http://127.0.0.1:{PORT}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
