#!/usr/bin/env python3
"""stack-bench — one-command honest local-inference battery.

Measures prefill and decode throughput against any llama-server-compatible
endpoint, with true-token sizing via /tokenize, per-rep distinct prompts,
and a same-request ambient A/B control (throughput numbers ride with
their ambient noise: same-boot reruns are not guaranteed identical).

Deltas vs llama-bench (named prior art, in-tree microbench): server-level
measurement, /tokenize-true prompt sizing, ambient-noise reporting,
reps-with-ranges JSON output. The ignore_eos decode convention is adopted
from llama-bench's tg measurement.

Zero dependencies. Python 3.10+.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

TOOL_VERSION = "0.1.1"
SCHEMA_VERSION = 1

WORDS = ("the manifold derivative encodes gradient information across layers "
         "quantization error accumulates in low-rank projections whereas attention "
         "heads specialize by syntactic role persistent caching amortizes the cost "
         "of repeated evaluation speculative drafting validates tokens in parallel "
         "batches memory bandwidth limits decode throughput on unified architectures "
         "kernel fusion reduces launch overhead for small matrix products the scheduler "
         "reorders requests to maximize batch occupancy under thermal constraints "
         "sparsity in mixture models activates subsets per token routing decisions").split()

BOUNDS = {"reps": (1, 10), "target": (512, 131072), "n_predict": (1, 2048)}


# ---------------------------------------------------------------- transport
class Transport:
    """Owns every HTTP concern. Batteries never touch the wire."""
    def post(self, path, payload, timeout):
        raise NotImplementedError


class UrllibTransport(Transport):
    def __init__(self, base_url, api_key=None):
        self.base = base_url.rstrip("/")
        self.api_key = api_key  # sent as Authorization: Bearer (llama-server --api-key)

    def _request(self, path, payload, timeout, method="POST"):
        url = self.base + path
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(
            url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", "replace")
                return {"ok": True, "status": r.status,
                        "json": _maybe_json(body), "body": body}
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace") if e.fp else ""
            except Exception:  # draining the error body must never crash the run
                body = ""
            return {"ok": False, "status": e.code,
                    "json": _maybe_json(body), "body": body, "http_error": True}
        except Exception as e:  # socket errors, timeouts — bounded by timeout
            return {"ok": False, "status": None, "json": None,
                    "body": "", "error": f"{type(e).__name__}: {e}"[:200]}

    def post(self, path, payload, timeout):
        return self._request(path, payload, timeout)

    def get(self, path, timeout):
        return self._request(path, None, timeout, method="GET")


def _maybe_json(body):
    try:
        return json.loads(body)
    except Exception:
        return None


# ---------------------------------------------------------------- batteries
def count_tokens(transport, text, timeout):
    r = transport.post("/tokenize", {"content": text}, timeout)
    if r["ok"] and isinstance(r["json"], dict):
        # defensive across /tokenize shape variants (old: tokens only;
        # current: tokens + pieces + bpe_offsets)
        toks = r["json"].get("tokens") or []
        if isinstance(toks, list):
            return len(toks)
    return None


def grow_prompt(transport, target, seed, timeout, state=None):
    """Grow a seeded word-soup prompt until /tokenize reports >= target
    tokens. The tokenize-dead latch lives in the shared `state` dict when
    provided (run scope): after the initial /tokenize failure, the WHOLE RUN
    stops calling /tokenize — a hung endpoint costs one timeout total."""
    import random
    rng = random.Random(seed)
    state = state if state is not None else {}
    chunks, text, n = [], "", None
    while True:
        chunks.append(" ".join(rng.choice(WORDS) for _ in range(120)))
        text = ("Document " + str(seed) + ": " + " ".join(chunks)
                + "\n\nSummarize this document in one word.")
        if not state.get("tokenize_dead"):
            n = count_tokens(transport, text, timeout)
            if n is not None:
                if n >= target:
                    return text, n, "tokenized"
                continue
            state["tokenize_dead"] = True  # flagged fallback for the run
        if sum(len(c.split()) for c in chunks) >= target:
            return text, None, "word-approx-fallback (token count unknown)"


def _base_row(battery, **kw):
    row = {"battery": battery}
    row.update(kw)
    return row


def _timed_completion(transport, prompt, n_predict, timeout, battery, meta):
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
               "cache_prompt": False}
    if battery == "decode":
        payload["ignore_eos"] = True  # llama-bench tg convention, credited
    r = transport.post("/completion", payload, timeout)
    row = _base_row(battery, **meta)
    if not r["ok"]:
        row["error"] = {"status": r.get("status"),
                        "body": (r.get("body") or "")[:300],
                        "transport_error": r.get("error")}
        return row, r
    d = r["json"] if isinstance(r["json"], dict) else {}
    t = d.get("timings") or {}
    content = d.get("content") or ""
    row.update({
        "content_sha": hashlib.sha256(content.encode()).hexdigest()[:16] if content else None,
        "stop_type": d.get("stop_type"),
        "tokens_cached": t.get("tokens_cached"),
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": round(t.get("prompt_ms", 0)) if t.get("prompt_ms") is not None else None,
        "prompt_tps": round(t.get("prompt_per_second", 0), 1)
                      if t.get("prompt_per_second") is not None else None,
        "predicted_n": t.get("predicted_n"),
        "predicted_tps": round(t.get("predicted_per_second", 0), 1)
                         if t.get("predicted_per_second") is not None else None,
    })
    if not t:
        row["timings_missing"] = True
    if battery == "decode" and row["predicted_n"] is not None:
        row["early_eos"] = bool(row["predicted_n"] < n_predict - 8)
    return row, r


def run_prefill(transport, targets, reps, timeouts, state, log=print):
    rows = []
    for tgt in targets:
        for rep in range(reps):
            seed = tgt * 10 + rep  # per-rep DISTINCT prompts: normative
            try:
                prompt, n_est, sizing = grow_prompt(transport, tgt, seed,
                                                    timeouts["tokenize"], state)
            except Exception as e:
                rows.append(_base_row("prefill", target_tokens=tgt, rep=rep,
                                      seed=seed,
                                      error={"transport_error": str(e)[:200]}))
                continue
            row, _ = _timed_completion(
                transport, prompt, 4, timeouts["completion"], "prefill",
                {"target_tokens": tgt, "rep": rep, "seed": seed,
                 "sizing": sizing, "sized_tokens": n_est})
            rows.append(row)
            log(f"[prefill {tgt} r{rep}] n={row.get('prompt_n')} "
                f"pps={row.get('prompt_tps')} ({sizing})")
    return rows


def run_decode(transport, target_tokens, reps, timeouts, state,
              n_predict=128, log=print):
    rows = []
    try:
        prompt, n_est, sizing = grow_prompt(transport, target_tokens, 999,
                                            timeouts["tokenize"], state)
    except Exception as e:
        return [_base_row("decode", error={"transport_error": str(e)[:200]})]
    for rep in range(reps):
        row, _ = _timed_completion(
            transport, prompt, n_predict, timeouts["completion"], "decode",
            {"rep": rep, "seed": 999, "sizing": sizing, "sized_tokens": n_est,
             "target_tokens": target_tokens, "n_predict": n_predict})
        rows.append(row)
        log(f"[decode r{rep}] predicted_n={row.get('predicted_n')} "
            f"tps={row.get('predicted_tps')}")
    return rows


def run_ambient(transport, targets, timeouts, state, n_predict_decode=128,
                log=print):
    """Same-request A/B pairs: one pair per prefill target
    plus one decode-shaped pair. Content-hash equality + timing deltas only —
    two samples per pair, informational. Errors surface as errors (sha_equal
    stays None = indeterminate), never as false divergence."""
    pairs, ambient_rows = [], []
    probes = [("prefill", tgt, 4) for tgt in targets] + [("decode", None, n_predict_decode)]
    for kind, tgt, n_predict in probes:
        seed = (tgt or 0) * 10 + 99
        try:
            prompt, n_est, sizing = grow_prompt(transport, tgt or 2048, seed,
                                                timeouts["tokenize"], state)
        except Exception as e:
            pairs.append({"kind": f"{kind}" + (f"@{tgt}" if tgt else ""),
                          "error": str(e)[:200]})
            continue
        row_a, _ = _timed_completion(
            transport, prompt, n_predict, timeouts["completion"], kind,
            {"probe": "a", "target_tokens": tgt, "seed": seed,
             "sizing": sizing, "sized_tokens": n_est})
        row_b, _ = _timed_completion(
            transport, prompt, n_predict, timeouts["completion"], kind,
            {"probe": "b", "target_tokens": tgt, "seed": seed,
             "sizing": sizing, "sized_tokens": n_est})
        ambient_rows.extend([row_a, row_b])
        if "error" in row_a or "error" in row_b:
            err = row_a.get("error") or row_b.get("error")
            pairs.append({"kind": f"{kind}" + (f"@{tgt}" if tgt else ""),
                          "sha_equal": None, "error": err})
            continue
        tps_key = "prompt_tps" if kind == "prefill" else "predicted_tps"
        ta, tb = row_a.get(tps_key), row_b.get(tps_key)
        pairs.append({
            "kind": f"{kind}" + (f"@{tgt}" if tgt else ""),
            "sha_equal": (None if (row_a.get("content_sha") is None
                                   and row_b.get("content_sha") is None)
                          else (row_a.get("content_sha") is not None
                                and row_a.get("content_sha")
                                == row_b.get("content_sha"))),
            f"{tps_key}_a": ta, f"{tps_key}_b": tb,
            "tps_delta": round(abs((ta or 0) - (tb or 0)), 1)
                         if ta is not None and tb is not None else None,
        })
        log(f"[ambient {kind}@{tgt}] sha_equal="
            f"{pairs[-1].get('sha_equal')} delta={pairs[-1].get('tps_delta')}")
    return pairs, ambient_rows


# ---------------------------------------------------------------- summarize
def bucket_of(n):
    return ((n + 4000) // 8000) * 8000


def summarize_prefill(rows):
    by_bucket, errors, missing, fallback = {}, 0, 0, 0
    for r in rows:
        if "error" in r:
            errors += 1
            continue
        if str(r.get("sizing", "")).startswith("word-approx"):
            fallback += 1  # count before the timings-missing skip — both
            # flags can co-occur; the counter must not depend on order
        if r.get("timings_missing"):
            missing += 1
            continue
        if r.get("prompt_tps") is not None and r.get("prompt_n") is not None:
            by_bucket.setdefault(str(bucket_of(r["prompt_n"])), []).append(
                r["prompt_tps"])
    return {
        "rows": len(rows), "errors": errors,
        "timings_missing_rows": missing, "word_approx_rows": fallback,
        "by_measured_prompt_n": {
            k: {"mean_tps": round(sum(v) / len(v), 1), "min": min(v),
                "max": max(v), "n": len(v)}
            for k, v in sorted(by_bucket.items())},
    }


def summarize_decode(rows):
    tps = [r["predicted_tps"] for r in rows
           if "error" not in r and not r.get("timings_missing")
           and r.get("predicted_tps") is not None]
    return {
        "rows": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "timings_missing_rows": sum(1 for r in rows if r.get("timings_missing")),
        "word_approx_rows": sum(1 for r in rows
                                if str(r.get("sizing", "")).startswith("word-approx")),
        "early_eos_rows": sum(1 for r in rows if r.get("early_eos")),
        "decode_tps": ({"mean_tps": round(sum(tps) / len(tps), 1),
                        "min": min(tps), "max": max(tps), "n": len(tps)}
                       if tps else None),
    }


def fetch_server_props(transport, timeout):
    r = transport.get("/props", timeout)
    if r["ok"] and isinstance(r["json"], dict):
        d = r["json"]
        spec_fields = {k: d[k] for k in d
                       if ("spec" in k.lower() or "draft" in k.lower())}
        return {
            "model_path": d.get("model_path"),
            "spec_dec": spec_fields or
                        "NOT EXPOSED by this server build — decode t/s is "
                        "END-TO-END (speculative-decoding gains included if "
                        "the server runs a draft model; not isolated)",
            "n_ctx": d.get("n_ctx"),
            "n_batch": d.get("n_batch"),
            "n_ubatch": d.get("n_ubatch"),
            "n_gpu_layers": d.get("n_gpu_layers"),
            "build": d.get("build_info") or d.get("build"),
            "commit": d.get("commit"),
            "default_generation_settings": bool(d.get("default_generation_settings")),
        }
    return None


# ---------------------------------------------------------------- runner
def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def out_dir_for(base, name):
    # sanitize: name flows into a filesystem path — letters/digits/dot/dash only
    safe = "".join(c for c in name if c.isalnum() or c in "._-") or "run"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = os.path.join(base, f"{ts}-{safe}")
    suffix = 0
    while True:  # race-free no-clobber: let makedirs decide
        try:
            os.makedirs(d)
            return d
        except FileExistsError:
            suffix += 1
            d = os.path.join(base, f"{ts}-{safe}-{suffix}")


def run_all(url, name, targets, reps, out_base, timeouts, transport=None,
            n_predict_decode=128, decode_ctx=2048, log=print):
    transport = transport or UrllibTransport(url)
    sizing_state = {}
    out = out_dir_for(out_base, name)
    t0 = time.time()
    env = {
        "url": url, "date": _utcnow(), "argv": sys.argv,
        "tool_version": TOOL_VERSION,
        "python_version": platform.python_version(),
    }
    server = fetch_server_props(transport, timeouts["tokenize"])

    prefill_rows = run_prefill(transport, targets, reps, timeouts,
                               sizing_state, log)
    t_prefill = time.time()
    decode_rows = run_decode(transport, decode_ctx, reps, timeouts,
                             sizing_state, n_predict_decode, log)
    t_decode = time.time()
    ambient_pairs, ambient_rows = run_ambient(
        transport, targets, timeouts, sizing_state, n_predict_decode, log)
    t_ambient = time.time()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "name": name, "env": env, "server": server,
        "prefill": dict(summarize_prefill(prefill_rows),
                        wall_s=round(t_prefill - t0, 1)),
        "decode": dict(summarize_decode(decode_rows),
                       wall_s=round(t_decode - t_prefill, 1)),
        "ambient": {"pairs": ambient_pairs,
                    "wall_s": round(t_ambient - t_decode, 1)},
        "wall_s_total": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out, "rows.jsonl"), "w") as f:
        for r in prefill_rows + decode_rows + ambient_rows:
            f.write(json.dumps(r) + "\n")
    # hard failure = ALL requests of ANY battery failed, OR a battery produced
    # ZERO usable measurements (a 200-with-garbage server must not exit 0)
    hard = (summary["prefill"]["errors"] >= max(1, len(prefill_rows))
            or summary["decode"]["errors"] >= max(1, len(decode_rows))
            or not summary["prefill"]["by_measured_prompt_n"]
            or summary["decode"]["decode_tps"] is None)
    log(json.dumps({"out": out, "buckets": summary["prefill"]["by_measured_prompt_n"],
                    "decode": summary["decode"]["decode_tps"]}))
    return summary, out, (1 if hard else 0)


# ---------------------------------------------------------------- selftest mock
class MockHandlerConfig:
    """Deterministic knobs the tests (and --selftest) inject."""
    def __init__(self):
        self.chars_per_token = 3
        self.tokenize_status = 200
        self.completion_500_next = 0
        self.completion_500_from_call = 0  # fail every call >= N (1-based)
        self.shape = "current"          # old | current
        self.timings_missing = False
        self.eos_at = None              # int → stop early unless ignore_eos
        self.server_ignores_ignore_eos = False  # M1: make early_eos testable
        self.vary_content = False       # ambient A/B demonstration: flip sha
        self.slow_tokenize_s = 0.0      # M3: hung-endpoint simulation


def make_mock_server(config):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    cfg = config

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/props":
                self._send(200, {"model_path": "/mock/model.gguf",
                                 "n_ctx": 32768, "n_batch": 512,
                                 "n_ubatch": 512, "n_gpu_layers": 0,
                                 "build_info": "mock-server for stack-bench"})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            payload = _maybe_json(self.rfile.read(length).decode()) or {}
            if self.path == "/tokenize":
                if cfg.slow_tokenize_s:
                    time.sleep(cfg.slow_tokenize_s)
                if cfg.tokenize_status != 200:
                    self._send(cfg.tokenize_status,
                               {"error": {"message": "injected"}})
                    return
                import math
                n = math.ceil(len(payload.get("content") or "") / cfg.chars_per_token)
                out = {"tokens": ["x"] * n}
                if cfg.shape == "current":
                    out["pieces"] = ["x"] * n
                    out["bpe_offsets"] = [[i * cfg.chars_per_token,
                                           (i + 1) * cfg.chars_per_token]
                                          for i in range(n)]
                self._send(200, out)
            elif self.path == "/completion":
                cfg._calls = getattr(cfg, "_calls", 0) + 1  # first — knobs read it
                if cfg.completion_500_next > 0:
                    cfg.completion_500_next -= 1
                    self._send(500, {"error": {"code": 500,
                                               "message": "injected completion failure"}})
                    return
                if (cfg.completion_500_from_call
                        and cfg._calls >= cfg.completion_500_from_call):
                    self._send(500, {"error": {"code": 500,
                                               "message": "injected completion failure"}})
                    return
                prompt = payload.get("prompt") or ""
                import math, hashlib as h
                prompt_n = math.ceil(len(prompt) / cfg.chars_per_token)
                n_predict = payload.get("n_predict", 4)
                stop = n_predict
                respects_ignore = not cfg.server_ignores_ignore_eos
                if cfg.eos_at is not None and (not payload.get("ignore_eos")
                                               or not respects_ignore):
                    stop = min(n_predict, cfg.eos_at)
                # vary_content: alternate per CALL so an A/B pair diverges
                salt = str(cfg._calls % 2) if cfg.vary_content else "a"
                content = h.sha256((prompt + salt).encode()).hexdigest()[:16]
                prompt_ms = prompt_n * 3.0          # 333.3 t/s deterministic
                predicted_ms = stop * 24.0           # 41.7 t/s deterministic
                timings = {"prompt_n": prompt_n,
                           "prompt_ms": prompt_ms,
                           "prompt_per_second": prompt_n / (prompt_ms / 1000.0),
                           "predicted_n": stop,
                           "predicted_ms": predicted_ms,
                           "predicted_per_second": stop / (predicted_ms / 1000.0),
                           "tokens_cached": 0}
                if cfg.timings_missing:
                    timings = {}
                self._send(200, {"content": content,
                                 "stop_type": "eos" if stop < n_predict else "length",
                                 "timings": timings,
                                 "tokens_cached": 0})
            else:
                self._send(404, {"error": {"message": "not found"}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def selftest():
    cfg = MockHandlerConfig()
    server = make_mock_server(cfg)
    port = server.server_address[1]
    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".selftest")
    os.makedirs(out_base, exist_ok=True)
    try:
        summary, out, rc = run_all(
            f"http://127.0.0.1:{port}", "selftest",
            targets=[8000, 16000], reps=2,
            out_base=out_base,
            timeouts={"tokenize": 30, "completion": 60},
            transport=UrllibTransport(f"http://127.0.0.1:{port}"),
            log=lambda *a: None)
        ok = (rc == 0
              and len(summary["prefill"]["by_measured_prompt_n"]) >= 1
              and summary["decode"]["decode_tps"] is not None
              and len(summary["ambient"]["pairs"]) >= 3)
        print(json.dumps({"selftest": "PASS" if ok else "FAIL",
                          "out": out,
                          "buckets": summary["prefill"]["by_measured_prompt_n"]}))
        return 0 if ok else 1
    finally:
        server.shutdown()


# ---------------------------------------------------------------- cli
def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="stack-bench",
        description="Honest local-inference battery (prefill/decode/ambient).")
    p.add_argument("--url", help="llama-server base URL")
    p.add_argument("--name", default="run", help="run name (output dir suffix)")
    p.add_argument("--profile", choices=["quick", "full"], default="full")
    p.add_argument("--out", default="./stack-bench-out")
    p.add_argument("--reps", type=int, default=3, help="repetitions per target (1..10)")
    p.add_argument("--api-key", dest="api_key", default=None,
                   help="Authorization Bearer key for servers using --api-key")
    p.add_argument("--decode-ctx", dest="decode_ctx", type=int, default=2048,
                   help="decode-battery prompt size in tokens (512..32768)")
    p.add_argument("--version", action="version",
                   version="stack-bench " + TOOL_VERSION)
    p.add_argument("--n-predict", dest="n_predict", type=int, default=128,
                   help="decode n_predict (1..2048)")
    p.add_argument("--targets", type=str,
                   default=None, help="comma-separated TRUE token targets, e.g. --targets 8000,16000,32000")
    p.add_argument("--timeout-tokenize", type=int, default=600)
    p.add_argument("--timeout-completion", type=int, default=1800)
    p.add_argument("--selftest", action="store_true",
                   help="run against the built-in mock server (no network)")
    a = p.parse_args(argv)
    if not a.selftest and not a.url:
        p.error("--url required (or use --selftest)")
    lo, hi = BOUNDS["reps"]
    if not (lo <= a.reps <= hi):
        p.error(f"--reps must be {lo}..{hi} (got {a.reps})")
    lo, hi = BOUNDS["n_predict"]
    if not (lo <= a.n_predict <= hi):
        p.error(f"--n-predict must be {lo}..{hi} (got {a.n_predict})")
    if not (512 <= a.decode_ctx <= 32768):
        p.error("--decode-ctx must be 512..32768")

    if a.targets:
        parsed = []
        for t in a.targets.split(","):
            try:
                v = int(t)
            except ValueError:
                p.error(f"bad target {t!r}")
            lo, hi = BOUNDS["target"]
            if not (lo <= v <= hi):
                p.error(f"target {v} outside {lo}..{hi}")
            parsed.append(v)
        a.targets = parsed
    else:
        a.targets = [8000, 16000] if a.profile == "quick" else [8000, 16000, 32000]
    if (a.timeout_completion == 1800 and a.targets
            and any(t > 64000 for t in a.targets)):
        print("NOTE: targets >64k with the default completion timeout "
              "(1800s) can exceed the per-request wall on large models; "
              "consider --timeout-completion 3600 or higher.")
    return a


def main(argv=None):
    a = parse_args(argv if argv is not None else sys.argv[1:])
    if a.selftest:
        return selftest()
    summary, out, rc = run_all(
        a.url, a.name, a.targets, a.reps, a.out,
        {"tokenize": a.timeout_tokenize, "completion": a.timeout_completion},
        n_predict_decode=a.n_predict, decode_ctx=a.decode_ctx,
        transport=UrllibTransport(a.url, a.api_key))
    print(f"output: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
