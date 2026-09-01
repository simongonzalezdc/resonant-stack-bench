"""addon.stack-bench wrapper tests — acceptance criteria A2,A3,A4,A6,A9,A10.

Run:  python3 -m unittest discover -s tests -v   (from the add-on root)
"""
import hashlib
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_ROOT = os.path.dirname(HERE)
UPSTREAM = os.path.expanduser("~/workspaces/stack-bench")
sys.path.insert(0, ADDON_ROOT)
sys.path.insert(0, os.path.join(ADDON_ROOT, "vendor"))

import server  # noqa: E402
import stack_bench  # noqa: E402

TEST_PORT = 4899
BASE = f"http://127.0.0.1:{TEST_PORT}"


def post(payload, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def post_err(payload, raw=None):
    try:
        return post(payload, raw)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Service:
    def __enter__(self):
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", TEST_PORT), server.Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        server._state.update({"busy": False, "last_run_id": None, "runs": {}})


class TestVendorPin(unittest.TestCase):  # A4
    def test_vendored_files_hash_identical_to_upstream(self):
        for rel in ("stack_bench.py", os.path.join("tests", "test_e2e_mock.py")):
            ours, theirs = os.path.join(ADDON_ROOT, "vendor", rel), os.path.join(UPSTREAM, rel)
            self.assertTrue(os.path.exists(theirs), f"upstream missing: {rel}")
            self.assertEqual(sha256(ours), sha256(theirs), f"vendor drift: {rel}")


class TestInternalApiPin(unittest.TestCase):  # A9
    def test_transport_signature(self):
        import inspect
        sig = inspect.signature(stack_bench.UrllibTransport.__init__)
        self.assertEqual(list(sig.parameters), ["self", "base_url", "api_key"])
        self.assertIsNone(sig.parameters["api_key"].default)

    def test_run_all_accepts_transport_and_out_base(self):
        import inspect
        params = list(inspect.signature(stack_bench.run_all).parameters)
        for expected in ("url", "name", "targets", "reps", "out_base", "timeouts", "transport"):
            self.assertIn(expected, params)

    def test_summary_schema_keys(self):
        # the keys the service promises in stackbench.results
        with open(os.path.join(ADDON_ROOT, "vendor", "stack_bench.py")) as f:
            src = f.read()
        for key in ('"schema_version"', '"prefill"', '"decode"', '"ambient"', '"wall_s_total"'):
            self.assertIn(key, src)


class TestStatus(unittest.TestCase):  # A2
    def test_status_roundtrip(self):
        with Service():
            code, body = post({"method": "stackbench.status"})
            self.assertEqual(code, 200)
            self.assertTrue(body["ok"])
            self.assertEqual(body["version"], stack_bench.TOOL_VERSION)
            self.assertFalse(body["busy"])
            self.assertIsNone(body["last_run_id"])

    def test_get_health(self):
        with Service():
            with urllib.request.urlopen(BASE + "/health", timeout=10) as resp:
                body = json.loads(resp.read().decode())
            self.assertTrue(body["ok"])


class TestRunSemantics(unittest.TestCase):  # A3 + A10
    def _mock_endpoint(self):
        cfg = stack_bench.MockHandlerConfig()
        srv = stack_bench.make_mock_server(cfg)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{port}"

    def test_full_job_lifecycle_against_mock_server(self):
        srv, url = self._mock_endpoint()
        try:
            with Service():
                code, body = post({"method": "stackbench.run", "params": {
                    "endpoint_url": url, "prompt_tokens": [32], "rounds": 1}})
                self.assertEqual(code, 200)
                run_id = body["run_id"]
                self.assertEqual(body["state"], "running")
                deadline = time.time() + 120
                while time.time() < deadline:
                    code, res = post({"method": "stackbench.results", "params": {"run_id": run_id}})
                    if res["state"] != "running":
                        break
                    time.sleep(0.2)
                self.assertEqual(res["state"], "done", res.get("error", "run did not finish"))
                summary = res["summary"]
                self.assertIn("prefill", summary)
                self.assertIn("decode", summary)
                self.assertIsNotNone(summary["decode"].get("decode_tps"))
                serialized = json.dumps(summary)
                self.assertNotIn(os.path.expanduser("~"), serialized)  # A6 privacy in OUTPUTS
                self.assertNotIn(os.sep + "Users" + os.sep, serialized)
                env_argv = summary.get("env", {}).get("argv")
                self.assertEqual(env_argv, ["stackbench", run_id])  # sanitized argv
        finally:
            srv.shutdown()

    def test_single_flight_and_busy_flag(self):
        original = server.stack_bench.run_all
        release = threading.Event()

        def slow_run(*args, **kwargs):
            release.wait(timeout=10)
            return original(*args, **kwargs)

        srv, url = self._mock_endpoint()
        try:
            with Service():
                server.stack_bench.run_all = slow_run
                code, first = post({"method": "stackbench.run", "params": {
                    "endpoint_url": url, "prompt_tokens": [32], "rounds": 1}})
                self.assertEqual(code, 200)
                code, status = post({"method": "stackbench.status"})
                self.assertTrue(status["busy"])  # status answers WHILE a run is live
                code, second = post_err({"method": "stackbench.run", "params": {"endpoint_url": url}})
                self.assertEqual(code, 409)  # second concurrent run rejected
                release.set()
                deadline = time.time() + 120
                while time.time() < deadline:
                    code, res = post({"method": "stackbench.results", "params": {"run_id": first["run_id"]}})
                    if res["state"] != "running":
                        break
                    time.sleep(0.2)
                self.assertEqual(res["state"], "done")
        finally:
            server.stack_bench.run_all = original
            srv.shutdown()

    def test_latest_results_without_run_id(self):
        srv, url = self._mock_endpoint()
        try:
            with Service():
                post({"method": "stackbench.run", "params": {"endpoint_url": url, "prompt_tokens": [32], "rounds": 1}})
                deadline = time.time() + 120
                while time.time() < deadline:
                    code, res = post({"method": "stackbench.results", "params": {}})
                    if res["state"] != "running":
                        break
                    time.sleep(0.2)
                self.assertEqual(res["state"], "done")
                self.assertTrue(res["summary_path"].startswith("var/"))
        finally:
            srv.shutdown()


class TestAdversarial(unittest.TestCase):  # A6
    def test_remote_url_refused(self):
        with Service():
            for url in ("http://example.com:8080", "https://10.0.0.1/v1", "http://[::ffff:127.0.0.1]/x"):
                code, body = post_err({"method": "stackbench.run", "params": {"endpoint_url": url}})
                self.assertEqual(code, 400, url)
                self.assertIn("loopback", body["error"])

    def test_bad_schemes_rejected(self):
        with Service():
            for url in ("file:///etc/passwd", "ftp://127.0.0.1", "gopher://127.0.0.1"):
                code, _ = post_err({"method": "stackbench.run", "params": {"endpoint_url": url}})
                self.assertEqual(code, 400, url)

    def test_api_key_field_never_accepted(self):
        with Service():
            code, body = post_err({"method": "stackbench.run", "params": {
                "endpoint_url": "http://127.0.0.1:9", "api_key": "sk-secret"}})
            self.assertEqual(code, 400)
            self.assertNotIn("sk-secret", json.dumps(body))

    def test_injection_laden_fields_rejected(self):
        with Service():
            code, _ = post_err({"method": "stackbench.run", "params": {
                "endpoint_url": f"http://127.0.0.1:9/'; rm -rf /",
                "run_id": "a; touch /tmp/pwned"}})
            self.assertEqual(code, 400)

    def test_unknown_fields_rejected(self):
        with Service():
            code, _ = post_err({"method": "stackbench.status", "params": {}, "extra": 1})
            self.assertEqual(code, 400)

    def test_oversized_body_rejected(self):
        with Service():
            big = json.dumps({"method": "stackbench.run", "params": {
                "endpoint_url": "http://127.0.0.1:9", "run_id": "x" * 100000}}).encode()
            self.assertGreater(len(big), server.MAX_BODY)
            code, _ = post_err(None, raw=big)
            self.assertEqual(code, 413)

    def test_garbage_endpoint_fails_without_fakery(self):
        with Service():
            code, body = post({"method": "stackbench.run", "params": {
                "endpoint_url": "http://127.0.0.1:1", "prompt_tokens": [32], "rounds": 1}})
            run_id = body["run_id"]
            deadline = time.time() + 60
            while time.time() < deadline:
                code, res = post({"method": "stackbench.results", "params": {"run_id": run_id}})
                if res["state"] != "running":
                    break
                time.sleep(0.2)
            self.assertEqual(res["state"], "failed")  # honest failure, no exit-0 fakery

    def test_unknown_run_id_404(self):
        with Service():
            code, _ = post_err({"method": "stackbench.results", "params": {"run_id": "nope"}})
            self.assertEqual(code, 404)

    def test_no_home_paths_in_tree(self):  # A7
        needle = (os.sep + "Users" + os.sep).encode()  # built at runtime so this file stays clean
        for root, dirs, files in os.walk(ADDON_ROOT):
            dirs[:] = [d for d in dirs if d not in ("var", "__pycache__")]
            for name in files:
                path = os.path.join(root, name)
                if path.endswith((".pyc",)):
                    continue
                with open(path, "rb") as f:
                    content = f.read()
                self.assertNotIn(needle, content, f"home path leaked in {path}")


class TestRedactionAndReviewFixes(unittest.TestCase):  # review C1 + I2
    def _stub_run_all_with_home_paths(self, original):
        home = os.path.expanduser("~")

        def fake_run_all(url, name, targets, reps, out_base, timeouts, transport=None, **kw):
            summary, out, rc = original(url, name, targets, reps, out_base, timeouts, transport=transport, **kw)
            summary = dict(summary)
            summary["server"] = dict(summary.get("server") or {})
            summary["server"]["model_path"] = os.path.join(home, "models", "test.gguf")  # realistic llama-server /props
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1)
            return summary, out, rc

        return fake_run_all

    def test_home_paths_redacted_in_response_and_on_disk(self):
        original = server.stack_bench.run_all
        srv, url = self._mock()
        try:
            with Service():
                server.stack_bench.run_all = self._stub_run_all_with_home_paths(original)
                code, body = post({"method": "stackbench.run", "params": {
                    "endpoint_url": url, "prompt_tokens": [32], "rounds": 1}})
                run_id = body["run_id"]
                deadline = time.time() + 120
                while time.time() < deadline:
                    code, res = post({"method": "stackbench.results", "params": {"run_id": run_id}})
                    if res["state"] != "running":
                        break
                    time.sleep(0.2)
                self.assertEqual(res["state"], "done")  # redaction, NOT false failure
                serialized = json.dumps(res["summary"])
                self.assertNotIn(os.path.expanduser("~"), serialized)
                self.assertIn("~" + os.sep + "models", serialized)  # redacted form present
                # on-disk files must be clean too (review C1)
                home = os.path.expanduser("~")
                with open(os.path.join(ADDON_ROOT, res["summary_path"])) as f:
                    on_disk = f.read()
                self.assertNotIn(home, on_disk)
                rows_path = os.path.join(os.path.dirname(os.path.join(ADDON_ROOT, res["summary_path"])), "rows.jsonl")
                with open(rows_path) as f:
                    rows = f.read()
                self.assertNotIn(home, rows)
        finally:
            server.stack_bench.run_all = original
            srv.shutdown()

    def test_immediate_results_poll_never_404(self):
        srv, url = self._mock()
        try:
            with Service():
                code, body = post({"method": "stackbench.run", "params": {
                    "endpoint_url": url, "prompt_tokens": [32], "rounds": 1}})
                code, res = post_err({"method": "stackbench.results", "params": {"run_id": body["run_id"]}})
                self.assertIn(code, (200, ))
                self.assertEqual(res["state"], "running")  # visible immediately, not 404
                deadline = time.time() + 120
                while time.time() < deadline:
                    code, res = post({"method": "stackbench.results", "params": {"run_id": body["run_id"]}})
                    if res["state"] != "running":
                        break
                    time.sleep(0.2)
                self.assertEqual(res["state"], "done")
        finally:
            srv.shutdown()

    def _mock(self):
        cfg = stack_bench.MockHandlerConfig()
        srv = stack_bench.make_mock_server(cfg)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{port}"

    def test_redact_helpers(self):
        home = os.path.expanduser("~")
        self.assertEqual(server._redact_text("x" + home + "/y"), "x~/y")
        self.assertEqual(server._redact_obj({"a": [home + "/b"], "c": 3}), {"a": ["~/b"], "c": 3})


if __name__ == "__main__":
    unittest.main()
