# Stack Bench — ResonantOS add-on

An honest local-inference benchmark, packaged as a ResonantOS 2.0.0-alpha
add-on. It measures a local llama-server endpoint: prefill and decode
tokens-per-second with true-token prompt sizing, plus an ambient-noise A/B
check. No dashboards, no leaderboards — just numbers you can check yourself,
written as plain JSON files you own.

The benchmark engine is [stack-bench](https://github.com/KyaniteLabs/stack-bench)
(MIT), vendored byte-identical under `vendor/` and wrapped by a thin local
service. The wrapper adds no dependencies: Python 3.10+ standard library only.

## What it does

- `stackbench.status` — report service version, busy state, last run id.
- `stackbench.run` — start one benchmark job against a local endpoint;
  returns a run id immediately (a full run takes minutes).
- `stackbench.results` — poll a run; returns the summary (prefill/decode
  tok/s, ambient pairs, wall times) or the honest failure verdict.

One run at a time. Results land under `var/<timestamp>-<run_id>/`
(`summary.json` + `rows.jsonl`), with any home paths redacted to `~`.

## Running it

    python3 server.py          # listens on http://127.0.0.1:4889 (the manifest entrypoint)

    curl -s http://127.0.0.1:4889/health
    curl -s -X POST http://127.0.0.1:4889/ -H 'Content-Type: application/json' \
      -d '{"method":"stackbench.run","params":{"endpoint_url":"http://127.0.0.1:8080","prompt_tokens":[8000],"rounds":3}}'
    curl -s -X POST http://127.0.0.1:4889/ -H 'Content-Type: application/json' \
      -d '{"method":"stackbench.results"}'

Environment: `STACKBENCH_API_KEY` (sent as Bearer to servers started with
`--api-key`; never accepted as a request field), `STACKBENCH_ALLOW_REMOTE=1`
(opt-in to benchmark a non-loopback endpoint), `STACKBENCH_PORT` (dev only —
the manifest declares 4889). The service refuses non-loopback endpoints by
default, spawns no subprocesses, and keeps no telemetry.

## Tests

    python3 -m unittest discover -s tests        # wrapper suite (21 tests)
    python3 -m unittest discover -s vendor/tests # upstream suite, unmodified (19 tests)
    sh run-validator-check.sh <path-to-2.0.0-alpha-clone>  # manifest vs the real validator

`vendor/` is hash-pinned to upstream; a wrapper test fails loudly if the
vendored files drift. The vendored upstream tests write scratch output under
`/tmp/sb-test-out/` (upstream behavior, left unmodified).

## License

MIT — see LICENSE. The vendored stack-bench engine is MIT, KyaniteLabs.
