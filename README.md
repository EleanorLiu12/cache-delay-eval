# Cache-Aware Routing Failure Cases

**Objective: test, with measurements, how often the failure condition of LMetric's multiplicative routing score ([Zhang et al., OSDI '26, §5.2](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf)) occurs in real traces, and what it costs in time to first token.**

LMetric routes each request to the instance with the smallest (queued prefill tokens + new prefill tokens) × (batch size + 1), and reports that hotspots that defeat this score are extremely rare. This repository replays public Qwen-Bailian traces against several vLLM instances with a client-side router, measures engine costs for a routing simulator, and keeps the earlier single-server cache-hit/TTFT study as dated evidence.

| Question | Entry point |
| --- | --- |
| How do I replay a trace across instances with a given routing policy? | `python -m cache_delay_eval.routing --help` ([source](src/cache_delay_eval/routing.py)) |
| How do I measure prefill and decode costs of one instance? | [`scripts/profile_engine.py`](scripts/profile_engine.py) |
| How do I split an A30 into four isolated instances and start the servers? | [`scripts/a30_cluster.sh`](scripts/a30_cluster.sh) |
| Which earlier data remain valid? | [Evidence review](results/evidence-review-2026-09-30/report.md) |
| What did the first single-server GPU experiment find? | [GPU experiment report](results/cloudlab-pattern-03/report.md) and [run guide](docs/pattern-pilot.md) |
| How are requests represented? | [Trace format](schemas/README.md) |

## Routing replay

The client is the router. Batch size and queued prefill tokens change at dispatch, first token and completion. The prefix index follows each engine's block events (`--index events`, exact), or assumes every routed prefix stays cached (`--index dispatch`, the approximate tree used by common open-source routers). Policies: `load` (batch size only), `lmetric`, `affinity` (most cached blocks).

Servers need `--enable-prefix-caching --enable-prompt-tokens-details` and a ZMQ block-event publisher; `scripts/a30_cluster.sh` sets these. Trace block hashes become deterministic 16-token blocks, so shared prefixes in the trace are shared prefixes on the engine.

```bash
python -m pip install -e '.[live,routing]'
python -m unittest discover -s tests
python -m cache_delay_eval.routing --instances x --policy load --workload trace \
  --trace data/qwen-bailian-thinking/<revision>/qwen_thinking_blksz_16.jsonl \
  --start-s 2100 --duration-s 600 --time-scale 6 --output /dev/null --dry-run
```

Source code is in `src/cache_delay_eval/`, scripts in `scripts/`, and measurements with their analyses in `results/`. Public traces are downloaded to `data/` (not tracked) by `scripts/fetch_*.py`.
