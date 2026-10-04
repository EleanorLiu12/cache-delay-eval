"""Replay LMetric, load-only routing, and LMetric with the paper's two-phase hotspot detector.

Each cell (trace, window, instances, KV capacity, load) runs four replays on the
same requests: load, lmetric, and lmetric_detector with a 60 s and a 10 s
detector window. Per-request results (arrival, class, number of instances
holding the class prefix at dispatch, time to first token, cache hit, detector
decisions) are written to ``requests/`` for offline analysis; one summary row
per cell goes to the output JSONL. The maximum rate comes from a load-only
flood run, as in run_routing_sim_sweep.py, and shares its capacity cache format.
"""

import argparse
import gzip
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

from cache_delay_eval.routing_sim import Cost, EngineConfig, load_trace, simulate, summarize

DATA = "data"
# name: (path, window length in trace seconds, loader options)
TRACES = {
    "chatbot": ("qwen-bailian/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_traceA_blksz_16.jsonl", 1800, {}),
    "trace-b": ("qwen-bailian-trace-b/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_traceB_blksz_16.jsonl", 600, {}),
    "coder": ("qwen-bailian-coder/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_coder_blksz_16.jsonl", 1800, {}),
    "kimi": ("mooncake-toolagent/0d1a8040faebb7c127c8901840a38c2ff57e80c5/toolagent_trace.jsonl", 900,
             dict(trace_block=512, timestamp_unit_s=1e-3)),
    "thinking": ("qwen-bailian-thinking/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_thinking_blksz_16.jsonl", 1800, {}),
}
RUNS = [("load", "load", None), ("lmetric", "lmetric", None),
        ("detector60", "lmetric_detector", 60.0), ("detector10", "lmetric_detector", 10.0)]
FLOOD_SCALE = 1000.0


def config(kv_mult):
    return EngineConfig(num_blocks=EngineConfig().num_blocks * kv_mult)


def load(trace, start, scale):
    path, length, opts = TRACES[trace]
    return load_trace(f"{DATA}/{path}", start, length, scale, **opts)


def capacity(job):
    trace, start, n, kv_mult, cost = job
    reqs = load(trace, start, FLOOD_SCALE)
    simulate(reqs, n, "load", "exact", Cost(**cost), config(kv_mult))
    return job[:4], len(reqs) / (max(r.done_ms for r in reqs) / 1000), len(reqs) / TRACES[trace][1]


def cell(job):
    trace, start, n, kv_mult, load_frac, scale, cost, out_dir = job
    c = Cost(**cost)
    row = dict(trace=trace, start_s=start, instances=n, kv_mult=kv_mult, load_frac=load_frac, time_scale=scale)
    per = {}
    for name, policy, window in RUNS:
        reqs = load(trace, start, scale)
        simulate(reqs, n, policy, "exact", c, config(kv_mult), detector_window_s=window or 60.0)
        row[name] = summarize(reqs, c)
        row[name]["alarmed"] = sum(r.alarm for r in reqs)
        row[name]["filtered"] = sum(r.filtered for r in reqs)
        per[name] = reqs
    base = per["lmetric"]
    classes = {}
    record = dict(row, cost_overhead=c.overhead,
                  arrival_ms=[r.arrival_ms for r in base],
                  cls=[classes.setdefault(r.cls, len(classes)) for r in base],
                  input_len=[r.input_len for r in base])
    for name, reqs in per.items():
        assert [r.rid for r in reqs] == [r.rid for r in base]
        record[name] = dict(ttft=[round(r.first_ms - r.arrival_ms + c.overhead, 3) for r in reqs],
                            hit=[r.engine_hit or 0 for r in reqs],
                            holders=[r.holders for r in reqs],
                            instance=[r.instance for r in reqs],
                            alarm=[int(r.alarm) for r in reqs],
                            filtered=[int(r.filtered) for r in reqs])
    name = f"{trace}-{start}-n{n}-kv{kv_mult}-l{load_frac}.json.gz"
    with gzip.open(Path(out_dir) / name, "wt") as f:
        json.dump(record, f)
    row["requests_file"] = f"requests/{name}"
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cost", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--capacity-cache", required=True, help="JSON of measured max rates, read and extended")
    p.add_argument("--seed-capacity", help="read-only capacity cache from an earlier sweep")
    p.add_argument("--traces", default=",".join(TRACES))
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--configs", default="16x16,4x1,16x4,4x4,16x1",
                   help="instances x KV multiplier, in run order")
    p.add_argument("--loads", default="0.5,0.7,0.9")
    args = p.parse_args()
    cost = json.load(open(args.cost))["cost"]
    out = Path(args.output)
    req_dir = out.parent / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["trace"], r["start_s"], r["instances"], r["kv_mult"], r["load_frac"]))
    configs = [tuple(map(int, c.split("x"))) for c in args.configs.split(",")]
    cells = [(trace, w * TRACES[trace][1], n, kv, cost)
             for (n, kv), trace, w in itertools.product(configs, args.traces.split(","), range(args.windows))]
    known = {}
    for path in (args.seed_capacity, args.capacity_cache):
        if path and Path(path).exists():
            known.update(json.load(open(path)))
    name = lambda key: "|".join(map(str, key))
    with Pool() as pool:
        todo = [c for c in cells if name(c[:4]) not in known]
        for key, max_rate, trace_rate in pool.map(capacity, todo):
            known[name(key)] = [max_rate, trace_rate]
        Path(args.capacity_cache).write_text(json.dumps(known, indent=1))
        jobs = []
        for c in cells:
            max_rate, trace_rate = known[name(c[:4])]
            for load_frac in map(float, args.loads.split(",")):
                if (*c[:4], load_frac) not in done:
                    jobs.append((*c[:4], load_frac, load_frac * max_rate / trace_rate, cost, str(req_dir)))
        print(f"{len(jobs)} cells", flush=True)
        with out.open("a") as f:
            for row in pool.imap_unordered(cell, jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row["trace"], row["start_s"], row["instances"], row["kv_mult"], row["load_frac"],
                      *(f"{k} {row[k]['ttft_mean']:.0f}/{row[k]['p99']:.0f}" for k, _, _ in RUNS),
                      f"alarm60 {row['detector60']['alarmed']} filt60 {row['detector60']['filtered']}"
                      f" filt10 {row['detector10']['filtered']}", flush=True)


if __name__ == "__main__":
    main()
