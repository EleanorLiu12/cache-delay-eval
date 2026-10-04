"""Sweep LMetric with exact vs approximate prefix indexes in the calibrated simulator.

For each (trace, window, KV capacity), a flood run under load-only routing gives
the cluster's maximum request rate; the window is then replayed at a fraction
of that rate (the paper uses half) under each router variant.
"""

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

from cache_delay_eval.routing_sim import Cost, EngineConfig, load_trace, simulate, summarize

DATA = "data"
TRACES = {
    "thinking": ("qwen-bailian-thinking/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_thinking_blksz_16.jsonl", 1800),
    "coder": ("qwen-bailian-coder/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_coder_blksz_16.jsonl", 1800),
    "chatbot": ("qwen-bailian/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_traceA_blksz_16.jsonl", 1800),
    "trace-b": ("qwen-bailian-trace-b/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_traceB_blksz_16.jsonl", 600),
}
VARIANTS = [("load", "exact"), ("lmetric", "exact"), ("lmetric", "dispatch"), ("lmetric", "lru"),
            ("sglang_ca", "dispatch")]
FLOOD_SCALE = 1000.0


def config(kv_mult):
    return EngineConfig(num_blocks=EngineConfig().num_blocks * kv_mult)


def capacity(job):
    trace, start, n, kv_mult, cost = job
    path, length = TRACES[trace]
    reqs = load_trace(f"{DATA}/{path}", start, length, FLOOD_SCALE)
    simulate(reqs, n, "load", "exact", Cost(**cost), config(kv_mult))
    return job[:4], len(reqs) / (max(r.done_ms for r in reqs) / 1000), len(reqs) / length


def replay(job):
    trace, start, n, kv_mult, load, scale, policy, index, cost = job
    path, length = TRACES[trace]
    reqs = load_trace(f"{DATA}/{path}", start, length, scale)
    c = Cost(**cost)
    simulate(reqs, n, policy, index, c, config(kv_mult))
    return dict(trace=trace, start_s=start, instances=n, kv_mult=kv_mult, load=load, time_scale=scale,
                policy=policy, index=index, **summarize(reqs, c))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cost", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--traces", default=",".join(TRACES))
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--instances", default="16")
    p.add_argument("--kv-mult", default="1,4,16")
    p.add_argument("--loads", default="0.5,0.7")
    p.add_argument("--capacity-cache", help="JSON of measured max rates, read and extended")
    p.add_argument("--variants", default=",".join(f"{a}/{b}" for a, b in VARIANTS),
                   help="comma-separated policy/index pairs")
    args = p.parse_args()
    cost = json.load(open(args.cost))["cost"]
    out = Path(args.output)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["trace"], r["start_s"], r["instances"], r["kv_mult"], r["load"], r["policy"], r["index"]))
    cells = []
    for trace in args.traces.split(","):
        length = TRACES[trace][1]
        for w, n, kv in itertools.product(range(args.windows), map(int, args.instances.split(",")),
                                          map(int, args.kv_mult.split(","))):
            cells.append((trace, w * length, n, kv, cost))
    known = json.load(open(args.capacity_cache)) if args.capacity_cache and Path(args.capacity_cache).exists() else {}
    name = lambda key: "|".join(map(str, key))
    with Pool() as pool:
        todo = [c for c in cells if name(c[:4]) not in known]
        for key, max_rate, trace_rate in pool.map(capacity, todo):
            known[name(key)] = [max_rate, trace_rate]
        if args.capacity_cache:
            Path(args.capacity_cache).write_text(json.dumps(known, indent=1))
        caps = [(c[:4], *known[name(c[:4])]) for c in cells]
        jobs = []
        for key, max_rate, trace_rate in caps:
            print("capacity", key, f"max {max_rate:.2f} req/s, trace {trace_rate:.2f} req/s", flush=True)
            for load in map(float, args.loads.split(",")):
                scale = load * max_rate / trace_rate
                for policy, index in (v.split("/") for v in args.variants.split(",")):
                    if (*key, load, policy, index) not in done:
                        jobs.append((*key, load, scale, policy, index, cost))
        with out.open("a") as f:
            for row in pool.imap_unordered(replay, jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row["trace"], row["start_s"], row["kv_mult"], row["load"], row["policy"], row["index"],
                      f"ttft {row['ttft_mean']:.0f} p99 {row['p99']:.0f} hit {row['cached_fraction']:.2f}", flush=True)


if __name__ == "__main__":
    main()
