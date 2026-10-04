"""Replay each A30 routing run in the simulator and compare against the measurement."""

import argparse
import json
import statistics
from pathlib import Path

from cache_delay_eval import routing
from cache_delay_eval.routing_sim import Cost, SimRequest, load_trace, simulate, summarize

TRACE = "data/qwen-bailian-thinking/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_thinking_blksz_16.jsonl"
SLICES = {"hot": 2100, "typical": 5100}
INDEX = {"events": "exact", "dispatch": "dispatch"}


def measured(path):
    rows = [json.loads(l) for l in open(path)]
    reqs = [r for r in rows if r.get("type") == "request" and r["status"] == "ok"]
    ttft = sorted(r["ttft_ms"] for r in reqs)
    pct = lambda q: ttft[min(len(ttft) - 1, int(q * len(ttft)))]
    tpot = [(r["e2e_ms"] - r["ttft_ms"]) / (r["output_len"] - 1) for r in reqs if r["output_len"] > 1]
    return dict(n=len(reqs), tpot_mean=statistics.fmean(tpot), ttft_mean=statistics.fmean(ttft), p50=pct(.5), p90=pct(.9), p99=pct(.99),
                cached_fraction=sum(r["engine_cached_tokens"] for r in reqs) / sum(r["input_len"] for r in reqs),
                per_instance=rows[-1]["per_instance"])


def burst_requests(bg_rate, hot_rate):
    book = routing.TokenBook("a30-2026-10-02")
    reqs = routing.synth_burst(300, bg_rate, [512, 1024, 2048, 4096], hot_rate, 60, 120, 4096, 256, 256, 699, book)
    return [SimRequest(r.rid, r.arrival_ms, r.full_blocks, r.input_len, r.output_len, r.cls) for r in reqs]


def workload(name):
    parts = name.removesuffix(".jsonl").split("-")
    if parts[0] == "burst":
        rate = 3.25 if parts[1] == "half" else 6.5
        return burst_requests(rate, rate), parts[-1], "exact"
    scale = float(parts[1][1:])
    return load_trace(TRACE, SLICES[parts[0]], 600, scale), parts[2], INDEX[parts[3]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs")
    p.add_argument("--cost", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--only", default="", help="substring filter on run names")
    p.add_argument("--set", action="append", default=[], help="override a cost field, e.g. decode_ctx=4e-4")
    args = p.parse_args()
    cost = Cost(**json.load(open(args.cost))["cost"])
    for item in args.set:
        k, v = item.split("=")
        setattr(cost, k, float(v))
    out = []
    for path in sorted(Path(args.runs).glob("*.jsonl")):
        if path.name == "smoke.jsonl" or args.only not in path.name:
            continue
        reqs, policy, index = workload(path.name)
        sim = summarize(simulate(reqs, 4, policy, index, cost), cost)
        real = measured(path)
        out.append(dict(run=path.name, real=real, sim=sim))
        f = lambda d: f"mean {d['ttft_mean']:8.0f} p50 {d['p50']:7.0f} p90 {d['p90']:7.0f} p99 {d['p99']:8.0f} hit {d['cached_fraction']:.2f} tpot {d['tpot_mean']:5.1f}"
        print(f"{path.name:36s} real {f(real)}\n{'':36s} sim  {f(sim)} preempt {sim['preemptions']}", flush=True)
    Path(args.output).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
