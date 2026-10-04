"""How much cache reuse LMetric gives up on agent-like traces, and what it buys.

Each cell (trace, window, instances, KV capacity, load) replays the same
requests under several policies: load-only, LMetric, LMetric with its
tie-break shifted by one (the noise floor), affinity (longest prefix hit,
then batch size), pin (follow-ups to their parent's instance, otherwise
LMetric) and, with ``--runs``, SMetric. It also records the ideal hit: the
prefix each request could reuse from any earlier request in the window, as
with one infinite cache shared by all instances. Load fractions and the
capacity cache follow run_detector_sweep.py. Per-request results go to
``requests/``; one summary row per cell goes to the output JSONL.
"""

import argparse
import gzip
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

from cache_delay_eval.routing_sim import BLOCK, Cost, SMetric, simulate, summarize
from run_detector_sweep import TRACES, capacity, config, load

# name: (policy, tie offset, SMetric parameters)
RUNS = {
    "load": ("load", 0, None),
    "lmetric": ("lmetric", 0, None),
    "lmetric_tie1": ("lmetric", 1, None),
    "affinity": ("affinity", 0, None),
    "pin": ("pin", 0, None),
    "smetric": ("smetric", 0, {}),
    "smetric_tie1": ("smetric", 1, {}),
    "smetric_slack2": ("smetric", 0, dict(slack=2.0)),
    "smetric_slack05": ("smetric", 0, dict(slack=0.5)),
}


def ideal_hit(reqs):
    """Cached fraction with one infinite cache shared by all instances."""
    seen, hit = set(), 0
    for r in reqs:
        n = 0
        for key in r.keys:
            if key not in seen:
                break
            n += 1
        hit += min(n, (r.input_len - 1) // BLOCK) * BLOCK
        seen.update(r.keys)
    return hit / max(1, sum(r.input_len for r in reqs))


def cell(job):
    trace, start, n, kv_mult, load_frac, scale, cost, runs, out_dir = job
    c = Cost(**cost)
    row = dict(trace=trace, start_s=start, instances=n, kv_mult=kv_mult, load_frac=load_frac, time_scale=scale)
    per = {}
    for name in runs:
        policy, tie, params = RUNS[name]
        reqs = load(trace, start, scale)
        simulate(reqs, n, policy, "exact", c, config(kv_mult), tie_offset=tie,
                 smetric=SMetric(**params) if params is not None else None)
        row[name] = summarize(reqs, c)
        row[name]["stuck"] = sum(r.stuck for r in reqs)
        per[name] = reqs
    base = per[runs[0]]
    row["followups"] = sum(r.parent is not None for r in base)
    row["ideal_cached_fraction"] = ideal_hit(base)
    record = dict(row, cost_overhead=c.overhead,
                  arrival_ms=[r.arrival_ms for r in base],
                  input_len=[r.input_len for r in base],
                  followup=[int(r.parent is not None) for r in base])
    for name, reqs in per.items():
        assert [r.rid for r in reqs] == [r.rid for r in base]
        record[name] = dict(ttft=[round(r.first_ms - r.arrival_ms + c.overhead, 3) for r in reqs],
                            hit=[r.engine_hit or 0 for r in reqs],
                            instance=[r.instance for r in reqs],
                            stuck=[int(r.stuck) for r in reqs])
    name = f"{trace}-{start}-n{n}-kv{kv_mult}-l{load_frac}-{runs[0]}.json.gz"
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
    p.add_argument("--traces", default="kimi,trace-b,coder")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--configs", default="16x16,4x16,16x4,4x4,16x1,4x1", help="instances x KV multiplier")
    p.add_argument("--loads", default="0.5,0.7,0.9")
    p.add_argument("--runs", default="load,lmetric,lmetric_tie1,affinity,pin")
    args = p.parse_args()
    cost = json.load(open(args.cost))["cost"]
    runs = args.runs.split(",")
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
                    jobs.append((*c[:4], load_frac, load_frac * max_rate / trace_rate, cost, runs, str(req_dir)))
        print(f"{len(jobs)} cells", flush=True)
        with out.open("a") as f:
            for row in pool.imap_unordered(cell, jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row["trace"], row["start_s"], row["instances"], row["kv_mult"], row["load_frac"],
                      f"ideal {row['ideal_cached_fraction']:.2f}",
                      *(f"{k} {row[k]['ttft_mean']:.0f}/{row[k]['p99']:.0f}/{row[k]['cached_fraction']:.2f}"
                        for k in runs), flush=True)


if __name__ == "__main__":
    main()
