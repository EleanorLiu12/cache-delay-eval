"""Summarize run_reuse_sweep.py: reuse LMetric gives up, and whether it pays.

For each (trace, instances, KV multiplier, load) the script reports medians
over non-overloaded windows: cached fraction under the ideal shared cache,
LMetric, affinity and pin; mean and 99th-percentile time to first token
relative to LMetric; and the follow-up requests' cached fraction. A latency
difference counts only if it exceeds the noise floor: the 90th percentile,
over that trace's non-overloaded cells, of |tie-shifted LMetric / LMetric - 1|.
``--compare`` names further runs (e.g. smetric) to report the same way.
"""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

OVERLOAD_MS = 2000.0     # load-only mean TTFT above this: the window is queueing-bound, skipped
TRACES = ["kimi", "trace-b", "coder"]


def q90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))] if xs else float("nan")


def followup_stats(paths, runs):
    """Cached fraction and mean TTFT of follow-up requests, per run, from the cell's request files."""
    out = {}
    for path in paths:
        rec = json.load(gzip.open(path, "rt"))
        idx = [i for i, f in enumerate(rec["followup"]) if f]
        tokens = sum(rec["input_len"][i] for i in idx) or 1
        out["share"] = len(idx) / len(rec["followup"])
        for run in runs:
            if run in rec and run not in out:
                r = rec[run]
                out[run] = dict(hit=sum(r["hit"][i] for i in idx) / tokens,
                                ttft=statistics.fmean(r["ttft"][i] for i in idx) if idx else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep", nargs="+")
    p.add_argument("--compare", default="affinity,pin")
    p.add_argument("--base", default="lmetric")
    args = p.parse_args()
    cells = {}
    for path in args.sweep:
        for line in open(path):
            r = json.loads(line)
            key = (r["trace"], r["start_s"], r["instances"], r["kv_mult"], r["load_frac"])
            cells.setdefault(key, {}).update(r)
            cells[key].setdefault("files", []).append(Path(path).parent / r["requests_file"])
    others = args.compare.split(",")
    base = args.base
    ok = {k: c for k, c in cells.items() if c["load"]["ttft_mean"] <= OVERLOAD_MS}
    print(f"{len(cells)} cells, {len(cells) - len(ok)} overloaded windows skipped")

    noise = {}
    for t in TRACES:
        cs = [c for k, c in ok.items() if k[0] == t and "lmetric_tie1" in c]
        noise[t] = dict(mean=q90([abs(c["lmetric_tie1"]["ttft_mean"] / c["lmetric"]["ttft_mean"] - 1) for c in cs]),
                        p99=q90([abs(c["lmetric_tie1"]["p99"] / c["lmetric"]["p99"] - 1) for c in cs]),
                        hit=q90([abs(c["lmetric_tie1"]["cached_fraction"] - c["lmetric"]["cached_fraction"])
                                 for c in cs]))
        print(f"noise floor {t}: mean {noise[t]['mean']:.1%}  p99 {noise[t]['p99']:.1%}  "
              f"hit {100 * noise[t]['hit']:.2f} pt")

    groups = defaultdict(list)
    for k, c in ok.items():
        groups[(k[0], k[2], k[3], k[4])].append((k, c))
    cols = " ".join(f"{o:>9s}" for o in others)
    print(f"\n## Medians over windows. hit = cached fraction (%); ratios are run / {base}; "
          f"! = beyond noise floor")
    print(f"trace    N  kv load  w | ideal {base:>8s} {cols} | fu% fu-hit {base}/{'/'.join(others)} | "
          + " | ".join(f"{o} mean p99" for o in others))
    flagged = []
    for key in sorted(groups, key=lambda k: (TRACES.index(k[0]), -k[1], -k[2], k[3])):
        cs = groups[key]
        med = lambda f: statistics.median(f(c) for _, c in cs)
        fu = [followup_stats(c["files"], [base] + others) for _, c in cs]
        n = noise[key[0]]
        hits = " ".join(f"{100 * med(lambda c, o=o: c[o]['cached_fraction']):9.1f}" for o in others)
        fuhit = "/".join(f"{100 * statistics.median(f[o]['hit'] for f in fu):.0f}" for o in [base] + others)
        lat = []
        for o in others:
            rm = med(lambda c, o=o: c[o]["ttft_mean"] / c[base]["ttft_mean"])
            rp = med(lambda c, o=o: c[o]["p99"] / c[base]["p99"])
            lat.append(f"{rm:5.2f}{'!' if abs(rm - 1) > n['mean'] else ' '} {rp:5.2f}{'!' if abs(rp - 1) > n['p99'] else ' '}")
        print(f"{key[0]:8s} {key[1]:2d} {key[2]:3d} {key[3]:.1f} {len(cs):2d} | "
              f"{100 * med(lambda c: c['ideal_cached_fraction']):5.1f} {100 * med(lambda c: c[base]['cached_fraction']):8.1f} "
              f"{hits} | {100 * statistics.median(f['share'] for f in fu):3.0f} {fuhit} | " + " | ".join(lat))
        # Per window: base loses hits to a comparison run AND is slower than it, both beyond noise.
        for k, c in cs:
            for o in others:
                lost = c[o]["cached_fraction"] - c[base]["cached_fraction"]
                slower = c[base]["ttft_mean"] / c[o]["ttft_mean"] - 1
                slower99 = c[base]["p99"] / c[o]["p99"] - 1
                if lost > n["hit"] and (slower > n["mean"] or slower99 > n["p99"]):
                    flagged.append((k, o, lost, slower, slower99))
    print(f"\n## Windows where {base} has fewer hits than a comparison run AND a higher mean or p99, beyond noise")
    for k, o, lost, slower, slower99 in sorted(flagged):
        print(f"{k} vs {o}: hit -{100 * lost:.1f} pt, mean {slower:+.1%}, p99 {slower99:+.1%}")
    if not flagged:
        print("none")


if __name__ == "__main__":
    main()
