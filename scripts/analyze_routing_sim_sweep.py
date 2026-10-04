"""Summarize the simulator sweep: how much of LMetric's gain survives an approximate index."""

import argparse
import json
import statistics
from collections import defaultdict

VARIANTS = ["load/exact", "lmetric/exact", "lmetric/dispatch", "lmetric/lru", "sglang_ca/dispatch"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep")
    p.add_argument("--metric", default="ttft_mean")
    p.add_argument("--output")
    args = p.parse_args()
    cells = defaultdict(dict)
    for line in open(args.sweep):
        r = json.loads(line)
        key = (r["trace"], r["instances"], r["kv_mult"], r["load"])
        cells[key, r["start_s"]][f"{r['policy']}/{r['index']}"] = r
    groups = defaultdict(list)
    for (key, start), runs in cells.items():
        if all(v in runs for v in VARIANTS):
            groups[key].append(runs)
    rows = []
    print(f"metric: {args.metric} (ms); ratios are approximate/exact LMetric; 'gain lost' is the share of "
          "LMetric's improvement over load-only routing that the approximate index gives back")
    print(f"{'trace':9s} {'N':>3s} {'kv':>3s} {'load':>4s} {'win':>3s} "
          + " ".join(f"{v:>18s}" for v in VARIANTS) + "  dispatch/exact  lru/exact  gain lost (dispatch, lru)")
    for key in sorted(groups):
        windows = groups[key]
        m = {v: [w[v][args.metric] for w in windows] for v in VARIANTS}
        ratio_d = [d / e for d, e in zip(m["lmetric/dispatch"], m["lmetric/exact"])]
        ratio_l = [d / e for d, e in zip(m["lmetric/lru"], m["lmetric/exact"])]
        lost_d = [(d - e) / (l - e) if l > e else float("nan")
                  for d, e, l in zip(m["lmetric/dispatch"], m["lmetric/exact"], m["load/exact"])]
        lost_l = [(d - e) / (l - e) if l > e else float("nan")
                  for d, e, l in zip(m["lmetric/lru"], m["lmetric/exact"], m["load/exact"])]
        over = [w["lmetric/dispatch"]["over_predicted"] / w["lmetric/dispatch"]["n"] for w in windows]
        med = statistics.median
        rows.append(dict(trace=key[0], instances=key[1], kv_mult=key[2], load=key[3], windows=len(windows),
                         median={v: med(m[v]) for v in VARIANTS},
                         dispatch_over_exact=ratio_d, lru_over_exact=ratio_l,
                         gain_lost_dispatch=lost_d, gain_lost_lru=lost_l,
                         dispatch_over_predicted_share=over,
                         hit={v: med([w[v]["cached_fraction"] for w in windows]) for v in VARIANTS}))
        span = lambda xs: f"{med(xs):.2f} [{min(xs):.2f}-{max(xs):.2f}]"
        print(f"{key[0]:9s} {key[1]:3d} {key[2]:3d} {key[3]:4.1f} {len(windows):3d} "
              + " ".join(f"{med(m[v]):18.0f}" for v in VARIANTS)
              + f"  {span(ratio_d):>16s} {span(ratio_l):>16s}  {med(lost_d):.2f}, {med(lost_l):.2f}")
    if args.output:
        json.dump(rows, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
