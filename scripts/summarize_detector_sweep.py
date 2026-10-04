"""Print the report tables from cells.json (written by analyze_detector_sweep.py)."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TRACES = ["chatbot", "trace-b", "coder", "kimi", "thinking"]
OVERLOAD_MS = 2000.0   # load-only mean TTFT above this: the cell is queueing-bound, skip in latency tables


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def groups(cells):
    g = defaultdict(list)
    for c in cells:
        g[c["trace"], c["instances"], c["kv_mult"], c["load_frac"]].append(c)
    return g


def eq2_table(cells):
    print("\n## Equation 2: share of windows violated, all classes / top-3 classes (sum over 4 windows)")
    g = groups(cells)
    cols = [(clock, w) for clock in ("trace", "replay") for w in (10, 60)]
    print("trace     N  kv load | " + " | ".join(f"{c}{w:>3}s" for c, w in cols) + " | viol. req %")
    for key in sorted(g, key=lambda k: (TRACES.index(k[0]), -k[1], k[2], k[3])):
        cs = g[key]
        parts = []
        for clock, w in cols:
            s = [c["stats"][f"{clock}{w}"] for c in cs]
            n = sum(x["windows"] for x in s)
            parts.append(f"{pct(sum(x['violating'] for x in s), n):4.0f}/{pct(sum(x['violating_top'] for x in s), n):3.0f}")
        s = [c["stats"]["replay60"] for c in cs]
        req = pct(sum(x["violating_requests"] for x in s), sum(x["eligible"] for x in s))
        print(f"{key[0]:9s} {key[1]:2d} {key[2]:3d} {key[3]:.1f} | " + " | ".join(parts) + f" | {req:5.1f}")


def detector_table(cells):
    print("\n## Detector vs plain LMetric (median over windows; ratios detector/LMetric); * = overloaded windows skipped")
    g = groups(cells)
    print("trace     N  kv load | alarm% filt% | 60s: mean  p99   hit-diff | 10s: mean  p99   hit-diff | "
          "filtered req: det/lm mean, % slower | LMetric/load mean")
    for key in sorted(g, key=lambda k: (TRACES.index(k[0]), -k[1], k[2], k[3])):
        cs = [c for c in g[key] if c["load"]["ttft_mean"] <= OVERLOAD_MS]
        skipped = len(g[key]) - len(cs)
        if not cs:
            print(f"{key[0]:9s} {key[1]:2d} {key[2]:3d} {key[3]:.1f} | all windows overloaded")
            continue
        n = sum(c["lmetric"]["n"] for c in cs)
        alarm = pct(sum(c["detector60"]["alarmed"] for c in cs), n)
        filt = pct(sum(c["detector60"]["filtered"] for c in cs), n)
        med = lambda f: statistics.median(f(c) for c in cs)
        parts = []
        for d in ("detector60", "detector10"):
            parts.append(f"{med(lambda c: c[d]['ttft_mean'] / c['lmetric']['ttft_mean']):5.3f} "
                         f"{med(lambda c: c[d]['p99'] / c['lmetric']['p99']):5.3f} "
                         f"{100 * med(lambda c: c[d]['cached_fraction'] - c['lmetric']['cached_fraction']):+5.1f}pt")
        fs = [c["stats"]["detector60"] for c in cs if c["stats"]["detector60"]["filtered"]]
        fn = sum(x["filtered"] for x in fs)
        if fn:
            det = sum(x["filtered_det_mean"] * x["filtered"] for x in fs) / fn
            lm = sum(x["filtered_lm_mean"] * x["filtered"] for x in fs) / fn
            ftxt = f"{det / lm:5.2f}, {pct(sum(x['filtered_slower'] for x in fs), fn):3.0f}%"
        else:
            ftxt = "    -,   -"
        print(f"{key[0]:9s} {key[1]:2d} {key[2]:3d} {key[3]:.1f} | {alarm:5.1f} {filt:5.2f} | " + " | ".join(parts)
              + f" | {ftxt} | {med(lambda c: c['lmetric']['ttft_mean'] / c['load']['ttft_mean']):5.3f}"
              + (f" *{skipped}" if skipped else ""))


def loss_table(cells):
    print("\n## LMetric loss windows (window mean TTFT > 1.2x load-only) vs Equation 2 violations, "
          "non-overloaded cells pooled per trace")
    print("trace     width | windows  loss  viol  both | P(viol) P(viol|loss) P(loss|viol) P(loss|no viol) "
          "| excess in viol. windows | mean in loss windows: LMetric / load / detector60 (ms)")
    for trace in TRACES:
        for w in (10, 60):
            cs = [c["stats"][f"loss{w}"] for c in cells
                  if c["trace"] == trace and c["load"]["ttft_mean"] <= OVERLOAD_MS]
            if not cs:
                continue
            W = sum(x["windows"] for x in cs)
            L = sum(x["loss"] for x in cs)
            V = sum(x["violating"] for x in cs)
            B = sum(x["both"] for x in cs)
            ex = sum(x["loss_excess_ms"] for x in cs)
            exv = sum(x["loss_excess_in_violating_ms"] for x in cs)
            lw = [x for x in cs if x["loss"]]
            avg = lambda k: sum(x[k] * x["loss"] for x in lw) / L if L else float("nan")
            print(f"{trace:9s} {w:4d}s | {W:7d} {L:5d} {V:5d} {B:5d} | {pct(V, W):6.1f} {pct(B, L):11.1f} "
                  f"{pct(B, V):12.1f} {pct(L - B, W - V):15.1f} | {pct(exv, ex):6.1f}% "
                  f"| {avg('lmetric_in_loss_ms'):7.0f} / {avg('load_in_loss_ms'):7.0f} / {avg('detector_in_loss_ms'):7.0f}")


def top_class_table(cells):
    print("\n## Busiest class: mean TTFT ratios and median holders (median over non-overloaded windows)")
    print("trace     N  kv load | share | LMetric/load  detector60/LMetric | holders load/LMetric/detector60")
    g = groups(cells)
    for key in sorted(g, key=lambda k: (TRACES.index(k[0]), -k[1], k[2], k[3])):
        cs = [c["stats"]["top_class"] for c in g[key] if c["load"]["ttft_mean"] <= OVERLOAD_MS]
        if not cs:
            continue
        med = lambda f: statistics.median(f(x) for x in cs)
        print(f"{key[0]:9s} {key[1]:2d} {key[2]:3d} {key[3]:.1f} | {med(lambda x: x['share']):5.2f} | "
              f"{med(lambda x: x['lmetric_ttft'] / x['load_ttft']):12.2f} {med(lambda x: x['detector60_ttft'] / x['lmetric_ttft']):19.2f} | "
              f"{med(lambda x: x['load_holders']):4.0f} {med(lambda x: x['lmetric_holders']):4.0f} {med(lambda x: x['detector60_holders']):4.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    cells = json.loads((args.root / "cells.json").read_text())
    eq2_table(cells)
    detector_table(cells)
    loss_table(cells)
    top_class_table(cells)


if __name__ == "__main__":
    main()
