"""How far Equation 2 violations exceed the boundary: x * N / |M| for violating requests (replay windows)."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_detector_sweep import violations, window_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--width", type=float, default=60)
    args = p.parse_args()
    out = defaultdict(list)
    for line in (args.root / "sweep.jsonl").read_text().splitlines():
        row = json.loads(line)
        with gzip.open(args.root / row["requests_file"], "rt") as f:
            rec = json.load(f)
        win = window_ids(rec, "replay", args.width)
        viol, _, _, w_idx = violations(rec, win)
        cls = np.asarray(rec["cls"])
        holders = np.asarray(rec["lmetric"]["holders"])
        key = w_idx * (cls.max() + 1) + cls
        _, k_idx, counts = np.unique(key, return_inverse=True, return_counts=True)
        share = counts[k_idx] / np.bincount(w_idx)[w_idx]
        margin = share[viol] * rec["instances"] / holders[viol]
        out[row["trace"], row["instances"], row["kv_mult"]].extend(margin.tolist())
    print(f"x*N/|M| over violating requests, {args.width:.0f} s replay windows, all loads pooled")
    print("trace     N  kv  violating   p10   p50   p90  share>1.25")
    for (trace, n, kv), m in sorted(out.items()):
        if m:
            a = np.asarray(m)
            print(f"{trace:9s} {n:2d} {kv:3d} {len(a):9d} {np.percentile(a, 10):5.2f} {np.median(a):5.2f} "
                  f"{np.percentile(a, 90):5.2f} {np.mean(a > 1.25):6.2f}")


if __name__ == "__main__":
    main()
