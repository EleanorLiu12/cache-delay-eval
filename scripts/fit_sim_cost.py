"""Fit the routing simulator's step-cost model to profile_engine.py output."""

import argparse
import json

import numpy as np

BUDGET = 2048


def features(rows):
    X, y = [], []
    for r in rows:
        if r["type"] == "prefill":
            if r.get("cached_tokens") is None:
                continue
            done, total = r["cached_tokens"], r["length"]
            f = np.zeros(5)
            while done < total:
                n = min(BUDGET, total - done)
                f += [1, n, n * (done + n / 2), 0, 0]
                done += n
            f[4] = 1
            X.append(f)
            y.append(r["ttft_ms"])
        elif r["type"] == "decode":
            b, ctx = r["batch"], r["context"]
            X.append([1, b, 0, b * ctx, 0])
            y.append(r["step_ms_median"])
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = [json.loads(l) for path in args.profiles for l in open(path)]
    X, y = features(rows)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rel = np.abs(pred - y) / y
    fit = dict(zip(["base", "per_token", "prefill_pair", "decode_ctx", "overhead"], coef.tolist()))
    out = dict(cost=fit, points=len(y), median_rel_error=float(np.median(rel)),
               max_rel_error=float(rel.max()), profiles=args.profiles)
    print(json.dumps(out, indent=1))
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
