"""Analyze the detector sweep: Equation 2 violation frequency, detector alarms and
filtering, and whether LMetric's losses to load-only routing line up with violations.

Equation 2 is checked per request under plain LMetric: x is the class's share of
all arrivals in the request's tumbling window, |M| the number of instances whose
cache held the class prefix (first 32 blocks) when the request was dispatched.
A request violates when 0 < |M| < N, its class has at least 5 arrivals in the
window, and x > |M|/N (equivalent to x/(1-x) > |M|/(N-|M|)). A window violates
when any of its requests does. "Top classes" restricts the check to the three
classes with the most cached tokens in the window, as in the paper's Figure 20.
"""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_CLASS = 5
TOP_CLASSES = 3
LOSS_RATIO = 1.2       # LMetric window mean TTFT at least 20% above load-only
MIN_WINDOW = 5         # requests per window for the loss comparison


def window_ids(rec, clock, width_s):
    t = np.asarray(rec["arrival_ms"]) / 1000
    if clock == "trace":
        t = rec["start_s"] + t * rec["time_scale"]
    return np.floor(t / width_s).astype(np.int64)


def violations(rec, win, run="lmetric"):
    """Per-request Equation 2 violation (all classes) and the top-class mask."""
    n = rec["instances"]
    cls = np.asarray(rec["cls"])
    holders = np.asarray(rec[run]["holders"])
    hit = np.asarray(rec[run]["hit"])
    _, w_idx, totals = np.unique(win, return_inverse=True, return_counts=True)
    key = w_idx.astype(np.int64) * (cls.max() + 1) + cls
    _, k_idx, counts = np.unique(key, return_inverse=True, return_counts=True)
    share = counts[k_idx] / totals[w_idx]
    eligible = holders >= 0
    viol = eligible & (counts[k_idx] >= MIN_CLASS) & (holders > 0) & (holders < n) & (share > holders / n)
    cached = np.bincount(k_idx, weights=hit * eligible)
    top = np.zeros(len(counts), bool)
    for w in np.unique(w_idx):
        ks = np.unique(k_idx[w_idx == w])
        best = ks[np.argsort(-cached[ks])[:TOP_CLASSES]]
        top[best[cached[best] > 0]] = True
    return viol, top[k_idx], eligible, w_idx


def cell_stats(rec):
    out = {}
    for clock in ("replay", "trace"):
        for width in (10, 60):
            win = window_ids(rec, clock, width)
            viol, top, eligible, w_idx = violations(rec, win)
            nwin = w_idx.max() + 1
            vwin = np.zeros(nwin, bool)
            vwin[w_idx[viol]] = True
            twin = np.zeros(nwin, bool)
            twin[w_idx[viol & top]] = True
            out[f"{clock}{width}"] = dict(windows=int(nwin), violating=int(vwin.sum()),
                                          violating_top=int(twin.sum()),
                                          eligible=int(eligible.sum()), violating_requests=int(viol.sum()))
    # LMetric vs load-only per replay window, and overlap with violations
    lm, ld = np.asarray(rec["lmetric"]["ttft"]), np.asarray(rec["load"]["ttft"])
    d60 = np.asarray(rec["detector60"]["ttft"])
    for width in (10, 60):
        win = window_ids(rec, "replay", width)
        viol, _, _, w_idx = violations(rec, win)
        size = np.bincount(w_idx)
        keep = size >= MIN_WINDOW
        m_lm = np.bincount(w_idx, lm) / np.maximum(size, 1)
        m_ld = np.bincount(w_idx, ld) / np.maximum(size, 1)
        m_d = np.bincount(w_idx, d60) / np.maximum(size, 1)
        vwin = np.zeros(len(size), bool)
        vwin[w_idx[viol]] = True
        loss = keep & (m_lm > LOSS_RATIO * m_ld)
        v = keep & vwin
        out[f"loss{width}"] = dict(windows=int(keep.sum()), loss=int(loss.sum()), violating=int(v.sum()),
                                   both=int((loss & v).sum()),
                                   loss_excess_ms=float((m_lm - m_ld)[loss].sum()),
                                   loss_excess_in_violating_ms=float((m_lm - m_ld)[loss & v].sum()),
                                   detector_in_loss_ms=float(m_d[loss].mean()) if loss.any() else None,
                                   lmetric_in_loss_ms=float(m_lm[loss].mean()) if loss.any() else None,
                                   load_in_loss_ms=float(m_ld[loss].mean()) if loss.any() else None)
    # requests the detector filtered, paired with the same request under plain LMetric
    for det in ("detector60", "detector10"):
        f = np.asarray(rec[det]["filtered"], bool)
        a = np.asarray(rec[det]["alarm"], bool)
        dt = np.asarray(rec[det]["ttft"])
        out[det] = dict(alarmed=int(a.sum()), filtered=int(f.sum()),
                        filtered_det_mean=float(dt[f].mean()) if f.any() else None,
                        filtered_lm_mean=float(lm[f].mean()) if f.any() else None,
                        filtered_slower=int((dt[f] > lm[f]).sum()))
    # the busiest class among requests long enough to carry a class prefix
    cls = np.asarray(rec["cls"])
    elig = np.asarray(rec["lmetric"]["holders"]) >= 0
    top = np.bincount(cls[elig]).argmax() if elig.any() else -1
    m = cls == top
    out["top_class"] = dict(share=float(m.mean()),
                            **{f"{k}_ttft": float(np.asarray(rec[k]["ttft"])[m].mean())
                               for k in ("load", "lmetric", "detector60")},
                            **{f"{k}_holders": float(np.median(np.asarray(rec[k]["holders"])[m]))
                               for k in ("load", "lmetric", "detector60")})
    out["n"] = len(lm)
    out["load_ttft_mean"] = float(ld.mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="results directory with sweep.jsonl and requests/")
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.root / "sweep.jsonl").read_text().splitlines()]
    cells = []
    for r in rows:
        with gzip.open(args.root / r["requests_file"], "rt") as f:
            stats = cell_stats(json.load(f))
        cells.append(dict(r, stats=stats))
    (args.root / "cells.json").write_text(json.dumps(cells, indent=1) + "\n")
    print(f"{len(cells)} cells written to {args.root / 'cells.json'}")


if __name__ == "__main__":
    main()
