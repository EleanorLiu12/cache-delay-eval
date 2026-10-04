"""Figure 21-style view of one cell: window mean TTFT under each policy, Equation 2
violations shaded, and how many instances hold the busiest class's prefix."""

import argparse
import gzip
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_detector_sweep import violations, window_ids

SERIES = [("load", "Load only", "#2a78d6", "o"), ("lmetric", "LMetric", "#eb6834", "s"),
          ("detector60", "LMetric + detector", "#1baf7a", "^")]
INK, MUTED, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb", "#ecebe6"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cell", help="requests/*.json.gz file")
    p.add_argument("--width", type=float, default=10)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with gzip.open(args.cell, "rt") as f:
        rec = json.load(f)
    win = window_ids(rec, "replay", args.width)
    viol, _, _, w_idx = violations(rec, win)
    size = np.bincount(w_idx)
    x = (np.arange(len(size)) + 0.5) * args.width
    vwin = np.zeros(len(size), bool)
    vwin[w_idx[viol]] = True

    cls = np.asarray(rec["cls"])
    elig = np.asarray(rec["lmetric"]["holders"]) >= 0
    top = np.bincount(cls[elig]).argmax()
    m = cls == top
    t = np.asarray(rec["arrival_ms"]) / 1000

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, facecolor=SURFACE,
                             gridspec_kw=dict(height_ratios=[3, 2]))
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for i in np.flatnonzero(vwin):
            ax.axvspan(i * args.width, (i + 1) * args.width, color=SHADE, lw=0, zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    ax = axes[0]
    for key, label, color, marker in SERIES:
        mean = np.bincount(w_idx, np.asarray(rec[key]["ttft"])) / np.maximum(size, 1)
        ax.plot(x, mean, color=color, lw=2, marker=marker, ms=6, label=label, zorder=3)
    ax.set_ylabel("Mean time to first token\nper window (ms)", color=INK, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK)

    ax = axes[1]
    for key, label, color, marker in SERIES[1:]:
        h = np.asarray(rec[key]["holders"])[m]
        ax.step(t[m], h, where="post", color=color, lw=2, label=label, zorder=3)
    share = m.mean()
    need = share * rec["instances"]
    ax.axhline(need, color=MUTED, lw=1, ls="--", zorder=2)
    ax.text(x[-1], need, f"  share x N = {need:.1f}", color=MUTED, fontsize=9, va="bottom", ha="right")
    ax.set_ylabel("Instances holding\nbusiest class prefix", color=INK, fontsize=10)
    ax.set_ylim(0, rec["instances"] + 0.5)
    ax.set_xlabel(f"Replay time (s); shaded = {args.width:.0f} s windows violating Equation 2 under LMetric",
                  color=INK, fontsize=10)
    fig.suptitle(f"{rec['trace']} trace, start {rec['start_s']} s, {rec['instances']} instances, "
                 f"{rec['kv_mult']}x cache, {rec['load_frac']:.0%} load; busiest class = {share:.0%} of requests",
                 color=INK, fontsize=10.5, x=0.01, y=0.995, ha="left")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)
    print("mean TTFT", {k: round(float(np.mean(rec[k]["ttft"])), 1) for k, *_ in SERIES},
          "busiest class", {k: round(float(np.mean(np.asarray(rec[k]["ttft"])[m])), 1) for k, *_ in SERIES})


if __name__ == "__main__":
    main()
