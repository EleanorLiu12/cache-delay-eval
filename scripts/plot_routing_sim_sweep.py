"""Plot the approximate-index penalty against cache size in seconds of offered traffic."""

import argparse
import json
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cache_delay_eval.routing_sim import load_trace
from run_routing_sim_sweep import TRACES

KV_BLOCK_TOKENS = 1733 * 16
SERIES = [("thinking", "#2a78d6", "o"), ("coder", "#eb6834", "s"), ("chatbot", "#1baf7a", "^"),
          ("trace-b", "#eda100", "D")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweeps", nargs="+", required=True)
    p.add_argument("--capacity", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    caps = json.load(open(args.capacity))
    mean_in = {}
    for trace, (path, length) in TRACES.items():
        for w in range(4):
            reqs = load_trace(f"data/{path}", w * length, length, 1.0)
            mean_in[trace, w * length] = statistics.fmean(r.input_len for r in reqs)
    cells = defaultdict(dict)
    for path in args.sweeps:
        for line in open(path):
            r = json.loads(line)
            key = (r["trace"], r["start_s"], r["instances"], r["kv_mult"], r["load"])
            cells[key][f"{r['policy']}/{r['index']}"] = r["ttft_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    for ax, n in zip(axes, (4, 16)):
        ax.set_facecolor(SURFACE)
        ax.axvspan(60, 600, color=GRID, alpha=0.6, lw=0)
        ax.text(190, 2.42, "estimated range of\nthe paper's setups", ha="center", va="top", fontsize=8, color=MUTED)
        ax.axhline(1.0, color=MUTED, lw=1)
        for trace, color, marker in SERIES:
            xs, ys = [], []
            for (t, start, inst, kv, load), v in cells.items():
                if t != trace or inst != n or "lmetric/dispatch" not in v:
                    continue
                max_rate, _ = caps[f"{t}|{start}|{inst}|{kv}"]
                xs.append(KV_BLOCK_TOKENS * kv / (load * max_rate / inst * mean_in[t, start]))
                ys.append(v["lmetric/dispatch"] / v["lmetric/exact"])
            if xs:
                ax.scatter(xs, ys, s=36, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.5,
                           label=trace, zorder=3)
        if n == 4:
            # A30 hot slice at 3x: 895 requests in 200 s, mean input 3,809 tokens, exact 310 ms, never-evicting 479 ms
            x = KV_BLOCK_TOKENS / (895 / 200 / 4 * 3809)
            ax.scatter([x], [479 / 310], s=90, marker="*", color=INK, zorder=4, label="A30 measurement")
            ax.annotate("A30 measurement", (x, 479 / 310), xytext=(8, 4), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_title(f"{n} instances", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("KV cache per instance, in seconds of offered input tokens", fontsize=9, color=MUTED)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("Mean TTFT, never-evicting index / exact index", fontsize=9, color=MUTED)
    axes[0].set_ylim(0.5, 2.5)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", bbox_to_anchor=(1.0, 0.85))
    fig.suptitle("LMetric with a never-evicting prefix index vs an exact one (simulated)", fontsize=11,
                 color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
