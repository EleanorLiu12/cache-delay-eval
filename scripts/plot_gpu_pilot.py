#!/usr/bin/env python3
"""Plot actual cache-hit/TTFT associations and the candidate's arrival pattern."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

BLUE = "#2B6CB0"
PURPLE = "#805AD5"
GREY = "#667085"
STYLES = {"deep_short": (BLUE, "Deep / short"),
          "shallow_long": (PURPLE, "Shallow / long")}


def trace_label(trace):
    stem = Path(trace).stem
    if stem == "P512-negative-seed100":
        return "Mixed sessions (candidate)"
    if stem == "P512-positive-seed100":
        return "Deep / short sessions (comparison)"
    return stem


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.color": "#E4E7EC",
        "legend.frameon": False, "savefig.dpi": 200,
    })


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def number(value):
    return "undefined" if value is None else f"{value:+.2f}"


def representative_figure(analysis, output_dir):
    """Show one run per trace for the short weekly discussion report."""
    summaries = analysis["summaries"]
    capacity = min(r["capacity_blocks"] for r in summaries)
    repeat = min(r["repeat"] for r in summaries if r["capacity_blocks"] == capacity)
    selected = sorted((r for r in summaries if r["capacity_blocks"] == capacity
                       and r["repeat"] == repeat), key=lambda r: r["trace"])
    fig, axes = plt.subplots(1, len(selected), squeeze=False, sharey=True,
                             figsize=(5.6 * len(selected), 4.6), layout="constrained")
    for ax, summary in zip(axes[0], selected):
        rows = [r for r in analysis["requests"] if r["trace"] == summary["trace"]
                and r["capacity_blocks"] == capacity and r["repeat"] == repeat]
        for archetype in sorted({r["archetype"] or "unknown" for r in rows}):
            group = [r for r in rows if (r["archetype"] or "unknown") == archetype]
            color, label = STYLES.get(archetype, (GREY, archetype))
            ax.scatter([r["cached_token_fraction"] * 100 for r in group],
                       [r["ttft_on_ms"] / 1000 for r in group],
                       s=20, alpha=.55, color=color,
                       marker="^" if archetype == "shallow_long" else "o",
                       linewidths=0, label=label)
        ax.set_title(f"{trace_label(summary['trace'])}\n"
                     f"Pearson r = {number(summary['pooled_hit_ttft_pearson'])}")
        ax.set_xlim(-3, 103)
        ax.set_yscale("log")
        ax.set_xlabel("Cache hit (%)")
        ax.legend(loc="best", fontsize=8)
    axes[0, 0].set_ylabel("TTFT (seconds, log scale)")
    fig.suptitle(f"Cache hit vs. TTFT | {capacity:,} KV blocks | repeat {repeat} | cache on",
                 fontsize=13)
    save_figure(fig, output_dir, "cache-hit-ttft-representative")


def summary_figure(analysis, output_dir):
    summaries = analysis["summaries"]
    traces = sorted({r["trace"] for r in summaries})
    capacities = sorted({r["capacity_blocks"] for r in summaries})
    fig, axes = plt.subplots(len(capacities), len(traces), squeeze=False,
                             figsize=(6.2 * len(traces), 4.2 * len(capacities)))
    for i, capacity in enumerate(capacities):
        for j, trace in enumerate(traces):
            ax = axes[i, j]
            rows = [r for r in analysis["requests"]
                    if r["trace"] == trace and r["capacity_blocks"] == capacity]
            repeats = sorted({r["repeat"] for r in rows})
            archetypes = sorted({r["archetype"] or "unknown" for r in rows})
            for archetype in archetypes:
                color, label = STYLES.get(archetype, (GREY, archetype))
                for k, repeat in enumerate(repeats):
                    selected = [r for r in rows if (r["archetype"] or "unknown") == archetype
                                and r["repeat"] == repeat]
                    ax.scatter([r["cached_token_fraction"] for r in selected],
                               [r["ttft_on_ms"] / 1000 for r in selected],
                               s=13, alpha=.35, color=color, marker=("o", "x", "^")[k % 3],
                               linewidths=.5, label=f"{label}, repeat {repeat}")
            relevant = [r for r in summaries
                        if r["trace"] == trace and r["capacity_blocks"] == capacity]
            notes = [f"Repeat {r['repeat']}: Pearson {number(r['pooled_hit_ttft_pearson'])}; "
                     f"Spearman {number(r['pooled_hit_ttft_spearman'])}\n"
                     f"  Prompt-length partial Pearson {number(r['prompt_length_partial_pearson'])}"
                     for r in relevant]
            ax.text(.03, .97, "\n".join(notes), transform=ax.transAxes,
                    va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.92,
                                                     edgecolor="#D0D5DD", boxstyle="round,pad=.35"))
            ax.set_title(f"{trace_label(trace)} | {capacity:,} KV blocks", loc="left")
            ax.set_xlim(-.03, 1.03)
            ax.set_yscale("log")
            ax.set_xlabel("Actual cached-token fraction (cache on)")
            ax.set_ylabel("TTFT (s, log scale)")
            ax.legend(loc="lower left", fontsize=7)
    fig.suptitle("Trace patterns: higher cache hit and higher TTFT", x=.07,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.07, .935, "Qwen3-4B / NVIDIA A30. Statistics use each run separately; points show both repeats.",
             color=GREY)
    fig.tight_layout(rect=(0, 0, 1, .91), h_pad=2)
    save_figure(fig, output_dir, "cache-hit-ttft")


def temporal_figure(analysis, output_dir):
    # Pilot-specific representative: use the mixed trace when it is present.
    traces = sorted({r["trace"] for r in analysis["requests"]})
    trace = next((t for t in traces if Path(t).stem == "P512-negative-seed100"), traces[0])
    rows = [r for r in analysis["requests"] if r["trace"] == trace]
    capacity = min(r["capacity_blocks"] for r in rows)
    repeat = min(r["repeat"] for r in rows if r["capacity_blocks"] == capacity)
    rows = sorted((r for r in rows if r["capacity_blocks"] == capacity and r["repeat"] == repeat),
                  key=lambda r: r["scheduled_ms"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    panels = [
        ("prompt_tokens", 1000, "Prompt length\n(thousand tokens)",
         "A. Long prompts are concentrated near the start"),
        ("cached_token_fraction", 1, "Actual cached-token\nfraction",
         "B. Later deep / short requests usually have high cache hit"),
        ("ttft_on_ms", 1000, "TTFT with cache on (s)",
         "C. Later high-hit requests can also have long TTFT"),
    ]
    for ax, (field, scale, ylabel, title) in zip(axes, panels):
        for archetype in sorted({r["archetype"] or "unknown" for r in rows}):
            selected = [r for r in rows if (r["archetype"] or "unknown") == archetype]
            color, label = STYLES.get(archetype, (GREY, archetype))
            ax.scatter([r["scheduled_ms"] / 1000 for r in selected],
                       [r[field] / scale for r in selected], s=17, alpha=.6,
                       color=color, marker="^" if archetype == "shallow_long" else "o",
                       linewidths=0, label=label)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    axes[0].legend(ncol=2, loc="upper right")
    axes[1].set_ylim(-.04, 1.04)
    axes[2].set_xlabel("Scheduled arrival time (s)")
    fig.suptitle("Candidate pattern: early long work, later high-hit requests", x=.08,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(.08, .935, f"{Path(trace).stem} | {capacity:,} KV blocks | repeat {repeat} | cache on",
             color=GREY)
    fig.text(.08, .015, "Arrival order suggests a queueing explanation; these measurements do not establish the mechanism.",
             color=GREY, fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .91), h_pad=1.5)
    save_figure(fig, output_dir, "trace-pattern-timeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    analysis = json.loads(args.analysis.read_text())
    representative_figure(analysis, args.output_dir)
    summary_figure(analysis, args.output_dir)
    temporal_figure(analysis, args.output_dir)
    print(f"Wrote cache-hit/TTFT and timeline figures to {args.output_dir}")


if __name__ == "__main__":
    main()
