#!/usr/bin/env python3
"""Plot observed Qwen Trace A structure, historical reuse, and arrival context."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np


BLUE = "#23628C"
TEAL = "#16847C"
ORANGE = "#C57B32"
PURPLE = "#8064AA"
GREY = "#526171"
COHORT = "observed_root_text"


def read_json(path):
    return json.loads(path.read_text())


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#AAB3BD",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "text.color": "#172B3A",
        "axes.labelcolor": "#273C4A",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
        "savefig.dpi": 220,
    })


def ecdf(ax, values, label, color, linestyle="-"):
    # Consolidating repeated values preserves the exact empirical distribution
    # while keeping the editable SVG small.
    values, counts = np.unique(np.asarray(values, dtype=float), return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum()
    ax.step(np.r_[values[0], values], np.r_[0, cumulative], where="post",
            label=label, color=color, linestyle=linestyle, linewidth=2)


def format_ecdf(ax):
    ax.set_ylim(0, 1.025)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Cumulative share of text requests")
    ax.grid(axis="y", color="#E6EAF0", linewidth=.8)


def validate_inputs(analysis, rows, arrivals):
    cohort = analysis["cohorts"][COHORT]
    if len(rows) != cohort["requests"]:
        raise ValueError("Text request file and analysis disagree on population")
    for name, bins in arrivals.items():
        expected = analysis["cohorts"][name]["requests"]
        if sum(row["requests"] for row in bins) != expected:
            raise ValueError(f"Arrival bins disagree with {name} population")
        if any(row["duration_s"] <= 0 for row in bins):
            raise ValueError("Arrival bin duration must be positive")
    if any(row["input_length"] <= 0 or row["output_length"] <= 0 for row in rows):
        raise ValueError("Logarithmic length plot requires positive token counts")
    return cohort


def plot(analysis_dir, output_dir):
    analysis = read_json(analysis_dir / "analysis.json")
    with (analysis_dir / f"requests-{COHORT}.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    arrivals = {
        name: read_json(analysis_dir / f"arrival-bins-{name}.json")["60"]
        for name in ("all_types", COHORT)
    }
    cohort = validate_inputs(analysis, rows, arrivals)
    comparisons = sorted(
        (row for row in cohort["prevalence_grid"]
         if row["reuse_threshold"] == .5
         and row["preceding_long_requires_reuse_below_25pct"] is False
         and row["window_seconds"] in (1, 10, 60)),
        key=lambda row: row["window_seconds"],
    )
    if [row["window_seconds"] for row in comparisons] != [1, 10, 60]:
        raise ValueError("Expected exactly the 1/10/60-second comparison rows")
    for row in comparisons:
        if row["long_context_all_requests"]["denominator"] != len(rows):
            raise ValueError("Co-occurrence baseline must include all text requests")

    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4))
    fig.subplots_adjust(left=.075, right=.975, bottom=.19, top=.845,
                        wspace=.27, hspace=.48)
    fig.suptitle("Qwen-Bailian Trace A | observed workload structure",
                 x=.075, y=.968, ha="left", fontsize=17, fontweight="bold")
    duration_min = max(row["start_s"] + row["duration_s"]
                       for row in arrivals["all_types"]) / 60
    all_count = analysis["cohorts"]["all_types"]["requests"]
    fig.text(.075, .928,
             f"{duration_min:g}-minute window  ·  {all_count:,} requests across all types"
             f"  ·  {len(rows):,} text requests in components with observed roots",
             fontsize=10, color=GREY)

    ax = axes[0, 0]
    ecdf(ax, [row["input_length"] for row in rows], "Input tokens", BLUE)
    ecdf(ax, [row["output_length"] for row in rows], "Output tokens", ORANGE, "--")
    ax.set_xscale("log")
    ax.set_xlabel("Token count (log scale)")
    ax.set_title("A. Text request lengths", loc="left", pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    format_ecdf(ax)
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    for field, label, color, linestyle in (
        ("potential_reuse_fraction", "Any earlier text request", BLUE, "-"),
        ("within_component_reuse_fraction", "Within parent component", TEAL, "--"),
        ("cross_component_reuse_fraction", "Across parent components", PURPLE, ":"),
    ):
        ecdf(ax, [row[field] for row in rows], label, color, linestyle)
    ax.set_title("B. Historical full-block prefix reuse", loc="left", pad=12)
    ax.set_xlabel("Matched historical prefix / input length")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    format_ecdf(ax)
    ax.legend(loc="lower right")
    ax.text(.02, .96, "Structural opportunity; not measured cache hits",
            transform=ax.transAxes, va="top", fontsize=9, color=GREY)

    ax = axes[1, 0]
    for name, label, color, linestyle in (
        ("all_types", "All request types", GREY, "-"),
        (COHORT, "Text requests", BLUE, "-"),
    ):
        bins = arrivals[name]
        times = [row["start_s"] / 60 for row in bins]
        rates = [row["requests"] / row["duration_s"] for row in bins]
        end = (bins[-1]["start_s"] + bins[-1]["duration_s"]) / 60
        ax.step(times + [end], rates + [rates[-1]], where="post",
                label=label, color=color, linestyle=linestyle, linewidth=1.65)
    ax.set_title("C. Arrivals throughout the observation window", loc="left", pad=12)
    ax.set_xlim(0, duration_min)
    ax.set_xticks(np.arange(0, duration_min + 1, 20))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Minutes since first arrival")
    ax.set_ylabel("Requests / second (60-second bins)")
    ax.grid(axis="y", color="#E6EAF0", linewidth=.8)
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    positions = np.arange(len(comparisons))
    baseline = np.array([row["long_context_all_requests"]["fraction"]
                         for row in comparisons])
    conditional = np.array([row["event_given_high_reuse"]["fraction"]
                            for row in comparisons])
    high_count = comparisons[0]["event_given_high_reuse"]["denominator"]
    for x, values, color, label in (
        (positions - .18, baseline, "#AFBDC7", f"All text requests (n = {len(rows):,})"),
        (positions + .18, conditional, BLUE, f"Historical reuse ≥50% (n = {high_count:,})"),
    ):
        bars = ax.bar(x, values, width=.32, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{value:.2%}" for value in values],
                     padding=4, fontsize=8.5)
    differences = (conditional - baseline) * 100
    ax.set_xticks(positions, [f"{row['window_seconds']} s\nΔ = {delta:+.2f} pp"
                              for row, delta in zip(comparisons, differences)])
    # Leave a clear legend band above the tallest bars.
    ax.set_ylim(0, 1.40)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("D. Recent long requests: similar co-occurrence rates",
                 loc="left", pad=12)
    ax.set_ylabel("Share with a recent long request\nfrom another parent component")
    ax.set_xlabel("Preceding arrival window; Δ = conditional − baseline")
    ax.grid(axis="y", color="#E6EAF0", linewidth=.8)
    ax.legend(loc="upper left", fontsize=8.5)

    long_at_least = cohort["thresholds"]["long_input_at_least"]
    short_at_most = cohort["thresholds"]["short_input_at_most"]
    block_size = analysis["methods"]["block_size"]
    fig.text(.075, .087,
             f"Text thresholds: long input ≥{long_at_least:,} tokens (p90); short input "
             f"≤{short_at_most:,} tokens (median). Panel D includes all current input lengths.",
             fontsize=9, color=GREY)
    fig.text(.075, .061,
             f"Reuse uses strictly earlier arrivals and full {block_size}-token blocks; "
             "partial blocks are excluded. Parent components follow observed parent links.",
             fontsize=9, color=GREY)
    fig.text(.075, .035,
             "No completion times, cache-residency measurements, or latency measurements. "
             "Co-occurrence does not establish queueing. pp = percentage points.",
             fontsize=9, color=GREY)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / f"qwen-trace-a-structure.{extension}"
        fig.savefig(path, facecolor="white", bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.analysis_dir, args.output_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
