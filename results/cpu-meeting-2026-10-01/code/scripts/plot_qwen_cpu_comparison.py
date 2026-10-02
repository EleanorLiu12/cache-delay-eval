"""Create three meeting figures from observed A/B CPU aggregates and evidence."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

COLORS = {"A": "#21618c", "B": "#bd572b"}
SCOPE_COLORS = {"all": "#21618c", "within": "#16847c", "cross": "#8960a4"}


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#adb5bd", "axes.axisbelow": True,
        "text.color": "#203342", "axes.labelcolor": "#203342",
        "legend.frameon": False, "legend.fontsize": 9,
        "svg.fonttype": "none", "savefig.dpi": 200,
    })


def save(fig, output, stem):
    for ext in ("png", "svg"):
        path = output / f"{stem}.{ext}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, facecolor="white")
        print(path)
    plt.close(fig)


def ecdf(ax, values, label, color):
    values, counts = np.unique(values, return_counts=True)
    y = np.cumsum(counts) / sum(counts)
    ax.step(np.r_[values[0], values], np.r_[0, y], where="post",
            color=color, linewidth=2, label=label)


def structure(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    fig.subplots_adjust(left=.065, right=.975, top=.76, bottom=.26, wspace=.30)
    fig.suptitle("Different request structures in the released Trace A and Trace B",
                 x=.065, y=.95, ha="left", fontsize=17, fontweight="bold")
    fig.text(.065, .88, "Trace A: observed text chains  |  Trace B: observed API chains  |  Cohorts are analyzed separately", color="#526171")
    for name, data in rows.items():
        color = COLORS[name]
        label = f"{name} ({len(data):,} requests)"
        ecdf(axes[0], [r["input_length"] for r in data], label, color)
        components = Counter(r["component_id"] for r in data)
        ecdf(axes[1], list(components.values()), f"{name} ({len(components):,} chains)", color)
        counts = np.zeros(120, dtype=int)
        for row in data:
            index = int(row["timestamp"] // 60)
            if not 0 <= index < 120:
                raise ValueError("Expected original two-hour observation window")
            counts[index] += 1
        axes[2].step(np.arange(121), np.r_[counts / 60, counts[-1] / 60], where="post",
                     color=color, label=name, linewidth=1.8)
    for ax, title, xlabel in zip(axes, ["Input length", "Observed chain size", "Arrival rate"],
                                ["Input tokens (log scale)", "Requests per observed chain", "Minutes from trace start"]):
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#e7ebef")
        ax.legend(loc="best")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("Cumulative share")
    axes[2].set_xlim(0, 120)
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("Requests / second (60-second bins)")
    fig.text(.065, .115, "Observed chains follow released parent links; the release does not certify complete conversations.", fontsize=9)
    fig.text(.065, .07, "These differences describe two released samples. Cluster arrivals do not specify utilization on one A30.", fontsize=9)
    save(fig, output, "01-trace-structure")


def recency(analyses, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=.08, right=.975, top=.75, bottom=.25, wspace=.22)
    fig.suptitle("How much prefix overlap survives a short history window?",
                 x=.08, y=.955, ha="left", fontsize=17, fontweight="bold")
    fig.text(.08, .887, "Share of all requests with at least 50% full-block prefix overlap; every curve uses its full cohort denominator", fontsize=10)
    labels = {"all": "Any chain", "within": "Within chain", "cross": "Other chains"}
    styles = {"all": "-", "within": "--", "cross": ":"}
    for ax, (name, analysis) in zip(axes, analyses.items()):
        entries = analysis["recency"]
        assert [r["window_label"] for r in entries] == ["1", "10", "60", "300", "unlimited"]
        for scope in ("all", "within", "cross"):
            values = [r["scopes"][scope]["at_least_50pct"]["fraction"] for r in entries]
            ax.plot(range(5), values, marker="o", color=SCOPE_COLORS[scope],
                    linestyle=styles[scope], linewidth=2, label=labels[scope])
            if scope == "all":
                for i, value in enumerate(values):
                    ax.annotate(f"{value:.1%}", (i, value), xytext=(0, 9),
                                textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(f"Trace {name} | n = {analysis['cohort']['requests']:,}", loc="left", pad=13)
        ax.set_xticks(range(5), ["1 s", "10 s", "60 s", "300 s", "Unlimited"])
        ax.set_xlabel("Maximum age of an earlier input")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#e7ebef")
        ax.legend(loc="upper left", ncol=1)
    axes[0].set_ylabel("Requests with overlap ≥50% / all requests")
    fig.text(.08, .13, "Only strictly earlier arrivals qualify; the lower window boundary is included. Final partial blocks are excluded.", fontsize=9)
    fig.text(.08, .085, "History-window sensitivity measures available input structure. It does not measure cache TTL, residency, completion, or hits.", fontsize=9)
    fig.text(.08, .04, "Within-chain and cross-chain curves can share matches and must not be added.", fontsize=9)
    save(fig, output, "02-prefix-recency")


def temporal(analyses, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=.085, right=.97, top=.81, bottom=.205, hspace=.48, wspace=.28)
    fig.suptitle("Does recent long-request context differ by historical overlap?",
                 x=.085, y=.965, ha="left", fontsize=16, fontweight="bold")
    fig.text(.085, .914, "Difference in request prevalence: overlap ≥50% minus overlap <50% (disjoint request groups)", fontsize=10)
    fig.text(.085, .875, "Positive: more frequent in high-overlap requests. Negative: more frequent in low-overlap requests.", fontsize=10)
    for col, (name, analysis) in enumerate(analyses.items()):
        for row, precursor in enumerate(("any_long", "low_overlap_long")):
            ax = axes[row, col]
            for definition, label, color, offset in (
                ("cohort_p90", "Cohort p90", "#21618c", -.07),
                ("absolute_4096", "Common ≥4,096 tokens", "#bd572b", .07),
            ):
                entries = sorted([r for r in analysis["mixed_arrival"]
                                  if r["long_definition"] == definition and r["precursor"] == precursor],
                                 key=lambda r: r["window_seconds"])
                assert [r["window_seconds"] for r in entries] == [1, 10, 60]
                values = [r["request_difference_pp"] for r in entries]
                ax.plot(np.arange(3) + offset, values, marker="o", linewidth=1.8,
                        color=color, label=label)
                for i, value in enumerate(values):
                    ax.annotate(f"{value:+.2f}", (i + offset, value),
                                textcoords="offset points", xytext=(0, 9 if offset < 0 else -15),
                                ha="center", fontsize=8.5, color=color)
            ax.axhline(0, color="#677783", linewidth=.9)
            ax.set_xlim(-.3, 2.3)
            ax.set_xticks(range(3), ["1 s", "10 s", "60 s"])
            ax.set_xlabel("Preceding arrival window")
            ax.set_ylabel("High − low overlap (percentage points)")
            ax.grid(axis="y", color="#e7ebef")
            ax.margins(y=.35)
            subtitle = "Any long predecessor" if row == 0 else "Predecessor overlap <25%"
            ax.set_title(f"Trace {name} | {subtitle}", loc="left", pad=12)
            if row == 0:
                ax.legend(loc="best", fontsize=8.5)
    fig.text(.085, .115, "Predecessors must arrive strictly earlier and belong to another observed chain. Historical overlap uses unlimited input history.", fontsize=9)
    fig.text(.085, .074, "These are observed differences, with no independence assumption or significance test. Stratified counts are retained in analysis.json.", fontsize=9)
    fig.text(.085, .034, "A temporal association does not identify a shared queue, unfinished prefill, contention, causation, or latency.", fontsize=9)
    save(fig, output, "03-temporal-comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    analyses = {name.upper(): read_json(args.temporal_dir / name / "analysis.json") for name in ("a", "b")}
    rows = {name.upper(): read_rows(args.temporal_dir / name / "requests.jsonl.gz") for name in ("a", "b")}
    for name in analyses:
        assert len(rows[name]) == analyses[name]["cohort"]["requests"]
    style()
    structure(rows, args.output_dir)
    recency(analyses, args.output_dir)
    temporal(analyses, args.output_dir)


if __name__ == "__main__":
    main()
