"""Build a compact research figure from retained, validated measurements."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
inputs = {}


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    inputs[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


struct = read("results/trace-structure-2026-10-01/summary.json")
sens = read("results/trace-structure-2026-10-01/qwen-a-sensitivity.json")
validation = read("results/trace-structure-2026-10-01/validation.json")
assert validation["status"] == "passed"
gpu = read("results/weekly-report-2026-10-01/measurements.json")
oracle = []
for suffix in ("slack0", "validated", "slack10"):
    base = "results/eviction-oracle-cpu-2026-09-29-" + suffix
    instance, result = read(base + "/instance.json"), read(base + "/result.json")
    assert result["exact"] and not result["hardware_validated"]
    stock, best = result["baseline"]["mean_ttft"], result["best"]["mean_ttft"]
    oracle.append(dict(budget=instance["slack_percent"], stock_ticks=stock, oracle_ticks=best,
                       relative_regret_pct=100 * (stock - best) / stock))

by_source = {s["source"]: {r["cohort"]: r for r in s["primary"]} for s in struct["sources"]}
order = [("qwen-a", "text"), ("qwen-a", "search"), ("qwen-a", "file"),
         ("qwen-a", "image"), ("qwen-b", "api"), ("qwen-b", "text")]
rates = [by_source[source][label] for source, label in order]
BLUE, ORANGE, GRAY = "#2463a6", "#b45712", "#737b84"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.4), layout="constrained",
                         gridspec_kw={"height_ratios": [1.3, 1]})
a, b, c, d = axes.flat

ys = np.arange(len(rates))
a.barh(ys, [r["percent"] for r in rates], height=.58, color=[BLUE]*4 + [GRAY]*2)
for i, r in enumerate(rates):
    a.text(r["percent"] + 1.5, i, f"{r['positive_windows']}/{r['total_windows']}  ({r['percent']:.1f}%)",
           va="center", fontsize=8)
a.set(yticks=ys, yticklabels=[f"{s[-1].upper()}: {label}" for s, label in order], xlim=(0, 120),
      xlabel="Matching windows (%)", title="(a) Structure: 300 s, at least 4 turns")
a.set_xticks((0, 25, 50, 75, 100)); a.invert_yaxis()

for width, color, shift in ((300, BLUE, -.07), (60, ORANGE, .07)):
    rows = [next(r for r in sens if r["cohort"] == "text" and r["window_seconds"] == width
                 and r["offset_seconds"] == 0 and r["long_root_min_tokens"] == 4096
                 and r["min_short_depth"] == depth) for depth in (4, 8, 30)]
    x = np.arange(3) + shift
    b.plot(x, [r["percent"] for r in rows], "o-", color=color, label=f"{width}-second windows", markersize=5)
    for pos, r in zip(x, rows):
        b.annotate(f"{r['positive_windows']}/{r['total_windows']}", (pos, r["percent"]),
                   xytext=(0, 8 if width == 300 else -13), textcoords="offset points", ha="center",
                   fontsize=8, color=color)
b.set(xticks=(0,1,2), xticklabels=(4,8,30), xlim=(-.35,2.35), ylim=(-16,111),
      yticks=(0,25,50,75,100), ylabel="Matching windows (%)", xlabel="Minimum turns in the short-root chain",
      title="(b) A text: sensitivity to the rule")
b.legend(loc="upper right", frameon=False, fontsize=8)

queue = gpu["server_mean_seconds"]["request_queue_time"]
total = gpu["server_mean_seconds"]["time_to_first_token"]
remainder = total - queue
c.barh([0], [queue], color=BLUE, height=.38)
c.barh([0], [remainder], left=[queue], color=ORANGE, height=.38)
c.text(queue / 2, 0, f"Before first scheduling: {queue:.2f} s", color="white", ha="center", va="center", fontsize=9)
c.text(.01, .93, f"Queueing = {100*queue/total:.2f}% of mean server TTFT", transform=c.transAxes, va="top", fontweight="bold")
c.text(.01, .72, f"Total: {total:.2f} s  |  Remaining interval: {remainder:.2f} s", transform=c.transAxes, fontsize=8)
c.text(.01, .06, f"A30; 426 synthetic requests; hit–TTFT r = {gpu['pearson']:+.3f}\n374/374 children dispatched before parent completion.",
       transform=c.transAxes, fontsize=8, va="bottom")
c.set(yticks=[], xlim=(0,610), ylim=(-.8,.8), xlabel="Mean server time to first token (s)",
      title="(c) Historical GPU: waiting dominates")
c.spines["left"].set_visible(False)

d.plot([r["budget"] for r in oracle], [r["relative_regret_pct"] for r in oracle], "o-", color=BLUE, markersize=6)
d.set(xlim=(-.5,10.5), xticks=(0,5,10), ylim=(-.15,1.0), yticks=(0,.5,1),
      xlabel="Allowed progress slack (%)", ylabel="Relative regret (%)",
      title="(d) CPU Oracle: one window, stock baseline")
d.text(.04, .85, "0% at all three tested budgets", transform=d.transAxes, fontweight="bold")
d.text(.04, .62, "Stock = Oracle = 2,525 model ticks\nSynthetic costs; GPU and other policies unmeasured",
       transform=d.transAxes, fontsize=8)
for ax in (a,b,c,d):
    ax.grid(axis="x" if ax in (a,c) else "both", alpha=.13)
    ax.set_axisbelow(True)
fig.get_layout_engine().set(rect=(0,.035,1,.965), h_pad=.09, w_pad=.09)
fig.text(.01,.006,"Source labels are not application purposes. B has only recorded singleton chains. Full high-hit/high-TTFT prevalence remains unknown.",fontsize=8)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"results-overview.{ext}", dpi=200)
plt.close(fig)
(OUT / "figure-data.json").write_text(json.dumps(dict(
    structural_primary=rates, oracle=oracle, gpu_mean_ttft_seconds=total,
    gpu_mean_queue_seconds=queue, source_sha256=inputs,
    note="Presentation of retained results; no new dataset analysis or serving experiment"), indent=2) + "\n")
print("Four panels generated from validated structural, GPU, and CPU evidence.")
