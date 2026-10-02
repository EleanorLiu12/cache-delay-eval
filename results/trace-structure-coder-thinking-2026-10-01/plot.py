"""Primary-rule window shares for all four Qwen-Bailian traces, labeled by trace, scenario and type label."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
rows = json.loads((OUT / "summary.json").read_text())["rows"]
SHORT = {"Qwen-Bailian Trace A": "Trace A (general chat)", "Qwen-Bailian Trace B": "Trace B (API automation)",
         "Qwen-Bailian Coder trace": "Coder (code generation)", "Qwen-Bailian Thinking trace": "Thinking (long reasoning)"}
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.2, 3.0), layout="constrained")
labels = [f"{SHORT[r['trace']]}: {r['type_label']}" for r in rows]
values = [r["percent"] for r in rows]
low = [r["percent"] - r["wilson95_percent"][0] for r in rows]
high = [r["wilson95_percent"][1] - r["percent"] for r in rows]
y = np.arange(len(rows))
ax.barh(y, values, color="#2166ac", height=.6)
sampled = [i for i, r in enumerate(rows) if not r["trace"].endswith("Trace B")]
ax.errorbar([values[i] for i in sampled], y[sampled], xerr=[[low[i] for i in sampled], [high[i] for i in sampled]],
            fmt="none", ecolor="#333333", elinewidth=.8, capsize=2)
for i, r in enumerate(rows):
    ax.text((r["wilson95_percent"][1] if i in sampled else r["percent"]) + 2, i, f"{r['positive_windows']}/{r['total_windows']} ({r['percent']:.1f}%)",
            va="center", fontsize=8)
ax.set(yticks=y, yticklabels=labels, xlim=(0, 135), xlabel="5-minute windows with the pattern (%)")
ax.set_xticks((0, 25, 50, 75, 100))
ax.invert_yaxis()
ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
fig.savefig(OUT / "primary-rates.png", dpi=200)
