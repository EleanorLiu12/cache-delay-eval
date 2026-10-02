"""Plot all retained source cohorts and the declared sensitivity results."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / "summary.json").read_text())
cohorts = [(s["source"], row) for s in summary["sources"] if not s["synthetic"] for row in s["primary"]]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "svg.fonttype":"none",
                     "axes.spines.top":False, "axes.spines.right":False})
fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.45), layout="constrained", gridspec_kw={"width_ratios":[1, 1.2]})
labels = [f"{source.replace('qwen-', '').upper()}: {row['cohort']}" for source,row in cohorts]
values = [row["percent"] for _,row in cohorts]
y = np.arange(len(labels))
left.barh(y, values, color=["#2166ac" if source=="qwen-a" else "#888888" for source,_ in cohorts], height=.6)
for i, (_,row) in enumerate(cohorts):
    left.text(row["percent"]+1.5, i, f"{row['positive_windows']}/{row['total_windows']} ({row['percent']:.1f}%)", va="center", fontsize=8)
left.set(yticks=y, yticklabels=labels, xlim=(0, 124), xlabel="Complete windows with a structural match (%)",
         title="(a) Primary rule: 300 s, at least 4 turns")
left.set_xticks((0,25,50,75,100))
left.invert_yaxis()
left.grid(axis="x",alpha=.15); left.set_axisbelow(True)

data = json.loads((OUT / "qwen-a-sensitivity.json").read_text())
depths = [4,8,30]
for width, color in ((60,"#b35806"),(300,"#2166ac")):
    rows=[next(r for r in data if r["cohort"]=="text" and r["window_seconds"]==width and r["offset_seconds"]==0
               and r["long_root_min_tokens"]==4096 and r["min_short_depth"]==depth) for depth in depths]
    x=np.arange(3)+(0.09 if width==60 else -0.09)
    right.plot(x,[r["percent"] for r in rows],marker="o",color=color,label=f"{width}-second windows",linewidth=1.5)
    for i,r in zip(x,rows):
        right.annotate(f"{r['positive_windows']}/{r['total_windows']}",(i,r['percent']),
                       xytext=(0,9 if width==300 else -15),textcoords="offset points",ha="center",color=color,fontsize=8)
right.set(xticks=np.arange(3),xticklabels=depths,ylim=(-12,108),xlim=(-.35,2.35),
          xlabel="Minimum turns in the short-root chain",
          ylabel="Windows with a structural match (%)",title="(b) Qwen A text: rule sensitivity")
right.set_yticks((0,25,50,75,100))
right.legend(loc="upper right",fontsize=8,frameon=False)
right.grid(alpha=.15);right.set_axisbelow(True)
fig.get_layout_engine().set(rect=(0,.14,1,.86))
fig.text(.01,.025,"Source labels are not application purposes. B has only recorded singleton chains.\nWildChat: unknown (no request arrivals). Actual high-hit/high-TTFT prevalence: unknown for all cohorts.",fontsize=8)
for ext in ("png","svg"):
    fig.savefig(OUT/f"structural-rates.{ext}",dpi=200)
plt.close(fig)

# Primary-rule panel alone, for the short meeting report.
fig, ax = plt.subplots(figsize=(5.2, 2.4), layout="constrained")
ax.barh(y, values, color=["#2166ac" if source=="qwen-a" else "#888888" for source,_ in cohorts], height=.6)
for i, (_,row) in enumerate(cohorts):
    ax.text(row["percent"]+1.5, i, f"{row['positive_windows']}/{row['total_windows']} ({row['percent']:.1f}%)", va="center", fontsize=8)
ax.set(yticks=y, yticklabels=labels, xlim=(0, 124), xlabel="Complete 300 s windows with a structural match (%)")
ax.set_xticks((0,25,50,75,100))
ax.invert_yaxis()
ax.grid(axis="x",alpha=.15); ax.set_axisbelow(True)
fig.savefig(OUT/"primary-rates.png",dpi=200)
plt.close(fig)

# Retain the complete grid as a second figure, including every zero.
fig,axes=plt.subplots(1,len(cohorts),figsize=(12,6.6),layout="constrained",sharey=True)
grid=[(w,o,l) for w in (60,300) for o in (0,w//2) for l in (4096,8192,16384)]
for ax,(source,cohort) in zip(axes,cohorts):
    records=json.loads((OUT/f"{source}-sensitivity.json").read_text())
    values=[[next(r["percent"] for r in records if r["cohort"]==cohort["cohort"] and
                  (r["window_seconds"],r["offset_seconds"],r["long_root_min_tokens"],r["min_short_depth"])==(w,o,l,d))
             for d in depths] for w,o,l in grid]
    im=ax.imshow(values,vmin=0,vmax=100,cmap="Blues",aspect="auto")
    for i,row in enumerate(values):
        for j,value in enumerate(row):
            ax.text(j,i,f"{value:.0f}",ha="center",va="center",color="white" if value>55 else "black",fontsize=8)
    ax.set(xticks=range(3),xticklabels=depths,title=f"{source[-1].upper()}: {cohort['cohort']}",xlabel="Min. turns")
axes[0].set_yticks(range(len(grid)),[f"{w}s / {o}s / {l}" for w,o,l in grid])
axes[0].set_ylabel("Window / offset / long-root tokens")
fig.colorbar(im,ax=axes,label="Observed structural window matches (%)",shrink=.7)
fig.suptitle("Full prespecified sensitivity grid (rounded labels; exact counts in JSON)")
fig.savefig(OUT/"full-sensitivity.png",dpi=160)
plt.close(fig)
