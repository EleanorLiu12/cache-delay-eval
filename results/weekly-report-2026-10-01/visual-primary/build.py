"""Plot the primary structural result with explicit dataset provenance."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
path = ROOT / "results/trace-structure-2026-10-01/summary.json"
summary = json.loads(path.read_text())
sources = {s["source"]: s for s in summary["sources"]}
order = [("qwen-a", "text"), ("qwen-a", "search"), ("qwen-a", "file"),
         ("qwen-a", "image"), ("qwen-b", "api"), ("qwen-b", "text")]
rows = [next(r for r in sources[source]["primary"] if r["cohort"] == label) for source,label in order]
assert [r["positive_windows"] for r in rows] == [20,5,4,0,0,0]
assert all(r["total_windows"] == 23 for r in rows)

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "svg.fonttype":"none",
                     "axes.spines.top":False, "axes.spines.right":False})
fig, ax = plt.subplots(figsize=(9.2,4.35))
fig.subplots_adjust(left=.19, right=.985, bottom=.28, top=.77)
fig.text(.035,.95,"Qwen-Bailian: observed structural matches",fontsize=15,fontweight="bold",va="top")
fig.text(.035,.865,"Alibaba Cloud Bailian production logs  |  Trace A: chat service  |  Trace B: API task automation",fontsize=9)
fig.text(.035,.81,"Primary rule: 300-second windows; at least four turns in the short-root chain",fontsize=9,color="#444444")
ys = [0,1,2,3,4.45,5.45]
ax.barh(ys,[r["percent"] for r in rows],height=.58,color=["#2463a6"]*4+["#7d8791"]*2)
for y,r in zip(ys,rows):
    ax.text(r["percent"]+1.3,y,f"{r['positive_windows']}/{r['total_windows']}  ({r['percent']:.1f}%)",va="center",fontsize=10)
ax.set(yticks=ys,yticklabels=[f"Trace {s[-1].upper()} / {label}" for s,label in order],
       xlim=(0,120),xlabel="Complete windows containing a structural match (%)")
ax.set_xticks([0,25,50,75,100]);ax.invert_yaxis()
ax.axhline(3.7,color="#dddddd",lw=.7)
ax.grid(axis="x",alpha=.16);ax.set_axisbelow(True)
fig.text(.035,.14,"Denominator: 23 complete five-minute windows per label, [0, 6900) seconds; final partial window excluded.",fontsize=8)
fig.text(.035,.096,"B records only singleton chains. These source labels do not identify application purpose; actual hits and TTFT are absent.",fontsize=8)
fig.text(.035,.047,"Source: github.com/alibaba-edu/qwen-bailian-usagetraces-anon  |  revision 5f7439c51ec2  |  A: 43,058; B: 172,800 requests",fontsize=8,color="#444444")
for ext in ("png","svg"):
    fig.savefig(OUT/f"structural-matches.{ext}",dpi=200)
plt.close(fig)
data = dict(source_summary=str(path.relative_to(ROOT)),source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            publisher="Alibaba",service="Alibaba Cloud Bailian Qwen serving",
            repository="https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
            revision="5f7439c51ec248a0c585f7d90a41a6f57773b912",rows=rows,
            source_files={name:dict(path=sources[name]["path"],sha256=sources[name]["sha256"]) for name in ("qwen-a","qwen-b")})
(OUT/"figure-data.json").write_text(json.dumps(data,indent=2)+"\n")
print("Primary figure generated; dataset, publisher, releases, and denominator are explicit.")
