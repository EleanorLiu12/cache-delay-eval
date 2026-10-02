"""Recompute historical GPU evidence without changing the original results."""

import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "results/cloudlab-pattern-03/run-009-on"


def counter(path, name):
    values = re.findall(r"^vllm:" + re.escape(name) + r"(?:\{[^\n]*\})? ([^\s]+)$",
                        path.read_text(), re.MULTILINE)
    assert len(values) == 1, (name, values)
    return float(values[0])


def delta(name):
    return counter(RUN / "metrics-after.txt", name) - counter(RUN / "metrics-before.txt", name)


rows = [json.loads(line) for line in (RUN / "requests.jsonl").read_text().splitlines()]
meta = next(row for row in rows if row["type"] == "run_meta")
reqs = [row for row in rows if row["type"] == "request"]
assert len(reqs) == meta["requests"] == 426
assert all(r["status"] == "ok" and 0 <= r["cached_tokens"] <= r["prompt_tokens"] for r in reqs)
hits = [r["cached_tokens"] / r["prompt_tokens"] * 100 for r in reqs]
ttft = [r["ttft_ms"] / 1000 for r in reqs]
means = {}
for name in ("request_queue_time", "time_to_first_token", "request_prefill_time"):
    assert delta(name + "_seconds_count") == len(reqs)
    means[name] = delta(name + "_seconds_sum") / len(reqs)
queue_share = 100 * means["request_queue_time"] / means["time_to_first_token"]
correlation = statistics.correlation(hits, ttft)
by_id = {r["request_id"]: r for r in reqs}
children = [r for r in reqs if r["parent_request_id"]]
overlaps = sum(r["dispatched_ms"] < by_id[r["parent_request_id"]]["completed_ms"] for r in children)

samples = []
pattern = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
for line_no, line in enumerate((RUN / "server.log").read_text().splitlines(), 1):
    match = pattern.search(line)
    if match:
        stamp, running, waiting, usage = match.groups()
        samples.append(dict(line=line_no, log_timestamp=stamp, running=int(running),
                            waiting=int(waiting), kv_usage_pct=float(usage)))
assert samples
assert len({r["log_timestamp"].split()[0] for r in samples}) == 1
origin = datetime.strptime(samples[0]["log_timestamp"].split()[1], "%H:%M:%S")
for row in samples:
    stamp = datetime.strptime(row["log_timestamp"].split()[1], "%H:%M:%S")
    row["minutes_since_first_sample"] = (stamp - origin).total_seconds() / 60

inputs = [RUN / name for name in ("requests.jsonl", "metrics-before.txt", "metrics-after.txt", "server.log")]
data = dict(
    source_run=str(RUN.relative_to(ROOT)), synthetic=True, requests=len(reqs),
    cache_hit_definition="cached_tokens / prompt_tokens, measured per request",
    client_ttft_definition=meta["ttft_definition"], pearson=correlation,
    server_mean_seconds=means, queue_share_of_server_ttft_pct=queue_share,
    server_ttft_minus_queue_seconds=means["time_to_first_token"] - means["request_queue_time"],
    children=len(children), children_dispatched_before_parent_completion=overlaps,
    preemption_counter_delta=delta("num_preemptions_total"),
    peak_sampled_waiting=max(r["waiting"] for r in samples), server_log_samples=samples,
    input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
)
(OUT / "measurements.json").write_text(json.dumps(data, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 2.65), layout="constrained")
left.scatter(hits, ttft, s=10, color="#2166ac", alpha=0.50, edgecolors="none")
left.set(title="(a) Measured hits and client TTFT", xlabel="Actual cached input tokens (%)",
         ylabel="Time to first token (s)", xlim=(-2, 102), ylim=(0, max(ttft) * 1.24))
left.text(0.04, 0.94, f"All {len(reqs)} requests; Pearson r = {correlation:.3f}",
          transform=left.transAxes, va="top", fontsize=8)
right.plot([r["minutes_since_first_sample"] for r in samples], [r["waiting"] for r in samples],
           color="#b35806", linewidth=1.1, marker=".", markersize=3)
right.set(title="(b) Observed server backlog", xlabel="Minutes since first log sample (19:51:10)",
          ylabel="Waiting requests", ylim=(0, 450), xlim=(0, samples[-1]["minutes_since_first_sample"]))
right.text(0.04, 0.94, f"Peak sampled queue: {data['peak_sampled_waiting']} requests",
           transform=right.transAxes, va="top", fontsize=8)
for ax in (left, right):
    ax.grid(alpha=0.14)
    ax.set_axisbelow(True)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"gpu-evidence.{ext}", dpi=200, metadata={"Creator": "Historical GPU evidence review"})
plt.close(fig)
print(json.dumps({k: data[k] for k in ("requests", "pearson", "server_mean_seconds", "queue_share_of_server_ttft_pct", "peak_sampled_waiting")}))
