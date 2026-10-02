"""Freeze meeting aggregates, exact analysis code, dependencies and artifact hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk:=stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-dir",type=Path,default=Path("results/cpu-meeting-2026-10-01"))
    parser.add_argument("--test-log",type=Path,required=True)
    args=parser.parse_args()
    root=Path.cwd()
    meeting=args.meeting_dir
    baseline=read(meeting/"preservation-baseline.json")
    for path,expected in baseline["files"].items():
        if digest(path)!=expected:
            raise ValueError(f"Historical file changed: {path}")
    structural={"A":read("results/qwen-trace-a-2026-09-28/analysis.json"),
                "B":read("results/qwen-trace-b-2026-09-28/analysis.json")}
    temporal={name.upper():read(Path("results/qwen-temporal-2026-09-28-exact-time")/name/"analysis.json") for name in ("a","b")}
    wild=read("results/wildchat-readiness-2026-09-28/analysis.json")
    wild_validation=read("results/wildchat-readiness-2026-09-28/validation-complete.json")
    qwen_validation=read("results/qwen-independent-validation-2026-09-28-exact-time/validation.json")
    figure_equal={}
    for name in ("a","b"):
        original=read(Path("results/qwen-temporal-2026-09-28")/name/"analysis.json")
        final=temporal[name.upper()]
        plotted=[(r["window_label"], {s:r["scopes"][s]["at_least_50pct"] for s in ("all","within","cross")}) for r in original["recency"]]
        final_plotted=[(r["window_label"], {s:r["scopes"][s]["at_least_50pct"] for s in ("all","within","cross")}) for r in final["recency"]]
        assert plotted==final_plotted and original["mixed_arrival"]==final["mixed_arrival"]
        figure_equal[name]=True
    summary=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                 meeting_date="2026-10-01",gpu_reservation=dict(node="CloudLab Wisc d7525/A30",start="2026-10-02T15:00:00-06:00",end="2026-10-02T19:00:00-06:00"),
                 new_gpu_experiments=0,new_inference_runs=0,changes_pushed=False,
                 main_cohorts={"A":structural["A"]["cohorts"]["observed_root_text"],"B":structural["B"]["cohorts"]["observed_root_api"]},
                 b_type_sensitivity={k:{f:v[f] for f in ("requests","components","types","input_tokens","output_tokens","prefix_opportunity")} for k,v in structural["B"]["cohorts"].items()},
                 temporal={k:{field:v[field] for field in ("cohort","source","methods","recency","mixed_arrival","standardized_mixed_arrival")} for k,v in temporal.items()},
                 wildchat={k:wild[k] for k in ("sampling","audit","reconstruction","replay_sets","provenance","content_checksums")},
                 wildchat_all_assistant_lengths=wild_validation["all_recorded_assistant_token_lengths"],
                 validation=dict(unit_tests=60,qwen_independent_passed=qwen_validation["passed"],wildchat_independent_passed=wild_validation["passed"],
                                 historical_files_unchanged=len(baseline["files"]),figure_values_equal_final_decimal_results=figure_equal),
                 limitations=["No measured cache hits, TTFT, completions, replica routing or queue contention in these traces.",
                              "History-window sensitivity is not TTL or finite-memory cache performance.",
                              "Descriptive co-occurrence and coarse standardization do not establish causation or significance.",
                              "WildChat uses a one-shard convenience sampling frame; generated-response replay remains to be implemented."])
    write(meeting/"summary.json",summary)
    code=meeting/"code"
    code.mkdir(exist_ok=False)
    paths=[Path(p) for p in [
        "src/cache_delay_eval/analyze_qwen_trace.py","src/cache_delay_eval/analyze_qwen_trace_b.py",
        "src/cache_delay_eval/analyze_qwen_temporal.py","src/cache_delay_eval/audit_wildchat.py",
        "src/cache_delay_eval/live_replay.py","scripts/fetch_qwen_trace_a.py","scripts/fetch_qwen_trace_b.py",
        "scripts/fetch_wildchat_audit.py","scripts/plot_qwen_cpu_comparison.py","scripts/validate_qwen_cpu.py",
        "scripts/validate_wildchat_audit.py","scripts/validate_wildchat_history.py","scripts/package_cpu_meeting.py",
        "tests/test_qwen_analysis.py","tests/test_qwen_trace_b.py","tests/test_qwen_temporal.py","tests/test_wildchat_audit.py"]]
    for path in paths:
        dest=code/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    shutil.copyfile(args.test_log,meeting/"test-suite.log")
    requirements="".join(f"{p}=={importlib.metadata.version(p)}\n" for p in ("numpy","matplotlib","pyarrow","transformers","tokenizers","jinja2"))
    (meeting/"requirements-cpu.txt").write_text(requirements)
    (meeting/"environment.txt").write_text(sys.version+"\n"+subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    directories=[meeting,Path("results/qwen-trace-b-2026-09-28"),Path("results/qwen-temporal-2026-09-28-exact-time"),Path("results/wildchat-readiness-2026-09-28"),Path("results/qwen-independent-validation-2026-09-28-exact-time")]
    artifacts={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for directory in directories for p in sorted(directory.rglob("*")) if p.is_file()}
    write(meeting/"artifact-manifest.json",dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
          repository=str(root),files=artifacts,code_sources={str(p):digest(p) for p in paths},
          original_source_manifests=["data/qwen-bailian/5f7439c51ec248a0c585f7d90a41a6f57773b912/manifest.json",
                                    "data/qwen-bailian-trace-b/5f7439c51ec248a0c585f7d90a41a6f57773b912/manifest.json",
                                    "data/wildchat-audit-2026-09-28/manifest.json"],
          historical_preservation=dict(files_checked=len(baseline["files"]),all_unchanged=True),
          manifest_excludes_itself=True))
    print(json.dumps({"artifacts":len(artifacts),"code_snapshots":len(paths),"historical_unchanged":len(baseline["files"])}))


if __name__=="__main__":
    main()
