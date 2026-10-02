"""Freeze disjoint replay/calibration sets and explicit synthetic root schedules."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from cache_delay_eval.chat_replay import ReplayConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    analysis_path=Path("results/wildchat-readiness-2026-09-28/analysis.json")
    analysis=json.loads(analysis_path.read_text())
    content=Path(analysis["content_directory"])
    for name, expected in analysis["content_checksums"].items():
        if sha(content/name)!=expected:
            raise ValueError(f"WildChat content changed: {name}")
    selection=json.loads(Path("results/wildchat-readiness-2026-09-28/selection.json").read_text())
    used=set(selection["smoke"]["sample_ids"]+selection["experiment"]["sample_ids"])
    evidence=[json.loads(l) for l in Path("results/wildchat-readiness-2026-09-28/conversations.jsonl").open()]
    calibration=[r for r in evidence if r["eligible_for_replay_preparation"] and r["sample_id"] not in used][:8]
    raw={r["sample_id"]:r for r in (json.loads(l) for l in (content/"source-sample-projection.jsonl").open())}
    groups={}
    for label in ("smoke","experiment","calibration"):
        if label=="calibration":
            scripts=[dict(sample_id=r["sample_id"],shard_row=r["shard_row"],source_transcript_sha256=r["source_transcript_sha256"],
                          user_messages=[m["content"] for m in raw[r["sample_id"]]["source"]["conversation"] if m["role"]=="user"]) for r in calibration]
            path=out/"calibration-user-scripts.jsonl"
            path.write_text("".join(json.dumps(s,ensure_ascii=False,sort_keys=True)+"\n" for s in scripts))
        else:
            path=content/f"{label}-user-scripts.jsonl"
            scripts=[json.loads(l) for l in path.open()]
        seed=699010+(0 if label=="smoke" else 1 if label=="experiment" else 2)
        rng=random.Random(seed)
        # Save unit-rate offsets; calibration later chooses one multiplier before comparisons.
        unit=0.0
        arrivals={}
        for index,script in enumerate(scripts):
            if index:
                unit+=rng.expovariate(1.0)
            arrivals[script["sample_id"]]=unit
        schedule=dict(source_scripts_sha256=sha(path),seed=seed,root_arrivals_s=arrivals,
                      assumption="Synthetic unit-rate Poisson session starts; first root at zero; no WildChat timestamp used",
                      interpretation="Calibration schedule template, not a calibrated load. Freeze a scaled copy and its hash before performance comparisons.")
        (out/f"{label}-unit-rate-schedule.json").write_text(json.dumps(schedule,indent=2)+"\n")
        mock=dict(schedule,root_arrivals_s={s["sample_id"]:i*.002 for i,s in enumerate(scripts)},
                  assumption="CPU mock validation only: deterministic roots two milliseconds apart; no production arrival claim")
        (out/f"{label}-mock-schedule.json").write_text(json.dumps(mock,indent=2)+"\n")
        config=asdict(ReplayConfig(run_id=f"{label}-candidate",max_tokens=512,think_delay_s=5.0,timeout_s=600))
        (out/f"{label}-config.json").write_text(json.dumps(config,indent=2)+"\n")
        groups[label]=dict(sessions=len(scripts),turns=sum(len(s["user_messages"]) for s in scripts),sample_ids=[s["sample_id"] for s in scripts],source_scripts=str(path),source_scripts_sha256=sha(path))
    assert not (set(groups["calibration"]["sample_ids"])&used)
    groups["calibration"]["selection"]="First 8 eligible rows in frozen sample order after excluding all smoke/experiment IDs; no length/overlap ranking"
    (out/"plan.json").write_text(json.dumps(dict(groups=groups,source_analysis_sha256=sha(analysis_path),
        calibration_disjoint_from_smoke_and_experiment=True,source_messages_truncated=0,
        gpu_executed=False,model_weights_downloaded=False,
        server_prerequisites=["Pinned vLLM source/revision and resolved config","Matched tokenizer/model revision","Telemetry patch verified on CPU, then validated on server","Cold cache or verified cache-reset probe before each timed condition","Actual EOS/output cap/token counts checked on smoke"],
        script_sha256=sha(__file__)),indent=2)+"\n")
    print(json.dumps({k:{f:v[f] for f in ("sessions","turns")} for k,v in groups.items()}))


if __name__=="__main__":
    main()
