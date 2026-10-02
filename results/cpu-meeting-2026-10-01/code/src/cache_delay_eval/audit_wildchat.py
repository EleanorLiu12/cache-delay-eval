"""Audit a reproducible bounded WildChat sample without model weights or inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

from .analyze_qwen_trace import describe, fraction

SEED = 69920261001
SAMPLE_SIZE = 200
CONTEXT_LIMIT = 32768
OUTPUT_CAP = 512


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),default=str)


def inspect_conversation(row):
    """Audit the released sequence; do not sort, repair, or truncate messages."""
    messages = row["conversation"]
    roles = [m.get("role") for m in messages]
    issues = []
    if not messages or len(messages)%2 or roles != ["user","assistant"]*(len(messages)//2):
        issues.append("nonalternating_or_incomplete_roles")
    if any(not isinstance(m.get("content"),str) or not m["content"].strip() for m in messages):
        issues.append("missing_or_empty_content")
    if row.get("turn") != roles.count("user"):
        issues.append("declared_turn_count_mismatch")
    if row.get("toxic") or any(m.get("toxic") for m in messages):
        issues.append("source_toxic_flag")
    if row.get("redacted") or any(m.get("redacted") for m in messages):
        issues.append("source_redacted_flag")
    times = [m.get("timestamp") for m in messages if m.get("role")=="assistant" and m.get("timestamp") is not None]
    if any(b<a for a,b in zip(times,times[1:])):
        issues.append("assistant_timestamp_order_decreases")
    return issues


def overlap_pairs(records):
    """Detect shared source turns, duplicate transcripts and history prefixes.

    Text-identical nonprefix messages alone do not identify a shared history.
    Source IDs and whole leading role/content sequences are checked separately.
    """
    pairs = []
    for i,a in enumerate(records):
        for b in records[i+1:]:
            shared = set(a["turn_identifiers"]) & set(b["turn_identifiers"])
            aa,bb = a["message_hashes"],b["message_hashes"]
            exact = aa == bb
            prefix = bool(aa and bb and (aa == bb[:len(aa)] or bb == aa[:len(bb)]))
            if shared or prefix:
                pairs.append(dict(left=a["sample_id"],right=b["sample_id"],
                                  shared_turn_identifiers=sorted(shared),exact_transcript=exact,
                                  whole_history_prefix=prefix))
    return pairs


def nonoverlap_representatives(records,pairs):
    """Keep one whole, longest released history per detected overlap component."""
    parent = {r["sample_id"]:r["sample_id"] for r in records}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    for p in pairs:
        parent[root(p["right"])]=root(p["left"])
    grouped = {}
    for r in records:
        grouped.setdefault(root(r["sample_id"]),[]).append(r)
    return {max(g,key=lambda r:(r["messages"],-r["sample_rank"]))["sample_id"] for g in grouped.values()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--content-output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output_dir.exists() or args.content_output_dir.exists():
        raise FileExistsError("Use new output directories; retained audits are immutable")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    manifest=json.loads((args.source_dir/"manifest.json").read_text())
    shard=args.source_dir/Path(manifest["shard"]).name
    expected=next(f for f in manifest["files"] if f["file"]==shard.name)
    if digest_file(shard)!=expected["sha256"] or shard.stat().st_size!=expected["bytes"]:
        raise ValueError("Source shard checksum mismatch")
    parquet=pq.ParquetFile(shard)
    indices=random.Random(SEED).sample(range(parquet.metadata.num_rows),SAMPLE_SIZE)
    wanted=set(indices)
    selected={}
    offset=0
    columns=["conversation_hash","model","timestamp","conversation","turn","language","toxic","redacted"]
    for group in range(parquet.num_row_groups):
        n=parquet.metadata.row_group(group).num_rows
        local=[i-offset for i in sorted(wanted) if offset<=i<offset+n]
        if local:
            table=parquet.read_row_group(group,columns=columns)
            for local_index in local:
                source=table.slice(local_index,1).to_pylist()[0]
                # Keep original conversation fields needed for audit; avoid copying demographic/header data.
                source["conversation"]=[{k:m.get(k) for k in ("role","content","turn_identifier","timestamp","toxic","redacted","language")} for m in source["conversation"]]
                selected[offset+local_index]=source
        offset+=n
    tokenizer=AutoTokenizer.from_pretrained(args.source_dir/"tokenizer",local_files_only=True,trust_remote_code=False)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    args.content_output_dir.mkdir(parents=True,exist_ok=False)
    source_rows=[]
    records=[]
    requests=[]
    prompt_evidence=[]
    for rank,index in enumerate(indices):
        source=selected[index]
        sample_id=f"shard00000-row{index:05d}"
        messages=source["conversation"]
        roles=Counter(m["role"] for m in messages)
        issues=inspect_conversation(source)
        ids=[m["turn_identifier"] for m in messages if m["turn_identifier"] is not None]
        template_controls=[s for s in ["<|im_start|>","<|im_end|>","<think>","</think>","<tool_response>"] if any(s in (m["content"] or "") for m in messages)]
        if template_controls:
            issues.append("template_control_strings_in_source")
        clean=[{"role":m["role"],"content":m["content"]} for m in messages]
        source_rows.append(dict(sample_id=sample_id,shard_row=index,sample_rank=rank,source=source))
        record=dict(sample_id=sample_id,shard_row=index,sample_rank=rank,
                    conversation_hash=source["conversation_hash"],source_transcript_sha256=digest_bytes(canonical(clean).encode()),
                    source_row_projection_sha256=digest_bytes(canonical(source).encode()),
                    source_model=source["model"],language=source["language"],declared_turn=source["turn"],
                    messages=len(messages),user_turns=roles["user"],assistant_turns=roles["assistant"],roles=dict(roles),
                    turn_identifiers=ids,missing_turn_identifiers=len(messages)-len(ids),
                    repeated_turn_identifiers_within_row=len(ids)-len(set(ids)),
                    message_hashes=[digest_bytes(canonical(m).encode()) for m in clean],
                    user_timestamps_missing=sum(m["timestamp"] is None for m in messages if m["role"]=="user"),
                    assistant_timestamps_missing=sum(m["timestamp"] is None for m in messages if m["role"]=="assistant"),
                    template_control_strings=template_controls,issues=issues,reconstructed_prompt_tokens=[],
                    recorded_assistant_tokens=[],input_growth_tokens=[])
        if not any(x in issues for x in ("nonalternating_or_incomplete_roles","missing_or_empty_content")):
            prior_count=None
            for turn,pos in enumerate(range(0,len(clean),2),1):
                history=clean[:pos+1]
                rendered=tokenizer.apply_chat_template(history,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                tokens=tokenizer.apply_chat_template(history,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                direct=tokenizer.encode(rendered,add_special_tokens=False)
                if tokens!=direct:
                    raise AssertionError("Rendered prompt and direct template tokenization disagree")
                response=clean[pos+1]["content"]
                response_ids=tokenizer.encode(response,add_special_tokens=False)
                count=len(tokens)
                record["reconstructed_prompt_tokens"].append(count)
                record["recorded_assistant_tokens"].append(len(response_ids))
                if prior_count is not None:
                    record["input_growth_tokens"].append(count-prior_count)
                request=dict(sample_id=sample_id,turn=turn,user_turn_identifier=messages[pos]["turn_identifier"],
                             assistant_turn_identifier=messages[pos+1]["turn_identifier"],
                             original_user_characters=len(clean[pos]["content"]),original_assistant_characters=len(response),
                             recorded_assistant_tokens=len(response_ids),reconstructed_prompt_tokens=count,
                             reconstructed_prompt_sha256=digest_bytes(rendered.encode()),
                             reconstructed_token_ids_sha256=digest_bytes(canonical(tokens).encode()),
                             exceeds_source_prompt_plus_cap=count+OUTPUT_CAP>CONTEXT_LIMIT,
                             growth_from_previous_prompt=None if prior_count is None else count-prior_count)
                requests.append(request)
                prompt_evidence.append(dict(**request,reconstructed_prompt=rendered,reconstructed_token_ids=tokens))
                prior_count=count
            if max(record["reconstructed_prompt_tokens"])+OUTPUT_CAP>CONTEXT_LIMIT:
                issues.append("source_context_plus_512_exceeds_32768")
        records.append(record)
    pairs=overlap_pairs(records)
    representatives=nonoverlap_representatives(records,pairs)
    for r in records:
        if r["sample_id"] not in representatives:
            r["issues"].append("overlap_component_nonrepresentative")
        r["eligible_for_replay_preparation"]=not r["issues"]
    eligible=[r for r in records if r["eligible_for_replay_preparation"]]
    # Deliberately choose multi-turn smoke sessions to exercise generated-output dependencies.
    smoke=[r for r in eligible if r["user_turns"]>=2][:8]
    smoke_ids={r["sample_id"] for r in smoke}
    experiment=[r for r in eligible if r["sample_id"] not in smoke_ids][:64]
    experiment_ids={r["sample_id"] for r in experiment}
    for label,chosen in (("smoke",smoke),("experiment",experiment)):
        ids_set={r["sample_id"] for r in chosen}
        scripts=[dict(sample_id=r["sample_id"],shard_row=r["shard_row"],
                      user_messages=[m["content"] for m in selected[r["shard_row"]]["conversation"] if m["role"]=="user"],
                      source_transcript_sha256=r["source_transcript_sha256"]) for r in chosen]
        (args.content_output_dir/f"{label}-user-scripts.jsonl").write_text("".join(canonical(r)+"\n" for r in scripts))
    def save_lines(path,items):
        with path.open("x") as stream:
            for item in items:
                stream.write(canonical(item)+"\n")
    save_lines(args.content_output_dir/"source-sample-projection.jsonl",source_rows)
    save_lines(args.output_dir/"conversations.jsonl",records)
    save_lines(args.output_dir/"requests.jsonl",requests)
    with gzip.GzipFile(filename=str(args.content_output_dir/"reconstructed-prompts.jsonl.gz"),mode="xb",mtime=0) as stream:
        for r in prompt_evidence:
            stream.write((canonical(r)+"\n").encode())
    selections=dict(smoke=dict(sessions=len(smoke),turns=sum(r["user_turns"] for r in smoke),sample_ids=[r["sample_id"] for r in smoke],selection="first 8 eligible rows with >=2 user turns in the seeded sample order; deliberate validity selection"),
                    experiment=dict(sessions=len(experiment),turns=sum(r["user_turns"] for r in experiment),sample_ids=[r["sample_id"] for r in experiment],selection="first 64 remaining eligible rows in seeded sample order; no overlap-based or length-based ranking",single_turn_sessions=sum(r["user_turns"]==1 for r in experiment)),
                    sets_disjoint=not(smoke_ids&experiment_ids),no_observed_cross_set_history_overlap=not any({p["left"],p["right"]}<=(smoke_ids|experiment_ids) for p in pairs),
                    preserves_complete_released_rows=True,source_messages_truncated=0,
                    context_screen="source transcript prompts plus a proposed 512-token output cap; generated histories can differ and require a runtime context check")
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                provenance=dict(source_manifest=str(args.source_dir/"manifest.json"),source_manifest_sha256=digest_file(args.source_dir/"manifest.json"),
                                shard_sha256=digest_file(shard),dataset_revision=manifest["dataset_revision"],tokenizer_revision=manifest["tokenizer_revision"],
                                audit_sha256=digest_file(__file__),chat_template_sha256=digest_bytes(tokenizer.chat_template.encode()),
                                versions={name:importlib.metadata.version(name) for name in ("pyarrow","transformers","tokenizers","jinja2")}),
                sampling=dict(seed=SEED,method="random.Random(seed).sample(range(first_shard_rows), 200), without replacement; sampled order retained",
                              frame=manifest["shard"],frame_rows=parquet.metadata.num_rows,selected_rows=len(records),shard_row_indices=indices,
                              limitation="The first shard is a convenience frame. This is not a corpus-wide probability sample; unknown duplicates outside the 200 rows remain possible."),
                audit=dict(roles=dict(sum((Counter(r["roles"]) for r in records),Counter())),
                           declared_turns=describe(r["declared_turn"] for r in records),user_turns=describe(r["user_turns"] for r in records),
                           multi_turn_rows=fraction(sum(r["user_turns"]>1 for r in records),len(records)),
                           languages=dict(Counter(r["language"] for r in records)),source_models=dict(Counter(r["source_model"] for r in records)),
                           issues=dict(Counter(x for r in records for x in r["issues"])),eligible_rows=len(eligible),excluded_rows=len(records)-len(eligible),
                           repeated_conversation_hashes=sum(n-1 for n in Counter(r["conversation_hash"] for r in records).values()),
                           overlap_pairs=pairs,exact_duplicate_transcript_pairs=sum(p["exact_transcript"] for p in pairs),
                           missing_turn_identifiers=sum(r["missing_turn_identifiers"] for r in records),
                           repeated_turn_identifiers_within_rows=sum(r["repeated_turn_identifiers_within_row"] for r in records),
                           missing_user_timestamps=sum(r["user_timestamps_missing"] for r in records),missing_assistant_timestamps=sum(r["assistant_timestamps_missing"] for r in records)),
                reconstruction=dict(template="pinned Qwen3, enable_thinking=False, add_generation_prompt=True; no invented system message; source assistant texts used only for CPU length audit",
                                    prompt_tokens=describe(r["reconstructed_prompt_tokens"] for r in requests),
                                    recorded_assistant_tokens=describe(r["recorded_assistant_tokens"] for r in requests),
                                    recorded_assistant_over_cap=fraction(sum(r["recorded_assistant_tokens"]>OUTPUT_CAP for r in requests),len(requests)),
                                    input_growth_tokens=describe(r["growth_from_previous_prompt"] for r in requests if r["growth_from_previous_prompt"] is not None),
                                    nonincreasing_edges=sum(r["growth_from_previous_prompt"]<=0 for r in requests if r["growth_from_previous_prompt"] is not None),
                                    direct_template_tokenization_checks=len(requests),context_limit=CONTEXT_LIMIT,proposed_generated_output_cap=OUTPUT_CAP),
                replay_sets=selections,content_directory=str(args.content_output_dir),
                limitations=["Recorded timestamps are not request arrivals or user think times and were not used as schedules.",
                             "Token lengths are our Qwen3 reconstruction, not provider token counts or original production serialization.",
                             "Generated responses and semantic suitability of scripted follow-ups are unmeasured; future histories must use actual generated responses.",
                             "No model weights, inference, GPU work, or new scheduler was used."])
    result["content_checksums"]={p.name:digest_file(p) for p in args.content_output_dir.iterdir() if p.is_file()}
    (args.output_dir/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.output_dir/"selection.json").write_text(json.dumps(selections,indent=2)+"\n")
    print(json.dumps({"selected":len(records),"eligible":len(eligible),"smoke":len(smoke),"experiment":len(experiment),"issues":result["audit"]["issues"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
