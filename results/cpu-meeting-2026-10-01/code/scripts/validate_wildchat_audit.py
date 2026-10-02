"""Independently compare sampled content/IDs with immutable Parquet rows."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def load_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--output-name",default="validation.json")
    args=parser.parse_args()
    a=json.loads((args.analysis_dir/"analysis.json").read_text())
    content=Path(a["content_directory"])
    source=load_lines(content/"source-sample-projection.jsonl")
    records=load_lines(args.analysis_dir/"conversations.jsonl")
    requests=load_lines(args.analysis_dir/"requests.jsonl")
    parquet=pq.ParquetFile(args.source_dir/"train-00000-of-00014.parquet")
    indices=random.Random(69920261001).sample(range(parquet.metadata.num_rows),200)
    assert [r["shard_row"] for r in source]==indices
    mapping={r["shard_row"]:r for r in source}
    offset=0
    comparisons=0
    for g in range(parquet.num_row_groups):
        n=parquet.metadata.row_group(g).num_rows
        chosen=[i for i in mapping if offset<=i<offset+n]
        if chosen:
            table=parquet.read_row_group(g,columns=["conversation","conversation_hash","turn"])
            for index in chosen:
                original=table.slice(index-offset,1).to_pylist()[0]
                projected=mapping[index]["source"]
                assert original["turn"]==projected["turn"] and original["conversation_hash"]==projected["conversation_hash"]
                assert len(original["conversation"])==len(projected["conversation"])
                for m,p in zip(original["conversation"],projected["conversation"]):
                    for key in ("role","content","turn_identifier","toxic","redacted"):
                        assert m[key]==p[key]
                    comparisons+=1
        offset+=n
    tokenizer=AutoTokenizer.from_pretrained(args.source_dir/"tokenizer",local_files_only=True,trust_remote_code=False)
    token_checks=0
    with gzip.open(content/"reconstructed-prompts.jsonl.gz","rt") as stream:
        for line in stream:
            r=json.loads(line)
            ids=tokenizer.encode(r["reconstructed_prompt"],add_special_tokens=False)
            assert ids==r["reconstructed_token_ids"] and len(ids)==r["reconstructed_prompt_tokens"]
            token_checks+=1
    paired=0
    repeated_exchanges=0
    empty_roles=Counter()
    assistant_lengths=[]
    for r in source:
        messages=r["source"]["conversation"]
        ids=[]
        for i in range(0,len(messages),2):
            paired+=messages[i]["turn_identifier"]==messages[i+1]["turn_identifier"]
            ids.append(messages[i]["turn_identifier"])
        repeated_exchanges+=len(ids)-len(set(ids))
        empty_roles.update(m["role"] for m in messages if not (m["content"] or "").strip())
        assistant_lengths.extend(len(tokenizer.encode(m["content"],add_special_tokens=False)) for m in messages if m["role"]=="assistant")
    by_id={r["sample_id"]:r for r in source}
    selections={}
    for label in ("smoke","experiment"):
        scripts=load_lines(content/f"{label}-user-scripts.jsonl")
        assert [r["sample_id"] for r in scripts]==a["replay_sets"][label]["sample_ids"]
        source_turns=0
        for script in scripts:
            expected=[m["content"] for m in by_id[script["sample_id"]]["source"]["conversation"] if m["role"]=="user"]
            assert script["user_messages"]==expected
            source_turns+=len(expected)
        selections[label]=dict(sessions=len(scripts),all_original_user_turns_preserved=source_turns)
    result=dict(passed=True,sampled_rows_compared_to_original=200,source_messages_compared=comparisons,
                reconstructed_prompts_retokenized=token_checks,user_assistant_pairs_sharing_turn_identifier=paired,
                repeated_exchange_identifiers_within_conversations=repeated_exchanges,
                interpretation="Each user/assistant exchange shares one turn_identifier; repeated message-level IDs are expected and do not indicate duplicate exchanges.",
                empty_content_messages_by_role=dict(empty_roles),selections=selections,
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    values=sorted(assistant_lengths)
    result["all_recorded_assistant_token_lengths"]={"count":len(values),"minimum":min(values),"maximum":max(values),"mean":sum(values)/len(values),
        **{f"p{q}":values[math.ceil(q*len(values)/100)-1] for q in (50,90,95,99)},"over_proposed_512_cap":sum(v>512 for v in values)}
    with (args.analysis_dir/args.output_name).open("x") as stream:
        json.dump(result,stream,indent=2)
        stream.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
