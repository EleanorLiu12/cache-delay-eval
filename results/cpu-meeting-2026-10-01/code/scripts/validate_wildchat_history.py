"""Check every sampled pair for partial leading-message history overlap."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    args=parser.parse_args()
    path=args.analysis_dir/"conversations.jsonl"
    rows=[json.loads(line) for line in path.open()]
    pairs=[]
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            n=0
            for a,b in zip(left["message_hashes"],right["message_hashes"]):
                if a!=b:
                    break
                n+=1
            if n:
                pairs.append(dict(left=left["sample_id"],right=right["sample_id"],
                                  shared_leading_messages=n,shared_complete_exchanges=n//2,
                                  shared_original_turn_ids=sorted(set(left["turn_identifiers"])&set(right["turn_identifiers"]))))
    result=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                pairs_checked=len(rows)*(len(rows)-1)//2,pairs_with_shared_leading_messages=pairs,
                pairs_with_shared_complete_exchange=sum(p["shared_complete_exchanges"]>0 for p in pairs),
                interpretation="A shared first user message alone can be a repeated prompt from distinct source exchanges. Preserve it as observed prefix sharing, without merging conversations.")
    with (args.analysis_dir/"history-overlap.json").open("x") as stream:
        json.dump(result,stream,indent=2)
        stream.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
