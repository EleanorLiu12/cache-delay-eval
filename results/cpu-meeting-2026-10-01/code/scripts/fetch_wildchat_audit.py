"""Acquire one immutable WildChat shard and only Qwen3 tokenizer artifacts.

The first shard is a bounded audit frame, not a random sample of the corpus.
Selection within this frame is performed by the separate audit program.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from fetch_qwen_trace_a import fetch

DATASET_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
TOKENIZER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SHARD = "data/train-00000-of-00014.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    base = "https://huggingface.co/datasets/allenai/WildChat-1M"
    tree_url = f"https://huggingface.co/api/datasets/allenai/WildChat-1M/tree/{DATASET_REVISION}?recursive=true&expand=false"
    entries.append(fetch(tree_url, args.output_dir / "dataset-tree.json"))
    tree = json.loads((args.output_dir / "dataset-tree.json").read_text())
    expected = next(x for x in tree if x["path"] == SHARD)
    for remote, local in [("README.md", "dataset-README.md"),("LICENSE.md", "dataset-LICENSE.md"),(SHARD,Path(SHARD).name)]:
        entry = fetch(f"{base}/resolve/{DATASET_REVISION}/{remote}",args.output_dir / local)
        entries.append(entry)
        if remote == SHARD:
            if entry["bytes"] != expected["size"] or entry["sha256"] != expected["lfs"]["oid"]:
                raise ValueError("WildChat shard differs from pinned publisher LFS metadata")
    tokenizer = args.output_dir / "tokenizer"
    tokenizer.mkdir()
    model_base = "https://huggingface.co/Qwen/Qwen3-4B"
    for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "LICENSE", "README.md"):
        entry = fetch(f"{model_base}/resolve/{TOKENIZER_REVISION}/{filename}",tokenizer / filename)
        entry["file"] = "tokenizer/" + filename
        entries.append(entry)
    result = dict(dataset="allenai/WildChat-1M", dataset_revision=DATASET_REVISION,
                  shard=SHARD, shard_verified_against_pinned_lfs_metadata=True,
                  sampling_frame="first lexicographic original Parquet shard only; no corpus-wide representativeness claim",
                  license="ODC-BY-1.0; publisher license and card retained",
                  tokenizer="Qwen/Qwen3-4B", tokenizer_revision=TOKENIZER_REVISION,
                  weights_downloaded=False, inference_run=False,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  acquisition_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=entries)
    (args.output_dir / "manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"directory":str(args.output_dir),"shard_bytes":expected["size"],"verified":True}))


if __name__ == "__main__":
    main()
