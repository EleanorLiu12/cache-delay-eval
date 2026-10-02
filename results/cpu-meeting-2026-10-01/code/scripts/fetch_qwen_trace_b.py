"""Acquire original Trace B; parse a fresh pinned publisher LFS pointer first."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from fetch_qwen_trace_a import fetch, REPOSITORY, REVISION


TRACE = "qwen_traceB_blksz_16.jsonl"
PREVIOUSLY_INSPECTED_SHA256 = "68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59"
PREVIOUSLY_INSPECTED_BYTES = 96209982


def parse_lfs_pointer(pointer):
    """Reject malformed pointers; publisher values, not prior notes, are authoritative."""
    match = re.fullmatch(
        r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?",
        pointer,
    )
    if match is None:
        raise ValueError("Unexpected Git LFS pointer format")
    size = int(match.group(2))
    if size <= 0:
        raise ValueError("LFS object size must be positive")
    return dict(sha256=match.group(1), bytes=size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/qwen-bailian-trace-b") / REVISION)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}"
    entries = []
    for source, filename in [(TRACE, "trace-lfs-pointer.txt"), ("README.md", "upstream-README.md"),
                             ("docs/qa-context-growth-pattern.md", "upstream-context-faq.md"),
                             ("LICENSE", "LICENSE")]:
        entries.append(fetch(f"{raw}/{source}", args.output_dir / filename))
    pointer = parse_lfs_pointer((args.output_dir / "trace-lfs-pointer.txt").read_text())
    url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/{TRACE}"
    trace = fetch(url, args.output_dir / TRACE)
    entries.append(trace)
    verified = trace["sha256"] == pointer["sha256"] and trace["bytes"] == pointer["bytes"]
    manifest = dict(
        dataset="Qwen-Bailian Trace B", repository=f"https://github.com/{REPOSITORY}",
        revision=REVISION, license="Apache-2.0", files=entries,
        acquired_at_utc=datetime.now(timezone.utc).isoformat(),
        expected_trace_sha256=pointer["sha256"], expected_trace_bytes=pointer["bytes"],
        verification_authority="freshly acquired LFS pointer at the pinned publisher revision",
        matches_previously_inspected_metadata=(pointer["sha256"] == PREVIOUSLY_INSPECTED_SHA256
                                              and pointer["bytes"] == PREVIOUSLY_INSPECTED_BYTES),
        trace_verified_against_lfs_pointer=verified,
        transformation="none; original source bytes retained",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not verified:
        raise ValueError("Downloaded bytes differ from fresh publisher LFS pointer; retained for inspection")
    print(json.dumps(dict(directory=str(args.output_dir), verified=verified, **pointer)))


if __name__ == "__main__":
    main()
