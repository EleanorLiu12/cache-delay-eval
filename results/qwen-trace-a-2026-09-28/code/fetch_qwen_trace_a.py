"""Fetch the pinned, unmodified Qwen-Bailian Trace A and its provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


REVISION = "5f7439c51ec248a0c585f7d90a41a6f57773b912"
REPOSITORY = "alibaba-edu/qwen-bailian-usagetraces-anon"
TRACE = "qwen_traceA_blksz_16.jsonl"
EXPECTED_SHA256 = "07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17"
EXPECTED_BYTES = 56354493


def fetch(url, destination):
    """Never replace an existing acquisition; retain original bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": "cache-delay-eval-research"})
    digest = hashlib.sha256()
    size = 0
    partial = destination.with_name(destination.name + ".partial")
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
        headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Type")}
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if destination.exists():
        raise FileExistsError(destination)
    partial.rename(destination)
    return dict(url=url, file=destination.name, bytes=size, sha256=digest.hexdigest(),
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(), headers=headers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/qwen-bailian") / REVISION)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}"
    entries = []
    for source, filename in [("README.md", "upstream-README.md"),
                             ("docs/qa-context-growth-pattern.md", "upstream-context-faq.md"),
                             ("LICENSE", "LICENSE"), (TRACE, "trace-lfs-pointer.txt")]:
        entries.append(fetch(f"{raw}/{source}", args.output_dir / filename))
    pointer = (args.output_dir / "trace-lfs-pointer.txt").read_text()
    if f"oid sha256:{EXPECTED_SHA256}" not in pointer or f"size {EXPECTED_BYTES}" not in pointer:
        raise ValueError("Pinned LFS pointer differs from the investigated object")
    url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/{TRACE}"
    trace = fetch(url, args.output_dir / TRACE)
    entries.append(trace)
    verified = trace["sha256"] == EXPECTED_SHA256 and trace["bytes"] == EXPECTED_BYTES
    manifest = dict(dataset="Qwen-Bailian Trace A", repository=f"https://github.com/{REPOSITORY}",
                    revision=REVISION, license="Apache-2.0", files=entries,
                    expected_trace_sha256=EXPECTED_SHA256, expected_trace_bytes=EXPECTED_BYTES,
                    trace_verified_against_lfs_pointer=verified,
                    transformation="none; original source bytes retained")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not verified:
        raise ValueError("Downloaded trace failed SHA-256/size verification; retained for inspection")
    print(json.dumps(dict(directory=str(args.output_dir), verified=verified,
                          bytes=trace["bytes"], sha256=trace["sha256"])))


if __name__ == "__main__":
    main()
