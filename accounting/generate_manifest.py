#!/usr/bin/env python3
"""
generate_manifest.py
Write manifest.json with sha256, rows (excluding header), path and timestamp for all CSV files
under the provided outdir.
"""
import argparse
import json
import hashlib
import time
from pathlib import Path
import glob

def sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def count_rows(p: Path) -> int:
    # returns number of data rows (excluding header) if file looks CSV-like
    with p.open("rb") as fh:
        # simple and memory-light
        n = 0
        for _ in fh:
            n += 1
    return max(0, n - 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./out", help="root directory to scan for CSVs")
    args = p.parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    records = {}
    for f in sorted(glob.glob(str(outdir) + "/**/*.csv", recursive=True)):
        fp = Path(f)
        try:
            sha = sha256_of_path(fp)
            rows = count_rows(fp)
            records[fp.name] = {
                "sha256": sha,
                "rows": rows,
                "path": str(fp),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        except Exception as e:
            records[fp.name] = {"error": str(e), "path": str(fp)}
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print("Wrote manifest:", manifest_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
