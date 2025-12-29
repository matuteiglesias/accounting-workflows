#!/usr/bin/env python3
"""
validate_manifest.py
Validate manifest.json entries:
 - file exists
 - sha256 matches
 - row count matches
 - for certain known files, validate header contains required columns
Exit code: 0 ok, 1 warnings, 2 errors
"""
import argparse
import json
import hashlib
from pathlib import Path
import sys

REQUIRED_HEADERS = {
    # filename (or suffix) : list of required header tokens
    "all_txns_parsed.csv": ["date", "source", "account", "payee", "amount", "Currency"],
    "fondos_report.csv": ["Net", "Date"],  # soft check: tokens present
    "renta_PM.csv": ["Date"],  # ensure at least Date column exists
}

def sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def read_header(p: Path):
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline().strip()
            return [c.strip() for c in first.split(",")]
    except Exception:
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to manifest.json")
    args = ap.parse_args()
    mpath = Path(args.manifest)
    if not mpath.exists():
        print("ERROR: manifest not found:", mpath)
        return 2
    data = json.loads(mpath.read_text(encoding="utf-8"))
    errors = 0
    warnings = 0
    for name, meta in data.items():
        if "path" not in meta:
            print(f"[WARN] {name} missing path in manifest")
            warnings += 1
            continue
        p = Path(meta["path"])
        if not p.exists():
            print(f"[ERROR] missing file: {p}")
            errors += 1
            continue
        # checksum
        if "sha256" in meta:
            actual = sha256_of_path(p)
            if actual != meta["sha256"]:
                print(f"[ERROR] checksum mismatch for {name}")
                errors += 1
        # rows check if available
        if "rows" in meta:
            # count lines - 1
            actual_rows = sum(1 for _ in p.open("rb")) - 1
            if actual_rows != meta["rows"]:
                print(f"[WARN] row count mismatch for {name}: manifest={meta['rows']} actual={actual_rows}")
                warnings += 1
        # header checks for known files
        for key, reqs in REQUIRED_HEADERS.items():
            if name.endswith(key):
                header = read_header(p)
                header_lower = [h.lower() for h in header]
                missing = [r for r in reqs if not any(r.lower() in h for h in header_lower)]
                if missing:
                    print(f"[ERROR] {name} header missing tokens: {missing} (header: {header[:10]})")
                    errors += 1
    print("Done. errors:", errors, "warnings:", warnings)
    if errors:
        return 2
    if warnings:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
