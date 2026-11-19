from __future__ import annotations

"""Cheap preflight for the on-demand PRIVATE evidence report trigger.

Does not build or mutate accounting sources. The renderer will revalidate
document hashes before accepting any evidence into a bundle.
"""

import argparse
import hashlib
import json
from pathlib import Path


def validate_source(run_root: Path, snapshot: Path) -> None:
    ledger = (Path(run_root) / "ledger_canonical.csv").resolve(strict=True)
    manifest = json.loads((Path(snapshot) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact") != "acct.transaction-evidence@1":
        raise ValueError("unsupported private evidence snapshot schema")
    if manifest.get("source_run_id") != ledger.parent.name:
        raise ValueError("private evidence source run does not match target accounting run")
    if manifest.get("publication_allowed") is not False:
        raise ValueError("private evidence publication_allowed must be explicitly false")
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    if manifest.get("source_ledger_sha256") != digest:
        raise ValueError("private evidence source ledger SHA256 does not match target accounting run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    args = parser.parse_args()
    validate_source(args.run_root, args.snapshot)
    print(f"[EVIDENCE] exact-run private source validated: {args.run_root.name}")


if __name__ == "__main__":
    main()
