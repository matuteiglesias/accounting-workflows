from __future__ import annotations

import argparse
from pathlib import Path

from accounting.reports.payment_evidence import build_payment_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the private Property Management payment-evidence report.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evidence-snapshot", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--browser-bin")
    args = parser.parse_args()
    result = build_payment_evidence_report(
        run_root=args.run_root,
        snapshot_dir=args.evidence_snapshot,
        out_root=args.out_root,
        browser_bin=args.browser_bin,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
