from __future__ import annotations

import argparse
from pathlib import Path

from accounting.reports.private_bundle import compose_private_report_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose one private catalog-driven report bundle.")
    parser.add_argument("--normal-reports", required=True, type=Path)
    parser.add_argument("--evidence-reports", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(compose_private_report_bundle(
        normal_reports_root=args.normal_reports,
        evidence_reports_root=args.evidence_reports,
        out_root=args.out,
    ))


if __name__ == "__main__":
    main()
