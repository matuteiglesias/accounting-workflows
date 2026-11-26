"""Private HH documentation report using the same hash-verified engine as PM."""
from __future__ import annotations

import argparse
from pathlib import Path

from accounting.reports.payment_evidence import build_payment_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-snapshot", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--browser-bin")
    args = parser.parse_args()
    result = build_payment_evidence_report(
        run_root=args.run_root,
        snapshot_dir=args.evidence_snapshot,
        out_root=args.out_root,
        browser_bin=args.browser_bin,
        report_id="hh_payment_evidence",
        report_title="Household · Boletas y comprobantes",
        report_description="Gastos de Household, facturas y pruebas de pago documentadas; revisión privada, sin datos de FB/PM.",
        scope_tag="HH_EVIDENCE",
    )
    for name, value in result.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
