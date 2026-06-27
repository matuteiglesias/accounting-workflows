# accounting/human/reports.py
from __future__ import annotations

"""
Deprecated module.

Stage E reports have been merged into Stage V views:
  - renta outputs are now derived directly from Stage D per_party_time_long
  - fondos wide output is now derived directly from Stage D per_party_time_long

Keep this module as a thin CLI wrapper to avoid breaking old entrypoints.
"""

from pathlib import Path
import argparse
import logging

from accounting.marts import export_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=str, required=True, help="Path to the reports folder or run folder used by views.load_reports_folder")
    ap.add_argument("--write-dir", type=str, required=True, help="Directory to write outputs (views marts and convenience tables)")
    ap.add_argument("--freq", type=str, default="M", help="Aggregation frequency (M, W, Q)")
    ap.add_argument("--allow-cross-currency-sum", action="store_true", help="Write unsafe currency-summed convenience outputs")
    args = ap.parse_args()

    logging.warning("accounting.human.reports is deprecated. It now calls views.export_views and does not generate separate reports-stage artifacts.")
    export_views(
        reports_dir=Path(args.reports_dir),
        write_dir=Path(args.write_dir),
        freq=str(args.freq),
        allow_cross_currency_sum=bool(args.allow_cross_currency_sum),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())