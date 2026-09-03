from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from accounting.professional.annual_dashboard_tables_legacy import (
    build_annual_debt_activity_by_pair,
    build_annual_debt_stock_by_pair,
    write_annual_long_and_wide,
)


def _monthly_matrix(source: pd.DataFrame, specs: list[tuple[str, str, str]]) -> pd.DataFrame:
    parts = []
    for selector_column, selector, measure in specs:
        rows = source.loc[source[selector_column].astype(str).eq(selector)].copy()
        if rows.empty:
            continue
        rows["measure"] = measure
        rows["pair"] = rows["debtor"].astype(str) + " → " + rows["creditor"].astype(str)
        rows["value"] = pd.to_numeric(rows[measure], errors="coerce").fillna(0.0)
        parts.append(rows[["measure", "Currency", "pair", "period", "value"]])
    if not parts:
        return pd.DataFrame(columns=["measure", "Currency", "pair"])
    long = pd.concat(parts, ignore_index=True)
    return (
        long.pivot_table(
            index=["measure", "Currency", "pair"],
            columns="period",
            values="value",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )


def build_debt_tables(run_root: Path, tables_dir: Path) -> list[Path]:
    run_root = Path(run_root)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    activity = pd.read_csv(run_root / "monthly_debt_activity.csv")
    position = pd.read_csv(run_root / "monthly_debt_position.csv")
    activity_matrix = _monthly_matrix(activity, [
        ("activity_type", "new_claim", "new_principal"),
        ("activity_type", "interest_accrual", "interest_accrued"),
        ("activity_type", "repayment", "repayments"),
        ("activity_type", "adjustment", "adjustments"),
        ("activity_type", "net_change", "net_change"),
    ])
    position_matrix = _monthly_matrix(position, [
        ("component", "principal", "open_principal"),
        ("component", "interest", "open_interest"),
        ("component", "total", "open_total"),
    ])
    paths = [
        tables_dir / "monthly_tables_debt_activity_matrix.csv",
        tables_dir / "monthly_tables_debt_position_matrix.csv",
    ]
    activity_matrix.to_csv(paths[0], index=False)
    position_matrix.to_csv(paths[1], index=False)
    activity_long, activity_wide = build_annual_debt_activity_by_pair(activity)
    stock_long, stock_wide = build_annual_debt_stock_by_pair(position)
    written = [
        *write_annual_long_and_wide(activity_long, activity_wide, tables_dir, "annual_debt_activity_by_pair").values(),
        *write_annual_long_and_wide(stock_long, stock_wide, tables_dir, "annual_debt_stock_by_pair").values(),
    ]
    return [*paths, *(path for path in written if path is not None)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build debt-only professional tables from one governed run.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--tables-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in build_debt_tables(args.run_root, args.tables_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
