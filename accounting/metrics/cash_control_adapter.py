"""Small adaptor from governed treasury accountability to control metrics.

This module deliberately contains no accounting arithmetic.  It validates the
producer's shape and applies the executable cash-control contract to select
the existing movement and closing-control columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from accounting.contracts.cash_control import CASH_CONTROL_SPECS_V1

SOURCE_TABLE = "monthly_cash_accountability.csv"
KEY_COLUMNS = ["period", "period_end", "Box", "Currency"]
REQUIRED_COLUMNS = [*KEY_COLUMNS, "net_cash_flow", "closing_control"]


def read_source(run_root: Path) -> pd.DataFrame | None:
    """Read and validate the optional exact-run source.

    Older synthetic fixtures may not contain this producer output.  A present
    source is different: malformed keys, duplicate grains, or non-numeric
    governed values are errors and must not become silently missing metrics.
    """

    path = Path(run_root) / SOURCE_TABLE
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{SOURCE_TABLE} missing required columns: {missing}")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{SOURCE_TABLE} has null control key columns")
    if frame.duplicated(KEY_COLUMNS, keep=False).any():
        duplicate_keys = frame.loc[frame.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(
            f"{SOURCE_TABLE} has duplicate period/period_end/Box/Currency keys: "
            f"{duplicate_keys.to_dict('records')}"
        )
    period_ends = pd.to_datetime(frame["period_end"], errors="coerce")
    if period_ends.isna().any():
        bad_rows = frame.index[period_ends.isna()].tolist()
        raise ValueError(f"{SOURCE_TABLE} has invalid period_end rows: {bad_rows}")
    for column in ("net_cash_flow", "closing_control"):
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | values.isin([float("inf"), float("-inf")])
        if invalid.any():
            bad_rows = frame.index[invalid].tolist()
            raise ValueError(f"{SOURCE_TABLE} has invalid numeric {column} rows: {bad_rows}")
        frame[column] = values
    return frame


def monthly_rows(
    source: pd.DataFrame,
    *,
    run_id: str,
    as_of_date: str,
) -> list[dict[str, Any]]:
    """Return one governed row per metric and source grain."""

    rows: list[dict[str, Any]] = []
    caveat = (
        "Recorded Box movement/control only: cumulative control uses the defined "
        "zero origin and is not actual cash or liquidity; unknown opening balances "
        "remain unknown; native currency and per-Box transfer legs are preserved."
    )
    for _, row in source.iterrows():
        for metric_id, spec in CASH_CONTROL_SPECS_V1.items():
            rows.append(
                {
                    "metric_id": metric_id,
                    "period": row["period"],
                    "period_end": row["period_end"],
                    "Currency": row["Currency"],
                    "Box": row["Box"],
                    "value": row[spec.value_ref],
                    "source_table": SOURCE_TABLE,
                    "run_id": run_id,
                    "as_of_date": as_of_date,
                    "caveat": caveat,
                }
            )
    return rows


def annual_rows(
    source: pd.DataFrame,
    *,
    run_id: str,
    as_of_date: str,
) -> list[dict[str, Any]]:
    """Sum flows and select the latest month-end control by year/Box/currency."""

    work = source.copy()
    work["year"] = work["period"].astype(str).str.slice(0, 4)
    work["_period_end_sort"] = pd.to_datetime(work["period_end"], errors="coerce")
    if work["_period_end_sort"].isna().any():
        raise ValueError(f"{SOURCE_TABLE} has invalid period_end values")
    caveat = (
        "Recorded Box movement/control only: cumulative control is the latest "
        "zero-origin month-end value, not actual cash or liquidity; unknown opening "
        "balances remain unknown; no FX or cross-Box aggregation."
    )
    rows: list[dict[str, Any]] = []
    groups = ["year", "Box", "Currency"]
    for (year, box, currency), group in work.groupby(groups, dropna=False, sort=False):
        latest = group.sort_values(["_period_end_sort", "period_end"]).iloc[-1]
        rows.extend(
            [
                {
                    "metric_id": "CTRL.BOX.NET_MOVEMENT",
                    "period": year,
                    "period_end": latest["period_end"],
                    "Currency": currency,
                    "Box": box,
                    "value": group["net_cash_flow"].sum(),
                    "source_table": SOURCE_TABLE,
                    "run_id": run_id,
                    "as_of_date": as_of_date,
                    "caveat": caveat,
                },
                {
                    "metric_id": "CTRL.BOX.CUM_NET",
                    "period": year,
                    "period_end": latest["period_end"],
                    "Currency": currency,
                    "Box": box,
                    "value": latest["closing_control"],
                    "source_table": SOURCE_TABLE,
                    "run_id": run_id,
                    "as_of_date": as_of_date,
                    "caveat": caveat,
                },
            ]
        )
    return rows
