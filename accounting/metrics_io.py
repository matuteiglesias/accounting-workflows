from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


METRIC_VALUES_COLUMNS = [
    "metric_id",
    "period_grain",
    "period",
    "currency",
    "value",
    "run_id",
    "as_of_date",
    "source_layer",
    "build_status",
    "build_detail",
]


@dataclass
class MetricsContext:
    ledger: Optional[pd.DataFrame] = None
    per_flow: Optional[pd.DataFrame] = None
    per_party: Optional[pd.DataFrame] = None
    daily_cash_position: Optional[pd.DataFrame] = None
    v_contributions_monthly: Optional[pd.DataFrame] = None
    v_opex_category_monthly: Optional[pd.DataFrame] = None
    party_balance_detailed: Optional[pd.DataFrame] = None
    debt_balance_monthly: Optional[pd.DataFrame] = None
    debt_balance_quarterly: Optional[pd.DataFrame] = None
    debt_balance_yearly: Optional[pd.DataFrame] = None
    deposits: Optional[pd.DataFrame] = None
    debts: Optional[pd.DataFrame] = None
    run_id: str = ""
    as_of_date: str = ""


def ensure_metric_values_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in METRIC_VALUES_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out["metric_id"] = out["metric_id"].astype(str).str.strip()
    out["period_grain"] = out["period_grain"].astype(str).str.strip()
    out["period"] = out["period"].astype(str).str.strip()
    out["currency"] = out["currency"].astype(str).str.strip()
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)
    out["run_id"] = out["run_id"].astype(str)
    out["as_of_date"] = out["as_of_date"].astype(str)
    out["source_layer"] = out["source_layer"].astype(str)
    out["build_status"] = out["build_status"].astype(str).replace("", "ok")
    out["build_detail"] = out["build_detail"].astype(str)

    return out[METRIC_VALUES_COLUMNS]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return

    raise ValueError(f"Unsupported file format: {path}")


def append_period_columns(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    copy: bool = True,
) -> pd.DataFrame:
    out = df.copy() if copy else df
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = out[date_col].dt.year.astype("Int64")
    out["quarter"] = out[date_col].dt.quarter.astype("Int64")
    out["period_q"] = out["year"].astype(str) + "Q" + out["quarter"].astype(str)
    out["period_y"] = out["year"].astype(str)
    return out


def append_period_columns_from_timeperiod(
    df: pd.DataFrame,
    *,
    col: str = "TimePeriod",
    freq: str = "M",
    copy: bool = True,
) -> pd.DataFrame:
    out = df.copy() if copy else df
    p = pd.PeriodIndex(out[col].astype(str), freq=freq)
    out[col] = p.astype(str)
    out["year"] = p.year
    out["quarter"] = p.quarter
    out["period_q"] = p.year.astype(str) + "Q" + pd.Series(p.quarter, index=out.index).astype(str)
    out["period_y"] = p.year.astype(str)
    return out


def build_metric_frame(
    *,
    metric_id: str,
    period_grain: str,
    period: str,
    currency: str,
    value: float,
    run_id: str = "",
    as_of_date: str = "",
    source_layer: str = "",
    build_status: str = "ok",
    build_detail: str = "",
) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "metric_id": metric_id,
                "period_grain": period_grain,
                "period": period,
                "currency": currency,
                "value": value,
                "run_id": run_id,
                "as_of_date": as_of_date,
                "source_layer": source_layer,
                "build_status": build_status,
                "build_detail": build_detail,
            }
        ]
    )
    return ensure_metric_values_schema(df)


def concat_metric_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return ensure_metric_values_schema(pd.DataFrame(columns=METRIC_VALUES_COLUMNS))
    return ensure_metric_values_schema(pd.concat(frames, ignore_index=True))
