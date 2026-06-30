from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

CASH_CLOSE_COLUMNS = [
    "period",
    "period_end",
    "as_of_date",
    "Box",
    "party",
    "account_id",
    "account_name",
    "Currency",
    "close_amount",
    "source_table",
    "source_date",
    "source_type",
    "source_reference",
    "validation_status",
    "validated_by",
    "position_type",
    "cash_suitability",
    "is_frontend_safe",
    "caveat",
    "notes",
    "n_source_rows",
    "calculation_rule",
]
VALIDATED_CASH_SOURCE_TYPES = {
    "bank_statement",
    "manual_cash_count",
    "account_snapshot",
    "reconciled_opening_plus_movements",
}
EXPLICIT_VALIDATION_STATUSES = {"validated", "approved", "reconciled"}
CASH_QA_COLUMNS = ["check", "status", "detail", "severity"]


def _empty_cash_close() -> pd.DataFrame:
    return pd.DataFrame(columns=CASH_CLOSE_COLUMNS)


def _qa(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CASH_QA_COLUMNS)


def _blank_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series([""] * len(df), index=df.index)


def _explicitly_validated(validated: pd.DataFrame) -> pd.Series:
    status = validated["validation_status"].astype(str).str.strip().str.lower()
    validated_by = validated["validated_by"].astype(str).str.strip()
    source_type = validated["source_type"].astype(str).str.strip().str.lower()
    return (
        status.isin(EXPLICIT_VALIDATION_STATUSES)
        & validated_by.ne("")
        & source_type.isin(VALIDATED_CASH_SOURCE_TYPES)
    )


def build_monthly_cash_close(out_dir: Path, freq: str = "M") -> Dict[str, Path]:
    out_dir = Path(out_dir)
    daily_path = out_dir / "daily_cash_position.csv"
    box_path = out_dir / f"box_balance_time_long.freq={freq}.csv"
    validated_path = out_dir / "validated_cash_close.csv"
    close_path = out_dir / "monthly_cash_close.csv"
    qa_path = out_dir / "monthly_cash_close_qa.csv"

    frames: list[pd.DataFrame] = []
    daily = pd.DataFrame()
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        required = ["Date", "Box", "party", "Currency", "balance"]
        missing = [c for c in required if c not in daily.columns]
        if missing:
            raise ValueError(
                f"daily_cash_position.csv missing required columns for monthly_cash_close: {missing}"
            )

        work = daily.copy()
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work[work["Date"].notna()].copy()
        work["balance"] = pd.to_numeric(work["balance"], errors="coerce").fillna(0.0)
        period = work["Date"].dt.to_period(freq)
        work["period"] = period.astype(str)
        work["period_end"] = period.dt.end_time.dt.date.astype(str)

        idx = work.groupby(["period", "Box", "party", "Currency"], dropna=False)[
            "Date"
        ].idxmax()
        close = work.loc[idx].copy().sort_values(["period", "Box", "party", "Currency"])
        close_rows = pd.DataFrame(
            {
                "period": close["period"],
                "period_end": close["period_end"],
                "as_of_date": close["Date"].dt.date.astype(str),
                "Box": close["Box"].fillna(""),
                "party": close["party"].fillna(""),
                "account_id": _blank_series(close),
                "account_name": _blank_series(close),
                "Currency": close["Currency"].fillna(""),
                "close_amount": close["balance"].astype(float),
                "source_table": "daily_cash_position.csv",
                "source_date": close["Date"].dt.date.astype(str),
                "source_type": "internal_party_balance",
                "source_reference": _blank_series(close),
                "validation_status": "not_validated_for_frontend_cash",
                "validated_by": _blank_series(close),
                "position_type": "internal_balance",
                "cash_suitability": "internal_only",
                "is_frontend_safe": False,
                "caveat": "Party-level daily cash position is an internal balance/claim view; do not sum as frontend-safe cash without account-level cash validation.",
                "notes": _blank_series(close),
                "n_source_rows": 1,
                "calculation_rule": "last observed daily_cash_position.balance by month, Box, party, Currency",
            }
        )
        frames.append(close_rows)

    box_loaded = False
    if box_path.exists():
        box = pd.read_csv(box_path)
        box_required = ["TimePeriod", "TimePeriod_end", "Box", "Currency", "cum_net"]
        if all(c in box.columns for c in box_required):
            box_loaded = True
            box_work = box.copy()
            box_rows = pd.DataFrame(
                {
                    "period": box_work["TimePeriod"].astype(str),
                    "period_end": box_work["TimePeriod_end"].astype(str),
                    "as_of_date": box_work["TimePeriod_end"].astype(str),
                    "Box": box_work["Box"].fillna(""),
                    "party": "",
                    "account_id": "",
                    "account_name": "",
                    "Currency": box_work["Currency"].fillna(""),
                    "close_amount": pd.to_numeric(
                        box_work["cum_net"], errors="coerce"
                    ).fillna(0.0),
                    "source_table": box_path.name,
                    "source_date": box_work["TimePeriod_end"].astype(str),
                    "source_type": "inferred_box_motor",
                    "source_reference": "",
                    "validation_status": "not_validated_for_frontend_cash",
                    "validated_by": "",
                    "position_type": "inferred_box_motor",
                    "cash_suitability": "safe_with_caveat",
                    "is_frontend_safe": False,
                    "caveat": "Box motor cumulative net is reconciliation/inferred movement, not real cash close.",
                    "notes": "",
                    "n_source_rows": 1,
                    "calculation_rule": "box_balance_time_long.cum_net by month, Box, Currency",
                }
            )
            frames.append(box_rows)

    validated_loaded = False
    validated_safe_rows = 0
    if validated_path.exists():
        validated = pd.read_csv(validated_path)
        required = [
            "period",
            "period_end",
            "as_of_date",
            "Box",
            "account_id",
            "account_name",
            "Currency",
            "close_amount",
            "source_type",
            "source_reference",
            "validation_status",
            "validated_by",
            "notes",
        ]
        missing = [c for c in required if c not in validated.columns]
        if missing:
            raise ValueError(
                f"validated_cash_close.csv missing required columns for monthly_cash_close: {missing}"
            )
        valid = validated.loc[_explicitly_validated(validated)].copy()
        validated_loaded = True
        if not valid.empty:
            valid["close_amount"] = pd.to_numeric(
                valid["close_amount"], errors="coerce"
            )
            valid = valid[valid["close_amount"].notna()].copy()
        if not valid.empty:
            valid_rows = pd.DataFrame(
                {
                    "period": valid["period"].astype(str),
                    "period_end": valid["period_end"].astype(str),
                    "as_of_date": valid["as_of_date"].astype(str),
                    "Box": valid["Box"].fillna(""),
                    "party": "",
                    "account_id": valid["account_id"].fillna(""),
                    "account_name": valid["account_name"].fillna(""),
                    "Currency": valid["Currency"].fillna("").astype(str).str.upper(),
                    "close_amount": valid["close_amount"].astype(float),
                    "source_table": validated_path.name,
                    "source_date": valid["as_of_date"].astype(str),
                    "source_type": valid["source_type"].astype(str),
                    "source_reference": valid["source_reference"].fillna(""),
                    "validation_status": valid["validation_status"].astype(str),
                    "validated_by": valid["validated_by"].astype(str),
                    "position_type": "cash_close",
                    "cash_suitability": "frontend_safe",
                    "is_frontend_safe": True,
                    "caveat": "Explicitly validated cash close; safe for frontend cash within its stated currency only.",
                    "notes": valid["notes"].fillna(""),
                    "n_source_rows": 1,
                    "calculation_rule": "validated cash close row; no inference from party balances or box motor",
                }
            )
            validated_safe_rows = len(valid_rows)
            frames.append(valid_rows)

    out = pd.concat(frames, ignore_index=True) if frames else _empty_cash_close()
    out = out[CASH_CLOSE_COLUMNS]
    out.to_csv(close_path, index=False)

    frontend_safe = (
        int(out["is_frontend_safe"].astype(bool).sum()) if not out.empty else 0
    )
    unsafe = int((~out["is_frontend_safe"].astype(bool)).sum()) if not out.empty else 0
    internal = (
        int(out["position_type"].eq("internal_balance").sum()) if not out.empty else 0
    )
    box_motor = (
        int(out["position_type"].eq("inferred_box_motor").sum()) if not out.empty else 0
    )
    caveats_present = (
        bool((out["caveat"].astype(str).str.strip() != "").all())
        if not out.empty
        else False
    )

    qa_rows = [
        {
            "check": "monthly_cash_close_exists",
            "status": "pass" if close_path.exists() else "fail",
            "detail": str(close_path),
            "severity": "error",
        },
        {
            "check": "validated_cash_source_loaded_or_absent",
            "status": "pass",
            "detail": f"loaded={validated_loaded}; frontend_safe_rows={validated_safe_rows}",
            "severity": "warning",
        },
        {
            "check": "daily_cash_position_loaded",
            "status": "pass" if daily_path.exists() else "warn",
            "detail": (
                f"rows={len(daily)}" if daily_path.exists() else f"absent {daily_path}"
            ),
            "severity": "warning",
        },
        {
            "check": "frontend_safe_cash_rows_count",
            "status": "pass",
            "detail": str(frontend_safe),
            "severity": "warning",
        },
        {
            "check": "unsafe_cash_rows_count",
            "status": "pass",
            "detail": str(unsafe),
            "severity": "warning",
        },
        {
            "check": "internal_balance_rows_count",
            "status": "pass",
            "detail": str(internal),
            "severity": "warning",
        },
        {
            "check": "box_motor_rows_count",
            "status": "pass" if box_loaded else "warn",
            "detail": str(box_motor),
            "severity": "warning",
        },
        {
            "check": "no_cash_total_without_frontend_safe_rows",
            "status": "pass",
            "detail": "no aggregate frontend-safe cash total emitted",
            "severity": "error",
        },
        {
            "check": "cash_rows_have_currency",
            "status": (
                "pass"
                if out.empty or out["Currency"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": ",".join(sorted(out["Currency"].dropna().astype(str).unique())),
            "severity": "error",
        },
        {
            "check": "cash_rows_have_position_type",
            "status": (
                "pass"
                if out.empty
                or out["position_type"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": ",".join(
                sorted(out["position_type"].dropna().astype(str).unique())
            ),
            "severity": "error",
        },
        {
            "check": "cash_rows_have_suitability",
            "status": (
                "pass"
                if out.empty
                or out["cash_suitability"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": ",".join(
                sorted(out["cash_suitability"].dropna().astype(str).unique())
            ),
            "severity": "error",
        },
        {
            "check": "cash_close_caveats_present",
            "status": "pass" if out.empty or caveats_present else "fail",
            "detail": (
                "all rows have caveats"
                if out.empty or caveats_present
                else "missing caveat"
            ),
            "severity": "error",
        },
        {
            "check": "daily_cash_position_rows_not_frontend_safe",
            "status": (
                "pass"
                if not out["source_table"]
                .astype(str)
                .eq("daily_cash_position.csv")
                .any()
                or out.loc[
                    out["source_table"].astype(str).eq("daily_cash_position.csv"),
                    "is_frontend_safe",
                ]
                .astype(bool)
                .eq(False)
                .all()
                else "fail"
            ),
            "detail": "daily cash rows are internal-only",
            "severity": "error",
        },
        {
            "check": "box_motor_rows_not_frontend_safe",
            "status": (
                "pass"
                if not out["position_type"].astype(str).eq("inferred_box_motor").any()
                or out.loc[
                    out["position_type"].astype(str).eq("inferred_box_motor"),
                    "is_frontend_safe",
                ]
                .astype(bool)
                .eq(False)
                .all()
                else "fail"
            ),
            "detail": "box motor rows are reconciliation-only",
            "severity": "error",
        },
        {
            "check": "no_cross_currency_cash_total",
            "status": "pass",
            "detail": "monthly_cash_close remains row-level/by-currency and emits no cross-currency aggregate",
            "severity": "error",
        },
    ]
    _qa(qa_rows).to_csv(qa_path, index=False)
    return {"monthly_cash_close": close_path, "monthly_cash_close_qa": qa_path}
