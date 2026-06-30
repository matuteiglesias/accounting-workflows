from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

CASH_CLOSE_COLUMNS = [
    "period", "period_end", "as_of_date", "Box", "party", "Currency", "close_amount",
    "source_table", "source_date", "position_type", "cash_suitability", "is_frontend_safe",
    "caveat", "n_source_rows", "calculation_rule",
]
CASH_QA_COLUMNS = ["check", "status", "detail", "severity"]


def _empty_cash_close() -> pd.DataFrame:
    return pd.DataFrame(columns=CASH_CLOSE_COLUMNS)


def _qa(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CASH_QA_COLUMNS)


def build_monthly_cash_close(out_dir: Path, freq: str = "M") -> Dict[str, Path]:
    out_dir = Path(out_dir)
    daily_path = out_dir / "daily_cash_position.csv"
    box_path = out_dir / f"box_balance_time_long.freq={freq}.csv"
    close_path = out_dir / "monthly_cash_close.csv"
    qa_path = out_dir / "monthly_cash_close_qa.csv"

    if not daily_path.exists():
        qa = _qa([
            {"check": "monthly_cash_close_exists", "status": "fail", "detail": "monthly_cash_close.csv not built", "severity": "error"},
            {"check": "daily_cash_position_loaded", "status": "fail", "detail": f"missing {daily_path}", "severity": "error"},
        ])
        _empty_cash_close().to_csv(close_path, index=False)
        qa.to_csv(qa_path, index=False)
        raise FileNotFoundError(f"monthly_cash_close requires {daily_path}; no legacy cash fallback was used")

    daily = pd.read_csv(daily_path)
    required = ["Date", "Box", "party", "Currency", "balance"]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise ValueError(f"daily_cash_position.csv missing required columns for monthly_cash_close: {missing}")

    work = daily.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work[work["Date"].notna()].copy()
    work["balance"] = pd.to_numeric(work["balance"], errors="coerce").fillna(0.0)
    period = work["Date"].dt.to_period(freq)
    work["period"] = period.astype(str)
    work["period_end"] = period.dt.end_time.dt.date.astype(str)

    idx = work.groupby(["period", "Box", "party", "Currency"], dropna=False)["Date"].idxmax()
    close = work.loc[idx].copy().sort_values(["period", "Box", "party", "Currency"])
    close_rows = pd.DataFrame({
        "period": close["period"],
        "period_end": close["period_end"],
        "as_of_date": close["Date"].dt.date.astype(str),
        "Box": close["Box"].fillna(""),
        "party": close["party"].fillna(""),
        "Currency": close["Currency"].fillna(""),
        "close_amount": close["balance"].astype(float),
        "source_table": "daily_cash_position.csv",
        "source_date": close["Date"].dt.date.astype(str),
        "position_type": "internal_balance",
        "cash_suitability": "internal_only",
        "is_frontend_safe": False,
        "caveat": "Party-level daily cash position is an internal balance/claim view; do not sum as frontend-safe cash without account-level cash validation.",
        "n_source_rows": 1,
        "calculation_rule": "last observed daily_cash_position.balance by month, Box, party, Currency",
    })

    frames = [close_rows]
    box_loaded = False
    if box_path.exists():
        box = pd.read_csv(box_path)
        box_required = ["TimePeriod", "TimePeriod_end", "Box", "Currency", "cum_net"]
        if all(c in box.columns for c in box_required):
            box_loaded = True
            box_work = box.copy()
            box_rows = pd.DataFrame({
                "period": box_work["TimePeriod"].astype(str),
                "period_end": box_work["TimePeriod_end"].astype(str),
                "as_of_date": box_work["TimePeriod_end"].astype(str),
                "Box": box_work["Box"].fillna(""),
                "party": "",
                "Currency": box_work["Currency"].fillna(""),
                "close_amount": pd.to_numeric(box_work["cum_net"], errors="coerce").fillna(0.0),
                "source_table": box_path.name,
                "source_date": box_work["TimePeriod_end"].astype(str),
                "position_type": "inferred_box_motor",
                "cash_suitability": "safe_with_caveat",
                "is_frontend_safe": False,
                "caveat": "Box motor cumulative net is reconciliation/inferred movement, not real cash close.",
                "n_source_rows": 1,
                "calculation_rule": "box_balance_time_long.cum_net by month, Box, Currency",
            })
            frames.append(box_rows)

    out = pd.concat(frames, ignore_index=True) if frames else _empty_cash_close()
    out = out[CASH_CLOSE_COLUMNS]
    out.to_csv(close_path, index=False)

    frontend_safe = int(out["is_frontend_safe"].astype(bool).sum()) if not out.empty else 0
    unsafe = int((~out["is_frontend_safe"].astype(bool)).sum()) if not out.empty else 0
    internal = int(out["position_type"].eq("internal_balance").sum()) if not out.empty else 0
    box_motor = int(out["position_type"].eq("inferred_box_motor").sum()) if not out.empty else 0
    caveats_present = bool((out["caveat"].astype(str).str.strip() != "").all()) if not out.empty else False

    qa_rows = [
        {"check": "monthly_cash_close_exists", "status": "pass" if close_path.exists() else "fail", "detail": str(close_path), "severity": "error"},
        {"check": "daily_cash_position_loaded", "status": "pass", "detail": f"rows={len(daily)}", "severity": "error"},
        {"check": "frontend_safe_cash_rows_count", "status": "pass", "detail": str(frontend_safe), "severity": "warning"},
        {"check": "unsafe_cash_rows_count", "status": "pass", "detail": str(unsafe), "severity": "warning"},
        {"check": "internal_balance_rows_count", "status": "pass", "detail": str(internal), "severity": "warning"},
        {"check": "box_motor_rows_count", "status": "pass" if box_loaded else "warn", "detail": str(box_motor), "severity": "warning"},
        {"check": "no_cash_total_without_frontend_safe_rows", "status": "pass", "detail": "no aggregate frontend-safe cash total emitted", "severity": "error"},
        {"check": "cash_close_by_currency_present", "status": "pass" if out["Currency"].astype(str).str.strip().ne("").any() else "fail", "detail": ",".join(sorted(out["Currency"].dropna().astype(str).unique())), "severity": "error"},
        {"check": "cash_close_caveats_present", "status": "pass" if caveats_present else "fail", "detail": "all rows have caveats" if caveats_present else "missing caveat", "severity": "error"},
        {"check": "daily_cash_position_not_frontend_safe", "status": "pass" if not out["source_table"].astype(str).eq("daily_cash_position.csv").any() or out.loc[out["source_table"].astype(str).eq("daily_cash_position.csv"), "is_frontend_safe"].astype(bool).eq(False).all() else "fail", "detail": "daily cash rows are internal-only", "severity": "error"},
        {"check": "box_balance_not_frontend_safe", "status": "pass" if not out["position_type"].astype(str).eq("inferred_box_motor").any() or out.loc[out["position_type"].astype(str).eq("inferred_box_motor"), "is_frontend_safe"].astype(bool).eq(False).all() else "fail", "detail": "box motor rows are reconciliation-only", "severity": "error"},
    ]
    _qa(qa_rows).to_csv(qa_path, index=False)
    return {"monthly_cash_close": close_path, "monthly_cash_close_qa": qa_path}
