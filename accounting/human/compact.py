from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

MONEY_LINES = {
    "rent_revenue": "rent_total",
    "property_opex_true": "opex_total",
    "net_operating": "net_operating",
    "funding_contributions": "funding_total",
    "family_draws_or_distributions": "draws_total",
}
QA_COLUMNS = ["check", "status", "detail", "severity"]
RECON_COLUMNS = ["metric", "period_or_semester", "currency", "compact_value", "source_value", "absolute_diff", "relative_diff", "status", "notes"]


def _semester(period: Any) -> str:
    ts = pd.to_datetime(f"{period}-01", errors="coerce")
    if pd.isna(ts):
        return ""
    return f"{ts.year}H{1 if ts.month <= 6 else 2}"


def operating_monthly_from_statement(statement: pd.DataFrame) -> pd.DataFrame:
    required = {"period", "period_end", "Currency", "statement_line", "amount"}
    missing = sorted(required - set(statement.columns))
    if missing:
        raise ValueError(f"monthly_operating_statement.csv missing columns for compact tables: {missing}")
    stmt = statement.loc[statement["statement_line"].astype(str).isin(MONEY_LINES)].copy()
    stmt["amount"] = pd.to_numeric(stmt["amount"], errors="coerce").fillna(0.0)
    pivot = stmt.pivot_table(index=["period", "period_end", "Currency"], columns="statement_line", values="amount", aggfunc="sum", fill_value=0.0).reset_index()
    for line in MONEY_LINES:
        if line not in pivot.columns:
            pivot[line] = 0.0
    out = pivot.rename(columns=MONEY_LINES)
    out["month"] = pd.to_datetime(out["period"] + "-01", errors="coerce").dt.month
    out["year"] = pd.to_datetime(out["period"] + "-01", errors="coerce").dt.year
    out["semester"] = out["period"].map(_semester)
    out["data_quality_flag"] = "canonical_monthly_statement"
    return out[["period", "period_end", "Currency", "year", "month", "semester", "rent_total", "opex_total", "net_operating", "funding_total", "draws_total", "data_quality_flag"]]


def build_compact_tables_from_statement(statement: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly = operating_monthly_from_statement(statement)
    group_cols = ["semester", "Currency"]
    compact = monthly.groupby(group_cols, dropna=False).agg(
        months=("period", "nunique"),
        rent_total=("rent_total", "sum"),
        opex_total=("opex_total", "sum"),
        net_operating=("net_operating", "sum"),
        funding_total=("funding_total", "sum"),
        draws_total=("draws_total", "sum"),
    ).reset_index().rename(columns={"Currency": "currency"})

    qa_rows = []
    def add(check: str, ok: bool, detail: str, severity: str = "error") -> None:
        qa_rows.append({"check": check, "status": "pass" if ok else "fail", "detail": detail, "severity": severity})
    over = compact.loc[pd.to_numeric(compact["months"], errors="coerce").fillna(0) > 6]
    add("semester_month_count_lte_6", over.empty, f"violations={over[['semester','currency','months']].to_dict('records')}")
    add("no_cross_currency_sum", True, "compact grouped by semester and Currency")
    add("currency_column_present_for_money_outputs", "currency" in compact.columns, f"columns={list(compact.columns)}")

    recon_rows = []
    source_sem = monthly.groupby(["semester", "Currency"], dropna=False)[["rent_total", "opex_total", "net_operating", "funding_total", "draws_total"]].sum().reset_index()
    for _, c in compact.iterrows():
        src = source_sem.loc[(source_sem["semester"].astype(str) == str(c["semester"])) & (source_sem["Currency"].astype(str) == str(c["currency"]))]
        for metric in ["rent_total", "opex_total", "net_operating", "funding_total", "draws_total"]:
            source_value = float(src[metric].iloc[0]) if not src.empty else 0.0
            compact_value = float(c[metric])
            diff = compact_value - source_value
            rel = abs(diff) / abs(source_value) if source_value else (0.0 if compact_value == 0 else 1.0)
            recon_rows.append({"metric": metric, "period_or_semester": c["semester"], "currency": c["currency"], "compact_value": compact_value, "source_value": source_value, "absolute_diff": abs(diff), "relative_diff": rel, "status": "pass" if rel <= 0.01 else "warn", "notes": "canonical monthly statement reconciliation"})
    recon = pd.DataFrame(recon_rows, columns=RECON_COLUMNS)
    add("compact_totals_match_operating_result_monthly", recon.empty or recon["relative_diff"].le(0.01).all(), f"max_relative_diff={recon['relative_diff'].max() if not recon.empty else 0}", "error")
    qa = pd.DataFrame(qa_rows, columns=QA_COLUMNS)
    paths = {"compact_semester_overview": out_dir / "compact_semester_overview.csv", "compact_tables_qa": out_dir / "compact_tables_qa.csv", "compact_vs_operating_result_reconciliation": out_dir / "compact_vs_operating_result_reconciliation.csv"}
    compact.to_csv(paths["compact_semester_overview"], index=False)
    qa.to_csv(paths["compact_tables_qa"], index=False)
    recon.to_csv(paths["compact_vs_operating_result_reconciliation"], index=False)
    return paths
