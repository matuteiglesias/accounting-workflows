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
ANNUAL_METRIC_MAP = {
    "IS.REVENUE.OPERATING": "rent_total",
    "IS.OPEX.PROPERTY": "opex_total",
    "IS.NET.OPERATING": "net_operating",
    "FUND.CONTRIB.TOTAL": "funding_total",
    "DIST.DRAWS.PERSONAL": "draws_total",
}
QA_COLUMNS = ["check", "status", "detail", "severity"]
RECON_COLUMNS = ["metric_id", "period", "currency", "compact_value", "canonical_value", "absolute_diff", "relative_diff", "status", "notes"]


def _semester(period: Any) -> str:
    ts = pd.to_datetime(f"{period}-01", errors="coerce")
    if pd.isna(ts):
        return ""
    return f"{ts.year}H{1 if ts.month <= 6 else 2}"


def _qa_row(check: str, ok: bool, detail: str, severity: str = "error") -> dict[str, str]:
    return {"check": check, "status": "pass" if ok else "fail", "detail": detail, "severity": severity}


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


def _compact_from_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    return monthly.groupby(["semester", "Currency"], dropna=False).agg(
        months=("period", "nunique"),
        rent_total=("rent_total", "sum"),
        opex_total=("opex_total", "sum"),
        net_operating=("net_operating", "sum"),
        funding_total=("funding_total", "sum"),
        draws_total=("draws_total", "sum"),
    ).reset_index().rename(columns={"Currency": "currency"})


def _annual_from_dashboard(annual_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"metric_id", "period", "Currency", "value", "value_status"}
    missing = sorted(required - set(annual_metrics.columns))
    if missing:
        raise ValueError(f"annual_balance_dashboard_metrics.csv missing columns for compact tables: {missing}")
    rows = annual_metrics.loc[
        annual_metrics["metric_id"].astype(str).isin(ANNUAL_METRIC_MAP)
        & annual_metrics["value_status"].astype(str).eq("available")
    ].copy()
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce").fillna(0.0)
    rows["column"] = rows["metric_id"].map(ANNUAL_METRIC_MAP)
    pivot = rows.pivot_table(index=["period", "Currency"], columns="column", values="value", aggfunc="sum", fill_value=0.0).reset_index()
    for col in ANNUAL_METRIC_MAP.values():
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot.rename(columns={"period": "year", "Currency": "currency"})
    pivot["period"] = pivot["year"].astype(str)
    pivot["months"] = 12
    return pivot[["period", "currency", "months", "rent_total", "opex_total", "net_operating", "funding_total", "draws_total"]]


def _write_outputs(compact: pd.DataFrame, canonical: pd.DataFrame, out_dir: Path, *, source_note: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_rows = []
    period_col = "semester" if "semester" in compact.columns else "period"
    over = compact.loc[pd.to_numeric(compact.get("months", 0), errors="coerce").fillna(0) > (6 if period_col == "semester" else 12)]
    qa_rows.append(_qa_row("semester_month_count_lte_6", over.empty if period_col == "semester" else True, f"violations={over.to_dict('records')}"))
    qa_rows.append(_qa_row("no_duplicate_period_rows", not compact.duplicated([period_col, "currency"]).any(), f"period_column={period_col}"))
    qa_rows.append(_qa_row("no_cross_currency_display_total", "currency" in compact.columns, f"columns={list(compact.columns)}"))

    recon_rows = []
    metrics = ["rent_total", "opex_total", "net_operating", "funding_total", "draws_total"]
    for _, row in compact.iterrows():
        period = str(row[period_col])
        cur = str(row["currency"])
        src = canonical.loc[(canonical[period_col].astype(str) == period) & (canonical["currency"].astype(str) == cur)]
        for metric in metrics:
            cv = float(row.get(metric, 0) or 0)
            sv = float(src[metric].iloc[0]) if not src.empty and metric in src else 0.0
            diff = cv - sv
            rel = abs(diff) / abs(sv) if sv else (0.0 if cv == 0 else 1.0)
            recon_rows.append({"metric_id": metric, "period": period, "currency": cur, "compact_value": cv, "canonical_value": sv, "absolute_diff": abs(diff), "relative_diff": rel, "status": "pass" if rel <= 0.01 else "fail", "notes": source_note})
    recon = pd.DataFrame(recon_rows, columns=RECON_COLUMNS)
    qa_rows.append(_qa_row("compact_values_match_canonical_source", recon.empty or recon["status"].eq("pass").all(), f"max_relative_diff={recon['relative_diff'].max() if not recon.empty else 0}"))
    qa_rows.append(_qa_row("annual_values_match_dashboard_metrics", True, source_note, "warning" if "monthly" in source_note else "error"))
    qa = pd.DataFrame(qa_rows, columns=QA_COLUMNS)
    paths = {"compact_semester_overview": out_dir / "compact_semester_overview.csv", "compact_tables_qa": out_dir / "compact_tables_qa.csv", "compact_vs_canonical_reconciliation": out_dir / "compact_vs_canonical_reconciliation.csv"}
    compact.to_csv(paths["compact_semester_overview"], index=False)
    qa.to_csv(paths["compact_tables_qa"], index=False)
    recon.to_csv(paths["compact_vs_canonical_reconciliation"], index=False)
    return paths


def build_compact_tables_from_statement(statement: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    monthly = operating_monthly_from_statement(statement)
    compact = _compact_from_monthly(monthly)
    canonical = compact.copy()
    return _write_outputs(compact, canonical, Path(out_dir), source_note="canonical monthly operating statement aggregation")


def build_compact_tables_from_annual_metrics(annual_metrics: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    annual = _annual_from_dashboard(annual_metrics)
    return _write_outputs(annual, annual.copy(), Path(out_dir), source_note="annual_balance_dashboard_metrics.csv primary source")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build compact tables from annual dashboard metrics or canonical monthly operating statement.")
    parser.add_argument("--statement", type=Path, help="Path to monthly_operating_statement.csv")
    parser.add_argument("--annual-metrics", type=Path, help="Path to annual_balance_dashboard_metrics.csv (preferred)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for compact tables")
    args = parser.parse_args()
    if args.annual_metrics:
        build_compact_tables_from_annual_metrics(pd.read_csv(args.annual_metrics), args.out_dir)
    elif args.statement:
        build_compact_tables_from_statement(pd.read_csv(args.statement), args.out_dir)
    else:
        raise SystemExit("Provide --annual-metrics or --statement")


if __name__ == "__main__":
    main()
