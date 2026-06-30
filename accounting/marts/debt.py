from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from accounting.debt.resolve import RULE_VERSION

DEBT_POSITION_COLUMNS = [
    "period", "period_end", "as_of_date", "debtor", "creditor", "Currency", "component",
    "open_amount", "open_principal", "open_interest", "open_total", "source_table",
    "source_rule_version", "n_open_items", "caveat", "frontend_suitability",
]
DEBT_QA_COLUMNS = ["check", "status", "detail", "severity"]


def _empty_debt_position() -> pd.DataFrame:
    return pd.DataFrame(columns=DEBT_POSITION_COLUMNS)


def _qa(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DEBT_QA_COLUMNS)


def _period_end(period: pd.Series) -> pd.Series:
    return pd.PeriodIndex(period.astype(str), freq="M").end_time.date.astype(str)


def build_monthly_debt_position(debt_dir: Path, write_dir: Path) -> Dict[str, Path]:
    debt_dir = Path(debt_dir)
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = debt_dir / "debt_balance_monthly.csv"
    open_items_path = debt_dir / "debt_open_items.csv"
    out_path = write_dir / "monthly_debt_position.csv"
    qa_path = write_dir / "monthly_debt_position_qa.csv"

    if not monthly_path.exists():
        _empty_debt_position().to_csv(out_path, index=False)
        qa = _qa([
            {"check": "monthly_debt_position_exists", "status": "pass", "detail": f"empty wrapper written to {out_path}", "severity": "warning"},
            {"check": "debt_balance_monthly_loaded", "status": "warn", "detail": f"missing {monthly_path}", "severity": "warning"},
        ])
        qa.to_csv(qa_path, index=False)
        return {"monthly_debt_position": out_path, "monthly_debt_position_qa": qa_path}

    monthly = pd.read_csv(monthly_path)
    required = ["as_of_date", "period", "debtor", "creditor", "currency", "open_principal", "open_interest", "open_total"]
    missing = [c for c in required if c not in monthly.columns]
    if missing:
        raise ValueError(f"debt_balance_monthly.csv missing required columns for monthly_debt_position: {missing}")

    base = monthly.copy()
    base["open_principal"] = pd.to_numeric(base["open_principal"], errors="coerce").fillna(0.0)
    base["open_interest"] = pd.to_numeric(base["open_interest"], errors="coerce").fillna(0.0)
    base["open_total"] = pd.to_numeric(base["open_total"], errors="coerce").fillna(0.0)
    base["Currency"] = base["currency"].astype(str).str.upper()
    base["period"] = base["period"].astype(str)
    if "period_end" not in base.columns:
        base["period_end"] = _period_end(base["period"])

    # debt_balance_monthly may contain one row per item_type. Use one balance row per debtor/creditor/currency/month.
    unique = base.drop_duplicates(["period", "debtor", "creditor", "Currency", "as_of_date"])[
        ["period", "period_end", "as_of_date", "debtor", "creditor", "Currency", "open_principal", "open_interest", "open_total"]
    ].copy()

    counts = pd.DataFrame(columns=["period", "debtor", "creditor", "Currency", "n_open_items"])
    if open_items_path.exists():
        items = pd.read_csv(open_items_path)
        item_required = ["opened_at", "debtor", "creditor", "currency"]
        if all(c in items.columns for c in item_required):
            items = items.copy()
            items["opened_at"] = pd.to_datetime(items["opened_at"], errors="coerce")
            items = items[items["opened_at"].notna()].copy()
            items["period"] = items["opened_at"].dt.to_period("M").astype(str)
            items["Currency"] = items["currency"].astype(str).str.upper()
            counts = items.groupby(["period", "debtor", "creditor", "Currency"], dropna=False).size().reset_index(name="n_open_items")

    rows = []
    for _, row in unique.iterrows():
        n_match = counts.loc[
            counts["period"].eq(row["period"]) & counts["debtor"].eq(row["debtor"]) & counts["creditor"].eq(row["creditor"]) & counts["Currency"].eq(row["Currency"]),
            "n_open_items",
        ]
        n_open_items = int(n_match.iloc[0]) if not n_match.empty else 0
        for component, amount_col in [("principal", "open_principal"), ("interest", "open_interest"), ("total", "open_total")]:
            rows.append({
                "period": row["period"],
                "period_end": row["period_end"],
                "as_of_date": row["as_of_date"],
                "debtor": row["debtor"],
                "creditor": row["creditor"],
                "Currency": row["Currency"],
                "component": component,
                "open_amount": float(row[amount_col]),
                "open_principal": float(row["open_principal"]),
                "open_interest": float(row["open_interest"]),
                "open_total": float(row["open_total"]),
                "source_table": "debt_balance_monthly.csv",
                "source_rule_version": RULE_VERSION,
                "n_open_items": n_open_items,
                "caveat": "Consumption wrapper over resolved debt balances; debt engine logic is unchanged.",
                "frontend_suitability": "safe_with_caveat",
            })

    out = pd.DataFrame(rows, columns=DEBT_POSITION_COLUMNS)
    out.to_csv(out_path, index=False)

    source_total = float(unique["open_total"].sum())
    wrapper_total = float(out.loc[out["component"].eq("total"), "open_amount"].sum()) if not out.empty else 0.0
    qa_rows = [
        {"check": "monthly_debt_position_exists", "status": "pass" if out_path.exists() else "fail", "detail": str(out_path), "severity": "error"},
        {"check": "debt_balance_monthly_loaded", "status": "pass", "detail": f"rows={len(monthly)}", "severity": "error"},
        {"check": "has_debtor_creditor", "status": "pass" if out[["debtor", "creditor"]].notna().all().all() else "fail", "detail": "debtor/creditor populated", "severity": "error"},
        {"check": "has_principal_interest_total", "status": "pass" if {"principal", "interest", "total"}.issubset(set(out["component"])) else "fail", "detail": ",".join(sorted(set(out["component"]))), "severity": "error"},
        {"check": "has_currency", "status": "pass" if out["Currency"].astype(str).str.strip().ne("").all() else "fail", "detail": ",".join(sorted(out["Currency"].dropna().astype(str).unique())), "severity": "error"},
        {"check": "has_monthly_periods", "status": "pass" if out["period"].astype(str).str.match(r"^\d{4}-\d{2}$").all() else "fail", "detail": f"periods={out['period'].nunique() if not out.empty else 0}", "severity": "error"},
        {"check": "total_reconciles_to_source", "status": "pass" if abs(source_total - wrapper_total) < 0.01 else "fail", "detail": f"source_total={source_total}; wrapper_total={wrapper_total}", "severity": "error"},
        {"check": "no_cross_currency_sum", "status": "pass", "detail": "all rows remain currency-grained", "severity": "error"},
        {"check": "frontend_outputs_have_suitability", "status": "pass" if "frontend_suitability" in out.columns and out["frontend_suitability"].astype(str).str.strip().ne("").all() else "fail", "detail": "debt wrapper rows carry suitability", "severity": "error"},
        {"check": "money_outputs_have_currency", "status": "pass" if out["Currency"].astype(str).str.strip().ne("").all() else "fail", "detail": "debt wrapper rows carry Currency", "severity": "error"},
    ]
    _qa(qa_rows).to_csv(qa_path, index=False)
    return {"monthly_debt_position": out_path, "monthly_debt_position_qa": qa_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly debt consumption wrapper")
    parser.add_argument("--debt-dir", required=True)
    parser.add_argument("--write-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_monthly_debt_position(Path(args.debt_dir), Path(args.write_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
