from __future__ import annotations

"""Governed constructive stakeholder settlements.

The optional override is private source-side governance.  It identifies cases
and legs explicitly; this module never links rows by date, amount, provider,
account, or party similarity.
"""

from pathlib import Path
from typing import Any

import pandas as pd


DETAIL_COLUMNS = [
    "settlement_case_id", "obligation_box", "Date", "period", "Currency",
    "gross_amount", "expense_category", "allocation_status", "allocation_basis",
    "stakeholder_actor", "actor_role", "allocated_amount", "settlement_mode",
    "cash_path", "physical_payment_id", "physical_payer", "physical_payee",
    "payment_method", "evidence_ref", "evidence_status", "source_tx_id",
    "mirror_group_id", "leg_role", "underlying_participant",
    "underlying_allocated_amount", "review_note",
]
MONTHLY_COLUMNS = [
    "period", "period_end", "Currency", "target_box", "funding_actor",
    "actor_role", "funding_channel", "settlement_mode", "cash_path",
    "obligation_category", "recognized_amount", "physical_cash_amount",
    "n_tx", "settlement_case_ids", "source_tx_ids",
]
QA_COLUMNS = ["check", "settlement_case_id", "status", "severity", "amount", "detail"]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _load_override(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = sorted(set(DETAIL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"stakeholder settlement override missing columns: {missing}")
    frame["gross_amount"] = pd.to_numeric(frame["gross_amount"], errors="raise")
    frame["allocated_amount"] = pd.to_numeric(frame["allocated_amount"], errors="raise")
    frame["underlying_allocated_amount"] = pd.to_numeric(
        frame["underlying_allocated_amount"], errors="coerce"
    ).fillna(0.0)
    return frame[DETAIL_COLUMNS].copy()


def apply_stakeholder_settlements(
    audit: pd.DataFrame, override_path: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply explicit case legs and return enriched audit plus governed marts."""
    detail = _load_override(override_path)
    enriched = audit.copy()
    for column in [
        "settlement_case_id", "actor_role", "settlement_mode", "physical_payment_id",
        "physical_payer", "physical_payee", "payment_method", "evidence_ref",
        "evidence_status", "mirror_group_id", "leg_role",
    ]:
        if column not in enriched.columns:
            enriched[column] = ""

    source_legs = detail.loc[detail["source_tx_id"].astype(str).str.strip().ne("")]
    duplicated = source_legs["source_tx_id"].duplicated(keep=False)
    if duplicated.any():
        ids = sorted(source_legs.loc[duplicated, "source_tx_id"].unique())
        raise ValueError(f"source transaction belongs to multiple settlement legs: {ids}")

    audit_ids = set(enriched["tx_id"].astype(str))
    for _, leg in source_legs.iterrows():
        tx_id = _text(leg["source_tx_id"])
        role = _text(leg["leg_role"])
        if tx_id not in audit_ids:
            if role == "responsibility_mirror":
                continue
            raise ValueError(f"governed settlement source transaction absent from scoped ledger: {tx_id}")
        idx = enriched.index[enriched["tx_id"].astype(str).eq(tx_id)]
        if len(idx) != 1:
            raise ValueError(f"settlement source transaction is not unique: {tx_id}")
        i = idx[0]
        source_amount = float(pd.to_numeric(pd.Series([enriched.at[i, "amount"]]), errors="raise").iloc[0])
        expected = float(leg["allocated_amount"] if role == "stakeholder_support" else leg["gross_amount"])
        if role in {"stakeholder_support", "economic_expense"} and abs(source_amount - expected) > 0.01:
            raise ValueError(f"settlement source amount mismatch for {tx_id}: {source_amount} != {expected}")
        for column in [
            "settlement_case_id", "actor_role", "settlement_mode", "physical_payment_id",
            "physical_payer", "physical_payee", "payment_method", "evidence_ref",
            "evidence_status", "mirror_group_id", "leg_role",
        ]:
            enriched.at[i, column] = leg[column]
        enriched.at[i, "target_box"] = leg["obligation_box"]
        enriched.at[i, "obligation_box"] = leg["obligation_box"]
        enriched.at[i, "cash_path"] = leg["cash_path"]
        if role == "stakeholder_support":
            enriched.at[i, "funding_actor"] = leg["stakeholder_actor"]
            enriched.at[i, "funding_channel"] = "constructive_stakeholder_settlement"
            enriched.at[i, "cash_effect"] = "no_cash_in_box_direct_payment"
        elif role == "economic_expense":
            enriched.at[i, "funding_actor"] = ""
            enriched.at[i, "funding_channel"] = ""
            enriched.at[i, "cash_effect"] = "no_cash_out_box_direct_payment"

    qa_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for case_id, case in detail.groupby("settlement_case_id", sort=False):
        support = case.loc[case["leg_role"].eq("stakeholder_support")]
        expense = case.loc[case["leg_role"].eq("economic_expense")]
        mirrors = case.loc[case["leg_role"].eq("responsibility_mirror")]
        allocations = case.loc[
            case["leg_role"].eq("allocation_component")
            & case["allocation_status"].eq("agreed")
        ]
        gross_values = case["gross_amount"].astype(float).unique()
        gross = float(gross_values[0]) if len(gross_values) == 1 else float("nan")
        support_total = float(support["allocated_amount"].sum())
        expense_total = float(expense["gross_amount"].sum())
        allocation_total = float(allocations["underlying_allocated_amount"].sum())

        def add(check: str, ok: bool, amount: float, detail_text: str) -> None:
            qa_rows.append({
                "check": check, "settlement_case_id": case_id,
                "status": "pass" if ok else "fail", "severity": "error",
                "amount": amount, "detail": detail_text,
            })

        add("stakeholder_support_equals_gross_expense", abs(support_total-gross) <= .01, support_total-gross, f"support={support_total}; gross={gross}")
        add("economic_expense_appears_once", len(expense) == 1 and abs(expense_total-gross) <= .01, expense_total-gross, f"rows={len(expense)}; expense={expense_total}")
        add("mirrors_do_not_count_as_target_support", float(mirrors["allocated_amount"].sum()) == 0.0, float(mirrors["allocated_amount"].sum()), f"mirror_rows={len(mirrors)}")
        add("agreed_allocation_components_reconcile", allocation_total <= gross + .01, allocation_total, "underlying allocation is separately scoped and never added to PM support")
        add("constructive_settlement_has_no_physical_box_cash", case["physical_payment_id"].astype(str).str.strip().eq("").all(), 0.0, "physical payment remains evidence-driven")
        add("physical_uncertainty_is_explicit", case["evidence_status"].astype(str).eq("evidence_pending").all(), 0.0, "evidence_status=evidence_pending")

        for _, leg in support.iterrows():
            date = pd.Timestamp(leg["Date"])
            monthly_rows.append({
                "period": date.to_period("M").strftime("%Y-%m"),
                "period_end": date.to_period("M").end_time.date().isoformat(),
                "Currency": leg["Currency"], "target_box": leg["obligation_box"],
                "funding_actor": leg["stakeholder_actor"], "actor_role": leg["actor_role"],
                "funding_channel": "constructive_stakeholder_settlement",
                "settlement_mode": leg["settlement_mode"], "cash_path": leg["cash_path"],
                "obligation_category": leg["expense_category"],
                "recognized_amount": float(leg["allocated_amount"]),
                "physical_cash_amount": 0.0, "n_tx": 1,
                "settlement_case_ids": leg["settlement_case_id"],
                "source_tx_ids": leg["source_tx_id"],
            })

    monthly = pd.DataFrame(monthly_rows, columns=MONTHLY_COLUMNS)
    if not monthly.empty:
        dims = MONTHLY_COLUMNS[:10]
        monthly = monthly.groupby(dims, dropna=False, as_index=False).agg(
            recognized_amount=("recognized_amount", "sum"),
            physical_cash_amount=("physical_cash_amount", "sum"),
            n_tx=("n_tx", "sum"),
            settlement_case_ids=("settlement_case_ids", lambda s: ";".join(sorted(set(s)))),
            source_tx_ids=("source_tx_ids", lambda s: ";".join(sorted(set(s)))),
        )[MONTHLY_COLUMNS]
    qa = pd.DataFrame(qa_rows, columns=QA_COLUMNS)
    failures = qa.loc[qa["status"].eq("fail")]
    if not failures.empty:
        raise ValueError(f"stakeholder settlement QA failed: {failures.to_dict('records')}")
    return enriched, detail, monthly, qa


def write_stakeholder_outputs(
    audit: pd.DataFrame, *, out_dir: Path, override_path: Path | None
) -> tuple[pd.DataFrame, dict[str, Path]]:
    enriched, detail, monthly, qa = apply_stakeholder_settlements(audit, override_path)
    paths = {
        "stakeholder_settlement_detail": Path(out_dir) / "stakeholder_settlement_detail.csv",
        "monthly_stakeholder_support": Path(out_dir) / "monthly_stakeholder_support.csv",
        "monthly_stakeholder_support_qa": Path(out_dir) / "monthly_stakeholder_support_qa.csv",
    }
    detail.to_csv(paths["stakeholder_settlement_detail"], index=False)
    monthly.to_csv(paths["monthly_stakeholder_support"], index=False)
    qa.to_csv(paths["monthly_stakeholder_support_qa"], index=False)
    return enriched, paths
