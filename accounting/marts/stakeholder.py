from __future__ import annotations

"""Governed constructive stakeholder settlements.

Private overrides identify cases and legs explicitly. This module never links
transactions by similarity of date, amount, provider, account, actor or place.
"""

import hashlib
import json
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
    "underlying_allocated_amount", "review_note", "settlement_nature",
    "obligation_period", "settlement_period", "known_box_cash_funding",
    "other_governed_funding", "unresolved_funding", "funding_status", "debt_origin",
]
MONTHLY_COLUMNS = [
    "period", "period_end", "Currency", "target_box", "funding_actor",
    "actor_role", "funding_channel", "settlement_mode", "cash_path",
    "obligation_category", "settlement_nature", "obligation_period",
    "settlement_period", "reporting_group", "recognized_amount",
    "physical_cash_amount", "n_tx", "settlement_case_ids", "source_tx_ids",
]
QA_COLUMNS = ["check", "settlement_case_id", "status", "severity", "amount", "detail"]
SETTLEMENT_NATURES = {
    "current_period_support", "prior_period_clearing", "designated_funding",
    "reimbursement_or_pass_through", "unknown",
}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _load_override(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in DETAIL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    for column in ["gross_amount", "allocated_amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ["underlying_allocated_amount", "known_box_cash_funding", "other_governed_funding", "unresolved_funding"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    bad = set(frame["settlement_nature"].map(_text)) - SETTLEMENT_NATURES - {""}
    if bad:
        raise ValueError(f"unsupported settlement_nature values: {sorted(bad)}")
    return frame[DETAIL_COLUMNS].copy()


def private_input_provenance(path: Path | None) -> dict[str, object]:
    source = Path(path) if path is not None else None
    if source is None or not source.is_file():
        return {"present": False, "schema_version": "stakeholder_settlement_override.v2", "row_count": 0, "sha256": None}
    return {
        "present": True, "schema_version": "stakeholder_settlement_override.v2",
        "row_count": len(pd.read_csv(source, dtype=str)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def apply_stakeholder_settlements(audit: pd.DataFrame, override_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail = _load_override(override_path)
    enriched = audit.copy()
    enrich_cols = [
        "settlement_case_id", "actor_role", "settlement_mode", "physical_payment_id",
        "physical_payer", "physical_payee", "payment_method", "evidence_ref",
        "evidence_status", "mirror_group_id", "leg_role", "settlement_nature",
        "obligation_period", "settlement_period", "debt_origin",
    ]
    for column in enrich_cols:
        if column not in enriched.columns:
            enriched[column] = ""

    source_legs = detail.loc[detail["source_tx_id"].map(_text).ne("")]
    duplicated = source_legs["source_tx_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(f"source transaction belongs to multiple settlement legs: {sorted(source_legs.loc[duplicated, 'source_tx_id'].unique())}")
    audit_ids = set(enriched["tx_id"].astype(str))
    valid_roles = {"stakeholder_support", "economic_expense", "stakeholder_direct_expense", "box_cash_expense", "responsibility_mirror", "allocation_component"}
    for _, leg in source_legs.iterrows():
        tx_id, role = _text(leg["source_tx_id"]), _text(leg["leg_role"])
        if role not in valid_roles:
            raise ValueError(f"unsupported stakeholder settlement leg_role: {role}")
        if tx_id not in audit_ids:
            if role in {"responsibility_mirror", "allocation_component"}:
                continue
            raise ValueError(f"governed settlement source transaction absent from scoped ledger: {tx_id}")
        idx = enriched.index[enriched["tx_id"].astype(str).eq(tx_id)]
        if len(idx) != 1:
            raise ValueError(f"settlement source transaction is not unique: {tx_id}")
        i = idx[0]
        source_amount = float(pd.to_numeric(pd.Series([enriched.at[i, "amount"]]), errors="raise").iloc[0])
        expected = float(leg["allocated_amount"] if role == "stakeholder_support" else leg["gross_amount"])
        amount_roles = {"stakeholder_support", "economic_expense", "stakeholder_direct_expense", "box_cash_expense"}
        if role in amount_roles and abs(source_amount - expected) > .01:
            raise ValueError(f"settlement source amount mismatch for {tx_id}: {source_amount} != {expected}")
        for column in enrich_cols:
            enriched.at[i, column] = leg[column]
        enriched.at[i, "target_box"] = leg["obligation_box"]
        enriched.at[i, "obligation_box"] = leg["obligation_box"]
        enriched.at[i, "cash_path"] = leg["cash_path"]
        if role in {"stakeholder_support", "stakeholder_direct_expense"}:
            enriched.at[i, "funding_actor"] = leg["stakeholder_actor"]
            enriched.at[i, "funding_channel"] = "constructive_stakeholder_settlement"
            enriched.at[i, "cash_effect"] = "no_cash_out_box_direct_payment" if role == "stakeholder_direct_expense" else "no_cash_in_box_direct_payment"
        elif role == "economic_expense":
            enriched.at[i, "funding_actor"] = ""
            enriched.at[i, "funding_channel"] = ""
            if _text(leg["cash_path"]) == "direct_obligation_payment":
                enriched.at[i, "cash_effect"] = "no_cash_out_box_direct_payment"
        elif role == "box_cash_expense":
            enriched.at[i, "funding_actor"] = ""
            enriched.at[i, "funding_channel"] = ""

    qa_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for case_id, case in detail.groupby("settlement_case_id", sort=False):
        support = case.loc[case["leg_role"].isin(["stakeholder_support", "stakeholder_direct_expense"])]
        expense = case.loc[case["leg_role"].isin(["economic_expense", "stakeholder_direct_expense", "box_cash_expense"])]
        mirrors = case.loc[case["leg_role"].eq("responsibility_mirror")]
        allocations = case.loc[case["leg_role"].eq("allocation_component") & case["allocation_status"].eq("agreed")]
        gross_values = case["gross_amount"].astype(float).unique()
        gross = float(gross_values[0]) if len(gross_values) == 1 else float("nan")
        support_total = float(support["allocated_amount"].sum())
        expense_total = float(expense["gross_amount"].sum())
        allocation_total = float(allocations["underlying_allocated_amount"].sum())
        box_cash = float(case["known_box_cash_funding"].max())
        other = float(case["other_governed_funding"].max())
        unresolved = float(case["unresolved_funding"].max())

        def add(check: str, ok: bool, amount: float, detail_text: str) -> None:
            qa_rows.append({"check": check, "settlement_case_id": case_id, "status": "pass" if ok else "fail", "severity": "error", "amount": amount, "detail": detail_text})

        add("stakeholder_support_does_not_exceed_gross", support_total <= gross + .01, support_total-gross, f"support={support_total}; gross={gross}")
        add("economic_expense_reconciles_to_gross", abs(expense_total-gross) <= .01, expense_total-gross, f"rows={len(expense)}; expense={expense_total}")
        residual = gross-support_total-box_cash-other-unresolved
        add("funding_composition_reconciles", abs(residual) <= .01, residual, f"stakeholder={support_total}; box_cash={box_cash}; other={other}; unresolved={unresolved}; gross={gross}")
        add("mirrors_do_not_count_as_target_support", float(mirrors["allocated_amount"].sum()) == 0.0, float(mirrors["allocated_amount"].sum()), f"mirror_rows={len(mirrors)}")
        add("agreed_allocation_components_reconcile", allocation_total <= gross + .01, allocation_total, "underlying allocation is separately scoped")
        constructive = case["settlement_mode"].map(_text).eq("constructive")
        add("constructive_settlement_has_no_physical_box_cash", case.loc[constructive, "physical_payment_id"].map(_text).eq("").all(), 0.0, "physical payment remains evidence-driven")
        add("physical_uncertainty_is_explicit", case.loc[constructive, "evidence_status"].map(_text).isin(["evidence_pending", "unavailable"]).all(), 0.0, "constructive path remains explicit")

        for _, leg in support.iterrows():
            date = pd.Timestamp(leg["Date"])
            settlement_period = _text(leg["settlement_period"]) or date.to_period("M").strftime("%Y-%m")
            monthly_rows.append({
                "period": settlement_period, "period_end": pd.Period(settlement_period, freq="M").end_time.date().isoformat(),
                "Currency": leg["Currency"], "target_box": leg["obligation_box"],
                "funding_actor": leg["stakeholder_actor"], "actor_role": leg["actor_role"],
                "funding_channel": "constructive_stakeholder_settlement", "settlement_mode": leg["settlement_mode"],
                "cash_path": leg["cash_path"], "obligation_category": leg["expense_category"],
                "settlement_nature": _text(leg["settlement_nature"]) or "unknown",
                "obligation_period": leg["obligation_period"], "settlement_period": settlement_period,
                "reporting_group": leg["stakeholder_actor"], "recognized_amount": float(leg["allocated_amount"]),
                "physical_cash_amount": 0.0, "n_tx": 1, "settlement_case_ids": leg["settlement_case_id"],
                "source_tx_ids": leg["source_tx_id"],
            })

    monthly = pd.DataFrame(monthly_rows, columns=MONTHLY_COLUMNS)
    if not monthly.empty:
        measures = {"recognized_amount", "physical_cash_amount", "n_tx", "settlement_case_ids", "source_tx_ids"}
        dims = [c for c in MONTHLY_COLUMNS if c not in measures]
        monthly = monthly.groupby(dims, dropna=False, as_index=False).agg(
            recognized_amount=("recognized_amount", "sum"), physical_cash_amount=("physical_cash_amount", "sum"),
            n_tx=("n_tx", "sum"), settlement_case_ids=("settlement_case_ids", lambda s: ";".join(sorted(set(s)))),
            source_tx_ids=("source_tx_ids", lambda s: ";".join(sorted(set(s)))),
        )[MONTHLY_COLUMNS]
    qa = pd.DataFrame(qa_rows, columns=QA_COLUMNS)
    failures = qa.loc[qa["status"].eq("fail")]
    if not failures.empty:
        raise ValueError(f"stakeholder settlement QA failed: {failures.to_dict('records')}")
    return enriched, detail, monthly, qa


def write_stakeholder_outputs(audit: pd.DataFrame, *, out_dir: Path, override_path: Path | None) -> tuple[pd.DataFrame, dict[str, Path]]:
    enriched, detail, monthly, qa = apply_stakeholder_settlements(audit, override_path)
    paths = {
        "stakeholder_settlement_detail": Path(out_dir) / "stakeholder_settlement_detail.csv",
        "monthly_stakeholder_support": Path(out_dir) / "monthly_stakeholder_support.csv",
        "monthly_stakeholder_support_qa": Path(out_dir) / "monthly_stakeholder_support_qa.csv",
    }
    detail.to_csv(paths["stakeholder_settlement_detail"], index=False)
    monthly.to_csv(paths["monthly_stakeholder_support"], index=False)
    qa.to_csv(paths["monthly_stakeholder_support_qa"], index=False)
    provenance_path = Path(out_dir) / "meta" / "stakeholder_private_input.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(private_input_provenance(override_path), indent=2) + "\n", encoding="utf-8")
    paths["stakeholder_private_input"] = provenance_path
    return enriched, paths
