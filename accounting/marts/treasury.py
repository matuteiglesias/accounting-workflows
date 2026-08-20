from __future__ import annotations

"""Governed monthly treasury movement and accountability marts.

Economic classification explains a transaction; it does not establish that a
Box actually moved cash. Actual Box cash requires physical counterparty
evidence (``direction_source == box_party_match``).
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from accounting.cash_authority import select_validated_cash_period
from accounting.cutoff import load_run_cutoff_if_present


TREASURY_FLOW_COLUMNS = [
    "period", "period_end", "Box", "Currency", "movement_basis", "cash_direction",
    "cash_category", "semantic_bucket", "semantic_subbucket", "funding_actor",
    "funding_channel", "cash_effect", "debt_effect", "direction_source",
    "classification_status", "classification_confidence", "review_required",
    "amount_in", "amount_out", "net_amount", "non_cash_amount", "gross_amount",
    "n_tx", "n_review_required", "source_tx_ids_sample", "rule_ids",
    "source_table", "notes",
]
TREASURY_QA_COLUMNS = [
    "check", "period", "Box", "Currency", "treasury_net", "box_flow_net",
    "box_balance_net", "gap", "status", "severity", "detail",
]
ACCOUNTABILITY_COMPONENT_COLUMNS = [
    "rent_in", "funding_cash_in", "debt_principal_in", "debt_repayment_in",
    "debt_interest_in", "internal_transfer_in", "fx_in", "other_cash_in",
    "unknown_cash_in", "taxes_out", "services_out", "maintenance_out",
    "legal_out", "personal_draws_out", "dividends_out", "debt_principal_out",
    "debt_repayments_out", "debt_interest_out", "internal_transfer_out",
    "fx_out", "fx_cost_out", "other_cash_out", "unknown_cash_out",
    "direct_tax_support_non_cash", "direct_service_support_non_cash",
    "other_non_cash_support",
]
ACCOUNTABILITY_COLUMNS = [
    "period", "period_end", "control_as_of_date", "Box", "Currency",
    "opening_control", *ACCOUNTABILITY_COMPONENT_COLUMNS, "total_cash_in",
    "total_cash_out", "net_cash_flow", "closing_control", "box_motor_net",
    "box_flow_net", "reconciliation_gap", "reconciliation_status",
    "validated_cash_close", "validated_cash_status", "validated_cash_reason",
    "validated_as_of_date", "validated_account_count", "validated_anchor_offset",
    "anchor_reconciliation_gap", "anchor_alignment_status",
    "debt_engine_repayments", "debt_repayment_gap", "debt_reconciliation_status",
    "n_tx", "n_review_required",
]
ACCOUNTABILITY_QA_COLUMNS = [
    "check", "period", "Box", "Currency", "amount", "status", "severity", "detail"
]
TOLERANCE = 0.01


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truth(value: Any) -> bool:
    return _text(value).casefold() in {"true", "1", "yes", "y"}


def _cash_category(row: pd.Series) -> str:
    bucket = _text(row.get("semantic_bucket"))
    sub = _text(row.get("semantic_subbucket"))
    if bucket == "operating_revenue" and sub == "rent":
        return "rent"
    if bucket == "funding_contribution":
        return "funding"
    if bucket == "property_opex" and sub in {"taxes", "services", "maintenance", "legal"}:
        return sub
    if bucket == "family_withdrawal_candidate":
        if sub == "dividend":
            return "dividends"
        if sub in {"personal_expense", "transfer_to_family_expense"}:
            return "personal_draws"
        return "family_withdrawal"
    if bucket == "debt_movement":
        return {
            "principal": "debt_principal",
            "repayment": "debt_repayment",
            "interest": "debt_interest",
        }.get(sub, "debt_other")
    if bucket == "internal_transfer":
        return "internal_transfer"
    if bucket == "treasury_fx":
        return "fx_cost" if sub == "fx_cost_or_spread" else "fx_conversion"
    if bucket == "unknown":
        return "unknown"
    return "other"


def _movement_basis(row: pd.Series) -> str:
    direction_source = _text(row.get("direction_source"))
    direction = _text(row.get("direction"))
    cash_effect = _text(row.get("cash_effect"))
    if direction_source == "box_party_match" and direction in {"in", "out"}:
        return "actual_cash"
    if direction_source == "box_party_match" and direction == "internal":
        return "internal_box_transfer"
    if cash_effect in {"no_cash_in_box_direct_payment", "non_cash_support"}:
        return "non_cash_support"
    return "economic_only"


def _cash_direction(row: pd.Series) -> str:
    basis = _movement_basis(row)
    if basis == "actual_cash":
        return _text(row.get("direction"))
    if basis == "internal_box_transfer":
        return "internal"
    if basis == "non_cash_support":
        return "non_cash"
    return "none"


def _motor_frame(path: Path, *, flow: bool) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["period", "Box", "Currency", "net"])
    df = pd.read_csv(path)
    needed = {"TimePeriod", "Box", "Currency", "net"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} missing treasury reconciliation columns: {missing}")
    df = df.copy()
    df["net"] = pd.to_numeric(df["net"], errors="coerce").fillna(0.0)
    df["period"] = df["TimePeriod"].astype(str)
    if flow:
        return (
            df.groupby(["period", "Box", "Currency"], dropna=False, as_index=False)["net"]
            .sum()
        )
    return df[["period", "Box", "Currency", "net"]].copy()


def _build_treasury_qa(
    treasury: pd.DataFrame,
    *,
    out_dir: Path,
    freq: str,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    relevant = treasury.loc[
        treasury["movement_basis"].isin(["actual_cash", "internal_box_transfer"])
    ].copy()
    treasury_net = (
        relevant.groupby(["period", "Box", "Currency"], dropna=False, as_index=False)["net_amount"]
        .sum()
        .rename(columns={"net_amount": "treasury_net"})
    )
    balance_path = out_dir / f"box_balance_time_long.freq={freq}.csv"
    flow_path = out_dir / f"box_flow_balance_time_long.freq={freq}.csv"
    balance = _motor_frame(balance_path, flow=False).rename(columns={"net": "box_balance_net"})
    flow = _motor_frame(flow_path, flow=True).rename(columns={"net": "box_flow_net"})

    if balance.empty or flow.empty:
        rows.append({
            "check": "treasury_motor_reconciliation_available",
            "period": "", "Box": "", "Currency": "",
            "treasury_net": pd.NA, "box_flow_net": pd.NA, "box_balance_net": pd.NA,
            "gap": pd.NA, "status": "warn", "severity": "warning",
            "detail": f"motor evidence absent balance={balance_path.exists()} flow={flow_path.exists()}",
        })
    else:
        keys = pd.concat([
            treasury_net[["period", "Box", "Currency"]],
            balance[["period", "Box", "Currency"]],
            flow[["period", "Box", "Currency"]],
        ], ignore_index=True).drop_duplicates()
        check = (
            keys.merge(treasury_net, on=["period", "Box", "Currency"], how="left")
            .merge(flow, on=["period", "Box", "Currency"], how="left")
            .merge(balance, on=["period", "Box", "Currency"], how="left")
        )
        for col in ["treasury_net", "box_flow_net", "box_balance_net"]:
            check[col] = pd.to_numeric(check[col], errors="coerce")
        for _, row in check.iterrows():
            values_present = not pd.isna(row["treasury_net"]) and not pd.isna(row["box_flow_net"]) and not pd.isna(row["box_balance_net"])
            gap = float("inf")
            if values_present:
                gap = max(
                    abs(float(row["treasury_net"]) - float(row["box_balance_net"])),
                    abs(float(row["box_flow_net"]) - float(row["box_balance_net"])),
                )
            ok = bool(gap <= tolerance)
            rows.append({
                "check": "treasury_net_equals_box_motors",
                "period": row["period"], "Box": row["Box"], "Currency": row["Currency"],
                "treasury_net": row["treasury_net"], "box_flow_net": row["box_flow_net"],
                "box_balance_net": row["box_balance_net"], "gap": gap,
                "status": "pass" if ok else "fail", "severity": "error",
                "detail": "semantic actual-cash net must equal both physical Box motor nets",
            })

    noncash_leak = treasury.loc[
        ~treasury["movement_basis"].eq("actual_cash")
        & (
            treasury["amount_in"].abs().gt(tolerance)
            | treasury["amount_out"].abs().gt(tolerance)
            | treasury["net_amount"].abs().gt(tolerance)
        )
    ]
    rows.append({
        "check": "non_cash_never_alters_cash_arithmetic",
        "period": "", "Box": "", "Currency": "",
        "treasury_net": pd.NA, "box_flow_net": pd.NA, "box_balance_net": pd.NA,
        "gap": float(len(noncash_leak)),
        "status": "pass" if noncash_leak.empty else "fail",
        "severity": "error",
        "detail": f"leaking_rows={len(noncash_leak)}",
    })
    bad_actual = treasury.loc[
        treasury["movement_basis"].eq("actual_cash")
        & ~treasury["direction_source"].eq("box_party_match")
    ]
    rows.append({
        "check": "actual_cash_requires_box_party_match",
        "period": "", "Box": "", "Currency": "",
        "treasury_net": pd.NA, "box_flow_net": pd.NA, "box_balance_net": pd.NA,
        "gap": float(len(bad_actual)),
        "status": "pass" if bad_actual.empty else "fail",
        "severity": "error",
        "detail": f"bad_rows={len(bad_actual)}",
    })
    qa = pd.DataFrame(rows, columns=TREASURY_QA_COLUMNS)
    failures = qa.loc[qa["severity"].eq("error") & qa["status"].eq("fail")]
    if not failures.empty:
        detail = failures[["check", "period", "Box", "Currency", "gap"]].to_dict("records")
        raise ValueError(f"Monthly Box treasury hard reconciliation failed: {detail[:10]}")
    return qa


def build_monthly_box_treasury_flow(
    audit: pd.DataFrame,
    *,
    out_dir: Path,
    freq: str = "M",
    tolerance: float = TOLERANCE,
) -> Dict[str, Path]:
    """Build semantic monthly cash movement using physical Box evidence only."""
    required = {
        "tx_id", "period", "period_end", "Box", "Currency", "amount", "direction",
        "direction_source", "semantic_bucket", "semantic_subbucket",
        "classification_status", "classification_confidence", "review_required",
        "rule_id", "funding_actor", "funding_channel", "cash_effect", "debt_effect",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(f"classified transaction frame missing treasury columns: {missing}")

    work = audit.copy()
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)
    work["movement_basis"] = work.apply(_movement_basis, axis=1)
    work["cash_direction"] = work.apply(_cash_direction, axis=1)
    work["cash_category"] = work.apply(_cash_category, axis=1)
    actual_in = work["movement_basis"].eq("actual_cash") & work["cash_direction"].eq("in")
    actual_out = work["movement_basis"].eq("actual_cash") & work["cash_direction"].eq("out")
    work["amount_in"] = work["amount"].where(actual_in, 0.0)
    work["amount_out"] = work["amount"].where(actual_out, 0.0)
    work["net_amount"] = work["amount_in"] - work["amount_out"]
    work["non_cash_amount"] = work["amount"].where(
        work["movement_basis"].eq("non_cash_support"), 0.0
    )
    work["gross_amount"] = work["amount"].abs()
    work["__review"] = work["review_required"].map(_truth).astype(int)

    group_cols = [
        "period", "period_end", "Box", "Currency", "movement_basis", "cash_direction",
        "cash_category", "semantic_bucket", "semantic_subbucket", "funding_actor",
        "funding_channel", "cash_effect", "debt_effect", "direction_source",
        "classification_status", "classification_confidence",
    ]
    monthly = (
        work.groupby(group_cols, dropna=False)
        .agg(
            amount_in=("amount_in", "sum"),
            amount_out=("amount_out", "sum"),
            net_amount=("net_amount", "sum"),
            non_cash_amount=("non_cash_amount", "sum"),
            gross_amount=("gross_amount", "sum"),
            n_tx=("tx_id", "size"),
            n_review_required=("__review", "sum"),
            source_tx_ids_sample=("tx_id", lambda s: ";".join(s.astype(str).head(20))),
            rule_ids=("rule_id", lambda s: ";".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )
    monthly["review_required"] = monthly["n_review_required"].gt(0)
    monthly["source_table"] = "ledger_canonical.csv"
    monthly["notes"] = (
        "Actual cash requires direction_source=box_party_match; semantic fallback "
        "explains economics but never manufactures Box cash."
    )
    monthly = monthly[TREASURY_FLOW_COLUMNS].sort_values(
        ["period", "Box", "Currency", "movement_basis", "cash_category", "cash_direction"]
    ).reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_path = out_dir / "monthly_box_treasury_flow.csv"
    qa_path = out_dir / "monthly_box_treasury_flow_qa.csv"
    monthly.to_csv(flow_path, index=False)
    qa = _build_treasury_qa(monthly, out_dir=out_dir, freq=freq, tolerance=tolerance)
    qa.to_csv(qa_path, index=False)
    return {
        "monthly_box_treasury_flow": flow_path,
        "monthly_box_treasury_flow_qa": qa_path,
    }


