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


def _cash_component(category: str, direction: str) -> str:
    if direction == "in":
        return {
            "rent": "rent_in",
            "funding": "funding_cash_in",
            "debt_principal": "debt_principal_in",
            "debt_repayment": "debt_repayment_in",
            "debt_interest": "debt_interest_in",
            "internal_transfer": "internal_transfer_in",
            "fx_conversion": "fx_in",
            "unknown": "unknown_cash_in",
        }.get(category, "other_cash_in")
    return {
        "taxes": "taxes_out",
        "services": "services_out",
        "maintenance": "maintenance_out",
        "legal": "legal_out",
        "personal_draws": "personal_draws_out",
        "dividends": "dividends_out",
        "debt_principal": "debt_principal_out",
        "debt_repayment": "debt_repayments_out",
        "debt_interest": "debt_interest_out",
        "internal_transfer": "internal_transfer_out",
        "fx_conversion": "fx_out",
        "fx_cost": "fx_cost_out",
        "unknown": "unknown_cash_out",
    }.get(category, "other_cash_out")


def _noncash_component(category: str) -> str:
    if category == "taxes":
        return "direct_tax_support_non_cash"
    if category == "services":
        return "direct_service_support_non_cash"
    return "other_non_cash_support"


def _load_monthly_motor(run_root: Path, name: str, *, flow: bool) -> pd.DataFrame:
    path = run_root / f"{name}.freq=M.csv"
    df = _motor_frame(path, flow=flow)
    if df.empty:
        raise FileNotFoundError(f"Missing monthly treasury motor: {path}")
    return df


def _period_end(period: str) -> str:
    return pd.Period(str(period), freq="M").end_time.date().isoformat()


def _numeric(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(parsed) else float(parsed)


def build_monthly_cash_accountability(
    run_root: Path,
    *,
    tolerance: float = TOLERANCE,
) -> Dict[str, Path]:
    """Assemble report-safe monthly treasury accountability after debt resolution."""
    run_root = Path(run_root)
    flow_path = run_root / "monthly_box_treasury_flow.csv"
    if not flow_path.exists():
        raise FileNotFoundError(f"Missing canonical treasury flow: {flow_path}")
    treasury = pd.read_csv(flow_path)

    raw_balance = pd.read_csv(run_root / "box_balance_time_long.freq=M.csv")
    raw_balance = raw_balance.rename(
        columns={
            "TimePeriod": "period",
            "TimePeriod_end": "period_end",
            "net": "box_motor_net",
            "cum_net": "closing_control",
        }
    )
    required_balance = {
        "period", "period_end", "Box", "Currency", "box_motor_net", "closing_control"
    }
    missing_balance = sorted(required_balance - set(raw_balance.columns))
    if missing_balance:
        raise ValueError(f"box balance missing accountability columns: {missing_balance}")
    for col in ["box_motor_net", "closing_control"]:
        raw_balance[col] = pd.to_numeric(raw_balance[col], errors="coerce").fillna(0.0)
    observed_controls = raw_balance[
        ["period", "period_end", "Box", "Currency", "box_motor_net", "closing_control"]
    ].copy().rename(columns={"closing_control": "observed_closing_control"})
    observed_box_flow = _load_monthly_motor(
        run_root, "box_flow_balance_time_long", flow=True
    ).rename(columns={"net": "box_flow_net"})

    # The physical Box motor is intentionally sparse. Accountability is monthly:
    # preserve explicit non-cash support in zero-movement months without inventing
    # cash, and carry the zero-origin control through those months.
    relevant_basis = {"actual_cash", "non_cash_support", "internal_box_transfer"}
    relevant_treasury = treasury.loc[
        treasury["movement_basis"].astype(str).isin(relevant_basis)
    ].copy()
    key_rows = pd.concat(
        [
            observed_controls[["period", "Box", "Currency"]],
            observed_box_flow[["period", "Box", "Currency"]],
            relevant_treasury[["period", "Box", "Currency"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    spine_rows: list[dict[str, str]] = []
    for (box, currency), group in key_rows.groupby(["Box", "Currency"], dropna=False):
        periods = group["period"].astype(str).map(lambda value: pd.Period(value, freq="M"))
        if periods.empty:
            continue
        for period in pd.period_range(periods.min(), periods.max(), freq="M"):
            period_text = str(period)
            spine_rows.append(
                {
                    "period": period_text,
                    "period_end": _period_end(period_text),
                    "Box": _text(box),
                    "Currency": _text(currency),
                }
            )
    controls = pd.DataFrame(spine_rows)
    controls = controls.merge(
        observed_controls.drop(columns=["period_end"]),
        on=["period", "Box", "Currency"],
        how="left",
    ).merge(
        observed_box_flow,
        on=["period", "Box", "Currency"],
        how="left",
    )
    controls["box_motor_observed"] = controls["box_motor_net"].notna()
    controls["box_flow_observed"] = controls["box_flow_net"].notna()
    presence_mismatch = controls.loc[
        controls["box_motor_observed"].ne(controls["box_flow_observed"])
    ]
    if not presence_mismatch.empty:
        bad = presence_mismatch[["period", "Box", "Currency"]].to_dict("records")
        raise ValueError(f"Treasury accountability motor presence mismatch: {bad[:10]}")
    controls["box_motor_net"] = pd.to_numeric(
        controls["box_motor_net"], errors="coerce"
    ).fillna(0.0)
    controls["box_flow_net"] = pd.to_numeric(
        controls["box_flow_net"], errors="coerce"
    ).fillna(0.0)
    controls = controls.sort_values(["Box", "Currency", "period"]).reset_index(drop=True)
    controls["closing_control"] = controls.groupby(
        ["Box", "Currency"], dropna=False
    )["box_motor_net"].cumsum()
    observed_gap = controls["closing_control"] - pd.to_numeric(
        controls["observed_closing_control"], errors="coerce"
    )
    bad_observed = controls.loc[
        controls["observed_closing_control"].notna() & observed_gap.abs().gt(tolerance)
    ]
    if not bad_observed.empty:
        bad = bad_observed.assign(observed_gap=observed_gap.loc[bad_observed.index])[
            ["period", "Box", "Currency", "observed_gap"]
        ].to_dict("records")
        raise ValueError(f"Treasury reconstructed control disagrees with Box cum_net: {bad[:10]}")

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, row in controls.iterrows():
        key = (_text(row["period"]), _text(row["Box"]), _text(row["Currency"]))
        records[key] = {
            "period": key[0],
            "period_end": _text(row["period_end"]) or _period_end(key[0]),
            "Box": key[1],
            "Currency": key[2],
            **{col: 0.0 for col in ACCOUNTABILITY_COMPONENT_COLUMNS},
            "n_tx": 0,
            "n_review_required": 0,
        }

    for _, row in relevant_treasury.iterrows():
        basis = _text(row["movement_basis"])
        key = (_text(row["period"]), _text(row["Box"]), _text(row["Currency"]))
        if key not in records:
            raise ValueError(f"Treasury accountability spine omitted relevant key: {key}")
        rec = records[key]
        if basis == "actual_cash":
            direction = _text(row["cash_direction"])
            col = _cash_component(_text(row["cash_category"]), direction)
            amount = row["amount_in"] if direction == "in" else row["amount_out"]
            rec[col] += _numeric(amount)
        elif basis == "non_cash_support":
            col = _noncash_component(_text(row["cash_category"]))
            rec[col] += _numeric(row["non_cash_amount"])
        rec["n_tx"] += int(_numeric(row.get("n_tx", 0)))
        rec["n_review_required"] += int(_numeric(row.get("n_review_required", 0)))

    out = pd.DataFrame(records.values())
    out = out.merge(
        controls[
            [
                "period", "period_end", "Box", "Currency", "box_motor_net",
                "closing_control", "box_flow_net",
            ]
        ],
        on=["period", "period_end", "Box", "Currency"],
        how="left",
    )
    if out[["box_motor_net", "closing_control", "box_flow_net"]].isna().any().any():
        bad = out.loc[
            out[["box_motor_net", "closing_control", "box_flow_net"]].isna().any(axis=1),
            ["period", "Box", "Currency"],
        ].to_dict("records")
        raise ValueError(f"Treasury accountability missing completed motor key(s): {bad[:10]}")

    cash_in_cols = [c for c in ACCOUNTABILITY_COMPONENT_COLUMNS if c.endswith("_in")]
    cash_out_cols = [c for c in ACCOUNTABILITY_COMPONENT_COLUMNS if c.endswith("_out")]
    out["total_cash_in"] = out[cash_in_cols].sum(axis=1)
    out["total_cash_out"] = out[cash_out_cols].sum(axis=1)
    out["net_cash_flow"] = out["total_cash_in"] - out["total_cash_out"]
    out["opening_control"] = out["closing_control"] - out["box_motor_net"]
    out["reconciliation_gap"] = out["net_cash_flow"] - out["box_motor_net"]
    motor_gap = out["box_flow_net"] - out["box_motor_net"]
    control_gap = out["opening_control"] + out["net_cash_flow"] - out["closing_control"]
    hard_gap = pd.concat(
        [out["reconciliation_gap"].abs(), motor_gap.abs(), control_gap.abs()],
        axis=1,
    ).max(axis=1)
    out["reconciliation_status"] = hard_gap.le(tolerance).map(
        {True: "reconciled", False: "fail"}
    )
    if out["reconciliation_status"].eq("fail").any():
        bad = out.loc[
            out["reconciliation_status"].eq("fail"),
            ["period", "Box", "Currency", "reconciliation_gap"],
        ].to_dict("records")
        raise ValueError(f"Monthly cash accountability hard reconciliation failed: {bad[:10]}")

    cutoff = load_run_cutoff_if_present(run_root)
    out["control_as_of_date"] = out["period_end"].astype(str)
    if cutoff is not None:
        cutoff_period = cutoff.date[:7]
        out.loc[out["period"].astype(str).eq(cutoff_period), "control_as_of_date"] = cutoff.date

    cash_path = run_root / "monthly_cash_close.csv"
    cash = pd.read_csv(cash_path) if cash_path.exists() else pd.DataFrame()
    selected_meta: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        if cash.empty:
            selected_meta.append({
                "validated_cash_close": pd.NA,
                "validated_cash_status": "unavailable",
                "validated_cash_reason": "missing_source",
                "validated_as_of_date": "",
                "validated_account_count": 0,
                "validated_anchor_offset": pd.NA,
                "anchor_reconciliation_gap": pd.NA,
                "anchor_alignment_status": "unavailable",
            })
            continue
        selection = select_validated_cash_period(
            cash,
            period=_text(row["period"]),
            currency=_text(row["Currency"]),
            box=_text(row["Box"]),
        )
        if not selection.available:
            selected_meta.append({
                "validated_cash_close": pd.NA,
                "validated_cash_status": selection.status,
                "validated_cash_reason": selection.reason,
                "validated_as_of_date": "",
                "validated_account_count": 0,
                "validated_anchor_offset": pd.NA,
                "anchor_reconciliation_gap": pd.NA,
                "anchor_alignment_status": "unavailable",
            })
            continue
        dates = sorted(
            selection.selected["as_of_date"]
            .fillna("")
            .astype(str)
            .str.strip()
            .loc[lambda s: s.ne("")]
            .unique()
            .tolist()
        )
        coherent_date = dates[0] if len(dates) == 1 else ""
        eligible = bool(
            coherent_date and coherent_date == _text(row["control_as_of_date"])
        )
        offset = (
            float(selection.value) - float(row["closing_control"])
            if eligible and selection.value is not None
            else pd.NA
        )
        reason = ""
        if not eligible:
            reason = (
                "anchor_date_misaligned"
                if len(dates) == 1
                else "selected_accounts_have_mixed_as_of_dates"
            )
        selected_meta.append({
            "validated_cash_close": float(selection.value) if selection.value is not None else pd.NA,
            "validated_cash_status": "available",
            "validated_cash_reason": reason,
            "validated_as_of_date": ";".join(dates),
            "validated_account_count": int(len(selection.selected)),
            "validated_anchor_offset": offset,
            "anchor_reconciliation_gap": pd.NA,
            "anchor_alignment_status": "pending" if eligible else "unavailable",
        })
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(selected_meta)], axis=1)

    out = out.sort_values(["Box", "Currency", "period"]).reset_index(drop=True)
    for _, idx in out.groupby(["Box", "Currency"], sort=False).groups.items():
        prior_offset: float | None = None
        for i in idx:
            value = pd.to_numeric(
                pd.Series([out.at[i, "validated_anchor_offset"]]), errors="coerce"
            ).iloc[0]
            if pd.isna(value):
                continue
            value = float(value)
            if prior_offset is None:
                out.at[i, "anchor_alignment_status"] = "first_anchor"
                out.at[i, "anchor_reconciliation_gap"] = pd.NA
            else:
                gap = value - prior_offset
                out.at[i, "anchor_reconciliation_gap"] = gap
                out.at[i, "anchor_alignment_status"] = (
                    "reconciled" if abs(gap) <= tolerance else "residual"
                )
            prior_offset = value

    debt_path = run_root / "monthly_debt_activity.csv"
    debt = pd.read_csv(debt_path) if debt_path.exists() else pd.DataFrame()
    if not debt.empty:
        debt["repayments"] = pd.to_numeric(debt["repayments"], errors="coerce").fillna(0.0)
        repayment = debt.loc[debt["activity_type"].astype(str).eq("repayment")].copy()
        debt_repayments = (
            repayment.groupby(["period", "debtor", "Currency"], dropna=False, as_index=False)["repayments"]
            .sum()
        )
        coverage = set(zip(debt["debtor"].astype(str), debt["Currency"].astype(str)))
    else:
        debt_repayments = pd.DataFrame(columns=["period", "debtor", "Currency", "repayments"])
        coverage = set()

    engine_vals = []
    for _, row in out.iterrows():
        pair = (_text(row["Box"]), _text(row["Currency"]))
        if pair not in coverage:
            engine_vals.append((pd.NA, pd.NA, "unavailable"))
            continue
        match = debt_repayments.loc[
            debt_repayments["period"].astype(str).eq(_text(row["period"]))
            & debt_repayments["debtor"].astype(str).eq(_text(row["Box"]))
            & debt_repayments["Currency"].astype(str).eq(_text(row["Currency"]))
        ]
        engine = float(match["repayments"].sum()) if not match.empty else 0.0
        gap = float(row["debt_repayments_out"]) - engine
        engine_vals.append(
            (engine, gap, "reconciled" if abs(gap) <= tolerance else "residual")
        )
    out["debt_engine_repayments"] = [v[0] for v in engine_vals]
    out["debt_repayment_gap"] = [v[1] for v in engine_vals]
    out["debt_reconciliation_status"] = [v[2] for v in engine_vals]

    qa_rows: list[dict[str, Any]] = [
        {
            "check": "cash_components_reconcile_to_box_motor",
            "period": "", "Box": "", "Currency": "",
            "amount": float(out["reconciliation_gap"].abs().max()) if not out.empty else 0.0,
            "status": "pass", "severity": "error",
            "detail": "total_cash_in-total_cash_out equals physical Box motor net",
        },
        {
            "check": "opening_plus_net_equals_closing_control",
            "period": "", "Box": "", "Currency": "",
            "amount": float(control_gap.abs().max()) if not out.empty else 0.0,
            "status": "pass", "severity": "error",
            "detail": "zero-origin inferred control only; not validated liquidity",
        },
    ]
    for _, row in out.iterrows():
        unknown = float(row["unknown_cash_in"]) + float(row["unknown_cash_out"])
        if abs(unknown) > tolerance:
            qa_rows.append({
                "check": "unknown_actual_cash_visible",
                "period": row["period"], "Box": row["Box"], "Currency": row["Currency"],
                "amount": unknown, "status": "warn", "severity": "warning",
                "detail": "actual Box cash exists with unknown semantic classification",
            })
        if int(row["n_review_required"]) > 0:
            qa_rows.append({
                "check": "review_required_cash_or_support_visible",
                "period": row["period"], "Box": row["Box"], "Currency": row["Currency"],
                "amount": float(row["n_review_required"]),
                "status": "warn", "severity": "warning",
                "detail": "review-required treasury evidence remains visible",
            })
        if row["debt_reconciliation_status"] == "residual":
            qa_rows.append({
                "check": "cash_debt_repayment_matches_debt_engine",
                "period": row["period"], "Box": row["Box"], "Currency": row["Currency"],
                "amount": row["debt_repayment_gap"], "status": "warn", "severity": "warning",
                "detail": "cash says money moved; debt engine says how much was allocated",
            })
        if row["anchor_alignment_status"] == "residual":
            qa_rows.append({
                "check": "validated_anchor_offsets_align",
                "period": row["period"], "Box": row["Box"], "Currency": row["Currency"],
                "amount": row["anchor_reconciliation_gap"], "status": "warn", "severity": "warning",
                "detail": "validated cash anchors imply inconsistent opening offset",
            })

    out = out[ACCOUNTABILITY_COLUMNS].sort_values(
        ["period", "Box", "Currency"]
    ).reset_index(drop=True)
    accountability_path = run_root / "monthly_cash_accountability.csv"
    qa_path = run_root / "monthly_cash_accountability_qa.csv"
    out.to_csv(accountability_path, index=False)
    pd.DataFrame(qa_rows, columns=ACCOUNTABILITY_QA_COLUMNS).to_csv(qa_path, index=False)
    return {
        "monthly_cash_accountability": accountability_path,
        "monthly_cash_accountability_qa": qa_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build monthly Box cash accountability after debt resolution."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    paths = build_monthly_cash_accountability(
        Path(args.run_root), tolerance=args.tolerance
    )
    for path in paths.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
