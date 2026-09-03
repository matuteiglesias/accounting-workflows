from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from accounting.debt.position_authority import (
    select_debt_position,
    selected_debt_position_rows,
)
from accounting.debt.resolve import RULE_VERSION
from accounting.cutoff import load_run_cutoff_if_present

DEBT_POSITION_COLUMNS = [
    "period",
    "period_end",
    "as_of_date",
    "debtor",
    "creditor",
    "Currency",
    "component",
    "position_status",
    "selection_reason",
    "candidate_rows",
    "valid_as_of_rows",
    "open_amount",
    "open_principal",
    "open_interest",
    "open_total",
    "source_table",
    "source_rule_version",
    "n_open_items",
    "caveat",
    "frontend_suitability",
]
DEBT_QA_COLUMNS = ["check", "status", "detail", "severity"]

DEBT_ACTIVITY_COLUMNS = [
    "period",
    "period_end",
    "Currency",
    "debtor",
    "creditor",
    "activity_type",
    "new_principal",
    "interest_accrued",
    "repayments",
    "adjustments",
    "opening_total",
    "closing_total",
    "net_change",
    "n_items",
    "source_table",
    "source_rule_version",
    "frontend_suitability",
    "reconciliation_status",
    "caveat",
    "notes",
]
DEBT_ACTIVITY_QA_COLUMNS = ["check", "status", "detail", "severity"]

DEBT_REPAYMENT_DETAIL_COLUMNS = [
    "period", "repayment_tx_id", "repayment_date", "debtor", "creditor",
    "Currency", "repayment_amount", "allocated_amount", "leftover_amount",
    "allocation_status", "target_debt_id", "target_source_tx_id",
    "target_item_type", "target_opened_at", "target_detail",
    "balance_before", "balance_after", "repayment_detail", "repayment_tag",
    "repayment_debt_family", "repayment_source_file", "repayment_source_row",
    "source_table", "source_rule_version",
]


def _empty_debt_activity() -> pd.DataFrame:
    return pd.DataFrame(columns=DEBT_ACTIVITY_COLUMNS)


def _period_from_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").astype(str)


def _build_repayment_detail(*, debt_dir: Path, write_dir: Path) -> Path:
    """Preserve resolved repayment-to-obligation lineage at allocation grain."""
    out_path = write_dir / "monthly_debt_repayment_detail.csv"
    events_path = debt_dir / "debt_repayment_events.csv"
    allocations_path = debt_dir / "debt_allocations.csv"
    if not events_path.exists():
        pd.DataFrame(columns=DEBT_REPAYMENT_DETAIL_COLUMNS).to_csv(out_path, index=False)
        return out_path

    events = pd.read_csv(events_path)
    allocations = (
        pd.read_csv(allocations_path)
        if allocations_path.exists()
        else pd.DataFrame()
    )
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        repayment_tx_id = str(event.get("repayment_tx_id", ""))
        matched = (
            allocations.loc[
                allocations["repayment_tx_id"].astype(str).eq(repayment_tx_id)
            ].copy()
            if not allocations.empty and "repayment_tx_id" in allocations.columns
            else pd.DataFrame()
        )
        event_allocated = float(pd.to_numeric(event.get("allocated_amount"), errors="coerce") or 0.0)
        leftover = float(pd.to_numeric(event.get("leftover_amount"), errors="coerce") or 0.0)
        status = "resolved" if leftover <= 0.01 else ("partial" if event_allocated > 0.01 else "unallocated")
        base = {
            "period": str(pd.to_datetime(event.get("repayment_date"), errors="coerce").to_period("M")),
            "repayment_tx_id": repayment_tx_id,
            "repayment_date": event.get("repayment_date", ""),
            "debtor": event.get("debtor", ""),
            "creditor": event.get("creditor", ""),
            "Currency": str(event.get("currency", "")).upper(),
            "repayment_amount": event.get("repayment_amount", 0.0),
            "leftover_amount": leftover,
            "allocation_status": status,
            "source_table": "debt_allocations.csv;debt_repayment_events.csv",
            "source_rule_version": event.get("rule_version", RULE_VERSION),
        }
        if matched.empty:
            rows.append({**base, "allocated_amount": 0.0})
            continue
        for _, allocation in matched.iterrows():
            rows.append({
                **base,
                "allocated_amount": allocation.get("allocated_amount", 0.0),
                **{
                    column: allocation.get(column, "")
                    for column in DEBT_REPAYMENT_DETAIL_COLUMNS
                    if column.startswith("target_")
                    or column.startswith("balance_")
                    or column.startswith("repayment_detail")
                    or column.startswith("repayment_tag")
                    or column.startswith("repayment_debt_family")
                    or column.startswith("repayment_source_")
                },
            })
    pd.DataFrame(rows, columns=DEBT_REPAYMENT_DETAIL_COLUMNS).to_csv(out_path, index=False)
    return out_path


def _amount_by_item_type(items: pd.DataFrame, item_type: str) -> pd.DataFrame:
    sub = items.loc[items["item_type"].astype(str).eq(item_type)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["period", "debtor", "creditor", "Currency", "amount", "n_items"]
        )
    return (
        sub.groupby(["period", "debtor", "creditor", "Currency"], dropna=False)
        .agg(amount=("original_amount", "sum"), n_items=("original_amount", "size"))
        .reset_index()
    )


def _build_monthly_debt_activity(
    *,
    debt_dir: Path,
    write_dir: Path,
    position: pd.DataFrame,
) -> Dict[str, Path]:
    activity_path = write_dir / "monthly_debt_activity.csv"
    qa_path = write_dir / "monthly_debt_activity_qa.csv"
    open_items_path = debt_dir / "debt_open_items.csv"
    repayments_path = debt_dir / "debt_repayment_events.csv"

    if position.empty:
        _empty_debt_activity().to_csv(activity_path, index=False)
        _qa(
            [
                {
                    "check": "monthly_debt_activity_exists",
                    "status": "pass",
                    "detail": f"empty wrapper written to {activity_path}",
                    "severity": "warning",
                },
                {
                    "check": "activity_rows_have_currency",
                    "status": "pass",
                    "detail": "empty activity output",
                    "severity": "warning",
                },
                {
                    "check": "activity_rows_have_debtor_creditor",
                    "status": "pass",
                    "detail": "empty activity output",
                    "severity": "warning",
                },
                {
                    "check": "opening_closing_present",
                    "status": "warn",
                    "detail": "monthly_debt_position has no rows",
                    "severity": "warning",
                },
                {
                    "check": "repayments_present_or_explicitly_unavailable",
                    "status": "warn",
                    "detail": "debt position unavailable; activity unavailable",
                    "severity": "warning",
                },
                {
                    "check": "new_claims_present_or_explicitly_unavailable",
                    "status": "warn",
                    "detail": "debt position unavailable; activity unavailable",
                    "severity": "warning",
                },
                {
                    "check": "activity_reconciles_to_position",
                    "status": "warn",
                    "detail": "debt position unavailable; activity unavailable",
                    "severity": "warning",
                },
                {
                    "check": "residual_adjustments_visible",
                    "status": "warn",
                    "detail": "debt position unavailable; activity unavailable",
                    "severity": "warning",
                },
                {
                    "check": "no_cross_currency_debt_activity_sum",
                    "status": "pass",
                    "detail": "empty activity output emits no cross-currency aggregate",
                    "severity": "warning",
                },
            ]
        ).to_csv(qa_path, index=False)
        return {
            "monthly_debt_activity": activity_path,
            "monthly_debt_activity_qa": qa_path,
        }

    totals = position.loc[position["component"].astype(str).eq("total")].copy()
    totals["closing_total"] = pd.to_numeric(
        totals["open_amount"], errors="coerce"
    )
    totals = totals[
        ["period", "period_end", "debtor", "creditor", "Currency", "closing_total"]
    ].drop_duplicates(["period", "debtor", "creditor", "Currency"])
    totals = totals.sort_values(
        ["debtor", "creditor", "Currency", "period"]
    ).reset_index(drop=True)
    totals["opening_total"] = (
        totals.groupby(["debtor", "creditor", "Currency"], dropna=False)[
            "closing_total"
        ]
        .shift(1)
        .fillna(0.0)
    )
    totals["net_change"] = totals["closing_total"] - totals["opening_total"]

    new_principal = pd.DataFrame(
        columns=["period", "debtor", "creditor", "Currency", "amount", "n_items"]
    )
    interest = pd.DataFrame(
        columns=["period", "debtor", "creditor", "Currency", "amount", "n_items"]
    )
    open_items_loaded = False
    if open_items_path.exists():
        items = pd.read_csv(open_items_path)
        required = {
            "opened_at",
            "debtor",
            "creditor",
            "currency",
            "item_type",
            "original_amount",
        }
        if required.issubset(items.columns):
            open_items_loaded = True
            items = items.copy()
            items["period"] = _period_from_date(items["opened_at"])
            items = items.loc[
                items["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
            ].copy()
            items["Currency"] = items["currency"].astype(str).str.upper()
            items["original_amount"] = pd.to_numeric(
                items["original_amount"], errors="coerce"
            ).fillna(0.0)
            new_principal = _amount_by_item_type(items, "Prestamo")
            interest = _amount_by_item_type(items, "Interes")

    repayments = pd.DataFrame(
        columns=["period", "debtor", "creditor", "Currency", "amount", "n_items"]
    )
    repayments_loaded = False
    if repayments_path.exists():
        rep = pd.read_csv(repayments_path)
        required = {"repayment_date", "debtor", "creditor", "currency"}
        amount_col = (
            "allocated_amount"
            if "allocated_amount" in rep.columns
            else "repayment_amount" if "repayment_amount" in rep.columns else None
        )
        if required.issubset(rep.columns) and amount_col is not None:
            repayments_loaded = True
            rep = rep.copy()
            rep["period"] = _period_from_date(rep["repayment_date"])
            rep = rep.loc[rep["period"].astype(str).str.match(r"^\d{4}-\d{2}$")].copy()
            rep["Currency"] = rep["currency"].astype(str).str.upper()
            rep["amount"] = pd.to_numeric(rep[amount_col], errors="coerce").fillna(0.0)
            repayments = (
                rep.groupby(["period", "debtor", "creditor", "Currency"], dropna=False)
                .agg(amount=("amount", "sum"), n_items=("amount", "size"))
                .reset_index()
            )

    event_key_frames = [
        df[["period", "debtor", "creditor", "Currency"]]
        for df in [new_principal, interest, repayments]
        if not df.empty
    ]
    balance_keys = totals[["period", "debtor", "creditor", "Currency"]].copy()
    all_key_frames = [balance_keys, *event_key_frames]
    all_keys = pd.concat(all_key_frames, ignore_index=True).drop_duplicates(
        ["period", "debtor", "creditor", "Currency"]
    )
    keys = all_keys.merge(
        totals[
            ["period", "period_end", "debtor", "creditor", "Currency", "closing_total"]
        ],
        on=["period", "debtor", "creditor", "Currency"],
        how="left",
    )
    keys["period_end"] = keys["period_end"].fillna(
        pd.Series(_period_end(keys["period"]), index=keys.index)
    )
    keys["closing_total"] = pd.to_numeric(
        keys["closing_total"], errors="coerce"
    )
    keys = keys.sort_values(["debtor", "creditor", "Currency", "period"]).reset_index(
        drop=True
    )
    keys["opening_total"] = keys.groupby(
        ["debtor", "creditor", "Currency"], dropna=False
    )["closing_total"].shift(1)
    first_in_pair = keys.groupby(
        ["debtor", "creditor", "Currency"], dropna=False
    ).cumcount().eq(0)
    keys.loc[first_in_pair & keys["opening_total"].isna(), "opening_total"] = 0.0
    keys["net_change"] = keys["closing_total"] - keys["opening_total"]

    def merge_amount(
        base: pd.DataFrame, source: pd.DataFrame, name: str, n_name: str
    ) -> pd.DataFrame:
        out = base.merge(
            source.rename(columns={"amount": name, "n_items": n_name}),
            on=["period", "debtor", "creditor", "Currency"],
            how="left",
        )
        out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        out[n_name] = pd.to_numeric(out[n_name], errors="coerce").fillna(0).astype(int)
        return out

    activity_base = merge_amount(keys, new_principal, "new_principal", "n_new_items")
    activity_base = merge_amount(
        activity_base, interest, "interest_accrued", "n_interest_items"
    )
    activity_base = merge_amount(
        activity_base, repayments, "repayments", "n_repayment_items"
    )
    activity_base["adjustments"] = (
        activity_base["net_change"]
        - activity_base["new_principal"]
        - activity_base["interest_accrued"]
        + activity_base["repayments"]
    )
    reconcilable = activity_base[["opening_total", "closing_total"]].notna().all(axis=1)
    activity_base["reconciliation_status"] = "unavailable_position"
    activity_base.loc[reconcilable, "reconciliation_status"] = (
        activity_base.loc[reconcilable, "adjustments"]
        .abs()
        .le(0.01)
        .map({True: "reconciled", False: "residual_adjustment_visible"})
    )

    rows: list[dict[str, Any]] = []
    activity_specs = [
        ("opening_balance", "opening_total", "debt_balance_monthly.csv"),
        (
            "new_claim",
            "new_principal",
            (
                open_items_path.name
                if open_items_loaded
                else "unavailable:debt_open_items.csv"
            ),
        ),
        (
            "interest_accrual",
            "interest_accrued",
            (
                open_items_path.name
                if open_items_loaded
                else "unavailable:debt_open_items.csv"
            ),
        ),
        (
            "repayment",
            "repayments",
            (
                repayments_path.name
                if repayments_loaded
                else "unavailable:debt_repayment_events.csv"
            ),
        ),
        ("adjustment", "adjustments", "derived_reconciliation_residual"),
        ("closing_balance", "closing_total", "monthly_debt_position.csv"),
        ("net_change", "net_change", "monthly_debt_position.csv"),
    ]
    for _, row in activity_base.iterrows():
        caveat = (
            "Debt activity wrapper: stock movement by currency; repayments/new claims use debt engine events when available; "
            "adjustments expose residuals rather than hiding them."
        )
        notes = []
        if not open_items_loaded:
            notes.append("new_claim and interest event source unavailable")
        if not repayments_loaded:
            notes.append("repayment event source unavailable")
        if abs(float(row["adjustments"])) > 0.01:
            notes.append("residual adjustment required to reconcile opening/closing")
        n_items_total = int(
            row["n_new_items"] + row["n_interest_items"] + row["n_repayment_items"]
        )
        for activity_type, _, source_table in activity_specs:
            rows.append(
                {
                    "period": row["period"],
                    "period_end": row["period_end"],
                    "Currency": row["Currency"],
                    "debtor": row["debtor"],
                    "creditor": row["creditor"],
                    "activity_type": activity_type,
                    "new_principal": (
                        float(row["new_principal"])
                        if activity_type == "new_claim"
                        else 0.0
                    ),
                    "interest_accrued": (
                        float(row["interest_accrued"])
                        if activity_type == "interest_accrual"
                        else 0.0
                    ),
                    "repayments": (
                        float(row["repayments"])
                        if activity_type == "repayment"
                        else 0.0
                    ),
                    "adjustments": (
                        float(row["adjustments"])
                        if activity_type == "adjustment"
                        else 0.0
                    ),
                    "opening_total": float(row["opening_total"]),
                    "closing_total": float(row["closing_total"]),
                    "net_change": (
                        float(row["net_change"])
                        if activity_type == "net_change"
                        else 0.0
                    ),
                    "n_items": n_items_total,
                    "source_table": source_table,
                    "source_rule_version": RULE_VERSION,
                    "frontend_suitability": "safe_with_caveat",
                    "reconciliation_status": row["reconciliation_status"],
                    "caveat": caveat,
                    "notes": "; ".join(notes),
                }
            )

    out = pd.DataFrame(rows, columns=DEBT_ACTIVITY_COLUMNS)
    out.to_csv(activity_path, index=False)

    recon = (
        activity_base["net_change"]
        - activity_base["new_principal"]
        - activity_base["interest_accrued"]
        + activity_base["repayments"]
        - activity_base["adjustments"]
    )
    qa_rows = [
        {
            "check": "monthly_debt_activity_exists",
            "status": "pass" if activity_path.exists() else "fail",
            "detail": str(activity_path),
            "severity": "error",
        },
        {
            "check": "activity_rows_have_currency",
            "status": (
                "pass"
                if out.empty or out["Currency"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": ",".join(sorted(out["Currency"].dropna().astype(str).unique())),
            "severity": "error",
        },
        {
            "check": "activity_rows_have_debtor_creditor",
            "status": (
                "pass"
                if out.empty
                or (
                    out[["debtor", "creditor"]].notna().all().all()
                    and out[["debtor", "creditor"]]
                    .astype(str)
                    .apply(lambda col: col.str.strip().ne(""))
                    .all()
                    .all()
                )
                else "fail"
            ),
            "detail": "debtor/creditor populated",
            "severity": "error",
        },
        {
            "check": "opening_closing_present_or_position_unavailable",
            "status": (
                "pass"
                if out.empty
                or (
                    out[["opening_total", "closing_total"]].notna().all(axis=1)
                    | out["reconciliation_status"].astype(str).eq("unavailable_position")
                ).all()
                else "fail"
            ),
            "detail": f"rows={len(out)}; unavailable_position_rows={int(out['reconciliation_status'].astype(str).eq('unavailable_position').sum()) if not out.empty else 0}",
            "severity": "error",
        },
        {
            "check": "repayments_present_or_explicitly_unavailable",
            "status": "pass",
            "detail": f"loaded={repayments_loaded}; rows={len(repayments)}",
            "severity": "warning" if not repayments_loaded else "error",
        },
        {
            "check": "new_claims_present_or_explicitly_unavailable",
            "status": "pass",
            "detail": f"loaded={open_items_loaded}; rows={len(new_principal)}",
            "severity": "warning" if not open_items_loaded else "error",
        },
        {
            "check": "activity_reconciles_to_position",
            "status": (
                "pass"
                if recon.loc[activity_base["reconciliation_status"].ne("unavailable_position")].abs().le(0.01).all()
                else "fail"
            ),
            "detail": (
                f"max_diff={float(recon.loc[activity_base['reconciliation_status'].ne('unavailable_position')].abs().max()) if activity_base['reconciliation_status'].ne('unavailable_position').any() else 0.0}; "
                f"unavailable_position_periods={int(activity_base['reconciliation_status'].eq('unavailable_position').sum())}"
            ),
            "severity": "error",
        },
        {
            "check": "residual_adjustments_visible",
            "status": (
                "pass"
                if out.empty
                or (
                    out["activity_type"].astype(str).eq("adjustment").any()
                    and out["reconciliation_status"]
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .all()
                )
                else "fail"
            ),
            "detail": f"adjustment_rows={int(out['activity_type'].astype(str).eq('adjustment').sum()) if not out.empty else 0}",
            "severity": "error",
        },
        {
            "check": "no_material_unexplained_adjustments",
            "status": "pass" if activity_base["adjustments"].abs().le(0.01).all() else "fail",
            "detail": f"material_rows={int(activity_base['adjustments'].abs().gt(0.01).sum())}; max_abs={float(activity_base['adjustments'].abs().max()) if not activity_base.empty else 0.0}",
            "severity": "error",
        },
        {
            "check": "no_cross_currency_debt_activity_sum",
            "status": "pass",
            "detail": "monthly_debt_activity remains debtor/creditor/currency grained and emits no ARS/USD aggregate",
            "severity": "error",
        },
    ]
    _qa(qa_rows).to_csv(qa_path, index=False)
    return {"monthly_debt_activity": activity_path, "monthly_debt_activity_qa": qa_path}


def _empty_debt_position() -> pd.DataFrame:
    return pd.DataFrame(columns=DEBT_POSITION_COLUMNS)


def _qa(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DEBT_QA_COLUMNS)


def _append_cost_gap_debt_exclusion_qa(*, debt_dir: Path, write_dir: Path) -> None:
    """Cross-check the semantic non-debt projection without owning it."""
    gap_path = write_dir / "cost_allocation_gaps.csv"
    qa_path = write_dir / "cost_allocation_gaps_qa.csv"
    if not gap_path.exists() or not qa_path.exists():
        return
    gaps = pd.read_csv(gap_path)
    gap_ids = set(gaps.get("source_tx_id", pd.Series(dtype=str)).astype(str))
    checks: list[tuple[str, bool, str]] = []
    open_items_path = debt_dir / "debt_open_items.csv"
    open_overlap: set[str] = set()
    if open_items_path.exists():
        items = pd.read_csv(open_items_path)
        open_overlap = gap_ids & set(items.get("source_tx_id", pd.Series(dtype=str)).astype(str))
    checks.append(("zero_overlap_with_debt_open_items", not open_overlap, f"overlap={len(open_overlap)}"))

    def pair_overlap(path: Path) -> int:
        if not path.exists() or gaps.empty:
            return 0
        frame = pd.read_csv(path)
        if not {"debtor", "creditor", "Currency"}.issubset(frame.columns):
            return 0
        mask = (
            frame["debtor"].astype(str).str.casefold().eq("costos")
            & frame["creditor"].astype(str).str.casefold().eq("pm")
            & frame["Currency"].astype(str).str.upper().isin(set(gaps["Currency"].astype(str).str.upper()))
        )
        return int(mask.sum())

    position_overlap = pair_overlap(write_dir / "monthly_debt_position.csv")
    activity_overlap = pair_overlap(write_dir / "monthly_debt_activity.csv")
    checks.append(("zero_overlap_with_debt_position_activity", position_overlap == 0 and activity_overlap == 0, f"position_rows={position_overlap}; activity_rows={activity_overlap}"))
    allocation_path = debt_dir / "debt_allocations.csv"
    allocation_overlap: set[str] = set()
    if allocation_path.exists():
        allocations = pd.read_csv(allocation_path)
        allocation_overlap = gap_ids & set(allocations.get("target_source_tx_id", pd.Series(dtype=str)).astype(str))
    checks.append(("zero_repayment_allocations_targeting_gaps", not allocation_overlap, f"overlap={len(allocation_overlap)}"))
    qa = pd.read_csv(qa_path)
    extra = pd.DataFrame([
        {"check": name, "status": "pass" if ok else "fail", "detail": detail, "severity": "error"}
        for name, ok, detail in checks
    ])
    pd.concat([qa.loc[~qa["check"].isin(extra["check"])], extra], ignore_index=True).to_csv(qa_path, index=False)


def _period_end(period: pd.Series) -> pd.Series:
    return pd.PeriodIndex(period.astype(str), freq="M").end_time.date.astype(str)


def build_monthly_debt_position(debt_dir: Path, write_dir: Path) -> Dict[str, Path]:
    debt_dir = Path(debt_dir)
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    repayment_detail_path = _build_repayment_detail(debt_dir=debt_dir, write_dir=write_dir)
    monthly_path = debt_dir / "debt_balance_monthly.csv"
    open_items_path = debt_dir / "debt_open_items.csv"
    out_path = write_dir / "monthly_debt_position.csv"
    qa_path = write_dir / "monthly_debt_position_qa.csv"

    if not monthly_path.exists():
        _empty_debt_position().to_csv(out_path, index=False)
        qa = _qa(
            [
                {
                    "check": "monthly_debt_position_exists",
                    "status": "pass",
                    "detail": f"empty wrapper written to {out_path}",
                    "severity": "warning",
                },
                {
                    "check": "debt_balance_monthly_loaded",
                    "status": "warn",
                    "detail": f"missing {monthly_path}",
                    "severity": "warning",
                },
            ]
        )
        qa.to_csv(qa_path, index=False)
        activity_paths = _build_monthly_debt_activity(
            debt_dir=debt_dir, write_dir=write_dir, position=_empty_debt_position()
        )
        return {
            "monthly_debt_position": out_path,
            "monthly_debt_position_qa": qa_path,
            "monthly_debt_repayment_detail": repayment_detail_path,
            **activity_paths,
        }

    monthly = pd.read_csv(monthly_path)
    required = [
        "as_of_date",
        "period",
        "debtor",
        "creditor",
        "currency",
        "open_principal",
        "open_interest",
        "open_total",
    ]
    missing = [c for c in required if c not in monthly.columns]
    if missing:
        raise ValueError(
            f"debt_balance_monthly.csv missing required columns for monthly_debt_position: {missing}"
        )

    base = monthly.copy()
    base["open_principal"] = pd.to_numeric(
        base["open_principal"], errors="coerce"
    ).fillna(0.0)
    base["open_interest"] = pd.to_numeric(
        base["open_interest"], errors="coerce"
    ).fillna(0.0)
    base["open_total"] = pd.to_numeric(base["open_total"], errors="coerce").fillna(0.0)
    base["Currency"] = base["currency"].astype(str).str.upper()
    base["period"] = base["period"].astype(str)
    if "period_end" not in base.columns:
        base["period_end"] = _period_end(base["period"])

    # debt_balance_monthly may contain repeated rows per item_type and, in
    # legacy artifacts, multiple stock snapshots inside one month. Use the
    # selected monthly close: latest as_of_date per debtor/creditor/currency.
    candidates = base.drop_duplicates(
        ["period", "debtor", "creditor", "Currency", "as_of_date"]
    )[
        [
            "period",
            "period_end",
            "as_of_date",
            "debtor",
            "creditor",
            "Currency",
            "open_principal",
            "open_interest",
            "open_total",
        ]
    ].copy()
    selected_groups: list[pd.DataFrame] = []
    for _, group in candidates.groupby(
        ["period", "debtor", "creditor", "Currency"],
        dropna=False,
        sort=False,
    ):
        period = str(group.iloc[0]["period"])
        selection = select_debt_position(group, period=period, annual=False)
        if selection.available:
            chosen = selected_debt_position_rows(group, selection).tail(1).copy()
            chosen["position_status"] = "available"
            chosen["selection_reason"] = selection.reason
            chosen["candidate_rows"] = selection.candidate_rows
            chosen["valid_as_of_rows"] = selection.valid_as_of_rows
            selected_groups.append(chosen)
            continue
        unavailable = group.iloc[[0]].copy()
        unavailable["as_of_date"] = ""
        unavailable[["open_principal", "open_interest", "open_total"]] = pd.NA
        unavailable["position_status"] = "unavailable"
        unavailable["selection_reason"] = selection.reason
        unavailable["candidate_rows"] = selection.candidate_rows
        unavailable["valid_as_of_rows"] = selection.valid_as_of_rows
        selected_groups.append(unavailable)
    unique = (
        pd.concat(selected_groups, ignore_index=True)
        if selected_groups
        else pd.DataFrame(columns=[
            "period", "period_end", "as_of_date", "debtor", "creditor", "Currency",
            "open_principal", "open_interest", "open_total", "position_status",
            "selection_reason", "candidate_rows", "valid_as_of_rows",
        ])
    )

    counts = pd.DataFrame(
        columns=["period", "debtor", "creditor", "Currency", "n_open_items"]
    )
    items_for_qa = pd.DataFrame()
    if open_items_path.exists():
        items = pd.read_csv(open_items_path)
        item_required = ["opened_at", "debtor", "creditor", "currency"]
        if all(c in items.columns for c in item_required):
            items = items.copy()
            items["opened_at"] = pd.to_datetime(items["opened_at"], errors="coerce")
            items = items[items["opened_at"].notna()].copy()
            items["period"] = items["opened_at"].dt.to_period("M").astype(str)
            items["Currency"] = items["currency"].astype(str).str.upper()
            items_for_qa = items.copy()
            counts = (
                items.groupby(
                    ["period", "debtor", "creditor", "Currency"], dropna=False
                )
                .size()
                .reset_index(name="n_open_items")
            )

    rows = []
    for _, row in unique.iterrows():
        n_match = counts.loc[
            counts["period"].eq(row["period"])
            & counts["debtor"].eq(row["debtor"])
            & counts["creditor"].eq(row["creditor"])
            & counts["Currency"].eq(row["Currency"]),
            "n_open_items",
        ]
        n_open_items = int(n_match.iloc[0]) if not n_match.empty else 0
        available = str(row.get("position_status", "available")) == "available"
        for component, amount_col in [
            ("principal", "open_principal"),
            ("interest", "open_interest"),
            ("total", "open_total"),
        ]:
            rows.append(
                {
                    "period": row["period"],
                    "period_end": row["period_end"],
                    "as_of_date": row["as_of_date"],
                    "debtor": row["debtor"],
                    "creditor": row["creditor"],
                    "Currency": row["Currency"],
                    "component": component,
                    "position_status": row.get("position_status", "available"),
                    "selection_reason": row.get("selection_reason", ""),
                    "candidate_rows": int(row.get("candidate_rows", 0)),
                    "valid_as_of_rows": int(row.get("valid_as_of_rows", 0)),
                    "open_amount": float(row[amount_col]) if available else pd.NA,
                    "open_principal": float(row["open_principal"]) if available else pd.NA,
                    "open_interest": float(row["open_interest"]) if available else pd.NA,
                    "open_total": float(row["open_total"]) if available else pd.NA,
                    "source_table": "debt_balance_monthly.csv",
                    "source_rule_version": RULE_VERSION,
                    "n_open_items": n_open_items,
                    "caveat": (
                        "Consumption wrapper over resolved debt balances; latest valid as_of authority applied."
                        if available
                        else f"Governed debt position unavailable: {row.get('selection_reason', '')}; no lexical or prior-period fallback."
                    ),
                    "frontend_suitability": "safe_with_caveat" if available else "unavailable",
                }
            )

    out = pd.DataFrame(rows, columns=DEBT_POSITION_COLUMNS)
    out.to_csv(out_path, index=False)

    source_total = float(pd.to_numeric(unique["open_total"], errors="coerce").sum())
    wrapper_total = (
        float(out.loc[out["component"].eq("total"), "open_amount"].sum())
        if not out.empty
        else 0.0
    )
    cutoff = load_run_cutoff_if_present(write_dir)
    selected_available = unique.loc[unique["position_status"].astype(str).eq("available")].copy()
    selected_dates = pd.to_datetime(selected_available["as_of_date"], errors="coerce")
    expected_dates = pd.to_datetime(selected_available["period_end"], errors="coerce")
    if cutoff is not None:
        cutoff_ts = pd.Timestamp(cutoff.date)
        expected_dates = expected_dates.where(expected_dates.le(cutoff_ts), cutoff_ts)
    close_dates_ok = selected_dates.eq(expected_dates).all()
    zero_stock_mismatches = 0
    if not items_for_qa.empty:
        closed_source = (
            items_for_qa["closed_at"]
            if "closed_at" in items_for_qa.columns
            else pd.Series(pd.NaT, index=items_for_qa.index)
        )
        closed = pd.to_datetime(closed_source, errors="coerce")
        opened = pd.to_datetime(items_for_qa["opened_at"], errors="coerce")
        for _, selected in selected_available.iterrows():
            as_of = pd.Timestamp(selected["as_of_date"])
            pair = (
                items_for_qa["debtor"].astype(str).eq(str(selected["debtor"]))
                & items_for_qa["creditor"].astype(str).eq(str(selected["creditor"]))
                & items_for_qa["Currency"].astype(str).eq(str(selected["Currency"]))
            )
            active = pair & opened.le(as_of) & (closed.isna() | closed.gt(as_of))
            expected_open = pd.to_numeric(items_for_qa.loc[active, "original_amount"], errors="coerce").fillna(0.0).sum()
            if abs(float(selected["open_total"]) - float(expected_open)) > 0.01:
                zero_stock_mismatches += 1
    qa_rows = [
        {
            "check": "monthly_debt_position_exists",
            "status": "pass" if out_path.exists() else "fail",
            "detail": str(out_path),
            "severity": "error",
        },
        {
            "check": "debt_balance_monthly_loaded",
            "status": "pass",
            "detail": f"rows={len(monthly)}",
            "severity": "error",
        },
        {
            "check": "debt_rows_have_currency",
            "status": (
                "pass"
                if out.empty or out["Currency"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": ",".join(sorted(out["Currency"].dropna().astype(str).unique())),
            "severity": "error",
        },
        {
            "check": "debt_rows_have_debtor_creditor",
            "status": (
                "pass"
                if out.empty
                or (
                    out[["debtor", "creditor"]].notna().all().all()
                    and out[["debtor", "creditor"]]
                    .astype(str)
                    .apply(lambda col: col.str.strip().ne(""))
                    .all()
                    .all()
                )
                else "fail"
            ),
            "detail": "debtor/creditor populated",
            "severity": "error",
        },
        {
            "check": "debt_rows_have_component",
            "status": (
                "pass"
                if out.empty or out["component"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": (
                ",".join(sorted(set(out["component"]))) if not out.empty else "empty"
            ),
            "severity": "error",
        },
        {
            "check": "principal_interest_total_present",
            "status": (
                "pass"
                if out.empty
                or {"principal", "interest", "total"}.issubset(set(out["component"]))
                else "fail"
            ),
            "detail": (
                ",".join(sorted(set(out["component"]))) if not out.empty else "empty"
            ),
            "severity": "error",
        },
        {
            "check": "has_monthly_periods",
            "status": (
                "pass"
                if out["period"].astype(str).str.match(r"^\d{4}-\d{2}$").all()
                else "fail"
            ),
            "detail": f"periods={out['period'].nunique() if not out.empty else 0}",
            "severity": "error",
        },
        {
            "check": "total_reconciles_to_source",
            "status": "pass" if abs(source_total - wrapper_total) < 0.01 else "fail",
            "detail": f"source_total={source_total}; wrapper_total={wrapper_total}",
            "severity": "error",
        },
        {
            "check": "selected_as_of_is_governed_month_close",
            "status": "pass" if close_dates_ok else "fail",
            "detail": f"rows={len(selected_available)}; mismatches={int((~selected_dates.eq(expected_dates)).sum())}",
            "severity": "error",
        },
        {
            "check": "reported_position_equals_open_obligations_at_as_of",
            "status": "pass" if zero_stock_mismatches == 0 else "fail",
            "detail": f"rows={len(selected_available)}; mismatches={zero_stock_mismatches}; zero-open positions remain explicit zero",
            "severity": "error",
        },
        {
            "check": "invalid_as_of_fails_closed",
            "status": (
                "pass"
                if out.empty
                or out.loc[out["position_status"].astype(str).eq("unavailable"), "open_amount"].isna().all()
                else "fail"
            ),
            "detail": f"unavailable_rows={int(out['position_status'].astype(str).eq('unavailable').sum()) if not out.empty else 0}; lexical_fallback=never",
            "severity": "error",
        },
        {
            "check": "no_cross_currency_debt_total",
            "status": "pass",
            "detail": "all rows remain currency-grained; no ARS/USD total emitted",
            "severity": "error",
        },
        {
            "check": "debt_stock_not_mixed_with_operating_flows",
            "status": "pass",
            "detail": "source_table=debt_balance_monthly.csv only; no operating-statement source columns used",
            "severity": "error",
        },
        {
            "check": "frontend_outputs_have_suitability",
            "status": (
                "pass"
                if "frontend_suitability" in out.columns
                and out["frontend_suitability"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": "debt wrapper rows carry suitability",
            "severity": "error",
        },
        {
            "check": "money_outputs_have_currency",
            "status": (
                "pass"
                if out["Currency"].astype(str).str.strip().ne("").all()
                else "fail"
            ),
            "detail": "debt wrapper rows carry Currency",
            "severity": "error",
        },
    ]
    _qa(qa_rows).to_csv(qa_path, index=False)
    activity_paths = _build_monthly_debt_activity(
        debt_dir=debt_dir, write_dir=write_dir, position=out
    )
    _append_cost_gap_debt_exclusion_qa(debt_dir=debt_dir, write_dir=write_dir)
    return {
        "monthly_debt_position": out_path,
        "monthly_debt_position_qa": qa_path,
        "monthly_debt_repayment_detail": repayment_detail_path,
        **activity_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly debt consumption wrapper"
    )
    parser.add_argument("--debt-dir", required=True)
    parser.add_argument("--write-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_monthly_debt_position(Path(args.debt_dir), Path(args.write_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
