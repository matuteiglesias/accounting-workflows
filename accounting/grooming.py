from __future__ import annotations

"""Private, source-preserving review queues.  This module never edits a ledger."""

import argparse
from pathlib import Path

import pandas as pd

from accounting.box_cash import infer_box_party
from accounting.marts.accountability import cycle_bounds


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()


def active_analysis_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    status = _text(ledger, "status").str.casefold()
    return ledger.loc[status.eq("pagado") & ~status.eq("x")].copy()


def build_legacy_inferred_net_grooming(ledger: pd.DataFrame) -> pd.DataFrame:
    work = active_analysis_rows(ledger)
    tag = _text(work, "tag")
    detail = (_text(work, "Detalle") + " " + _text(work, "notes")).str.casefold()
    family = _text(work, "Box").eq("Family Business")
    residual_shape = detail.str.contains(r"gastos\s+personales|family_withdrawal_candidate", regex=True)
    lacks_affirmative_tag = tag.eq("")
    candidates = work.loc[family & residual_shape & lacks_affirmative_tag].copy()
    keep = [c for c in ["tx_id", "Date", "Currency", "amount", "Box", "payer", "receiver", "Flujo", "Tipo", "Detalle", "status", "tag", "source_file", "source_row"] if c in candidates]
    out = candidates[keep].copy()
    out["recommended_tag"] = "legacy_inferred_net"
    out["recommendation"] = "REVIEW"
    out["review_question"] = "¿Este importe tiene evidencia afirmativa de retiro/destino, o es un neto/residual derivado?"
    out["if_confirmed_effect"] = "audit_only; no cash; no governed withdrawal; no OPEX"
    return out


def build_household_mirror_review(ledger: pd.DataFrame) -> pd.DataFrame:
    work = active_analysis_rows(ledger)
    work["Date"] = pd.to_datetime(work.get("Date"), errors="coerce")
    work["period"] = work["Date"].dt.to_period("M").astype(str)
    amount = pd.to_numeric(work.get("amount"), errors="coerce")
    actor = _text(work, "payer").str.casefold().eq("alejandro")
    hh = _text(work, "receiver").str.casefold().isin({"hh", "household"})
    months = work["period"].isin({"2026-05", "2026-06", "2026-07"})
    ars_200k = _text(work, "Currency").str.upper().eq("ARS") & amount.sub(200_000).abs().le(0.01)
    candidates = work.loc[actor & hh & months & ars_200k].copy()
    rows = []
    for period, group in candidates.groupby("period", sort=True):
        boxes = sorted(set(_text(group, "Box")))
        rows.append({
            "period": period, "Currency": "ARS", "nominal_amount": 200000.0,
            "candidate_row_count": len(group), "recording_boxes": ";".join(boxes),
            "source_tx_ids": ";".join(_text(group, "tx_id")),
            "source_rows": ";".join(_text(group, "source_row")),
            "review_status": "HUMAN_REVIEW_REQUIRED",
            "review_question": "¿Hubo una sola transferencia física Alejandro→HH o movimientos físicos separados? Describir cuenta/caja de origen y destino.",
            "cash_rule_if_one_transfer": "count canonical cash once; preserve PM-distribution economic linkage separately",
            "debt_rule": "agreed funding contribution; do not create Alejandro→HH debt absent stronger evidence",
        })
    return pd.DataFrame(rows, columns=[
        "period", "Currency", "nominal_amount", "candidate_row_count", "recording_boxes",
        "source_tx_ids", "source_rows", "review_status", "review_question",
        "cash_rule_if_one_transfer", "debt_rule",
    ])


def build_pm_cost_allocation_gaps(ledger: pd.DataFrame) -> pd.DataFrame:
    status = _text(ledger, "status").str.casefold()
    mask = (
        ~status.eq("x")
        & _text(ledger, "Tipo").str.casefold().eq("prestamo")
        & _text(ledger, "payer").str.casefold().eq("costos")
        & _text(ledger, "receiver").str.casefold().eq("pm")
        & _text(ledger, "Box").eq("Property Management")
    )
    out = ledger.loc[mask].copy()
    keep = [c for c in ["tx_id", "Date", "Currency", "amount", "Box", "payer", "receiver", "Flujo", "Tipo", "status", "Detalle", "tag", "source_file", "source_row"] if c in out]
    out = out[keep]
    out["accounting_treatment"] = "cost_allocation_gap"
    out["established_debt_effect"] = "excluded"
    out["ordinary_repayment_eligible"] = False
    out["control_note"] = "unresolved economic burden within Property Management; Box is not a legal person"
    return out


def build_status_x_impact(ledger: pd.DataFrame) -> pd.DataFrame:
    work = ledger.loc[_text(ledger, "status").str.casefold().eq("x")].copy()
    columns = [
        "Box", "Currency", "period", "accountability_cycle", "excluded_rows",
        "excluded_source_amount", "cash_before", "cash_after", "opex_before", "opex_after",
        "established_debt_before", "established_debt_after", "repayments_before", "repayments_after",
        "funding_before", "funding_after", "distributions_before", "distributions_after",
        "accountability_balance_delta", "professional_drilldown_members_before",
        "professional_drilldown_members_after",
    ]
    if work.empty:
        return pd.DataFrame(columns=columns)
    dates = pd.to_datetime(work.get("Date"), errors="coerce")
    work["period"] = dates.dt.to_period("M").astype(str)
    bounds = dates.map(lambda value: cycle_bounds(value) if pd.notna(value) else (pd.NaT, pd.NaT))
    work["accountability_cycle"] = bounds.map(
        lambda pair: f"{pair[0].date().isoformat()}_{pair[1].date().isoformat()}" if pd.notna(pair[0]) else ""
    )
    amount = pd.to_numeric(work.get("amount"), errors="coerce").fillna(0.0)
    party = _text(work, "Box").map(lambda box: infer_box_party(box) if box else "")
    incoming = _text(work, "receiver").eq(party) & party.ne("")
    outgoing = _text(work, "payer").eq(party) & party.ne("")
    actual = incoming ^ outgoing
    work["cash_before"] = amount.where(incoming & actual, 0.0) - amount.where(outgoing & actual, 0.0)
    detail = (_text(work, "Detalle") + " " + _text(work, "notes")).str.casefold()
    work["distributions_before"] = amount.where(outgoing & detail.str.contains(r"gastos\s+personales|dividendo|retiro", regex=True), 0.0)
    work["opex_before"] = amount.where(outgoing & _text(work, "Tipo").str.casefold().isin({"impuestos", "servicio", "servicios", "mantenimiento", "legal"}), 0.0)
    work["established_debt_before"] = amount.where(_text(work, "Tipo").str.casefold().isin({"prestamo", "interes"}), 0.0)
    work["repayments_before"] = amount.where(_text(work, "Tipo").str.casefold().eq("repago"), 0.0)
    work["funding_before"] = amount.where(_text(work, "Tipo").str.casefold().isin({"contribucion", "contribuciones"}), 0.0)
    grouped = work.groupby(["Box", "Currency", "period", "accountability_cycle"], dropna=False).agg(
        excluded_rows=("tx_id", "size"), excluded_source_amount=("amount", "sum"),
        cash_before=("cash_before", "sum"), opex_before=("opex_before", "sum"),
        established_debt_before=("established_debt_before", "sum"),
        repayments_before=("repayments_before", "sum"), funding_before=("funding_before", "sum"),
        distributions_before=("distributions_before", "sum"),
        professional_drilldown_members_before=("tx_id", "size"),
    ).reset_index()
    for name in ["cash_after", "opex_after", "established_debt_after", "repayments_after", "funding_after", "distributions_after", "professional_drilldown_members_after"]:
        grouped[name] = 0.0
    grouped["accountability_balance_delta"] = -grouped["cash_before"]
    return grouped[columns]


def build_debt_status_grooming(reconciliation: pd.DataFrame) -> pd.DataFrame:
    if reconciliation.empty:
        return pd.DataFrame()
    work = reconciliation.copy()
    ledger = _text(work, "ledger_status").str.casefold()
    engine = _text(work, "engine_status").str.casefold()
    mismatch = ledger.ne(engine) & ~(
        ledger.isin({"cerrado", "closed"}) & engine.eq("closed")
    ) & ~(
        ledger.isin({"abierto", "open", "pagado"}) & engine.eq("open")
    )
    out = work.loc[mismatch].copy()
    out["recommendation"] = "REVIEW"
    supported_close = (
        _text(out, "engine_status").str.casefold().eq("closed")
        & pd.to_numeric(out.get("open_amount"), errors="coerce").fillna(float("inf")).abs().le(0.01)
        & _text(out, "closed_at").ne("")
    )
    out.loc[supported_close, "recommendation"] = "RECOMMEND_CERRADO"
    ledger_closed_engine_open = (
        _text(out, "ledger_status").str.casefold().isin({"cerrado", "closed"})
        & _text(out, "engine_status").str.casefold().eq("open")
    )
    out.loc[ledger_closed_engine_open, "recommendation"] = "REVIEW"
    out["live_source_edit_performed"] = False
    return out


def write_grooming_outputs(*, ledger_path: Path, out_dir: Path, debt_reconciliation_path: Path | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ledger_path, keep_default_na=False)
    outputs = {
        "legacy_inferred_net_grooming": out_dir / "legacy_inferred_net_grooming.csv",
        "household_mirror_review": out_dir / "household_mirror_review.csv",
        "pm_cost_allocation_gaps": out_dir / "pm_cost_allocation_gaps.csv",
        "status_x_exclusion_impact": out_dir / "status_x_exclusion_impact.csv",
    }
    build_legacy_inferred_net_grooming(ledger).to_csv(outputs["legacy_inferred_net_grooming"], index=False)
    build_household_mirror_review(ledger).to_csv(outputs["household_mirror_review"], index=False)
    build_pm_cost_allocation_gaps(ledger).to_csv(outputs["pm_cost_allocation_gaps"], index=False)
    build_status_x_impact(ledger).to_csv(outputs["status_x_exclusion_impact"], index=False)
    if debt_reconciliation_path and debt_reconciliation_path.exists():
        outputs["debt_status_grooming"] = out_dir / "debt_status_grooming.csv"
        build_debt_status_grooming(pd.read_csv(debt_reconciliation_path, keep_default_na=False)).to_csv(outputs["debt_status_grooming"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--debt-reconciliation", type=Path)
    args = parser.parse_args()
    for path in write_grooming_outputs(ledger_path=args.ledger, out_dir=args.out_dir, debt_reconciliation_path=args.debt_reconciliation).values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
