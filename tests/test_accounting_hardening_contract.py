from pathlib import Path

import pandas as pd

from accounting.debt.resolve import build_open_items
from accounting.grooming import (
    build_debt_status_grooming,
    build_household_mirror_review,
    build_legacy_inferred_net_grooming,
    build_pm_cost_allocation_gaps,
)
from accounting.ledger.ingest import filter_ledger_statuses
from accounting.marts.accountability import (
    build_family_business_accountability_cycles,
    build_household_monthly_control,
    cycle_bounds,
)
from accounting.marts.semantic import build_cost_allocation_gap_outputs, build_semantic_outputs
from accounting.stage_d.materialize import _analytical_ledger


def _ledger_row(**overrides):
    row = {
        "tx_id": "tx", "Date": "2026-05-10", "amount": 100.0,
        "Currency": "ARS", "payer": "FB", "receiver": "Gasto",
        "Flujo": "Transfer", "Tipo": "Gasto", "status": "pagado",
        "Box": "Family Business", "Detalle": "Gastos Personales", "tag": "",
        "source_file": "synthetic", "source_row": 1,
    }
    row.update(overrides)
    return row


def test_status_x_is_provenance_only_and_never_analytical() -> None:
    frame = pd.DataFrame([_ledger_row(tx_id="valid"), _ledger_row(tx_id="invalid", status="X")])
    assert list(filter_ledger_statuses(frame, None)["tx_id"]) == ["valid"]
    assert list(filter_ledger_statuses(frame, ["pagado", "X"])["tx_id"]) == ["valid"]


def test_costos_pm_is_gap_not_debt_or_cash(tmp_path: Path) -> None:
    row = _ledger_row(
        tx_id="gap", payer="Costos", receiver="PM", Tipo="Prestamo",
        Flujo="Transfer", Box="Property Management", Detalle="Costo pendiente",
    )
    frame = pd.DataFrame([row])
    paths = build_semantic_outputs(frame, tmp_path)
    audit = pd.read_csv(paths["classification_audit"])
    assert audit.iloc[0]["semantic_bucket"] == "cost_allocation_gap"
    treasury = pd.read_csv(paths["monthly_box_treasury_flow"])
    assert treasury.iloc[0]["movement_basis"] == "economic_only"
    assert treasury.iloc[0]["amount_in"] == 0
    debt_input = frame.assign(Lugar="", Issuer="")
    assert build_open_items(debt_input) == []


def test_governed_cost_gap_projection_includes_open_but_excludes_x(tmp_path: Path) -> None:
    frame = pd.DataFrame([
        _ledger_row(tx_id="gap", payer="Costos", receiver="PM", Tipo="Prestamo", status="abierto", Box="Property Management", Currency="usd", Lugar="CABA", Detalle="Costo pendiente"),
        _ledger_row(tx_id="gap-x", payer="Costos", receiver="PM", Tipo="Prestamo", status="X", Box="Property Management", Currency="USD"),
    ])
    paths = build_cost_allocation_gap_outputs(frame, tmp_path)
    gaps = pd.read_csv(paths["cost_allocation_gaps"])
    assert list(gaps["source_tx_id"]) == ["gap"]
    assert gaps.iloc[0]["Currency"] == "USD"
    assert gaps.iloc[0]["accounting_nature"] == "unresolved_cost_allocation"
    assert gaps.iloc[0]["debt_effect"] == "none"
    assert gaps.iloc[0]["economic_scope"] == "Property Management"
    assert "legal_debtor" not in gaps.columns
    assert pd.read_csv(paths["cost_allocation_gaps_qa"])["status"].eq("pass").all()


def test_legacy_inferred_net_is_audit_only(tmp_path: Path) -> None:
    frame = pd.DataFrame([
        _ledger_row(tx_id="legacy", amount=80, tag="legacy_inferred_net"),
        _ledger_row(tx_id="cash", amount=20, tag="affirmative_cash"),
    ])
    assert list(_analytical_ledger(frame)["tx_id"]) == ["cash"]
    paths = build_semantic_outputs(frame, tmp_path)
    legacy = pd.read_csv(paths["legacy_inferred_net_audit"])
    split = pd.read_csv(paths["monthly_flow_semantic_split"])
    assert list(legacy["tx_id"]) == ["legacy"]
    assert split["amount_out"].sum() == 20


def _accountability_rows(box: str = "Family Business") -> pd.DataFrame:
    rows = []
    for period, cash_in, draws, uses in [
        ("2026-03", 100, 10, 20), ("2026-08", 50, 5, 15), ("2026-09", 30, 0, 10),
    ]:
        rows.append({
            "period": period, "period_end": str(pd.Period(period, "M").end_time.date()),
            "control_as_of_date": str(pd.Period(period, "M").end_time.date()),
            "Box": box, "Currency": "ARS", "opening_control": 0,
            "total_cash_in": cash_in, "total_cash_out": draws + uses,
            "personal_draws_out": draws, "dividends_out": 0,
            "internal_transfer_out": 0, "fx_out": 0, "n_tx": 2,
            "validated_cash_status": "unavailable", "validated_cash_reason": "missing_source",
            "validated_cash_close": pd.NA, "validated_as_of_date": "",
            "funding_cash_in": cash_in, "net_cash_flow": cash_in - draws - uses,
        })
    return pd.DataFrame(rows)


def test_cycles_are_fixed_mar_aug_and_sep_feb_and_cash_gap_is_unavailable() -> None:
    assert tuple(str(value.date()) for value in cycle_bounds("2026-08-31")) == ("2026-03-01", "2026-08-31")
    assert tuple(str(value.date()) for value in cycle_bounds("2026-09-01")) == ("2026-09-01", "2027-02-28")
    out = build_family_business_accountability_cycles(_accountability_rows(), as_of_date="2026-09-30")
    closed = out.loc[out["cycle_start"].eq("2026-03-01")].iloc[0]
    assert closed["view_type"] == "completed_cycle"
    assert closed["closing_accountability_balance"] == 100
    assert closed["accountability_gap_status"] == "unavailable_no_validated_cash"
    assert pd.isna(closed["accountability_gap"])
    assert set(out["view_type"]) == {"completed_cycle", "current_since_last_cut"}


def test_household_has_own_contribution_use_and_surplus_deficit() -> None:
    out = build_household_monthly_control(_accountability_rows("Household"))
    assert out.iloc[0]["effective_funding_contributions"] == 100
    assert out.iloc[0]["domestic_uses"] == 30
    assert out.iloc[0]["position_label"] == "surplus"


def test_private_grooming_is_review_only_and_never_edits_source() -> None:
    ledger = pd.DataFrame([
        _ledger_row(tx_id="old", Date="2025-01-10"),
        _ledger_row(tx_id="tagged", tag="FB_OWNER_DRAW_GASTOS_PERSONALES"),
        _ledger_row(tx_id="x", status="X"),
        _ledger_row(tx_id="hh", Date="2026-05-11", payer="Alejandro", receiver="HH", amount=200000, Box="Household", Currency="ARS"),
    ])
    legacy = build_legacy_inferred_net_grooming(ledger)
    assert list(legacy["tx_id"]) == ["old"]
    household = build_household_mirror_review(ledger)
    assert household.iloc[0]["review_status"] == "HUMAN_REVIEW_REQUIRED"
    gaps = build_pm_cost_allocation_gaps(pd.DataFrame([
        _ledger_row(payer="Costos", receiver="PM", Tipo="Prestamo", status="abierto", Box="Property Management")
    ]))
    assert gaps.iloc[0]["accounting_treatment"] == "cost_allocation_gap"
    assert not bool(gaps.iloc[0]["ordinary_repayment_eligible"])
    debt = pd.DataFrame([
        {"debt_id": "a", "ledger_status": "abierto", "engine_status": "closed", "open_amount": 0, "closed_at": "2026-01-01"},
        {"debt_id": "b", "ledger_status": "cerrado", "engine_status": "open", "open_amount": 10, "closed_at": ""},
    ])
    recommendations = build_debt_status_grooming(debt).set_index("debt_id")["recommendation"]
    assert recommendations["a"] == "RECOMMEND_CERRADO"
    assert recommendations["b"] == "REVIEW"
