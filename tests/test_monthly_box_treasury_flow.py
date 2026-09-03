from __future__ import annotations

import pandas as pd

from accounting.marts.semantic import build_semantic_outputs
from accounting.stage_d.materialize import (
    materialize_box_balance_time_long,
    materialize_box_flow_balance_time_long,
)


def _ledger() -> pd.DataFrame:
    base = {
        "Currency": "ARS",
        "Box": "Property Management",
        "Lugar": "CABA",
        "status": "pagado",
    }
    rows = [
        dict(base, tx_id="rent", Date="2026-05-01", amount=1000.0, payer="Tenant", receiver="PM", Flujo="Cobros", Tipo="Renta", Detalle="rent"),
        dict(base, tx_id="tax", Date="2026-05-02", amount=200.0, payer="PM", receiver="ABL", Flujo="Pagos", Tipo="Impuestos", Detalle="tax"),
        dict(base, tx_id="direct_tax", Date="2026-05-03", amount=50.0, payer="Inquilino", receiver="ABL", Flujo="Pagos", Tipo="Impuestos", Detalle="Inquilino paga impuesto directo"),
        dict(base, tx_id="dividend", Date="2026-05-04", amount=100.0, payer="PM", receiver="Family", Flujo="Pagos", Tipo="Dividendo", Detalle="dividend"),
        dict(base, tx_id="unknown", Date="2026-05-05", amount=30.0, payer="PM", receiver="Vendor", Flujo="Otro", Tipo="Misterio", Detalle="unknown cash"),
        dict(base, tx_id="internal", Date="2026-05-06", amount=25.0, payer="PM", receiver="PM", Flujo="Transfer", Tipo="Transfer", Detalle="self transfer"),
    ]
    return pd.DataFrame(rows)


def test_treasury_flow_requires_physical_box_cash_and_reconciles(tmp_path):
    ledger = _ledger()
    materialize_box_balance_time_long(ledger, tmp_path, freq="M")
    materialize_box_flow_balance_time_long(ledger, tmp_path, freq="M")
    paths = build_semantic_outputs(ledger, tmp_path, freq="M")

    treasury = pd.read_csv(paths["monthly_box_treasury_flow"])
    actual = treasury.loc[treasury["movement_basis"].eq("actual_cash")]
    assert round(float(actual["net_amount"].sum()), 2) == 670.0

    direct = treasury.loc[
        treasury["movement_basis"].eq("non_cash_support")
        & treasury["cash_category"].eq("taxes")
    ]
    assert round(float(direct["non_cash_amount"].sum()), 2) == 50.0
    assert float(direct["amount_out"].sum()) == 0.0

    unknown = treasury.loc[
        treasury["movement_basis"].eq("actual_cash")
        & treasury["cash_category"].eq("unknown")
    ]
    assert round(float(unknown["amount_out"].sum()), 2) == 30.0

    internal = treasury.loc[treasury["movement_basis"].eq("internal_box_transfer")]
    assert round(float(internal["net_amount"].sum()), 2) == 0.0

    qa = pd.read_csv(paths["monthly_box_treasury_flow_qa"])
    hard = qa.loc[qa["severity"].eq("error")]
    assert hard["status"].eq("pass").all()


def test_semantic_fallback_cannot_manufacture_box_cash(tmp_path):
    ledger = _ledger().loc[lambda d: d["tx_id"].eq("direct_tax")].copy()
    materialize_box_balance_time_long(ledger, tmp_path, freq="M")
    materialize_box_flow_balance_time_long(ledger, tmp_path, freq="M")
    paths = build_semantic_outputs(ledger, tmp_path, freq="M")
    treasury = pd.read_csv(paths["monthly_box_treasury_flow"])
    assert float(treasury["amount_in"].sum()) == 0.0
    assert float(treasury["amount_out"].sum()) == 0.0
    assert round(float(treasury["non_cash_amount"].sum()), 2) == 50.0


def test_residual_cash_is_transaction_drillable_and_materiality_is_warn_only(tmp_path):
    ledger = pd.DataFrame([{
        "tx_id": "residual-in", "Date": "2026-05-03", "amount": 150000.0,
        "Currency": "ARS", "Box": "Property Management", "Lugar": "CABA",
        "status": "pagado", "payer": "MI", "receiver": "PM",
        "Flujo": "Contribucion", "Tipo": "Impuestos", "Detalle": "source evidence",
    }])
    materialize_box_balance_time_long(ledger, tmp_path, freq="M")
    materialize_box_flow_balance_time_long(ledger, tmp_path, freq="M")
    paths = build_semantic_outputs(ledger, tmp_path, freq="M")

    audit = pd.read_csv(paths["treasury_residual_cash_audit"])
    assert list(audit["tx_id"]) == ["residual-in"]
    assert audit.iloc[0]["movement_basis"] == "actual_cash"
    assert audit.iloc[0]["direction_source"] == "box_party_match"

    qa = pd.read_csv(paths["treasury_residual_cash_materiality_qa"])
    row = qa.iloc[0]
    assert row["other_cash_in"] == 150000.0
    assert row["total_cash_in"] == 150000.0
    assert row["other_cash_in_share"] == 1.0
    assert row["status"] == "warn"
    assert row["severity"] == "warning"
