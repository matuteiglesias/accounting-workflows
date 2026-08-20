from __future__ import annotations

import pandas as pd

from accounting.marts.semantic import build_semantic_outputs
from accounting.marts.treasury import build_monthly_cash_accountability
from accounting.stage_d.materialize import (
    materialize_box_balance_time_long,
    materialize_box_flow_balance_time_long,
)


def _ledger() -> pd.DataFrame:
    base = {
        "Box": "Property Management",
        "Lugar": "CABA",
        "status": "pagado",
    }
    return pd.DataFrame(
        [
            dict(base, tx_id="rent_ars", Date="2026-05-01", amount=1000.0, Currency="ARS", payer="Tenant", receiver="PM", Flujo="Cobros", Tipo="Renta", Detalle="rent"),
            dict(base, tx_id="tax_ars", Date="2026-05-02", amount=200.0, Currency="ARS", payer="PM", receiver="ABL", Flujo="Pagos", Tipo="Impuestos", Detalle="tax"),
            dict(base, tx_id="direct_tax", Date="2026-05-03", amount=50.0, Currency="ARS", payer="Inquilino", receiver="ABL", Flujo="Pagos", Tipo="Impuestos", Detalle="Inquilino paga impuesto directo"),
            dict(base, tx_id="rent_usd", Date="2026-05-04", amount=500.0, Currency="USD", payer="Tenant", receiver="PM", Flujo="Cobros", Tipo="Renta", Detalle="rent usd"),
            dict(base, tx_id="repay_usd", Date="2026-05-05", amount=100.0, Currency="USD", payer="PM", receiver="MI", Flujo="Pagos", Tipo="Repago", Detalle="debt repayment"),
        ]
    )


def _build_base(tmp_path):
    ledger = _ledger()
    materialize_box_balance_time_long(ledger, tmp_path, freq="M")
    materialize_box_flow_balance_time_long(ledger, tmp_path, freq="M")
    build_semantic_outputs(ledger, tmp_path, freq="M")


def test_accountability_is_native_currency_and_reconciles(tmp_path):
    _build_base(tmp_path)
    pd.DataFrame(
        [
            {
                "period": "2026-05",
                "period_end": "2026-05-31",
                "as_of_date": "2026-05-31",
                "Box": "Property Management",
                "party": "",
                "account_id": "bank-ars",
                "account_name": "ARS",
                "Currency": "ARS",
                "close_amount": 1800.0,
                "source_table": "validated_cash_close.csv",
                "source_date": "2026-05-31",
                "source_type": "account_snapshot",
                "source_reference": "fixture",
                "validation_status": "validated",
                "validated_by": "controller",
                "position_type": "cash_close",
                "cash_suitability": "frontend_safe",
                "is_frontend_safe": True,
                "caveat": "fixture",
                "notes": "",
                "n_source_rows": 1,
                "calculation_rule": "fixture",
            }
        ]
    ).to_csv(tmp_path / "monthly_cash_close.csv", index=False)
    pd.DataFrame(
        [
            {
                "period": "2026-05",
                "period_end": "2026-05-31",
                "Currency": "USD",
                "debtor": "Property Management",
                "creditor": "MI",
                "activity_type": "repayment",
                "new_principal": 0.0,
                "interest_accrued": 0.0,
                "repayments": 100.0,
                "adjustments": 0.0,
                "opening_total": 500.0,
                "closing_total": 400.0,
                "net_change": 0.0,
                "n_items": 1,
                "source_table": "debt_repayment_events.csv",
                "source_rule_version": "fixture",
                "frontend_suitability": "safe_with_caveat",
                "reconciliation_status": "reconciled",
                "caveat": "fixture",
                "notes": "",
            }
        ]
    ).to_csv(tmp_path / "monthly_debt_activity.csv", index=False)

    paths = build_monthly_cash_accountability(tmp_path)
    out = pd.read_csv(paths["monthly_cash_accountability"])
    assert set(out["Currency"]) == {"ARS", "USD"}

    ars = out.loc[out["Currency"].eq("ARS")].iloc[0]
    assert ars["rent_in"] == 1000.0
    assert ars["taxes_out"] == 200.0
    assert ars["direct_tax_support_non_cash"] == 50.0
    assert ars["net_cash_flow"] == 800.0
    assert ars["box_motor_net"] == 800.0
    assert ars["validated_cash_close"] == 1800.0
    assert ars["validated_anchor_offset"] == 1000.0
    assert ars["anchor_alignment_status"] == "first_anchor"

    usd = out.loc[out["Currency"].eq("USD")].iloc[0]
    assert usd["rent_in"] == 500.0
    assert usd["debt_repayments_out"] == 100.0
    assert usd["net_cash_flow"] == 400.0
    assert usd["debt_engine_repayments"] == 100.0
    assert usd["debt_reconciliation_status"] == "reconciled"


def test_two_validated_anchors_expose_offset_gap_instead_of_hiding_it(tmp_path):
    motor_rows = []
    for month, net, cum in [("2026-01", 100.0, 100.0), ("2026-02", 50.0, 150.0)]:
        motor_rows.append(
            {
                "TimePeriod": month,
                "TimePeriod_end": pd.Period(month, freq="M").end_time.date().isoformat(),
                "Box": "Property Management",
                "Currency": "USD",
                "in_amt": net,
                "out_amt": 0.0,
                "net": net,
                "cum_net": cum,
            }
        )
    pd.DataFrame(motor_rows).to_csv(tmp_path / "box_balance_time_long.freq=M.csv", index=False)
    pd.DataFrame(
        [
            {
                "TimePeriod": r["TimePeriod"],
                "TimePeriod_end": r["TimePeriod_end"],
                "Box": r["Box"],
                "Currency": r["Currency"],
                "Flujo": "Cobros",
                "Tipo": "Renta",
                "in_amt": r["in_amt"],
                "out_amt": r["out_amt"],
                "net": r["net"],
                "n_tx": 1,
            }
            for r in motor_rows
        ]
    ).to_csv(tmp_path / "box_flow_balance_time_long.freq=M.csv", index=False)
    pd.DataFrame(
        [
            {
                "period": r["TimePeriod"],
                "period_end": r["TimePeriod_end"],
                "Box": r["Box"],
                "Currency": r["Currency"],
                "movement_basis": "actual_cash",
                "cash_direction": "in",
                "cash_category": "rent",
                "semantic_bucket": "operating_revenue",
                "semantic_subbucket": "rent",
                "funding_actor": "",
                "funding_channel": "",
                "cash_effect": "cash_in_box",
                "debt_effect": "none",
                "direction_source": "box_party_match",
                "classification_status": "classified",
                "classification_confidence": "high",
                "review_required": False,
                "amount_in": r["net"],
                "amount_out": 0.0,
                "net_amount": r["net"],
                "non_cash_amount": 0.0,
                "gross_amount": r["net"],
                "n_tx": 1,
                "n_review_required": 0,
                "source_tx_ids_sample": r["TimePeriod"],
                "rule_ids": "R001",
                "source_table": "ledger_canonical.csv",
                "notes": "",
            }
            for r in motor_rows
        ]
    ).to_csv(tmp_path / "monthly_box_treasury_flow.csv", index=False)
    cash_rows = []
    for period, close in [("2026-01", 1100.0), ("2026-02", 1200.0)]:
        end = pd.Period(period, freq="M").end_time.date().isoformat()
        cash_rows.append(
            {
                "period": period,
                "period_end": end,
                "as_of_date": end,
                "Box": "Property Management",
                "party": "",
                "account_id": "usd",
                "account_name": "USD",
                "Currency": "USD",
                "close_amount": close,
                "source_table": "validated_cash_close.csv",
                "source_date": end,
                "source_type": "account_snapshot",
                "source_reference": "fixture",
                "validation_status": "validated",
                "validated_by": "controller",
                "position_type": "cash_close",
                "cash_suitability": "frontend_safe",
                "is_frontend_safe": True,
                "caveat": "fixture",
                "notes": "",
                "n_source_rows": 1,
                "calculation_rule": "fixture",
            }
        )
    pd.DataFrame(cash_rows).to_csv(tmp_path / "monthly_cash_close.csv", index=False)

    paths = build_monthly_cash_accountability(tmp_path)
    out = pd.read_csv(paths["monthly_cash_accountability"])
    jan, feb = out.sort_values("period").to_dict("records")
    assert jan["validated_anchor_offset"] == 1000.0
    assert feb["validated_anchor_offset"] == 1050.0
    assert feb["anchor_reconciliation_gap"] == 50.0
    assert feb["anchor_alignment_status"] == "residual"
