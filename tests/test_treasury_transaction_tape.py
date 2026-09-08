from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from accounting.marts.semantic import build_semantic_outputs
from accounting.reports.treasury_transaction_tape.render import render_report
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
    return pd.DataFrame(
        [
            dict(
                base,
                tx_id="rent",
                Date="2026-05-01",
                amount=1000.0,
                payer="Tenant",
                receiver="PM",
                Flujo="Cobros",
                Tipo="Renta",
                Detalle="rent",
            ),
            dict(
                base,
                tx_id="tax",
                Date="2026-05-02",
                amount=200.0,
                payer="PM",
                receiver="ABL",
                Flujo="Pagos",
                Tipo="Impuestos",
                Detalle="tax",
            ),
            dict(
                base,
                tx_id="direct_tax",
                Date="2026-05-03",
                amount=50.0,
                payer="Inquilino",
                receiver="ABL",
                Flujo="Pagos",
                Tipo="Impuestos",
                Detalle="tenant direct tax",
            ),
            dict(
                base,
                tx_id="dividend",
                Date="2026-05-04",
                amount=100.0,
                payer="PM",
                receiver="Family",
                Flujo="Pagos",
                Tipo="Dividendo",
                Detalle="distribution",
            ),
        ]
    )


def test_semantic_treasury_persists_atomic_actual_cash_without_reclassifying(tmp_path: Path) -> None:
    ledger = _ledger()
    materialize_box_balance_time_long(ledger, tmp_path, freq="M")
    materialize_box_flow_balance_time_long(ledger, tmp_path, freq="M")
    paths = build_semantic_outputs(ledger, tmp_path, freq="M")

    detail = pd.read_csv(paths["box_treasury_transaction_detail"])
    assert set(detail["tx_id"]) == {"rent", "tax", "dividend"}
    assert "direct_tax" not in set(detail["tx_id"])
    assert detail["movement_basis"].eq("actual_cash").all()
    assert detail["direction_source"].eq("box_party_match").all()
    assert float(detail["amount_in"].sum()) == 1000.0
    assert float(detail["amount_out"].sum()) == 300.0
    assert float(detail["net_amount"].sum()) == 700.0

    qa = pd.read_csv(paths["box_treasury_transaction_detail_qa"])
    assert qa["status"].eq("pass").all()
    assert "atomic_detail_matches_monthly_treasury" in set(qa["check"])

    contract = pd.read_csv(paths["box_treasury_transaction_detail_contract"])
    row = contract.loc[contract["name"].eq("box_treasury_transaction_detail.csv")].iloc[0]
    assert row["artifact_role"] == "canonical_source"
    assert row["grain"] == "tx"
    assert row["currency_policy"] == "by_currency"
    assert row["source_authority"] == "source_of_truth_for_treasury_flow"


def _write_report_sources(tmp_path: Path, *, closing_control: float = 700.0) -> tuple[Path, Path, Path]:
    detail_path = tmp_path / "box_treasury_transaction_detail.csv"
    qa_path = tmp_path / "box_treasury_transaction_detail_qa.csv"
    accountability_path = tmp_path / "monthly_cash_accountability.csv"

    pd.DataFrame(
        [
            {
                "tx_id": "rent",
                "Date": "2026-05-01",
                "period": "2026-05",
                "period_end": "2026-05-31",
                "Box": "Property Management",
                "Currency": "ARS",
                "movement_basis": "actual_cash",
                "cash_direction": "in",
                "cash_category": "rent",
                "amount_in": 1000.0,
                "amount_out": 0.0,
                "net_amount": 1000.0,
                "payer": "Tenant",
                "receiver": "PM",
                "Lugar": "CABA",
                "Detalle": "Renta mayo",
                "semantic_bucket": "operating_revenue",
                "semantic_subbucket": "rent",
                "direction_source": "box_party_match",
                "cash_effect": "cash_in_box",
                "debt_effect": "none",
                "classification_status": "classified",
                "classification_confidence": "high",
                "review_required": False,
                "rule_id": "R001",
                "source_file": "ledger.csv",
                "source_row": 1,
            },
            {
                "tx_id": "tax",
                "Date": "2026-05-02",
                "period": "2026-05",
                "period_end": "2026-05-31",
                "Box": "Property Management",
                "Currency": "ARS",
                "movement_basis": "actual_cash",
                "cash_direction": "out",
                "cash_category": "taxes",
                "amount_in": 0.0,
                "amount_out": 200.0,
                "net_amount": -200.0,
                "payer": "PM",
                "receiver": "ABL",
                "Lugar": "CABA",
                "Detalle": "ABL mayo",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
                "direction_source": "box_party_match",
                "cash_effect": "cash_out_box",
                "debt_effect": "none",
                "classification_status": "classified",
                "classification_confidence": "high",
                "review_required": False,
                "rule_id": "R002",
                "source_file": "ledger.csv",
                "source_row": 2,
            },
            {
                "tx_id": "distribution",
                "Date": "2026-06-01",
                "period": "2026-06",
                "period_end": "2026-06-30",
                "Box": "Property Management",
                "Currency": "ARS",
                "movement_basis": "actual_cash",
                "cash_direction": "out",
                "cash_category": "dividends",
                "amount_in": 0.0,
                "amount_out": 100.0,
                "net_amount": -100.0,
                "payer": "PM",
                "receiver": "Family",
                "Lugar": "",
                "Detalle": "Distribución",
                "semantic_bucket": "family_withdrawal_candidate",
                "semantic_subbucket": "dividend",
                "direction_source": "box_party_match",
                "cash_effect": "cash_out_box",
                "debt_effect": "none",
                "classification_status": "classified",
                "classification_confidence": "medium",
                "review_required": False,
                "rule_id": "R010",
                "source_file": "ledger.csv",
                "source_row": 3,
            },
        ]
    ).to_csv(detail_path, index=False)
    pd.DataFrame(
        [
            {
                "check": "atomic_detail_matches_monthly_treasury",
                "period": "",
                "Box": "",
                "Currency": "",
                "amount_in_gap": 0.0,
                "amount_out_gap": 0.0,
                "net_gap": 0.0,
                "status": "pass",
                "severity": "error",
                "detail": "synthetic",
            }
        ]
    ).to_csv(qa_path, index=False)
    pd.DataFrame(
        [
            {
                "period": "2026-05",
                "Box": "Property Management",
                "Currency": "ARS",
                "total_cash_in": 1000.0,
                "total_cash_out": 200.0,
                "net_cash_flow": 800.0,
                "closing_control": 800.0,
                "validated_cash_status": "unavailable",
                "validated_cash_close": pd.NA,
            },
            {
                "period": "2026-06",
                "Box": "Property Management",
                "Currency": "ARS",
                "total_cash_in": 0.0,
                "total_cash_out": 100.0,
                "net_cash_flow": -100.0,
                "closing_control": closing_control,
                "validated_cash_status": "unavailable",
                "validated_cash_close": pd.NA,
            },
        ]
    ).to_csv(accountability_path, index=False)
    return detail_path, qa_path, accountability_path


def test_transaction_tape_renders_running_arithmetic_and_control_boundary(tmp_path: Path) -> None:
    detail, qa, accountability = _write_report_sources(tmp_path)
    out = render_report(
        detail_path=detail,
        detail_qa_path=qa,
        accountability_path=accountability,
        out_dir=tmp_path / "report",
        as_of_date="2026-06-30",
    )

    trace = pd.read_csv(out["trace"])
    assert list(trace["accum_in"]) == [1000.0, 1000.0, 1000.0]
    assert list(trace["accum_out"]) == [0.0, 200.0, 300.0]
    assert list(trace["accum_net"]) == [1000.0, 800.0, 700.0]
    validation = pd.read_csv(out["validation"])
    assert validation["status"].eq("pass").all()

    text = out["html"].read_text(encoding="utf-8")
    assert "Rendición transaccional de tesorería" in text
    assert "Property Management · ARS" in text
    assert "Σ entradas" in text
    assert "Caja validada" in text
    assert "no constituye por sí solo efectivo físico validado" in text


def test_transaction_tape_fails_closed_when_control_does_not_reconcile(tmp_path: Path) -> None:
    detail, qa, accountability = _write_report_sources(tmp_path, closing_control=699.0)
    with pytest.raises(ValueError, match="validation failed"):
        render_report(
            detail_path=detail,
            detail_qa_path=qa,
            accountability_path=accountability,
            out_dir=tmp_path / "report",
            as_of_date="2026-06-30",
        )
