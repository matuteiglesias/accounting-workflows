from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from accounting.management.usd_ccl_flows import (
    _measure_column,
    build_usd_ccl_management_flows,
)
from accounting.marts.semantic import build_semantic_outputs
from accounting.valuation.usd_ccl import build_usd_ccl_valuation


ROOT = Path(__file__).resolve().parents[1]
BASE_LEDGER = ROOT / "fixtures" / "management_usd_ccl_flow_fixture.csv"
RATES = ROOT / "fixtures" / "synthetic_ccl_rates.csv"
POLICY = ROOT / "fixtures" / "valuation_policy_v1.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_extended_ledger(tmp_path: Path, extra_rows: list[dict[str, str]]) -> Path:
    with BASE_LEDGER.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    ledger = tmp_path / "ledger.csv"
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows + extra_rows)
    return ledger


def _build(tmp_path: Path, ledger: Path) -> dict[str, Path]:
    semantic = build_semantic_outputs(pd.read_csv(ledger), tmp_path / "semantic")
    valuation = build_usd_ccl_valuation(
        ledger_path=ledger,
        rates_path=RATES,
        policy_path=POLICY,
        output_dir=tmp_path / "valuation",
        run_id="semantic-measure-fixture",
        source_scope_tag="synthetic",
    )
    return build_usd_ccl_management_flows(
        ledger_path=ledger,
        semantic_audit_path=semantic["classification_audit"],
        valuation_sidecar_path=valuation["sidecar"],
        valuation_manifest_path=valuation["manifest"],
        output_dir=tmp_path / "management",
    )


def test_property_opex_uses_outflow_measure_and_keeps_inbound_mirror_visible(tmp_path: Path) -> None:
    mirror = {
        "tx_id": "mgmt-opex-in-mirror",
        "Date": "2026-01-05",
        "amount": "1200",
        "amount_cents": "120000",
        "Currency": "ARS",
        "payer": "Tax Authority",
        "receiver": "PM",
        "Flujo": "Gastos",
        "Tipo": "Impuestos",
        "status": "recognized",
        "Box": "Property Management",
        "Lugar": "CABA",
        "Detalle": "Inbound mirror of tax movement",
        "source_file": "synthetic",
        "source_row": "11",
        "notes": "",
        "channel": "",
        "cash_path": "Gastos:Impuestos",
    }
    ledger = _write_extended_ledger(tmp_path, [mirror])
    outputs = _build(tmp_path, ledger)

    audit = {row["tx_id"]: row for row in _rows(outputs["audit"])}
    mirrored = audit["mgmt-opex-in-mirror"]
    assert mirrored["semantic_bucket"] == "property_opex"
    assert mirrored["direction"] == "in"
    assert mirrored["measure_direction"] == "out"
    assert mirrored["measure_inclusion"] == "excluded_direction"
    assert mirrored["management_eligibility"] == "eligible"

    components = _rows(outputs["components"])
    opex = next(
        row for row in components
        if row["Box"] == "Property Management"
        and row["semantic_bucket"] == "property_opex"
        and row["semantic_subbucket"] == "taxes"
    )
    assert opex["projection_status"] == "complete"
    assert opex["measure_direction"] == "out"
    assert opex["source_rows"] == "2"
    assert opex["contributing_rows"] == "1"
    assert opex["eligible_rows"] == "1"
    assert opex["measure_direction_excluded_rows"] == "1"
    assert opex["value_usd_ccl"] == "1.000000"
    assert opex["available_value_usd_ccl"] == "1.000000"


def test_unapproved_debt_component_is_excluded_not_review_required(tmp_path: Path) -> None:
    debt = {
        "tx_id": "mgmt-debt-repayment",
        "Date": "2026-01-05",
        "amount": "1200",
        "amount_cents": "120000",
        "Currency": "ARS",
        "payer": "PM",
        "receiver": "Lender",
        "Flujo": "Deuda",
        "Tipo": "Repago",
        "status": "recognized",
        "Box": "Property Management",
        "Lugar": "CABA",
        "Detalle": "Debt repayment",
        "source_file": "synthetic",
        "source_row": "11",
        "notes": "",
        "channel": "",
        "cash_path": "Deuda:Repago",
    }
    ledger = _write_extended_ledger(tmp_path, [debt])
    outputs = _build(tmp_path, ledger)

    audit = {row["tx_id"]: row for row in _rows(outputs["audit"])}
    debt_audit = audit["mgmt-debt-repayment"]
    assert _measure_column(debt_audit) == "amount_abs"
    assert debt_audit["semantic_bucket"] == "debt_movement"
    assert debt_audit["management_eligibility"] == "excluded_not_approved_v1"
    assert debt_audit["measure_inclusion"] == "excluded_not_approved_v1"

    components = _rows(outputs["components"])
    debt_component = next(
        row for row in components
        if row["Box"] == "Property Management"
        and row["semantic_bucket"] == "debt_movement"
        and row["semantic_subbucket"] == "repayment"
    )
    assert debt_component["projection_status"] == "excluded_not_approved_v1"
    assert debt_component["source_rows"] == "1"
    assert debt_component["contributing_rows"] == "0"
    assert debt_component["excluded_not_approved_rows"] == "1"
    assert debt_component["value_usd_ccl"] == ""
    assert debt_component["reportable_value_usd_ccl"] == ""
    assert debt_component["review_required_rows"] == "0"
