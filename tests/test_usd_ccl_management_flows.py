from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.artifacts.manifest import artifact_contract_for_name
from accounting.management.usd_ccl_flows import (
    MANAGEMENT_IMPLEMENTATION_ID,
    ManagementProjectionContractError,
    build_usd_ccl_management_flows,
)
from accounting.management.usd_ccl_run import run_usd_ccl_management_flows
from accounting.marts.semantic import build_semantic_outputs
from accounting.valuation.usd_ccl import build_usd_ccl_valuation


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "fixtures" / "management_usd_ccl_flow_fixture.csv"
RATES = ROOT / "fixtures" / "synthetic_ccl_rates.csv"
POLICY = ROOT / "fixtures" / "valuation_policy_v1.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build(tmp_path: Path) -> dict[str, Path]:
    semantic = build_semantic_outputs(pd.read_csv(LEDGER), tmp_path / "semantic")
    valuation = build_usd_ccl_valuation(
        ledger_path=LEDGER,
        rates_path=RATES,
        policy_path=POLICY,
        output_dir=tmp_path / "valuation",
        run_id="management-fixture",
        source_scope_tag="synthetic",
    )
    return build_usd_ccl_management_flows(
        ledger_path=LEDGER,
        semantic_audit_path=semantic["classification_audit"],
        valuation_sidecar_path=valuation["sidecar"],
        valuation_manifest_path=valuation["manifest"],
        output_dir=tmp_path / "management",
    )


def test_row_audit_quarantines_locally_without_rewriting_evidence(tmp_path: Path) -> None:
    outputs = _build(tmp_path)
    audit = {row["tx_id"]: row for row in _rows(outputs["audit"])}

    negative = audit["mgmt-negative-opex"]
    assert negative["amount"] == "-1200"
    assert negative["amount_usd_ccl"] == "-1.000000"
    assert negative["management_eligibility"] == "review_required"
    assert negative["eligibility_reason"] == "negative_native_amount"

    overlap = audit["mgmt-fx-overlap"]
    assert overlap["semantic_bucket"] == "operating_revenue"
    assert overlap["management_eligibility"] == "review_required"
    assert overlap["eligibility_reason"] == "fx_semantic_overlap"

    unavailable = audit["mgmt-missing-rate"]
    assert unavailable["amount_usd_ccl"] == ""
    assert unavailable["management_eligibility"] == "unavailable_valuation"
    assert unavailable["eligibility_reason"] == "unavailable_missing_rate"


def test_complete_cells_reconcile_and_incomplete_cells_are_na(tmp_path: Path) -> None:
    outputs = _build(tmp_path)
    components = _rows(outputs["components"])
    keyed = {
        (row["Box"], row["semantic_bucket"], row["semantic_subbucket"]): row
        for row in components
    }

    rent = keyed[("Property Management", "operating_revenue", "rent")]
    assert rent["projection_status"] == "complete"
    assert rent["contributing_rows"] == "3"
    assert rent["eligible_rows"] == "3"
    assert rent["value_usd_ccl"] == "110.000000"
    assert rent["available_value_usd_ccl"] == "110.000000"

    opex = keyed[("Property Management", "property_opex", "taxes")]
    funding = keyed[("Property Management", "funding_contribution", "family_or_tenant_contribution")]
    draw = keyed[("Property Management", "family_withdrawal_candidate", "personal_expense")]
    assert opex["value_usd_ccl"] == "1.000000"
    assert funding["value_usd_ccl"] == "2.000000"
    assert draw["value_usd_ccl"] == "1.000000"

    negative_opex = keyed[("Family Business", "property_opex", "taxes")]
    assert negative_opex["projection_status"] == "incomplete_review_required"
    assert negative_opex["value_usd_ccl"] == ""
    assert negative_opex["reportable_value_usd_ccl"] == ""
    assert negative_opex["available_value_usd_ccl"] == "0"
    assert negative_opex["negative_amount_rows"] == "1"

    overlap_rent = keyed[("Family Business", "operating_revenue", "rent")]
    assert overlap_rent["value_usd_ccl"] == ""
    assert overlap_rent["fx_overlap_rows"] == "1"

    clean_fx = keyed[("Property Management", "treasury_fx", "fx_conversion_proceeds")]
    assert clean_fx["projection_status"] == "complete"
    assert clean_fx["value_usd_ccl"] == "1.000000"

    missing = keyed[("Family Business", "funding_contribution", "family_or_tenant_contribution")]
    assert missing["value_usd_ccl"] == ""
    assert missing["missing_valuation_rows"] == "1"


def test_eligibility_characterization_counts(tmp_path: Path) -> None:
    audit = _rows(_build(tmp_path)["audit"])
    assert sum(row["management_eligibility"] == "eligible" for row in audit) == 7
    assert sum(row["eligibility_reason"] == "negative_native_amount" for row in audit) == 1
    assert sum(row["eligibility_reason"] == "fx_semantic_overlap" for row in audit) == 1
    assert sum(row["management_eligibility"] == "unavailable_valuation" for row in audit) == 1


def test_inputs_must_have_identical_unique_tx_coverage(tmp_path: Path) -> None:
    semantic = build_semantic_outputs(pd.read_csv(LEDGER), tmp_path / "semantic")
    valuation = build_usd_ccl_valuation(
        ledger_path=LEDGER,
        rates_path=RATES,
        policy_path=POLICY,
        output_dir=tmp_path / "valuation",
        run_id="management-fixture",
        source_scope_tag="synthetic",
    )
    short_audit = tmp_path / "short_audit.csv"
    lines = semantic["classification_audit"].read_text(encoding="utf-8").splitlines()
    short_audit.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ManagementProjectionContractError, match="identical tx_id coverage"):
        build_usd_ccl_management_flows(
            ledger_path=LEDGER,
            semantic_audit_path=short_audit,
            valuation_sidecar_path=valuation["sidecar"],
            valuation_manifest_path=valuation["manifest"],
            output_dir=tmp_path / "management",
        )


def test_valuation_manifest_must_bind_ledger_and_sidecar(tmp_path: Path) -> None:
    semantic = build_semantic_outputs(pd.read_csv(LEDGER), tmp_path / "semantic")
    valuation = build_usd_ccl_valuation(
        ledger_path=LEDGER,
        rates_path=RATES,
        policy_path=POLICY,
        output_dir=tmp_path / "valuation",
        run_id="management-fixture",
        source_scope_tag="synthetic",
    )
    manifest = json.loads(valuation["manifest"].read_text(encoding="utf-8"))
    manifest["source_ledger_sha256"] = "0" * 64
    mismatched = tmp_path / "mismatched_manifest.json"
    mismatched.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManagementProjectionContractError, match="supplied ledger SHA"):
        build_usd_ccl_management_flows(
            ledger_path=LEDGER,
            semantic_audit_path=semantic["classification_audit"],
            valuation_sidecar_path=valuation["sidecar"],
            valuation_manifest_path=mismatched,
            output_dir=tmp_path / "management",
        )


def test_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")
    assert first["audit"].read_bytes() == second["audit"].read_bytes()
    assert first["components"].read_bytes() == second["components"].read_bytes()


def test_management_stage_does_not_modify_native_or_valuation_evidence(tmp_path: Path) -> None:
    semantic = build_semantic_outputs(pd.read_csv(LEDGER), tmp_path / "semantic")
    valuation = build_usd_ccl_valuation(
        ledger_path=LEDGER,
        rates_path=RATES,
        policy_path=POLICY,
        output_dir=tmp_path / "valuation",
        run_id="management-fixture",
        source_scope_tag="synthetic",
    )
    evidence = [LEDGER, semantic["classification_audit"], valuation["sidecar"]]
    before = {path: path.read_bytes() for path in evidence}
    build_usd_ccl_management_flows(
        ledger_path=LEDGER,
        semantic_audit_path=semantic["classification_audit"],
        valuation_sidecar_path=valuation["sidecar"],
        valuation_manifest_path=valuation["manifest"],
        output_dir=tmp_path / "management",
    )
    assert {path: path.read_bytes() for path in evidence} == before


def test_artifact_contracts_keep_management_projection_noncanonical() -> None:
    audit = artifact_contract_for_name("management_usd_ccl_flow_audit.csv")
    components = artifact_contract_for_name("monthly_management_usd_ccl_components.csv")
    assert audit["artifact_role"] == "diagnostic"
    assert components["artifact_role"] == "derived_valuation"
    assert audit["source_authority"] == "derived_valuation_evidence"
    assert components["frontend_suitability"] == "internal_only"


def test_existing_run_orchestration_is_offline_and_content_addressed(tmp_path: Path) -> None:
    run_root = tmp_path / "exact-run"
    run_root.mkdir()
    ledger = run_root / "ledger_canonical.csv"
    ledger.write_bytes(LEDGER.read_bytes())
    build_semantic_outputs(pd.read_csv(ledger), run_root)
    before = ledger.read_bytes()

    outputs = run_usd_ccl_management_flows(
        run_root=run_root,
        rates_path=RATES,
        policy_path=ROOT / "reference" / "fx" / "ccl_txn_prev_available_v1.json",
    )
    valuation_dir = outputs["sidecar"].parent
    assert json.loads(outputs["manifest"].read_text())["mode"] == "offline"
    assert valuation_dir.parent == run_root / "valuations" / "usd_ccl"
    assert valuation_dir.name == json.loads(outputs["manifest"].read_text())["valuation_id"]
    assert outputs["management_audit"].parent == valuation_dir / "management" / MANAGEMENT_IMPLEMENTATION_ID
    assert outputs["management_components"].is_file()
    assert ledger.read_bytes() == before

    rerun = run_usd_ccl_management_flows(
        run_root=run_root,
        rates_path=RATES,
        policy_path=ROOT / "reference" / "fx" / "ccl_txn_prev_available_v1.json",
    )
    assert rerun["sidecar"] == outputs["sidecar"]
