from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from accounting.management.usd_ccl_flows import build_usd_ccl_management_flows
from accounting.marts.semantic import build_semantic_outputs
from accounting.valuation.usd_ccl import build_usd_ccl_valuation


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "fixtures" / "management_usd_ccl_flow_fixture.csv"
HOUSEHOLD_LEDGER = ROOT / "fixtures" / "management_usd_ccl_household_fixture.csv"
RATES = ROOT / "fixtures" / "synthetic_ccl_rates.csv"
POLICY = ROOT / "fixtures" / "valuation_policy_v1.json"
EXPECTED_AUDIT = ROOT / "fixtures" / "management_usd_ccl_flow_audit_expected.csv"
EXPECTED_COMPONENTS = (
    ROOT / "fixtures" / "monthly_management_usd_ccl_components_expected.csv"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_management_outputs_match_pre_contract_migration_and_preserve_valuation(
    tmp_path: Path,
) -> None:
    semantic = build_semantic_outputs(pd.read_csv(LEDGER), tmp_path / "semantic")
    valuation = build_usd_ccl_valuation(
        ledger_path=LEDGER,
        rates_path=RATES,
        policy_path=POLICY,
        output_dir=tmp_path / "valuation",
        run_id="management-contract-parity",
        source_scope_tag="synthetic",
    )
    manifest_before = valuation["manifest"].read_bytes()
    sidecar_sha_before = _sha256(valuation["sidecar"])

    outputs = build_usd_ccl_management_flows(
        ledger_path=LEDGER,
        semantic_audit_path=semantic["classification_audit"],
        valuation_sidecar_path=valuation["sidecar"],
        valuation_manifest_path=valuation["manifest"],
        output_dir=tmp_path / "management",
    )

    assert outputs["audit"].read_bytes() == EXPECTED_AUDIT.read_bytes()
    assert outputs["components"].read_bytes() == EXPECTED_COMPONENTS.read_bytes()
    assert valuation["manifest"].read_bytes() == manifest_before
    assert _sha256(valuation["sidecar"]) == sidecar_sha_before
    assert json.loads(manifest_before)["valuation_artifact_sha256"] == sidecar_sha_before


def test_household_fixture_preserves_corrected_pr52_measures(tmp_path: Path) -> None:
    semantic = build_semantic_outputs(
        pd.read_csv(HOUSEHOLD_LEDGER), tmp_path / "household-semantic"
    )
    valuation = build_usd_ccl_valuation(
        ledger_path=HOUSEHOLD_LEDGER,
        rates_path=RATES,
        policy_path=POLICY,
        output_dir=tmp_path / "household-valuation",
        run_id="household-management-contract",
        source_scope_tag="HH",
    )
    outputs = build_usd_ccl_management_flows(
        ledger_path=HOUSEHOLD_LEDGER,
        semantic_audit_path=semantic["classification_audit"],
        valuation_sidecar_path=valuation["sidecar"],
        valuation_manifest_path=valuation["manifest"],
        output_dir=tmp_path / "household-management",
    )

    with outputs["components"].open(encoding="utf-8", newline="") as handle:
        components = list(csv.DictReader(handle))
    values = {row["semantic_bucket"]: row["value_usd_ccl"] for row in components}
    assert values == {
        "family_withdrawal_candidate": "1.000000",
        "funding_contribution": "2.000000",
        "operating_revenue": "1.000000",
        "property_opex": "1.000000",
    }
    assert {row["projection_status"] for row in components} == {"complete"}
