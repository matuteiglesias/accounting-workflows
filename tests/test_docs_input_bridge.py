from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.docs_input import DOCS_INPUT_SCHEMA
from accounting.docs_input.build import (
    build_docs_input_bundle,
    build_required_reserve_schedule,
)


def _write_core_sources(run: Path, *, duplicate_audit: bool = False) -> dict[str, bytes]:
    run.mkdir(parents=True, exist_ok=True)

    audit_rows = [
        {"tx_id": "tx-support", "Lugar": "CABA Balbin", "Box": "Property Management"},
        {"tx_id": "tx-gap", "Lugar": "Tigre 01", "Box": "Property Management"},
    ]
    if duplicate_audit:
        audit_rows.append(
            {"tx_id": "tx-support", "Lugar": "CABA Balbin", "Box": "Property Management"}
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(run / "classification_audit.csv", index=False)

    settlement_columns = [
        "settlement_case_id", "source_tx_id", "Date", "Currency", "gross_amount",
        "expense_category", "allocation_status", "allocation_basis",
        "stakeholder_actor", "actor_role", "allocated_amount", "settlement_mode",
        "cash_path", "physical_payment_id", "physical_payer", "physical_payee",
        "payment_method", "evidence_ref", "evidence_status", "leg_role",
        "obligation_box", "underlying_participant", "underlying_allocated_amount",
        "obligation_period", "settlement_period", "funding_status", "debt_origin",
    ]
    settlement = pd.DataFrame(
        [
            {
                "settlement_case_id": "case-1",
                "source_tx_id": "tx-support",
                "Date": "2026-08-10",
                "Currency": "ARS",
                "gross_amount": 100.0,
                "expense_category": "taxes",
                "allocation_status": "agreed",
                "allocation_basis": "documented",
                "stakeholder_actor": "Matias",
                "actor_role": "stakeholder",
                "allocated_amount": 100.0,
                "settlement_mode": "constructive",
                "cash_path": "direct_obligation_payment",
                "physical_payment_id": "",
                "physical_payer": "Matias",
                "physical_payee": "Tax authority",
                "payment_method": "transfer",
                "evidence_ref": "ev-1",
                "evidence_status": "approved",
                "leg_role": "stakeholder_direct_expense",
                "obligation_box": "Property Management",
                "underlying_participant": "",
                "underlying_allocated_amount": 0.0,
                "obligation_period": "2026-08",
                "settlement_period": "2026-08",
                "funding_status": "reconciled",
                "debt_origin": "",
            },
            {
                "settlement_case_id": "case-1",
                "source_tx_id": "",
                "Date": "2026-08-10",
                "Currency": "ARS",
                "gross_amount": 100.0,
                "expense_category": "taxes",
                "allocation_status": "agreed",
                "allocation_basis": "documented",
                "stakeholder_actor": "",
                "actor_role": "",
                "allocated_amount": 0.0,
                "settlement_mode": "constructive",
                "cash_path": "direct_obligation_payment",
                "physical_payment_id": "",
                "physical_payer": "",
                "physical_payee": "",
                "payment_method": "unknown",
                "evidence_ref": "",
                "evidence_status": "evidence_pending",
                "leg_role": "allocation_component",
                "obligation_box": "Property Management",
                "underlying_participant": "Other actor",
                "underlying_allocated_amount": 40.0,
                "obligation_period": "2026-08",
                "settlement_period": "2026-08",
                "funding_status": "reconciled",
                "debt_origin": "",
            },
        ],
        columns=settlement_columns,
    )
    settlement.to_csv(run / "stakeholder_settlement_detail.csv", index=False)

    gaps = pd.DataFrame(
        [
            {
                "source_tx_id": "tx-gap",
                "Date": "2026-08-12",
                "period": "2026-08",
                "Currency": "USD",
                "amount": 2000.0,
                "Lugar": "Tigre 01",
                "description": "Donation cost allocation",
                "status": "review",
                "source_file": "fixture.csv",
                "source_row": "9",
                "economic_scope": "Property Management",
                "accounting_nature": "unresolved_cost_allocation",
                "debt_effect": "none",
                "allocation_status": "unresolved",
                "asserted_bearer": "",
            }
        ]
    )
    gaps.to_csv(run / "cost_allocation_gaps.csv", index=False)

    tracked = {}
    for filename in (
        "classification_audit.csv",
        "stakeholder_settlement_detail.csv",
        "cost_allocation_gaps.csv",
    ):
        tracked[filename] = (run / filename).read_bytes()
    return tracked


def _write_pack(pack: Path) -> None:
    detail_dir = pack / "drilldown" / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(
        [
            {
                "tx_id": "tx-support",
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA Balbin",
                "actor": "Matias",
            },
            {
                "tx_id": "tx-other",
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA Balbin",
                "actor": "Matias",
            },
        ]
    )
    detail.to_csv(detail_dir / "population.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric_id": "TEST.POPULATION",
                "detail_csv_relpath": "drilldown/details/population.csv",
            }
        ]
    ).to_csv(pack / "drilldown" / "professional_drilldown_index.csv", index=False)


def _write_evidence(run: Path) -> None:
    pd.DataFrame(
        [
            {
                "evidence_id": "ev-approved",
                "content_sha256": "a" * 64,
                "media_type": "application/pdf",
                "display_name": "Approved proof",
                "href": "private/approved.pdf",
            },
            {
                "evidence_id": "ev-candidate",
                "content_sha256": "b" * 64,
                "media_type": "image/png",
                "display_name": "Candidate proof",
                "href": "private/candidate.png",
            },
        ]
    ).to_csv(run / "evidence_documents.csv", index=False)
    pd.DataFrame(
        [
            {
                "tx_id": "tx-support",
                "evidence_id": "ev-approved",
                "relation": "payment_proof",
                "status": "approved",
            },
            {
                "tx_id": "tx-other",
                "evidence_id": "ev-candidate",
                "relation": "statement_context",
                "status": "candidate",
            },
        ]
    ).to_csv(run / "transaction_evidence.csv", index=False)


def _write_commitments(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "commitment_id": "ars-due",
                "due_date": "2026-12-15",
                "property": "Tigre 27",
                "Box": "Property Management",
                "Currency": "ARS",
                "expected_amount": "100",
                "amount_status": "known",
                "commitment_status": "known_due",
                "obligation_category": "legal",
                "source_ref": "contract-1",
                "notes": "known upcoming obligation",
            },
            {
                "commitment_id": "ars-scenario",
                "due_date": "2027-01-15",
                "property": "Tigre 27",
                "Box": "Property Management",
                "Currency": "ARS",
                "expected_amount": "999",
                "amount_status": "estimated",
                "commitment_status": "scenario",
                "obligation_category": "maintenance",
                "source_ref": "scenario-1",
                "notes": "visible but not required reserve",
            },
            {
                "commitment_id": "usd-plan",
                "due_date": "2027-02-01",
                "property": "CABA Balbin",
                "Box": "Property Management",
                "Currency": "USD",
                "expected_amount": "50",
                "amount_status": "estimated",
                "commitment_status": "approved_plan",
                "obligation_category": "taxes",
                "source_ref": "plan-1",
                "notes": "approved plan",
            },
        ]
    ).to_csv(path, index=False)


def test_docs_input_bundle_projects_existing_authorities_without_semantic_drift(tmp_path: Path) -> None:
    run = tmp_path / "20260917T000000Z_FBPM"
    tracked = _write_core_sources(run)
    _write_evidence(run)
    pack = tmp_path / "professional_pack"
    _write_pack(pack)
    commitments = tmp_path / "treasury_commitments.csv"
    _write_commitments(commitments)
    out = tmp_path / "docs_input" / run.name

    paths = build_docs_input_bundle(
        run_root=run,
        out_dir=out,
        pack_dir=pack,
        commitments_path=commitments,
        generated_at_utc="2026-09-17T22:00:00+00:00",
    )

    coverage = pd.read_csv(paths["accounting_evidence_coverage"])
    assert len(coverage) == 1
    row = coverage.iloc[0]
    assert row["coverage_status"] == "available"
    assert int(row["transaction_rows"]) == 2
    assert int(row["distinct_tx_ids"]) == 2
    assert int(row["approved_evidence_rows"]) == 1
    assert int(row["candidate_evidence_rows"]) == 1
    assert int(row["missing_evidence_rows"]) == 0
    assert float(row["coverage_pct"]) == 50.0
    assert row["property"] == "CABA Balbin"

    actor_detail = pd.read_csv(paths["actor_property_cost_support_detail"])
    assert len(actor_detail) == 2
    support = actor_detail.loc[actor_detail["leg_role"].eq("stakeholder_direct_expense")].iloc[0]
    assert support["property"] == "CABA Balbin"
    assert support["Box"] == "Property Management"
    assert float(support["recognized_support"]) == 100.0
    component = actor_detail.loc[actor_detail["leg_role"].eq("allocation_component")].iloc[0]
    assert float(component["recognized_support"]) == 0.0
    assert not any(column.startswith("legal_") for column in actor_detail.columns)

    unresolved = pd.read_csv(paths["unresolved_allocation_detail"])
    assert len(unresolved) == 1
    assert float(unresolved.iloc[0]["amount"]) == 2000.0
    assert unresolved.iloc[0]["debt_effect"] == "none"
    assert pd.isna(unresolved.iloc[0]["asserted_bearer"]) or unresolved.iloc[0]["asserted_bearer"] == ""
    assert unresolved.iloc[0]["evidence_status"] == "missing"

    reserve = pd.read_csv(paths["required_reserve_schedule"])
    assert "distribution_capacity" not in reserve.columns
    ars = reserve.loc[reserve["Currency"].eq("ARS")].reset_index(drop=True)
    assert list(ars["commitment_id"]) == ["ars-due", "ars-scenario"]
    assert list(ars["required_reserve_amount"].astype(float)) == [100.0, 0.0]
    assert list(ars["cumulative_required_reserve"].astype(float)) == [100.0, 100.0]
    usd = reserve.loc[reserve["Currency"].eq("USD")].iloc[0]
    assert float(usd["required_reserve_amount"]) == 50.0

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == DOCS_INPUT_SCHEMA
    assert manifest["source_run_id"] == run.name
    assert manifest["accounting_authority_changed"] is False
    assert manifest["legal_interpretation_included"] is False
    assert manifest["distribution_capacity_included"] is False
    assert manifest["prospective_input"]["present"] is True
    names = {item["name"] for item in manifest["artifacts"]}
    assert {
        "accounting_evidence_coverage",
        "actor_property_cost_support_detail",
        "unresolved_allocation_detail",
        "required_reserve_schedule",
        "required_reserve_schedule_qa",
    } <= names

    for filename, before in tracked.items():
        assert (run / filename).read_bytes() == before


def test_missing_evidence_sidecar_is_explicit_and_nonfatal(tmp_path: Path) -> None:
    run = tmp_path / "20260917T000000Z_FBPM"
    _write_core_sources(run)
    pack = tmp_path / "professional_pack"
    _write_pack(pack)
    out = tmp_path / "docs_input" / run.name

    paths = build_docs_input_bundle(run_root=run, out_dir=out, pack_dir=pack)
    coverage = pd.read_csv(paths["accounting_evidence_coverage"])
    assert set(coverage["coverage_status"]) == {"sidecar_absent"}
    assert int(coverage.iloc[0]["approved_evidence_rows"]) == 0
    assert int(coverage.iloc[0]["candidate_evidence_rows"]) == 0
    assert int(coverage.iloc[0]["missing_evidence_rows"]) == 2

    unresolved = pd.read_csv(paths["unresolved_allocation_detail"])
    assert unresolved.iloc[0]["evidence_status"] == "sidecar_absent"
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["evidence_input"]["status"] == "sidecar_absent"
    assert manifest["prospective_input"]["present"] is False
    assert "required_reserve_schedule" not in paths


def test_actor_property_projection_fails_closed_on_nonunique_explicit_tx_join(tmp_path: Path) -> None:
    run = tmp_path / "20260917T000000Z_FBPM"
    _write_core_sources(run, duplicate_audit=True)
    with pytest.raises(ValueError, match="not unique for referenced source tx_id"):
        build_docs_input_bundle(run_root=run, out_dir=tmp_path / "docs")


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate treasury commitment_id"),
        (
            lambda frame: frame.assign(commitment_status="invented"),
            "unsupported treasury commitment commitment_status",
        ),
        (
            lambda frame: frame.assign(amount_status="range_midpoint"),
            "unsupported treasury commitment amount_status",
        ),
    ],
)
def test_prospective_commitments_fail_closed(tmp_path: Path, mutator, match: str) -> None:
    path = tmp_path / "commitments.csv"
    _write_commitments(path)
    frame = pd.read_csv(path, dtype=str)
    mutated = mutator(frame)
    mutated.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        build_required_reserve_schedule(path)


def test_docs_input_rejects_detail_path_escape(tmp_path: Path) -> None:
    run = tmp_path / "20260917T000000Z_FBPM"
    _write_core_sources(run)
    pack = tmp_path / "professional_pack"
    (pack / "drilldown").mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    pd.DataFrame([{"tx_id": "tx-support"}]).to_csv(outside, index=False)
    pd.DataFrame(
        [{"metric_id": "BAD", "detail_csv_relpath": "../outside.csv"}]
    ).to_csv(pack / "drilldown" / "professional_drilldown_index.csv", index=False)

    with pytest.raises(ValueError, match="relative path"):
        build_docs_input_bundle(run_root=run, out_dir=tmp_path / "docs", pack_dir=pack)
