from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from accounting.cutoff import cutoff_metadata
from accounting.rendition.build import MANDATORY_ACTIVITY_CAVEAT, compile_rendition
from accounting.rendition.scope import PropertyRegistry
from accounting.rendition.spec import load_rendition_spec
from accounting.scope import scope_metadata


def _run(tmp_path: Path, *, cutoff: str = "2026-09-30") -> Path:
    run = tmp_path / "20260930T000000Z_FBPM"
    (run / "meta").mkdir(parents=True)
    params = {}
    params.update(scope_metadata({"Family Business", "Property Management"}))
    params.update(cutoff_metadata(cutoff))
    (run / "meta" / "stage_A_ingest.json").write_text(
        json.dumps({"params": params}), encoding="utf-8"
    )

    cash = pd.DataFrame(
        [
            {
                "tx_id": "rent-ars",
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
                "Lugar": "Balbin 4148",
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
                "tx_id": "tax-ars",
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
                "receiver": "AGIP",
                "Lugar": "Balbin 4148",
                "Detalle": "ABL",
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
                "tx_id": "rent-usd",
                "Date": "2026-05-03",
                "period": "2026-05",
                "period_end": "2026-05-31",
                "Box": "Property Management",
                "Currency": "USD",
                "movement_basis": "actual_cash",
                "cash_direction": "in",
                "cash_category": "rent",
                "amount_in": 100.0,
                "amount_out": 0.0,
                "net_amount": 100.0,
                "payer": "Tenant",
                "receiver": "PM",
                "Lugar": "Balbin 4148",
                "Detalle": "Renta USD",
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
                "source_row": 3,
            },
            {
                "tx_id": "other-property",
                "Date": "2026-05-04",
                "period": "2026-05",
                "period_end": "2026-05-31",
                "Box": "Property Management",
                "Currency": "ARS",
                "movement_basis": "actual_cash",
                "cash_direction": "in",
                "cash_category": "rent",
                "amount_in": 999.0,
                "amount_out": 0.0,
                "net_amount": 999.0,
                "payer": "Tenant",
                "receiver": "PM",
                "Lugar": "Other Property",
                "Detalle": "Outside scope",
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
                "source_row": 4,
            },
        ]
    )
    cash.to_csv(run / "box_treasury_transaction_detail.csv", index=False)
    pd.DataFrame(
        [{"check": "atomic_detail_matches_monthly_treasury", "status": "pass"}]
    ).to_csv(run / "box_treasury_transaction_detail_qa.csv", index=False)

    audit = cash.rename(columns={"net_amount": "amount"}).copy()
    audit["amount"] = audit["amount"].abs()
    audit["direction"] = audit["cash_direction"]
    audit["funding_actor"] = ""
    audit["funding_channel"] = ""
    audit["settlement_case_id"] = ""
    audit["settlement_mode"] = ""
    audit["leg_role"] = ""
    direct = pd.DataFrame(
        [
            {
                "tx_id": "direct-tax",
                "Date": "2026-05-05",
                "Box": "Property Management",
                "Currency": "ARS",
                "Lugar": "Balbin 4148",
                "amount": 50.0,
                "payer": "Tenant",
                "receiver": "AGIP",
                "Detalle": "Tenant direct tax",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
                "cash_effect": "no_cash_in_box_direct_payment",
                "debt_effect": "none",
                "rule_id": "R002",
                "review_required": False,
                "source_file": "ledger.csv",
                "source_row": 5,
                "direction": "out",
                "funding_actor": "Tenant",
                "funding_channel": "tenant_direct_tax_payment",
                "settlement_case_id": "S1",
                "settlement_mode": "constructive",
                "leg_role": "stakeholder_direct_expense",
            }
        ]
    )
    audit = pd.concat([audit, direct], ignore_index=True, sort=False)
    audit.to_csv(run / "classification_audit.csv", index=False)

    pd.DataFrame(
        [
            {
                "period": "2025-12",
                "Box": "Property Management",
                "Currency": "ARS",
                "closing_control": 321.0,
                "validated_cash_status": "unavailable",
            },
            {
                "period": "2025-12",
                "Box": "Property Management",
                "Currency": "USD",
                "closing_control": 12.0,
                "validated_cash_status": "unavailable",
            },
            {
                "period": "2026-05",
                "Box": "Property Management",
                "Currency": "ARS",
                "closing_control": 1799.0,
                "validated_cash_status": "unavailable",
            },
        ]
    ).to_csv(run / "monthly_cash_accountability.csv", index=False)
    return run


def _registry(tmp_path: Path) -> PropertyRegistry:
    path = tmp_path / "properties.csv"
    path.write_text(
        "property_id,display_name,lugar_selector,box_selector,optional_notes\n"
        "BALBIN_4148,Balbin 4148,Balbin 4148,Property Management,reporting only\n",
        encoding="utf-8",
    )
    return PropertyRegistry.from_csv(path)


def _spec(tmp_path: Path, run: Path, opening: str = "unavailable") -> Path:
    path = tmp_path / "rendition.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "accounting.rendition_spec.v1",
                "rendition_id": "RDC-MI-2026-09-30-001",
                "source_run_id": run.name,
                "rendidor": {"display_name": "Matias Iglesias"},
                "period": {"from": "2026-01-01", "through": "2026-09-30"},
                "properties": ["BALBIN_4148"],
                "boxes": ["Property Management"],
                "recipients": ["Eduardo Iglesias"],
                "purpose": "private_account_rendition",
                "language": "es-AR",
                "opening_basis": {"type": opening},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_compiler_builds_cash_non_cash_and_summary_without_cross_property_leakage(tmp_path: Path) -> None:
    run = _run(tmp_path)
    spec = load_rendition_spec(_spec(tmp_path, run))
    paths = compile_rendition(
        spec=spec,
        registry=_registry(tmp_path),
        run_root=run,
        out_base=tmp_path / "out",
    )

    cash = pd.read_csv(paths["cash_tape"])
    assert set(cash["tx_id"]) == {"rent-ars", "tax-ars", "rent-usd"}
    assert "other-property" not in set(cash["tx_id"])
    assert set(cash["property_id"]) == {"BALBIN_4148"}

    non_cash = pd.read_csv(paths["non_cash_events"])
    assert set(non_cash["tx_id"]) == {"direct-tax"}
    assert float(non_cash["amount"].sum()) == 50.0
    assert "rent-ars" not in set(non_cash["tx_id"])

    summary = pd.read_csv(paths["summary"])
    ars_result = summary.loc[
        summary["property_id"].eq("BALBIN_4148")
        & summary["Currency"].eq("ARS")
        & summary["account_line_id"].eq("activity_result"),
        "amount",
    ].iloc[0]
    usd_result = summary.loc[
        summary["property_id"].eq("BALBIN_4148")
        & summary["Currency"].eq("USD")
        & summary["account_line_id"].eq("activity_result"),
        "amount",
    ].iloc[0]
    assert ars_result == 800.0
    assert usd_result == 100.0
    assert not ((summary["Currency"].eq("ARS+USD"))).any()

    context = json.loads(paths["context"].read_text(encoding="utf-8"))
    assert context["mandatory_activity_caveat"] == MANDATORY_ACTIVITY_CAVEAT

    validation = pd.read_csv(paths["validation"])
    assert validation["status"].eq("pass").all()


def test_governed_prior_close_remains_box_level_and_not_property_cash(tmp_path: Path) -> None:
    run = _run(tmp_path)
    spec = load_rendition_spec(_spec(tmp_path, run, opening="governed_prior_close"))
    paths = compile_rendition(
        spec=spec,
        registry=_registry(tmp_path),
        run_root=run,
        out_base=tmp_path / "out",
    )

    opening = pd.read_csv(paths["opening_basis"])
    ars = opening.loc[opening["Currency"].eq("ARS")].iloc[0]
    assert ars["grain"] == "box_currency"
    assert pd.isna(ars["property_id"]) or ars["property_id"] == ""
    assert float(ars["opening_amount"]) == 321.0
    assert "not_property_attributable" in ars["status"]


def test_zero_origin_fails_if_selected_property_has_earlier_activity(tmp_path: Path) -> None:
    run = _run(tmp_path)
    cash_path = run / "box_treasury_transaction_detail.csv"
    cash = pd.read_csv(cash_path)
    earlier = cash.iloc[0].copy()
    earlier["tx_id"] = "earlier-rent"
    earlier["Date"] = "2025-12-15"
    cash = pd.concat([cash, earlier.to_frame().T], ignore_index=True)
    cash.to_csv(cash_path, index=False)

    audit_path = run / "classification_audit.csv"
    audit = pd.read_csv(audit_path)
    earlier_audit = audit.iloc[0].copy()
    earlier_audit["tx_id"] = "earlier-rent"
    earlier_audit["Date"] = "2025-12-15"
    audit = pd.concat([audit, earlier_audit.to_frame().T], ignore_index=True)
    audit.to_csv(audit_path, index=False)

    spec = load_rendition_spec(_spec(tmp_path, run, opening="zero_origin_from_management_start"))
    with pytest.raises(ValueError, match="earlier selected-property activity"):
        compile_rendition(
            spec=spec,
            registry=_registry(tmp_path),
            run_root=run,
            out_base=tmp_path / "out",
        )


def test_compiler_fails_closed_when_upstream_treasury_qa_failed(tmp_path: Path) -> None:
    run = _run(tmp_path)
    pd.DataFrame([{"check": "x", "status": "fail"}]).to_csv(
        run / "box_treasury_transaction_detail_qa.csv", index=False
    )
    spec = load_rendition_spec(_spec(tmp_path, run))
    with pytest.raises(ValueError, match="QA contains failures"):
        compile_rendition(
            spec=spec,
            registry=_registry(tmp_path),
            run_root=run,
            out_base=tmp_path / "out",
        )
