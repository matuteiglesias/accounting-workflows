from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.diagnostics.funding_lineage import build_audit, write_outputs


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_funding_lineage_audit_detects_cash_direct_and_debt_candidates(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"

    _write(run / "monthly_flow_semantic_split.csv", [
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Property Management",
            "payer": "Inq", "receiver": "PM", "actor": "Inq", "counterparty": "FB",
            "semantic_bucket": "funding_contribution", "semantic_subbucket": "family_or_tenant_contribution",
            "amount_in": 50, "amount_out": 0, "net_amount": 50, "amount_abs": 50,
            "source_tx_ids_sample": "inq_cash", "rule_ids": "R006_contribution",
        },
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Property Management",
            "payer": "Inquilino", "receiver": "Tax authority", "actor": "Inquilino",
            "semantic_bucket": "property_opex", "semantic_subbucket": "taxes",
            "amount_in": 0, "amount_out": 30, "net_amount": -30, "amount_abs": 30,
            "source_tx_ids_sample": "inq_tax", "rule_ids": "R002_property_taxes",
            "notes": "Inquilino directo a pagar impuestos",
        },
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Family Business",
            "payer": "Alejandro", "receiver": "FB", "actor": "Alejandro",
            "semantic_bucket": "funding_contribution", "semantic_subbucket": "family_or_tenant_contribution",
            "amount_in": 70, "amount_out": 0, "net_amount": 70, "amount_abs": 70,
            "source_tx_ids_sample": "ale_fb", "rule_ids": "R006_contribution",
        },
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Property Management",
            "payer": "Matias", "receiver": "PM", "actor": "Matias",
            "semantic_bucket": "debt_movement", "semantic_subbucket": "principal",
            "amount_in": 100, "amount_out": 0, "net_amount": 100, "amount_abs": 100,
            "source_tx_ids_sample": "debt1", "rule_ids": "R007_debt_principal",
            "notes": "Matias funding deuda",
        },
    ])
    _write(run / "classification_audit.csv", [
        {"tx_id": "inq_cash", "period": "2026-01", "Currency": "ARS", "Box": "Property Management", "payer": "Inq", "receiver": "PM", "semantic_bucket": "funding_contribution", "semantic_subbucket": "family_or_tenant_contribution", "amount_in": 50},
    ])
    _write(pack / "tables" / "overview_balance_dashboard.csv", [
        {"Currency": "ARS", "metric": "Funding / aportes", "2026": 120},
        {"Currency": "ARS", "metric": "Inquilinos directo a pagar impuestos", "2026": 30},
    ])

    audit, summary = build_audit(repo, pack, run)

    assert not audit.empty
    assert set(audit["funding_actor"]) >= {"Inquilino", "Alejandro", "Matías"}
    assert "tenant_direct_tax_payment" in set(audit["funding_channel"])
    assert "family_business_contribution" in set(audit["funding_channel"])
    assert "debt_creation" in set(audit["funding_channel"])
    direct = audit[audit["funding_channel"].eq("tenant_direct_tax_payment")].iloc[0]
    assert direct["is_direct_obligation_payment"] == "true"
    assert direct["obligation_box"] == "Property Management"
    assert "third_party_obligation_payment_not_explicit" in direct["classification_problem"]
    assert not summary.empty

    paths = write_outputs(audit, summary, pack, repo / "docs")
    assert paths["audit"].exists()
    assert paths["summary"].exists()
    assert paths["html"].exists()
    assert paths["markdown"].exists()
    written = pd.read_csv(paths["audit"])
    assert list(written.columns) == list(audit.columns)
