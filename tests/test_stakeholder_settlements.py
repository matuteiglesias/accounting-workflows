from pathlib import Path

import pandas as pd

from accounting.marts.semantic import build_semantic_outputs
from accounting.marts.stakeholder import DETAIL_COLUMNS


def test_constructive_case_preserves_opex_support_and_zero_cash(tmp_path: Path) -> None:
    # Synthetic actors and scaled amounts preserve the private case structure.
    rows = [
        ("mirror", "Household", "HH", "PM", "Pagos", "Servicio", 30),
        ("support_a", "Property Management", "Actor A", "PM", "Contribucion", "Servicio", 6),
        ("support_hh", "Property Management", "HH", "PM", "Contribucion", "Servicio", 30),
        ("expense", "Property Management", "PM", "Servicios", "Pagos", "Servicio", 42),
        ("support_b", "Property Management", "Actor B", "PM", "Contribucion", "Servicio", 6),
        ("unrelated", "Property Management", "Actor A", "PM", "Contribucion", "Servicio", 4.6334),
    ]
    ledger = pd.DataFrame([
        {"tx_id": tx, "Date": "2025-10-08" if tx != "unrelated" else "2025-10-02",
         "amount": amount, "Currency": "ARS", "Box": box, "Lugar": "Site",
         "payer": payer, "receiver": receiver, "Flujo": flujo, "Tipo": tipo,
         "Detalle": "synthetic provider", "status": "pagado"}
        for tx, box, payer, receiver, flujo, tipo, amount in rows
        if tx != "mirror"  # mirror is outside the FBPM Box universe
    ])
    private = tmp_path / "private_review"
    private.mkdir()
    base = dict(
        settlement_case_id="case-1", obligation_box="Property Management",
        Date="2025-10-08", period="2025-10", Currency="ARS", gross_amount=42,
        expense_category="services", allocation_status="agreed",
        allocation_basis="stakeholder_consensus", actor_role="other",
        settlement_mode="constructive", cash_path="direct_obligation_payment",
        physical_payment_id="", physical_payer="", physical_payee="",
        payment_method="unknown", evidence_ref="", evidence_status="evidence_pending",
        mirror_group_id="mirror-1", underlying_participant="",
        underlying_allocated_amount=0, review_note="",
    )
    override = [
        base | {"stakeholder_actor":"Household", "allocated_amount":0, "source_tx_id":"mirror", "leg_role":"responsibility_mirror"},
        base | {"stakeholder_actor":"Actor A", "allocated_amount":6, "source_tx_id":"support_a", "leg_role":"stakeholder_support"},
        base | {"stakeholder_actor":"Household", "allocated_amount":30, "source_tx_id":"support_hh", "leg_role":"stakeholder_support"},
        base | {"stakeholder_actor":"", "allocated_amount":0, "source_tx_id":"expense", "leg_role":"economic_expense"},
        base | {"stakeholder_actor":"Actor B", "allocated_amount":6, "source_tx_id":"support_b", "leg_role":"stakeholder_support"},
        base | {"stakeholder_actor":"", "allocated_amount":0, "source_tx_id":"", "leg_role":"allocation_component", "underlying_participant":"Participant 1", "underlying_allocated_amount":6},
    ]
    pd.DataFrame(override, columns=DETAIL_COLUMNS).to_csv(
        private / "stakeholder_settlement_overrides.csv", index=False
    )

    paths = build_semantic_outputs(ledger, tmp_path)
    audit = pd.read_csv(paths["classification_audit"])
    monthly = pd.read_csv(paths["monthly_stakeholder_support"])
    treasury = pd.read_csv(paths["monthly_box_treasury_flow"])
    split = pd.read_csv(paths["monthly_flow_semantic_split"])

    assert monthly["recognized_amount"].sum() == 42
    assert monthly["physical_cash_amount"].sum() == 0
    case_ids = set(audit.loc[audit["settlement_case_id"].eq("case-1"), "tx_id"])
    assert case_ids == {"support_a", "support_hh", "expense", "support_b"}
    assert "unrelated" not in case_ids
    case_treasury = treasury[treasury["source_tx_ids_sample"].str.contains("support_|expense", na=False)]
    assert case_treasury["amount_in"].sum() == 0
    assert case_treasury["amount_out"].sum() == 0
    opex = split[(split["semantic_bucket"] == "property_opex") & (split["semantic_subbucket"] == "services")]
    assert opex["amount_out"].sum() == 42


def test_explicit_cash_path_outranks_party_match(tmp_path: Path) -> None:
    # Covered end-to-end above: all support rows point to PM but remain non-cash.
    test_constructive_case_preserves_opex_support_and_zero_cash(tmp_path)
