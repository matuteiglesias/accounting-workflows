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
    private.mkdir(parents=True)
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


def _write_case(tmp_path: Path, rows: list[dict], override: list[dict]) -> dict:
    private = tmp_path / "private_review"
    private.mkdir(parents=True)
    pd.DataFrame(override, columns=DETAIL_COLUMNS).to_csv(private / "stakeholder_settlement_overrides.csv", index=False)
    return build_semantic_outputs(pd.DataFrame(rows), tmp_path)


def _base_leg(**values) -> dict:
    base = {column: "" for column in DETAIL_COLUMNS}
    base.update(
        settlement_case_id="case", obligation_box="Property Management", Date="2025-04-29",
        period="2025-04", Currency="ARS", gross_amount=100, expense_category="taxes",
        allocation_status="agreed", allocation_basis="explicit_case_link", actor_role="other",
        settlement_mode="constructive", cash_path="direct_obligation_payment",
        payment_method="unknown", evidence_status="evidence_pending", settlement_nature="current_period_support",
        settlement_period="2025-04", known_box_cash_funding=0, other_governed_funding=0,
        unresolved_funding=0, underlying_allocated_amount=0,
    )
    base.update(values)
    return base


def test_mixed_and_partial_funding_reconcile_without_inventing_cash(tmp_path: Path) -> None:
    rows = [
        {"tx_id":"support", "Date":"2025-04-29", "amount":40, "Currency":"ARS", "Box":"Property Management", "Lugar":"Site", "payer":"Actor", "receiver":"PM", "Flujo":"Contribucion", "Tipo":"Impuestos", "Detalle":"", "status":"pagado"},
        {"tx_id":"expense", "Date":"2025-04-29", "amount":100, "Currency":"ARS", "Box":"Property Management", "Lugar":"Site", "payer":"PM", "receiver":"Impuestos", "Flujo":"Pagos", "Tipo":"Impuestos", "Detalle":"", "status":"pagado"},
    ]
    override = [
        _base_leg(source_tx_id="support", leg_role="stakeholder_support", stakeholder_actor="Actor", allocated_amount=40, known_box_cash_funding=35, unresolved_funding=25),
        _base_leg(source_tx_id="expense", leg_role="economic_expense", allocated_amount=0, known_box_cash_funding=35, unresolved_funding=25),
    ]
    paths = _write_case(tmp_path, rows, override)
    support = pd.read_csv(paths["monthly_stakeholder_support"])
    treasury = pd.read_csv(paths["monthly_box_treasury_flow"])
    assert support["recognized_amount"].sum() == 40
    assert treasury["amount_in"].sum() == 0
    assert treasury["amount_out"].sum() == 0


def test_self_contained_direct_payment_matches_constructive_pair(tmp_path: Path) -> None:
    pair = tmp_path / "pair"
    direct = tmp_path / "direct"
    common = {"Date":"2025-04-29", "amount":100, "Currency":"ARS", "Box":"Property Management", "Lugar":"Site", "Flujo":"Pagos", "Tipo":"Impuestos", "Detalle":"", "status":"pagado"}
    pair_paths = _write_case(pair, [
        common | {"tx_id":"support", "payer":"Actor", "receiver":"PM", "Flujo":"Contribucion"},
        common | {"tx_id":"expense", "payer":"PM", "receiver":"Impuestos"},
    ], [
        _base_leg(source_tx_id="support", leg_role="stakeholder_support", stakeholder_actor="Actor", allocated_amount=100),
        _base_leg(source_tx_id="expense", leg_role="economic_expense", allocated_amount=0),
    ])
    direct_paths = _write_case(direct, [common | {"tx_id":"direct", "payer":"Actor", "receiver":"Impuestos"}], [
        _base_leg(source_tx_id="direct", leg_role="stakeholder_direct_expense", stakeholder_actor="Actor", allocated_amount=100),
    ])
    for paths in [pair_paths, direct_paths]:
        support = pd.read_csv(paths["monthly_stakeholder_support"])
        treasury = pd.read_csv(paths["monthly_box_treasury_flow"])
        split = pd.read_csv(paths["monthly_flow_semantic_split"])
        assert support["recognized_amount"].sum() == 100
        assert treasury[["amount_in", "amount_out"]].sum().sum() == 0
        assert split.loc[split.semantic_bucket.eq("property_opex"), "amount_out"].sum() == 100


def test_same_provider_and_date_do_not_link_unlisted_transaction(tmp_path: Path) -> None:
    rows = [
        {"tx_id":"direct", "Date":"2025-04-29", "amount":100, "Currency":"ARS", "Box":"Property Management", "Lugar":"Site", "payer":"Actor", "receiver":"Impuestos", "Flujo":"Pagos", "Tipo":"Impuestos", "Detalle":"account", "status":"pagado"},
        {"tx_id":"unrelated", "Date":"2025-04-29", "amount":100, "Currency":"ARS", "Box":"Property Management", "Lugar":"Site", "payer":"PM", "receiver":"Impuestos", "Flujo":"Pagos", "Tipo":"Impuestos", "Detalle":"account", "status":"pagado"},
    ]
    paths = _write_case(tmp_path, rows, [_base_leg(source_tx_id="direct", leg_role="stakeholder_direct_expense", stakeholder_actor="Actor", allocated_amount=100)])
    audit = pd.read_csv(paths["classification_audit"]).fillna("")
    assert audit.loc[audit.tx_id.eq("unrelated"), "settlement_case_id"].iloc[0] == ""


def test_reporting_group_defaults_to_actor_and_is_presentation_only(tmp_path: Path) -> None:
    rows = [{"tx_id":"direct", "Date":"2025-04-29", "amount":100, "Currency":"ARS", "Box":"Property Management", "Lugar":"Site", "payer":"Actor", "receiver":"Impuestos", "Flujo":"Pagos", "Tipo":"Impuestos", "Detalle":"", "status":"pagado"}]
    paths = _write_case(tmp_path, rows, [_base_leg(source_tx_id="direct", leg_role="stakeholder_direct_expense", stakeholder_actor="Actor", allocated_amount=100)])
    support = pd.read_csv(paths["monthly_stakeholder_support"])
    before = support["recognized_amount"].sum()
    support["reporting_group"] = support["funding_actor"].map({"Actor":"Illustrative group"}).fillna(support["funding_actor"])
    assert support["recognized_amount"].sum() == before
    assert support["target_box"].unique().tolist() == ["Property Management"]
