from __future__ import annotations

import pandas as pd

from accounting.box_cash import box_party_match_masks, infer_box_party
from accounting.marts.semantic import build_semantic_outputs
from accounting.stage_d.materialize import materialize_box_balance_time_long


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tx_id": "rent",
                "Date": "2026-05-05",
                "amount": 1000.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "payer": "Tenant",
                "receiver": "PM",
                "Flujo": "Cobros",
                "Tipo": "Renta",
                "Lugar": "CABA",
                "Detalle": "rent",
                "status": "pagado",
            },
            {
                "tx_id": "tax",
                "Date": "2026-05-06",
                "amount": 200.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "payer": "PM",
                "receiver": "ABL",
                "Flujo": "Pagos",
                "Tipo": "Impuestos",
                "Lugar": "CABA",
                "Detalle": "tax",
                "status": "pagado",
            },
            {
                "tx_id": "direct",
                "Date": "2026-05-07",
                "amount": 50.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "payer": "Inquilino",
                "receiver": "ABL",
                "Flujo": "Pagos",
                "Tipo": "Impuestos",
                "Lugar": "CABA",
                "Detalle": "Inquilino paga impuesto directo",
                "status": "pagado",
            },
        ]
    )


def test_shared_box_party_mapping_preserves_existing_tokens():
    assert infer_box_party("Property Management") == "PM"
    assert infer_box_party("Family Business") == "FB"
    assert infer_box_party("Household") == "HH"
    assert infer_box_party("") == ""


def test_stage_d_motor_and_semantic_direction_share_physical_match(tmp_path):
    ledger = _ledger()
    matched_in, matched_out = box_party_match_masks(
        ledger, require_nonempty_box_party=True
    )
    assert matched_in.tolist() == [True, False, False]
    assert matched_out.tolist() == [False, True, False]

    motor, _ = materialize_box_balance_time_long(ledger, tmp_path, freq="M")
    assert float(motor["net"].sum()) == 800.0

    paths = build_semantic_outputs(ledger, tmp_path, freq="M")
    audit = pd.read_csv(paths["classification_audit"]).set_index("tx_id")
    assert audit.loc["rent", "direction_source"] == "box_party_match"
    assert audit.loc["rent", "direction"] == "in"
    assert audit.loc["tax", "direction_source"] == "box_party_match"
    assert audit.loc["tax", "direction"] == "out"
    assert audit.loc["direct", "direction_source"] == "semantic_fallback"
