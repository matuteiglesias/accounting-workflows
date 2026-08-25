from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.contracts.funding_support import classify_funding_support
from accounting.marts.semantic import build_semantic_outputs


def test_keyword_only_rows_do_not_acquire_explicit_support_metadata(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        [
            {
                "tx_id": "keyword-only",
                "Date": "2026-01-10",
                "amount": 123.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA",
                "payer": "Matias",
                "receiver": "Vendor",
                "Flujo": "Pago",
                "Tipo": "Otro",
                "Detalle": "funding support deuda mentioned in narrative only",
            }
        ]
    )

    paths = build_semantic_outputs(ledger, tmp_path)
    audit = pd.read_csv(paths["classification_audit"])
    split = pd.read_csv(paths["monthly_flow_semantic_split"])

    row = audit.iloc[0]
    assert row["semantic_bucket"] == "unknown"
    assert row["semantic_subbucket"] == "review_required"
    assert pd.isna(row["funding_actor"]) or row["funding_actor"] == ""
    assert pd.isna(row["funding_channel"]) or row["funding_channel"] == ""
    assert row["debt_effect"] == "none"

    # The strict downstream contract remains strict; the producer simply stops
    # claiming support membership where none has been established.
    assert classify_funding_support(split, strict=True).empty
