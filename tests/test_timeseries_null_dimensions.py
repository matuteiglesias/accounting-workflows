from __future__ import annotations

import pandas as pd

from accounting.core.timeseries import aggregate_per_flow, aggregate_per_party, expand_party_rows


def test_per_flow_preserves_amounts_with_missing_dimensions() -> None:
    ledger = pd.DataFrame(
        [
            {
                "tx_id": "complete",
                "Date": "2026-01-01",
                "Box": "Household",
                "Currency": "ARS",
                "Flujo": "income",
                "Tipo": "rent",
                "amount": 100.0,
            },
            {
                "tx_id": "missing_tipo",
                "Date": "2026-01-02",
                "Box": "Household",
                "Currency": "ARS",
                "Flujo": "income",
                "Tipo": pd.NA,
                "amount": 110000.0,
            },
        ]
    )

    out = aggregate_per_flow(ledger, freq="M")

    assert out["amount"].sum() == ledger["amount"].sum()
    assert out["Tipo"].isna().any()


def test_per_party_preserves_rows_with_missing_flow_dimensions() -> None:
    ledger = pd.DataFrame(
        [
            {
                "tx_id": "missing_flow_dimension",
                "Date": "2026-01-02",
                "Box": "Household",
                "Currency": "ARS",
                "payer": "HH",
                "receiver": "Vendor",
                "Flujo": pd.NA,
                "Tipo": "expense",
                "amount": 110000.0,
            },
        ]
    )

    expanded = expand_party_rows(ledger)
    out = aggregate_per_party(expanded, freq="M")

    assert out["n_tx"].sum() == 2
    assert out["Flujo"].isna().all()
    assert out["amount"].sum() == 0.0
