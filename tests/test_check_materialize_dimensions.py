from __future__ import annotations

import pandas as pd

from scripts.check_materialize import (
    REQUIRED_LEDGER_DIMENSIONS,
    _empty_required_dimension_mask,
    _format_empty_dimension_error,
)


def test_materialize_check_flags_empty_required_dimensions_with_human_context() -> None:
    ledger = pd.DataFrame(
        [
            {
                "tx_id": "ok",
                "Date": "2026-01-01",
                "amount": 100.0,
                "Box": "Household",
                "Currency": "ARS",
                "Flujo": "income",
                "Tipo": "rent",
                "source_file": "sheet",
                "source_row": 2,
            },
            {
                "tx_id": "missing_box_and_tipo",
                "Date": "2026-01-02",
                "amount": 110000.0,
                "Box": " ",
                "Currency": "ARS",
                "Flujo": "income",
                "Tipo": pd.NA,
                "source_file": "sheet",
                "source_row": 3,
                "notes": "forgotten classification",
            },
        ]
    )

    mask = _empty_required_dimension_mask(ledger, REQUIRED_LEDGER_DIMENSIONS)
    message = _format_empty_dimension_error(ledger, REQUIRED_LEDGER_DIMENSIONS)

    assert mask.tolist() == [False, True]
    assert "rows=1" in message
    assert "amount_sum=110000.0" in message
    assert "'Box': 1" in message
    assert "'Tipo': 1" in message
    assert "missing_box_and_tipo" in message
    assert "source_row" in message
    assert "Fill Box/Currency/Flujo/Tipo" in message
