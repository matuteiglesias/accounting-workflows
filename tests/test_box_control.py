from __future__ import annotations

import pandas as pd
import pytest

from accounting.box_control import (
    inferred_control_schema_supported,
    select_inferred_box_control_period,
)
from accounting.cash_authority import (
    inferred_control_schema_supported as legacy_schema_supported,
    select_inferred_box_control_period as legacy_select,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Property Management",
                "close_amount": 100.0,
                "position_type": "inferred_box_motor",
                "source_type": "inferred_box_motor",
                "cash_suitability": "safe_with_caveat",
                "is_frontend_safe": False,
                "as_of_date": "2026-01-31",
            },
            {
                "period": "2026-01",
                "Currency": "USD",
                "Box": "Property Management",
                "close_amount": 40.0,
                "position_type": "inferred_box_motor",
                "source_type": "inferred_box_motor",
                "cash_suitability": "safe_with_caveat",
                "is_frontend_safe": False,
                "as_of_date": "2026-01-31",
            },
        ]
    )


def _signature(result) -> tuple:
    return (
        result.status,
        result.reason,
        result.value,
        result.period,
        result.currency,
        result.box,
        result.selected.to_dict("records"),
        result.candidates.to_dict("records"),
        result.excluded_inferred.to_dict("records"),
        result.excluded_internal.to_dict("records"),
        result.excluded_other.to_dict("records"),
    )


def test_legacy_import_is_the_new_selector_and_schema_check() -> None:
    assert legacy_select is select_inferred_box_control_period
    assert legacy_schema_supported is inferred_control_schema_supported


@pytest.mark.parametrize(
    ("period", "currency", "box", "expected_status", "expected_reason", "expected_value"),
    [
        ("2026-01", "ARS", "Property Management", "available", "", 100.0),
        ("2026-01", "GBP", "Property Management", "unavailable", "no_inferred_control_candidate", None),
        ("2026-01", "", "Property Management", "unavailable", "missing_currency_or_box", None),
        ("2026-01", "ARS", "", "unavailable", "missing_currency_or_box", None),
        ("2026-02", "ARS", "Property Management", "unavailable", "no_inferred_control_candidate", None),
    ],
)
def test_selector_preserves_scope_and_fail_closed_behavior(
    period: str,
    currency: str,
    box: str,
    expected_status: str,
    expected_reason: str,
    expected_value: float | None,
) -> None:
    result = select_inferred_box_control_period(_rows(), period=period, currency=currency, box=box)
    assert (result.status, result.reason, result.value) == (
        expected_status,
        expected_reason,
        expected_value,
    )


def test_old_and_new_paths_have_identical_results_for_valid_and_invalid_inputs() -> None:
    cases = [
        (_rows(), "2026-01", "ARS", "Property Management"),
        (_rows().iloc[0:0], "2026-01", "ARS", "Property Management"),
        (_rows().assign(as_of_date="not-a-date"), "2026-01", "ARS", "Property Management"),
        (pd.concat([_rows(), _rows().iloc[[0]]], ignore_index=True), "2026-01", "ARS", "Property Management"),
    ]
    for cash, period, currency, box in cases:
        assert _signature(legacy_select(cash, period=period, currency=currency, box=box)) == _signature(
            select_inferred_box_control_period(cash, period=period, currency=currency, box=box)
        )


def test_incomplete_schema_is_unsupported_and_non_numeric_value_is_unavailable() -> None:
    incomplete = _rows().drop(columns=["as_of_date"])
    assert not inferred_control_schema_supported(incomplete)
    assert select_inferred_box_control_period(
        incomplete, period="2026-01", currency="ARS", box="Property Management"
    ).reason == "legacy_or_incomplete_cash_schema"

    non_numeric = _rows().copy()
    non_numeric["close_amount"] = non_numeric["close_amount"].astype(object)
    non_numeric.loc[0, "close_amount"] = "not-a-number"
    result = select_inferred_box_control_period(
        non_numeric, period="2026-01", currency="ARS", box="Property Management"
    )
    assert result.status == "unavailable"
    assert result.reason == "selected_control_value_not_numeric"
