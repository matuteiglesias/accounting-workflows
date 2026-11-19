from __future__ import annotations

import pandas as pd
import pytest

from accounting.debt.position_authority import (
    select_debt_position,
    selected_debt_position_rows,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"period": "2025-03", "as_of_date": "2025-03-19", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 880.0},
            {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 850.0},
            {"period": "2025-04", "as_of_date": "not-a-date-a", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 800.0},
            {"period": "2025-04", "as_of_date": "not-a-date-z", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 700.0},
        ]
    )


def test_monthly_selector_uses_latest_valid_as_of() -> None:
    rows = _rows()
    selection = select_debt_position(rows, period="2025-03")
    assert selection.available
    assert selection.selected_period == "2025-03"
    assert selection.selected_as_of_date == "2025-03-31"
    assert selection.valid_as_of_rows == 2
    selected = selected_debt_position_rows(rows, selection)
    assert selected.iloc[0]["open_amount"] == 850.0


def test_monthly_selector_fails_closed_when_all_as_of_dates_invalid() -> None:
    selection = select_debt_position(_rows(), period="2025-04")
    assert not selection.available
    assert selection.selected_period == "2025-04"
    assert selection.valid_as_of_rows == 0
    assert selection.selected_positions == ()
    assert "no valid as_of_date" in selection.reason


def test_annual_selector_does_not_backfill_when_latest_period_is_invalid() -> None:
    rows = _rows()
    selection = select_debt_position(rows, period="2025", annual=True)
    assert not selection.available
    assert selection.selected_period == "2025-04"
    assert selection.valid_as_of_rows == 0
    assert "prior periods are not substituted" in selection.reason


def test_selector_preserves_native_currency_scope_and_never_aggregates() -> None:
    rows = _rows()
    ars = pd.DataFrame(
        [{"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "ARS", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 900000.0}]
    )
    scoped = pd.concat([rows, ars], ignore_index=True)
    usd = scoped[scoped["Currency"].eq("USD")]
    selection = select_debt_position(usd, period="2025-03")
    selected = selected_debt_position_rows(usd, selection)
    assert set(selected["Currency"]) == {"USD"}
    assert selected.iloc[0]["open_amount"] == 850.0


@pytest.mark.parametrize(
    "observation,requested,annual,available,selected_period,valid_count,reason",
    [
        ("2025-04-30", "2025-04", False, True, "2025-04", 1, "latest valid as_of_date selected"),
        ("2025-04-30", "2025", True, True, "2025-04", 1, "latest valid as_of_date selected"),
        ("invalid-date", "2025-04", False, False, "2025-04", 0, "no valid as_of_date in selected monthly debt-position candidates"),
        ("invalid-date", "2025", True, False, "2025-04", 0, "latest debt-position period has no valid as_of_date; prior periods are not substituted"),
        ("2025-04-30", "2025-03", False, False, "", 0, "no debt-position candidates in requested period scope"),
        ("2025-04-30", "2024", True, False, "", 0, "no debt-position candidates in requested period scope"),
        (pd.NA, "2025-04", False, False, "2025-04", 0, "no valid as_of_date in selected monthly debt-position candidates"),
    ],
)
def test_single_snapshot_fast_path_preserves_governed_selection(
    observation, requested, annual, available, selected_period, valid_count, reason
):
    frame = pd.DataFrame([{
        "period": "2025-04", "as_of_date": observation,
        "debtor": "PM", "creditor": "MI", "Currency": "USD", "open_total": 15,
    }])
    selection = select_debt_position(frame, period=requested, annual=annual)
    assert selection.available is available
    assert selection.selected_period == selected_period
    assert selection.valid_as_of_rows == valid_count
    assert selection.candidate_rows == (1 if selected_period else 0)
    assert selection.reason == reason
    selected = selected_debt_position_rows(frame, selection)
    assert len(selected) == int(available)
    if available:
        assert selection.selected_positions == (0,)
        assert selection.selected_as_of_date == "2025-04-30"
        pd.testing.assert_frame_equal(selected, frame)


def test_single_snapshot_with_missing_period_cannot_be_selected():
    frame = pd.DataFrame([{"period": pd.NA, "as_of_date": "2025-04-30"}])
    selection = select_debt_position(frame)
    assert not selection.available
    assert selection.reason == "no usable debt-position period"
    assert selection.candidate_rows == 1
