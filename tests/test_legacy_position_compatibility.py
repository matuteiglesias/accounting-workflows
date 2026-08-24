from __future__ import annotations

"""Compatibility characterization for pre-governed cash/debt position helpers.

These assertions deliberately preserve historical helper behavior that is not
allowed to override modern governed cash/debt identities.

Removal condition: delete the relevant cases when supported professional inputs
always satisfy the governed validated-cash and component-grained debt schemas,
and no supported artifact can reach these legacy helpers.
"""

import pandas as pd

from accounting.professional.drilldown_legacy import (
    _build_annual_cash_close_companion_cell,
    _build_annual_debt_stock_companion_cell,
    _build_cash_control_cell,
    _build_debt_position_cell,
)


def _legacy_cash_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, inferred, bank_a, bank_b in [
        ("2025-12", 80, 50, 30),
        ("2026-01", 100, 70, 30),
    ]:
        end = f"{period}-31"
        rows.extend(
            [
                {
                    "period": period,
                    "period_end": end,
                    "as_of_date": end,
                    "Box": "Property Management",
                    "party": "Alice",
                    "account_id": "",
                    "Currency": "ARS",
                    "close_amount": 30 if period == "2025-12" else 40,
                    "position_type": "internal_balance",
                    "source_type": "internal_party_balance",
                },
                {
                    "period": period,
                    "period_end": end,
                    "as_of_date": end,
                    "Box": "Property Management",
                    "party": "Bob",
                    "account_id": "",
                    "Currency": "ARS",
                    "close_amount": 20 if period == "2025-12" else 10,
                    "position_type": "internal_balance",
                    "source_type": "internal_party_balance",
                },
                {
                    "period": period,
                    "period_end": end,
                    "as_of_date": end,
                    "Box": "Property Management",
                    "party": "",
                    "account_id": "",
                    "Currency": "ARS",
                    "close_amount": inferred,
                    "position_type": "inferred_box_motor",
                    "source_type": "inferred_box_motor",
                },
                {
                    "period": period,
                    "period_end": end,
                    "as_of_date": end,
                    "Box": "Property Management",
                    "party": "",
                    "account_id": "bank-a",
                    "Currency": "ARS",
                    "close_amount": bank_a,
                    "position_type": "cash_close",
                    "source_type": "bank_statement",
                },
                {
                    "period": period,
                    "period_end": end,
                    "as_of_date": end,
                    "Box": "Property Management",
                    "party": "",
                    "account_id": "bank-b",
                    "Currency": "ARS",
                    "close_amount": bank_b,
                    "position_type": "cash_close",
                    "source_type": "bank_statement",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_legacy_monthly_cash_helper_mixes_validated_and_inferred_box_rows() -> None:
    cash = _legacy_cash_rows()
    row = pd.Series(
        {"Currency": "ARS", "Box": "Property Management", "metric": "cash_close"}
    )
    result = _build_cash_control_cell(
        row=row,
        period="2026-01",
        display_value=200,
        source_df=cash,
        source_name="monthly_cash_close.csv",
        default_metric="cash_close",
        tolerance=1e-6,
    )

    assert result[1] == 200
    assert set(result[7]["position_type"]) == {"cash_close", "inferred_box_motor"}
    assert not result[7]["position_type"].eq("internal_balance").any()


def test_legacy_annual_cash_and_diagnostic_population_rules_remain_characterized() -> None:
    cash = _legacy_cash_rows()

    annual = _build_annual_cash_close_companion_cell(
        row=pd.Series({"Currency": "ARS", "Box": "Property Management"}),
        period="2026",
        display_value=250,
        cash_close=cash,
        tolerance=1e-6,
    )
    assert annual[1] == 250
    assert set(annual[7]["position_type"]) == {
        "internal_balance",
        "inferred_box_motor",
        "cash_close",
    }

    diagnostic_row = pd.Series(
        {
            "Currency": "ARS",
            "Box": "Property Management",
            "metric": "diagnostic_box_level",
        }
    )
    diagnostic = _build_cash_control_cell(
        row=diagnostic_row,
        period="2026-01",
        display_value=40,
        source_df=cash,
        source_name="monthly_cash_close.csv",
        default_metric="diagnostic_box_level",
        tolerance=1e-6,
    )
    assert diagnostic[1] == 40

    missing_previous = _build_cash_control_cell(
        row=diagnostic_row,
        period="2025-12",
        display_value=160,
        source_df=cash,
        source_name="monthly_cash_close.csv",
        default_metric="diagnostic_box_level",
        tolerance=1e-6,
    )
    assert missing_previous[1] == 160


def _legacy_debt_position() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, as_of_date, principal, interest in [
        ("2025-02", "2025-02-28", 1000.0, 50.0),
        ("2025-03", "2025-03-31", 850.0, 20.0),
        ("2025-04", "not-a-date-z", 700.0, 0.0),
    ]:
        total = principal + interest
        for component, amount in [
            ("principal", principal),
            ("interest", interest),
            ("total", total),
        ]:
            rows.append(
                {
                    "period": period,
                    "as_of_date": as_of_date,
                    "Currency": "USD",
                    "debtor": "PM",
                    "creditor": "MI",
                    "component": component,
                    "open_amount": amount,
                    "open_principal": principal,
                    "open_interest": interest,
                    "open_total": total,
                }
            )
    return pd.DataFrame(rows)


def test_legacy_debt_position_helper_can_select_invalid_latest_period() -> None:
    position = _legacy_debt_position()

    monthly = _build_debt_position_cell(
        row=pd.Series(
            {"measure": "open_principal", "Currency": "USD", "pair": "PM → MI"}
        ),
        period="2025-03",
        display_value=850,
        debt_position=position,
        tolerance=1e-6,
    )
    assert monthly[1] == 850
    assert monthly[7].iloc[0]["as_of_date"] == "2025-03-31"

    annual = _build_annual_debt_stock_companion_cell(
        row=pd.Series(
            {
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "open_principal",
            }
        ),
        period="2025",
        display_value=700,
        debt_position=position,
        tolerance=1e-6,
    )
    assert annual[1] == 700
