from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.professional import drilldown as professional
from accounting.professional import drilldown_legacy as legacy


def _position_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": "2025-02",
                "as_of_date": "2025-02-28",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 1000.0,
                "open_principal": 1000.0,
                "open_interest": 50.0,
                "open_total": 1050.0,
            },
            {
                "period": "2025-03",
                "as_of_date": "2025-03-15",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 900.0,
                "open_principal": 900.0,
                "open_interest": 30.0,
                "open_total": 930.0,
            },
            {
                "period": "2025-03",
                "as_of_date": "2025-03-31",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 850.0,
                "open_principal": 850.0,
                "open_interest": 20.0,
                "open_total": 870.0,
            },
            {
                "period": "2025-03",
                "as_of_date": "2025-03-31",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "total",
                "open_amount": 870.0,
                "open_principal": 850.0,
                "open_interest": 20.0,
                "open_total": 870.0,
            },
        ]
    )


def _invalid_latest_period_rows() -> pd.DataFrame:
    rows = _position_rows()
    invalid = pd.DataFrame(
        [
            {
                "period": "2025-04",
                "as_of_date": "not-a-date-z",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 700.0,
                "open_principal": 700.0,
                "open_interest": 0.0,
                "open_total": 700.0,
            }
        ]
    )
    return pd.concat([rows, invalid], ignore_index=True)


def test_monthly_debt_position_consumes_contract_and_preserves_valid_snapshot_parity() -> None:
    row = pd.Series(
        {"measure": "open_principal", "Currency": "USD", "pair": "PM → MI"}
    )
    result = professional._build_debt_position_cell(
        row=row,
        period="2025-03",
        display_value=850.0,
        debt_position=_position_rows(),
        tolerance=1e-6,
    )

    assert result[0] == "ok"
    assert result[1] == 850.0
    assert result[2] == 0.0
    assert result[3] == "governed_debt_position:monthly"
    selected = result[7]
    assert len(selected) == 1
    assert selected.iloc[0]["component"] == "principal"
    assert selected.iloc[0]["as_of_date"] == "2025-03-31"

    filters = result[5]
    assert filters["spec_id"] == "debt.position.principal"
    assert filters["measure"] == "open_principal"
    assert filters["aggregation"] == "snapshot"
    assert filters["selection"] == "latest_valid_as_of_date"
    assert filters["executor"] == "governed_debt_position_v1"
    assert filters["candidate_rows"] == 2
    assert filters["valid_as_of_rows"] == 2


def test_monthly_debt_position_filters_component_before_snapshot_selection() -> None:
    row = pd.Series(
        {"measure": "open_principal", "Currency": "USD", "pair": "PM → MI"}
    )
    result = professional._build_debt_position_cell(
        row=row,
        period="2025-03",
        display_value=850.0,
        debt_position=_position_rows(),
        tolerance=1e-6,
    )

    candidates = result[8][1][1]
    assert set(candidates["component"]) == {"principal"}
    assert result[1] == 850.0


def test_monthly_invalid_as_of_is_unavailable_not_lexical_or_undated_fallback() -> None:
    row = pd.Series(
        {"measure": "open_principal", "Currency": "USD", "pair": "PM → MI"}
    )
    source = _invalid_latest_period_rows()
    april = source[source["period"].eq("2025-04")].copy()

    # PR11 froze the legacy helper as the before-state: an invalid as_of row is
    # still selectable there. PR13 changes only the governed professional path.
    legacy_result = legacy._build_debt_position_cell(
        row=row,
        period="2025-04",
        display_value=700.0,
        debt_position=april,
        tolerance=1e-6,
    )
    assert legacy_result[1] == 700.0

    governed = professional._build_debt_position_cell(
        row=row,
        period="2025-04",
        display_value=700.0,
        debt_position=april,
        tolerance=1e-6,
    )
    assert governed[0] == "unavailable"
    assert governed[1] == 0.0
    assert governed[2] == -700.0
    assert governed[5]["availability_status"] == "unavailable"
    assert governed[5]["invalid_as_of_policy"] == "unavailable"
    assert governed[5]["valid_as_of_rows"] == 0
    assert len(governed[7]) == 1


def test_annual_debt_stock_reuses_snapshot_primitive_and_never_sums_periods() -> None:
    row = pd.Series(
        {
            "Currency": "USD",
            "debtor": "PM",
            "creditor": "MI",
            "component": "open_principal",
        }
    )
    result = professional._build_annual_debt_stock_companion_cell(
        row=row,
        period="2025",
        display_value=850.0,
        debt_position=_position_rows(),
        tolerance=1e-6,
    )

    assert result[0] == "ok"
    assert result[1] == 850.0
    assert result[3] == "governed_debt_position:annual"
    selected = result[7]
    assert len(selected) == 1
    assert selected.iloc[0]["period"] == "2025-03"
    assert selected.iloc[0]["as_of_date"] == "2025-03-31"
    assert result[5]["selected_period"] == "2025-03"
    assert result[5]["annualization"] == "latest_period_then_latest_valid_as_of_date"


def test_annual_latest_period_with_invalid_as_of_is_unavailable_and_does_not_backfill() -> None:
    row = pd.Series(
        {
            "Currency": "USD",
            "debtor": "PM",
            "creditor": "MI",
            "component": "open_principal",
        }
    )
    source = _invalid_latest_period_rows()

    legacy_result = legacy._build_annual_debt_stock_companion_cell(
        row=row,
        period="2025",
        display_value=700.0,
        debt_position=source,
        tolerance=1e-6,
    )
    assert legacy_result[1] == 700.0

    governed = professional._build_annual_debt_stock_companion_cell(
        row=row,
        period="2025",
        display_value=700.0,
        debt_position=source,
        tolerance=1e-6,
    )
    assert governed[0] == "unavailable"
    assert governed[1] == 0.0
    assert governed[5]["selected_period"] == "2025-04"
    assert governed[5]["valid_as_of_rows"] == 0
    assert "prior periods are not substituted" in governed[5]["reason"]


def test_componentless_legacy_source_stays_on_compatibility_path() -> None:
    source = pd.DataFrame(
        [
            {
                "period": "2026-12",
                "as_of_date": "2026-12-31",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "open_principal": 70.0,
                "open_interest": 7.0,
                "open_total": 77.0,
            }
        ]
    )
    row = pd.Series(
        {
            "Currency": "USD",
            "debtor": "PM",
            "creditor": "MI",
            "component": "open_total",
        }
    )

    result = professional._build_annual_debt_stock_companion_cell(
        row=row,
        period="2026",
        display_value=77.0,
        debt_position=source,
        tolerance=1e-6,
    )
    legacy_result = legacy._build_annual_debt_stock_companion_cell(
        row=row,
        period="2026",
        display_value=77.0,
        debt_position=source,
        tolerance=1e-6,
    )

    assert result[0] == "ok"
    assert result[1] == 77.0
    assert result[3] == legacy_result[3]
    assert not str(result[3]).startswith("governed_debt_position")
    assert "spec_id" not in result[5]


def test_unknown_position_measure_keeps_legacy_compatibility_path() -> None:
    row = pd.Series(
        {"measure": "open_amount", "Currency": "USD", "pair": "PM → MI"}
    )
    result = professional._build_debt_position_cell(
        row=row,
        period="2025-03",
        display_value=870.0,
        debt_position=_position_rows(),
        tolerance=1e-6,
    )
    assert not str(result[3]).startswith("governed_debt_position")


def test_position_executor_remains_isolated_after_pr14_activity_migration() -> None:
    executor = Path("accounting/professional/debt_position_executor.py").read_text(
        encoding="utf-8"
    )
    facade = Path("accounting/professional/drilldown.py").read_text(encoding="utf-8")
    legacy_source = Path("accounting/professional/drilldown_legacy.py").read_text(
        encoding="utf-8"
    )

    assert "resolve_debt_position_spec" in executor
    assert "resolve_debt_activity_spec" not in executor
    assert "monthly_tables_debt_position_matrix" in facade
    assert "annual_debt_stock_by_pair_wide" in facade
    assert "execute_monthly_debt_position" in facade
    assert "execute_annual_debt_position" in facade
    assert "accounting.contracts.debt_position_activity" not in legacy_source

    # PR14 routes activity separately; the position executor remains untouched.
    assert professional._build_debt_activity_cell is not legacy._build_debt_activity_cell
