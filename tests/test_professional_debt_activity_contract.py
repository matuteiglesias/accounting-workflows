from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.professional import drilldown as professional
from accounting.professional import drilldown_legacy as legacy


def _activity_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "opening_balance",
                "new_principal": 0.0,
                "interest_accrued": 0.0,
                "repayments": 0.0,
                "adjustments": 0.0,
                "net_change": 0.0,
            },
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "new_principal": 0.0,
                "interest_accrued": 0.0,
                "repayments": 100.0,
                "adjustments": 0.0,
                "net_change": 0.0,
            },
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "new_principal": 0.0,
                "interest_accrued": 0.0,
                "repayments": 80.0,
                "adjustments": 0.0,
                "net_change": 0.0,
            },
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "adjustment",
                "new_principal": 0.0,
                "interest_accrued": 0.0,
                "repayments": 0.0,
                "adjustments": 5.0,
                "net_change": 0.0,
            },
            {
                "period": "2025-04",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "new_principal": 0.0,
                "interest_accrued": 0.0,
                "repayments": 170.0,
                "adjustments": 0.0,
                "net_change": 0.0,
            },
        ]
    )


def test_monthly_debt_activity_consumes_contract_and_sums_owning_rows() -> None:
    row = pd.Series(
        {"measure": "repayments", "Currency": "USD", "pair": "PM → MI"}
    )
    result = professional._build_debt_activity_cell(
        row=row,
        period="2025-03",
        display_value=180.0,
        debt_activity=_activity_rows(),
        tolerance=1e-6,
    )

    assert result[0] == "ok"
    assert result[1] == 180.0
    assert result[2] == 0.0
    assert result[3] == "governed_debt_activity:monthly"
    assert set(result[7]["activity_type"]) == {"repayment"}
    assert len(result[7]) == 2

    filters = result[5]
    assert filters["spec_id"] == "debt.activity.repayment"
    assert filters["activity_type"] == "repayment"
    assert filters["measure"] == "repayments"
    assert filters["aggregation"] == "sum_flow"
    assert filters["executor"] == "governed_debt_activity_v1"
    assert filters["matched_activity_rows"] == 2


def test_annual_debt_activity_alias_resolves_to_same_spec_and_sums_periods() -> None:
    row = pd.Series(
        {
            "Currency": "USD",
            "debtor": "PM",
            "creditor": "MI",
            "pair": "PM → MI",
            "activity_type": "repayments",
        }
    )
    result = professional._build_annual_debt_activity_companion_cell(
        row=row,
        period="2025",
        display_value=350.0,
        debt_activity=_activity_rows(),
        tolerance=1e-6,
    )

    assert result[0] == "ok"
    assert result[1] == 350.0
    assert result[2] == 0.0
    assert result[3] == "governed_debt_activity:annual"
    assert set(result[7]["activity_type"]) == {"repayment"}
    assert set(result[7]["period"]) == {"2025-03", "2025-04"}
    assert result[5]["spec_id"] == "debt.activity.repayment"
    assert result[5]["annualization"] == "sum_periods"


def test_annual_governed_activity_exposes_mart_semantic_leakage_as_residual() -> None:
    source = pd.DataFrame(
        [
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "repayments": 10.0,
            },
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "adjustment",
                # Deliberate mart invariant violation: a repayment measure sits
                # on a non-repayment activity row.
                "repayments": 5.0,
            },
        ]
    )
    row = pd.Series(
        {
            "Currency": "USD",
            "debtor": "PM",
            "creditor": "MI",
            "pair": "PM → MI",
            "activity_type": "repayments",
        }
    )

    legacy_result = legacy._build_annual_debt_activity_companion_cell(
        row=row,
        period="2025",
        display_value=15.0,
        debt_activity=source,
        tolerance=1e-6,
    )
    assert legacy_result[1] == 15.0

    governed = professional._build_annual_debt_activity_companion_cell(
        row=row,
        period="2025",
        display_value=15.0,
        debt_activity=source,
        tolerance=1e-6,
    )
    assert governed[0] == "residual_warning"
    assert governed[1] == 10.0
    assert governed[2] == -5.0
    assert set(governed[7]["activity_type"]) == {"repayment"}


def test_settlements_remains_legacy_until_contract_explicitly_governs_it() -> None:
    row = pd.Series(
        {
            "Currency": "USD",
            "debtor": "PM",
            "creditor": "MI",
            "pair": "PM → MI",
            "activity_type": "settlements",
        }
    )
    result = professional._build_annual_debt_activity_companion_cell(
        row=row,
        period="2025",
        display_value=350.0,
        debt_activity=_activity_rows(),
        tolerance=1e-6,
    )
    assert not str(result[3]).startswith("governed_debt_activity")
    assert result[1] == 350.0


def test_activity_source_without_activity_type_keeps_legacy_compatibility() -> None:
    source = pd.DataFrame(
        [
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "repayments": 180.0,
            }
        ]
    )
    row = pd.Series(
        {"measure": "repayments", "Currency": "USD", "pair": "PM → MI"}
    )
    result = professional._build_debt_activity_cell(
        row=row,
        period="2025-03",
        display_value=180.0,
        debt_activity=source,
        tolerance=1e-6,
    )
    assert not str(result[3]).startswith("governed_debt_activity")
    assert result[1] == 180.0


def test_position_and_activity_executors_are_structurally_non_interchangeable() -> None:
    activity_executor = Path(
        "accounting/professional/debt_activity_executor.py"
    ).read_text(encoding="utf-8")
    position_executor = Path(
        "accounting/professional/debt_position_executor.py"
    ).read_text(encoding="utf-8")

    # DebtActivitySpec cannot enter the position/snapshot executor.
    assert "DebtActivitySpec" not in position_executor
    assert "resolve_debt_activity_spec" not in position_executor
    assert "debt_activity_executor" not in position_executor
    assert "sum_flow" not in position_executor

    # DebtPositionSpec cannot enter the activity/sum executor, and activity has
    # no as-of selection machinery at all.
    assert "DebtPositionSpec" not in activity_executor
    assert "resolve_debt_position_spec" not in activity_executor
    assert "debt_position_executor" not in activity_executor
    assert "latest_valid_as_of_date" not in activity_executor
    assert "pd.to_datetime" not in activity_executor
    assert "as_of_date" not in activity_executor

    assert "resolve_debt_activity_spec" in activity_executor
    assert "resolve_debt_position_spec" in position_executor
