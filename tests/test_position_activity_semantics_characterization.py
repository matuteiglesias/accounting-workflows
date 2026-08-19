from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.marts.cash import build_monthly_cash_close
from accounting.marts.debt import build_monthly_debt_position
from accounting.professional.drilldown_legacy import (
    _build_annual_cash_close_companion_cell,
    _build_annual_debt_activity_companion_cell,
    _build_annual_debt_stock_companion_cell,
    _build_cash_control_cell,
    _build_debt_activity_cell,
    _build_debt_position_cell,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_cash_fixture(tmp_path: Path) -> pd.DataFrame:
    root = tmp_path / "cash"
    _write(
        root / "daily_cash_position.csv",
        [
            {"Date": "2025-12-31", "Box": "Property Management", "party": "Alice", "Currency": "ARS", "balance": 30},
            {"Date": "2025-12-31", "Box": "Property Management", "party": "Bob", "Currency": "ARS", "balance": 20},
            {"Date": "2026-01-31", "Box": "Property Management", "party": "Alice", "Currency": "ARS", "balance": 40},
            {"Date": "2026-01-31", "Box": "Property Management", "party": "Bob", "Currency": "ARS", "balance": 10},
        ],
    )
    _write(
        root / "box_balance_time_long.freq=M.csv",
        [
            {"TimePeriod": "2025-12", "TimePeriod_end": "2025-12-31", "Box": "Property Management", "Currency": "ARS", "cum_net": 80},
            {"TimePeriod": "2026-01", "TimePeriod_end": "2026-01-31", "Box": "Property Management", "Currency": "ARS", "cum_net": 100},
        ],
    )
    _write(
        root / "validated_cash_close.csv",
        [
            {"period": "2025-12", "period_end": "2025-12-31", "as_of_date": "2025-12-31", "Box": "Property Management", "account_id": "bank-a", "account_name": "Bank A", "Currency": "ARS", "close_amount": 50, "source_type": "bank_statement", "source_reference": "stmt-a-dec", "validation_status": "validated", "validated_by": "fixture-controller", "notes": ""},
            {"period": "2025-12", "period_end": "2025-12-31", "as_of_date": "2025-12-31", "Box": "Property Management", "account_id": "bank-b", "account_name": "Bank B", "Currency": "ARS", "close_amount": 30, "source_type": "bank_statement", "source_reference": "stmt-b-dec", "validation_status": "validated", "validated_by": "fixture-controller", "notes": ""},
            {"period": "2026-01", "period_end": "2026-01-31", "as_of_date": "2026-01-31", "Box": "Property Management", "account_id": "bank-a", "account_name": "Bank A", "Currency": "ARS", "close_amount": 70, "source_type": "bank_statement", "source_reference": "stmt-a-jan", "validation_status": "validated", "validated_by": "fixture-controller", "notes": ""},
            {"period": "2026-01", "period_end": "2026-01-31", "as_of_date": "2026-01-31", "Box": "Property Management", "account_id": "bank-b", "account_name": "Bank B", "Currency": "ARS", "close_amount": 30, "source_type": "bank_statement", "source_reference": "stmt-b-jan", "validation_status": "validated", "validated_by": "fixture-controller", "notes": ""},
        ],
    )
    paths = build_monthly_cash_close(root)
    return pd.read_csv(paths["monthly_cash_close"])


def _build_debt_fixture(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    debt_dir = tmp_path / "debt"
    out_dir = tmp_path / "debt_out"
    _write(
        debt_dir / "debt_balance_monthly.csv",
        [
            {"as_of_date": "2025-02-28", "period": "2025-02", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 1000, "open_interest": 50, "open_total": 1050},
            {"as_of_date": "2025-03-15", "period": "2025-03", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 900, "open_interest": 30, "open_total": 930},
            {"as_of_date": "2025-03-31", "period": "2025-03", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 850, "open_interest": 20, "open_total": 870},
            {"as_of_date": "not-a-date-a", "period": "2025-04", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 800, "open_interest": 0, "open_total": 800},
            {"as_of_date": "not-a-date-z", "period": "2025-04", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 700, "open_interest": 0, "open_total": 700},
        ],
    )
    _write(
        debt_dir / "debt_open_items.csv",
        [
            {"opened_at": "2025-02-01", "debtor": "PM", "creditor": "MI", "currency": "usd", "item_type": "Prestamo", "original_amount": 1000},
            {"opened_at": "2025-02-02", "debtor": "PM", "creditor": "MI", "currency": "usd", "item_type": "Interes", "original_amount": 50},
        ],
    )
    _write(
        debt_dir / "debt_repayment_events.csv",
        [
            {"repayment_date": "2025-03-20", "debtor": "PM", "creditor": "MI", "currency": "usd", "allocated_amount": 180},
            {"repayment_date": "2025-04-20", "debtor": "PM", "creditor": "MI", "currency": "usd", "allocated_amount": 170},
        ],
    )
    paths = build_monthly_debt_position(debt_dir, out_dir)
    return (
        pd.read_csv(paths["monthly_debt_position"]),
        pd.read_csv(paths["monthly_debt_activity"]),
    )


def test_cash_mart_preserves_three_distinct_position_populations(tmp_path: Path) -> None:
    cash = _build_cash_fixture(tmp_path)

    assert set(cash["position_type"]) == {
        "internal_balance",
        "inferred_box_motor",
        "cash_close",
    }
    assert len(cash[cash["position_type"].eq("internal_balance")]) == 4
    assert len(cash[cash["position_type"].eq("inferred_box_motor")]) == 2
    validated = cash[cash["position_type"].eq("cash_close")]
    assert len(validated) == 4
    assert validated["is_frontend_safe"].astype(bool).all()
    assert not cash.loc[
        ~cash["position_type"].eq("cash_close"), "is_frontend_safe"
    ].astype(bool).any()


def test_current_professional_cash_monthly_mix_is_characterized_not_fixed(tmp_path: Path) -> None:
    cash = _build_cash_fixture(tmp_path)
    row = pd.Series({"Currency": "ARS", "Box": "Property Management", "metric": "cash_close"})

    result = _build_cash_control_cell(
        row=row,
        period="2026-01",
        display_value=200,
        source_df=cash,
        source_name="monthly_cash_close.csv",
        default_metric="cash_close",
        tolerance=1e-6,
    )
    matched = result[1]
    selected = result[7]

    # Current behavior: blank-party validated rows and inferred_box_motor are
    # both treated as box-level and summed. The validated accounts total 100
    # and inferred_box_motor is also 100, so the professional helper returns
    # 200. This is intentionally a characterization of the current ambiguity.
    assert matched == 200
    assert set(selected["position_type"]) == {"cash_close", "inferred_box_motor"}
    assert not selected["position_type"].eq("internal_balance").any()


def test_current_professional_cash_annual_and_diagnostic_population_rules_are_frozen(
    tmp_path: Path,
) -> None:
    cash = _build_cash_fixture(tmp_path)

    annual_row = pd.Series({"Currency": "ARS", "Box": "Property Management"})
    annual = _build_annual_cash_close_companion_cell(
        row=annual_row,
        period="2026",
        display_value=250,
        cash_close=cash,
        tolerance=1e-6,
    )
    annual_selected = annual[7]
    assert annual[1] == 250
    assert set(annual_selected["position_type"]) == {
        "internal_balance",
        "inferred_box_motor",
        "cash_close",
    }

    diagnostic_row = pd.Series(
        {"Currency": "ARS", "Box": "Property Management", "metric": "diagnostic_box_level"}
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
    # With no 2025-11 row, current behavior treats the missing prior close as 0.
    assert missing_previous[1] == 160


def test_debt_position_builder_selects_latest_valid_snapshot_and_exposes_component_contract(
    tmp_path: Path,
) -> None:
    position, _ = _build_debt_fixture(tmp_path)

    march = position[position["period"].eq("2025-03")]
    assert set(march["as_of_date"]) == {"2025-03-31"}
    assert set(march["component"]) == {"principal", "interest", "total"}

    expected = {"principal": 850.0, "interest": 20.0, "total": 870.0}
    for component, amount in expected.items():
        row = march[march["component"].eq(component)].iloc[0]
        assert row["open_amount"] == amount
        assert row["open_principal"] == 850.0
        assert row["open_interest"] == 20.0
        assert row["open_total"] == 870.0


def test_debt_position_all_invalid_as_of_dates_currently_fail_open_lexically(
    tmp_path: Path,
) -> None:
    position, _ = _build_debt_fixture(tmp_path)
    april = position[position["period"].eq("2025-04")]

    # Both source as_of_date values are unparseable. Current mart behavior does
    # not reject the month; secondary string ordering selects not-a-date-z.
    assert set(april["as_of_date"]) == {"not-a-date-z"}
    assert april[april["component"].eq("total")].iloc[0]["open_amount"] == 700.0


def test_professional_debt_position_monthly_and_annual_are_snapshot_not_sum(
    tmp_path: Path,
) -> None:
    position, _ = _build_debt_fixture(tmp_path)

    monthly_row = pd.Series(
        {"measure": "open_principal", "Currency": "USD", "pair": "PM → MI"}
    )
    monthly = _build_debt_position_cell(
        row=monthly_row,
        period="2025-03",
        display_value=850,
        debt_position=position,
        tolerance=1e-6,
    )
    assert monthly[1] == 850
    assert len(monthly[7]) == 1
    assert monthly[7].iloc[0]["component"] == "principal"
    assert monthly[7].iloc[0]["as_of_date"] == "2025-03-31"

    annual_row = pd.Series(
        {"Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "open_principal"}
    )
    annual = _build_annual_debt_stock_companion_cell(
        row=annual_row,
        period="2025",
        display_value=700,
        debt_position=position,
        tolerance=1e-6,
    )
    # Annual stock chooses the latest available period (2025-04) rather than
    # summing 2025-02 + 2025-03 + 2025-04. The latest period in this adversarial
    # fixture carries only invalid as_of_date values, which is a separate
    # fail-open behavior frozen by the preceding test.
    assert annual[1] == 700


def test_debt_activity_is_sparse_period_flow_and_annualization_sums_months(
    tmp_path: Path,
) -> None:
    _, activity = _build_debt_fixture(tmp_path)

    assert set(activity["activity_type"]) == {
        "opening_balance",
        "new_claim",
        "interest_accrual",
        "repayment",
        "adjustment",
        "closing_balance",
        "net_change",
    }
    nonzero_specs = {
        "new_principal": "new_claim",
        "interest_accrued": "interest_accrual",
        "repayments": "repayment",
        "adjustments": "adjustment",
        "net_change": "net_change",
    }
    for measure, activity_type in nonzero_specs.items():
        nonzero = activity[pd.to_numeric(activity[measure], errors="coerce").fillna(0).abs().gt(1e-9)]
        assert set(nonzero["activity_type"]) <= {activity_type}

    monthly_row = pd.Series(
        {"measure": "repayments", "Currency": "USD", "pair": "PM → MI"}
    )
    monthly = _build_debt_activity_cell(
        row=monthly_row,
        period="2025-03",
        display_value=180,
        debt_activity=activity,
        tolerance=1e-6,
    )
    assert monthly[1] == 180
    assert set(monthly[7]["activity_type"]) == {"repayment"}

    annual_row = pd.Series(
        {
            "Currency": "USD",
            "debtor": "PM",
            "creditor": "MI",
            "activity_type": "repayments",
        }
    )
    annual = _build_annual_debt_activity_companion_cell(
        row=annual_row,
        period="2025",
        display_value=350,
        debt_activity=activity,
        tolerance=1e-6,
    )
    assert annual[1] == 350


def test_wave4_semantics_inventory_records_blockers_and_boundaries() -> None:
    inventory = pd.read_csv(
        Path("diagnostics/position_activity_semantics_inventory_20260819.csv")
    )
    assert set(inventory["domain"]) == {"debt_position", "debt_activity", "cash_position"}
    assert set(inventory[inventory["domain"].eq("debt_position")]["nature"]) == {"stock"}
    assert set(inventory[inventory["domain"].eq("debt_activity")]["nature"]) == {"flow"}
    cash = inventory[inventory["domain"].eq("cash_position")]
    assert set(cash["contract_readiness"]) == {"BLOCKED_CASH_AUTHORITY"}
    assert cash["current_population_rule"].str.contains("mix", case=False).any()
