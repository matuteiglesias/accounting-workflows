from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.cash_authority import (
    select_inferred_box_control_period,
    select_validated_cash_period,
    select_validated_cash_year,
)
from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.metrics.frontier import build_metrics_frontier
from accounting.professional.annual_dashboard_tables import build_annual_cash_close_by_box
from accounting.professional.drilldown import _build_derived_cell
from accounting.professional.drilldown_legacy import (
    _build_annual_cash_close_companion_cell as legacy_annual_cash,
    _build_cash_control_cell as legacy_cash_control,
)


def _row(
    *,
    period: str,
    amount: float,
    position_type: str,
    as_of_date: str,
    account_id: str = "",
    party: str = "",
    source_type: str = "",
    safe: bool = False,
    suitability: str = "",
    validation_status: str = "",
    validated_by: str = "",
    box: str = "Property Management",
    currency: str = "ARS",
) -> dict[str, object]:
    return {
        "period": period,
        "period_end": f"{period}-28" if period.endswith("02") else f"{period}-31",
        "as_of_date": as_of_date,
        "Box": box,
        "party": party,
        "account_id": account_id,
        "account_name": account_id,
        "Currency": currency,
        "close_amount": amount,
        "source_table": (
            "validated_cash_close.csv"
            if position_type == "cash_close"
            else "box_balance_time_long.freq=M.csv"
            if position_type == "inferred_box_motor"
            else "daily_cash_position.csv"
        ),
        "source_date": as_of_date,
        "source_type": source_type,
        "source_reference": "fixture",
        "validation_status": validation_status,
        "validated_by": validated_by,
        "position_type": position_type,
        "cash_suitability": suitability,
        "is_frontend_safe": safe,
        "caveat": "fixture",
        "notes": "",
        "n_source_rows": 1,
        "calculation_rule": "fixture",
    }


def cash_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, internal_a, internal_b, inferred, bank_a, bank_b in [
        ("2025-12", 30, 20, 80, 50, 30),
        ("2026-01", 40, 10, 100, 70, 30),
    ]:
        end = f"{period}-31"
        rows.extend(
            [
                _row(
                    period=period,
                    amount=internal_a,
                    position_type="internal_balance",
                    as_of_date=end,
                    party="Alice",
                    source_type="internal_party_balance",
                    suitability="internal_only",
                ),
                _row(
                    period=period,
                    amount=internal_b,
                    position_type="internal_balance",
                    as_of_date=end,
                    party="Bob",
                    source_type="internal_party_balance",
                    suitability="internal_only",
                ),
                _row(
                    period=period,
                    amount=inferred,
                    position_type="inferred_box_motor",
                    as_of_date=end,
                    source_type="inferred_box_motor",
                    suitability="safe_with_caveat",
                ),
                _row(
                    period=period,
                    amount=bank_a,
                    position_type="cash_close",
                    as_of_date=end,
                    account_id="bank-a",
                    source_type="bank_statement",
                    safe=True,
                    suitability="frontend_safe",
                    validation_status="validated",
                    validated_by="controller",
                ),
                _row(
                    period=period,
                    amount=bank_b,
                    position_type="cash_close",
                    as_of_date=end,
                    account_id="bank-b",
                    source_type="bank_statement",
                    safe=True,
                    suitability="frontend_safe",
                    validation_status="validated",
                    validated_by="controller",
                ),
            ]
        )
    return pd.DataFrame(rows)


def test_shared_selector_separates_validated_cash_from_control_and_internal() -> None:
    cash = cash_fixture()
    selected = select_validated_cash_period(
        cash, period="2026-01", currency="ARS", box="Property Management"
    )
    assert selected.available
    assert selected.value == 100
    assert set(selected.selected["account_id"]) == {"bank-a", "bank-b"}
    assert set(selected.selected["position_type"]) == {"cash_close"}
    assert selected.excluded_inferred["close_amount"].sum() == 100
    assert selected.excluded_internal["close_amount"].sum() == 50

    control = select_inferred_box_control_period(
        cash, period="2026-01", currency="ARS", box="Property Management"
    )
    assert control.available
    assert control.value == 100
    assert set(control.selected["position_type"]) == {"inferred_box_motor"}


def test_validated_cash_fails_closed_on_incomplete_or_duplicate_account_snapshot() -> None:
    cash = cash_fixture()
    broken = cash.copy()
    mask = broken["account_id"].eq("bank-b") & broken["period"].eq("2026-01")
    broken.loc[mask, "as_of_date"] = "not-a-date"
    result = select_validated_cash_period(
        broken, period="2026-01", currency="ARS", box="Property Management"
    )
    assert result.status == "unavailable"
    assert result.reason == "candidate_account_has_no_valid_as_of"
    assert result.value is None

    duplicate = pd.concat(
        [cash, cash[cash["account_id"].eq("bank-a") & cash["period"].eq("2026-01")]],
        ignore_index=True,
    )
    result = select_validated_cash_period(
        duplicate, period="2026-01", currency="ARS", box="Property Management"
    )
    assert result.status == "unavailable"
    assert result.reason == "duplicate_latest_account_as_of"


def test_monthly_and_annual_use_identical_validated_population() -> None:
    cash = cash_fixture()
    monthly = select_validated_cash_period(
        cash, period="2026-01", currency="ARS", box="Property Management"
    )
    annual = select_validated_cash_year(
        cash, year="2026", currency="ARS", box="Property Management"
    )
    assert monthly.available and annual.available
    assert monthly.value == annual.value == 100
    assert annual.period == "2026-01"
    assert set(monthly.selected["account_id"]) == set(annual.selected["account_id"])


def test_professional_before_after_and_diagnostic_boundary() -> None:
    cash = cash_fixture()
    row = pd.Series(
        {"Currency": "ARS", "Box": "Property Management", "metric": "cash_close"}
    )
    before_month = legacy_cash_control(
        row=row,
        period="2026-01",
        display_value=200,
        source_df=cash,
        source_name="monthly_cash_close.csv",
        default_metric="cash_close",
        tolerance=1e-6,
    )
    assert before_month[1] == 200

    after_month = _build_derived_cell(
        table_id="monthly_tables_cash_close_matrix",
        row=row,
        period="2026-01",
        display_value=100,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=cash,
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert after_month[0] == "ok"
    assert after_month[1] == 100
    assert set(after_month[7]["account_id"]) == {"bank-a", "bank-b"}
    section_names = [name for name, _ in after_month[8]]
    assert "Excluded inferred control rows" in section_names
    assert "Excluded internal balance rows" in section_names

    annual_row = pd.Series({"Currency": "ARS", "Box": "Property Management"})
    before_annual = legacy_annual_cash(
        row=annual_row,
        period="2026",
        display_value=250,
        cash_close=cash,
        tolerance=1e-6,
    )
    assert before_annual[1] == 250
    after_annual = _build_derived_cell(
        table_id="annual_cash_close_by_box_wide",
        row=annual_row,
        period="2026",
        display_value=100,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=cash,
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert after_annual[0] == "ok"
    assert after_annual[1] == 100

    # Wave 5 boundary: diagnostic still delegates to the characterized legacy
    # period-delta implementation and therefore remains 40 for this fixture.
    diagnostic = _build_derived_cell(
        table_id="monthly_tables_diagnostic_box_level_matrix",
        row=pd.Series(
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "metric": "diagnostic_box_level",
            }
        ),
        period="2026-01",
        display_value=40,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=cash,
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert diagnostic[1] == 40


def test_frontier_monthly_cash_is_governed_and_not_double_counted(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"
    run_root.mkdir()
    cash_fixture().to_csv(run_root / "monthly_cash_close.csv", index=False)

    paths = build_metrics_frontier(run_root, metrics_dir, "fixture", "2026-01-31")
    series = pd.read_csv(paths["frontend_metric_series"])
    jan = series[
        series["period"].astype(str).eq("2026-01")
        & series["Currency"].astype(str).eq("ARS")
    ]
    total = jan[jan["metric_id"].eq("BS.CASH.TOTAL")]
    box = jan[jan["metric_id"].eq("BS.CASH.CLOSE.BOX")]
    assert len(total) == 1 and total.iloc[0]["value"] == 100
    assert len(box) == 1 and box.iloc[0]["value"] == 100

    frontier = pd.read_csv(paths["metric_contract_frontier"])
    cash_contract = frontier[frontier["metric_id"].eq("BS.CASH.TOTAL")].iloc[0]
    assert "cash.position.validated" in cash_contract["calculation_rule"]
    assert "inferred" in cash_contract["caveat"].lower()


def test_annual_metrics_and_companion_use_same_cash_selector(tmp_path: Path) -> None:
    cash = cash_fixture()
    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "annual"
    run_root.mkdir()
    cash.to_csv(run_root / "monthly_cash_close.csv", index=False)

    paths = build_annual_balance_dashboard(run_root, metrics_dir, "fixture", "2026-01-31")
    annual = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    box = annual[
        annual["metric_id"].eq("BS.CASH.CLOSE.BOX")
        & annual["period"].astype(str).eq("2026")
        & annual["Currency"].eq("ARS")
        & annual["dimension_value"].eq("Property Management")
    ]
    total = annual[
        annual["metric_id"].eq("BS.CASH.TOTAL")
        & annual["period"].astype(str).eq("2026")
        & annual["Currency"].eq("ARS")
    ]
    assert len(box) == 1 and box.iloc[0]["value"] == 100
    assert len(total) == 1 and total.iloc[0]["value"] == 100

    companion_long, _ = build_annual_cash_close_by_box(cash, year_columns=("2025", "2026"))
    companion = companion_long[
        companion_long["period"].astype(str).eq("2026")
        & companion_long["Currency"].eq("ARS")
        & companion_long["Box"].eq("Property Management")
    ]
    assert len(companion) == 1
    assert companion.iloc[0]["value"] == 100
    assert companion.iloc[0]["selected_month"] == "2026-01"


def test_no_validated_cash_never_falls_back_to_inferred() -> None:
    cash = cash_fixture()
    cash = cash.loc[~cash["position_type"].eq("cash_close")].copy()
    selected = select_validated_cash_period(
        cash, period="2026-01", currency="ARS", box="Property Management"
    )
    assert selected.status == "unavailable"
    assert selected.value is None
    assert selected.excluded_inferred["close_amount"].sum() == 100
