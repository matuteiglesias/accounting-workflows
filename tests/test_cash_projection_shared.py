from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

import accounting.metrics.annual as annual_module
import accounting.metrics.frontier as frontier_module
import accounting.professional.annual_dashboard_tables as professional_annual_module
from accounting.cash_authority import (
    select_validated_cash_period,
    select_validated_cash_year,
)
from accounting.cash_projection import (
    iter_validated_annual_cash_positions,
    iter_validated_monthly_cash_positions,
)
from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.metrics.frontier import build_metrics_frontier
from accounting.professional.annual_dashboard_tables import build_annual_cash_close_by_box


def _validated_row(
    *,
    period: str,
    currency: str,
    box: str,
    account_id: str,
    amount: float,
) -> dict[str, object]:
    as_of = f"{period}-28" if period.endswith("02") else f"{period}-31"
    return {
        "period": period,
        "period_end": as_of,
        "as_of_date": as_of,
        "Box": box,
        "party": "",
        "account_id": account_id,
        "account_name": account_id,
        "Currency": currency,
        "close_amount": amount,
        "source_table": "validated_cash_close.csv",
        "source_date": as_of,
        "source_type": "bank_statement",
        "source_reference": "phase5-fixture",
        "validation_status": "validated",
        "validated_by": "controller",
        "position_type": "cash_close",
        "cash_suitability": "frontend_safe",
        "is_frontend_safe": True,
        "caveat": "fixture",
        "notes": "",
        "n_source_rows": 1,
        "calculation_rule": "fixture",
    }


def disjoint_scope_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _validated_row(
                period="2025-12",
                currency="ARS",
                box="Property Management",
                account_id="ars-bank",
                amount=120.0,
            ),
            _validated_row(
                period="2026-01",
                currency="USD",
                box="Family Box",
                account_id="usd-bank",
                amount=7.0,
            ),
        ]
    )


def test_shared_projection_enumerates_only_source_backed_scopes() -> None:
    cash = disjoint_scope_fixture()

    monthly = list(iter_validated_monthly_cash_positions(cash))
    monthly_keys = [
        (p.reporting_period, p.scope, p.currency, p.box) for p in monthly
    ]
    assert monthly_keys == [
        ("2025-12", "currency", "ARS", ""),
        ("2025-12", "box", "ARS", "Property Management"),
        ("2026-01", "currency", "USD", ""),
        ("2026-01", "box", "USD", "Family Box"),
    ]

    annual = list(iter_validated_annual_cash_positions(cash))
    annual_keys = [
        (p.reporting_period, p.scope, p.currency, p.box) for p in annual
    ]
    assert annual_keys == [
        ("2025", "currency", "ARS", ""),
        ("2025", "box", "ARS", "Property Management"),
        ("2026", "currency", "USD", ""),
        ("2026", "box", "USD", "Family Box"),
    ]

    # No report-layer Cartesian combinations such as 2025/USD or 2026/ARS.
    assert not any(p.reporting_period == "2025" and p.currency == "USD" for p in annual)
    assert not any(p.reporting_period == "2026" and p.currency == "ARS" for p in annual)


def test_shared_projection_is_only_scope_wiring_over_cash_authority() -> None:
    cash = disjoint_scope_fixture()
    monthly = list(iter_validated_monthly_cash_positions(cash))
    annual = list(iter_validated_annual_cash_positions(cash))

    monthly_box = next(
        p
        for p in monthly
        if p.scope == "box" and p.reporting_period == "2025-12"
    )
    direct_monthly = select_validated_cash_period(
        cash,
        period="2025-12",
        currency="ARS",
        box="Property Management",
    )
    assert monthly_box.selection.status == direct_monthly.status == "available"
    assert monthly_box.value == direct_monthly.value == 120.0
    assert set(monthly_box.selection.selected["account_id"]) == set(
        direct_monthly.selected["account_id"]
    )

    annual_box = next(
        p for p in annual if p.scope == "box" and p.reporting_period == "2026"
    )
    direct_annual = select_validated_cash_year(
        cash,
        year="2026",
        currency="USD",
        box="Family Box",
    )
    assert annual_box.selection.status == direct_annual.status == "available"
    assert annual_box.value == direct_annual.value == 7.0
    assert annual_box.selected_period == direct_annual.period == "2026-01"


def test_annual_frontier_and_professional_project_same_source_population(
    tmp_path: Path,
) -> None:
    cash = disjoint_scope_fixture()
    run_root = tmp_path / "run"
    run_root.mkdir()
    cash.to_csv(run_root / "monthly_cash_close.csv", index=False)

    annual_paths = build_annual_balance_dashboard(
        run_root,
        tmp_path / "annual",
        "phase5",
        "2026-01-31",
    )
    annual = pd.read_csv(annual_paths["annual_balance_dashboard_metrics"])
    annual_cash = annual[
        annual["metric_id"].isin({"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"})
    ].copy()
    annual_cash["period"] = annual_cash["period"].astype(str)
    assert set(zip(annual_cash["period"], annual_cash["Currency"])) == {
        ("2025", "ARS"),
        ("2026", "USD"),
    }
    assert not (
        annual_cash["period"].eq("2025") & annual_cash["Currency"].eq("USD")
    ).any()
    assert not (
        annual_cash["period"].eq("2026") & annual_cash["Currency"].eq("ARS")
    ).any()

    companion_long, _ = build_annual_cash_close_by_box(
        cash,
        year_columns=("2025", "2026"),
    )
    assert set(
        zip(
            companion_long["period"].astype(str),
            companion_long["Currency"],
            companion_long["Box"],
            companion_long["value"],
        )
    ) == {
        ("2025", "ARS", "Property Management", 120.0),
        ("2026", "USD", "Family Box", 7.0),
    }

    frontier_paths = build_metrics_frontier(
        run_root,
        tmp_path / "frontier",
        "phase5",
        "2026-01-31",
    )
    series = pd.read_csv(frontier_paths["frontend_metric_series"])
    cash_series = series[
        series["metric_id"].isin({"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"})
    ].copy()
    expected = {
        (p.reporting_period, p.currency, p.scope, p.box, float(p.value))
        for p in iter_validated_monthly_cash_positions(cash)
        if p.available
    }
    got = {
        (
            str(row.period),
            str(row.Currency),
            "box" if row.metric_id == "BS.CASH.CLOSE.BOX" else "currency",
            "" if pd.isna(row.dimension_value) else str(row.dimension_value),
            float(row.value),
        )
        for row in cash_series.itertuples(index=False)
    }
    assert got == expected


def test_bulk_cash_consumers_delegate_scope_discovery_to_projection_module() -> None:
    annual_source = inspect.getsource(annual_module)
    frontier_source = inspect.getsource(frontier_module)
    professional_source = inspect.getsource(professional_annual_module)

    assert "iter_validated_annual_cash_positions" in annual_source
    assert "select_validated_cash_year" not in annual_source

    assert "iter_validated_monthly_cash_positions" in frontier_source
    assert "select_validated_cash_period" not in frontier_source

    assert "iter_validated_annual_cash_positions" in professional_source
    assert "select_validated_cash_year" not in professional_source
