"""Contract: historical time-series axes AND relevant tables start January 2022."""
from __future__ import annotations

import pandas as pd

from accounting.reports.time_series import HISTORY_START_MONTH
from accounting.reports.annual_management.distribution_time_series import operating_vs_distributions_chart
from accounting.reports.debt_accountability.time_series_inputs import (
    build_debt_position_chart, monthly_debt_position_table,
)
from accounting.reports.specialized.render import _period_selection, _monthly_history_table
from accounting.reports.specialized.time_series_inputs import monthly_rent_chart
from accounting.reports.treasury_accountability.render import complete_calendar, _svg_chart, _table_html
from accounting.reports.treasury_accountability.time_series_inputs import cash_inout_chart


def test_contract_boundary():
    assert HISTORY_START_MONTH == "2022-01"


def test_specialized_month_selection_is_full_history_with_missing_opex_explained():
    source = pd.DataFrame([
        {"period": "2022-02", "Currency": "ARS", "value": 20, "month": "2022-02",
         "scope": "FBPM", "period_basis": "monthly", "source_table": "governed.csv"},
        {"period": "2026-01", "Currency": "ARS", "value": 100, "month": "2026-01",
         "scope": "FBPM", "period_basis": "monthly", "source_table": "governed.csv"},
        {"period": "2026-11", "Currency": "ARS", "value": 30, "month": "2026-11",
         "scope": "FBPM", "period_basis": "monthly", "source_table": "governed.csv"},
    ])
    selected, label = _period_selection(source, "all_history_months", "2026-11-19")
    assert selected.period.tolist() == ["2022-02", "2026-01", "2026-11"]
    assert "2022-01" in label
    doc = _monthly_history_table(selected, "ARS", "2026-11-19", "OPEX")
    assert '<td>2022-01</td><td>Sin registro</td><td>Sin dato</td>' in doc
    assert '<td>2022-02</td><td>ARS 20,00</td><td>Registrado</td>' in doc
    assert '<td>2026-11</td>' in doc
    html, trace = monthly_rent_chart(
        selected, currency="ARS", as_of_date="2026-11-19", metric="opex"
    )
    assert "22/01" in html  # complete axis even before first OPEX record
    assert "2022-01" not in set(trace.period)  # no fabricated value
    assert "2026-11" not in set(trace.period)  # partial month excluded from line
    assert trace.period.tolist() == ["2022-02", "2026-01"]


def test_treasury_full_calendar_and_plot_do_not_forge_pre_activity():
    # The official complete_calendar zero-origin behavior is explicit and
    # tagged, and the old legacy line starts at the first recorded transaction.
    original = pd.DataFrame([
        {"period": "2024-05", "Box": "Property Management", "Currency": "ARS",
         "net_cash_flow": 100, "closing_control": 100,
         "opening_control": 0, "total_cash_in": 100, "total_cash_out": 0,
         "reconciliation_status": "reconciled", "reconciliation_gap": 0, "n_tx": 1,
         "n_review_required": 0},
        {"period": "2024-06", "Box": "Property Management", "Currency": "ARS",
         "net_cash_flow": -30, "closing_control": 70,
         "opening_control": 100, "total_cash_in": 0, "total_cash_out": 30,
         "reconciliation_status": "reconciled", "reconciliation_gap": 0, "n_tx": 1,
         "n_review_required": 0},
    ])
    group = complete_calendar(original, "2022-01", "2024-06")
    assert group.period.iloc[0] == "2022-01"
    assert group.reconciliation_status.iloc[0] == "calendar_zero"
    assert group.reconciliation_status.iloc[-1] == "reconciled"
    doc = _table_html(group)
    assert '<td class="month">2022-01</td>' in doc
    assert 'class="pre-activity"' in doc
    chart = _svg_chart(group, "ARS")
    assert chart.count('class="cash-chart"') == 1
    # There are no pre-inception bars.
    assert chart.count("<rect ") == 3  # 2 actual bars plus one legend glyph
    cash_doc, trace = cash_inout_chart(group, as_of_date="2024-06-30")
    assert "22/01" in cash_doc
    assert "2022-01" not in set(trace.period)
    assert trace.period.unique().tolist() == ["2024-05", "2024-06"]


def test_operating_line_has_full_calendar_domain_not_12_month_tail():
    rows = []
    for period, net, draw in [("2024-01", 130, 0), ("2024-02", 80, 5)]:
        for category, amount in (("net_operating", net), ("family_draws_or_distributions", draw)):
            rows.append({"period": period, "Currency": "ARS", "statement_line": category,
                         "amount": amount, "classification_coverage_ratio": 1,
                         "unknown_amount": 0, "review_required_amount": 0})
    html, trace = operating_vs_distributions_chart(
        pd.DataFrame(rows), as_of_date="2024-03-05"
    )
    assert "22/01" in html
    assert "2022-01" not in set(trace.period)
    assert trace.period.unique().tolist() == ["2024-01", "2024-02"]


def test_debt_monthly_table_from_2022_shows_unavailable_prefix():
    positions = pd.DataFrame([
        {"period": "2023-05", "debtor": "PM", "creditor": "MI", "Currency": "USD",
         "component": "total", "position_status": "available", "open_amount": 80},
        {"period": "2023-06", "debtor": "PM", "creditor": "MI", "Currency": "USD",
         "component": "total", "position_status": "available", "open_amount": 75},
    ])
    html, trace = build_debt_position_chart(positions, currency="USD")
    assert "22/01" in html
    assert trace.period.unique().tolist() == ["2023-05", "2023-06"]
    table, table_trace = monthly_debt_position_table(positions)
    assert '<td>2022-01</td><td class="num">Sin registro</td><td>Sin dato</td>' in table
    assert '<td>2023-05</td><td class="num">USD 80,00</td><td>Observado</td>' in table
    assert table_trace.iloc[0].status == "unavailable"
    assert pd.isna(table_trace.iloc[0].open_total)
    assert table_trace.loc[table_trace.period.eq("2023-06"), "open_total"].iloc[0] == 75
