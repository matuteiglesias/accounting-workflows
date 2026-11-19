"""Time-series renderer is presentation-only, with strong population guards."""
from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.time_series import (
    TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)
from accounting.reports.annual_management.time_series_inputs import build_operating_chart_pair
from accounting.reports.debt_accountability.time_series_inputs import build_debt_position_chart


def _chart_spec(**overrides):
    params = {
        "chart_id": "fixture_line", "title": "Recorded history",
        "frequency": "M", "mark": "line", "unit": "USD",
        "scope": "FBPM", "series_ids": ("a",), "series_labels": ("Obligation A",),
    }
    params.update(overrides)
    return TimeSeriesSpec(**params)


def _points():
    return pd.DataFrame([
        {"period": p, "series_id": "a", "value": v, "unit": "USD",
         "scope": "FBPM", "phase": "observed", "source_table": "synthetic.csv"}
        for p, v in [("2026-01", 100), ("2026-02", 80), ("2026-03", 70)]
    ])


def test_reusable_svg_is_self_contained_and_has_exact_trace():
    spec = _chart_spec()
    svg, trace = render_timeseries_svg(spec, _points())
    assert '<svg class="governed-timeseries"' in svg
    assert "<script" not in svg and "http" not in svg
    assert len(trace) == 3 and trace.value.sum() == 250
    assert set(trace.chart_id) == {"fixture_line"}
    assert set(trace.source_table) == {"synthetic.csv"}
    assert "<figure" in timeseries_figure(spec, svg)


@pytest.mark.parametrize("column,value", [
    ("unit", "ARS"), ("scope", "Property Management"),
    ("phase", "estimated"), ("value", None), ("period", "2026-W02"),
    ("source_table", ""),
])
def test_renderer_rejects_unsafe_points(column, value):
    points = _points()
    points.loc[0, column] = value
    with pytest.raises(ValueError):
        render_timeseries_svg(_chart_spec(), points)


def test_renderer_rejects_duplicates_and_missing_stacked_components():
    points = pd.concat([_points(), _points().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        render_timeseries_svg(_chart_spec(), points)
    spec = _chart_spec(
        chart_id="stacked", frequency="Y", mark="stacked_bars", unit="ARS",
        series_ids=("tax", "service"), series_labels=("Tax", "Service"),
    )
    rows = pd.DataFrame([
        {"period": "2024", "series_id": "tax", "value": 10, "unit": "ARS",
         "scope": "FBPM", "phase": "observed", "source_table": "s"},
        {"period": "2025", "series_id": "tax", "value": 20, "unit": "ARS",
         "scope": "FBPM", "phase": "observed", "source_table": "s"},
    ])
    with pytest.raises(ValueError, match="all component"):
        render_timeseries_svg(spec, rows)


def _monthly_operating():
    rows = []
    for year, multiplier in [(2024, 1), (2025, 2), (2026, 3)]:
        for month in range(1, 12):
            period = f"{year}-{month:02d}"
            current = {
                "taxes": 5 * multiplier, "services": 3 * multiplier,
                "maintenance": multiplier, "legal": multiplier,
                "other_property_opex": 0,
                "property_opex_true": 10 * multiplier,
                "rent_revenue": 100 * multiplier,
            }
            # November is deliberately a large incomplete-month value.
            if month == 11:
                current = {k: v * 100 for k, v in current.items()}
            for line, value in current.items():
                rows.append({
                    "period": period, "Currency": "ARS",
                    "statement_line": line, "amount": value,
                    "classification_coverage_ratio": 1,
                    "unknown_amount": 0, "review_required_amount": 0,
                })
    return pd.DataFrame(rows)


def test_annual_cost_bars_and_ratio_use_same_complete_months():
    costs, ratio, trace = build_operating_chart_pair(
        _monthly_operating(), as_of_date="2026-11-19",
    )
    assert "¿Cómo evolucionaron" in costs and "¿Qué proporción" in ratio
    assert "enero–octubre" in costs
    assert set(trace.chart_id) == {"ts01_operating_costs", "ts02_opex_to_rent"}
    cost_points = trace.query("chart_id == 'ts01_operating_costs'")
    assert len(cost_points) == 15
    assert cost_points.query("period == '2026'").value.sum() == 300
    share_points = trace.query("chart_id == 'ts02_opex_to_rent'")
    assert share_points.value.tolist() == pytest.approx([.1, .1, .1])
    assert all(p in costs for p in ["2024", "2025", "2026"])
    assert "<script" not in costs


def test_annual_chart_refuses_nonreconciled_or_incomplete_source():
    rows = _monthly_operating()
    rows.loc[
        (rows.period.eq("2026-02")) & rows.statement_line.eq("taxes"), "amount"
    ] += 1
    with pytest.raises(ValueError, match="reconcile"):
        build_operating_chart_pair(rows, as_of_date="2026-11-19")

    rows = _monthly_operating()
    rows = rows.loc[~(
        rows.period.eq("2024-04") & rows.statement_line.eq("services")
    )]
    with pytest.raises(ValueError, match="missing source"):
        build_operating_chart_pair(rows, as_of_date="2026-11-19")


def test_debt_series_preserves_stock_and_uses_latest_leaders():
    rows = []
    for period, values in [
        ("2026-01", [90, 15, 2]),
        ("2026-02", [60, 30, 3]),
        ("2026-03", [50, 40, 4]),
    ]:
        for (debtor, creditor), value in zip(
            [("Alejandro", "PM"), ("PM", "MI"), ("Hector", "MI")], values
        ):
            rows.append({
                "period": period, "debtor": debtor, "creditor": creditor,
                "Currency": "USD", "component": "total",
                "position_status": "available", "open_amount": value,
            })
    doc, trace = build_debt_position_chart(pd.DataFrame(rows))
    assert 'data-chart-id="ts07_open_debt_usd"' in doc
    assert set(trace.series_id) == {"total_open", "Alejandro → PM", "PM → MI"}
    assert trace.query("series_id == 'total_open'").value.tolist() == [107, 93, 94]
    assert trace.query("period == '2026-03' and series_id == 'total_open'").value.iloc[0] == 94
    assert "Hector → MI" not in doc
    assert "nunca suma saldos entre meses" in doc


def test_debt_adapter_rejects_duplicated_relation_month():
    frame = pd.DataFrame([
        {"period": "2026-01", "debtor": "PM", "creditor": "MI", "Currency": "USD",
         "component": "total", "position_status": "available", "open_amount": 10},
        {"period": "2026-01", "debtor": "PM", "creditor": "MI", "Currency": "USD",
         "component": "total", "position_status": "available", "open_amount": 10},
    ])
    with pytest.raises(ValueError, match="duplicated"):
        build_debt_position_chart(frame)
