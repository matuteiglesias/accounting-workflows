import pandas as pd
import pytest

from accounting.reports.charts import PieSpec, professional_support_view, render_pie_svg


def _spec(**values):
    base = dict(chart_id="x", source_metric="M", measure="value", slice_dimension="actor", currency="ARS", scope="FBPM", period_basis="annual", period="2026", title="Test")
    base.update(values)
    return PieSpec(**base)


def _rows(**values):
    base = {"actor": "A", "value": 60, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"}
    rows = [base.copy(), {**base, "actor": "B", "value": 40}]
    for row in rows:
        row.update(values)
    return pd.DataFrame(rows)


def test_pie_reconciles_and_has_inline_svg_trace():
    svg, trace = render_pie_svg(_spec(), _rows(), 100)
    assert "<svg" in svg and "<script" not in svg
    assert trace["value"].sum() == 100
    assert trace["denominator"].eq(100).all()


@pytest.mark.parametrize("bad", [
    {"Currency": "USD"},
    {"value": -1},
])
def test_pie_rejects_mixed_currency_or_negative_values(bad):
    rows = _rows()
    rows.loc[1, list(bad)] = list(bad.values())
    with pytest.raises(ValueError):
        render_pie_svg(_spec(), rows, 100)


def test_support_reporting_group_remap_preserves_denominator():
    support = pd.DataFrame([
        {"period": "2026-01", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "Actor A", "reporting_group": "Actor A", "recognized_amount": 60},
        {"period": "2026-02", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "Actor B", "reporting_group": "Actor B", "recognized_amount": 40},
    ])
    view = professional_support_view(support)
    before = view.value.sum()
    view["reporting_group"] = view["funding_actor"].map({"Actor A": "FB", "Actor B": "FB"})
    assert view.value.sum() == before
