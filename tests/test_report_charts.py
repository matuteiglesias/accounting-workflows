import re

import pandas as pd
import pytest

from accounting.reports.charts import (
    PieSpec,
    build_stable_color_map,
    professional_fb_receipts_view,
    professional_rent_receipts_view,
    professional_support_view,
    professional_tax_service_payment_view,
    professional_tax_service_support_view,
    render_pie_svg,
)


def _spec(**values):
    base = dict(
        chart_id="x",
        source_metric="M",
        measure="value",
        slice_dimension="actor",
        currency="ARS",
        scope="FBPM",
        period_basis="annual",
        period="2026",
        title="Test",
    )
    base.update(values)
    return PieSpec(**base)


def _rows(**values):
    base = {
        "actor": "A",
        "value": 60,
        "Currency": "ARS",
        "scope": "FBPM",
        "period_basis": "annual",
        "period": "2026",
    }
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


def test_pie_orders_geometry_legend_and_trace_by_value_then_identity():
    rows = pd.DataFrame([
        {"actor": "C", "value": 60, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"},
        {"actor": "A", "value": 20, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"},
        {"actor": "B", "value": 60, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"},
    ])
    svg, trace = render_pie_svg(_spec(max_slices=3), rows, 140)
    assert trace["slice_key"].tolist() == ["B", "C", "A"]
    assert svg.index(">B</text>") < svg.index(">C</text>") < svg.index(">A</text>")


def test_default_colors_are_stable_by_identity_when_rank_changes():
    family = pd.DataFrame([
        {"actor": "A", "value": 90, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2025"},
        {"actor": "B", "value": 10, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2025"},
        {"actor": "A", "value": 5, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"},
        {"actor": "B", "value": 95, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"},
    ])
    trace_2025 = render_pie_svg(_spec(period="2025"), family.loc[family.period.eq("2025")], 100)[1]
    trace_2026 = render_pie_svg(_spec(period="2026"), family.loc[family.period.eq("2026")], 100)[1]
    for actor in ["A", "B"]:
        color_2025 = trace_2025.loc[trace_2025.slice_key.eq(actor), "color"].iloc[0]
        color_2026 = trace_2026.loc[trace_2026.slice_key.eq(actor), "color"].iloc[0]
        assert color_2025 == color_2026


def test_optional_family_color_map_is_deterministic():
    family = pd.DataFrame([
        {"actor": "A", "value": 90},
        {"actor": "B", "value": 10},
        {"actor": "A", "value": 5},
        {"actor": "B", "value": 95},
    ])
    assert build_stable_color_map(family, "actor") == build_stable_color_map(family, "actor")


def test_pie_dynamic_height_keeps_long_legend_and_total_inside_viewbox():
    rows = pd.DataFrame([
        {"actor": f"Actor {index:02d}", "value": index + 1, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"}
        for index in range(12)
    ])
    svg, trace = render_pie_svg(_spec(max_slices=12), rows, float(rows["value"].sum()))
    viewbox = re.search(r'viewBox="0 0 520 (\d+)"', svg)
    total_y = re.search(r'<text x="210" y="(\d+)" class="pie-total">', svg)
    legend_y = [int(value) for value in re.findall(r'<g transform="translate\(210,(\d+)\)">', svg)]
    assert viewbox and total_y and len(legend_y) == 12
    height = int(viewbox.group(1))
    assert height > 245
    assert max(legend_y) + 21 < int(total_y.group(1)) < height
    assert len(trace) == 12


def test_one_slice_population_uses_compact_fallback_without_pie_geometry():
    rows = pd.DataFrame([
        {"actor": "Only", "value": 60, "Currency": "ARS", "scope": "FBPM", "period_basis": "annual", "period": "2026"}
    ])
    svg, trace = render_pie_svg(_spec(max_slices=1), rows, 60)
    assert 'class="governed-pie pie-single"' in svg
    assert "<path" not in svg
    assert "Only" in svg and "100.0%" in svg and "Total: ARS 60" in svg
    assert trace["slice_key"].tolist() == ["Only"]


def test_tax_service_spec_uses_source_of_coverage_wording():
    spec = PieSpec(
        chart_id="tax",
        source_metric="TAX_SERVICE.PAYMENTS.BY_ACTOR",
        measure="value",
        slice_dimension="funding_actor",
        currency="ARS",
        scope="FBPM",
        period_basis="annual",
        period="2026",
        title="Impuestos y servicios pagados o aplicados por actor · 2026 YTD · corte 31/08/2026",
    )
    assert spec.title == "Impuestos y servicios por fuente de cobertura · 2026 YTD · corte 31/08/2026"


def test_support_reporting_group_remap_preserves_denominator():
    support = pd.DataFrame([
        {"period": "2026-01", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "Actor A", "reporting_group": "Actor A", "recognized_amount": 60},
        {"period": "2026-02", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "Actor B", "reporting_group": "Actor B", "recognized_amount": 40},
    ])
    view = professional_support_view(support)
    before = view.value.sum()
    view["reporting_group"] = view["funding_actor"].map({"Actor A": "FB", "Actor B": "FB"})
    assert view.value.sum() == before


def test_support_view_merges_confirmed_hector_spelling_without_total_change():
    support = pd.DataFrame([
        {"period": "2026-01", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "Hector", "reporting_group": "Hector", "recognized_amount": 60},
        {"period": "2026-02", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "Héctor", "reporting_group": "Héctor", "recognized_amount": 40},
    ])
    view = professional_support_view(support)
    assert view["funding_actor"].tolist() == ["Héctor"]
    assert view["value"].sum() == 100


def test_fb_receipts_view_is_cash_receipt_population_not_distribution():
    source = pd.DataFrame([
        {"Date": "2026-01-01", "Box": "Family Business", "Currency": "ARS", "direction": "in", "cash_effect": "cash_in_box", "semantic_subbucket": "rent", "amount": 100},
        {"Date": "2026-01-02", "Box": "Family Business", "Currency": "ARS", "direction": "in", "cash_effect": "cash_in_box", "semantic_subbucket": "rent", "amount": 50},
        {"Date": "2026-01-03", "Box": "Property Management", "Currency": "ARS", "direction": "in", "cash_effect": "cash_in_box", "semantic_subbucket": "rent", "amount": 999},
    ])
    view = professional_fb_receipts_view(source)
    assert view["value"].sum() == 150
    assert set(view["receipt_nature"]) == {"rent"}


def test_tax_service_support_view_excludes_non_service_support():
    source = pd.DataFrame([
        {"period": "2026-01", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "A", "obligation_category": "taxes", "recognized_amount": 60},
        {"period": "2026-01", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "A", "obligation_category": "services", "recognized_amount": 40},
        {"period": "2026-01", "Currency": "ARS", "target_box": "Property Management", "funding_actor": "A", "obligation_category": "maintenance", "recognized_amount": 999},
    ])
    view = professional_tax_service_support_view(source)
    assert view["value"].sum() == 100


def test_tax_service_payment_view_merges_hector_and_preserves_unidentified_tenant_and_box_labels():
    source = pd.DataFrame([
        {"Date": "2026-01-01", "Box": "Property Management", "Currency": "ARS", "semantic_subbucket": "taxes", "cash_effect": "no_cash_out_box_direct_payment", "leg_role": "stakeholder_direct_expense", "funding_actor": "Hector", "amount": 60},
        {"Date": "2026-01-02", "Box": "Property Management", "Currency": "ARS", "semantic_subbucket": "services", "cash_effect": "no_cash_out_box_direct_payment", "leg_role": "stakeholder_direct_expense", "funding_actor": "Héctor", "amount": 40},
        {"Date": "2026-01-03", "Box": "Property Management", "Currency": "ARS", "semantic_subbucket": "services", "cash_effect": "no_cash_out_box_direct_payment", "leg_role": "stakeholder_direct_expense", "funding_actor": "Inquilino", "amount": 30},
        {"Date": "2026-01-04", "Box": "Property Management", "Currency": "ARS", "semantic_subbucket": "taxes", "cash_effect": "cash_out_box", "leg_role": "box_cash_expense", "funding_actor": "", "amount": 20},
    ])
    view = professional_tax_service_payment_view(source)
    assert view["value"].sum() == 150
    assert set(view["funding_actor"]) == {"Héctor", "Inquilino", "Property Management"}
    labels = dict(zip(view["funding_actor"], view["display_label"]))
    assert labels["Héctor"] == "Héctor"
    assert labels["Inquilino"] == "Inquilino no identificado"
    assert labels["Property Management"] == "Caja PM"


def test_rent_receipts_view_keeps_both_accounting_boxes_separate():
    source = pd.DataFrame([
        {"Date": "2026-01-01", "Box": "Family Business", "Currency": "ARS", "direction": "in", "cash_effect": "cash_in_box", "semantic_subbucket": "rent", "amount": 100},
        {"Date": "2026-01-02", "Box": "Property Management", "Currency": "ARS", "direction": "in", "cash_effect": "cash_in_box", "semantic_subbucket": "rent", "amount": 40},
    ])
    view = professional_rent_receipts_view(source)
    assert view["value"].sum() == 140
    assert set(view["box"]) == {"Family Business", "Property Management"}
