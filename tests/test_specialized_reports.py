from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.specialized.render import render_specialized
from accounting.reports.specialized.spec import REPORT_SPECS
from accounting.reports.specialized.views import build_specialized_view, view_is_available


def _semantic_split() -> pd.DataFrame:
    return pd.DataFrame([
        {"period":"2026-01","Currency":"ARS","Box":"Family Business","Lugar":"Site A","semantic_bucket":"operating_revenue","semantic_subbucket":"rent","amount_in":100,"amount_out":0},
        {"period":"2026-02","Currency":"ARS","Box":"Family Business","Lugar":"Site B","semantic_bucket":"operating_revenue","semantic_subbucket":"rent","amount_in":50,"amount_out":0},
        {"period":"2026-01","Currency":"ARS","Box":"Property Management","Lugar":"Site A","semantic_bucket":"property_opex","semantic_subbucket":"taxes","amount_in":0,"amount_out":30},
        {"period":"2026-02","Currency":"ARS","Box":"Property Management","Lugar":"Site B","semantic_bucket":"property_opex","semantic_subbucket":"services","amount_in":0,"amount_out":20},
    ])


def _annual_metrics() -> pd.DataFrame:
    base = {"period":"2026","Currency":"ARS","value_status":"available"}
    return pd.DataFrame([
        base | {"metric_id":"IS.RENT.BY_PROPERTY","dimension_name":"Lugar","dimension_value":"Site A","value":100,"source_table":"monthly_flow_semantic_split.csv","source_filter":"rent; Lugar=Site A","calculation_rule":"governed annual rent by property"},
        base | {"metric_id":"IS.RENT.BY_PROPERTY","dimension_name":"Lugar","dimension_value":"Site B","value":50,"source_table":"monthly_flow_semantic_split.csv","source_filter":"rent; Lugar=Site B","calculation_rule":"governed annual rent by property"},
        base | {"metric_id":"IS.RENT.TOTAL","dimension_name":"","dimension_value":"","value":150,"source_table":"annual_flow_membership.csv","source_filter":"rent","calculation_rule":"governed annual rent total"},
        base | {"metric_id":"IS.OPEX.BY_CATEGORY","dimension_name":"semantic_subbucket","dimension_value":"taxes","value":30,"source_table":"monthly_flow_semantic_split.csv","source_filter":"property_opex; taxes","calculation_rule":"governed annual OPEX by category"},
        base | {"metric_id":"IS.OPEX.BY_CATEGORY","dimension_name":"semantic_subbucket","dimension_value":"services","value":20,"source_table":"monthly_flow_semantic_split.csv","source_filter":"property_opex; services","calculation_rule":"governed annual OPEX by category"},
    ])


def _write_round1_sources(run: Path, metrics: Path) -> None:
    _semantic_split().to_csv(run / "monthly_flow_semantic_split.csv", index=False)
    _annual_metrics().to_csv(metrics / "annual_balance_dashboard_metrics.csv", index=False)


def test_specialized_vertical_has_explicit_recipe_seam_and_round1_reports():
    ids = [spec.report_id for spec in REPORT_SPECS]
    assert ids[:4] == [
        "pm_tax_accountability",
        "pm_services_accountability",
        "stakeholder_support",
        "distributions_by_recipient",
    ]
    assert ids[4:] == [
        "rent_by_property",
        "rent_monthly_evolution",
        "opex_by_category",
        "taxes_by_property",
        "services_by_property",
        "distributions_vs_rent",
    ]
    for spec in REPORT_SPECS:
        assert spec.question and spec.establishes and spec.caveat
        assert spec.view_key and spec.scope and spec.period_policy and spec.currency_policy
        assert spec.section_plan


def test_round1_semantic_views_reconcile_governed_sources(tmp_path: Path):
    run = tmp_path / "run"
    metrics = tmp_path / "metrics"
    run.mkdir(); metrics.mkdir()
    _write_round1_sources(run, metrics)

    rent = build_specialized_view("rent_by_property", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    opex = build_specialized_view("opex_by_category", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    taxes = build_specialized_view("taxes_by_property", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    services = build_specialized_view("services_by_property", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    monthly = build_specialized_view("rent_monthly_evolution", run_root=run, metrics_dir=metrics, scope="FBPM").frame

    assert rent["value"].sum() == 150
    assert opex["value"].sum() == 50
    assert taxes["value"].sum() == 30
    assert services["value"].sum() == 20
    assert monthly["value"].sum() == 150
    assert set(rent["property"]) == {"Site A", "Site B"}
    assert set(opex["category"]) == {"Impuestos", "Servicios"}
    assert set(rent["metric_id"]) == {"IS.RENT.BY_PROPERTY"}
    assert set(opex["metric_id"]) == {"IS.OPEX.BY_CATEGORY"}


def test_split_based_round1_views_fail_closed_on_annual_reconciliation_gap(tmp_path: Path):
    run = tmp_path / "run"
    metrics = tmp_path / "metrics"
    run.mkdir(); metrics.mkdir()
    _write_round1_sources(run, metrics)
    bad = _annual_metrics()
    bad.loc[bad["metric_id"].eq("IS.RENT.TOTAL"), "value"] = 149
    bad.to_csv(metrics / "annual_balance_dashboard_metrics.csv", index=False)

    with pytest.raises(ValueError, match="monthly rent -> IS.RENT.TOTAL"):
        build_specialized_view("rent_monthly_evolution", run_root=run, metrics_dir=metrics, scope="FBPM")

    bad = _annual_metrics()
    bad.loc[
        bad["metric_id"].eq("IS.OPEX.BY_CATEGORY") & bad["dimension_value"].eq("taxes"),
        "value",
    ] = 29
    bad.to_csv(metrics / "annual_balance_dashboard_metrics.csv", index=False)
    with pytest.raises(ValueError, match="taxes by property"):
        build_specialized_view("taxes_by_property", run_root=run, metrics_dir=metrics, scope="FBPM")


def test_specialized_view_availability_is_per_recipe(tmp_path: Path):
    run = tmp_path / "run"
    metrics = tmp_path / "metrics"
    run.mkdir(); metrics.mkdir()
    _semantic_split().to_csv(run / "monthly_flow_semantic_split.csv", index=False)

    assert not view_is_available("rent_monthly_evolution", run, metrics)
    assert not view_is_available("taxes_by_property", run, metrics)
    assert not view_is_available("rent_by_property", run, metrics)
    assert not view_is_available("pm_support_by_actor", run, metrics)
    assert not view_is_available("distributions_vs_rent", run, metrics)

    _annual_metrics().to_csv(metrics / "annual_balance_dashboard_metrics.csv", index=False)
    assert view_is_available("rent_monthly_evolution", run, metrics)
    assert view_is_available("taxes_by_property", run, metrics)
    assert view_is_available("rent_by_property", run, metrics)
    assert view_is_available("opex_by_category", run, metrics)


def test_round1_renderer_is_self_contained_and_cutoff_driven(tmp_path: Path):
    run = tmp_path / "run"
    metrics = tmp_path / "metrics"
    out = tmp_path / "out"
    run.mkdir(); metrics.mkdir()
    _write_round1_sources(run, metrics)

    outputs = render_specialized(
        report_id="rent_by_property",
        run_root=run,
        metrics_dir=metrics,
        out_dir=out,
        as_of_date="2026-08-31",
        require_pdf=False,
    )
    document = outputs["html"].read_text(encoding="utf-8")
    trace = pd.read_csv(outputs["trace"])

    assert "31/08/2026" in document
    assert "Renta por inmueble / fuente" in document
    assert "Qué establece:" in document and "Qué no establece:" in document
    assert "<style>" in document and '<link rel="stylesheet"' not in document
    assert trace["value"].sum() == 150
    assert set(trace["slice_key"]) == {"Site A", "Site B"}


def test_pm_tax_report_selector_does_not_leak_fb_tax_rows(tmp_path: Path):
    run = tmp_path / "run"
    metrics = tmp_path / "metrics"
    run.mkdir(); metrics.mkdir()
    pd.DataFrame([
        {"Date":"2026-01-01","Box":"Property Management","Currency":"ARS","semantic_subbucket":"taxes","cash_effect":"cash_out_box","leg_role":"box_cash_expense","funding_actor":"","amount":30},
        {"Date":"2026-01-02","Box":"Family Business","Currency":"ARS","semantic_subbucket":"taxes","cash_effect":"cash_out_box","leg_role":"box_cash_expense","funding_actor":"","amount":70},
    ]).to_csv(run / "classification_audit.csv", index=False)

    view = build_specialized_view("pm_tax_by_actor", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert view["value"].sum() == 30
    assert set(view["funding_actor"]) == {"Property Management"}
    assert set(view["metric_id"]) == {"PM.TAXES.COVERAGE.BY_ACTOR"}


def test_distributions_vs_rent_keeps_measures_separate(tmp_path: Path):
    run = tmp_path / "run"
    metrics = tmp_path / "metrics"
    run.mkdir(); metrics.mkdir()
    pd.DataFrame([
        {"Date":"2026-01-10","semantic_bucket":"family_withdrawal_candidate","Box":"Family Business","receiver":"Actor A","amount":40,"Currency":"ARS"},
        {"Date":"2026-02-10","semantic_bucket":"family_withdrawal","Box":"Family Business","receiver":"Actor B","amount":10,"Currency":"ARS"},
    ]).to_csv(run / "classification_audit.csv", index=False)
    _annual_metrics().to_csv(metrics / "annual_balance_dashboard_metrics.csv", index=False)

    view = build_specialized_view("distributions_vs_rent", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    values = dict(zip(view["concept"], view["value"]))
    assert values == {"Renta reconocida": 150, "Distribuciones registradas": 50}
    assert view["calculation_rule"].str.contains("no netting").all()
