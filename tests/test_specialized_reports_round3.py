from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.specialized.render import render_specialized
from accounting.reports.specialized.spec import REPORT_SPECS
from accounting.reports.specialized.views import build_specialized_view, view_is_available


def _write_round3_sources(run: Path, metrics: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"period":"2026-01","Currency":"ARS","Lugar":"A","semantic_bucket":"property_opex","semantic_subbucket":"maintenance","amount_in":0,"amount_out":30},
        {"period":"2026-02","Currency":"ARS","Lugar":"B","semantic_bucket":"property_opex","semantic_subbucket":"maintenance","amount_in":0,"amount_out":40},
        {"period":"2026-03","Currency":"ARS","Lugar":"A","semantic_bucket":"property_opex","semantic_subbucket":"legal","amount_in":0,"amount_out":50},
    ]).to_csv(run / "monthly_flow_semantic_split.csv", index=False)
    pd.DataFrame([
        {"period":"2026","Currency":"ARS","metric_id":"IS.OPEX.PROPERTY","dimension_name":"","dimension_value":"","value":120,"value_status":"available"},
        {"period":"2026","Currency":"ARS","metric_id":"IS.OPEX.BY_CATEGORY","dimension_name":"semantic_subbucket","dimension_value":"maintenance","value":70,"value_status":"available"},
        {"period":"2026","Currency":"ARS","metric_id":"IS.OPEX.BY_CATEGORY","dimension_name":"semantic_subbucket","dimension_value":"legal","value":50,"value_status":"available"},
    ]).to_csv(metrics / "annual_balance_dashboard_metrics.csv", index=False)
    pd.DataFrame([
        {"period":"2026-01","Currency":"ARS","target_box":"Property Management","funding_actor":"Ana","recognized_amount":80,"obligation_category":"taxes","funding_channel":"direct_payment","settlement_nature":"current_period_support"},
        {"period":"2026-02","Currency":"ARS","target_box":"Property Management","funding_actor":"Beto","recognized_amount":20,"obligation_category":"services","funding_channel":"constructive_pair","settlement_nature":"prior_period_clearing"},
        {"period":"2026-02","Currency":"ARS","target_box":"Household","funding_actor":"Ana","recognized_amount":999,"obligation_category":"services","funding_channel":"direct_payment","settlement_nature":"current_period_support"},
    ]).to_csv(run / "monthly_stakeholder_support.csv", index=False)
    pd.DataFrame([
        {"period":"2026-01","Box":"Property Management","Currency":"ARS","movement_basis":"actual_cash","cash_direction":"in","cash_category":"rent","amount_in":100,"amount_out":0},
        {"period":"2026-01","Box":"Family Business","Currency":"ARS","movement_basis":"actual_cash","cash_direction":"in","cash_category":"funding","amount_in":20,"amount_out":0},
        {"period":"2026-01","Box":"Property Management","Currency":"ARS","movement_basis":"actual_cash","cash_direction":"out","cash_category":"taxes","amount_in":0,"amount_out":30},
        {"period":"2026-01","Box":"Family Business","Currency":"ARS","movement_basis":"actual_cash","cash_direction":"out","cash_category":"personal_draws","amount_in":0,"amount_out":10},
        {"period":"2026-01","Box":"Property Management","Currency":"ARS","movement_basis":"non_cash_support","cash_direction":"out","cash_category":"services","amount_in":0,"amount_out":0},
    ]).to_csv(run / "monthly_box_treasury_flow.csv", index=False)
    pd.DataFrame([
        {"period":"2026-01","Box":"Property Management","Currency":"ARS","total_cash_in":100,"total_cash_out":30,"other_cash_in":5,"unknown_cash_in":0,"other_cash_out":0,"unknown_cash_out":2},
        {"period":"2026-01","Box":"Family Business","Currency":"ARS","total_cash_in":20,"total_cash_out":10,"other_cash_in":0,"unknown_cash_in":1,"other_cash_out":3,"unknown_cash_out":0},
    ]).to_csv(run / "monthly_cash_accountability.csv", index=False)


def test_round3_reports_are_registered_with_explicit_boundaries() -> None:
    ids = {spec.report_id for spec in REPORT_SPECS}
    expected = {
        "opex_monthly_evolution",
        "maintenance_by_property",
        "legal_costs_by_property",
        "support_by_obligation_category",
        "support_by_funding_channel",
        "support_by_settlement_nature",
        "physical_inflows_by_category",
        "physical_outflows_by_category",
        "cash_residuals",
    }
    assert expected <= ids
    for spec in REPORT_SPECS:
        if spec.report_id in expected:
            assert spec.question.strip()
            assert spec.establishes.strip()
            assert spec.caveat.strip()
            assert spec.currency_policy == "separate_native"


def test_round3_cost_views_reconcile_to_annual_authorities(tmp_path: Path) -> None:
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round3_sources(run, metrics)
    monthly = build_specialized_view("opex_monthly_evolution", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    maintenance = build_specialized_view("maintenance_by_property", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    legal = build_specialized_view("legal_costs_by_property", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert monthly["value"].sum() == 120
    assert maintenance["value"].sum() == 70
    assert legal["value"].sum() == 50
    assert set(maintenance["property"]) == {"A", "B"}


def test_round3_cost_view_fails_closed_on_reconciliation_gap(tmp_path: Path) -> None:
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round3_sources(run, metrics)
    annual = pd.read_csv(metrics / "annual_balance_dashboard_metrics.csv")
    annual.loc[annual["metric_id"].eq("IS.OPEX.PROPERTY"), "value"] = 119
    annual.to_csv(metrics / "annual_balance_dashboard_metrics.csv", index=False)
    with pytest.raises(ValueError, match="monthly OPEX"):
        build_specialized_view("opex_monthly_evolution", run_root=run, metrics_dir=metrics, scope="FBPM")


def test_round3_support_views_keep_target_box_scope_and_do_not_promote_household(tmp_path: Path) -> None:
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round3_sources(run, metrics)
    by_category = build_specialized_view("support_by_obligation_category", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    by_channel = build_specialized_view("support_by_funding_channel", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    by_nature = build_specialized_view("support_by_settlement_nature", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert by_category["value"].sum() == 100
    assert by_channel["value"].sum() == 100
    assert by_nature["value"].sum() == 100
    assert set(by_category["obligation_category"]) == {"taxes", "services"}


def test_round3_physical_category_views_reconcile_to_cash_accountability(tmp_path: Path) -> None:
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round3_sources(run, metrics)
    inflows = build_specialized_view("physical_inflows_by_category", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    outflows = build_specialized_view("physical_outflows_by_category", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert inflows["value"].sum() == 120
    assert outflows["value"].sum() == 40
    assert set(inflows["cash_category"]) == {"rent", "funding"}
    assert set(outflows["cash_category"]) == {"taxes", "personal_draws"}

    cash = pd.read_csv(run / "monthly_cash_accountability.csv")
    cash.loc[cash["Box"].eq("Property Management"), "total_cash_in"] = 99
    cash.to_csv(run / "monthly_cash_accountability.csv", index=False)
    with pytest.raises(ValueError, match="physical cash in category"):
        build_specialized_view("physical_inflows_by_category", run_root=run, metrics_dir=metrics, scope="FBPM")


def test_round3_cash_residuals_are_diagnostic_and_renderer_stays_self_contained(tmp_path: Path) -> None:
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    out = tmp_path / "out"
    _write_round3_sources(run, metrics)
    residuals = build_specialized_view("cash_residuals", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert residuals["value"].sum() == 11
    assert set(residuals["residual_type"]) == {"Otras entradas", "Entradas sin clasificar", "Otras salidas", "Salidas sin clasificar"}
    assert view_is_available("cash_residuals", run, metrics)

    outputs = render_specialized(
        report_id="cash_residuals",
        run_root=run,
        metrics_dir=metrics,
        out_dir=out,
        as_of_date="2026-08-31",
        require_pdf=False,
    )
    document = outputs["html"].read_text(encoding="utf-8")
    assert "Residuales de caja para revisión" in document
    assert "no demuestra error" in document
    assert "<style>" in document
