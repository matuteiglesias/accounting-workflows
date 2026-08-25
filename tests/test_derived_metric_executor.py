from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.professional import drilldown as professional
from accounting.professional.derived_metric_metadata import enrich_derived_metric_table


def _annual(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "period_grain": "Y",
        "dimension_name": "",
        "dimension_value": "",
        "source_table": "monthly_operating_statement.csv",
        "value_status": "available",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _call(row: dict, annual: pd.DataFrame, display: float):
    return professional._build_derived_cell(
        table_id="overview_balance_dashboard",
        row=pd.Series(row),
        period="2026",
        display_value=display,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=annual,
        cash_close=pd.DataFrame(),
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )


def test_metadata_adapter_attaches_stable_ids_before_execution() -> None:
    source = pd.DataFrame(
        [
            {"Currency": "ARS", "metric": "Margen operativo", "2026": 0.75},
            {"Currency": "ARS", "metric": "OPEX / renta", "2026": 0.25},
            {"Currency": "ARS", "metric": "Retiros / resultado operativo", "2026": 0.4},
            {"Currency": "ARS", "metric": "Cobertura después de funding y retiros", "2026": 550.0},
        ]
    )
    enriched = enrich_derived_metric_table(source, "overview_balance_dashboard")
    assert list(enriched["derived_metric_id"]) == [
        "derived.operating_margin",
        "derived.opex_to_rent",
        "derived.draws_to_operating_result",
        "derived.coverage_after_draws",
    ]
    assert set(enriched["derived_metric_id_source"]) == {"compatibility_presentation_mapping"}


def test_opex_to_rent_uses_rent_not_total_operating_revenue() -> None:
    annual = _annual(
        [
            {"metric_id": "IS.REVENUE.OPERATING", "period": "2026", "Currency": "ARS", "value": 1000.0},
            {"metric_id": "IS.RENT.TOTAL", "period": "2026", "Currency": "ARS", "value": 800.0},
            {"metric_id": "IS.OPEX.PROPERTY", "period": "2026", "Currency": "ARS", "value": 200.0},
        ]
    )
    governed = _call(
        {"Currency": "ARS", "metric": "OPEX / renta", "derived_metric_id": "derived.opex_to_rent"},
        annual,
        0.25,
    )
    assert governed[0] == "ok"
    assert governed[1] == 0.25
    assert governed[3] == "governed_derived_formula"


def test_ratio_zero_denominator_is_not_applicable_not_false_zero() -> None:
    annual = _annual(
        [
            {"metric_id": "IS.NET.OPERATING", "period": "2026", "Currency": "ARS", "value": 0.0},
            {"metric_id": "IS.REVENUE.OPERATING", "period": "2026", "Currency": "ARS", "value": 0.0},
        ]
    )
    governed = _call(
        {"Currency": "ARS", "metric": "Margen operativo", "derived_metric_id": "derived.operating_margin"},
        annual,
        0.0,
    )
    assert governed[0] == "unsupported"
    assert "zero_denominator:not_applicable" in governed[5]["selection_reason"]


def test_missing_modern_component_fails_closed_instead_of_zero_default() -> None:
    annual = _annual(
        [
            {"metric_id": "IS.NET.OPERATING", "period": "2026", "Currency": "ARS", "value": 100.0},
            {"metric_id": "DIST.DRAWS.PERSONAL", "period": "2026", "Currency": "ARS", "value": 30.0},
            {"metric_id": "COV.NET.AFTER_DRAWS", "period": "2026", "Currency": "ARS", "value": 70.0},
        ]
    )
    governed = _call(
        {"Currency": "ARS", "metric": "Cobertura después de funding y retiros", "derived_metric_id": "derived.coverage_after_draws"},
        annual,
        70.0,
    )
    assert governed[0] == "unsupported"
    assert governed[5]["selection_reason"] == "missing_component:FUND.CONTRIB.TOTAL"


def test_source_value_is_authority_and_formula_reconciliation_can_warn() -> None:
    annual = _annual(
        [
            {"metric_id": "IS.NET.OPERATING", "period": "2026", "Currency": "ARS", "value": 750.0},
            {"metric_id": "FUND.CONTRIB.TOTAL", "period": "2026", "Currency": "ARS", "value": 100.0},
            {"metric_id": "DIST.DRAWS.PERSONAL", "period": "2026", "Currency": "ARS", "value": 300.0},
            {"metric_id": "COV.NET.AFTER_DRAWS", "period": "2026", "Currency": "ARS", "value": 560.0},
        ]
    )
    result = _call(
        {"Currency": "ARS", "metric": "Cobertura después de funding y retiros", "derived_metric_id": "derived.coverage_after_draws"},
        annual,
        560.0,
    )
    assert result[1] == 560.0
    assert result[0] == "residual_warning"
    assert result[5]["formula_residual"] == -10.0
    assert result[3] == "governed_source_value_with_formula_reconciliation"


def _cash_rows(include_previous: bool = True) -> pd.DataFrame:
    rows = [
        {
            "period": "2026-02", "Currency": "ARS", "Box": "Property Management", "party": "",
            "close_amount": 100.0, "position_type": "inferred_box_motor", "source_type": "inferred_box_motor",
            "cash_suitability": "safe_with_caveat", "is_frontend_safe": False, "as_of_date": "2026-02-28",
        },
        {
            "period": "2026-02", "Currency": "ARS", "Box": "Property Management", "party": "",
            "close_amount": 1000.0, "position_type": "cash_close", "source_type": "bank_statement",
            "cash_suitability": "frontend_safe", "is_frontend_safe": True, "as_of_date": "2026-02-28",
        },
    ]
    if include_previous:
        rows.extend(
            [
                {
                    "period": "2026-01", "Currency": "ARS", "Box": "Property Management", "party": "",
                    "close_amount": 80.0, "position_type": "inferred_box_motor", "source_type": "inferred_box_motor",
                    "cash_suitability": "safe_with_caveat", "is_frontend_safe": False, "as_of_date": "2026-01-31",
                },
                {
                    "period": "2026-01", "Currency": "ARS", "Box": "Property Management", "party": "",
                    "close_amount": 500.0, "position_type": "cash_close", "source_type": "bank_statement",
                    "cash_suitability": "frontend_safe", "is_frontend_safe": True, "as_of_date": "2026-01-31",
                },
            ]
        )
    return pd.DataFrame(rows)


def _diagnostic(cash: pd.DataFrame, display: float):
    return professional._build_derived_cell(
        table_id="monthly_tables_diagnostic_box_level_matrix",
        row=pd.Series(
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "metric": "diagnostic_box_level",
                "derived_metric_id": "derived.diagnostic_box_level",
            }
        ),
        period="2026-02",
        display_value=display,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=cash,
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )


def test_diagnostic_uses_only_inferred_control_not_blank_party_validated_cash() -> None:
    cash = _cash_rows()
    governed = _diagnostic(cash, 20.0)
    assert governed[0] == "ok"
    assert governed[1] == 20.0
    assert governed[3] == "governed_inferred_box_control_period_delta"
    assert governed[5]["validated_cash_fallback"] == "never"


def test_diagnostic_missing_previous_is_unavailable_not_zero_baseline() -> None:
    result = _diagnostic(_cash_rows(include_previous=False), 100.0)
    assert result[0] == "unsupported"
    assert "previous:unavailable:no_inferred_control_candidate" in result[5]["selection_reason"]


def test_full_orchestration_enriches_and_executes_modern_derived_rows(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"
    run.mkdir(parents=True)
    tables.mkdir(parents=True)

    annual = _annual(
        [
            {"metric_id": "IS.REVENUE.OPERATING", "period": "2026", "Currency": "ARS", "value": 1000.0},
            {"metric_id": "IS.RENT.TOTAL", "period": "2026", "Currency": "ARS", "value": 800.0, "source_table": "monthly_flow_semantic_split.csv"},
            {"metric_id": "IS.OPEX.PROPERTY", "period": "2026", "Currency": "ARS", "value": 200.0},
            {"metric_id": "IS.NET.OPERATING", "period": "2026", "Currency": "ARS", "value": 800.0},
            {"metric_id": "FUND.CONTRIB.TOTAL", "period": "2026", "Currency": "ARS", "value": 100.0},
            {"metric_id": "DIST.DRAWS.PERSONAL", "period": "2026", "Currency": "ARS", "value": 300.0},
            {"metric_id": "COV.NET.AFTER_DRAWS", "period": "2026", "Currency": "ARS", "value": 600.0},
        ]
    )
    annual.to_csv(run / "annual_balance_dashboard_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"Currency": "ARS", "metric": "Margen operativo", "2026": 0.8},
            {"Currency": "ARS", "metric": "OPEX / renta", "2026": 0.25},
            {"Currency": "ARS", "metric": "Cobertura después de funding y retiros", "2026": 600.0},
        ]
    ).to_csv(tables / "overview_balance_dashboard.csv", index=False)

    paths = professional.build_professional_flow_drilldowns(repo, pack, run)
    enriched = pd.read_csv(tables / "overview_balance_dashboard.csv")
    assert set(enriched["derived_metric_id"]) == {
        "derived.operating_margin",
        "derived.opex_to_rent",
        "derived.coverage_after_draws",
    }
    index = pd.read_csv(paths["index"])
    rows = index[index["table_id"].eq("overview_balance_dashboard")]
    assert set(rows["status"]) == {"ok"}
    assert set(rows["lineage_level"]) == {
        "governed_derived_formula",
        "governed_source_value_with_formula_reconciliation",
    }
