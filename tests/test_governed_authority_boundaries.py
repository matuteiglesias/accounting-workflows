from __future__ import annotations

import pandas as pd

from accounting.professional import drilldown as professional
from accounting.professional.drilldown_wave4_base import _ORIGINAL_SPEC_FOR_CELL
from accounting.professional.table_contracts import enrich_professional_table


def test_proven_dead_atomic_cellspec_routes_stay_physically_pruned() -> None:
    cases = [
        (
            "monthly_tables_draws_by_box_amount_out",
            {"Currency": "ARS", "Box": "Household", "2026-01": 10},
            "flow.draws.by_box",
        ),
        (
            "monthly_tables_draws_by_type_amount_out",
            {
                "Currency": "ARS",
                "semantic_subbucket": "personal_expense",
                "2026-01": 10,
            },
            "flow.draws.by_type",
        ),
        (
            "monthly_tables_opex_by_type_amount_out",
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_subbucket": "services",
                "2026-01": 10,
            },
            "flow.property_opex.by_box_category",
        ),
    ]

    for table_id, raw, governed_id in cases:
        row = enrich_professional_table(pd.DataFrame([raw]), table_id).iloc[0]
        assert row["drilldown_cell_id"] == governed_id
        assert _ORIGINAL_SPEC_FOR_CELL(table_id, row) is None


def test_governed_identity_cannot_fall_back_to_legacy_execution(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("governed identity reached legacy execution fallback")

    atomic_row = enrich_professional_table(
        pd.DataFrame(
            [
                {
                    "statement_line": "property_opex_true",
                    "Currency": "ARS",
                    "2026-01": 30.0,
                }
            ]
        ),
        "monthly_tables_operating_statement_matrix",
    ).iloc[0]
    assert atomic_row["drilldown_cell_id"] == "flow.property_opex.total"

    monkeypatch.setattr(professional._base, "_ORIGINAL_BUILD_DERIVED_CELL", explode)
    atomic_result = professional._base._build_derived_cell(
        table_id="monthly_tables_operating_statement_matrix",
        row=atomic_row,
        period="2026-01",
        display_value=30.0,
        split=pd.DataFrame(
            [
                {
                    "period": "2026-01",
                    "Currency": "ARS",
                    "Box": "Property Management",
                    "semantic_bucket": "property_opex",
                    "semantic_subbucket": "services",
                    "amount_in": 0.0,
                    "amount_out": 30.0,
                    "net_amount": -30.0,
                    "amount_abs": 30.0,
                }
            ]
        ),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=pd.DataFrame(),
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert atomic_result[0] == "ok"
    assert str(atomic_result[3]).startswith("governed_atomic_flow")

    monkeypatch.setattr(professional, "_ORIGINAL_BUILD_DERIVED_CELL", explode)
    annual = pd.DataFrame(
        [
            {
                "metric_id": "IS.NET.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 75.0,
                "value_status": "available",
            },
            {
                "metric_id": "IS.REVENUE.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 100.0,
                "value_status": "available",
            },
        ]
    )
    result = professional._build_derived_cell(
        table_id="overview_balance_dashboard",
        row=pd.Series(
            {
                "Currency": "ARS",
                "derived_metric_id": "derived.operating_margin",
            }
        ),
        period="2026",
        display_value=0.75,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=annual,
        cash_close=pd.DataFrame(),
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert result[0] == "ok"
    assert result[3] == "governed_derived_formula"
