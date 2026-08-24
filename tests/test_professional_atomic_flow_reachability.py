from __future__ import annotations

import pandas as pd

from accounting.professional.drilldown import (
    _execute_governed_derived_flow,
    _governed_flow_resolution,
    _spec_for_cell,
)
from accounting.professional.table_contracts import enrich_professional_table


def _enrich(table_id: str, row: dict[str, object]) -> pd.Series:
    return enrich_professional_table(pd.DataFrame([row]), table_id).iloc[0]


def test_all_simple_monthly_atomic_rows_resolve_to_governed_specs() -> None:
    cases = [
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "operating_revenue", "Currency": "ARS", "2026-01": 10},
            "flow.operating_revenue",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "rent_revenue", "Currency": "ARS", "2026-01": 10},
            "flow.rent.total",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "property_opex_true", "Currency": "ARS", "2026-01": 10},
            "flow.property_opex.total",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "taxes", "Currency": "ARS", "2026-01": 10},
            "flow.property_opex.taxes",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "services", "Currency": "ARS", "2026-01": 10},
            "flow.property_opex.services",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "maintenance", "Currency": "ARS", "2026-01": 10},
            "flow.property_opex.maintenance",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "legal", "Currency": "ARS", "2026-01": 10},
            "flow.property_opex.legal",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "funding_contributions", "Currency": "ARS", "2026-01": 10},
            "flow.funding_contribution.total",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "family_draws_or_distributions", "Currency": "ARS", "2026-01": 10},
            "flow.family_draws_or_distributions.total",
        ),
        (
            "monthly_tables_draws_by_box_amount_out",
            {"Currency": "ARS", "Box": "Household", "2026-01": 10},
            "flow.draws.by_box",
        ),
        (
            "monthly_tables_draws_by_type_amount_out",
            {"Currency": "ARS", "semantic_subbucket": "personal_expense", "2026-01": 10},
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

    for table_id, raw, expected_id in cases:
        row = _enrich(table_id, raw)
        assert row["drilldown_cell_id"] == expected_id
        assert _governed_flow_resolution(row) is not None


def test_opex_by_box_category_preserves_box_and_subbucket_grain() -> None:
    row = _enrich(
        "monthly_tables_opex_by_type_amount_out",
        {
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_subbucket": "services",
            "2026-01": 10,
        },
    )
    spec = _spec_for_cell("monthly_tables_opex_by_type_amount_out", row)
    assert spec is not None
    assert spec.measure == "amount_out"

    split = pd.DataFrame(
        [
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "amount_out": 10,
            },
            {
                "Currency": "ARS",
                "Box": "Household",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "amount_out": 99,
            },
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
                "amount_out": 88,
            },
        ]
    )
    selected = split.loc[spec.filter_func(split, row)]
    assert selected["amount_out"].sum() == 10
    assert set(selected["Box"]) == {"Property Management"}
    assert set(selected["semantic_subbucket"]) == {"services"}


def test_deferred_semantics_remain_explicit_and_fail_closed_to_legacy() -> None:
    funding = _enrich(
        "monthly_tables_operating_statement_matrix",
        {
            "drilldown_cell_id": "flow.funding_contribution.by_actor",
            "Currency": "ARS",
            "funding_actor": "Matías",
            "2026-01": 10,
        },
    )
    assert funding["drilldown_cell_id"] == "flow.funding_contribution.by_actor"
    assert _governed_flow_resolution(funding) is None

    fx = _enrich(
        "monthly_tables_operating_statement_matrix",
        {"statement_line": "treasury_fx_conversion_out", "Currency": "ARS", "2026-01": 10},
    )
    assert fx["drilldown_cell_id"] == "flow.fx.conversion_outflow"
    assert _governed_flow_resolution(fx) is None


def test_annual_atomic_identity_preserves_annual_lineage_path() -> None:
    row = _enrich(
        "income_operating_statement",
        {"metric_id": "IS.RENT.TOTAL", "Currency": "ARS", "2026": 30},
    )
    assert row["drilldown_cell_id"] == "flow.rent.total"
    assert _governed_flow_resolution(row) is not None

    result = _execute_governed_derived_flow(
        table_id="income_operating_statement",
        row=row,
        period="2026",
        display_value=30,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert result is None


def test_compatibility_inference_does_not_opt_into_atomic_execution() -> None:
    row = _enrich(
        "overview_balance_dashboard",
        {"Currency": "ARS", "metric": "Funding / aportes", "2026": 100},
    )
    assert row["metric_id"] == "FUND.CONTRIB.TOTAL"
    assert row["drilldown_cell_id"] == ""
    assert _governed_flow_resolution(row) is None
