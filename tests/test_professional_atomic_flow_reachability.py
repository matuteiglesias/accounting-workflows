from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from accounting.professional.drilldown import (
    _execute_governed_derived_flow,
    _governed_flow_resolution,
    _spec_for_cell,
)
from accounting.professional.table_contracts import enrich_professional_table


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "diagnostics" / "atomic_flow_reachability_20260819.csv"


def _rows() -> list[dict[str, str]]:
    with MATRIX.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _enrich(table_id: str, row: dict[str, object]) -> pd.Series:
    return enrich_professional_table(pd.DataFrame([row]), table_id).iloc[0]


def test_reachability_inventory_has_expected_wave3_boundary() -> None:
    rows = _rows()
    assert len(rows) == 20
    counts = pd.Series([row["routing_strategy"] for row in rows]).value_counts().to_dict()
    assert counts == {
        "GOVERNED_ATOMIC": 6,
        "LEGACY_ATOMIC": 6,
        "NET_DIAGNOSTIC": 3,
        "ANNUAL_METRIC": 2,
        "DEFERRED_FUNDING_SUPPORT": 1,
        "DEFERRED_FX": 1,
        "COMPATIBILITY": 1,
    }
    safe = {row["case_id"] for row in rows if row["safe_to_delete"] == "true"}
    assert safe == {"draws_by_box_monthly", "draws_by_type_monthly"}


def test_current_monthly_governed_rows_resolve_without_legacy_membership() -> None:
    cases = [
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "operating_revenue", "Currency": "ARS", "2026-01": 10},
            "flow.operating_revenue",
        ),
        (
            "monthly_tables_operating_statement_matrix",
            {"statement_line": "property_opex_true", "Currency": "ARS", "2026-01": 10},
            "flow.property_opex.total",
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
    ]

    for table_id, raw, expected_id in cases:
        row = _enrich(table_id, raw)
        assert row["drilldown_cell_id"] == expected_id
        assert _governed_flow_resolution(row) is not None


def test_simple_atomic_gaps_are_observed_as_legacy_not_guessed() -> None:
    statement_lines = ["rent_revenue", "taxes", "services", "maintenance", "legal"]
    for statement_line in statement_lines:
        row = _enrich(
            "monthly_tables_operating_statement_matrix",
            {"statement_line": statement_line, "Currency": "ARS", "2026-01": 10},
        )
        assert row.get("drilldown_cell_id", "") == ""
        assert _governed_flow_resolution(row) is None

    opex = _enrich(
        "monthly_tables_opex_by_type_amount_out",
        {
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_subbucket": "services",
            "2026-01": 10,
        },
    )
    assert opex.get("drilldown_cell_id", "") == ""
    legacy_spec = _spec_for_cell("monthly_tables_opex_by_type_amount_out", opex)
    assert legacy_spec is not None
    assert legacy_spec.measure == "amount_out"


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
