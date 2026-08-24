from __future__ import annotations

"""Compatibility tests for pre-stable-ID professional derived formulas.

These tests deliberately preserve behavior that is *not* the current governed
meaning. They exist only while supported historical/minimal professional rows
can reach the label-selected legacy formula/diagnostic path.

Removal condition: delete this module when supported professional inputs always
carry stable ``derived_metric_id`` metadata and the legacy label formula and
legacy diagnostic paths are no longer reachable.
"""

import pandas as pd

from accounting.professional import drilldown as professional


def test_legacy_formula_identity_is_bound_to_supported_human_labels() -> None:
    expected = {
        "Margen operativo": (
            "operating_margin",
            ("IS.NET.OPERATING", "IS.REVENUE.OPERATING"),
        ),
        "OPEX / renta": (
            "opex_to_rent",
            ("IS.OPEX.PROPERTY", "IS.REVENUE.OPERATING"),
        ),
        "Retiros / resultado operativo": (
            "draws_to_operating_result",
            ("DIST.DRAWS.PERSONAL", "IS.NET.OPERATING"),
        ),
        "Cobertura después de funding y retiros": (
            "coverage_after_funding_and_draws",
            (
                "COV.NET.AFTER_DRAWS",
                "IS.NET.OPERATING",
                "FUND.CONTRIB.TOTAL",
                "DIST.DRAWS.PERSONAL",
            ),
        ),
    }
    for label, (formula_id, components) in expected.items():
        spec = professional._annual_formula_spec(pd.Series({"metric": label}))
        assert spec is not None
        assert spec.formula_id == formula_id
        assert spec.component_metric_ids == components

    assert professional._annual_formula_spec(
        pd.Series({"metric": "future stable metric"})
    ) is None


def test_legacy_ratio_zero_denominator_returns_zero() -> None:
    annual = pd.DataFrame(
        [
            {
                "metric_id": "IS.NET.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 0.0,
            },
            {
                "metric_id": "IS.REVENUE.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 0.0,
            },
        ]
    )
    result = professional._build_annual_formula_cell(
        table_id="overview_balance_dashboard",
        row=pd.Series({"metric": "Margen operativo", "Currency": "ARS"}),
        period="2026",
        currency="ARS",
        display_value=0.0,
        annual=annual,
        tolerance=1e-6,
    )
    assert result is not None
    assert result[0] == "ok"
    assert result[1] == 0.0
    assert professional._safe_div(100.0, 0.0) == 0.0


def test_legacy_coverage_recomputes_with_zero_defaults_when_source_is_missing() -> None:
    row = pd.Series(
        {"metric": "Cobertura después de funding y retiros", "Currency": "ARS"}
    )
    partial = pd.DataFrame(
        [
            {
                "metric_id": "IS.NET.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 100.0,
            },
            {
                "metric_id": "DIST.DRAWS.PERSONAL",
                "period": "2026",
                "Currency": "ARS",
                "value": 30.0,
            },
        ]
    )
    recomputed = professional._build_annual_formula_cell(
        table_id="overview_balance_dashboard",
        row=row,
        period="2026",
        currency="ARS",
        display_value=70.0,
        annual=partial,
        tolerance=1e-6,
    )
    assert recomputed is not None
    assert recomputed[1] == 70.0

    source_first = pd.concat(
        [
            partial,
            pd.DataFrame(
                [
                    {
                        "metric_id": "COV.NET.AFTER_DRAWS",
                        "period": "2026",
                        "Currency": "ARS",
                        "value": 55.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    sourced = professional._build_annual_formula_cell(
        table_id="overview_balance_dashboard",
        row=row,
        period="2026",
        currency="ARS",
        display_value=55.0,
        annual=source_first,
        tolerance=1e-6,
    )
    assert sourced is not None
    assert sourced[1] == 55.0


def test_legacy_diagnostic_can_mix_blank_party_cash_and_inferred_control() -> None:
    cash = pd.DataFrame(
        [
            {
                "period": "2026-02",
                "Currency": "ARS",
                "Box": "Property Management",
                "party": "",
                "position_type": "inferred_box_motor",
                "source_type": "inferred_box_motor",
                "close_amount": 100.0,
            },
            {
                "period": "2026-02",
                "Currency": "ARS",
                "Box": "Property Management",
                "party": "",
                "position_type": "cash_close",
                "source_type": "bank_statement",
                "close_amount": 100.0,
            },
        ]
    )
    result = professional._build_derived_cell(
        table_id="monthly_tables_diagnostic_box_level_matrix",
        row=pd.Series(
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "metric": "diagnostic_box_level",
            }
        ),
        period="2026-02",
        display_value=200.0,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=cash,
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert result[0] == "ok"
    assert result[1] == 200.0
    assert result[5]["previous_period"] == "2026-01"
