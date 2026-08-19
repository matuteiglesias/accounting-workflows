from __future__ import annotations

import pandas as pd
import pytest

from accounting.professional import drilldown
from accounting.professional.table_contracts import enrich_professional_table


DIRECT_MIGRATED_CASES = (
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
)


@pytest.mark.parametrize("table_id,raw,expected_id", DIRECT_MIGRATED_CASES)
def test_migrated_direct_atomic_tables_cannot_reach_legacy_membership(
    monkeypatch: pytest.MonkeyPatch,
    table_id: str,
    raw: dict[str, object],
    expected_id: str,
) -> None:
    row = enrich_professional_table(pd.DataFrame([raw]), table_id).iloc[0]
    assert row["drilldown_cell_id"] == expected_id

    def forbidden_legacy(*args, **kwargs):
        raise AssertionError(
            f"migrated direct atomic table reached legacy routing: {table_id}"
        )

    monkeypatch.setattr(drilldown, "_ORIGINAL_SPEC_FOR_CELL", forbidden_legacy)
    spec = drilldown._spec_for_cell(table_id, row)

    assert spec is not None
    resolution = drilldown._governed_flow_resolution(row)
    assert resolution is not None
    flow_spec, governed_measure, _, missing = resolution
    assert flow_spec.cell_id == expected_id
    assert spec.measure == governed_measure
    assert missing == ()


def test_wave3_deferred_ids_are_explicit_not_accidental_fallbacks() -> None:
    assert drilldown._DEFERRED_FLOW_IDS == {
        "flow.funding_contribution.by_actor",
        "flow.funding_contribution.by_channel",
        "flow.funding_contribution.by_cash_effect",
        "flow.funding_contribution.by_target_box",
        "flow.fx.conversion_proceeds",
        "flow.fx.conversion_outflow",
        "flow.fx.cost_or_spread",
    }


def test_wave3_annual_periods_remain_outside_monthly_governed_recomputation() -> None:
    row = enrich_professional_table(
        pd.DataFrame(
            [{"metric_id": "IS.RENT.TOTAL", "Currency": "ARS", "2026": 30}]
        ),
        "income_operating_statement",
    ).iloc[0]

    assert row["drilldown_cell_id"] == "flow.rent.total"
    assert drilldown._governed_flow_resolution(row) is not None
    assert (
        drilldown._execute_governed_derived_flow(
            table_id="income_operating_statement",
            row=row,
            period="2026",
            display_value=30,
            split=pd.DataFrame(),
            audit=pd.DataFrame(),
            tolerance=1e-6,
        )
        is None
    )
