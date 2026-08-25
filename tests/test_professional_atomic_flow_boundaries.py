from __future__ import annotations

import pandas as pd

from accounting.professional import drilldown
from accounting.professional.table_contracts import enrich_professional_table


def _enrich(table_id: str, row: dict[str, object]) -> pd.Series:
    return enrich_professional_table(pd.DataFrame([row]), table_id).iloc[0]


def test_only_unmigrated_funding_dimensions_remain_deferred() -> None:
    """Typed funding dimensions remain deferred; landed FX authority is active."""

    funding_id = "flow.funding_contribution.by_actor"
    fx_id = "flow.fx.conversion_outflow"
    migrated_id = "flow.property_opex.total"

    assert funding_id in drilldown._DEFERRED_FLOW_IDS
    assert fx_id not in drilldown._DEFERRED_FLOW_IDS
    assert migrated_id not in drilldown._DEFERRED_FLOW_IDS

    funding = _enrich(
        "monthly_tables_operating_statement_matrix",
        {
            "drilldown_cell_id": funding_id,
            "Currency": "ARS",
            "funding_actor": "Matías",
            "2026-01": 10,
        },
    )
    fx = _enrich(
        "monthly_tables_operating_statement_matrix",
        {
            "statement_line": "treasury_fx_conversion_out",
            "Currency": "ARS",
            "2026-01": 10,
        },
    )

    assert funding["drilldown_cell_id"] == funding_id
    assert fx["drilldown_cell_id"] == fx_id
    assert drilldown._governed_flow_resolution(funding) is None
    resolution = drilldown._governed_flow_resolution(fx)
    assert resolution is not None
    assert resolution[1] == "amount_out"
    assert resolution[3] == ("Box",)


def test_stable_governed_identity_keeps_derived_cell_dispatch_boundary() -> None:
    """Stable atomic IDs execute in the derived hook without changing filename grain."""

    row = _enrich(
        "monthly_tables_operating_statement_matrix",
        {
            "statement_line": "property_opex_true",
            "Currency": "ARS",
            "2026-06": 40,
        },
    )

    assert row["drilldown_cell_id"] == "flow.property_opex.total"
    assert drilldown._governed_flow_resolution(row) is not None

    # Returning a direct CellSpec here would alter the measure used before
    # governed derived-cell dispatch, including stable detail-file identity.
    assert drilldown._spec_for_cell(
        "monthly_tables_operating_statement_matrix", row
    ) is None
