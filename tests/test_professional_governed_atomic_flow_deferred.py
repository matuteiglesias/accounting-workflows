from __future__ import annotations

import pandas as pd

from accounting.professional.drilldown import (
    _governed_flow_resolution,
    _spec_for_cell,
)
from accounting.professional.table_contracts import enrich_professional_table


def test_fx_atomic_id_is_deferred_until_total_vs_box_grain_is_explicit() -> None:
    row = enrich_professional_table(
        pd.DataFrame(
            [
                {
                    "statement_line": "treasury_fx_conversion_out",
                    "Currency": "ARS",
                    "2026-06": 5,
                }
            ]
        ),
        "monthly_tables_operating_statement_matrix",
    ).iloc[0]

    assert row["drilldown_cell_id"] == "flow.fx.conversion_outflow"
    assert _governed_flow_resolution(row) is None


def test_derived_rows_keep_legacy_row_measure_for_stable_drilldown_ids() -> None:
    row = enrich_professional_table(
        pd.DataFrame(
            [
                {
                    "statement_line": "property_opex_true",
                    "Currency": "ARS",
                    "2026-06": 40,
                }
            ]
        ),
        "monthly_tables_operating_statement_matrix",
    ).iloc[0]

    assert row["drilldown_cell_id"] == "flow.property_opex.total"
    # The governed executor is invoked inside the derived-cell hook. Returning a
    # direct CellSpec here would alter the measure used to construct the stable
    # drilldown/detail filenames before dispatch.
    assert _spec_for_cell("monthly_tables_operating_statement_matrix", row) is None
