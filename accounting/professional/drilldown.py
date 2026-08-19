from __future__ import annotations

"""Governed professional drilldown facade after Wave 4 cash migration.

The pre-PR15B facade is preserved byte-for-byte in ``drilldown_wave4_base``.
This module adds only governed validated-cash routing. Atomic flow and debt
position/activity behavior remain delegated to that preserved facade; formula,
diagnostic, FX, funding-support, and historical compatibility routes remain
unchanged.

Architectural authorities intentionally remain visible at this public boundary:
``governed_atomic_flow``, ``monthly_tables_debt_position_matrix``,
``annual_debt_stock_by_pair_wide``, ``monthly_tables_debt_activity_matrix``, and
``annual_debt_activity_by_pair_wide`` all continue through the governed Wave 3/4
facade preserved below.
"""

from typing import Any

import pandas as pd

# Keep semantic/FlowCellSpec authorities physically visible in the public
# facade. Existing architecture regressions intentionally inspect this module.
from accounting.contracts.atomic_flow_drilldowns import (
    FlowCellSpec,
    resolve_flow_cell_spec,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.professional import drilldown_wave4_base as _base
from accounting.professional.cash_position_executor import (
    execute_annual_cash_position,
    execute_monthly_cash_position,
)
# Re-import these governed executors at the public boundary so PR13/PR14
# isolation contracts remain explicit even though their routing is preserved in
# drilldown_wave4_base.
from accounting.professional.debt_activity_executor import (
    execute_annual_debt_activity,
    execute_monthly_debt_activity,
)
from accounting.professional.debt_position_executor import (
    execute_annual_debt_position,
    execute_monthly_debt_position,
)


# Re-export the complete pre-15B public/private surface. Only the derived-cell
# routing hook below is changed for cash headline tables.
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


_ORIGINAL_BUILD_DERIVED_CELL = _base._build_derived_cell


def _build_derived_cell(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    display_value: float,
    split: pd.DataFrame,
    audit: pd.DataFrame,
    stmt: pd.DataFrame,
    annual: pd.DataFrame,
    cash_close: pd.DataFrame,
    debt_activity: pd.DataFrame,
    debt_position: pd.DataFrame,
    tolerance: float,
):
    if table_id == "monthly_tables_cash_close_matrix":
        governed_cash = execute_monthly_cash_position(
            row=row,
            period=period,
            display_value=display_value,
            cash_close=cash_close,
            tolerance=tolerance,
        )
        if governed_cash is not None:
            return governed_cash

    if table_id == "annual_cash_close_by_box_wide":
        governed_cash = execute_annual_cash_position(
            row=row,
            period=period,
            display_value=display_value,
            cash_close=cash_close,
            tolerance=tolerance,
        )
        if governed_cash is not None:
            return governed_cash

    # In particular, monthly_tables_diagnostic_box_level_matrix is deliberately
    # not intercepted here. Its period-delta formula belongs to Wave 5.
    return _ORIGINAL_BUILD_DERIVED_CELL(
        table_id=table_id,
        row=row,
        period=period,
        display_value=display_value,
        split=split,
        audit=audit,
        stmt=stmt,
        annual=annual,
        cash_close=cash_close,
        debt_activity=debt_activity,
        debt_position=debt_position,
        tolerance=tolerance,
    )


# Keep the characterized FX compatibility surface physically visible in this
# public module so the anti-shadowing regression continues to protect it.
FX_TREASURY_TABLE_IDS = _base.FX_TREASURY_TABLE_IDS
FX_MEASURES = _base.FX_MEASURES


def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    return _base._fx_treasury_measure_for_row(table_id, row)


# The preserved facade already patched drilldown_legacy with governed flow/debt
# hooks. Replace only its derived hook so historical orchestration sees cash too.
_base._legacy._build_derived_cell = _build_derived_cell

build_professional_flow_drilldowns = _base._legacy.build_professional_flow_drilldowns
main = _base._legacy.main


if __name__ == "__main__":
    raise SystemExit(main())
