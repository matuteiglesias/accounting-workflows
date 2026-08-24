from __future__ import annotations

"""Governed professional drilldown facade after Wave 5 derived migration.

Wave 3/4 atomic flow, debt position/activity, and validated-cash authorities are
preserved through ``drilldown_wave4_base``. Wave 5 adds stable derived-metric
metadata plus a closed DerivedMetricSpec executor. Historical/minimal artifacts
remain on compatibility paths; modern governed sources fail closed.
"""

# Architecture markers intentionally remain physically visible at this public
# boundary for cross-wave regressions and human audits. Their implementations
# remain delegated to the preserved Wave 3/4 facade:
# governed_atomic_flow
# monthly_tables_debt_position_matrix / annual_debt_stock_by_pair_wide
# monthly_tables_debt_activity_matrix / annual_debt_activity_by_pair_wide
# monthly_tables_cash_close_matrix / annual_cash_close_by_box_wide
# monthly_tables_diagnostic_box_level_matrix

import pandas as pd

from accounting.contracts.atomic_flow_drilldowns import (
    FlowCellSpec,
    resolve_flow_cell_spec,
)
from accounting.contracts.derived_metrics import DerivedMetricSpec, resolve_derived_metric_spec
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.professional import drilldown_wave4_base as _base
from accounting.professional.cash_position_executor import (
    execute_annual_cash_position,
    execute_monthly_cash_position,
)
from accounting.professional.debt_activity_executor import (
    execute_annual_debt_activity,
    execute_monthly_debt_activity,
)
from accounting.professional.debt_position_executor import (
    execute_annual_debt_position,
    execute_monthly_debt_position,
)
from accounting.professional.derived_metric_executor import execute_derived_metric
from accounting.professional.derived_metric_metadata import enrich_derived_metric_tables


# Explicit compatibility surface derived from repository caller census.
# Do not broaden this list: every retained legacy symbol must have a caller
# or an independently documented compatibility contract/removal condition.
LEGACY_COMPAT_EXPORTS = (
    'DEFAULT_TOLERANCE',
    'INDEX_FILENAME',
    'STATUS_OK',
    'STATUS_UNSUPPORTED',
    '_DEFERRED_FLOW_IDS',
    '_annual_formula_spec',
    '_build_annual_debt_activity_companion_cell',
    '_build_annual_debt_stock_companion_cell',
    '_build_annual_formula_cell',
    '_build_debt_activity_cell',
    '_build_debt_position_cell',
    '_cash_bridge_line_spec',
    '_execute_governed_derived_flow',
    '_governed_flow_resolution',
    '_safe_div',
    '_semantic_filter_for_statement_line',
    '_spec_for_cell',
    'row_context_id',
)

DEFAULT_TOLERANCE = _base.DEFAULT_TOLERANCE
INDEX_FILENAME = _base.INDEX_FILENAME
STATUS_OK = _base.STATUS_OK
STATUS_UNSUPPORTED = _base.STATUS_UNSUPPORTED
_DEFERRED_FLOW_IDS = _base._DEFERRED_FLOW_IDS
_annual_formula_spec = _base._annual_formula_spec
_build_annual_debt_activity_companion_cell = _base._build_annual_debt_activity_companion_cell
_build_annual_debt_stock_companion_cell = _base._build_annual_debt_stock_companion_cell
_build_annual_formula_cell = _base._build_annual_formula_cell
_build_debt_activity_cell = _base._build_debt_activity_cell
_build_debt_position_cell = _base._build_debt_position_cell
_cash_bridge_line_spec = _base._cash_bridge_line_spec
_execute_governed_derived_flow = _base._execute_governed_derived_flow
_governed_flow_resolution = _base._governed_flow_resolution
_safe_div = _base._safe_div
_semantic_filter_for_statement_line = _base._semantic_filter_for_statement_line
_spec_for_cell = _base._spec_for_cell
row_context_id = _base.row_context_id


_ORIGINAL_BUILD_DERIVED_CELL = _base._build_derived_cell
_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS = (
    _base._legacy.enrich_professional_table_contracts
)


def _enrich_professional_table_contracts(tables_dir):
    """Run established table enrichment, then attach stable derived IDs."""

    written = list(_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS(tables_dir))
    written.extend(enrich_derived_metric_tables(tables_dir))
    return list(dict.fromkeys(written))


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

    governed_derived = execute_derived_metric(
        table_id=table_id,
        row=row,
        period=period,
        display_value=display_value,
        annual=annual,
        cash_close=cash_close,
        tolerance=tolerance,
    )
    if governed_derived is not None:
        return governed_derived

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


FX_TREASURY_TABLE_IDS = _base.FX_TREASURY_TABLE_IDS
FX_MEASURES = _base.FX_MEASURES


def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    return _base._fx_treasury_measure_for_row(table_id, row)


_base._legacy._build_derived_cell = _build_derived_cell
_base._legacy.enrich_professional_table_contracts = _enrich_professional_table_contracts

build_professional_flow_drilldowns = _base._legacy.build_professional_flow_drilldowns
main = _base._legacy.main


if __name__ == "__main__":
    raise SystemExit(main())
