from __future__ import annotations

"""Governed professional drilldown facade after annual-lineage and FX-grain work.

Wave 3/4 atomic flow, debt position/activity, and validated-cash authorities are
preserved through ``drilldown_wave4_base``. Later facades add stable derived
metrics, governed annual-flow lineage, and explicit FX reporting grain.
Historical/minimal artifacts remain on compatibility paths; modern governed
sources fail closed.
"""

# Architecture markers intentionally remain physically visible at this public
# boundary for cross-wave regressions and human audits. Their implementations
# remain delegated to the preserved Wave 3/4 facade:
# governed_atomic_flow
# monthly_tables_debt_position_matrix / annual_debt_stock_by_pair_wide
# monthly_tables_debt_activity_matrix / annual_debt_activity_by_pair_wide
# monthly_tables_cash_close_matrix / annual_cash_close_by_box_wide
# monthly_tables_diagnostic_box_level_matrix

from contextvars import ContextVar
from pathlib import Path

import pandas as pd

from accounting.contracts.atomic_flow_drilldowns import (
    FlowCellSpec,
    resolve_flow_cell_spec,
)
from accounting.contracts.derived_metrics import DerivedMetricSpec, resolve_derived_metric_spec
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.professional import drilldown_wave4_base as _base
from accounting.professional.annual_flow_executor import execute_annual_flow_membership
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
from accounting.professional.fx_reporting_executor import (
    build_fx_cell_spec,
    execute_fx_reporting_cell,
)
from accounting.professional.fx_reporting_metadata import enrich_fx_reporting_grain_tables


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
_ORIGINAL_RUNTIME_SPEC_FOR_CELL = _base._legacy._spec_for_cell
_ORIGINAL_BUILD_PROFESSIONAL_FLOW_DRILLDOWNS = _base._legacy.build_professional_flow_drilldowns
_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS = (
    _base._legacy.enrich_professional_table_contracts
)
_CURRENT_ANNUAL_FLOW_MEMBERSHIP: ContextVar[pd.DataFrame | None] = ContextVar(
    "current_annual_flow_membership", default=None
)


def _enrich_professional_table_contracts(tables_dir):
    """Run established enrichment, then attach governed derived/FX identity."""

    written = list(_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS(tables_dir))
    written.extend(enrich_derived_metric_tables(tables_dir))
    written.extend(enrich_fx_reporting_grain_tables(tables_dir))
    return list(dict.fromkeys(written))


def _fx_aware_runtime_spec_for_cell(table_id: str, row: pd.Series):
    """Prefer explicit FX grain for modern rows; preserve other routes."""

    governed_fx = build_fx_cell_spec(table_id, row)
    if governed_fx is not None:
        return governed_fx
    return _ORIGINAL_RUNTIME_SPEC_FOR_CELL(table_id, row)


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
    governed_fx = execute_fx_reporting_cell(
        table_id=table_id,
        row=row,
        period=period,
        display_value=display_value,
        split=split,
        tolerance=tolerance,
    )
    if governed_fx is not None:
        return governed_fx

    annual_flow_membership = _CURRENT_ANNUAL_FLOW_MEMBERSHIP.get()
    if annual_flow_membership is not None and not annual_flow_membership.empty:
        governed_annual_flow = execute_annual_flow_membership(
            row=row,
            period=period,
            display_value=display_value,
            annual_flow_membership=annual_flow_membership,
            tolerance=tolerance,
        )
        if governed_annual_flow is not None:
            return governed_annual_flow

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
    """Compatibility resolver; governed grain is resolved separately."""

    return _base._fx_treasury_measure_for_row(table_id, row)


_base._legacy._build_derived_cell = _build_derived_cell
_base._legacy._spec_for_cell = _fx_aware_runtime_spec_for_cell
_base._legacy.enrich_professional_table_contracts = _enrich_professional_table_contracts


def build_professional_flow_drilldowns(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None = None,
    tables_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    fast: bool = False,
):
    """Run professional drilldowns with optional governed annual lineage."""

    membership_path = _base._legacy._find_source(
        Path(repo_root), Path(pack_dir), Path(run_root) if run_root is not None else None,
        "annual_flow_membership.csv",
    )
    membership = (
        _base._legacy._read_csv(membership_path)
        if membership_path is not None
        else pd.DataFrame()
    )
    token = _CURRENT_ANNUAL_FLOW_MEMBERSHIP.set(membership)
    try:
        return _ORIGINAL_BUILD_PROFESSIONAL_FLOW_DRILLDOWNS(
            repo_root=repo_root,
            pack_dir=pack_dir,
            run_root=run_root,
            tables_dir=tables_dir,
            tolerance=tolerance,
            fast=fast,
        )
    finally:
        _CURRENT_ANNUAL_FLOW_MEMBERSHIP.reset(token)


main = _base._legacy.main


if __name__ == "__main__":
    raise SystemExit(main())
