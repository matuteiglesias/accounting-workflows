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
from accounting.professional.fx_drilldown_authority import (
    FX_MEASURES,
    FX_TREASURY_TABLE_IDS,
    _fx_treasury_measure_for_row,
    resolve_fx_drilldown,
)


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
row_context_id = _base.row_context_id


_ORIGINAL_SPEC_FOR_CELL = _base._spec_for_cell
_ORIGINAL_BUILD_DERIVED_CELL = _base._build_derived_cell
_ORIGINAL_BUILD_PROFESSIONAL_FLOW_DRILLDOWNS = _base._legacy.build_professional_flow_drilldowns
_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS = (
    _base._legacy.enrich_professional_table_contracts
)
_CURRENT_ANNUAL_FLOW_MEMBERSHIP: ContextVar[pd.DataFrame | None] = ContextVar(
    "current_annual_flow_membership", default=None
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


def _unsupported_fx_spec(table_id: str, row: pd.Series, reason: str, measure: str):
    return _base._legacy.CellSpec(
        table_id,
        measure or _base._legacy._norm(row.get("measure")) or _base._legacy._metric_name(row),
        lambda df, _row: pd.Series(False, index=df.index),
        caveat_func=lambda _row, why=reason: f"FX drilldown unsupported: {why}.",
        unsupported_if=lambda _row: True,
    )


def _fx_spec_for_cell(table_id: str, row: pd.Series):
    resolution = resolve_fx_drilldown(table_id, row)
    if resolution is None:
        return None
    if not resolution.supported:
        return _unsupported_fx_spec(
            table_id,
            row,
            resolution.unsupported_reason,
            resolution.measure,
        )

    def _filter(df: pd.DataFrame, _row: pd.Series) -> pd.Series:
        if "Currency" not in df.columns or "semantic_bucket" not in df.columns:
            return pd.Series(False, index=df.index)
        mask = (
            df["Currency"].fillna("").astype(str).str.strip().eq(resolution.currency)
            & df["semantic_bucket"].fillna("").astype(str).str.strip().eq("treasury_fx")
        )
        if resolution.grain == "box_currency":
            if "Box" not in df.columns:
                return pd.Series(False, index=df.index)
            mask &= df["Box"].fillna("").astype(str).str.strip().eq(resolution.box)
        if resolution.semantic_subbucket:
            if "semantic_subbucket" not in df.columns:
                return pd.Series(False, index=df.index)
            mask &= (
                df["semantic_subbucket"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq(resolution.semantic_subbucket)
            )
        return mask

    return _base._legacy.CellSpec(
        table_id,
        resolution.measure,
        _filter,
        caveat_func=lambda _row: (
            "FX drilldown authority: "
            f"grain={resolution.grain}; measure={resolution.measure}."
        ),
    )


def _spec_for_cell(table_id: str, row: pd.Series):
    fx_spec = _fx_spec_for_cell(table_id, row)
    if fx_spec is not None:
        return fx_spec
    return _ORIGINAL_SPEC_FOR_CELL(table_id, row)


# The legacy/base modules remain import-compatible, but every live FX selector
# points at this facade's imported single authority. Their historical resolver
# bodies are therefore no longer on the execution path.
_base.FX_TREASURY_TABLE_IDS = FX_TREASURY_TABLE_IDS
_base.FX_MEASURES = FX_MEASURES
_base._fx_treasury_measure_for_row = _fx_treasury_measure_for_row
_base._spec_for_cell = _spec_for_cell
_base._legacy.FX_TREASURY_TABLE_IDS = FX_TREASURY_TABLE_IDS
_base._legacy.FX_MEASURES = FX_MEASURES
_base._legacy._fx_treasury_measure_for_row = _fx_treasury_measure_for_row
_base._legacy._spec_for_cell = _spec_for_cell
_base._legacy._build_derived_cell = _build_derived_cell
_base._legacy.enrich_professional_table_contracts = _enrich_professional_table_contracts


def build_professional_flow_drilldowns(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None = None,
    tables_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    fast: bool = False,
):
    """Run professional drilldowns with optional governed annual lineage.

    The lineage artifact is loaded once at this public boundary and passed to
    annual governed execution through invocation-local context. Historical packs
    without the artifact continue through the characterized compatibility path.
    """

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
