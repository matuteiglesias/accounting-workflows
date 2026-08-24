from __future__ import annotations

"""Governed professional drilldown facade.

The historical implementation is preserved in ``drilldown_legacy`` for
compatibility-only routes. Rows with an explicit governed ``drilldown_cell_id``
are resolved through ``FlowCellSpec`` and ``semantic_measure_registry_v1``
before falling back to the legacy implementation.

This module intentionally keeps atomic-flow execution distinct from cash/debt
snapshots, formulas, ratios, and compatibility fallbacks.
"""

from typing import Any

import pandas as pd

from accounting.contracts.atomic_flow_drilldowns import (
    FlowCellSpec,
    resolve_flow_cell_spec,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.professional import drilldown_legacy as _legacy
from accounting.professional.debt_activity_executor import (
    execute_annual_debt_activity,
    execute_monthly_debt_activity,
)
from accounting.professional.debt_position_executor import (
    execute_annual_debt_position,
    execute_monthly_debt_position,
)


# Re-export the historical public/private surface so existing callers and
# regression tests retain their import contract while routing is migrated.
# Explicit compatibility surface derived from repository caller census.
# Do not broaden this list: every retained legacy symbol must have a caller
# or an independently documented compatibility contract/removal condition.
LEGACY_COMPAT_EXPORTS = (
    'DEFAULT_TOLERANCE',
    'INDEX_FILENAME',
    'STATUS_OK',
    'STATUS_UNSUPPORTED',
    '_annual_formula_spec',
    '_build_annual_formula_cell',
    '_cash_bridge_line_spec',
    '_safe_div',
    '_semantic_filter_for_statement_line',
    'row_context_id',
)

DEFAULT_TOLERANCE = _legacy.DEFAULT_TOLERANCE
INDEX_FILENAME = _legacy.INDEX_FILENAME
STATUS_OK = _legacy.STATUS_OK
STATUS_UNSUPPORTED = _legacy.STATUS_UNSUPPORTED
_annual_formula_spec = _legacy._annual_formula_spec
_build_annual_formula_cell = _legacy._build_annual_formula_cell
_cash_bridge_line_spec = _legacy._cash_bridge_line_spec
_safe_div = _legacy._safe_div
_semantic_filter_for_statement_line = _legacy._semantic_filter_for_statement_line
row_context_id = _legacy.row_context_id


_ORIGINAL_SPEC_FOR_CELL = _legacy._spec_for_cell
_ORIGINAL_BUILD_DERIVED_CELL = _legacy._build_derived_cell
_ORIGINAL_BUILD_DEBT_ACTIVITY_CELL = _legacy._build_debt_activity_cell
_ORIGINAL_BUILD_ANNUAL_DEBT_ACTIVITY_COMPANION_CELL = (
    _legacy._build_annual_debt_activity_companion_cell
)
_ORIGINAL_BUILD_DEBT_POSITION_CELL = _legacy._build_debt_position_cell
_ORIGINAL_BUILD_ANNUAL_DEBT_STOCK_COMPANION_CELL = (
    _legacy._build_annual_debt_stock_companion_cell
)

# These IDs are intentionally deferred because their current professional
# surfaces are broader than the corresponding atomic FlowCellSpec contract.
# Funding dimensions can include direct-obligation/debt-linked support, and the
# current FX specs require Box while statement rows may be total-by-currency.
# Neither case may be simplified by silently dropping membership/grain.
_DEFERRED_FLOW_IDS = {
    "flow.funding_contribution.by_actor",
    "flow.funding_contribution.by_channel",
    "flow.funding_contribution.by_cash_effect",
    "flow.funding_contribution.by_target_box",
    "flow.fx.conversion_proceeds",
    "flow.fx.conversion_outflow",
    "flow.fx.cost_or_spread",
}


def _row_dimension_value(row: pd.Series, dimension: str) -> str:
    direct = _legacy._norm(row.get(dimension)) if dimension in row.index else ""
    if direct:
        return direct
    if _legacy._norm(row.get("dimension_name")) == dimension:
        return _legacy._norm(row.get("dimension_value"))
    return ""


def _semantic_members_mask(df: pd.DataFrame, spec: FlowCellSpec) -> pd.Series:
    if "semantic_bucket" not in df.columns:
        return pd.Series(False, index=df.index)
    bucket = df["semantic_bucket"].fillna("").astype(str).str.strip()
    subbucket = (
        df["semantic_subbucket"].fillna("").astype(str).str.strip()
        if "semantic_subbucket" in df.columns
        else pd.Series("", index=df.index)
    )
    mask = pd.Series(False, index=df.index)
    for member_bucket, member_subbucket in spec.semantic_members:
        member = bucket.eq(member_bucket)
        if member_subbucket:
            member &= subbucket.eq(member_subbucket)
        mask |= member
    return mask


def _strict_dimension_mask(
    df: pd.DataFrame,
    dimensions: dict[str, str],
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for dimension, value in dimensions.items():
        if dimension not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df[dimension].fillna("").astype(str).str.strip().eq(value)
    return mask


def _governed_flow_resolution(
    row: pd.Series,
) -> tuple[FlowCellSpec, str, dict[str, str], tuple[str, ...]] | None:
    cell_id = _legacy._norm(row.get("drilldown_cell_id"))
    if not cell_id or cell_id in _DEFERRED_FLOW_IDS:
        return None

    spec = resolve_flow_cell_spec(cell_id)
    if spec is None:
        raise ValueError(f"Unknown governed atomic-flow cell id: {cell_id!r}")

    measure = resolve_semantic_measure(*spec.measure_ref)
    if measure is None:
        raise ValueError(
            "FlowCellSpec measure_ref is no longer governed: "
            f"cell_id={cell_id!r}; measure_ref={spec.measure_ref!r}"
        )

    dimensions: dict[str, str] = {}
    missing: list[str] = []
    for dimension in spec.grain[2:]:
        value = _row_dimension_value(row, dimension)
        if not value:
            missing.append(dimension)
        else:
            dimensions[dimension] = value

    return spec, measure, dimensions, tuple(missing)


def _governed_filter(
    spec: FlowCellSpec,
    dimensions: dict[str, str],
    row: pd.Series,
):
    currency = _legacy._norm(row.get("Currency"))

    def _filter(df: pd.DataFrame, _row: pd.Series) -> pd.Series:
        if not currency or "Currency" not in df.columns:
            return pd.Series(False, index=df.index)
        mask = df["Currency"].fillna("").astype(str).str.strip().eq(currency)
        mask &= _semantic_members_mask(df, spec)
        mask &= _strict_dimension_mask(df, dimensions)
        return mask

    return _filter


def _unsupported_governed_spec(
    table_id: str,
    measure: str,
) -> Any:
    return _legacy.CellSpec(
        table_id,
        measure,
        lambda df, row: pd.Series(False, index=df.index),
        unsupported_if=lambda row: True,
    )


def _spec_for_cell(table_id: str, row: pd.Series):
    # Derived tables build drilldown IDs before dispatch. Preserve their legacy
    # row-level measure/path identity; governed execution happens inside the
    # derived hook instead of changing filenames as an incidental side effect.
    if table_id in _legacy.DERIVED_TABLE_IDS:
        return _ORIGINAL_SPEC_FOR_CELL(table_id, row)

    resolution = _governed_flow_resolution(row)
    if resolution is None:
        return _ORIGINAL_SPEC_FOR_CELL(table_id, row)

    spec, measure, dimensions, missing = resolution
    if missing:
        return _unsupported_governed_spec(table_id, measure)

    cell_id = spec.cell_id
    return _legacy.CellSpec(
        table_id,
        measure,
        _governed_filter(spec, dimensions, row),
        caveat_func=lambda _row, cid=cell_id: (
            f"Governed atomic-flow membership from {cid}; measure from "
            "semantic_measure_registry_v1."
        ),
    )


def _period_mask(df: pd.DataFrame, period: str) -> pd.Series:
    if _legacy.YEAR_RE.match(str(period)):
        return _legacy._year_mask(df, str(period))
    if "period" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["period"].fillna("").astype(str).eq(str(period))


def _execute_governed_derived_flow(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    display_value: float,
    split: pd.DataFrame,
    audit: pd.DataFrame,
    tolerance: float,
):
    # Annual professional rows already have a governed annual metric artifact
    # with established lineage/detail sections. Recomputing them from monthly
    # semantic rows would change provenance even when totals reconcile. Keep
    # annual rows on that existing path until a dedicated annual membership
    # contract composes the governed monthly measure without discarding lineage.
    if _legacy.YEAR_RE.match(str(period)):
        return None

    resolution = _governed_flow_resolution(row)
    if resolution is None:
        return None

    spec, measure, dimensions, missing = resolution
    currency = _legacy._norm(row.get("Currency"))
    base_filters = {
        "period": period,
        "Currency": currency,
        "drilldown_cell_id": spec.cell_id,
        "semantic_members": [list(member) for member in spec.semantic_members],
        "grain": list(spec.grain),
        "measure": measure,
        "dimensions": dimensions,
        "row_context": _legacy._row_context(table_id, row),
    }

    if missing:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "unsupported": True, "reason": f"missing governed grain dimensions: {list(missing)}"},
            "Governed atomic-flow row is missing required grain metadata; no broad fallback aggregation was allowed.",
            pd.DataFrame(),
            [],
        )

    if not currency:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "unsupported": True, "reason": "missing Currency would risk cross-currency aggregation"},
            "Governed atomic-flow execution requires explicit currency.",
            pd.DataFrame(),
            [],
        )

    if split.empty:
        return (
            _legacy.STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "error": "missing monthly_flow_semantic_split.csv"},
            "",
            pd.DataFrame(),
            [],
        )

    if measure not in split.columns:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "unsupported": True, "reason": f"measure column unavailable: {measure}"},
            "Governed atomic-flow measure is not available in the semantic split.",
            pd.DataFrame(),
            [],
        )

    row_filter = _governed_filter(spec, dimensions, row)
    semantic_mask = _period_mask(split, period) & row_filter(split, row)
    semantic_rows = split.loc[semantic_mask].copy()
    matched = _legacy._measure_sum(semantic_rows, measure)
    residual = matched - display_value

    detail_rows, lineage = _legacy._detail_from_audit(
        audit,
        semantic_rows,
        lambda df: _period_mask(df, period) & row_filter(df, row),
    )
    if detail_rows.empty and lineage != "semantic_only":
        detail_rows = semantic_rows.copy()

    status = (
        _legacy.STATUS_EMPTY
        if semantic_rows.empty
        else _legacy.STATUS_OK
        if abs(residual) <= tolerance
        else _legacy.STATUS_RESIDUAL_WARNING
    )
    filters = {**base_filters, "executor": "governed_atomic_flow_v1"}
    caveat = (
        f"Governed atomic-flow membership from {spec.cell_id}; measure from "
        "semantic_measure_registry_v1."
    )
    sections = [
        ("Governed semantic rows", semantic_rows),
        ("Classification rows", detail_rows),
    ]
    return (
        status,
        matched,
        residual,
        f"governed_atomic_flow:{lineage}",
        "monthly_flow_semantic_split.csv",
        filters,
        caveat,
        detail_rows if not detail_rows.empty else semantic_rows,
        sections,
    )


def _build_debt_activity_cell(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_activity: pd.DataFrame,
    tolerance: float,
):
    governed = execute_monthly_debt_activity(
        row=row,
        period=period,
        display_value=display_value,
        debt_activity=debt_activity,
        tolerance=tolerance,
    )
    if governed is not None:
        return governed
    return _ORIGINAL_BUILD_DEBT_ACTIVITY_CELL(
        row=row,
        period=period,
        display_value=display_value,
        debt_activity=debt_activity,
        tolerance=tolerance,
    )


def _build_annual_debt_activity_companion_cell(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_activity: pd.DataFrame,
    tolerance: float,
):
    governed = execute_annual_debt_activity(
        row=row,
        period=period,
        display_value=display_value,
        debt_activity=debt_activity,
        tolerance=tolerance,
    )
    if governed is not None:
        return governed
    return _ORIGINAL_BUILD_ANNUAL_DEBT_ACTIVITY_COMPANION_CELL(
        row=row,
        period=period,
        display_value=display_value,
        debt_activity=debt_activity,
        tolerance=tolerance,
    )


def _build_debt_position_cell(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_position: pd.DataFrame,
    tolerance: float,
):
    governed = execute_monthly_debt_position(
        row=row,
        period=period,
        display_value=display_value,
        debt_position=debt_position,
        tolerance=tolerance,
    )
    if governed is not None:
        return governed
    return _ORIGINAL_BUILD_DEBT_POSITION_CELL(
        row=row,
        period=period,
        display_value=display_value,
        debt_position=debt_position,
        tolerance=tolerance,
    )


def _build_annual_debt_stock_companion_cell(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_position: pd.DataFrame,
    tolerance: float,
):
    governed = execute_annual_debt_position(
        row=row,
        period=period,
        display_value=display_value,
        debt_position=debt_position,
        tolerance=tolerance,
    )
    if governed is not None:
        return governed
    return _ORIGINAL_BUILD_ANNUAL_DEBT_STOCK_COMPANION_CELL(
        row=row,
        period=period,
        display_value=display_value,
        debt_position=debt_position,
        tolerance=tolerance,
    )


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
    if table_id == "monthly_tables_debt_activity_matrix":
        governed_activity = execute_monthly_debt_activity(
            row=row,
            period=period,
            display_value=display_value,
            debt_activity=debt_activity,
            tolerance=tolerance,
        )
        if governed_activity is not None:
            return governed_activity

    if table_id == "annual_debt_activity_by_pair_wide":
        governed_activity = execute_annual_debt_activity(
            row=row,
            period=period,
            display_value=display_value,
            debt_activity=debt_activity,
            tolerance=tolerance,
        )
        if governed_activity is not None:
            return governed_activity

    if table_id == "monthly_tables_debt_position_matrix":
        governed_position = execute_monthly_debt_position(
            row=row,
            period=period,
            display_value=display_value,
            debt_position=debt_position,
            tolerance=tolerance,
        )
        if governed_position is not None:
            return governed_position

    if table_id == "annual_debt_stock_by_pair_wide":
        governed_position = execute_annual_debt_position(
            row=row,
            period=period,
            display_value=display_value,
            debt_position=debt_position,
            tolerance=tolerance,
        )
        if governed_position is not None:
            return governed_position

    governed = _execute_governed_derived_flow(
        table_id=table_id,
        row=row,
        period=period,
        display_value=display_value,
        split=split,
        audit=audit,
        tolerance=tolerance,
    )
    if governed is not None:
        return governed
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
# public module so the anti-shadowing AST regression remains meaningful.
FX_TREASURY_TABLE_IDS = _legacy.FX_TREASURY_TABLE_IDS
FX_MEASURES = _legacy.FX_MEASURES


def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    return _legacy._fx_treasury_measure_for_row(table_id, row)


# Patch only the two routing hooks used by the historical orchestration. The
# orchestration itself remains unchanged, so generated index/detail/QA contracts
# are preserved while governed rows bypass local semantic/position/activity
# routing.
_legacy._spec_for_cell = _spec_for_cell
_legacy._build_derived_cell = _build_derived_cell

build_professional_flow_drilldowns = _legacy.build_professional_flow_drilldowns
main = _legacy.main


if __name__ == "__main__":
    raise SystemExit(main())
