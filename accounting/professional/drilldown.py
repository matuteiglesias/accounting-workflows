from __future__ import annotations

"""Current professional drilldown authority and compatibility boundary.

The supported professional surface resolves governed identities directly in this
module. The historical ``drilldown_legacy`` module now supplies only the
remaining compatibility routes plus stable orchestration/rendering helpers; the
Wave-4 facade layer is retired.

Governed identities never fall back to historical semantic execution. A legacy
route is reachable only when the row has no governed identity and the current
supported-surface census still permits that compatibility family.
"""

from contextvars import ContextVar
from pathlib import Path

import pandas as pd

from accounting.contracts.annual_flow_membership import resolve_annual_flow_membership_spec
from accounting.contracts.atomic_flow_drilldowns import (
    FlowCellSpec,
    resolve_flow_cell_spec,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.professional import drilldown_legacy as _legacy
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
from accounting.professional.funding_support_executor import execute_annual_funding_support
from accounting.professional.fx_drilldown_authority import (
    FX_MEASURES,
    FX_TREASURY_TABLE_IDS,
    _fx_treasury_measure_for_row,
    resolve_fx_drilldown,
)

# Small compatibility/export seam still used by renderer and durable contract tests.
LEGACY_COMPAT_EXPORTS = (
    "DEFAULT_TOLERANCE",
    "INDEX_FILENAME",
    "STATUS_OK",
    "STATUS_UNSUPPORTED",
    "_annual_formula_spec",
    "_build_annual_formula_cell",
    "_cash_bridge_line_spec",
    "_safe_div",
    "_semantic_filter_for_statement_line",
    "row_context_id",
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

# Explicitly retained test/audit hooks. These are not separate production
# authorities; they expose the current governed functions directly.
STATUS_EMPTY = _legacy.STATUS_EMPTY
STATUS_RESIDUAL_WARNING = _legacy.STATUS_RESIDUAL_WARNING
STATUS_ERROR = _legacy.STATUS_ERROR

_ORIGINAL_SPEC_FOR_CELL = _legacy._spec_for_cell
_ORIGINAL_BUILD_DERIVED_CELL = _legacy._build_derived_cell
_ORIGINAL_BUILD_PROFESSIONAL_FLOW_DRILLDOWNS = _legacy.build_professional_flow_drilldowns
_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS = _legacy.enrich_professional_table_contracts

_CURRENT_ANNUAL_FLOW_MEMBERSHIP: ContextVar[pd.DataFrame | None] = ContextVar(
    "current_annual_flow_membership", default=None
)
_CURRENT_REPAYMENT_DETAIL: ContextVar[pd.DataFrame | None] = ContextVar(
    "current_repayment_detail", default=None
)

# Broader support needs FundingSupportSpec rather than the narrow atomic
# funding_contribution specs. FX is no longer deferred: its dedicated
# single-authority resolver is handled before atomic-flow dispatch.
_DEFERRED_FLOW_IDS = {
    "flow.funding_contribution.by_actor",
    "flow.funding_contribution.by_channel",
    "flow.funding_contribution.by_cash_effect",
    "flow.funding_contribution.by_target_box",
}


def _enrich_professional_table_contracts(tables_dir: Path) -> list[Path]:
    written = list(_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS(tables_dir))
    written.extend(enrich_derived_metric_tables(tables_dir))
    return list(dict.fromkeys(written))


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


def _unsupported_spec(
    table_id: str,
    row: pd.Series,
    *,
    measure: str = "",
    reason: str = "governed route unavailable",
):
    return _legacy.CellSpec(
        table_id,
        measure
        or _legacy._norm(row.get("measure"))
        or _legacy._metric_name(row),
        lambda df, _row: pd.Series(False, index=df.index),
        caveat_func=lambda _row, why=reason: f"Drilldown unsupported: {why}.",
        unsupported_if=lambda _row: True,
    )


def _fx_spec_for_cell(table_id: str, row: pd.Series):
    resolution = resolve_fx_drilldown(table_id, row)
    if resolution is None:
        return None
    if not resolution.supported:
        return _unsupported_spec(
            table_id,
            row,
            measure=resolution.measure,
            reason=f"FX {resolution.unsupported_reason}",
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

    return _legacy.CellSpec(
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

    # Derived rows execute in _build_derived_cell. Keep their established row
    # identity/filename behavior until the remaining diagnostic/bridge families
    # are harvested.
    if table_id in _legacy.DERIVED_TABLE_IDS:
        return _ORIGINAL_SPEC_FOR_CELL(table_id, row)

    resolution = _governed_flow_resolution(row)
    if resolution is None:
        return _ORIGINAL_SPEC_FOR_CELL(table_id, row)

    spec, measure, dimensions, missing = resolution
    if missing:
        return _unsupported_spec(
            table_id,
            row,
            measure=measure,
            reason=f"missing governed grain dimensions: {list(missing)}",
        )

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
    # Annual additive-flow rows are governed by annual_flow_membership_v1.
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
            STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {
                **base_filters,
                "unsupported": True,
                "reason": f"missing governed grain dimensions: {list(missing)}",
            },
            "Governed atomic-flow row is missing required grain metadata; no broad fallback aggregation was allowed.",
            pd.DataFrame(),
            [],
        )

    if not currency:
        return (
            STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {
                **base_filters,
                "unsupported": True,
                "reason": "missing Currency would risk cross-currency aggregation",
            },
            "Governed atomic-flow execution requires explicit currency.",
            pd.DataFrame(),
            [],
        )

    if split.empty:
        return (
            STATUS_ERROR,
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
            STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {
                **base_filters,
                "unsupported": True,
                "reason": f"measure column unavailable: {measure}",
            },
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
        STATUS_EMPTY
        if semantic_rows.empty
        else STATUS_OK
        if abs(residual) <= tolerance
        else STATUS_RESIDUAL_WARNING
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


def _execute_annual_bridge_fx(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    split: pd.DataFrame,
    audit: pd.DataFrame,
    tolerance: float,
):
    """Execute annual FX bridge rows through the single FX authority."""

    if not _legacy.YEAR_RE.match(str(period)):
        return None
    resolution = resolve_fx_drilldown("cash_annual_box_flow_bridge_wide", row)
    if resolution is None:
        return None

    filters = {
        "year": period,
        "Currency": resolution.currency,
        "Box": resolution.box,
        "measure": resolution.measure,
        "semantic_subbucket": resolution.semantic_subbucket,
        "grain": resolution.grain,
        "executor": "governed_fx_bridge_v1",
    }
    if not resolution.supported:
        return (
            STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {
                **filters,
                "unsupported": True,
                "reason": resolution.unsupported_reason,
            },
            "FX bridge row failed the explicit measure/grain authority.",
            pd.DataFrame(),
            [],
        )
    if split.empty or resolution.measure not in split.columns:
        return (
            STATUS_ERROR if split.empty else STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "missing_source" if split.empty else "unsupported",
            "monthly_flow_semantic_split.csv",
            {
                **filters,
                "error" if split.empty else "unsupported": True,
                "reason": "missing semantic split or governed FX measure column",
            },
            "",
            pd.DataFrame(),
            [],
        )

    mask = (
        _legacy._year_mask(split, period)
        & _legacy._source_filter_eq(split, "Currency", resolution.currency)
        & _legacy._source_filter_eq(split, "semantic_bucket", "treasury_fx")
    )
    if resolution.grain == "box_currency":
        mask &= _legacy._source_filter_eq(split, "Box", resolution.box)
    if resolution.semantic_subbucket:
        mask &= _legacy._source_filter_eq(
            split,
            "semantic_subbucket",
            resolution.semantic_subbucket,
        )
    semantic_rows = split.loc[mask].copy()
    matched = _legacy._measure_sum(semantic_rows, resolution.measure)
    residual = matched - display_value

    def audit_filter(df: pd.DataFrame) -> pd.Series:
        audit_mask = (
            _legacy._year_mask(df, period)
            & _legacy._source_filter_eq(df, "Currency", resolution.currency)
            & _legacy._source_filter_eq(df, "semantic_bucket", "treasury_fx")
        )
        if resolution.grain == "box_currency":
            audit_mask &= _legacy._source_filter_eq(df, "Box", resolution.box)
        if resolution.semantic_subbucket:
            audit_mask &= _legacy._source_filter_eq(
                df,
                "semantic_subbucket",
                resolution.semantic_subbucket,
            )
        return audit_mask

    detail_rows, lineage = _legacy._detail_from_audit(
        audit,
        semantic_rows,
        audit_filter,
    )
    if detail_rows.empty and lineage != "semantic_only":
        detail_rows = semantic_rows
    status = (
        STATUS_EMPTY
        if semantic_rows.empty
        else STATUS_OK
        if abs(residual) <= tolerance
        else STATUS_RESIDUAL_WARNING
    )
    return (
        status,
        matched,
        residual,
        f"governed_fx_bridge:{lineage}",
        "monthly_flow_semantic_split.csv",
        filters,
        (
            "FX bridge authority uses explicit "
            f"grain={resolution.grain} and measure={resolution.measure}; "
            "no compact default, Box dropping, or cross-currency aggregation."
        ),
        detail_rows if not detail_rows.empty else semantic_rows,
        [
            ("Governed FX semantic rows", semantic_rows),
            ("Classification rows", detail_rows),
        ],
    )


def _fail_closed_current_route(
    *,
    display_value: float,
    source_artifact: str,
    family: str,
):
    return (
        STATUS_UNSUPPORTED,
        0.0,
        -display_value,
        "unsupported",
        source_artifact,
        {
            "unsupported": True,
            "reason": f"{family} does not satisfy the current governed source schema",
        },
        f"{family} is a governed current route; historical/minimal schema fallback has been retired.",
        pd.DataFrame(),
        [],
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
        repayment_detail=_CURRENT_REPAYMENT_DETAIL.get(),
        tolerance=tolerance,
    )
    if governed is not None:
        return governed
    return _fail_closed_current_route(
        display_value=display_value,
        source_artifact="monthly_debt_activity.csv",
        family="Debt activity",
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
        repayment_detail=_CURRENT_REPAYMENT_DETAIL.get(),
        tolerance=tolerance,
    )
    if governed is not None:
        return governed
    return _fail_closed_current_route(
        display_value=display_value,
        source_artifact="monthly_debt_activity.csv",
        family="Annual debt activity",
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
    return _fail_closed_current_route(
        display_value=display_value,
        source_artifact="monthly_debt_position.csv",
        family="Debt position",
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
    return _fail_closed_current_route(
        display_value=display_value,
        source_artifact="monthly_debt_position.csv",
        family="Annual debt position",
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
    # Governed annual additive flows never re-filter the monthly split. For a
    # registered annual metric, even a missing lineage artifact fails closed.
    annual_spec = resolve_annual_flow_membership_spec(
        _legacy._norm(row.get("metric_id"))
    )
    if annual_spec is not None and _legacy.YEAR_RE.match(str(period)):
        annual_flow_membership = _CURRENT_ANNUAL_FLOW_MEMBERSHIP.get()
        return execute_annual_flow_membership(
            row=row,
            period=period,
            display_value=display_value,
            annual_flow_membership=(
                annual_flow_membership
                if annual_flow_membership is not None
                else pd.DataFrame()
            ),
            tolerance=tolerance,
        )

    if _legacy.YEAR_RE.match(str(period)):
        funding = execute_annual_funding_support(
            row=row,
            period=period,
            display_value=display_value,
            split=split,
            tolerance=tolerance,
        )
        if funding is not None:
            return funding

    if table_id == "cash_annual_box_flow_bridge_wide":
        annual_fx = _execute_annual_bridge_fx(
            row=row,
            period=period,
            display_value=display_value,
            split=split,
            audit=audit,
            tolerance=tolerance,
        )
        if annual_fx is not None:
            return annual_fx

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
        return _fail_closed_current_route(
            display_value=display_value,
            source_artifact="monthly_cash_close.csv",
            family="Validated monthly cash",
        )

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
        return _fail_closed_current_route(
            display_value=display_value,
            source_artifact="monthly_cash_close.csv",
            family="Validated annual cash",
        )

    if table_id == "monthly_tables_debt_activity_matrix":
        return _build_debt_activity_cell(
            row=row,
            period=period,
            display_value=display_value,
            debt_activity=debt_activity,
            tolerance=tolerance,
        )

    if table_id == "annual_debt_activity_by_pair_wide":
        return _build_annual_debt_activity_companion_cell(
            row=row,
            period=period,
            display_value=display_value,
            debt_activity=debt_activity,
            tolerance=tolerance,
        )

    if table_id == "monthly_tables_debt_position_matrix":
        return _build_debt_position_cell(
            row=row,
            period=period,
            display_value=display_value,
            debt_position=debt_position,
            tolerance=tolerance,
        )

    if table_id == "annual_debt_stock_by_pair_wide":
        return _build_annual_debt_stock_companion_cell(
            row=row,
            period=period,
            display_value=display_value,
            debt_position=debt_position,
            tolerance=tolerance,
        )

    governed_flow = _execute_governed_derived_flow(
        table_id=table_id,
        row=row,
        period=period,
        display_value=display_value,
        split=split,
        audit=audit,
        tolerance=tolerance,
    )
    if governed_flow is not None:
        return governed_flow

    derived_id = _legacy._norm(row.get("derived_metric_id"))
    if derived_id:
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
        return _fail_closed_current_route(
            display_value=display_value,
            source_artifact="annual_balance_dashboard_metrics.csv",
            family=f"Derived metric {derived_id}",
        )

    # Remaining compatibility is intentionally narrow: current statement/bridge
    # families without a stable governed identity still use the characterized
    # implementation. Each family is listed in the harvest evidence note.
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


# Route the historical orchestration through one modern authority boundary.
_legacy.FX_TREASURY_TABLE_IDS = FX_TREASURY_TABLE_IDS
_legacy.FX_MEASURES = FX_MEASURES
_legacy._fx_treasury_measure_for_row = _fx_treasury_measure_for_row
_legacy._spec_for_cell = _spec_for_cell
_legacy._build_derived_cell = _build_derived_cell
_legacy.enrich_professional_table_contracts = _enrich_professional_table_contracts

# Duplicate FX ID was migration debris; diagnostic Box-level presentation was
# explicitly retired by the current supported-surface census. Underlying Box
# evidence and diagnostic contracts remain available.
_legacy.SUPPORTED_TABLE_IDS = tuple(
    dict.fromkeys(
        table_id
        for table_id in _legacy.SUPPORTED_TABLE_IDS
        if table_id != "monthly_tables_diagnostic_box_level_matrix"
    )
)


def build_professional_flow_drilldowns(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None = None,
    tables_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    fast: bool = False,
):
    membership_path = _legacy._find_source(
        Path(repo_root),
        Path(pack_dir),
        Path(run_root) if run_root is not None else None,
        "annual_flow_membership.csv",
    )
    membership = (
        _legacy._read_csv(membership_path)
        if membership_path is not None
        else pd.DataFrame()
    )
    repayment_detail_path = _legacy._find_source(
        Path(repo_root),
        Path(pack_dir),
        Path(run_root) if run_root is not None else None,
        "monthly_debt_repayment_detail.csv",
    )
    repayment_detail = (
        _legacy._read_csv(repayment_detail_path)
        if repayment_detail_path is not None
        else pd.DataFrame()
    )
    token = _CURRENT_ANNUAL_FLOW_MEMBERSHIP.set(membership)
    repayment_token = _CURRENT_REPAYMENT_DETAIL.set(repayment_detail)
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
        _CURRENT_REPAYMENT_DETAIL.reset(repayment_token)


_legacy.build_professional_flow_drilldowns = build_professional_flow_drilldowns
main = _legacy.main


if __name__ == "__main__":
    raise SystemExit(main())
