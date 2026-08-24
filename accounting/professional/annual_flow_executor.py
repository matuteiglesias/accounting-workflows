"""Professional executor for governed annual additive-flow membership."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from accounting.contracts.annual_flow_membership import (
    ANNUAL_FLOW_MEMBERSHIP_VERSION,
    resolve_annual_flow_membership_spec,
)
from accounting.professional import drilldown_legacy as _legacy


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _dimension_value(row: pd.Series, dimension_name: str) -> str:
    if not dimension_name:
        return ""
    explicit_name = _text(row.get("dimension_name"))
    explicit_value = _text(row.get("dimension_value"))
    if explicit_name == dimension_name and explicit_value:
        return explicit_value
    return _text(row.get(dimension_name))


def execute_annual_flow_membership(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    annual_flow_membership: pd.DataFrame,
    tolerance: float,
):
    """Resolve one annual professional flow cell from materialized lineage.

    Returns ``None`` for rows outside the governed annual-flow registry, so
    historical compatibility routing remains available.  Once a modern row has
    a governed metric identity, missing/incompatible lineage fails closed and
    never reclassifies the monthly semantic split.
    """

    if not _legacy.YEAR_RE.match(str(period)):
        return None
    metric_id = _text(row.get("metric_id"))
    spec = resolve_annual_flow_membership_spec(metric_id)
    if spec is None:
        return None

    cell_id = _text(row.get("drilldown_cell_id"))
    if cell_id and cell_id != spec.monthly_flow_cell_id:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "annual_flow_membership.csv",
            {
                "unsupported": True,
                "reason": "annual metric identity conflicts with governed monthly flow cell",
                "metric_id": metric_id,
                "expected_drilldown_cell_id": spec.monthly_flow_cell_id,
                "actual_drilldown_cell_id": cell_id,
            },
            "Annual governed flow identity conflicted with producer metadata; no compatibility reclassification was allowed.",
            pd.DataFrame(),
            [],
        )

    currency = _text(row.get("Currency"))
    if not currency:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "annual_flow_membership.csv",
            {"unsupported": True, "reason": "missing Currency would risk cross-currency aggregation", "metric_id": metric_id},
            "Annual governed flow execution requires explicit native currency.",
            pd.DataFrame(),
            [],
        )

    dimension_value = _dimension_value(row, spec.dimension_name)
    if spec.dimension_name and not dimension_value:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "annual_flow_membership.csv",
            {
                "unsupported": True,
                "reason": f"missing governed annual dimension {spec.dimension_name}",
                "metric_id": metric_id,
            },
            "Annual governed flow row is missing required dimension metadata; no broad aggregation was allowed.",
            pd.DataFrame(),
            [],
        )

    if annual_flow_membership is None or annual_flow_membership.empty:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "missing_source",
            "annual_flow_membership.csv",
            {
                "unsupported": True,
                "reason": "missing annual_flow_membership.csv",
                "metric_id": metric_id,
                "lineage_version": ANNUAL_FLOW_MEMBERSHIP_VERSION,
            },
            "Governed annual membership artifact is absent; modern annual flow rows are not recomputed from monthly semantic rows.",
            pd.DataFrame(),
            [],
        )

    required = {
        "metric_id", "period", "Currency", "dimension_name", "dimension_value",
        "monthly_flow_cell_id", "aggregation_rule", "monthly_governed_cell_ids",
        "source_member_ids", "measure_id", "lineage_version", "value", "member_months",
    }
    missing = sorted(required - set(annual_flow_membership.columns))
    if missing:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "annual_flow_membership.csv",
            {"unsupported": True, "reason": f"annual lineage schema missing columns: {missing}", "metric_id": metric_id},
            "Annual governed membership artifact has an incompatible schema; no fallback classification was allowed.",
            pd.DataFrame(),
            [],
        )

    candidates = annual_flow_membership.loc[
        annual_flow_membership["metric_id"].fillna("").astype(str).eq(metric_id)
        & annual_flow_membership["period"].fillna("").astype(str).eq(str(period))
        & annual_flow_membership["Currency"].fillna("").astype(str).eq(currency)
    ].copy()
    if spec.dimension_name:
        candidates = candidates.loc[
            candidates["dimension_name"].fillna("").astype(str).eq(spec.dimension_name)
            & candidates["dimension_value"].fillna("").astype(str).eq(dimension_value)
        ].copy()
    else:
        candidates = candidates.loc[
            candidates["dimension_name"].fillna("").astype(str).eq("")
        ].copy()

    if len(candidates) != 1:
        reason = "no governed annual membership row" if candidates.empty else "ambiguous governed annual membership rows"
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "annual_flow_membership.csv",
            {
                "unsupported": True,
                "reason": reason,
                "metric_id": metric_id,
                "period": period,
                "Currency": currency,
                "dimension_name": spec.dimension_name,
                "dimension_value": dimension_value,
                "candidate_rows": len(candidates),
            },
            "Governed annual flow membership must resolve exactly one annual cell; no report-layer membership inference was allowed.",
            candidates,
            [],
        )

    selected = candidates.iloc[[0]].copy()
    if _text(selected.iloc[0]["lineage_version"]) != ANNUAL_FLOW_MEMBERSHIP_VERSION:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "annual_flow_membership.csv",
            {"unsupported": True, "reason": "unsupported annual lineage version", "metric_id": metric_id},
            "Annual flow lineage version is not governed by this executor.",
            selected,
            [],
        )

    matched = float(pd.to_numeric(selected.iloc[0]["value"], errors="coerce"))
    residual = matched - display_value
    status = _legacy.STATUS_OK if abs(residual) <= tolerance else _legacy.STATUS_RESIDUAL_WARNING
    filters = {
        "metric_id": metric_id,
        "period": period,
        "Currency": currency,
        "dimension_name": spec.dimension_name,
        "dimension_value": dimension_value,
        "drilldown_cell_id": spec.monthly_flow_cell_id,
        "aggregation_rule": _text(selected.iloc[0]["aggregation_rule"]),
        "measure": _text(selected.iloc[0]["measure_id"]),
        "lineage_version": _text(selected.iloc[0]["lineage_version"]),
        "monthly_governed_cell_ids": _text(selected.iloc[0]["monthly_governed_cell_ids"]),
        "source_member_ids": _text(selected.iloc[0]["source_member_ids"]),
        "member_months": _text(selected.iloc[0]["member_months"]),
    }
    caveat = (
        "Annual additive flow membership consumed from annual_flow_membership_v1; "
        "professional reporting did not reclassify monthly semantic rows."
    )
    sections = [("Governed annual flow membership", selected)]
    return (
        status,
        matched,
        residual,
        "annual_governed_membership",
        "annual_flow_membership.csv",
        filters,
        caveat,
        selected,
        sections,
    )
