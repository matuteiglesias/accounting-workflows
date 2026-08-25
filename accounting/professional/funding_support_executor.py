from __future__ import annotations

"""Governed professional executor for typed funding/support membership.

This module does not redefine core funding. It consumes
``funding_support_specs_v1`` and preserves the distinction between the narrow
``FUND.CONTRIB.TOTAL`` accounting metric and broader support views.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from accounting.contracts.funding_support import (
    FUNDING_SUPPORT_SPECS_VERSION,
    classify_funding_support,
)


@dataclass(frozen=True, slots=True)
class FundingRoute:
    metric_id: str
    selector: Callable[[pd.DataFrame], pd.Series]
    dimension: str = ""


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _dimension_value(row: pd.Series, dimension: str) -> str:
    direct = _text(row.get(dimension))
    if direct:
        return direct
    if _text(row.get("dimension_name")) == dimension:
        return _text(row.get("dimension_value"))
    return ""


def _route(metric_id: str) -> FundingRoute | None:
    routes = {
        "FUND.CONTRIB.TOTAL": FundingRoute(
            metric_id,
            lambda members: members["support_kind"].eq("core_contribution"),
        ),
        "FUND.CONTRIB.BY_FUNDING_ACTOR": FundingRoute(
            metric_id,
            lambda members: pd.Series(True, index=members.index),
            "funding_actor",
        ),
        "FUND.CONTRIB.BY_CHANNEL": FundingRoute(
            metric_id,
            lambda members: pd.Series(True, index=members.index),
            "funding_channel",
        ),
        "FUND.CONTRIB.BY_CASH_EFFECT": FundingRoute(
            metric_id,
            lambda members: pd.Series(True, index=members.index),
            "cash_effect",
        ),
        "FUND.CONTRIB.BY_TARGET_BOX": FundingRoute(
            metric_id,
            lambda members: pd.Series(True, index=members.index),
            "target_box",
        ),
        "FUND.CONTRIB.DIRECT_OBLIGATION": FundingRoute(
            metric_id,
            lambda members: members["support_kind"].eq("direct_obligation_payment"),
        ),
        "FUND.CONTRIB.CASH_TO_BOX": FundingRoute(
            metric_id,
            lambda members: members["cash_effect"]
            .fillna("")
            .astype(str)
            .eq("cash_in_box"),
        ),
        "FUND.CONTRIB.DEBT_LINKED": FundingRoute(
            metric_id,
            lambda members: members["support_kind"].eq("debt_linked_support"),
        ),
    }
    return routes.get(metric_id)


def execute_annual_funding_support(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    split: pd.DataFrame,
    tolerance: float,
):
    """Resolve one supported annual funding/support cell or return ``None``.

    ``None`` means the metric is outside the typed funding/support surface.
    Once a typed metric is recognized, missing source/grain fails closed.
    """

    metric_id = _text(row.get("metric_id"))
    route = _route(metric_id)
    if route is None:
        return None
    if not str(period).isdigit() or len(str(period)) != 4:
        return None

    currency = _text(row.get("Currency"))
    base_filters = {
        "metric_id": metric_id,
        "period": str(period),
        "Currency": currency,
        "contract_version": FUNDING_SUPPORT_SPECS_VERSION,
        "executor": "governed_funding_support_v1",
    }
    if not currency:
        return (
            "unsupported",
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "unsupported": True, "reason": "missing Currency"},
            "Typed funding/support execution requires explicit native Currency.",
            pd.DataFrame(),
            [],
        )
    if split is None or split.empty:
        return (
            "error",
            0.0,
            -display_value,
            "missing_source",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "error": "missing monthly_flow_semantic_split.csv"},
            "",
            pd.DataFrame(),
            [],
        )

    members = classify_funding_support(split, strict=True).copy()
    if members.empty:
        selected = members
    else:
        selected = members.loc[
            members["period"].fillna("").astype(str).str.startswith(f"{period}-")
            & members["Currency"].fillna("").astype(str).str.strip().eq(currency)
        ].copy()
        selected = selected.loc[route.selector(selected)].copy()

    dimension_value = ""
    if route.dimension:
        dimension_value = _dimension_value(row, route.dimension)
        if not dimension_value:
            return (
                "unsupported",
                0.0,
                -display_value,
                "unsupported",
                "monthly_flow_semantic_split.csv",
                {
                    **base_filters,
                    "unsupported": True,
                    "reason": f"missing governed dimension {route.dimension}",
                },
                "Typed funding/support row is missing required dimension metadata.",
                pd.DataFrame(),
                [],
            )
        if route.dimension not in selected.columns:
            return (
                "unsupported",
                0.0,
                -display_value,
                "unsupported",
                "monthly_flow_semantic_split.csv",
                {
                    **base_filters,
                    "unsupported": True,
                    "reason": f"source missing dimension {route.dimension}",
                },
                "Typed funding/support source does not satisfy the requested grain.",
                pd.DataFrame(),
                [],
            )
        selected = selected.loc[
            selected[route.dimension]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(dimension_value)
        ].copy()

    matched = float(
        pd.to_numeric(
            selected.get("support_amount", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    residual = matched - display_value
    status = (
        "empty"
        if selected.empty
        else "ok"
        if abs(residual) <= tolerance
        else "residual_warning"
    )
    filters = {
        **base_filters,
        "support_kinds": sorted(
            selected.get("support_kind", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .unique()
            .tolist()
        ),
        "dimension_name": route.dimension,
        "dimension_value": dimension_value,
        "measure": "support_amount",
    }
    return (
        status,
        matched,
        residual,
        "governed_funding_support",
        "monthly_flow_semantic_split.csv",
        filters,
        (
            "Funding/support membership comes from funding_support_specs_v1; "
            "core contribution remains distinct from broader direct-obligation "
            "and debt-linked support."
        ),
        selected,
        [("Governed funding/support members", selected)],
    )
