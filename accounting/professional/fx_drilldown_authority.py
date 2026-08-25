from __future__ import annotations

"""Single FX measure/grain authority for professional drilldowns.

This module is deliberately FX-specific. It resolves only the two supported FX
row grains (Currency total and Box x Currency) and the approved physical amount
measures. Ambiguous rows fail closed instead of acquiring a default measure or
silently losing a Box dimension.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from accounting.contracts.atomic_flow_drilldowns import resolve_flow_cell_spec
from accounting.contracts.semantic_measures import resolve_semantic_measure

FX_TREASURY_TABLE_IDS = frozenset(
    {
        "monthly_tables_fx_treasury_all_measures",
        "monthly_tables_fx_treasury_amount_in",
        "monthly_tables_fx_treasury_amount_out",
        "monthly_tables_fx_treasury_net_amount",
        "monthly_tables_fx_treasury_compact",
    }
)
FX_MEASURES = frozenset({"amount_in", "amount_out", "net_amount", "amount_abs"})
FXGrain = Literal["currency_total", "box_currency"]

_SINGLE_MEASURE_TABLES = {
    "monthly_tables_fx_treasury_amount_in": "amount_in",
    "monthly_tables_fx_treasury_amount_out": "amount_out",
    "monthly_tables_fx_treasury_net_amount": "net_amount",
}
_METRIC_SUBBUCKETS = {
    "fx_conversion_proceeds": "fx_conversion_proceeds",
    "fx_conversion_outflow": "fx_conversion_outflow",
    "fx_cost_or_spread": "fx_cost_or_spread",
}


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True, slots=True)
class FXDrilldownResolution:
    measure: str
    grain: FXGrain
    currency: str
    box: str = ""
    semantic_subbucket: str = ""
    unsupported_reason: str = ""

    @property
    def supported(self) -> bool:
        return not self.unsupported_reason


def _metric_measure(metric: str) -> str:
    if metric in FX_MEASURES:
        return metric
    if metric in _METRIC_SUBBUCKETS:
        return resolve_semantic_measure("treasury_fx", metric) or ""
    if metric == "fx_net":
        return "net_amount"
    return ""


def resolve_fx_drilldown(
    table_id: str,
    row: pd.Series,
) -> FXDrilldownResolution | None:
    """Resolve one FX row to explicit measure and explicit supported grain."""

    if table_id not in FX_TREASURY_TABLE_IDS:
        return None

    currency = _text(row.get("Currency"))
    box = _text(row.get("Box"))
    cell_id = _text(row.get("drilldown_cell_id"))
    subbucket = _text(row.get("semantic_subbucket"))

    grain: FXGrain = "box_currency" if box else "currency_total"
    contract_measure = ""

    if cell_id:
        spec = resolve_flow_cell_spec(cell_id)
        if spec is None or not cell_id.startswith("flow.fx."):
            return FXDrilldownResolution(
                "", grain, currency, box, subbucket,
                f"unsupported FX drilldown_cell_id: {cell_id}",
            )
        contract_measure = resolve_semantic_measure(*spec.measure_ref) or ""
        if "Box" in spec.grain:
            grain = "box_currency"
            if not box:
                return FXDrilldownResolution(
                    contract_measure,
                    grain,
                    currency,
                    "",
                    subbucket,
                    "missing Box for Box x Currency FX contract",
                )
        if len(spec.semantic_members) == 1:
            contract_subbucket = spec.semantic_members[0][1]
            if subbucket and contract_subbucket and subbucket != contract_subbucket:
                return FXDrilldownResolution(
                    contract_measure,
                    grain,
                    currency,
                    box,
                    subbucket,
                    "semantic_subbucket conflicts with FX drilldown contract",
                )
            subbucket = subbucket or contract_subbucket

    explicit_measure = _text(row.get("measure"))
    if explicit_measure and explicit_measure not in FX_MEASURES:
        return FXDrilldownResolution(
            "", grain, currency, box, subbucket,
            f"unsupported FX measure: {explicit_measure}",
        )

    metric = _text(row.get("metric")).casefold()
    metric_measure = _metric_measure(metric)
    table_measure = _SINGLE_MEASURE_TABLES.get(table_id, "")
    measure = explicit_measure or contract_measure or table_measure or metric_measure

    if contract_measure and explicit_measure and explicit_measure != contract_measure:
        return FXDrilldownResolution(
            explicit_measure,
            grain,
            currency,
            box,
            subbucket,
            "explicit measure conflicts with FX drilldown contract",
        )
    if not measure:
        return FXDrilldownResolution(
            "", grain, currency, box, subbucket,
            "ambiguous FX row has no explicit recognized measure",
        )
    if not currency:
        return FXDrilldownResolution(
            measure, grain, "", box, subbucket,
            "missing Currency would risk cross-currency aggregation",
        )

    if not subbucket and metric in _METRIC_SUBBUCKETS:
        subbucket = _METRIC_SUBBUCKETS[metric]

    return FXDrilldownResolution(measure, grain, currency, box, subbucket)


def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    """Compatibility measure selector backed by the single FX authority."""

    resolution = resolve_fx_drilldown(table_id, row)
    if resolution is None or resolution.measure not in FX_MEASURES:
        return ""
    return resolution.measure
