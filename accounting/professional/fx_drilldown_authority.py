from __future__ import annotations

"""Single FX measure/grain authority for professional drilldowns."""

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
        "cash_annual_box_flow_bridge_wide",
    }
)
FX_MEASURES = frozenset({"amount_in", "amount_out", "net_amount", "amount_abs"})
FXGrain = Literal["currency_total", "box_currency"]

_TABLE_MEASURE = {
    "monthly_tables_fx_treasury_amount_in": "amount_in",
    "monthly_tables_fx_treasury_amount_out": "amount_out",
    "monthly_tables_fx_treasury_net_amount": "net_amount",
}
_METRIC_SUBBUCKET = {
    "fx_conversion_proceeds": "fx_conversion_proceeds",
    "fx_conversion_outflow": "fx_conversion_outflow",
    "fx_cost_or_spread": "fx_cost_or_spread",
}


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


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
    if metric == "fx_net":
        return "net_amount"
    if metric in _METRIC_SUBBUCKET:
        return resolve_semantic_measure("treasury_fx", metric) or ""
    return ""


def _bridge_is_fx(row: pd.Series) -> bool:
    cell_id = _text(row.get("drilldown_cell_id"))
    metric = _text(row.get("metric")).casefold()
    line = _text(row.get("line")).casefold()
    return (
        _text(row.get("semantic_bucket")) == "treasury_fx"
        or cell_id.startswith("flow.fx.")
        or metric.startswith("fx_")
        or "fx" in line
        or "treasury" in line
        or "cambio" in line
    )


def resolve_fx_drilldown(
    table_id: str,
    row: pd.Series,
) -> FXDrilldownResolution | None:
    if table_id not in FX_TREASURY_TABLE_IDS:
        return None
    if table_id == "cash_annual_box_flow_bridge_wide" and not _bridge_is_fx(row):
        return None

    currency = _text(row.get("Currency"))
    box = _text(row.get("Box"))
    subbucket = _text(row.get("semantic_subbucket"))
    cell_id = _text(row.get("drilldown_cell_id"))
    grain: FXGrain = "box_currency" if box else "currency_total"
    contract_measure = ""

    def result(
        measure: str,
        reason: str = "",
    ) -> FXDrilldownResolution:
        return FXDrilldownResolution(
            measure,
            grain,
            currency,
            box,
            subbucket,
            reason,
        )

    if cell_id:
        spec = resolve_flow_cell_spec(cell_id)
        if spec is None or not cell_id.startswith("flow.fx."):
            return result("", f"unsupported FX drilldown_cell_id: {cell_id}")
        contract_measure = resolve_semantic_measure(*spec.measure_ref) or ""
        if "Box" in spec.grain:
            grain = "box_currency"
            if not box:
                return result(
                    contract_measure,
                    "missing Box for Box x Currency FX contract",
                )
        if len(spec.semantic_members) == 1:
            contract_subbucket = spec.semantic_members[0][1]
            if (
                subbucket
                and contract_subbucket
                and subbucket != contract_subbucket
            ):
                return result(
                    contract_measure,
                    "semantic_subbucket conflicts with FX drilldown contract",
                )
            subbucket = subbucket or contract_subbucket

    explicit = _text(row.get("measure"))
    if explicit and explicit not in FX_MEASURES:
        return result("", f"unsupported FX measure: {explicit}")
    if contract_measure and explicit and explicit != contract_measure:
        return result(
            explicit,
            "explicit measure conflicts with FX drilldown contract",
        )

    metric = _text(row.get("metric")).casefold()
    measure = (
        explicit
        or contract_measure
        or _TABLE_MEASURE.get(table_id, "")
        or _metric_measure(metric)
    )
    if not measure:
        return result("", "ambiguous FX row has no explicit recognized measure")
    if not currency:
        return result(
            measure,
            "missing Currency would risk cross-currency aggregation",
        )
    if not subbucket and metric in _METRIC_SUBBUCKET:
        subbucket = _METRIC_SUBBUCKET[metric]
    return result(measure)


def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    """Compatibility selector; grain enforcement belongs to displayed cells."""

    resolved = resolve_fx_drilldown(table_id, row)
    return (
        resolved.measure
        if resolved and resolved.measure in FX_MEASURES
        else ""
    )
