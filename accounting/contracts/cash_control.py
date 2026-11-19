"""Replacement contract for recorded Box movement controls.

These specs are additive to the validated-cash contracts.  They identify the
authoritative columns already produced by ``monthly_cash_accountability.csv``;
they do not create another calculation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping


CASH_CONTROL_SPECS_VERSION: Final = "cash_control_specs_v1"
CashControlMetricId = Literal["CTRL.BOX.NET_MOVEMENT", "CTRL.BOX.CUM_NET"]
CashControlAggregation = Literal["period_flow", "month_end_stock"]
CashControlAnnualization = Literal["sum_period_flows", "latest_month_end"]


@dataclass(frozen=True, slots=True)
class CashControlSpec:
    """Executable identity and interpretation of one Box control metric."""

    metric_id: CashControlMetricId
    source_contract: Literal["monthly_cash_accountability"]
    grain: tuple[Literal["period", "period_end", "Box", "Currency"], ...]
    value_ref: Literal["net_cash_flow", "closing_control"]
    aggregation: CashControlAggregation
    annualization: CashControlAnnualization
    origin: Literal["defined_zero_origin"] = "defined_zero_origin"
    currency_policy: Literal["native_only_no_fx_conversion"] = (
        "native_only_no_fx_conversion"
    )
    transfer_policy: Literal["include_recorded_per_box_legs_no_consolidation"] = (
        "include_recorded_per_box_legs_no_consolidation"
    )
    opening_balance_policy: Literal["unknown_does_not_become_zero"] = (
        "unknown_does_not_become_zero"
    )
    liquidity_claim: Literal["not_actual_cash_or_liquidity"] = (
        "not_actual_cash_or_liquidity"
    )

    def __post_init__(self) -> None:
        if self.source_contract != "monthly_cash_accountability":
            raise ValueError("cash controls must source monthly_cash_accountability")
        if self.grain != ("period", "period_end", "Box", "Currency"):
            raise ValueError("cash controls require month, month-end, Box, Currency grain")
        if self.metric_id == "CTRL.BOX.NET_MOVEMENT":
            if self.value_ref != "net_cash_flow" or self.aggregation != "period_flow":
                raise ValueError("NET_MOVEMENT must be the monthly net movement flow")
            if self.annualization != "sum_period_flows":
                raise ValueError("NET_MOVEMENT annualization must sum monthly flows")
        elif self.metric_id == "CTRL.BOX.CUM_NET":
            if self.value_ref != "closing_control" or self.aggregation != "month_end_stock":
                raise ValueError("CUM_NET must be the zero-origin month-end control stock")
            if self.annualization != "latest_month_end":
                raise ValueError("CUM_NET annualization must select the latest month-end")
        else:
            raise ValueError(f"unsupported cash control metric: {self.metric_id!r}")
        if self.origin != "defined_zero_origin":
            raise ValueError("cash controls require an explicit zero-origin convention")
        if self.currency_policy != "native_only_no_fx_conversion":
            raise ValueError("cash controls cannot convert or combine currencies")
        if self.transfer_policy != "include_recorded_per_box_legs_no_consolidation":
            raise ValueError("cash controls must retain per-Box transfer legs")
        if self.opening_balance_policy != "unknown_does_not_become_zero":
            raise ValueError("unknown opening balances must not be silently treated as zero")
        if self.liquidity_claim != "not_actual_cash_or_liquidity":
            raise ValueError("cash controls cannot claim actual liquidity")


_SPECS = (
    CashControlSpec(
        metric_id="CTRL.BOX.NET_MOVEMENT",
        source_contract="monthly_cash_accountability",
        grain=("period", "period_end", "Box", "Currency"),
        value_ref="net_cash_flow",
        aggregation="period_flow",
        annualization="sum_period_flows",
    ),
    CashControlSpec(
        metric_id="CTRL.BOX.CUM_NET",
        source_contract="monthly_cash_accountability",
        grain=("period", "period_end", "Box", "Currency"),
        value_ref="closing_control",
        aggregation="month_end_stock",
        annualization="latest_month_end",
    ),
)

CASH_CONTROL_SPECS_V1: Final[Mapping[str, CashControlSpec]] = MappingProxyType(
    {spec.metric_id: spec for spec in _SPECS}
)

if len(CASH_CONTROL_SPECS_V1) != len(_SPECS):
    raise ValueError("duplicate cash control metric ID")


def resolve_cash_control_spec(metric_id: str) -> CashControlSpec | None:
    """Resolve a replacement metric without changing existing metric IDs."""

    return CASH_CONTROL_SPECS_V1.get(str(metric_id).strip())
