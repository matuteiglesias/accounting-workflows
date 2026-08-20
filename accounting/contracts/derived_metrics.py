"""Typed, declarative contracts for governed derived metrics.

Wave 5 PR17 defines formula identity and policy only. It deliberately has no
production consumer: professional execution migrates in a later PR.

Derived metrics may combine only already-governed scalar/component authorities.
This contract never reclassifies ledger rows and never embeds semantic filters,
labels, lambdas, or executable expressions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final, Literal, Mapping


DerivedAuthorityMode = Literal[
    "source_value_with_formula_reconciliation",
    "computed_derived",
]
DerivedOperation = Literal[
    "subtract",
    "add_subtract",
    "ratio",
    "period_delta",
]
DerivedGrainDimension = Literal["period", "Currency", "Box"]
DerivedPeriodGrain = Literal["M", "Y"]
DerivedAnnualPolicy = Literal[
    "source_authority_annual_value",
    "recompute_from_annual_components",
    "not_applicable",
]
DerivedMissingPolicy = Literal["unavailable"]
DerivedZeroDenominatorPolicy = Literal["not_applicable"]

DERIVED_METRIC_SPECS_VERSION: Final = "derived_metric_specs_v1"
DEFAULT_DERIVED_TOLERANCE: Final = 1e-6

_PERIOD_CURRENCY_GRAIN: Final[tuple[DerivedGrainDimension, ...]] = (
    "period",
    "Currency",
)
_PERIOD_CURRENCY_BOX_GRAIN: Final[tuple[DerivedGrainDimension, ...]] = (
    "period",
    "Currency",
    "Box",
)

# Stable component references approved for the v1 DAG. These are identities,
# not instructions for how to rediscover rows. The future executor must obtain
# the value from the already-governed producer for each reference.
APPROVED_DERIVED_COMPONENT_REFS: Final[frozenset[str]] = frozenset(
    {
        "metric:IS.REVENUE.OPERATING",
        "metric:IS.RENT.TOTAL",
        "metric:IS.OPEX.PROPERTY",
        "metric:IS.NET.OPERATING",
        "metric:FUND.CONTRIB.TOTAL",
        "metric:DIST.DRAWS.PERSONAL",
        "metric:COV.NET.AFTER_DRAWS",
        "cash.control.inferred_box_motor",
    }
)


def _validate_spec_id(spec_id: str) -> None:
    if not isinstance(spec_id, str):
        raise TypeError("spec_id must be a string")
    if not spec_id or spec_id != spec_id.strip():
        raise ValueError("spec_id must be non-empty and normalized")
    if not spec_id.startswith("derived."):
        raise ValueError("spec_id must begin with 'derived.'")


@dataclass(frozen=True, slots=True)
class DerivedMetricSpec:
    """Governed identity and closed formula policy for one derived metric.

    ``source_value_with_formula_reconciliation`` means the upstream metric is
    authoritative and the formula is explanatory/reconciliatory only. It must
    never create a competing production value.

    ``computed_derived`` means the value is computed from the declared governed
    component references using the declared closed operation.
    """

    spec_id: str
    authority_mode: DerivedAuthorityMode
    operation: DerivedOperation
    component_refs: tuple[str, ...]
    grain: tuple[DerivedGrainDimension, ...]
    period_grains: tuple[DerivedPeriodGrain, ...]
    annual_policy: DerivedAnnualPolicy
    source_value_ref: str | None = None
    missing_component_policy: DerivedMissingPolicy = "unavailable"
    zero_denominator_policy: DerivedZeroDenominatorPolicy | None = None
    tolerance: float = DEFAULT_DERIVED_TOLERANCE

    def __post_init__(self) -> None:
        _validate_spec_id(self.spec_id)

        if self.authority_mode not in {
            "source_value_with_formula_reconciliation",
            "computed_derived",
        }:
            raise ValueError(f"Unsupported authority mode: {self.authority_mode!r}")
        if self.operation not in {
            "subtract",
            "add_subtract",
            "ratio",
            "period_delta",
        }:
            raise ValueError(f"Unsupported derived operation: {self.operation!r}")
        if not self.component_refs:
            raise ValueError("DerivedMetricSpec.component_refs must be non-empty")
        if len(set(self.component_refs)) != len(self.component_refs):
            raise ValueError("DerivedMetricSpec.component_refs must be unique")
        unknown_refs = set(self.component_refs) - APPROVED_DERIVED_COMPONENT_REFS
        if unknown_refs:
            raise ValueError(f"Unknown governed component refs: {sorted(unknown_refs)!r}")
        if self.grain not in {_PERIOD_CURRENCY_GRAIN, _PERIOD_CURRENCY_BOX_GRAIN}:
            raise ValueError(
                "DerivedMetricSpec.grain must be (period, Currency) or "
                "(period, Currency, Box)"
            )
        if not self.period_grains or any(g not in {"M", "Y"} for g in self.period_grains):
            raise ValueError("period_grains must contain only M and/or Y")
        if len(set(self.period_grains)) != len(self.period_grains):
            raise ValueError("period_grains must be unique")
        if self.missing_component_policy != "unavailable":
            raise ValueError("Missing components must fail closed as unavailable")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

        if self.authority_mode == "source_value_with_formula_reconciliation":
            if not self.source_value_ref or not self.source_value_ref.startswith("metric:"):
                raise ValueError(
                    "Source-authority derived specs require a governed metric source_value_ref"
                )
        elif self.source_value_ref is not None:
            raise ValueError("Computed derived specs must not declare source_value_ref")

        if self.operation == "ratio":
            if len(self.component_refs) != 2:
                raise ValueError("ratio requires exactly numerator and denominator refs")
            if self.zero_denominator_policy != "not_applicable":
                raise ValueError("ratio zero denominator policy must be not_applicable")
        elif self.zero_denominator_policy is not None:
            raise ValueError("zero_denominator_policy applies only to ratio specs")

        if self.operation == "subtract" and len(self.component_refs) != 2:
            raise ValueError("subtract requires exactly two component refs")
        if self.operation == "add_subtract" and len(self.component_refs) != 3:
            raise ValueError("add_subtract requires exactly three component refs")
        if self.operation == "period_delta":
            if self.component_refs != ("cash.control.inferred_box_motor",):
                raise ValueError(
                    "v1 period_delta is governed only for inferred box control"
                )
            if self.grain != _PERIOD_CURRENCY_BOX_GRAIN:
                raise ValueError("period_delta requires period/Currency/Box grain")
            if self.period_grains != ("M",):
                raise ValueError("period_delta v1 is monthly only")
            if self.annual_policy != "not_applicable":
                raise ValueError("period_delta v1 has no annualization")

        if self.annual_policy == "source_authority_annual_value":
            if self.authority_mode != "source_value_with_formula_reconciliation":
                raise ValueError(
                    "source_authority_annual_value requires source-value authority mode"
                )
            if "Y" not in self.period_grains:
                raise ValueError("source annual authority requires Y period grain")
        if self.annual_policy == "recompute_from_annual_components":
            if self.operation != "ratio" or self.period_grains != ("Y",):
                raise ValueError(
                    "v1 annual component recomputation is reserved for annual ratios"
                )


def _source_reconciliation_spec(
    *,
    spec_id: str,
    source_metric_id: str,
    operation: Literal["subtract", "add_subtract"],
    component_refs: tuple[str, ...],
) -> DerivedMetricSpec:
    return DerivedMetricSpec(
        spec_id=spec_id,
        authority_mode="source_value_with_formula_reconciliation",
        operation=operation,
        component_refs=component_refs,
        grain=_PERIOD_CURRENCY_GRAIN,
        period_grains=("M", "Y"),
        annual_policy="source_authority_annual_value",
        source_value_ref=f"metric:{source_metric_id}",
    )


def _annual_ratio_spec(
    *,
    spec_id: str,
    numerator_ref: str,
    denominator_ref: str,
) -> DerivedMetricSpec:
    return DerivedMetricSpec(
        spec_id=spec_id,
        authority_mode="computed_derived",
        operation="ratio",
        component_refs=(numerator_ref, denominator_ref),
        grain=_PERIOD_CURRENCY_GRAIN,
        period_grains=("Y",),
        annual_policy="recompute_from_annual_components",
        zero_denominator_policy="not_applicable",
    )


_SPECS = (
    _source_reconciliation_spec(
        spec_id="derived.net_operating",
        source_metric_id="IS.NET.OPERATING",
        operation="subtract",
        component_refs=(
            "metric:IS.REVENUE.OPERATING",
            "metric:IS.OPEX.PROPERTY",
        ),
    ),
    _source_reconciliation_spec(
        spec_id="derived.coverage_after_draws",
        source_metric_id="COV.NET.AFTER_DRAWS",
        operation="add_subtract",
        component_refs=(
            "metric:IS.NET.OPERATING",
            "metric:FUND.CONTRIB.TOTAL",
            "metric:DIST.DRAWS.PERSONAL",
        ),
    ),
    _annual_ratio_spec(
        spec_id="derived.savings_rate",
        numerator_ref="metric:COV.NET.AFTER_DRAWS",
        denominator_ref="metric:IS.NET.OPERATING",
    ),
    _annual_ratio_spec(
        spec_id="derived.operating_margin",
        numerator_ref="metric:IS.NET.OPERATING",
        denominator_ref="metric:IS.REVENUE.OPERATING",
    ),
    _annual_ratio_spec(
        spec_id="derived.opex_to_rent",
        numerator_ref="metric:IS.OPEX.PROPERTY",
        denominator_ref="metric:IS.RENT.TOTAL",
    ),
    _annual_ratio_spec(
        spec_id="derived.draws_to_operating_result",
        numerator_ref="metric:DIST.DRAWS.PERSONAL",
        denominator_ref="metric:IS.NET.OPERATING",
    ),
    DerivedMetricSpec(
        spec_id="derived.diagnostic_box_level",
        authority_mode="computed_derived",
        operation="period_delta",
        component_refs=("cash.control.inferred_box_motor",),
        grain=_PERIOD_CURRENCY_BOX_GRAIN,
        period_grains=("M",),
        annual_policy="not_applicable",
    ),
)

DERIVED_METRIC_SPECS_V1: Final[Mapping[str, DerivedMetricSpec]] = MappingProxyType(
    {spec.spec_id: spec for spec in _SPECS}
)

if len(DERIVED_METRIC_SPECS_V1) != len(_SPECS):
    raise ValueError("Duplicate DerivedMetricSpec.spec_id in derived_metric_specs_v1")
if any(
    callable(getattr(spec, field.name))
    for spec in _SPECS
    for field in fields(spec)
):
    raise TypeError("DerivedMetricSpec fields must remain declarative")


def resolve_derived_metric_spec(spec_id: str) -> DerivedMetricSpec | None:
    """Return a v1 derived metric spec by stable ID."""

    return DERIVED_METRIC_SPECS_V1.get(str(spec_id).strip())
