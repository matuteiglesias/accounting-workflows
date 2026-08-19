"""Typed, declarative contracts for resolved-debt position and activity drilldowns.

This module defines Wave 4 contracts only. Professional consumers are migrated in
separate PRs. Debt position remains a stock/snapshot contract; debt activity
remains a period-flow contract. Cash position is deliberately out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final, Literal, Mapping


DebtPositionSourceContract = Literal["monthly_debt_position"]
DebtActivitySourceContract = Literal["monthly_debt_activity"]
DebtPositionComponent = Literal["principal", "interest", "total"]
DebtPositionValueRef = Literal["open_principal", "open_interest", "open_total"]
DebtActivityType = Literal[
    "new_claim",
    "interest_accrual",
    "repayment",
    "adjustment",
    "net_change",
]
DebtActivityMeasureRef = Literal[
    "new_principal",
    "interest_accrued",
    "repayments",
    "adjustments",
    "net_change",
]
DebtPositionGrainDimension = Literal[
    "period",
    "Currency",
    "debtor",
    "creditor",
    "component",
]
DebtActivityGrainDimension = Literal[
    "period",
    "Currency",
    "debtor",
    "creditor",
    "activity_type",
]

DEBT_POSITION_SPECS_VERSION: Final = "debt_position_specs_v1"
DEBT_ACTIVITY_SPECS_VERSION: Final = "debt_activity_specs_v1"

_POSITION_SOURCE: Final[DebtPositionSourceContract] = "monthly_debt_position"
_ACTIVITY_SOURCE: Final[DebtActivitySourceContract] = "monthly_debt_activity"
_POSITION_GRAIN: Final[tuple[DebtPositionGrainDimension, ...]] = (
    "period",
    "Currency",
    "debtor",
    "creditor",
    "component",
)
_ACTIVITY_GRAIN: Final[tuple[DebtActivityGrainDimension, ...]] = (
    "period",
    "Currency",
    "debtor",
    "creditor",
    "activity_type",
)
_POSITION_VALUE_FOR_COMPONENT: Final[dict[str, str]] = {
    "principal": "open_principal",
    "interest": "open_interest",
    "total": "open_total",
}
_ACTIVITY_MEASURE_FOR_TYPE: Final[dict[str, str]] = {
    "new_claim": "new_principal",
    "interest_accrual": "interest_accrued",
    "repayment": "repayments",
    "adjustment": "adjustments",
    "net_change": "net_change",
}


def _validate_spec_id(spec_id: str, prefix: str) -> None:
    if not isinstance(spec_id, str):
        raise TypeError("spec_id must be a string")
    if not spec_id or spec_id != spec_id.strip():
        raise ValueError("spec_id must be non-empty and normalized")
    if not spec_id.startswith(prefix):
        raise ValueError(f"spec_id must begin with {prefix!r}")


@dataclass(frozen=True, slots=True)
class DebtPositionSpec:
    """Governed identity and selection policy for one debt-position component.

    The contract preserves the current physical value columns rather than
    prematurely normalizing all components to ``open_amount``. A future
    consumer must select the latest *valid* ``as_of_date`` and return
    unavailable when no valid as-of value exists; lexical fallback is not an
    approved stock-selection rule.
    """

    spec_id: str
    source_contract: DebtPositionSourceContract
    grain: tuple[DebtPositionGrainDimension, ...]
    component: DebtPositionComponent
    value_ref: DebtPositionValueRef
    aggregation: Literal["snapshot"] = "snapshot"
    selection: Literal["latest_valid_as_of_date"] = "latest_valid_as_of_date"
    as_of_field: Literal["as_of_date"] = "as_of_date"
    invalid_as_of_policy: Literal["unavailable"] = "unavailable"
    annualization: Literal[
        "latest_period_then_latest_valid_as_of_date"
    ] = "latest_period_then_latest_valid_as_of_date"

    def __post_init__(self) -> None:
        _validate_spec_id(self.spec_id, "debt.position.")
        if self.source_contract != _POSITION_SOURCE:
            raise ValueError(
                f"Unsupported debt-position source contract: {self.source_contract!r}"
            )
        if self.grain != _POSITION_GRAIN:
            raise ValueError(
                "DebtPositionSpec.grain must be exactly "
                "(period, Currency, debtor, creditor, component)"
            )
        expected_value = _POSITION_VALUE_FOR_COMPONENT.get(str(self.component))
        if expected_value is None:
            raise ValueError(f"Unsupported debt-position component: {self.component!r}")
        if self.value_ref != expected_value:
            raise ValueError(
                "DebtPositionSpec.value_ref must preserve the characterized physical "
                f"mapping for component={self.component!r}: expected {expected_value!r}"
            )
        if self.aggregation != "snapshot":
            raise ValueError("DebtPositionSpec.aggregation must be 'snapshot'")
        if self.selection != "latest_valid_as_of_date":
            raise ValueError(
                "DebtPositionSpec.selection must be 'latest_valid_as_of_date'"
            )
        if self.as_of_field != "as_of_date":
            raise ValueError("DebtPositionSpec.as_of_field must be 'as_of_date'")
        if self.invalid_as_of_policy != "unavailable":
            raise ValueError(
                "DebtPositionSpec.invalid_as_of_policy must fail closed as 'unavailable'"
            )
        if self.annualization != "latest_period_then_latest_valid_as_of_date":
            raise ValueError(
                "DebtPositionSpec annualization must select a closing snapshot, not sum periods"
            )


@dataclass(frozen=True, slots=True)
class DebtActivitySpec:
    """Governed identity and measure for one resolved-debt activity flow."""

    spec_id: str
    source_contract: DebtActivitySourceContract
    grain: tuple[DebtActivityGrainDimension, ...]
    activity_type: DebtActivityType
    measure_ref: DebtActivityMeasureRef
    aggregation: Literal["sum_flow"] = "sum_flow"
    annualization: Literal["sum_periods"] = "sum_periods"

    def __post_init__(self) -> None:
        _validate_spec_id(self.spec_id, "debt.activity.")
        if self.source_contract != _ACTIVITY_SOURCE:
            raise ValueError(
                f"Unsupported debt-activity source contract: {self.source_contract!r}"
            )
        if self.grain != _ACTIVITY_GRAIN:
            raise ValueError(
                "DebtActivitySpec.grain must be exactly "
                "(period, Currency, debtor, creditor, activity_type)"
            )
        expected_measure = _ACTIVITY_MEASURE_FOR_TYPE.get(str(self.activity_type))
        if expected_measure is None:
            raise ValueError(f"Unsupported debt activity type: {self.activity_type!r}")
        if self.measure_ref != expected_measure:
            raise ValueError(
                "DebtActivitySpec.measure_ref must preserve the characterized mapping "
                f"for activity_type={self.activity_type!r}: expected {expected_measure!r}"
            )
        if self.aggregation != "sum_flow":
            raise ValueError("DebtActivitySpec.aggregation must be 'sum_flow'")
        if self.annualization != "sum_periods":
            raise ValueError(
                "DebtActivitySpec annualization must sum period flows, not select a snapshot"
            )


def _position_spec(
    component: DebtPositionComponent,
    value_ref: DebtPositionValueRef,
) -> DebtPositionSpec:
    return DebtPositionSpec(
        spec_id=f"debt.position.{component}",
        source_contract=_POSITION_SOURCE,
        grain=_POSITION_GRAIN,
        component=component,
        value_ref=value_ref,
    )


def _activity_spec(
    activity_type: DebtActivityType,
    measure_ref: DebtActivityMeasureRef,
) -> DebtActivitySpec:
    return DebtActivitySpec(
        spec_id=f"debt.activity.{activity_type}",
        source_contract=_ACTIVITY_SOURCE,
        grain=_ACTIVITY_GRAIN,
        activity_type=activity_type,
        measure_ref=measure_ref,
    )


_POSITION_SPECS = (
    _position_spec("principal", "open_principal"),
    _position_spec("interest", "open_interest"),
    _position_spec("total", "open_total"),
)
_ACTIVITY_SPECS = (
    _activity_spec("new_claim", "new_principal"),
    _activity_spec("interest_accrual", "interest_accrued"),
    _activity_spec("repayment", "repayments"),
    _activity_spec("adjustment", "adjustments"),
    _activity_spec("net_change", "net_change"),
)

DEBT_POSITION_SPECS_V1: Final[Mapping[str, DebtPositionSpec]] = MappingProxyType(
    {spec.spec_id: spec for spec in _POSITION_SPECS}
)
DEBT_ACTIVITY_SPECS_V1: Final[Mapping[str, DebtActivitySpec]] = MappingProxyType(
    {spec.spec_id: spec for spec in _ACTIVITY_SPECS}
)

if len(DEBT_POSITION_SPECS_V1) != len(_POSITION_SPECS):
    raise ValueError("Duplicate DebtPositionSpec.spec_id in debt_position_specs_v1")
if len(DEBT_ACTIVITY_SPECS_V1) != len(_ACTIVITY_SPECS):
    raise ValueError("Duplicate DebtActivitySpec.spec_id in debt_activity_specs_v1")
if set(DEBT_POSITION_SPECS_V1).intersection(DEBT_ACTIVITY_SPECS_V1):
    raise ValueError("Debt position and activity spec IDs must be disjoint")
if any(
    callable(getattr(spec, field.name))
    for spec in (*_POSITION_SPECS, *_ACTIVITY_SPECS)
    for field in fields(spec)
):
    raise TypeError("Debt position/activity contract fields must remain declarative")


def resolve_debt_position_spec(spec_id: str) -> DebtPositionSpec | None:
    """Return a v1 debt-position spec by stable ID."""

    return DEBT_POSITION_SPECS_V1.get(str(spec_id).strip())


def resolve_debt_activity_spec(spec_id: str) -> DebtActivitySpec | None:
    """Return a v1 debt-activity spec by stable ID."""

    return DEBT_ACTIVITY_SPECS_V1.get(str(spec_id).strip())
