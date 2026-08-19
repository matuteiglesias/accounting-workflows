"""Typed, declarative contracts for validated cash position and inferred box control.

Wave 4 PR15A defines reporting authority only. It does not migrate any
professional consumer and does not change current accounting outputs.

The contract makes two epistemically distinct positions impossible to confuse:
- validated cash position is eligible for the cash-close headline;
- inferred box motor is reconciliation/control evidence and is never a cash
  headline fallback.

Internal party balances remain outside both contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final, Literal, Mapping


CashSourceContract = Literal["monthly_cash_close"]
ValidatedCashGrainDimension = Literal["period", "Currency", "Box", "account_id"]
InferredControlGrainDimension = Literal["period", "Currency", "Box"]
ValidatedCashStatus = Literal["validated", "approved", "reconciled"]
ValidatedCashSourceType = Literal[
    "bank_statement",
    "manual_cash_count",
    "account_snapshot",
    "reconciled_opening_plus_movements",
]

VALIDATED_CASH_POSITION_SPECS_VERSION: Final = "validated_cash_position_specs_v1"
INFERRED_BOX_CONTROL_SPECS_VERSION: Final = "inferred_box_control_specs_v1"

APPROVED_VALIDATED_CASH_STATUSES: Final[tuple[ValidatedCashStatus, ...]] = (
    "validated",
    "approved",
    "reconciled",
)
APPROVED_VALIDATED_CASH_SOURCE_TYPES: Final[tuple[ValidatedCashSourceType, ...]] = (
    "bank_statement",
    "manual_cash_count",
    "account_snapshot",
    "reconciled_opening_plus_movements",
)

_SOURCE: Final[CashSourceContract] = "monthly_cash_close"
_VALIDATED_GRAIN: Final[tuple[ValidatedCashGrainDimension, ...]] = (
    "period",
    "Currency",
    "Box",
    "account_id",
)
_INFERRED_GRAIN: Final[tuple[InferredControlGrainDimension, ...]] = (
    "period",
    "Currency",
    "Box",
)


def _validate_spec_id(spec_id: str, prefix: str) -> None:
    if not isinstance(spec_id, str):
        raise TypeError("spec_id must be a string")
    if not spec_id or spec_id != spec_id.strip():
        raise ValueError("spec_id must be non-empty and normalized")
    if not spec_id.startswith(prefix):
        raise ValueError(f"spec_id must begin with {prefix!r}")


@dataclass(frozen=True, slots=True)
class ValidatedCashPositionSpec:
    """Governed account-level cash-close authority for professional headlines.

    Consumers must select one latest valid snapshot per candidate account and
    then sum the selected account closes within period/Currency/Box. A candidate
    account with no valid as-of date makes the whole position unavailable;
    silently dropping that account would understate cash while pretending the
    headline is complete.
    """

    spec_id: str
    source_contract: CashSourceContract
    grain: tuple[ValidatedCashGrainDimension, ...]
    value_ref: Literal["close_amount"] = "close_amount"
    position_type: Literal["cash_close"] = "cash_close"
    cash_suitability: Literal["frontend_safe"] = "frontend_safe"
    frontend_safe_required: Literal[True] = True
    allowed_validation_statuses: tuple[ValidatedCashStatus, ...] = (
        "validated",
        "approved",
        "reconciled",
    )
    allowed_source_types: tuple[ValidatedCashSourceType, ...] = (
        "bank_statement",
        "manual_cash_count",
        "account_snapshot",
        "reconciled_opening_plus_movements",
    )
    validated_by_policy: Literal["required_nonblank"] = "required_nonblank"
    account_id_policy: Literal["required_nonblank"] = "required_nonblank"
    as_of_field: Literal["as_of_date"] = "as_of_date"
    selection: Literal[
        "latest_valid_as_of_date_per_account"
    ] = "latest_valid_as_of_date_per_account"
    aggregation: Literal["sum_selected_accounts"] = "sum_selected_accounts"
    duplicate_account_as_of_policy: Literal["unavailable"] = "unavailable"
    incomplete_account_snapshot_policy: Literal[
        "unavailable_if_any_candidate_account_has_no_valid_as_of"
    ] = "unavailable_if_any_candidate_account_has_no_valid_as_of"
    missing_policy: Literal["unavailable"] = "unavailable"
    annualization: Literal[
        "latest_governed_period_then_same_account_snapshot_selection"
    ] = "latest_governed_period_then_same_account_snapshot_selection"
    fallback_to_inferred: Literal["never"] = "never"
    headline_eligible: Literal[True] = True

    def __post_init__(self) -> None:
        _validate_spec_id(self.spec_id, "cash.position.")
        if self.source_contract != _SOURCE:
            raise ValueError(
                f"Unsupported validated-cash source contract: {self.source_contract!r}"
            )
        if self.grain != _VALIDATED_GRAIN:
            raise ValueError(
                "ValidatedCashPositionSpec.grain must be exactly "
                "(period, Currency, Box, account_id)"
            )
        if self.value_ref != "close_amount":
            raise ValueError("Validated cash must use close_amount")
        if self.position_type != "cash_close":
            raise ValueError("Validated cash position_type must be 'cash_close'")
        if self.cash_suitability != "frontend_safe" or self.frontend_safe_required is not True:
            raise ValueError("Validated cash must require frontend-safe rows")
        if tuple(self.allowed_validation_statuses) != APPROVED_VALIDATED_CASH_STATUSES:
            raise ValueError("Validated cash statuses must match the approved v1 status set")
        if tuple(self.allowed_source_types) != APPROVED_VALIDATED_CASH_SOURCE_TYPES:
            raise ValueError("Validated cash source types must match the approved v1 source set")
        if self.validated_by_policy != "required_nonblank":
            raise ValueError("validated_by must be required and nonblank")
        if self.account_id_policy != "required_nonblank":
            raise ValueError("account_id must be required and nonblank")
        if self.as_of_field != "as_of_date":
            raise ValueError("Validated cash as-of field must be as_of_date")
        if self.selection != "latest_valid_as_of_date_per_account":
            raise ValueError("Validated cash must select latest valid as_of_date per account")
        if self.aggregation != "sum_selected_accounts":
            raise ValueError("Validated cash aggregation must sum selected account snapshots")
        if self.duplicate_account_as_of_policy != "unavailable":
            raise ValueError("Duplicate account/as-of snapshots must fail closed as unavailable")
        if (
            self.incomplete_account_snapshot_policy
            != "unavailable_if_any_candidate_account_has_no_valid_as_of"
        ):
            raise ValueError("Incomplete account snapshot sets must fail closed")
        if self.missing_policy != "unavailable":
            raise ValueError("Missing validated cash must be unavailable, not zero")
        if (
            self.annualization
            != "latest_governed_period_then_same_account_snapshot_selection"
        ):
            raise ValueError(
                "Annual validated cash must reuse the monthly account-snapshot primitive"
            )
        if self.fallback_to_inferred != "never":
            raise ValueError("Validated cash headline must never fall back to inferred control")
        if self.headline_eligible is not True:
            raise ValueError("ValidatedCashPositionSpec must be headline eligible")


@dataclass(frozen=True, slots=True)
class InferredBoxControlSpec:
    """Governed inferred box-level control position, explicitly not cash."""

    spec_id: str
    source_contract: CashSourceContract
    grain: tuple[InferredControlGrainDimension, ...]
    value_ref: Literal["close_amount"] = "close_amount"
    position_type: Literal["inferred_box_motor"] = "inferred_box_motor"
    source_type: Literal["inferred_box_motor"] = "inferred_box_motor"
    cash_suitability: Literal["safe_with_caveat"] = "safe_with_caveat"
    frontend_safe_required: Literal[False] = False
    as_of_field: Literal["as_of_date"] = "as_of_date"
    selection: Literal["latest_valid_as_of_date"] = "latest_valid_as_of_date"
    aggregation: Literal["snapshot"] = "snapshot"
    duplicate_snapshot_policy: Literal["unavailable"] = "unavailable"
    missing_policy: Literal["unavailable"] = "unavailable"
    annualization: Literal[
        "latest_governed_period_then_same_snapshot_selection"
    ] = "latest_governed_period_then_same_snapshot_selection"
    headline_eligible: Literal[False] = False
    fallback_role: Literal["never_cash_headline"] = "never_cash_headline"

    def __post_init__(self) -> None:
        _validate_spec_id(self.spec_id, "cash.control.")
        if self.source_contract != _SOURCE:
            raise ValueError(
                f"Unsupported inferred-control source contract: {self.source_contract!r}"
            )
        if self.grain != _INFERRED_GRAIN:
            raise ValueError(
                "InferredBoxControlSpec.grain must be exactly (period, Currency, Box)"
            )
        if self.value_ref != "close_amount":
            raise ValueError("Inferred box control must use close_amount")
        if self.position_type != "inferred_box_motor" or self.source_type != "inferred_box_motor":
            raise ValueError("Inferred control must require inferred_box_motor identity")
        if self.cash_suitability != "safe_with_caveat":
            raise ValueError("Inferred box control must remain safe_with_caveat")
        if self.frontend_safe_required is not False:
            raise ValueError("Inferred box control must not be frontend-safe cash")
        if self.as_of_field != "as_of_date":
            raise ValueError("Inferred box control as-of field must be as_of_date")
        if self.selection != "latest_valid_as_of_date":
            raise ValueError("Inferred box control must select a valid snapshot")
        if self.aggregation != "snapshot":
            raise ValueError("Inferred box control is one stock/control snapshot, not a sum")
        if self.duplicate_snapshot_policy != "unavailable":
            raise ValueError("Duplicate inferred snapshots must fail closed as unavailable")
        if self.missing_policy != "unavailable":
            raise ValueError("Missing inferred control must remain unavailable")
        if self.annualization != "latest_governed_period_then_same_snapshot_selection":
            raise ValueError("Annual inferred control must reuse the same snapshot primitive")
        if self.headline_eligible is not False:
            raise ValueError("Inferred box control can never be cash-headline eligible")
        if self.fallback_role != "never_cash_headline":
            raise ValueError("Inferred control must never act as cash-headline fallback")


_VALIDATED_SPEC = ValidatedCashPositionSpec(
    spec_id="cash.position.validated",
    source_contract=_SOURCE,
    grain=_VALIDATED_GRAIN,
)
_INFERRED_SPEC = InferredBoxControlSpec(
    spec_id="cash.control.inferred_box_motor",
    source_contract=_SOURCE,
    grain=_INFERRED_GRAIN,
)

VALIDATED_CASH_POSITION_SPECS_V1: Final[
    Mapping[str, ValidatedCashPositionSpec]
] = MappingProxyType({_VALIDATED_SPEC.spec_id: _VALIDATED_SPEC})
INFERRED_BOX_CONTROL_SPECS_V1: Final[
    Mapping[str, InferredBoxControlSpec]
] = MappingProxyType({_INFERRED_SPEC.spec_id: _INFERRED_SPEC})

if set(VALIDATED_CASH_POSITION_SPECS_V1).intersection(INFERRED_BOX_CONTROL_SPECS_V1):
    raise ValueError("Validated cash and inferred control spec IDs must be disjoint")
if any(
    callable(getattr(spec, field.name))
    for spec in (_VALIDATED_SPEC, _INFERRED_SPEC)
    for field in fields(spec)
):
    raise TypeError("Cash position/control contract fields must remain declarative")


def resolve_validated_cash_position_spec(
    spec_id: str,
) -> ValidatedCashPositionSpec | None:
    """Return a v1 validated-cash position spec by stable ID."""

    return VALIDATED_CASH_POSITION_SPECS_V1.get(str(spec_id).strip())


def resolve_inferred_box_control_spec(spec_id: str) -> InferredBoxControlSpec | None:
    """Return a v1 inferred box-control spec by stable ID."""

    return INFERRED_BOX_CONTROL_SPECS_V1.get(str(spec_id).strip())
