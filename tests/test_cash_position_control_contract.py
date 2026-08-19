from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType

import pytest

import accounting.contracts.cash_position_control as contract
from accounting.contracts.cash_position_control import (
    APPROVED_VALIDATED_CASH_SOURCE_TYPES,
    APPROVED_VALIDATED_CASH_STATUSES,
    INFERRED_BOX_CONTROL_SPECS_V1,
    INFERRED_BOX_CONTROL_SPECS_VERSION,
    VALIDATED_CASH_POSITION_SPECS_V1,
    VALIDATED_CASH_POSITION_SPECS_VERSION,
    InferredBoxControlSpec,
    ValidatedCashPositionSpec,
    resolve_inferred_box_control_spec,
    resolve_validated_cash_position_spec,
)
from accounting.marts.cash import (
    EXPLICIT_VALIDATION_STATUSES,
    VALIDATED_CASH_SOURCE_TYPES,
)


def _validated(**overrides) -> ValidatedCashPositionSpec:
    values = {
        "spec_id": "cash.position.validated",
        "source_contract": "monthly_cash_close",
        "grain": ("period", "Currency", "Box", "account_id"),
    }
    values.update(overrides)
    return ValidatedCashPositionSpec(**values)


def _inferred(**overrides) -> InferredBoxControlSpec:
    values = {
        "spec_id": "cash.control.inferred_box_motor",
        "source_contract": "monthly_cash_close",
        "grain": ("period", "Currency", "Box"),
    }
    values.update(overrides)
    return InferredBoxControlSpec(**values)


def test_validated_cash_registry_is_versioned_immutable_and_headline_only() -> None:
    assert VALIDATED_CASH_POSITION_SPECS_VERSION == "validated_cash_position_specs_v1"
    assert isinstance(VALIDATED_CASH_POSITION_SPECS_V1, MappingProxyType)
    assert set(VALIDATED_CASH_POSITION_SPECS_V1) == {"cash.position.validated"}

    spec = resolve_validated_cash_position_spec("cash.position.validated")
    assert spec is not None
    assert spec.source_contract == "monthly_cash_close"
    assert spec.grain == ("period", "Currency", "Box", "account_id")
    assert spec.value_ref == "close_amount"
    assert spec.position_type == "cash_close"
    assert spec.cash_suitability == "frontend_safe"
    assert spec.frontend_safe_required is True
    assert spec.headline_eligible is True
    assert spec.fallback_to_inferred == "never"


def test_validated_cash_requires_account_level_snapshot_completeness() -> None:
    spec = resolve_validated_cash_position_spec("cash.position.validated")
    assert spec is not None
    assert spec.account_id_policy == "required_nonblank"
    assert spec.validated_by_policy == "required_nonblank"
    assert spec.as_of_field == "as_of_date"
    assert spec.selection == "latest_valid_as_of_date_per_account"
    assert spec.aggregation == "sum_selected_accounts"
    assert spec.duplicate_account_as_of_policy == "unavailable"
    assert (
        spec.incomplete_account_snapshot_policy
        == "unavailable_if_any_candidate_account_has_no_valid_as_of"
    )
    assert spec.missing_policy == "unavailable"
    assert (
        spec.annualization
        == "latest_governed_period_then_same_account_snapshot_selection"
    )


def test_validated_cash_contract_matches_current_mart_validation_vocabulary() -> None:
    # PR15A does not change the mart. This parity guard prevents the new
    # contract from quietly approving a broader/different source vocabulary.
    assert set(APPROVED_VALIDATED_CASH_STATUSES) == set(EXPLICIT_VALIDATION_STATUSES)
    assert set(APPROVED_VALIDATED_CASH_SOURCE_TYPES) == set(VALIDATED_CASH_SOURCE_TYPES)


def test_inferred_control_registry_is_separate_and_never_headline_cash() -> None:
    assert INFERRED_BOX_CONTROL_SPECS_VERSION == "inferred_box_control_specs_v1"
    assert isinstance(INFERRED_BOX_CONTROL_SPECS_V1, MappingProxyType)
    assert set(INFERRED_BOX_CONTROL_SPECS_V1) == {"cash.control.inferred_box_motor"}

    spec = resolve_inferred_box_control_spec("cash.control.inferred_box_motor")
    assert spec is not None
    assert spec.source_contract == "monthly_cash_close"
    assert spec.grain == ("period", "Currency", "Box")
    assert spec.value_ref == "close_amount"
    assert spec.position_type == "inferred_box_motor"
    assert spec.source_type == "inferred_box_motor"
    assert spec.cash_suitability == "safe_with_caveat"
    assert spec.frontend_safe_required is False
    assert spec.aggregation == "snapshot"
    assert spec.selection == "latest_valid_as_of_date"
    assert spec.duplicate_snapshot_policy == "unavailable"
    assert spec.missing_policy == "unavailable"
    assert spec.headline_eligible is False
    assert spec.fallback_role == "never_cash_headline"


def test_validated_and_inferred_contracts_are_epistemically_disjoint() -> None:
    validated = resolve_validated_cash_position_spec("cash.position.validated")
    inferred = resolve_inferred_box_control_spec("cash.control.inferred_box_motor")
    assert validated is not None and inferred is not None

    assert validated.position_type != inferred.position_type
    assert validated.cash_suitability != inferred.cash_suitability
    assert validated.frontend_safe_required is True
    assert inferred.frontend_safe_required is False
    assert validated.headline_eligible is True
    assert inferred.headline_eligible is False
    assert validated.fallback_to_inferred == "never"
    assert inferred.fallback_role == "never_cash_headline"
    assert set(VALIDATED_CASH_POSITION_SPECS_V1).isdisjoint(
        INFERRED_BOX_CONTROL_SPECS_V1
    )


def test_internal_balance_is_deliberately_not_a_cash_position_contract() -> None:
    assert not hasattr(contract, "InternalBalanceSpec")
    assert not hasattr(contract, "INTERNAL_BALANCE_SPECS_V1")
    assert not hasattr(contract, "CashPositionSpec")

    positions = {spec.position_type for spec in VALIDATED_CASH_POSITION_SPECS_V1.values()}
    controls = {spec.position_type for spec in INFERRED_BOX_CONTROL_SPECS_V1.values()}
    assert "internal_balance" not in positions | controls


def test_cash_contracts_are_frozen_and_declarative() -> None:
    validated = resolve_validated_cash_position_spec("cash.position.validated")
    inferred = resolve_inferred_box_control_spec("cash.control.inferred_box_motor")
    assert validated is not None and inferred is not None

    with pytest.raises(FrozenInstanceError):
        validated.position_type = "inferred_box_motor"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        inferred.headline_eligible = True  # type: ignore[misc]

    for spec in (validated, inferred):
        assert not any(callable(getattr(spec, field.name)) for field in fields(spec))


def test_validated_cash_contract_fails_closed_on_unsafe_configuration() -> None:
    with pytest.raises(ValueError, match="source contract"):
        _validated(source_contract="validated_cash_close")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grain"):
        _validated(grain=("period", "Currency", "Box"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frontend-safe"):
        _validated(frontend_safe_required=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="account_id"):
        _validated(account_id_policy="optional")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="as_of_date per account"):
        _validated(selection="latest_row")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Duplicate account/as-of"):
        _validated(duplicate_account_as_of_policy="sum")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Incomplete account snapshot"):
        _validated(incomplete_account_snapshot_policy="drop_invalid_accounts")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Missing validated cash"):
        _validated(missing_policy="zero")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="never fall back"):
        _validated(fallback_to_inferred="if_missing")  # type: ignore[arg-type]


def test_inferred_control_contract_fails_closed_on_cash_like_configuration() -> None:
    with pytest.raises(ValueError, match="grain"):
        _inferred(grain=("period", "Currency", "Box", "account_id"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inferred_box_motor identity"):
        _inferred(position_type="cash_close")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not be frontend-safe"):
        _inferred(frontend_safe_required=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one stock/control snapshot"):
        _inferred(aggregation="sum")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Duplicate inferred snapshots"):
        _inferred(duplicate_snapshot_policy="sum")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="never be cash-headline"):
        _inferred(headline_eligible=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="never act as cash-headline fallback"):
        _inferred(fallback_role="fallback_if_validated_missing")  # type: ignore[arg-type]


def test_resolvers_fail_closed_on_unknown_or_cross_domain_ids() -> None:
    assert resolve_validated_cash_position_spec("cash.position.unknown") is None
    assert resolve_inferred_box_control_spec("cash.control.unknown") is None
    assert resolve_validated_cash_position_spec("cash.control.inferred_box_motor") is None
    assert resolve_inferred_box_control_spec("cash.position.validated") is None


def test_pr15a_adds_contracts_without_production_consumers() -> None:
    needle = "accounting.contracts.cash_position_control"
    for consumer in [
        "accounting/marts/cash.py",
        "accounting/professional/drilldown.py",
        "accounting/professional/drilldown_legacy.py",
        "accounting/professional/annual_dashboard_tables.py",
        "accounting/metrics/annual.py",
    ]:
        assert needle not in Path(consumer).read_text(encoding="utf-8"), consumer
