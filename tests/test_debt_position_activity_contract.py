from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType

import pytest

import accounting.contracts.debt_position_activity as contract
from accounting.contracts.debt_position_activity import (
    DEBT_ACTIVITY_SPECS_V1,
    DEBT_ACTIVITY_SPECS_VERSION,
    DEBT_POSITION_SPECS_V1,
    DEBT_POSITION_SPECS_VERSION,
    DebtActivitySpec,
    DebtPositionSpec,
    resolve_debt_activity_spec,
    resolve_debt_position_spec,
)


def _position(**overrides) -> DebtPositionSpec:
    values = {
        "spec_id": "debt.position.principal",
        "source_contract": "monthly_debt_position",
        "grain": ("period", "Currency", "debtor", "creditor", "component"),
        "component": "principal",
        "value_ref": "open_principal",
    }
    values.update(overrides)
    return DebtPositionSpec(**values)


def _activity(**overrides) -> DebtActivitySpec:
    values = {
        "spec_id": "debt.activity.repayment",
        "source_contract": "monthly_debt_activity",
        "grain": ("period", "Currency", "debtor", "creditor", "activity_type"),
        "activity_type": "repayment",
        "measure_ref": "repayments",
    }
    values.update(overrides)
    return DebtActivitySpec(**values)


def test_position_registry_is_versioned_immutable_and_snapshot_only() -> None:
    assert DEBT_POSITION_SPECS_VERSION == "debt_position_specs_v1"
    assert isinstance(DEBT_POSITION_SPECS_V1, MappingProxyType)
    assert set(DEBT_POSITION_SPECS_V1) == {
        "debt.position.principal",
        "debt.position.interest",
        "debt.position.total",
    }
    expected = {
        "principal": "open_principal",
        "interest": "open_interest",
        "total": "open_total",
    }
    for spec in DEBT_POSITION_SPECS_V1.values():
        assert isinstance(spec, DebtPositionSpec)
        assert spec.source_contract == "monthly_debt_position"
        assert spec.grain == ("period", "Currency", "debtor", "creditor", "component")
        assert spec.value_ref == expected[spec.component]
        assert spec.aggregation == "snapshot"
        assert spec.selection == "latest_valid_as_of_date"
        assert spec.as_of_field == "as_of_date"
        assert spec.invalid_as_of_policy == "unavailable"
        assert spec.annualization == "latest_period_then_latest_valid_as_of_date"


def test_activity_registry_is_versioned_immutable_and_sum_flow_only() -> None:
    assert DEBT_ACTIVITY_SPECS_VERSION == "debt_activity_specs_v1"
    assert isinstance(DEBT_ACTIVITY_SPECS_V1, MappingProxyType)
    expected = {
        "new_claim": "new_principal",
        "interest_accrual": "interest_accrued",
        "repayment": "repayments",
        "adjustment": "adjustments",
        "net_change": "net_change",
    }
    assert set(DEBT_ACTIVITY_SPECS_V1) == {
        f"debt.activity.{activity_type}" for activity_type in expected
    }
    for spec in DEBT_ACTIVITY_SPECS_V1.values():
        assert isinstance(spec, DebtActivitySpec)
        assert spec.source_contract == "monthly_debt_activity"
        assert spec.grain == (
            "period",
            "Currency",
            "debtor",
            "creditor",
            "activity_type",
        )
        assert spec.measure_ref == expected[spec.activity_type]
        assert spec.aggregation == "sum_flow"
        assert spec.annualization == "sum_periods"


def test_position_and_activity_contracts_cannot_cross_resolve() -> None:
    assert resolve_debt_position_spec("debt.position.total") is not None
    assert resolve_debt_activity_spec("debt.activity.net_change") is not None
    assert resolve_debt_position_spec("debt.activity.repayment") is None
    assert resolve_debt_activity_spec("debt.position.principal") is None
    assert set(DEBT_POSITION_SPECS_V1).isdisjoint(DEBT_ACTIVITY_SPECS_V1)
    assert {spec.aggregation for spec in DEBT_POSITION_SPECS_V1.values()} == {"snapshot"}
    assert {spec.aggregation for spec in DEBT_ACTIVITY_SPECS_V1.values()} == {"sum_flow"}


def test_contracts_are_frozen_and_declarative() -> None:
    position = resolve_debt_position_spec("debt.position.total")
    activity = resolve_debt_activity_spec("debt.activity.repayment")
    assert position is not None and activity is not None
    with pytest.raises(FrozenInstanceError):
        position.component = "principal"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        activity.activity_type = "new_claim"  # type: ignore[misc]
    for spec in (*DEBT_POSITION_SPECS_V1.values(), *DEBT_ACTIVITY_SPECS_V1.values()):
        assert not any(callable(getattr(spec, field.name)) for field in fields(spec))


def test_position_contract_fails_closed_on_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="source contract"):
        _position(source_contract="monthly_debt_activity")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grain"):
        _position(grain=("period", "Currency", "debtor", "creditor"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value_ref"):
        _position(value_ref="open_total")
    with pytest.raises(ValueError, match="aggregation"):
        _position(aggregation="sum_flow")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="selection"):
        _position(selection="latest_as_of_date")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fail closed"):
        _position(invalid_as_of_policy="lexical_fallback")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="annualization"):
        _position(annualization="sum_periods")  # type: ignore[arg-type]


def test_activity_contract_rejects_stock_or_sparse_mapping_shortcuts() -> None:
    with pytest.raises(ValueError, match="source contract"):
        _activity(source_contract="monthly_debt_position")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grain"):
        _activity(grain=("period", "Currency", "debtor", "creditor"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="measure_ref"):
        _activity(measure_ref="net_change")
    with pytest.raises(ValueError, match="aggregation"):
        _activity(aggregation="snapshot")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="annualization"):
        _activity(annualization="latest_period")  # type: ignore[arg-type]


def test_activity_contract_excludes_opening_and_closing_balance_rows() -> None:
    activity_types = {spec.activity_type for spec in DEBT_ACTIVITY_SPECS_V1.values()}
    assert "opening_balance" not in activity_types
    assert "closing_balance" not in activity_types
    assert "opening_total" not in {spec.measure_ref for spec in DEBT_ACTIVITY_SPECS_V1.values()}
    assert "closing_total" not in {spec.measure_ref for spec in DEBT_ACTIVITY_SPECS_V1.values()}


def test_cash_position_remains_out_of_scope() -> None:
    assert not hasattr(contract, "CashPositionSpec")
    assert not hasattr(contract, "CASH_POSITION_SPECS_V1")


def test_pr12_is_contract_only_no_production_consumer_wiring() -> None:
    consumers = [
        "accounting/professional/drilldown.py",
        "accounting/professional/drilldown_legacy.py",
        "accounting/marts/debt.py",
        "accounting/metrics/annual.py",
    ]
    needle = "accounting.contracts.debt_position_activity"
    for consumer in consumers:
        assert needle not in Path(consumer).read_text(encoding="utf-8"), consumer
