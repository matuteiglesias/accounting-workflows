from __future__ import annotations

from pathlib import Path

from accounting.contracts.cash_position_control import (
    resolve_inferred_box_control_spec,
    resolve_validated_cash_position_spec,
)
from accounting.contracts.debt_position_activity import (
    resolve_debt_activity_spec,
    resolve_debt_position_spec,
)
from accounting.contracts.atomic_flow_drilldowns import resolve_flow_cell_spec
from accounting.contracts.derived_metrics import resolve_derived_metric_spec


def test_wave4_authorities_exist_and_are_type_separated() -> None:
    assert resolve_flow_cell_spec("flow.property_opex.total") is not None
    assert resolve_debt_position_spec("debt.position.total") is not None
    assert resolve_debt_activity_spec("debt.activity.repayment") is not None

    cash = resolve_validated_cash_position_spec("cash.position.validated")
    control = resolve_inferred_box_control_spec("cash.control.inferred_box_motor")
    assert cash is not None and cash.headline_eligible is True
    assert cash.fallback_to_inferred == "never"
    assert control is not None and control.headline_eligible is False
    assert control.fallback_role == "never_cash_headline"


def test_wave4_runtime_consumers_remain_distinct_after_wave5_diagnostic_migration() -> None:
    root = Path(__file__).resolve().parents[1]
    cash_authority = (root / "accounting" / "cash_authority.py").read_text(encoding="utf-8")
    cash_executor = (root / "accounting" / "professional" / "cash_position_executor.py").read_text(encoding="utf-8")
    position_executor = (root / "accounting" / "professional" / "debt_position_executor.py").read_text(encoding="utf-8")
    activity_executor = (root / "accounting" / "professional" / "debt_activity_executor.py").read_text(encoding="utf-8")
    derived_executor = (root / "accounting" / "professional" / "derived_metric_executor.py").read_text(encoding="utf-8")
    drilldown = (root / "accounting" / "professional" / "drilldown.py").read_text(encoding="utf-8")

    assert "cash.position.validated" in cash_authority
    assert "cash.control.inferred_box_motor" in cash_authority
    assert "select_validated_cash_period" in cash_executor
    assert "select_validated_cash_year" in cash_executor
    assert "monthly_tables_cash_close_matrix" in drilldown
    assert "annual_cash_close_by_box_wide" in drilldown
    assert "monthly_tables_diagnostic_box_level_matrix" in drilldown

    diagnostic = resolve_derived_metric_spec("derived.diagnostic_box_level")
    assert diagnostic is not None
    assert diagnostic.component_refs == ("cash.control.inferred_box_motor",)
    assert "select_inferred_box_control_period" in derived_executor
    assert "validated_cash_fallback" in derived_executor

    assert "DebtPositionSpec" in position_executor
    assert "DebtActivitySpec" not in position_executor
    assert "DebtActivitySpec" in activity_executor
    assert "DebtPositionSpec" not in activity_executor
    assert "as_of_date" not in activity_executor
    assert "select_inferred_box_control_period" not in cash_executor


def test_wave4_cash_consumers_delegate_to_shared_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    frontier = (root / "accounting" / "metrics" / "frontier.py").read_text(encoding="utf-8")
    annual = (root / "accounting" / "metrics" / "annual.py").read_text(encoding="utf-8")
    companion = (root / "accounting" / "professional" / "annual_dashboard_tables.py").read_text(encoding="utf-8")

    assert "select_validated_cash_period" in frontier
    assert "select_validated_cash_year" in annual
    assert "select_validated_cash_year" in companion
    assert "fallback_to_inferred=never" in frontier
    assert "fallback_to_inferred=never" in annual
    assert "inferred/internal excluded" in companion
