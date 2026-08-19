from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from accounting.contracts.derived_metrics import (
    APPROVED_DERIVED_COMPONENT_REFS,
    DERIVED_METRIC_SPECS_V1,
    DERIVED_METRIC_SPECS_VERSION,
    resolve_derived_metric_spec,
)


EXPECTED_SPEC_IDS = {
    "derived.net_operating",
    "derived.coverage_after_draws",
    "derived.savings_rate",
    "derived.operating_margin",
    "derived.opex_to_rent",
    "derived.draws_to_operating_result",
    "derived.diagnostic_box_level",
}


def test_registry_v1_is_exact_and_resolvable() -> None:
    assert DERIVED_METRIC_SPECS_VERSION == "derived_metric_specs_v1"
    assert set(DERIVED_METRIC_SPECS_V1) == EXPECTED_SPEC_IDS
    for spec_id in EXPECTED_SPEC_IDS:
        assert resolve_derived_metric_spec(spec_id) is DERIVED_METRIC_SPECS_V1[spec_id]
    assert resolve_derived_metric_spec("derived.not_registered") is None


def test_source_authority_specs_reconcile_without_competing_value_authority() -> None:
    net = DERIVED_METRIC_SPECS_V1["derived.net_operating"]
    assert net.authority_mode == "source_value_with_formula_reconciliation"
    assert net.source_value_ref == "metric:IS.NET.OPERATING"
    assert net.operation == "subtract"
    assert net.component_refs == (
        "metric:IS.REVENUE.OPERATING",
        "metric:IS.OPEX.PROPERTY",
    )
    assert net.period_grains == ("M", "Y")
    assert net.annual_policy == "source_authority_annual_value"

    coverage = DERIVED_METRIC_SPECS_V1["derived.coverage_after_draws"]
    assert coverage.authority_mode == "source_value_with_formula_reconciliation"
    assert coverage.source_value_ref == "metric:COV.NET.AFTER_DRAWS"
    assert coverage.operation == "add_subtract"
    assert coverage.component_refs == (
        "metric:IS.NET.OPERATING",
        "metric:FUND.CONTRIB.TOTAL",
        "metric:DIST.DRAWS.PERSONAL",
    )


def test_ratio_specs_fail_closed_and_recompute_from_annual_components() -> None:
    ratio_ids = {
        "derived.savings_rate",
        "derived.operating_margin",
        "derived.opex_to_rent",
        "derived.draws_to_operating_result",
    }
    for spec_id in ratio_ids:
        spec = DERIVED_METRIC_SPECS_V1[spec_id]
        assert spec.authority_mode == "computed_derived"
        assert spec.operation == "ratio"
        assert spec.period_grains == ("Y",)
        assert spec.annual_policy == "recompute_from_annual_components"
        assert spec.missing_component_policy == "unavailable"
        assert spec.zero_denominator_policy == "not_applicable"
        assert spec.source_value_ref is None


def test_opex_to_rent_uses_rent_authority_not_total_operating_revenue() -> None:
    spec = DERIVED_METRIC_SPECS_V1["derived.opex_to_rent"]
    assert spec.component_refs == (
        "metric:IS.OPEX.PROPERTY",
        "metric:IS.RENT.TOTAL",
    )
    assert "metric:IS.REVENUE.OPERATING" not in spec.component_refs


def test_diagnostic_box_level_is_only_inferred_control_period_delta() -> None:
    spec = DERIVED_METRIC_SPECS_V1["derived.diagnostic_box_level"]
    assert spec.authority_mode == "computed_derived"
    assert spec.operation == "period_delta"
    assert spec.component_refs == ("cash.control.inferred_box_motor",)
    assert spec.grain == ("period", "Currency", "Box")
    assert spec.period_grains == ("M",)
    assert spec.annual_policy == "not_applicable"
    assert spec.missing_component_policy == "unavailable"
    assert spec.zero_denominator_policy is None


def test_specialized_fx_debt_and_bridge_authorities_are_not_forced_into_v1() -> None:
    joined = " ".join(
        [
            *DERIVED_METRIC_SPECS_V1,
            *APPROVED_DERIVED_COMPONENT_REFS,
        ]
    )
    assert "TR.FX.NET" not in joined
    assert "ID.DEBT.NET_PM_POSITION" not in joined
    assert "net_flow" not in joined


def test_contract_rejects_missing_zero_and_callable_style_shortcuts() -> None:
    base = DERIVED_METRIC_SPECS_V1["derived.operating_margin"]

    with pytest.raises(ValueError, match="not_applicable"):
        replace(base, zero_denominator_policy=None)

    with pytest.raises(ValueError, match="Unknown governed component refs"):
        replace(
            base,
            component_refs=("metric:IS.NET.OPERATING", "label:rent"),
        )

    with pytest.raises(ValueError, match="unavailable"):
        replace(base, missing_component_policy="zero")  # type: ignore[arg-type]

    with pytest.raises(FrozenInstanceError):
        base.operation = "subtract"  # type: ignore[misc]


def test_period_delta_cannot_be_retargeted_to_validated_cash() -> None:
    diagnostic = DERIVED_METRIC_SPECS_V1["derived.diagnostic_box_level"]
    with pytest.raises(ValueError, match="inferred box control"):
        replace(
            diagnostic,
            component_refs=("metric:IS.REVENUE.OPERATING",),
        )


def test_contract_is_not_consumed_by_production_in_pr17() -> None:
    repo = Path(__file__).resolve().parents[1]
    contract_path = repo / "accounting" / "contracts" / "derived_metrics.py"
    consumers: list[str] = []
    for path in (repo / "accounting").rglob("*.py"):
        if path == contract_path:
            continue
        text = path.read_text(encoding="utf-8")
        if "contracts.derived_metrics" in text or "DerivedMetricSpec" in text:
            consumers.append(str(path.relative_to(repo)))
    assert consumers == [], f"PR17 must remain contract-only; consumers={consumers}"


def test_contract_contains_no_human_label_or_executable_formula_dispatch() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "accounting" / "contracts" / "derived_metrics.py"
    text = path.read_text(encoding="utf-8")
    for label in [
        "Margen operativo",
        "OPEX / renta",
        "Retiros / resultado operativo",
        "Cobertura después de funding y retiros",
    ]:
        assert label not in text

    tree = ast.parse(text)
    assert not any(isinstance(node, ast.Lambda) for node in ast.walk(tree))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "eval"
        for node in ast.walk(tree)
    )
