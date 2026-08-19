from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import pytest

from accounting.contracts.atomic_flow_drilldowns import (
    ATOMIC_FLOW_DRILLDOWN_SPECS_V1,
    ATOMIC_FLOW_DRILLDOWN_SPECS_VERSION,
    DEFAULT_FLOW_CELL_TOLERANCE,
    FlowCellSpec,
    resolve_flow_cell_spec,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure


def test_v1_specs_are_typed_immutable_and_atomic_flow_only() -> None:
    assert ATOMIC_FLOW_DRILLDOWN_SPECS_VERSION == "atomic_flow_drilldown_specs_v1"
    assert isinstance(ATOMIC_FLOW_DRILLDOWN_SPECS_V1, MappingProxyType)
    assert len(ATOMIC_FLOW_DRILLDOWN_SPECS_V1) == 21

    for cell_id, spec in ATOMIC_FLOW_DRILLDOWN_SPECS_V1.items():
        assert isinstance(spec, FlowCellSpec)
        assert spec.cell_id == cell_id
        assert spec.source_contract == "monthly_flow_semantic_split"
        assert spec.grain[:2] == ("period", "Currency")
        assert spec.tolerance == DEFAULT_FLOW_CELL_TOLERANCE
        assert not any(token in cell_id for token in ("cash_close", "snapshot", "formula", "ratio"))
        assert not any(callable(getattr(spec, field.name)) for field in fields(spec))

    with pytest.raises(TypeError):
        ATOMIC_FLOW_DRILLDOWN_SPECS_V1["flow.future"] = next(  # type: ignore[index]
            iter(ATOMIC_FLOW_DRILLDOWN_SPECS_V1.values())
        )
    with pytest.raises(FrozenInstanceError):
        next(iter(ATOMIC_FLOW_DRILLDOWN_SPECS_V1.values())).cell_id = "flow.changed"  # type: ignore[misc]


def test_every_measure_ref_delegates_to_semantic_measure_registry_v1() -> None:
    for spec in ATOMIC_FLOW_DRILLDOWN_SPECS_V1.values():
        assert spec.measure_ref == (spec.semantic_bucket, spec.semantic_subbucket)
        assert resolve_semantic_measure(*spec.measure_ref) is not None

    physical_measure_fields = {"measure", "amount_in", "amount_out", "amount_abs", "net_amount"}
    assert physical_measure_fields.isdisjoint(field.name for field in fields(FlowCellSpec))


def test_registry_contains_expected_membership_and_grain_contracts() -> None:
    taxes = resolve_flow_cell_spec("flow.property_opex.taxes")
    assert taxes is not None
    assert taxes.semantic_bucket == "property_opex"
    assert taxes.semantic_subbucket == "taxes"
    assert taxes.grain == ("period", "Currency")
    assert taxes.measure_ref == ("property_opex", "taxes")

    rent = resolve_flow_cell_spec("flow.rent.by_property")
    assert rent is not None and rent.grain == ("period", "Currency", "Lugar")

    funding = resolve_flow_cell_spec("flow.funding_contribution.by_channel")
    assert funding is not None and funding.grain == (
        "period",
        "Currency",
        "funding_channel",
    )

    assert resolve_flow_cell_spec("flow.future.unsupported") is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_id": "", "semantic_bucket": "property_opex", "semantic_subbucket": "taxes", "grain": ("period", "Currency"), "measure_ref": ("property_opex", "taxes")},
        {"cell_id": "flow.bad_source", "source_contract": "ledger", "semantic_bucket": "property_opex", "semantic_subbucket": "taxes", "grain": ("period", "Currency"), "measure_ref": ("property_opex", "taxes")},
        {"cell_id": "flow.bad_grain", "semantic_bucket": "property_opex", "semantic_subbucket": "taxes", "grain": ("Currency", "period"), "measure_ref": ("property_opex", "taxes")},
        {"cell_id": "flow.bad_ref", "semantic_bucket": "property_opex", "semantic_subbucket": "taxes", "grain": ("period", "Currency"), "measure_ref": ("operating_revenue", "rent")},
        {"cell_id": "flow.unknown_ref", "semantic_bucket": "future", "semantic_subbucket": "unknown", "grain": ("period", "Currency"), "measure_ref": ("future", "unknown")},
        {"cell_id": "flow.bad_tolerance", "semantic_bucket": "property_opex", "semantic_subbucket": "taxes", "grain": ("period", "Currency"), "measure_ref": ("property_opex", "taxes"), "tolerance": -1.0},
    ],
)
def test_invalid_or_ungoverned_specs_fail_closed(kwargs: dict[str, object]) -> None:
    defaults: dict[str, object] = {"source_contract": "monthly_flow_semantic_split"}
    with pytest.raises(ValueError):
        FlowCellSpec(**(defaults | kwargs))  # type: ignore[arg-type]


def test_no_consumer_is_wired_to_flow_specs_yet() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    consumers = [
        root / "accounting" / "professional" / "drilldown.py",
        root / "accounting" / "metrics" / "annual.py",
    ]
    for consumer in consumers:
        assert "contracts.atomic_flow_drilldowns" not in consumer.read_text(encoding="utf-8")
