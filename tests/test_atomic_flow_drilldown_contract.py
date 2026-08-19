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
    assert len(ATOMIC_FLOW_DRILLDOWN_SPECS_V1) == 22

    for cell_id, spec in ATOMIC_FLOW_DRILLDOWN_SPECS_V1.items():
        assert isinstance(spec, FlowCellSpec)
        assert spec.cell_id == cell_id
        assert spec.source_contract == "monthly_flow_semantic_split"
        assert spec.grain[:2] == ("period", "Currency")
        assert spec.tolerance == DEFAULT_FLOW_CELL_TOLERANCE
        assert spec.semantic_members
        assert spec.measure_ref in spec.semantic_members
        assert not any(
            token in cell_id for token in ("cash_close", "snapshot", "formula", "ratio")
        )
        assert not any(callable(getattr(spec, field.name)) for field in fields(spec))

    assert not any(
        "direct_obligation" in cell_id for cell_id in ATOMIC_FLOW_DRILLDOWN_SPECS_V1
    )
    assert not any(
        "debt_linked" in cell_id for cell_id in ATOMIC_FLOW_DRILLDOWN_SPECS_V1
    )

    with pytest.raises(TypeError):
        ATOMIC_FLOW_DRILLDOWN_SPECS_V1["flow.future"] = next(  # type: ignore[index]
            iter(ATOMIC_FLOW_DRILLDOWN_SPECS_V1.values())
        )
    with pytest.raises(FrozenInstanceError):
        next(iter(ATOMIC_FLOW_DRILLDOWN_SPECS_V1.values())).cell_id = "flow.changed"  # type: ignore[misc]


def test_every_semantic_member_delegates_to_one_governed_measure() -> None:
    for spec in ATOMIC_FLOW_DRILLDOWN_SPECS_V1.values():
        reference_measure = resolve_semantic_measure(*spec.measure_ref)
        assert reference_measure is not None
        assert {
            resolve_semantic_measure(*member) for member in spec.semantic_members
        } == {reference_measure}

    physical_measure_fields = {
        "measure",
        "amount_in",
        "amount_out",
        "amount_abs",
        "net_amount",
    }
    assert physical_measure_fields.isdisjoint(
        field.name for field in fields(FlowCellSpec)
    )


def test_registry_contains_expected_membership_and_grain_contracts() -> None:
    taxes = resolve_flow_cell_spec("flow.property_opex.taxes")
    assert taxes is not None
    assert taxes.semantic_members == (("property_opex", "taxes"),)
    assert taxes.grain == ("period", "Currency")
    assert taxes.measure_ref == ("property_opex", "taxes")

    rent = resolve_flow_cell_spec("flow.rent.by_property")
    assert rent is not None and rent.grain == ("period", "Currency", "Lugar")
    assert rent.semantic_members == (("operating_revenue", "rent"),)

    funding = resolve_flow_cell_spec("flow.funding_contribution.by_channel")
    assert funding is not None and funding.grain == (
        "period",
        "Currency",
        "funding_channel",
    )
    assert funding.semantic_members == (("funding_contribution", ""),)

    assert resolve_flow_cell_spec("flow.future.unsupported") is None


def test_statement_draws_union_is_governed_without_a_callable() -> None:
    draws = resolve_flow_cell_spec("flow.family_draws_or_distributions.total")
    assert draws is not None
    assert draws.semantic_members == (
        ("family_withdrawal_candidate", ""),
        ("family_withdrawal", ""),
    )
    assert draws.measure_ref == ("family_withdrawal_candidate", "")
    assert {
        resolve_semantic_measure(*member) for member in draws.semantic_members
    } == {"amount_out"}


def _valid_spec_kwargs() -> dict[str, object]:
    return {
        "cell_id": "flow.test",
        "source_contract": "monthly_flow_semantic_split",
        "semantic_members": (("property_opex", "taxes"),),
        "grain": ("period", "Currency"),
        "measure_ref": ("property_opex", "taxes"),
    }


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"cell_id": ""}, "cell_id"),
        ({"source_contract": "ledger"}, "Unsupported atomic-flow source contract"),
        ({"grain": ("Currency", "period")}, "grain must begin"),
        ({"semantic_members": ()}, "at least one"),
        (
            {
                "semantic_members": (
                    ("property_opex", "taxes"),
                    ("property_opex", "taxes"),
                )
            },
            "duplicates",
        ),
        (
            {
                "semantic_members": (
                    ("property_opex", "taxes"),
                    ("future", "unknown"),
                )
            },
            "ungoverned",
        ),
        (
            {
                "semantic_members": (("future", "unknown"),),
                "measure_ref": ("future", "unknown"),
            },
            "not governed",
        ),
        (
            {
                "semantic_members": (
                    ("operating_revenue", "rent"),
                    ("property_opex", "taxes"),
                ),
                "measure_ref": ("operating_revenue", "rent"),
            },
            "same governed measure",
        ),
        ({"measure_ref": ("property_opex", "services")}, "one of semantic_members"),
        ({"tolerance": -1.0}, "tolerance"),
    ],
)
def test_invalid_or_ungoverned_specs_fail_closed(
    overrides: dict[str, object], match: str
) -> None:
    kwargs = _valid_spec_kwargs() | overrides
    with pytest.raises(ValueError, match=match):
        FlowCellSpec(**kwargs)  # type: ignore[arg-type]


def test_semantic_members_must_be_declarative_normalized_tuples() -> None:
    kwargs = _valid_spec_kwargs()
    with pytest.raises(TypeError, match="semantic_members must be a tuple"):
        FlowCellSpec(  # type: ignore[arg-type]
            **(kwargs | {"semantic_members": [("property_opex", "taxes")]})
        )
    with pytest.raises(ValueError, match="bucket must be non-empty and normalized"):
        FlowCellSpec(  # type: ignore[arg-type]
            **(kwargs | {"semantic_members": ((" property_opex", "taxes"),)})
        )
    with pytest.raises(ValueError, match="subbucket must be normalized"):
        FlowCellSpec(  # type: ignore[arg-type]
            **(kwargs | {"semantic_members": (("property_opex", " taxes"),)})
        )


def test_professional_is_the_only_flow_spec_consumer_after_pr10c() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    professional = root / "accounting" / "professional" / "drilldown.py"
    annual = root / "accounting" / "metrics" / "annual.py"

    assert "contracts.atomic_flow_drilldowns" in professional.read_text(
        encoding="utf-8"
    )
    assert "resolve_flow_cell_spec" in professional.read_text(encoding="utf-8")
    # Annual metrics consume governed semantic measures, not professional
    # FlowCellSpec membership. That boundary remains intentional.
    assert "contracts.atomic_flow_drilldowns" not in annual.read_text(
        encoding="utf-8"
    )
