"""Typed, declarative contracts for atomic-flow drilldown membership.

This module defines contracts only. No drilldown consumer is wired to this
registry yet; migration requires separate characterization and parity evidence.
Stock selection, formulas, quality ratios, compatibility fallbacks, and
unsupported routing deliberately remain outside this contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final, Literal, Mapping

from accounting.contracts.semantic_measures import (
    SemanticMeasureKey,
    resolve_semantic_measure,
)


FlowSourceContract = Literal["monthly_flow_semantic_split"]
FlowGrainDimension = Literal[
    "period",
    "Currency",
    "Box",
    "Lugar",
    "actor",
    "funding_actor",
    "funding_channel",
    "cash_effect",
    "target_box",
    "semantic_subbucket",
]

ATOMIC_FLOW_DRILLDOWN_SPECS_VERSION: Final = "atomic_flow_drilldown_specs_v1"
DEFAULT_FLOW_CELL_TOLERANCE: Final = 1e-6
_SOURCE_CONTRACT: Final[FlowSourceContract] = "monthly_flow_semantic_split"
_BASE_GRAIN: Final[tuple[FlowGrainDimension, ...]] = ("period", "Currency")


def _normalize_semantic_key(
    key: SemanticMeasureKey,
    *,
    field_name: str,
) -> SemanticMeasureKey:
    if (
        not isinstance(key, tuple)
        or len(key) != 2
        or not all(isinstance(value, str) for value in key)
    ):
        raise TypeError(
            f"FlowCellSpec.{field_name} entries must be "
            "(bucket, subbucket) string tuples"
        )
    bucket, subbucket = key
    if not bucket or bucket != bucket.strip():
        raise ValueError(
            f"FlowCellSpec.{field_name} bucket must be non-empty and normalized"
        )
    if subbucket != subbucket.strip():
        raise ValueError(f"FlowCellSpec.{field_name} subbucket must be normalized")
    return bucket, subbucket


@dataclass(frozen=True, slots=True)
class FlowCellSpec:
    """Declarative membership for one governed atomic-flow cell family.

    ``semantic_members`` contains one or more approved semantic pairs. Unions
    are allowed only when every member resolves to the same governed measure.
    ``measure_ref`` identifies that shared measure authority; it never stores a
    physical amount column.
    """

    cell_id: str
    source_contract: FlowSourceContract
    semantic_members: tuple[SemanticMeasureKey, ...]
    grain: tuple[FlowGrainDimension, ...]
    measure_ref: SemanticMeasureKey
    tolerance: float = DEFAULT_FLOW_CELL_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str):
            raise TypeError("FlowCellSpec.cell_id must be a string")
        if not self.cell_id or self.cell_id != self.cell_id.strip():
            raise ValueError("FlowCellSpec.cell_id must be non-empty and normalized")
        if self.source_contract != _SOURCE_CONTRACT:
            raise ValueError(
                f"Unsupported atomic-flow source contract: {self.source_contract!r}"
            )
        if not isinstance(self.semantic_members, tuple):
            raise TypeError("FlowCellSpec.semantic_members must be a tuple")
        if not self.semantic_members:
            raise ValueError(
                "FlowCellSpec.semantic_members must contain at least one semantic pair"
            )

        members = tuple(
            _normalize_semantic_key(member, field_name="semantic_members")
            for member in self.semantic_members
        )
        if len(set(members)) != len(members):
            raise ValueError(
                "FlowCellSpec.semantic_members must not contain duplicates"
            )

        measure_ref = _normalize_semantic_key(
            self.measure_ref,
            field_name="measure_ref",
        )
        if measure_ref not in members:
            raise ValueError(
                "FlowCellSpec.measure_ref must identify one of semantic_members"
            )

        reference_measure = resolve_semantic_measure(*measure_ref)
        if reference_measure is None:
            raise ValueError(
                "measure_ref is not governed by semantic_measure_registry_v1: "
                f"{measure_ref!r}"
            )

        unresolved = [
            member for member in members if resolve_semantic_measure(*member) is None
        ]
        if unresolved:
            raise ValueError(
                "semantic_members contains ungoverned semantic pair(s): "
                f"{unresolved!r}"
            )

        inconsistent = [
            member
            for member in members
            if resolve_semantic_measure(*member) != reference_measure
        ]
        if inconsistent:
            raise ValueError(
                "semantic_members must all resolve to the same governed measure "
                "as measure_ref: "
                f"measure_ref={measure_ref!r}; inconsistent={inconsistent!r}"
            )

        if not isinstance(self.grain, tuple) or not all(
            isinstance(dimension, str) for dimension in self.grain
        ):
            raise TypeError("FlowCellSpec.grain must be a tuple of dimensions")
        if not self.grain or len(set(self.grain)) != len(self.grain):
            raise ValueError(
                "FlowCellSpec.grain must be non-empty and contain no duplicates"
            )
        if self.grain[:2] != _BASE_GRAIN:
            raise ValueError("Atomic-flow grain must begin with period and Currency")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("FlowCellSpec.tolerance must be finite and non-negative")


def _spec(
    cell_id: str,
    bucket: str,
    subbucket: str = "",
    *,
    dimensions: tuple[FlowGrainDimension, ...] = (),
) -> FlowCellSpec:
    member = (bucket, subbucket)
    return FlowCellSpec(
        cell_id=cell_id,
        source_contract=_SOURCE_CONTRACT,
        semantic_members=(member,),
        grain=(*_BASE_GRAIN, *dimensions),
        measure_ref=member,
    )


def _union_spec(
    cell_id: str,
    semantic_members: tuple[SemanticMeasureKey, ...],
    *,
    measure_ref: SemanticMeasureKey,
    dimensions: tuple[FlowGrainDimension, ...] = (),
) -> FlowCellSpec:
    return FlowCellSpec(
        cell_id=cell_id,
        source_contract=_SOURCE_CONTRACT,
        semantic_members=semantic_members,
        grain=(*_BASE_GRAIN, *dimensions),
        measure_ref=measure_ref,
    )


_SPECS = (
    _spec("flow.operating_revenue", "operating_revenue"),
    _spec("flow.rent.total", "operating_revenue", "rent"),
    _spec("flow.rent.by_box", "operating_revenue", "rent", dimensions=("Box",)),
    _spec(
        "flow.rent.by_property",
        "operating_revenue",
        "rent",
        dimensions=("Lugar",),
    ),
    _spec("flow.property_opex.total", "property_opex"),
    _spec("flow.property_opex.taxes", "property_opex", "taxes"),
    _spec("flow.property_opex.services", "property_opex", "services"),
    _spec("flow.property_opex.maintenance", "property_opex", "maintenance"),
    _spec("flow.property_opex.legal", "property_opex", "legal"),
    _spec(
        "flow.property_opex.by_category",
        "property_opex",
        dimensions=("semantic_subbucket",),
    ),
    _spec("flow.funding_contribution.total", "funding_contribution"),
    _spec(
        "flow.funding_contribution.by_actor",
        "funding_contribution",
        dimensions=("funding_actor",),
    ),
    _spec(
        "flow.funding_contribution.by_channel",
        "funding_contribution",
        dimensions=("funding_channel",),
    ),
    _spec(
        "flow.funding_contribution.by_cash_effect",
        "funding_contribution",
        dimensions=("cash_effect",),
    ),
    _spec(
        "flow.funding_contribution.by_target_box",
        "funding_contribution",
        dimensions=("target_box",),
    ),
    _spec("flow.draws.total", "family_withdrawal_candidate"),
    _spec(
        "flow.draws.by_box",
        "family_withdrawal_candidate",
        dimensions=("Box",),
    ),
    _spec(
        "flow.draws.by_type",
        "family_withdrawal_candidate",
        dimensions=("semantic_subbucket",),
    ),
    _union_spec(
        "flow.family_draws_or_distributions.total",
        (
            ("family_withdrawal_candidate", ""),
            ("family_withdrawal", ""),
        ),
        measure_ref=("family_withdrawal_candidate", ""),
    ),
    _spec(
        "flow.fx.conversion_proceeds",
        "treasury_fx",
        "fx_conversion_proceeds",
        dimensions=("Box",),
    ),
    _spec(
        "flow.fx.conversion_outflow",
        "treasury_fx",
        "fx_conversion_outflow",
        dimensions=("Box",),
    ),
    _spec(
        "flow.fx.cost_or_spread",
        "treasury_fx",
        "fx_cost_or_spread",
        dimensions=("Box",),
    ),
)

ATOMIC_FLOW_DRILLDOWN_SPECS_V1: Final[Mapping[str, FlowCellSpec]] = MappingProxyType(
    {spec.cell_id: spec for spec in _SPECS}
)

if len(ATOMIC_FLOW_DRILLDOWN_SPECS_V1) != len(_SPECS):
    raise ValueError("Duplicate FlowCellSpec.cell_id in atomic_flow_drilldown_specs_v1")
if any(
    callable(getattr(spec, field.name))
    for spec in _SPECS
    for field in fields(spec)
):
    raise TypeError("FlowCellSpec fields must remain declarative; callables are not allowed")


def resolve_flow_cell_spec(cell_id: str) -> FlowCellSpec | None:
    """Return a v1 atomic-flow spec by stable cell ID."""

    return ATOMIC_FLOW_DRILLDOWN_SPECS_V1.get(str(cell_id).strip())
