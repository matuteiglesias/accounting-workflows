"""Typed, declarative contracts for atomic-flow drilldown membership.

This module defines contracts only.  No drilldown consumer is wired to this
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


@dataclass(frozen=True, slots=True)
class FlowCellSpec:
    """Declarative membership for one governed atomic-flow cell family.

    ``measure_ref`` is a semantic registry key, never a physical amount column.
    Executors must resolve it through ``semantic_measure_registry_v1`` when they
    are migrated in a later change.
    """

    cell_id: str
    source_contract: FlowSourceContract
    semantic_bucket: str
    semantic_subbucket: str
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
        if not isinstance(self.semantic_bucket, str) or not isinstance(
            self.semantic_subbucket, str
        ):
            raise TypeError("FlowCellSpec semantic membership must use strings")
        if (
            not self.semantic_bucket
            or self.semantic_bucket != self.semantic_bucket.strip()
        ):
            raise ValueError("FlowCellSpec.semantic_bucket must be non-empty and normalized")
        if self.semantic_subbucket != self.semantic_subbucket.strip():
            raise ValueError("FlowCellSpec.semantic_subbucket must be normalized")
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
        if self.measure_ref != (self.semantic_bucket, self.semantic_subbucket):
            raise ValueError("measure_ref must identify the spec's semantic membership")
        if resolve_semantic_measure(*self.measure_ref) is None:
            raise ValueError(
                "measure_ref is not governed by semantic_measure_registry_v1: "
                f"{self.measure_ref!r}"
            )
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("FlowCellSpec.tolerance must be finite and non-negative")


def _spec(
    cell_id: str,
    bucket: str,
    subbucket: str = "",
    *,
    dimensions: tuple[FlowGrainDimension, ...] = (),
) -> FlowCellSpec:
    return FlowCellSpec(
        cell_id=cell_id,
        source_contract=_SOURCE_CONTRACT,
        semantic_bucket=bucket,
        semantic_subbucket=subbucket,
        grain=(*_BASE_GRAIN, *dimensions),
        measure_ref=(bucket, subbucket),
    )


_SPECS = (
    _spec("flow.operating_revenue", "operating_revenue"),
    _spec("flow.rent.total", "operating_revenue", "rent"),
    _spec("flow.rent.by_box", "operating_revenue", "rent", dimensions=("Box",)),
    _spec("flow.rent.by_property", "operating_revenue", "rent", dimensions=("Lugar",)),
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
    _spec("flow.draws.by_box", "family_withdrawal_candidate", dimensions=("Box",)),
    _spec(
        "flow.draws.by_type",
        "family_withdrawal_candidate",
        dimensions=("semantic_subbucket",),
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
if any(callable(getattr(spec, field.name)) for spec in _SPECS for field in fields(spec)):
    raise TypeError("FlowCellSpec fields must remain declarative; callables are not allowed")


def resolve_flow_cell_spec(cell_id: str) -> FlowCellSpec | None:
    """Return a v1 atomic-flow spec by stable cell ID."""

    return ATOMIC_FLOW_DRILLDOWN_SPECS_V1.get(str(cell_id).strip())
