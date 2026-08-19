"""Versioned authority for approved semantic flow measures.

This registry describes semantic meaning only. Consumers are intentionally not
wired to it yet; migration must happen separately with characterization parity.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal, Mapping


SemanticMeasure = Literal["amount_in", "amount_out", "amount_abs"]
SemanticMeasureKey = tuple[str, str]

SEMANTIC_MEASURE_REGISTRY_VERSION: Final = "semantic_measure_registry_v1"

_WILDCARD: Final = "*"
SEMANTIC_MEASURE_REGISTRY_V1: Final[Mapping[SemanticMeasureKey, SemanticMeasure]] = (
    MappingProxyType(
        {
            ("operating_revenue", _WILDCARD): "amount_in",
            ("property_opex", _WILDCARD): "amount_out",
            ("funding_contribution", _WILDCARD): "amount_in",
            ("family_withdrawal_candidate", _WILDCARD): "amount_out",
            ("family_withdrawal", _WILDCARD): "amount_out",
            ("debt_movement", _WILDCARD): "amount_abs",
            ("internal_transfer", _WILDCARD): "amount_abs",
            ("treasury_fx", "fx_conversion_proceeds"): "amount_in",
            ("treasury_fx", "fx_conversion_outflow"): "amount_out",
            ("treasury_fx", "fx_cost_or_spread"): "amount_out",
        }
    )
)


def resolve_semantic_measure(bucket: str, subbucket: str) -> SemanticMeasure | None:
    """Return the approved measure for a semantic pair, or ``None`` if unknown."""

    key = (str(bucket).strip(), str(subbucket).strip())
    return SEMANTIC_MEASURE_REGISTRY_V1.get(
        key,
        SEMANTIC_MEASURE_REGISTRY_V1.get((key[0], _WILDCARD)),
    )
