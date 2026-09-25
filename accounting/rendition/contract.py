from __future__ import annotations

"""Contract constants and guardrails for private formal account renditions.

The rendition layer is a reporting projection over governed accounting facts. It
must never become a source of title, ownership shares, creditor status, FX
valuation, or any other legal/accounting authority that belongs upstream or
outside the accounting backend.
"""

from collections.abc import Mapping, Sequence
from typing import Any


RENDITION_SPEC_SCHEMA = "accounting.rendition_spec.v1"
RENDITION_CONTRACT_SCHEMA = "accounting.rendition.v1"
SUPPORTED_PURPOSES = frozenset({"private_account_rendition"})
SUPPORTED_LANGUAGES = frozenset({"es-AR"})

# Phase 1 intentionally keeps currency authority source-driven. A rendition
# specification selects a governed run/scope; it never creates a reporting
# currency, conversion, or synthetic ARS+USD total.
FORBIDDEN_CURRENCY_KEYS = frozenset(
    {
        "currency",
        "currencies",
        "base_currency",
        "reporting_currency",
        "synthetic_currency",
        "valuation_currency",
        "fx",
        "fx_rate",
        "exchange_rate",
    }
)

# Title/governance semantics do not belong in the accounting rendition spec.
# Keep this list narrow and explicit: recipients and property identifiers are
# routing metadata, not legal conclusions.
FORBIDDEN_LEGAL_KEYS = frozenset(
    {
        "ownership",
        "ownership_percentage",
        "ownership_share",
        "owner_share",
        "title_share",
        "legal_share",
        "usufruct_share",
        "usufruct_percentage",
        "beneficial_share",
        "percentage",
    }
)


def _normalize_key(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def assert_no_forbidden_semantics(payload: Any, *, path: str = "spec") -> None:
    """Reject legal-title or synthetic-currency authority in rendition inputs.

    The scan is recursive so a later caller cannot hide prohibited semantics in
    a nested block. Values are not interpreted; only field names are governed.
    """

    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = _normalize_key(raw_key)
            here = f"{path}.{raw_key}"
            if key in FORBIDDEN_CURRENCY_KEYS:
                raise ValueError(
                    f"{here} is not allowed in {RENDITION_SPEC_SCHEMA}: "
                    "native currency remains source-driven and ARS/USD must not "
                    "be synthesized or converted by the rendition spec"
                )
            if key in FORBIDDEN_LEGAL_KEYS:
                raise ValueError(
                    f"{here} is not allowed in {RENDITION_SPEC_SCHEMA}: "
                    "property/title percentages and legal entitlement semantics "
                    "must remain outside the accounting rendition contract"
                )
            assert_no_forbidden_semantics(value, path=here)
        return

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            assert_no_forbidden_semantics(value, path=f"{path}[{index}]")
