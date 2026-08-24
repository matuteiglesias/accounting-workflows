"""Typed authority for broader economic-support membership.

Core funding remains the semantic ``funding_contribution`` bucket.  This
contract describes the deliberately broader support surface used by some
annual/professional views without redefining that core accounting bucket.

The registry is intentionally small and evidence-driven:

* core contribution;
* direct obligation payment;
* debt-linked support.

Rows that match more than one support kind fail closed.  Rows with explicit
funding/support metadata that match none of the governed kinds also fail closed
in strict mode.  Ordinary rent/OPEX/debt rows without support metadata are not
members.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

import pandas as pd

from accounting.contracts.semantic_measures import resolve_semantic_measure


FundingSupportKind = Literal[
    "core_contribution",
    "direct_obligation_payment",
    "debt_linked_support",
]

FUNDING_SUPPORT_SPECS_VERSION: Final = "funding_support_specs_v1"
SUPPORT_KIND_COLUMN: Final = "support_kind"
SUPPORT_SPEC_ID_COLUMN: Final = "funding_support_spec_id"
SUPPORT_MEASURE_COLUMN: Final = "support_measure"
SUPPORT_AMOUNT_COLUMN: Final = "support_amount"
SOURCE_MEMBER_IDS_COLUMN: Final = "source_member_ids"


@dataclass(frozen=True, slots=True)
class FundingSupportSpec:
    spec_id: str
    support_kind: FundingSupportKind
    semantic_bucket: str = ""
    cash_effect: str = ""
    requires_debt_effect: bool = False
    requires_funding_channel: bool = False

    def __post_init__(self) -> None:
        if not self.spec_id or self.spec_id != self.spec_id.strip():
            raise ValueError("FundingSupportSpec.spec_id must be non-empty and normalized")
        if self.semantic_bucket != self.semantic_bucket.strip():
            raise ValueError("FundingSupportSpec.semantic_bucket must be normalized")
        if self.cash_effect != self.cash_effect.strip():
            raise ValueError("FundingSupportSpec.cash_effect must be normalized")


_SPECS = (
    FundingSupportSpec(
        spec_id="funding.support.core_contribution",
        support_kind="core_contribution",
        semantic_bucket="funding_contribution",
    ),
    FundingSupportSpec(
        spec_id="funding.support.direct_obligation_payment",
        support_kind="direct_obligation_payment",
        cash_effect="no_cash_in_box_direct_payment",
        requires_funding_channel=True,
    ),
    FundingSupportSpec(
        spec_id="funding.support.debt_linked_support",
        support_kind="debt_linked_support",
        requires_debt_effect=True,
        requires_funding_channel=True,
    ),
)

FUNDING_SUPPORT_SPECS: Final[Mapping[str, FundingSupportSpec]] = MappingProxyType(
    {spec.spec_id: spec for spec in _SPECS}
)


def resolve_funding_support_spec(spec_id: str) -> FundingSupportSpec | None:
    return FUNDING_SUPPORT_SPECS.get(str(spec_id).strip())


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str).str.strip()


def _nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _debt_effect_present(frame: pd.DataFrame) -> pd.Series:
    value = _text(frame, "debt_effect", "none").str.lower()
    return ~value.isin({"", "none", "nan", "n/a", "na"})


def funding_support_spec_mask(frame: pd.DataFrame, spec: FundingSupportSpec) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if spec.semantic_bucket:
        mask &= _text(frame, "semantic_bucket").eq(spec.semantic_bucket)
    if spec.cash_effect:
        mask &= _text(frame, "cash_effect").eq(spec.cash_effect)
    if spec.requires_debt_effect:
        mask &= _debt_effect_present(frame)
    if spec.requires_funding_channel:
        mask &= _nonempty(_text(frame, "funding_channel"))
    return mask


def _explicit_support_metadata(frame: pd.DataFrame) -> pd.Series:
    return (
        _text(frame, "semantic_bucket").eq("funding_contribution")
        | _nonempty(_text(frame, "funding_channel"))
        | _debt_effect_present(frame)
        | _text(frame, "cash_effect").eq("no_cash_in_box_direct_payment")
    )


def _source_member_ids(frame: pd.DataFrame) -> pd.Series:
    for column in ("source_tx_ids_sample", "tx_id", "source_member_id"):
        if column in frame.columns:
            return _text(frame, column)
    return pd.Series("", index=frame.index, dtype="object")


def classify_funding_support(
    frame: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Return governed support members with explicit kind, measure and amount.

    The returned rows retain every source column and add stable support contract
    fields.  Support amount is always read through the canonical semantic-measure
    registry; this contract never invents a second amount convention.
    """

    if frame is None or frame.empty:
        columns = list(frame.columns) if isinstance(frame, pd.DataFrame) else []
        return pd.DataFrame(
            columns=[
                *columns,
                SUPPORT_KIND_COLUMN,
                SUPPORT_SPEC_ID_COLUMN,
                SUPPORT_MEASURE_COLUMN,
                SUPPORT_AMOUNT_COLUMN,
                SOURCE_MEMBER_IDS_COLUMN,
            ]
        )

    masks = {spec.spec_id: funding_support_spec_mask(frame, spec) for spec in _SPECS}
    match_count = sum(mask.astype(int) for mask in masks.values())
    overlapping = frame.index[match_count.gt(1)]
    if len(overlapping):
        raise ValueError(
            "Funding support row matches multiple governed support kinds; "
            f"indexes={list(overlapping)}"
        )

    explicit = _explicit_support_metadata(frame)
    unmatched_explicit = frame.index[explicit & match_count.eq(0)]
    if strict and len(unmatched_explicit):
        raise ValueError(
            "Explicit funding/support metadata does not match funding_support_specs_v1; "
            f"indexes={list(unmatched_explicit)}"
        )

    out = frame.loc[match_count.eq(1)].copy()
    if out.empty:
        out[SUPPORT_KIND_COLUMN] = pd.Series(dtype="object")
        out[SUPPORT_SPEC_ID_COLUMN] = pd.Series(dtype="object")
        out[SUPPORT_MEASURE_COLUMN] = pd.Series(dtype="object")
        out[SUPPORT_AMOUNT_COLUMN] = pd.Series(dtype="float64")
        out[SOURCE_MEMBER_IDS_COLUMN] = pd.Series(dtype="object")
        return out

    spec_id = pd.Series("", index=out.index, dtype="object")
    support_kind = pd.Series("", index=out.index, dtype="object")
    for spec in _SPECS:
        selected = masks[spec.spec_id].reindex(out.index, fill_value=False)
        spec_id.loc[selected] = spec.spec_id
        support_kind.loc[selected] = spec.support_kind
    out[SUPPORT_SPEC_ID_COLUMN] = spec_id
    out[SUPPORT_KIND_COLUMN] = support_kind

    measures = pd.Series("", index=out.index, dtype="object")
    amounts = pd.Series(pd.NA, index=out.index, dtype="Float64")
    bucket = _text(out, "semantic_bucket")
    subbucket = _text(out, "semantic_subbucket")
    for idx in out.index:
        measure = resolve_semantic_measure(bucket.loc[idx], subbucket.loc[idx])
        if measure is None:
            raise ValueError(
                "Governed funding-support member lacks semantic measure: "
                f"index={idx}; bucket={bucket.loc[idx]!r}; subbucket={subbucket.loc[idx]!r}"
            )
        if measure not in out.columns:
            raise ValueError(
                f"Funding-support source is missing governed amount column {measure!r}"
            )
        measures.loc[idx] = measure
        amounts.loc[idx] = pd.to_numeric(pd.Series([out.loc[idx, measure]]), errors="coerce").iloc[0]
    out[SUPPORT_MEASURE_COLUMN] = measures
    out[SUPPORT_AMOUNT_COLUMN] = amounts.fillna(0.0)
    out[SOURCE_MEMBER_IDS_COLUMN] = _source_member_ids(out)
    return out


def funding_support_membership_mask(frame: pd.DataFrame, *, strict: bool = True) -> pd.Series:
    members = classify_funding_support(frame, strict=strict)
    return frame.index.to_series().isin(members.index)
