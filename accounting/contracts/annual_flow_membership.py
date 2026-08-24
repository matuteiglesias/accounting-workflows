"""Governed annual lineage for additive flow metrics.

Annual flow values may only be composed from governed monthly atomic-flow cell
membership.  This module deliberately excludes stocks, ratios, quality metrics,
and formulas: their time aggregation rules are different authorities.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

import pandas as pd

from accounting.contracts.atomic_flow_drilldowns import (
    FlowCellSpec,
    resolve_flow_cell_spec,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure


ANNUAL_FLOW_MEMBERSHIP_VERSION: Final = "annual_flow_membership_v1"
ANNUAL_FLOW_MEMBERSHIP_COLUMNS: Final = [
    "annual_cell_id",
    "metric_id",
    "period",
    "Currency",
    "dimension_name",
    "dimension_value",
    "monthly_flow_cell_id",
    "aggregation_rule",
    "monthly_governed_cell_ids",
    "source_member_ids",
    "measure_id",
    "lineage_version",
    "value",
    "member_months",
]


@dataclass(frozen=True, slots=True)
class AnnualFlowMembershipSpec:
    annual_cell_family: str
    metric_id: str
    monthly_flow_cell_id: str
    dimension_name: str = ""
    flow_or_stock: Literal["flow"] = "flow"
    aggregation_rule: Literal["sum_monthly_governed_values"] = (
        "sum_monthly_governed_values"
    )

    def __post_init__(self) -> None:
        for field_name in ("annual_cell_family", "metric_id", "monthly_flow_cell_id"):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise ValueError(f"{field_name} must be non-empty and normalized")
        if self.flow_or_stock != "flow":
            raise ValueError("AnnualFlowMembershipSpec is flow-only; stocks need a closing selector")
        monthly = resolve_flow_cell_spec(self.monthly_flow_cell_id)
        if monthly is None:
            raise ValueError(
                f"Unknown monthly governed flow cell: {self.monthly_flow_cell_id!r}"
            )
        expected_dims = tuple(monthly.grain[2:])
        if self.dimension_name:
            if expected_dims != (self.dimension_name,):
                raise ValueError(
                    "Annual dimension must match the monthly governed cell grain; "
                    f"expected={expected_dims!r}; got={self.dimension_name!r}"
                )
        elif expected_dims:
            raise ValueError(
                "Annual total spec cannot discard monthly governed dimensions; "
                f"monthly dimensions={expected_dims!r}"
            )


_SPECS = (
    AnnualFlowMembershipSpec(
        "annual.rent.total", "IS.RENT.TOTAL", "flow.rent.total"
    ),
    AnnualFlowMembershipSpec(
        "annual.opex.property", "IS.OPEX.PROPERTY", "flow.property_opex.total"
    ),
    AnnualFlowMembershipSpec(
        "annual.draws.personal",
        "DIST.DRAWS.PERSONAL",
        "flow.family_draws_or_distributions.total",
    ),
    AnnualFlowMembershipSpec(
        "annual.opex.by_category",
        "IS.OPEX.BY_CATEGORY",
        "flow.property_opex.by_category",
        dimension_name="semantic_subbucket",
    ),
)

ANNUAL_FLOW_MEMBERSHIP_SPECS: Final[Mapping[str, AnnualFlowMembershipSpec]] = (
    MappingProxyType({spec.metric_id: spec for spec in _SPECS})
)
if len(ANNUAL_FLOW_MEMBERSHIP_SPECS) != len(_SPECS):
    raise ValueError("Duplicate metric_id in annual_flow_membership_v1")


def resolve_annual_flow_membership_spec(
    metric_id: str,
) -> AnnualFlowMembershipSpec | None:
    return ANNUAL_FLOW_MEMBERSHIP_SPECS.get(str(metric_id).strip())


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def _semantic_members_mask(frame: pd.DataFrame, spec: FlowCellSpec) -> pd.Series:
    bucket = _text(frame, "semantic_bucket")
    subbucket = _text(frame, "semantic_subbucket")
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for member_bucket, member_subbucket in spec.semantic_members:
        member = bucket.eq(member_bucket)
        if member_subbucket:
            member &= subbucket.eq(member_subbucket)
        mask |= member
    return mask


def _source_ids(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {token.strip() for token in text.replace(",", ";").split(";") if token.strip()}


def _source_member_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("source_tx_ids_sample", "tx_id", "source_member_id"):
        if column in frame.columns:
            return frame[column]
    return pd.Series("", index=frame.index, dtype="object")


def _monthly_instance_id(
    spec: AnnualFlowMembershipSpec,
    month: str,
    currency: str,
    dimension_value: str,
) -> str:
    suffix = f"|{spec.dimension_name}={dimension_value}" if spec.dimension_name else ""
    return f"{spec.monthly_flow_cell_id}|period={month}|Currency={currency}{suffix}"


def _annual_instance_id(
    spec: AnnualFlowMembershipSpec,
    year: str,
    currency: str,
    dimension_value: str,
) -> str:
    suffix = f"|{spec.dimension_name}={dimension_value}" if spec.dimension_name else ""
    return f"{spec.annual_cell_family}|period={year}|Currency={currency}{suffix}"


def build_annual_flow_membership(split: pd.DataFrame) -> pd.DataFrame:
    """Materialize annual flow values and lineage from governed monthly cells."""

    if split is None or split.empty:
        return pd.DataFrame(columns=ANNUAL_FLOW_MEMBERSHIP_COLUMNS)
    required = {"period", "Currency", "semantic_bucket", "semantic_subbucket"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"monthly_flow_semantic_split missing annual-lineage columns: {missing}")

    rows: list[dict[str, object]] = []
    for annual_spec in _SPECS:
        monthly_spec = resolve_flow_cell_spec(annual_spec.monthly_flow_cell_id)
        assert monthly_spec is not None
        measure_id = resolve_semantic_measure(*monthly_spec.measure_ref)
        if measure_id is None or measure_id not in split.columns:
            raise ValueError(
                "Annual flow lineage cannot resolve governed monthly measure; "
                f"cell={monthly_spec.cell_id!r}; measure={measure_id!r}"
            )

        members = split.loc[_semantic_members_mask(split, monthly_spec)].copy()
        if members.empty:
            continue
        members["period"] = _text(members, "period")
        members = members.loc[members["period"].str.match(r"^\d{4}-\d{2}$")].copy()
        if members.empty:
            continue
        members["year"] = members["period"].str.slice(0, 4)
        members["Currency"] = _text(members, "Currency")
        if members["Currency"].eq("").any():
            raise ValueError("Annual flow membership requires explicit Currency")
        members["governed_value"] = pd.to_numeric(
            members[measure_id], errors="coerce"
        ).fillna(0.0)
        members["__source_members"] = _source_member_series(members).map(_source_ids)

        group_dims = ["period", "year", "Currency"]
        if annual_spec.dimension_name:
            if annual_spec.dimension_name not in members.columns:
                raise ValueError(
                    f"Annual flow membership missing dimension {annual_spec.dimension_name!r}"
                )
            members[annual_spec.dimension_name] = _text(
                members, annual_spec.dimension_name
            )
            if members[annual_spec.dimension_name].eq("").any():
                raise ValueError(
                    "Annual dimensioned flow membership cannot silently aggregate a "
                    f"blank {annual_spec.dimension_name}"
                )
            group_dims.append(annual_spec.dimension_name)

        monthly_rows: list[dict[str, object]] = []
        for key, group in members.groupby(group_dims, dropna=False, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            month, year, currency = map(str, key[:3])
            dim_value = str(key[3]) if annual_spec.dimension_name else ""
            source_ids = sorted(set().union(*group["__source_members"].tolist()))
            monthly_rows.append(
                {
                    "month": month,
                    "year": year,
                    "Currency": currency,
                    "dimension_value": dim_value,
                    "value": float(group["governed_value"].sum()),
                    "monthly_cell_id": _monthly_instance_id(
                        annual_spec, month, currency, dim_value
                    ),
                    "source_ids": source_ids,
                }
            )

        monthly = pd.DataFrame(monthly_rows)
        annual_dims = ["year", "Currency"]
        if annual_spec.dimension_name:
            annual_dims.append("dimension_value")
        for key, group in monthly.groupby(annual_dims, dropna=False, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            year, currency = map(str, key[:2])
            dim_value = str(key[2]) if annual_spec.dimension_name else ""
            source_ids = sorted(set().union(*group["source_ids"].tolist()))
            month_cells = sorted(group["monthly_cell_id"].astype(str).tolist())
            member_months = sorted(group["month"].astype(str).tolist())
            rows.append(
                {
                    "annual_cell_id": _annual_instance_id(
                        annual_spec, year, currency, dim_value
                    ),
                    "metric_id": annual_spec.metric_id,
                    "period": year,
                    "Currency": currency,
                    "dimension_name": annual_spec.dimension_name,
                    "dimension_value": dim_value,
                    "monthly_flow_cell_id": annual_spec.monthly_flow_cell_id,
                    "aggregation_rule": annual_spec.aggregation_rule,
                    "monthly_governed_cell_ids": ";".join(month_cells),
                    "source_member_ids": ";".join(source_ids),
                    "measure_id": measure_id,
                    "lineage_version": ANNUAL_FLOW_MEMBERSHIP_VERSION,
                    "value": float(group["value"].sum()),
                    "member_months": ";".join(member_months),
                }
            )

    return pd.DataFrame(rows, columns=ANNUAL_FLOW_MEMBERSHIP_COLUMNS).sort_values(
        ["metric_id", "period", "Currency", "dimension_name", "dimension_value"]
    ).reset_index(drop=True)
