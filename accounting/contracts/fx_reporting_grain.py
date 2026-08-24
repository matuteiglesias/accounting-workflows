"""Explicit reporting grain authority for native-currency FX flows.

`Box=None` is never interpreted as Currency-total.  A governed FX row must carry
an explicit grain identity assigned from stable producer metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

import pandas as pd

from accounting.contracts.semantic_measures import resolve_semantic_measure


FXReportingGrain = Literal["currency_total", "box_currency"]
FXMeasureKind = Literal["conversion_in", "conversion_out", "fx_cost", "net"]
FX_REPORTING_GRAIN_VERSION: Final = "fx_reporting_grain_v1"
FX_REPORTING_GRAIN_COLUMN: Final = "fx_reporting_grain"


@dataclass(frozen=True, slots=True)
class FXReportingSpec:
    spec_id: str
    measure_kind: FXMeasureKind
    grain: FXReportingGrain
    measure_id: str
    semantic_subbucket: str = ""

    def __post_init__(self) -> None:
        if not self.spec_id or self.spec_id != self.spec_id.strip():
            raise ValueError("FXReportingSpec.spec_id must be non-empty and normalized")
        if self.grain not in {"currency_total", "box_currency"}:
            raise ValueError(f"Unsupported FX reporting grain: {self.grain!r}")
        if self.measure_kind == "net":
            if self.measure_id != "net_amount" or self.semantic_subbucket:
                raise ValueError("FX net must use net_amount over the treasury_fx bucket")
        else:
            governed = resolve_semantic_measure("treasury_fx", self.semantic_subbucket)
            if governed != self.measure_id:
                raise ValueError(
                    "FX reporting measure must match semantic_measure_registry_v1; "
                    f"subbucket={self.semantic_subbucket!r}; expected={governed!r}; got={self.measure_id!r}"
                )


def _spec(kind: FXMeasureKind, grain: FXReportingGrain) -> FXReportingSpec:
    semantic = {
        "conversion_in": ("fx_conversion_proceeds", "amount_in"),
        "conversion_out": ("fx_conversion_outflow", "amount_out"),
        "fx_cost": ("fx_cost_or_spread", "amount_out"),
        "net": ("", "net_amount"),
    }
    subbucket, measure = semantic[kind]
    return FXReportingSpec(
        spec_id=f"fx.{kind}.{grain}",
        measure_kind=kind,
        grain=grain,
        measure_id=measure,
        semantic_subbucket=subbucket,
    )


_SPECS = tuple(
    _spec(kind, grain)
    for grain in ("currency_total", "box_currency")
    for kind in ("conversion_in", "conversion_out", "fx_cost", "net")
)
FX_REPORTING_SPECS: Final[Mapping[str, FXReportingSpec]] = MappingProxyType(
    {spec.spec_id: spec for spec in _SPECS}
)

DEDICATED_FX_BOX_TABLE_IDS: Final = frozenset(
    {
        "monthly_tables_fx_treasury_all_measures",
        "monthly_tables_fx_treasury_amount_in",
        "monthly_tables_fx_treasury_amount_out",
        "monthly_tables_fx_treasury_net_amount",
        "monthly_tables_fx_treasury_compact",
    }
)

_METRIC_KIND = MappingProxyType(
    {
        "TR.FX.CONVERSION.IN": "conversion_in",
        "TR.FX.CONVERSION.OUT": "conversion_out",
        "TR.FX.COST.OUT": "fx_cost",
        "TR.FX.NET": "net",
    }
)
_STATEMENT_KIND = MappingProxyType(
    {
        "treasury_fx_conversion_in": "conversion_in",
        "treasury_fx_conversion_out": "conversion_out",
        "treasury_fx_cost": "fx_cost",
        "treasury_fx_net": "net",
    }
)
_LEGACY_METRIC_KIND = MappingProxyType(
    {
        "fx_conversion_proceeds": "conversion_in",
        "fx_conversion_outflow": "conversion_out",
        "fx_cost_or_spread": "fx_cost",
        "fx_net": "net",
        "net_amount": "net",
    }
)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def producer_fx_reporting_grain(table_id: str, row: pd.Series) -> FXReportingGrain | None:
    """Resolve grain from stable producer identity, never from Box nullability."""

    if table_id in DEDICATED_FX_BOX_TABLE_IDS:
        return "box_currency"
    metric_id = _text(row.get("metric_id"))
    statement_line = _text(row.get("statement_line"))
    if metric_id in _METRIC_KIND or statement_line in _STATEMENT_KIND:
        return "currency_total"
    return None


def producer_fx_measure_kind(table_id: str, row: pd.Series) -> FXMeasureKind | None:
    metric_id = _text(row.get("metric_id"))
    if metric_id in _METRIC_KIND:
        return _METRIC_KIND[metric_id]  # type: ignore[return-value]
    statement_line = _text(row.get("statement_line"))
    if statement_line in _STATEMENT_KIND:
        return _STATEMENT_KIND[statement_line]  # type: ignore[return-value]

    if table_id in DEDICATED_FX_BOX_TABLE_IDS:
        metric = _text(row.get("metric"))
        if metric in _LEGACY_METRIC_KIND:
            return _LEGACY_METRIC_KIND[metric]  # type: ignore[return-value]
        explicit_measure = _text(row.get("measure"))
        if table_id == "monthly_tables_fx_treasury_amount_in":
            return "conversion_in"
        if table_id == "monthly_tables_fx_treasury_amount_out":
            if metric == "fx_cost_or_spread":
                return "fx_cost"
            return "conversion_out"
        if table_id in {"monthly_tables_fx_treasury_net_amount", "monthly_tables_fx_treasury_compact"}:
            return "net"
        if explicit_measure == "amount_in":
            return "conversion_in"
        if explicit_measure == "amount_out":
            return None  # amount_out is ambiguous between conversion out and FX cost
        if explicit_measure == "net_amount":
            return "net"
    return None


def resolve_fx_reporting_spec(table_id: str, row: pd.Series) -> FXReportingSpec | None:
    grain = _text(row.get(FX_REPORTING_GRAIN_COLUMN))
    if grain not in {"currency_total", "box_currency"}:
        return None
    expected = producer_fx_reporting_grain(table_id, row)
    if expected is not None and grain != expected:
        raise ValueError(
            "FX reporting grain conflicts with stable producer identity: "
            f"table_id={table_id!r}; expected={expected!r}; got={grain!r}"
        )
    kind = producer_fx_measure_kind(table_id, row)
    if kind is None:
        return None
    return FX_REPORTING_SPECS[f"fx.{kind}.{grain}"]


def validate_fx_row_grain(row: pd.Series, spec: FXReportingSpec) -> tuple[bool, str]:
    currency = _text(row.get("Currency"))
    if not currency:
        return False, "missing Currency would risk cross-currency aggregation"
    box = _text(row.get("Box"))
    if spec.grain == "box_currency" and not box:
        return False, "box_currency FX row requires explicit Box"
    if spec.grain == "currency_total" and box:
        return False, "currency_total FX row must not carry Box; Box aggregation must be explicit upstream"
    return True, ""
