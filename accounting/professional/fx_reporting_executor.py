"""Governed professional execution for explicit native-currency FX grains."""

from __future__ import annotations

import pandas as pd

from accounting.contracts.fx_reporting_grain import (
    FX_REPORTING_GRAIN_VERSION,
    FXReportingSpec,
    resolve_fx_reporting_spec,
    validate_fx_row_grain,
)
from accounting.professional import drilldown_legacy as _legacy


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    if _legacy.YEAR_RE.match(str(period)):
        if "period" not in frame.columns:
            return pd.Series(False, index=frame.index)
        return frame["period"].fillna("").astype(str).str.startswith(f"{period}-")
    if "period" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["period"].fillna("").astype(str).eq(str(period))


def fx_semantic_mask(frame: pd.DataFrame, spec: FXReportingSpec, row: pd.Series) -> pd.Series:
    currency = _text(row.get("Currency"))
    if not currency or "Currency" not in frame.columns:
        return pd.Series(False, index=frame.index)
    mask = frame["Currency"].fillna("").astype(str).str.strip().eq(currency)
    if "semantic_bucket" not in frame.columns:
        return pd.Series(False, index=frame.index)
    mask &= frame["semantic_bucket"].fillna("").astype(str).str.strip().eq("treasury_fx")
    if spec.semantic_subbucket:
        if "semantic_subbucket" not in frame.columns:
            return pd.Series(False, index=frame.index)
        mask &= frame["semantic_subbucket"].fillna("").astype(str).str.strip().eq(
            spec.semantic_subbucket
        )
    if spec.grain == "box_currency":
        if "Box" not in frame.columns:
            return pd.Series(False, index=frame.index)
        mask &= frame["Box"].fillna("").astype(str).str.strip().eq(_text(row.get("Box")))
    return mask


def build_fx_cell_spec(table_id: str, row: pd.Series):
    """Return a legacy CellSpec backed by the explicit FX grain contract."""

    spec = resolve_fx_reporting_spec(table_id, row)
    if spec is None:
        return None
    valid, reason = validate_fx_row_grain(row, spec)
    if not valid:
        return _legacy.CellSpec(
            table_id,
            spec.measure_id,
            lambda df, r: pd.Series(False, index=df.index),
            unsupported_if=lambda r: True,
            caveat_func=lambda r, why=reason: f"Governed FX grain incompatible: {why}",
        )
    return _legacy.CellSpec(
        table_id,
        spec.measure_id,
        lambda df, r, fx_spec=spec: fx_semantic_mask(df, fx_spec, r),
        caveat_func=lambda r, fx_spec=spec: (
            f"Governed FX reporting grain={fx_spec.grain}; contract={FX_REPORTING_GRAIN_VERSION}; "
            "native currencies remain separate."
        ),
    )


def execute_fx_reporting_cell(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    display_value: float,
    split: pd.DataFrame,
    tolerance: float,
):
    """Execute derived/statement FX cells without Box-shape inference."""

    spec = resolve_fx_reporting_spec(table_id, row)
    if spec is None:
        return None
    valid, reason = validate_fx_row_grain(row, spec)
    base_filters = {
        "period": period,
        "Currency": _text(row.get("Currency")),
        "Box": _text(row.get("Box")) if spec.grain == "box_currency" else "",
        "fx_reporting_grain": spec.grain,
        "fx_measure_kind": spec.measure_kind,
        "measure": spec.measure_id,
        "lineage_version": FX_REPORTING_GRAIN_VERSION,
    }
    if not valid:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "unsupported": True, "reason": reason},
            "FX grain is explicit; incompatible Box/Currency metadata fails closed.",
            pd.DataFrame(),
            [],
        )
    if split is None or split.empty:
        return (
            _legacy.STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "error": "missing monthly_flow_semantic_split.csv"},
            "",
            pd.DataFrame(),
            [],
        )
    if spec.measure_id not in split.columns:
        return (
            _legacy.STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_flow_semantic_split.csv",
            {**base_filters, "unsupported": True, "reason": f"missing measure {spec.measure_id}"},
            "Governed FX measure is unavailable; no alternate amount column used.",
            pd.DataFrame(),
            [],
        )

    selected = split.loc[_period_mask(split, period) & fx_semantic_mask(split, spec, row)].copy()
    matched = _legacy._measure_sum(selected, spec.measure_id)
    residual = matched - display_value
    status = (
        _legacy.STATUS_EMPTY
        if selected.empty
        else _legacy.STATUS_OK
        if abs(residual) <= tolerance
        else _legacy.STATUS_RESIDUAL_WARNING
    )
    return (
        status,
        matched,
        residual,
        "governed_fx_grain",
        "monthly_flow_semantic_split.csv",
        base_filters,
        (
            f"FX reporting grain={spec.grain} is explicit under {FX_REPORTING_GRAIN_VERSION}; "
            "Box absence is never interpreted as Currency-total."
        ),
        selected,
        [("Governed FX semantic members", selected)],
    )
