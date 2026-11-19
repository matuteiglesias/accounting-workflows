from __future__ import annotations

"""Governed selection of inferred Box-control snapshots.

This module owns the accounting-control primitive independently of validated
cash selection.  Inferred control is evidence for reconciliation and never a
cash-headline fallback.
"""

from dataclasses import dataclass
from typing import Final

import pandas as pd

from accounting.contracts.cash_position_control import resolve_inferred_box_control_spec


INFERRED_SPEC_ID: Final = "cash.control.inferred_box_motor"
_INFERRED_REQUIRED_COLUMNS: Final[set[str]] = {
    "period",
    "Currency",
    "Box",
    "close_amount",
    "position_type",
    "source_type",
    "cash_suitability",
    "is_frontend_safe",
    "as_of_date",
}
BOX_MOTOR_SOURCE: Final = "box_balance_time_long.freq=M.csv"
_BOX_MOTOR_REQUIRED_COLUMNS: Final[set[str]] = {
    "TimePeriod", "TimePeriod_end", "Box", "Currency", "cum_net",
}


@dataclass(frozen=True)
class BoxControlSelection:
    status: str
    reason: str
    value: float | None
    period: str
    currency: str
    box: str
    selected: pd.DataFrame
    candidates: pd.DataFrame
    excluded_inferred: pd.DataFrame
    excluded_internal: pd.DataFrame
    excluded_other: pd.DataFrame

    @property
    def available(self) -> bool:
        return self.status == "available"


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _truth(series: pd.Series) -> pd.Series:
    return _text(series).str.lower().isin({"true", "1", "yes", "y"})


def _base_scope(
    cash: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str = "",
) -> pd.DataFrame:
    if cash is None or cash.empty:
        return pd.DataFrame(columns=[] if cash is None else cash.columns)
    if not {"period", "Currency", "Box"}.issubset(cash.columns):
        return cash.iloc[0:0].copy()
    mask = _text(cash["period"]).eq(str(period).strip())
    mask &= _text(cash["Currency"]).eq(str(currency).strip())
    if str(box).strip():
        mask &= _text(cash["Box"]).eq(str(box).strip())
    return cash.loc[mask].copy()


def _partition_exclusions(scope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scope.empty or "position_type" not in scope.columns:
        return pd.DataFrame(), pd.DataFrame(), scope.copy()
    position = _text(scope["position_type"])
    inferred = scope.loc[position.eq("inferred_box_motor")].copy()
    internal = scope.loc[position.eq("internal_balance")].copy()
    other = scope.loc[~position.isin({"inferred_box_motor", "internal_balance", "cash_close"})].copy()
    return inferred, internal, other


def inferred_control_schema_supported(cash: pd.DataFrame) -> bool:
    return cash is not None and _INFERRED_REQUIRED_COLUMNS.issubset(cash.columns)


def box_motor_schema_supported(box_motor: pd.DataFrame) -> bool:
    """Return whether a frame is the native monthly Box motor artifact."""

    return box_motor is not None and _BOX_MOTOR_REQUIRED_COLUMNS.issubset(box_motor.columns)


def _box_motor_scope(
    box_motor: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
    cutoff_date: str | None,
) -> tuple[pd.DataFrame, str]:
    if not box_motor_schema_supported(box_motor):
        return pd.DataFrame(columns=[] if box_motor is None else box_motor.columns), "legacy_or_incomplete_box_motor_schema"
    work = box_motor.copy()
    work["__period"] = _text(work["TimePeriod"])
    work["__period_end"] = pd.to_datetime(work["TimePeriod_end"], errors="coerce")
    requested = str(period).strip()
    try:
        requested_period = pd.Period(requested, freq="M")
    except Exception:
        return work.iloc[0:0].copy(), "invalid_month_period"
    if not requested or requested != str(requested_period):
        return work.iloc[0:0].copy(), "invalid_month_period"
    if cutoff_date:
        cutoff = pd.to_datetime(cutoff_date, errors="coerce")
        if pd.isna(cutoff):
            return work.iloc[0:0].copy(), "invalid_cutoff_date"
        if requested_period.end_time.normalize() > cutoff.normalize():
            return work.iloc[0:0].copy(), "period_after_cutoff"
    scope = work.loc[
        work["__period"].eq(requested)
        & _text(work["Currency"]).eq(currency)
        & _text(work["Box"]).eq(box)
    ].copy()
    if scope.empty:
        return scope, "no_box_motor_candidate"
    ends = scope["__period_end"]
    if ends.isna().any() or (ends.dt.to_period("M") != requested_period).any():
        return scope, "malformed_box_motor_period"
    if scope.duplicated(["__period", "__period_end", "Box", "Currency"], keep=False).any():
        return scope, "duplicate_box_motor_snapshot"
    values = pd.to_numeric(scope["cum_net"], errors="coerce")
    if values.isna().any():
        return scope, "box_motor_value_not_numeric"
    return scope, ""


def select_box_motor_control_period(
    box_motor: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
    cutoff_date: str | None = None,
) -> BoxControlSelection:
    """Select one native monthly Box-motor cumulative control snapshot.

    This selector deliberately knows nothing about bank accounts, validation,
    or frontend suitability. It fails closed on malformed or duplicate native
    keys and preserves negative and zero controls as valid numeric values.
    """

    period, currency, box = str(period).strip(), str(currency).strip(), str(box).strip()
    empty = pd.DataFrame(columns=[] if box_motor is None else box_motor.columns)
    if not currency or not box:
        return BoxControlSelection("unavailable", "missing_currency_or_box", None, period, currency, box, empty, empty, empty, empty, empty)
    if box_motor is None or box_motor.empty:
        return BoxControlSelection("unavailable", "missing_source", None, period, currency, box, empty, empty, empty, empty, empty)
    if not box_motor_schema_supported(box_motor):
        return BoxControlSelection("unsupported", "legacy_or_incomplete_box_motor_schema", None, period, currency, box, empty, empty, empty, empty, box_motor.copy())
    candidates, reason = _box_motor_scope(box_motor, period=period, currency=currency, box=box, cutoff_date=cutoff_date)
    clean = candidates.drop(columns=["__period", "__period_end"], errors="ignore")
    if reason:
        status = "unsupported" if reason in {"legacy_or_incomplete_box_motor_schema", "invalid_month_period", "invalid_cutoff_date", "malformed_box_motor_period"} else "unavailable"
        return BoxControlSelection(status, reason, None, period, currency, box, empty, clean, empty, empty, empty)
    value = float(pd.to_numeric(candidates.iloc[0]["cum_net"], errors="coerce"))
    selected = clean.copy()
    return BoxControlSelection("available", "", value, period, currency, box, selected, clean, empty, empty, empty)


def select_inferred_box_control_period(
    cash: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
) -> BoxControlSelection:
    """Select the governed inferred Box-control snapshot."""

    spec = resolve_inferred_box_control_spec(INFERRED_SPEC_ID)
    if spec is None:  # pragma: no cover
        raise RuntimeError(f"Missing inferred control contract: {INFERRED_SPEC_ID}")
    period, currency, box = str(period).strip(), str(currency).strip(), str(box).strip()
    empty = pd.DataFrame(columns=[] if cash is None else cash.columns)
    if not currency or not box:
        return BoxControlSelection("unavailable", "missing_currency_or_box", None, period, currency, box, empty, empty, empty, empty, empty)
    if cash is None or cash.empty:
        return BoxControlSelection("unavailable", "missing_source", None, period, currency, box, empty, empty, empty, empty, empty)
    if not inferred_control_schema_supported(cash):
        return BoxControlSelection("unsupported", "legacy_or_incomplete_cash_schema", None, period, currency, box, empty, empty, empty, empty, cash.copy())

    scope = _base_scope(cash, period=period, currency=currency, box=box)
    inferred, internal, other = _partition_exclusions(scope)
    mask = _text(scope["position_type"]).eq(spec.position_type)
    mask &= _text(scope["source_type"]).eq(spec.source_type)
    mask &= _text(scope["cash_suitability"]).eq(spec.cash_suitability)
    mask &= ~_truth(scope["is_frontend_safe"])
    candidates = scope.loc[mask].copy()
    if candidates.empty:
        return BoxControlSelection("unavailable", "no_inferred_control_candidate", None, period, currency, box, empty, candidates, inferred, internal, other)
    candidates["__as_of_date"] = pd.to_datetime(candidates[spec.as_of_field], errors="coerce")
    valid = candidates.loc[candidates["__as_of_date"].notna()].copy()
    if valid.empty:
        return BoxControlSelection("unavailable", "no_valid_inferred_as_of", None, period, currency, box, empty, candidates.drop(columns=["__as_of_date"], errors="ignore"), inferred, internal, other)
    latest_as_of = valid["__as_of_date"].max()
    selected = valid.loc[valid["__as_of_date"].eq(latest_as_of)].copy()
    if len(selected) != 1:
        return BoxControlSelection("unavailable", "duplicate_inferred_snapshot", None, period, currency, box, empty, candidates.drop(columns=["__as_of_date"], errors="ignore"), inferred, internal, other)
    value = pd.to_numeric(selected.iloc[0].get(spec.value_ref), errors="coerce")
    if pd.isna(value):
        return BoxControlSelection("unavailable", "selected_control_value_not_numeric", None, period, currency, box, empty, candidates.drop(columns=["__as_of_date"], errors="ignore"), inferred, internal, other)
    return BoxControlSelection(
        "available", "", float(value), period, currency, box,
        selected.drop(columns=["__as_of_date"], errors="ignore"),
        candidates.drop(columns=["__as_of_date"], errors="ignore"), inferred, internal, other,
    )
