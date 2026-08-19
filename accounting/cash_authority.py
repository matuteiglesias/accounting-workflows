from __future__ import annotations

"""Shared governed selectors for validated cash and inferred box control.

This module is the runtime authority for Wave 4 cash/control selection.  It
consumes the declarative PR15A contracts and deliberately does not depend on
professional rendering, metrics presentation, or human labels.
"""

from dataclasses import dataclass
from typing import Final

import pandas as pd

from accounting.contracts.cash_position_control import (
    InferredBoxControlSpec,
    ValidatedCashPositionSpec,
    resolve_inferred_box_control_spec,
    resolve_validated_cash_position_spec,
)


VALIDATED_SPEC_ID: Final = "cash.position.validated"
INFERRED_SPEC_ID: Final = "cash.control.inferred_box_motor"

_VALIDATED_REQUIRED_COLUMNS: Final[set[str]] = {
    "period",
    "Currency",
    "Box",
    "account_id",
    "close_amount",
    "position_type",
    "cash_suitability",
    "is_frontend_safe",
    "validation_status",
    "validated_by",
    "source_type",
    "as_of_date",
}
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


@dataclass(frozen=True)
class CashSelection:
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


def validated_cash_schema_supported(cash: pd.DataFrame) -> bool:
    return cash is not None and _VALIDATED_REQUIRED_COLUMNS.issubset(cash.columns)


def inferred_control_schema_supported(cash: pd.DataFrame) -> bool:
    return cash is not None and _INFERRED_REQUIRED_COLUMNS.issubset(cash.columns)


def _partition_exclusions(scope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scope.empty or "position_type" not in scope.columns:
        return pd.DataFrame(), pd.DataFrame(), scope.copy()
    position = _text(scope["position_type"])
    inferred = scope.loc[position.eq("inferred_box_motor")].copy()
    internal = scope.loc[position.eq("internal_balance")].copy()
    other = scope.loc[~position.isin({"inferred_box_motor", "internal_balance", "cash_close"})].copy()
    return inferred, internal, other


def _validated_eligibility_mask(scope: pd.DataFrame, spec: ValidatedCashPositionSpec) -> pd.Series:
    mask = _text(scope["position_type"]).eq(spec.position_type)
    mask &= _text(scope["cash_suitability"]).eq(spec.cash_suitability)
    mask &= _truth(scope["is_frontend_safe"])
    mask &= _text(scope["validation_status"]).str.lower().isin(spec.allowed_validation_statuses)
    mask &= _text(scope["source_type"]).str.lower().isin(spec.allowed_source_types)
    mask &= _text(scope["validated_by"]).ne("")
    mask &= _text(scope["account_id"]).ne("")
    return mask


def select_validated_cash_period(
    cash: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str = "",
) -> CashSelection:
    """Select governed validated cash for one period/currency[/box].

    One latest valid as-of snapshot is selected per ``Box/account_id``.  The
    selected account snapshots are then summed.  Inferred and internal rows are
    evidence only and can never contribute to the matched value.
    """

    spec = resolve_validated_cash_position_spec(VALIDATED_SPEC_ID)
    if spec is None:  # pragma: no cover - import-time contract regression owns this
        raise RuntimeError(f"Missing validated cash contract: {VALIDATED_SPEC_ID}")

    currency = str(currency).strip()
    box = str(box).strip()
    period = str(period).strip()
    empty = pd.DataFrame(columns=[] if cash is None else cash.columns)
    if not currency:
        return CashSelection("unavailable", "missing_currency", None, period, currency, box, empty, empty, empty, empty, empty)
    if cash is None or cash.empty:
        return CashSelection("unavailable", "missing_source", None, period, currency, box, empty, empty, empty, empty, empty)
    if not validated_cash_schema_supported(cash):
        return CashSelection("unsupported", "legacy_or_incomplete_cash_schema", None, period, currency, box, empty, empty, empty, empty, cash.copy())

    scope = _base_scope(cash, period=period, currency=currency, box=box)
    inferred, internal, other = _partition_exclusions(scope)
    if scope.empty:
        return CashSelection("unavailable", "no_rows_for_period_scope", None, period, currency, box, empty, empty, inferred, internal, other)

    eligible_mask = _validated_eligibility_mask(scope, spec)
    candidates = scope.loc[eligible_mask].copy()
    # Non-authoritative cash_close rows are retained as excluded evidence.
    other = pd.concat([other, scope.loc[_text(scope["position_type"]).eq("cash_close") & ~eligible_mask].copy()], ignore_index=False)
    if candidates.empty:
        return CashSelection("unavailable", "no_validated_cash_candidates", None, period, currency, box, empty, candidates, inferred, internal, other)

    candidates = candidates.copy()
    candidates["__as_of_date"] = pd.to_datetime(candidates[spec.as_of_field], errors="coerce")
    account_keys = ["Box", "account_id"]
    selected_parts: list[pd.DataFrame] = []
    for _, account_rows in candidates.groupby(account_keys, dropna=False, sort=True):
        valid = account_rows.loc[account_rows["__as_of_date"].notna()].copy()
        if valid.empty:
            return CashSelection(
                "unavailable",
                "candidate_account_has_no_valid_as_of",
                None,
                period,
                currency,
                box,
                empty,
                candidates.drop(columns=["__as_of_date"], errors="ignore"),
                inferred,
                internal,
                other,
            )
        latest_as_of = valid["__as_of_date"].max()
        selected_account = valid.loc[valid["__as_of_date"].eq(latest_as_of)].copy()
        if len(selected_account) != 1:
            return CashSelection(
                "unavailable",
                "duplicate_latest_account_as_of",
                None,
                period,
                currency,
                box,
                empty,
                candidates.drop(columns=["__as_of_date"], errors="ignore"),
                inferred,
                internal,
                other,
            )
        selected_parts.append(selected_account)

    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else empty
    selected[spec.value_ref] = pd.to_numeric(selected[spec.value_ref], errors="coerce")
    if selected[spec.value_ref].isna().any():
        return CashSelection(
            "unavailable",
            "selected_cash_value_not_numeric",
            None,
            period,
            currency,
            box,
            empty,
            candidates.drop(columns=["__as_of_date"], errors="ignore"),
            inferred,
            internal,
            other,
        )
    value = float(selected[spec.value_ref].sum())
    selected = selected.drop(columns=["__as_of_date"], errors="ignore")
    candidates = candidates.drop(columns=["__as_of_date"], errors="ignore")
    return CashSelection("available", "", value, period, currency, box, selected, candidates, inferred, internal, other)


def _periods_for_year(cash: pd.DataFrame, year: str, currency: str, box: str) -> list[str]:
    if cash is None or cash.empty or not {"period", "Currency", "Box"}.issubset(cash.columns):
        return []
    periods = _text(cash["period"])
    mask = periods.str.startswith(f"{str(year).strip()}-")
    mask &= _text(cash["Currency"]).eq(str(currency).strip())
    if str(box).strip():
        mask &= _text(cash["Box"]).eq(str(box).strip())
    return sorted(periods.loc[mask].loc[lambda s: s.str.match(r"^20\d{2}-(0[1-9]|1[0-2])$")].unique().tolist())


def select_validated_cash_year(
    cash: pd.DataFrame,
    *,
    year: str,
    currency: str,
    box: str = "",
) -> CashSelection:
    """Select annual closing cash using the same monthly account primitive.

    The latest period containing headline-eligible validated candidates is
    selected.  If that period is incomplete/invalid, annual cash is unavailable;
    the selector never falls back to an older period merely to manufacture a
    value.
    """

    year = str(year).strip()
    if cash is None or cash.empty or not validated_cash_schema_supported(cash):
        return select_validated_cash_period(cash, period=f"{year}-12", currency=currency, box=box)

    periods = _periods_for_year(cash, year, currency, box)
    candidate_periods: list[str] = []
    spec = resolve_validated_cash_position_spec(VALIDATED_SPEC_ID)
    assert spec is not None
    for period in periods:
        scope = _base_scope(cash, period=period, currency=currency, box=box)
        if not scope.empty and _validated_eligibility_mask(scope, spec).any():
            candidate_periods.append(period)
    if not candidate_periods:
        empty = cash.iloc[0:0].copy()
        return CashSelection("unavailable", "no_validated_cash_period_in_year", None, "", str(currency).strip(), str(box).strip(), empty, empty, empty, empty, empty)
    return select_validated_cash_period(
        cash,
        period=candidate_periods[-1],
        currency=currency,
        box=box,
    )


def select_inferred_box_control_period(
    cash: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
) -> CashSelection:
    """Select the governed inferred box-motor control snapshot.

    This primitive is intentionally separate from validated cash and can never
    be used as a cash-headline fallback.
    """

    spec = resolve_inferred_box_control_spec(INFERRED_SPEC_ID)
    if spec is None:  # pragma: no cover
        raise RuntimeError(f"Missing inferred control contract: {INFERRED_SPEC_ID}")
    period, currency, box = str(period).strip(), str(currency).strip(), str(box).strip()
    empty = pd.DataFrame(columns=[] if cash is None else cash.columns)
    if not currency or not box:
        return CashSelection("unavailable", "missing_currency_or_box", None, period, currency, box, empty, empty, empty, empty, empty)
    if cash is None or cash.empty:
        return CashSelection("unavailable", "missing_source", None, period, currency, box, empty, empty, empty, empty, empty)
    if not inferred_control_schema_supported(cash):
        return CashSelection("unsupported", "legacy_or_incomplete_cash_schema", None, period, currency, box, empty, empty, empty, empty, cash.copy())

    scope = _base_scope(cash, period=period, currency=currency, box=box)
    inferred, internal, other = _partition_exclusions(scope)
    mask = _text(scope["position_type"]).eq(spec.position_type)
    mask &= _text(scope["source_type"]).eq(spec.source_type)
    mask &= _text(scope["cash_suitability"]).eq(spec.cash_suitability)
    mask &= ~_truth(scope["is_frontend_safe"])
    candidates = scope.loc[mask].copy()
    if candidates.empty:
        return CashSelection("unavailable", "no_inferred_control_candidate", None, period, currency, box, empty, candidates, inferred, internal, other)
    candidates["__as_of_date"] = pd.to_datetime(candidates[spec.as_of_field], errors="coerce")
    valid = candidates.loc[candidates["__as_of_date"].notna()].copy()
    if valid.empty:
        return CashSelection("unavailable", "no_valid_inferred_as_of", None, period, currency, box, empty, candidates.drop(columns=["__as_of_date"], errors="ignore"), inferred, internal, other)
    latest_as_of = valid["__as_of_date"].max()
    selected = valid.loc[valid["__as_of_date"].eq(latest_as_of)].copy()
    if len(selected) != 1:
        return CashSelection("unavailable", "duplicate_inferred_snapshot", None, period, currency, box, empty, candidates.drop(columns=["__as_of_date"], errors="ignore"), inferred, internal, other)
    value = pd.to_numeric(selected.iloc[0].get(spec.value_ref), errors="coerce")
    if pd.isna(value):
        return CashSelection("unavailable", "selected_control_value_not_numeric", None, period, currency, box, empty, candidates.drop(columns=["__as_of_date"], errors="ignore"), inferred, internal, other)
    return CashSelection(
        "available",
        "",
        float(value),
        period,
        currency,
        box,
        selected.drop(columns=["__as_of_date"], errors="ignore"),
        candidates.drop(columns=["__as_of_date"], errors="ignore"),
        inferred,
        internal,
        other,
    )
