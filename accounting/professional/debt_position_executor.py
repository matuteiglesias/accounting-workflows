from __future__ import annotations

"""Governed professional execution for resolved debt-position snapshots.

This module consumes ``DebtPositionSpec`` only. Debt activity remains on the
legacy path until PR14, and the debt resolver/mart remain unchanged.
"""

from typing import Any

import pandas as pd

from accounting.contracts.debt_position_activity import (
    DEBT_POSITION_SPECS_VERSION,
    DebtPositionSpec,
    resolve_debt_position_spec,
)
from accounting.professional import drilldown_legacy as _legacy

STATUS_UNAVAILABLE = "unavailable"

_VALUE_OR_COMPONENT_TO_SPEC_ID = {
    "open_principal": "debt.position.principal",
    "principal": "debt.position.principal",
    "open_interest": "debt.position.interest",
    "interest": "debt.position.interest",
    "open_total": "debt.position.total",
    "total": "debt.position.total",
}


def _resolve_row_spec(row: pd.Series) -> DebtPositionSpec | None:
    """Resolve only explicit characterized debt-position identities.

    Unknown/blank identities return ``None`` so the facade can preserve legacy
    compatibility rather than broadening the governed contract.
    """

    token = _legacy._norm(row.get("measure")) or _legacy._norm(row.get("component"))
    spec_id = _VALUE_OR_COMPONENT_TO_SPEC_ID.get(token)
    return resolve_debt_position_spec(spec_id) if spec_id else None


def _pair_identity(row: pd.Series) -> tuple[str, str, str]:
    pair = _legacy._norm(row.get("pair"))
    pair_debtor, pair_creditor = _legacy._pair_parts(pair)
    debtor = _legacy._norm(row.get("debtor")) or pair_debtor
    creditor = _legacy._norm(row.get("creditor")) or pair_creditor
    return pair, debtor, creditor


def _contract_filters(
    *,
    spec: DebtPositionSpec,
    period: str,
    currency: str,
    pair: str,
    debtor: str,
    creditor: str,
    annual: bool,
    selected_period: str = "",
) -> dict[str, Any]:
    return {
        "period": period,
        "Currency": currency,
        "pair": pair,
        "debtor": debtor,
        "creditor": creditor,
        "component": spec.component,
        "measure": spec.value_ref,
        "spec_id": spec.spec_id,
        "contract_version": DEBT_POSITION_SPECS_VERSION,
        "aggregation": spec.aggregation,
        "selection": spec.selection,
        "as_of_field": spec.as_of_field,
        "invalid_as_of_policy": spec.invalid_as_of_policy,
        "annualization": spec.annualization if annual else "n/a",
        "selected_period": selected_period,
        "source": "monthly_debt_position.csv",
        "executor": "governed_debt_position_v1",
    }


def _unsupported(
    *,
    display_value: float,
    source: str,
    filters: dict[str, Any],
    reason: str,
    caveat: str,
):
    return (
        _legacy.STATUS_UNSUPPORTED,
        0.0,
        -display_value,
        "unsupported",
        source,
        {**filters, "unsupported": True, "reason": reason},
        caveat,
        pd.DataFrame(),
        [],
    )


def _unavailable(
    *,
    display_value: float,
    filters: dict[str, Any],
    candidates: pd.DataFrame,
    reason: str,
    sections: list[tuple[str, pd.DataFrame]],
):
    return (
        STATUS_UNAVAILABLE,
        0.0,
        -display_value,
        "governed_debt_position:unavailable",
        "monthly_debt_position.csv",
        {
            **filters,
            "availability_status": "unavailable",
            "reason": reason,
            "candidate_rows": int(len(candidates)),
            "valid_as_of_rows": 0,
        },
        "Debt position requires a valid as-of observation; lexical or undated fallback is prohibited by debt_position_specs_v1.",
        candidates,
        sections,
    )


def _strict_candidates(
    debt_position: pd.DataFrame,
    *,
    spec: DebtPositionSpec,
    currency: str,
    debtor: str,
    creditor: str,
    period: str,
    annual: bool,
) -> pd.DataFrame:
    if debt_position.empty:
        return pd.DataFrame()

    required = {"period", "Currency", "debtor", "creditor", "component"}
    if not required.issubset(debt_position.columns):
        return pd.DataFrame()

    period_mask = (
        _legacy._year_mask(debt_position, period)
        if annual
        else _legacy._period_eq(debt_position, period)
    )
    mask = (
        period_mask
        & _legacy._source_filter_eq(debt_position, "Currency", currency)
        & _legacy._source_filter_eq(debt_position, "debtor", debtor)
        & _legacy._source_filter_eq(debt_position, "creditor", creditor)
        & _legacy._source_filter_eq(debt_position, "component", spec.component)
    )
    return debt_position.loc[mask].copy()


def _latest_period_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if rows.empty or "period" not in rows.columns:
        return rows.iloc[0:0].copy(), ""
    periods = sorted(
        value
        for value in rows["period"].dropna().astype(str).unique().tolist()
        if value
    )
    if not periods:
        return rows.iloc[0:0].copy(), ""
    selected_period = periods[-1]
    return rows.loc[rows["period"].astype(str).eq(selected_period)].copy(), selected_period


def _latest_valid_as_of(
    rows: pd.DataFrame,
    spec: DebtPositionSpec,
) -> tuple[pd.DataFrame, int]:
    if rows.empty or spec.as_of_field not in rows.columns:
        return rows.iloc[0:0].copy(), 0

    work = rows.copy()
    work["__governed_as_of"] = pd.to_datetime(
        work[spec.as_of_field], errors="coerce"
    )
    valid = work.loc[work["__governed_as_of"].notna()].copy()
    if valid.empty:
        return rows.iloc[0:0].copy(), 0

    selected = (
        valid.sort_values(["__governed_as_of"], kind="stable")
        .tail(1)
        .drop(columns=["__governed_as_of"])
        .copy()
    )
    return selected, int(len(valid))


def execute_monthly_debt_position(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_position: pd.DataFrame,
    tolerance: float,
):
    """Execute a governed monthly debt-position cell or return ``None``.

    ``None`` means the row has no recognized DebtPositionSpec and should retain
    its legacy compatibility path.
    """

    spec = _resolve_row_spec(row)
    if spec is None:
        return None

    currency = _legacy._norm(row.get("Currency"))
    pair, debtor, creditor = _pair_identity(row)
    filters = _contract_filters(
        spec=spec,
        period=period,
        currency=currency,
        pair=pair,
        debtor=debtor,
        creditor=creditor,
        annual=False,
    )

    if debt_position.empty:
        return (
            _legacy.STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_debt_position.csv",
            {**filters, "error": "missing monthly_debt_position.csv"},
            "Debt position drilldown requires monthly_debt_position.csv.",
            pd.DataFrame(),
            [],
        )

    missing_identity = [
        name
        for name, value in (("Currency", currency), ("debtor", debtor), ("creditor", creditor))
        if not value
    ]
    if missing_identity:
        return _unsupported(
            display_value=display_value,
            source="monthly_debt_position.csv",
            filters=filters,
            reason=f"missing governed debt-position identity: {missing_identity}",
            caveat="Governed debt-position execution requires explicit native-currency debtor/creditor identity.",
        )

    required_columns = {
        "period",
        "Currency",
        "debtor",
        "creditor",
        "component",
        spec.value_ref,
        spec.as_of_field,
    }
    missing_columns = sorted(required_columns.difference(debt_position.columns))
    if missing_columns:
        return _unsupported(
            display_value=display_value,
            source="monthly_debt_position.csv",
            filters=filters,
            reason=f"missing governed debt-position columns: {missing_columns}",
            caveat="Debt position source does not satisfy debt_position_specs_v1.",
        )

    candidates = _strict_candidates(
        debt_position,
        spec=spec,
        currency=currency,
        debtor=debtor,
        creditor=creditor,
        period=period,
        annual=False,
    )
    if candidates.empty:
        return (
            _legacy.STATUS_EMPTY,
            0.0,
            -display_value,
            "governed_debt_position:empty",
            "monthly_debt_position.csv",
            {**filters, "candidate_rows": 0, "valid_as_of_rows": 0},
            "No debt-position snapshot matches the governed identity.",
            candidates,
            [("All candidate snapshots in period", candidates)],
        )

    selected, valid_count = _latest_valid_as_of(candidates, spec)
    sections = [
        ("Selected governed monthly close snapshot", selected),
        ("All candidate snapshots in period", candidates),
    ]
    if selected.empty:
        return _unavailable(
            display_value=display_value,
            filters=filters,
            candidates=candidates,
            reason="no valid as_of_date in selected monthly debt-position candidates",
            sections=sections,
        )

    matched = _legacy._num(selected.iloc[0].get(spec.value_ref))
    residual = matched - display_value
    status = (
        _legacy.STATUS_OK
        if abs(residual) <= tolerance
        else _legacy.STATUS_RESIDUAL_WARNING
    )
    filters = {
        **filters,
        "candidate_rows": int(len(candidates)),
        "valid_as_of_rows": valid_count,
        "selected_as_of_date": _legacy._norm(selected.iloc[0].get(spec.as_of_field)),
    }
    return (
        status,
        matched,
        residual,
        "governed_debt_position:monthly",
        "monthly_debt_position.csv",
        filters,
        "Debt position is governed snapshot lineage; the selected value comes from DebtPositionSpec and the latest valid as-of observation.",
        selected,
        sections,
    )


def execute_annual_debt_position(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_position: pd.DataFrame,
    tolerance: float,
):
    """Execute annual debt stock as latest period then latest valid as-of."""

    spec = _resolve_row_spec(row)
    if spec is None:
        return None

    currency = _legacy._norm(row.get("Currency"))
    pair, debtor, creditor = _pair_identity(row)
    base_filters = _contract_filters(
        spec=spec,
        period=period,
        currency=currency,
        pair=pair,
        debtor=debtor,
        creditor=creditor,
        annual=True,
    )

    if debt_position.empty:
        return (
            _legacy.STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_debt_position.csv",
            {**base_filters, "error": "missing monthly_debt_position.csv"},
            "Annual debt stock drilldown requires monthly_debt_position.csv.",
            pd.DataFrame(),
            [],
        )

    missing_identity = [
        name
        for name, value in (("Currency", currency), ("debtor", debtor), ("creditor", creditor))
        if not value
    ]
    if missing_identity:
        return _unsupported(
            display_value=display_value,
            source="monthly_debt_position.csv",
            filters=base_filters,
            reason=f"missing governed debt-position identity: {missing_identity}",
            caveat="Governed annual debt stock requires explicit native-currency debtor/creditor identity.",
        )

    required_columns = {
        "period",
        "Currency",
        "debtor",
        "creditor",
        "component",
        spec.value_ref,
        spec.as_of_field,
    }
    missing_columns = sorted(required_columns.difference(debt_position.columns))
    if missing_columns:
        return _unsupported(
            display_value=display_value,
            source="monthly_debt_position.csv",
            filters=base_filters,
            reason=f"missing governed debt-position columns: {missing_columns}",
            caveat="Debt position source does not satisfy debt_position_specs_v1.",
        )

    year_candidates = _strict_candidates(
        debt_position,
        spec=spec,
        currency=currency,
        debtor=debtor,
        creditor=creditor,
        period=period,
        annual=True,
    )
    if year_candidates.empty:
        return (
            _legacy.STATUS_EMPTY,
            0.0,
            -display_value,
            "governed_debt_position:empty",
            "monthly_debt_position.csv",
            {**base_filters, "candidate_rows": 0, "valid_as_of_rows": 0},
            "No debt-position snapshot matches the governed annual identity.",
            year_candidates,
            [("Candidate debt position rows in year", year_candidates)],
        )

    month_candidates, selected_period = _latest_period_rows(year_candidates)
    filters = {**base_filters, "selected_period": selected_period}
    selected, valid_count = _latest_valid_as_of(month_candidates, spec)
    sections = [
        ("Annual companion row", _legacy._annual_companion_long_row(row, period, display_value)),
        ("Selected governed annual close snapshot", selected),
        ("Candidates in selected closing period", month_candidates),
        ("Candidate debt position rows in year", year_candidates),
    ]
    if selected.empty:
        return _unavailable(
            display_value=display_value,
            filters=filters,
            candidates=month_candidates,
            reason="latest debt-position period has no valid as_of_date; prior periods are not substituted",
            sections=sections,
        )

    matched = _legacy._num(selected.iloc[0].get(spec.value_ref))
    residual = matched - display_value
    status = (
        _legacy.STATUS_OK
        if abs(residual) <= tolerance
        else _legacy.STATUS_RESIDUAL_WARNING
    )
    filters = {
        **filters,
        "year_candidate_rows": int(len(year_candidates)),
        "selected_period_candidate_rows": int(len(month_candidates)),
        "valid_as_of_rows": valid_count,
        "selected_as_of_date": _legacy._norm(selected.iloc[0].get(spec.as_of_field)),
    }
    return (
        status,
        matched,
        residual,
        "governed_debt_position:annual",
        "monthly_debt_position.csv",
        filters,
        "Debt stock is governed snapshot lineage: latest period in year, then latest valid as-of within that period; monthly stocks are never summed.",
        selected,
        sections,
    )
