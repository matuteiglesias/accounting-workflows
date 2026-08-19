from __future__ import annotations

"""Governed professional execution for resolved debt-activity flows.

This module consumes ``DebtActivitySpec`` only. Debt position remains in the
snapshot executor introduced by PR13. The debt resolver/mart remain unchanged.
"""

from typing import Any

import pandas as pd

from accounting.contracts.debt_position_activity import (
    DEBT_ACTIVITY_SPECS_VERSION,
    DebtActivitySpec,
    resolve_debt_activity_spec,
)
from accounting.professional import drilldown_legacy as _legacy


_VIEW_TOKEN_TO_SPEC_ID = {
    "new_claim": "debt.activity.new_claim",
    "new_principal": "debt.activity.new_claim",
    "interest_accrual": "debt.activity.interest_accrual",
    "interest_accrued": "debt.activity.interest_accrual",
    "repayment": "debt.activity.repayment",
    "repayments": "debt.activity.repayment",
    "adjustment": "debt.activity.adjustment",
    "adjustments": "debt.activity.adjustment",
    "net_change": "debt.activity.net_change",
}


def _resolve_row_spec(row: pd.Series) -> DebtActivitySpec | None:
    """Resolve a characterized activity view to one governed activity spec.

    Monthly tables expose the physical measure name while annual companion
    tables expose a view/activity token. Both resolve to the same atomic
    DebtActivitySpec. ``settlements`` deliberately remains outside v1.
    """

    measure = _legacy._norm(row.get("measure"))
    activity_type = _legacy._norm(row.get("activity_type"))
    token = measure or activity_type
    spec_id = _VIEW_TOKEN_TO_SPEC_ID.get(token)
    return resolve_debt_activity_spec(spec_id) if spec_id else None


def _pair_identity(row: pd.Series) -> tuple[str, str, str]:
    pair = _legacy._norm(row.get("pair"))
    pair_debtor, pair_creditor = _legacy._pair_parts(pair)
    debtor = _legacy._norm(row.get("debtor")) or pair_debtor
    creditor = _legacy._norm(row.get("creditor")) or pair_creditor
    return pair, debtor, creditor


def _contract_filters(
    *,
    spec: DebtActivitySpec,
    period: str,
    currency: str,
    pair: str,
    debtor: str,
    creditor: str,
    annual: bool,
) -> dict[str, Any]:
    return {
        "period": period,
        "Currency": currency,
        "pair": pair,
        "debtor": debtor,
        "creditor": creditor,
        "activity_type": spec.activity_type,
        "measure": spec.measure_ref,
        "spec_id": spec.spec_id,
        "contract_version": DEBT_ACTIVITY_SPECS_VERSION,
        "aggregation": spec.aggregation,
        "annualization": spec.annualization if annual else "n/a",
        "source": "monthly_debt_activity.csv",
        "executor": "governed_debt_activity_v1",
    }


def _unsupported(
    *,
    display_value: float,
    filters: dict[str, Any],
    reason: str,
    caveat: str,
):
    return (
        _legacy.STATUS_UNSUPPORTED,
        0.0,
        -display_value,
        "unsupported",
        "monthly_debt_activity.csv",
        {**filters, "unsupported": True, "reason": reason},
        caveat,
        pd.DataFrame(),
        [],
    )


def _strict_activity_rows(
    debt_activity: pd.DataFrame,
    *,
    spec: DebtActivitySpec,
    currency: str,
    debtor: str,
    creditor: str,
    period: str,
    annual: bool,
) -> pd.DataFrame:
    period_mask = (
        _legacy._year_mask(debt_activity, period)
        if annual
        else _legacy._period_eq(debt_activity, period)
    )
    mask = (
        period_mask
        & _legacy._source_filter_eq(debt_activity, "Currency", currency)
        & _legacy._source_filter_eq(debt_activity, "debtor", debtor)
        & _legacy._source_filter_eq(debt_activity, "creditor", creditor)
        & _legacy._source_filter_eq(
            debt_activity, "activity_type", spec.activity_type
        )
    )
    return debt_activity.loc[mask].copy()


def _execute_debt_activity(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_activity: pd.DataFrame,
    tolerance: float,
    annual: bool,
):
    spec = _resolve_row_spec(row)
    if spec is None:
        return None

    # Historical sources without activity_type predate the typed grain. Keep
    # them on the compatibility helper rather than inventing activity identity
    # from a requested measure.
    if not debt_activity.empty and "activity_type" not in debt_activity.columns:
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
        annual=annual,
    )

    if debt_activity.empty:
        return (
            _legacy.STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_debt_activity.csv",
            {**filters, "error": "missing monthly_debt_activity.csv"},
            "Debt activity drilldown requires monthly_debt_activity.csv.",
            pd.DataFrame(),
            [],
        )

    missing_identity = [
        name
        for name, value in (
            ("Currency", currency),
            ("debtor", debtor),
            ("creditor", creditor),
        )
        if not value
    ]
    if missing_identity:
        return _unsupported(
            display_value=display_value,
            filters=filters,
            reason=f"missing governed debt-activity identity: {missing_identity}",
            caveat="Governed debt activity requires explicit native-currency debtor/creditor identity.",
        )

    required_columns = {
        "period",
        "Currency",
        "debtor",
        "creditor",
        "activity_type",
        spec.measure_ref,
    }
    missing_columns = sorted(required_columns.difference(debt_activity.columns))
    if missing_columns:
        return _unsupported(
            display_value=display_value,
            filters=filters,
            reason=f"missing governed debt-activity columns: {missing_columns}",
            caveat="Debt activity source does not satisfy debt_activity_specs_v1.",
        )

    source = _strict_activity_rows(
        debt_activity,
        spec=spec,
        currency=currency,
        debtor=debtor,
        creditor=creditor,
        period=period,
        annual=annual,
    )
    matched = _legacy._measure_sum(source, spec.measure_ref)
    residual = matched - display_value
    status = (
        _legacy.STATUS_EMPTY
        if source.empty
        else _legacy.STATUS_OK
        if abs(residual) <= tolerance
        else _legacy.STATUS_RESIDUAL_WARNING
    )
    filters = {
        **filters,
        "matched_activity_rows": int(len(source)),
        "calculation_rule": (
            "annual flow = sum governed activity rows across months"
            if annual
            else "monthly flow = sum governed activity rows in period"
        ),
    }
    sections = (
        [
            (
                "Annual companion row",
                _legacy._annual_companion_long_row(row, period, display_value),
            ),
            ("Matched monthly_debt_activity rows", source),
        ]
        if annual
        else [("Debt activity rows", source)]
    )
    return (
        status,
        matched,
        residual,
        "governed_debt_activity:annual"
        if annual
        else "governed_debt_activity:monthly",
        "monthly_debt_activity.csv",
        filters,
        "Debt activity is governed resolved-debt flow evidence; values are summed, never selected as snapshots.",
        source,
        sections,
    )


def execute_monthly_debt_activity(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_activity: pd.DataFrame,
    tolerance: float,
):
    """Execute one governed monthly debt-activity flow or return ``None``."""

    return _execute_debt_activity(
        row=row,
        period=period,
        display_value=display_value,
        debt_activity=debt_activity,
        tolerance=tolerance,
        annual=False,
    )


def execute_annual_debt_activity(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_activity: pd.DataFrame,
    tolerance: float,
):
    """Execute annual debt activity as a sum of governed monthly activity."""

    return _execute_debt_activity(
        row=row,
        period=period,
        display_value=display_value,
        debt_activity=debt_activity,
        tolerance=tolerance,
        annual=True,
    )
