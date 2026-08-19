from __future__ import annotations

"""Professional drilldown adapter for governed validated cash positions."""

from typing import Any

import pandas as pd

from accounting.cash_authority import (
    select_validated_cash_period,
    select_validated_cash_year,
    validated_cash_schema_supported,
)


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _result(
    *,
    selection,
    display_value: float,
    tolerance: float,
    annual: bool,
):
    matched = float(selection.value) if selection.value is not None else 0.0
    residual = matched - display_value
    if selection.available:
        status = "ok" if abs(residual) <= tolerance else "residual_warning"
    elif selection.status == "unsupported":
        return None
    else:
        status = "unsupported"
    filters = {
        "cash_position_spec": "cash.position.validated",
        "executor": "governed_validated_cash_v1",
        "period": selection.period,
        "Currency": selection.currency,
        "Box": selection.box,
        "selection_status": selection.status,
        "selection_reason": selection.reason,
        "selection": "latest valid as_of_date per Box/account_id; sum selected accounts",
        "annualization": (
            "latest period with validated candidates, then same account snapshot primitive"
            if annual
            else "monthly account snapshot primitive"
        ),
        "fallback_to_inferred": "never",
    }
    sections = [
        ("Selected validated account snapshots", selection.selected),
        ("Validated cash candidates", selection.candidates),
        ("Excluded inferred control rows", selection.excluded_inferred),
        ("Excluded internal balance rows", selection.excluded_internal),
    ]
    if not selection.excluded_other.empty:
        sections.append(("Other excluded cash rows", selection.excluded_other))
    caveat = (
        "Cash headline uses explicitly validated account snapshots only. "
        "Inferred box control and internal party balances are excluded and never used as fallback."
    )
    return (
        status,
        matched,
        residual,
        "governed_validated_cash",
        "monthly_cash_close.csv",
        filters,
        caveat,
        selection.selected,
        sections,
    )


def execute_monthly_cash_position(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    cash_close: pd.DataFrame,
    tolerance: float,
):
    metric = _norm(row.get("metric")) or _norm(row.get("measure")) or "cash_close"
    if metric != "cash_close":
        return None
    if cash_close is not None and not cash_close.empty and not validated_cash_schema_supported(cash_close):
        return None
    selection = select_validated_cash_period(
        cash_close,
        period=period,
        currency=_norm(row.get("Currency")),
        box=_norm(row.get("Box")),
    )
    return _result(selection=selection, display_value=display_value, tolerance=tolerance, annual=False)


def execute_annual_cash_position(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    cash_close: pd.DataFrame,
    tolerance: float,
):
    if cash_close is not None and not cash_close.empty and not validated_cash_schema_supported(cash_close):
        return None
    selection = select_validated_cash_year(
        cash_close,
        year=period,
        currency=_norm(row.get("Currency")),
        box=_norm(row.get("Box")),
    )
    return _result(selection=selection, display_value=display_value, tolerance=tolerance, annual=True)
