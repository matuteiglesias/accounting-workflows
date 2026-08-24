from __future__ import annotations

"""Shared reporting projection over the governed validated-cash selector.

This module does not define cash eligibility or valuation rules. Those remain
owned by :mod:`accounting.cash_authority`. It only enumerates source-backed
reporting scopes and applies the existing governed selector once per scope so
annual metrics, frontend series, and professional companion tables consume the
same selected population.
"""

from dataclasses import dataclass
from typing import Iterator, Literal

import pandas as pd

from accounting.cash_authority import (
    CashSelection,
    select_validated_cash_period,
    select_validated_cash_year,
    validated_cash_schema_supported,
)


CashFrequency = Literal["monthly", "annual"]
CashScope = Literal["currency", "box"]


@dataclass(frozen=True)
class ValidatedCashProjection:
    """One source-backed reporting scope and its governed selection result."""

    frequency: CashFrequency
    reporting_period: str
    scope: CashScope
    currency: str
    box: str
    selection: CashSelection

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def value(self) -> float | None:
        return self.selection.value

    @property
    def selected_period(self) -> str:
        return self.selection.period


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _source_scopes(cash: pd.DataFrame) -> pd.DataFrame:
    """Return normalized valid period/currency/Box source scopes.

    Scope discovery is intentionally mechanical. It does not inspect cash
    suitability, validation status, or position type; those semantics belong to
    ``cash_authority``. A source-backed scope can therefore project to an
    unavailable governed selection, which is important evidence for fail-closed
    reporting.
    """

    if cash is None or cash.empty or not validated_cash_schema_supported(cash):
        return pd.DataFrame(columns=["period", "year", "Currency", "Box"])

    scopes = pd.DataFrame(
        {
            "period": _text(cash["period"]),
            "Currency": _text(cash["Currency"]),
            "Box": _text(cash["Box"]),
        }
    )
    valid_period = scopes["period"].str.match(r"^20\d{2}-(0[1-9]|1[0-2])$")
    scopes = scopes.loc[valid_period & scopes["Currency"].ne("")].copy()
    scopes["year"] = scopes["period"].str.slice(0, 4)
    return scopes[["period", "year", "Currency", "Box"]]


def iter_validated_monthly_cash_positions(
    cash: pd.DataFrame,
) -> Iterator[ValidatedCashProjection]:
    """Yield governed monthly cash selections for actual source scopes.

    For every source-backed ``period/Currency`` pair, yield the total-currency
    selection followed by each source-backed ``period/Currency/Box`` selection.
    No Cartesian period/currency/Box combinations are invented by report code.
    """

    scopes = _source_scopes(cash)
    if scopes.empty:
        return

    currency_scopes = (
        scopes[["period", "Currency"]]
        .drop_duplicates()
        .sort_values(["period", "Currency"], kind="stable")
    )
    for row in currency_scopes.itertuples(index=False):
        period, currency = str(row.period), str(row.Currency)
        yield ValidatedCashProjection(
            frequency="monthly",
            reporting_period=period,
            scope="currency",
            currency=currency,
            box="",
            selection=select_validated_cash_period(
                cash,
                period=period,
                currency=currency,
                box="",
            ),
        )

        box_scopes = (
            scopes.loc[
                scopes["period"].eq(period)
                & scopes["Currency"].eq(currency)
                & scopes["Box"].ne(""),
                ["Box"],
            ]
            .drop_duplicates()
            .sort_values(["Box"], kind="stable")
        )
        for box_row in box_scopes.itertuples(index=False):
            box = str(box_row.Box)
            yield ValidatedCashProjection(
                frequency="monthly",
                reporting_period=period,
                scope="box",
                currency=currency,
                box=box,
                selection=select_validated_cash_period(
                    cash,
                    period=period,
                    currency=currency,
                    box=box,
                ),
            )


def iter_validated_annual_cash_positions(
    cash: pd.DataFrame,
) -> Iterator[ValidatedCashProjection]:
    """Yield governed annual closing-cash selections for actual source scopes.

    Each ``year/Currency`` and ``year/Currency/Box`` scope is discovered from
    the monthly cash source and delegated to ``select_validated_cash_year``.
    The selector still owns the annual stock rule: choose the latest period with
    validated candidates and fail closed if that selected position is invalid.
    """

    scopes = _source_scopes(cash)
    if scopes.empty:
        return

    currency_scopes = (
        scopes[["year", "Currency"]]
        .drop_duplicates()
        .sort_values(["year", "Currency"], kind="stable")
    )
    for row in currency_scopes.itertuples(index=False):
        year, currency = str(row.year), str(row.Currency)
        yield ValidatedCashProjection(
            frequency="annual",
            reporting_period=year,
            scope="currency",
            currency=currency,
            box="",
            selection=select_validated_cash_year(
                cash,
                year=year,
                currency=currency,
                box="",
            ),
        )

        box_scopes = (
            scopes.loc[
                scopes["year"].eq(year)
                & scopes["Currency"].eq(currency)
                & scopes["Box"].ne(""),
                ["Box"],
            ]
            .drop_duplicates()
            .sort_values(["Box"], kind="stable")
        )
        for box_row in box_scopes.itertuples(index=False):
            box = str(box_row.Box)
            yield ValidatedCashProjection(
                frequency="annual",
                reporting_period=year,
                scope="box",
                currency=currency,
                box=box,
                selection=select_validated_cash_year(
                    cash,
                    year=year,
                    currency=currency,
                    box=box,
                ),
            )
