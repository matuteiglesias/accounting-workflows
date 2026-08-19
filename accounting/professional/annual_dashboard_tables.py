from __future__ import annotations

"""Annual professional companion-table facade with governed cash authority.

All non-cash builders are preserved in ``annual_dashboard_tables_legacy``.
The cash companion alone is migrated to the PR15A validated-cash contract.
"""

from typing import Sequence

import pandas as pd

from accounting.cash_authority import (
    select_validated_cash_year,
    validated_cash_schema_supported,
)
from accounting.professional import annual_dashboard_tables_legacy as _legacy


for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


def build_annual_cash_close_by_box(
    cash: pd.DataFrame,
    year_columns: Sequence[str] = _legacy.DEFAULT_YEAR_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build validated annual cash close by Box/Currency.

    Modern ``monthly_cash_close`` inputs use the shared governed selector:
    latest period containing validated candidates in the year, then latest
    valid as-of per account, then sum selected account closes. Inferred box
    control and internal balances are never additive and never fallback cash.

    Historical/non-modern table shapes retain the pre-15B compatibility path.
    """

    if cash is None or cash.empty or not validated_cash_schema_supported(cash):
        return _legacy.build_annual_cash_close_by_box(cash, year_columns)

    long_cols = [
        "metric_id",
        "line_id",
        "period",
        "Box",
        "Currency",
        "value",
        "selected_month",
        "source_table",
        "source_filter",
        "calculation_rule",
    ]
    empty_wide_cols = [
        "metric_id",
        "line_id",
        "Box",
        "Currency",
        "source_table",
        "source_filter",
        "calculation_rule",
        *year_columns,
    ]

    period_text = cash["period"].fillna("").astype(str).str.strip()
    years = sorted(period_text.str.extract(r"^(20\d{2})-", expand=False).dropna().unique())
    currencies = sorted(cash["Currency"].fillna("").astype(str).str.strip().loc[lambda s: s.ne("")].unique())
    boxes = sorted(cash["Box"].fillna("").astype(str).str.strip().loc[lambda s: s.ne("")].unique())

    rows: list[dict[str, object]] = []
    for year in years:
        for currency in currencies:
            for box in boxes:
                selection = select_validated_cash_year(
                    cash,
                    year=str(year),
                    currency=str(currency),
                    box=str(box),
                )
                if not selection.available:
                    continue
                rows.append(
                    {
                        "metric_id": "CASH.CLOSE.BY_BOX",
                        "line_id": f"CASH.CLOSE.BY_BOX.{box}.{currency}",
                        "period": str(year),
                        "Box": str(box),
                        "Currency": str(currency),
                        "value": float(selection.value),
                        "selected_month": selection.period,
                        "source_table": "monthly_cash_close.csv",
                        "source_filter": (
                            "cash.position.validated; latest valid as_of_date per "
                            "Box/account_id; inferred/internal excluded"
                        ),
                        "calculation_rule": (
                            "annual stock = last governed validated cash period in year; "
                            "same account snapshot selector as monthly; sum selected "
                            "accounts; never sum monthly positions"
                        ),
                    }
                )

    if not rows:
        return _legacy._empty(long_cols), _legacy._empty(empty_wide_cols)

    long_df = pd.DataFrame(rows, columns=long_cols).sort_values(
        ["period", "Currency", "Box"]
    ).reset_index(drop=True)
    wide_df = _legacy._annual_wide(
        long_df,
        [
            "metric_id",
            "line_id",
            "Box",
            "Currency",
            "source_table",
            "source_filter",
            "calculation_rule",
        ],
        year_columns,
    )
    return long_df, wide_df
