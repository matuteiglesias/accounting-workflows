from __future__ import annotations

"""Annual professional companion-table facade with governed cash projection.

All non-cash builders are preserved in ``annual_dashboard_tables_legacy``.
The cash companion consumes the same source-backed governed annual cash
projection as the annual metrics facade.
"""

from typing import Sequence

import pandas as pd

from accounting.cash_authority import validated_cash_schema_supported
from accounting.cash_projection import iter_validated_annual_cash_positions
from accounting.professional import annual_dashboard_tables_legacy as _legacy


# Explicit compatibility surface derived from repository caller census.
# Do not broaden this list: every retained legacy symbol must have a caller
# or an independently documented compatibility contract/removal condition.
LEGACY_COMPAT_EXPORTS = (
    "build_annual_debt_activity_by_pair",
    "build_annual_debt_stock_by_pair",
    "build_annual_funding_by_actor_channel",
    "write_annual_long_and_wide",
)

build_annual_debt_activity_by_pair = _legacy.build_annual_debt_activity_by_pair
build_annual_debt_stock_by_pair = _legacy.build_annual_debt_stock_by_pair
build_annual_funding_by_actor_channel = _legacy.build_annual_funding_by_actor_channel
write_annual_long_and_wide = _legacy.write_annual_long_and_wide


def build_annual_cash_close_by_box(
    cash: pd.DataFrame,
    year_columns: Sequence[str] = _legacy.DEFAULT_YEAR_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build validated annual cash close by Box/Currency.

    Modern ``monthly_cash_close`` inputs consume the shared annual projection:
    source-backed year/Currency/Box scope, latest period containing validated
    candidates in the year, then latest valid as-of per account, then sum
    selected account closes. Inferred box control and internal balances are
    never additive and never fallback cash.

    Historical/non-modern table shapes retain the compatibility path.
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

    rows: list[dict[str, object]] = []
    for projection in iter_validated_annual_cash_positions(cash):
        if projection.scope != "box" or not projection.available:
            continue
        selection = projection.selection
        rows.append(
            {
                "metric_id": "CASH.CLOSE.BY_BOX",
                "line_id": (
                    f"CASH.CLOSE.BY_BOX.{projection.box}.{projection.currency}"
                ),
                "period": projection.reporting_period,
                "Box": projection.box,
                "Currency": projection.currency,
                "value": float(selection.value),
                "selected_month": projection.selected_period,
                "source_table": "monthly_cash_close.csv",
                "source_filter": (
                    "cash.position.validated; source-backed year/Currency/Box scope; "
                    "latest valid as_of_date per Box/account_id; inferred/internal excluded"
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
