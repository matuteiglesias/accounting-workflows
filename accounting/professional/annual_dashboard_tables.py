from __future__ import annotations

"""Annual professional companion-table facade with governed authorities."""

from typing import Sequence

import pandas as pd

from accounting.contracts.funding_support import (
    FUNDING_SUPPORT_SPECS_VERSION,
    classify_funding_support,
)
from accounting.professional import annual_dashboard_tables_legacy as _legacy



_MODERN_FUNDING_COLUMNS = {
    "period",
    "Currency",
    "semantic_bucket",
    "semantic_subbucket",
    "funding_channel",
    "cash_effect",
    "debt_effect",
}


def build_annual_funding_by_actor_channel(
    flow: pd.DataFrame,
    year_columns: Sequence[str] = _legacy.DEFAULT_YEAR_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual support views from governed support membership.

    ``FUND.CONTRIB.TOTAL`` is not produced here and remains the narrower core
    contribution metric. This companion table intentionally represents the
    broader support surface, but only when explicit modern semantic dimensions
    are present. Missing modern columns use the historical compatibility path;
    modern rows never fall back to label/blob inference.
    """

    dims = [
        "Currency",
        "funding_actor",
        "funding_channel",
        "cash_effect",
        "target_box",
        "beneficiary_box",
        "obligation_box",
    ]
    long_cols = [
        "metric_id",
        "line_id",
        "period",
        *dims,
        "value",
        "source_table",
        "source_filter",
        "calculation_rule",
    ]
    empty_wide = _legacy._empty(
        [
            "metric_id",
            "line_id",
            *dims,
            "source_table",
            "source_filter",
            "calculation_rule",
            *year_columns,
        ]
    )
    if flow is None or flow.empty:
        return _legacy._empty(long_cols), empty_wide
    if not _MODERN_FUNDING_COLUMNS.issubset(flow.columns):
        return _legacy.build_annual_funding_by_actor_channel(flow, year_columns)

    members = classify_funding_support(flow, strict=True)
    if members.empty:
        return _legacy._empty(long_cols), empty_wide

    work = _legacy._ensure_columns(
        members,
        {
            "funding_actor": "",
            "funding_channel": "",
            "cash_effect": "",
            "target_box": "",
            "beneficiary_box": "",
            "obligation_box": "",
        },
    )
    work = _legacy._add_year(work)
    if work.empty:
        return _legacy._empty(long_cols), empty_wide
    for column in dims:
        work[column] = _legacy._clean_text(work[column])
    work["support_amount"] = pd.to_numeric(
        work["support_amount"], errors="coerce"
    ).fillna(0.0)

    specs = [
        ("FUND.CONTRIB.BY_FUNDING_ACTOR", pd.Series(True, index=work.index), "funding_actor"),
        ("FUND.CONTRIB.BY_CHANNEL", pd.Series(True, index=work.index), "funding_channel"),
        ("FUND.CONTRIB.BY_CASH_EFFECT", pd.Series(True, index=work.index), "cash_effect"),
        ("FUND.CONTRIB.BY_TARGET_BOX", pd.Series(True, index=work.index), "target_box"),
        (
            "FUND.CONTRIB.DIRECT_OBLIGATION",
            work["support_kind"].eq("direct_obligation_payment"),
            "funding_channel",
        ),
        (
            "FUND.CONTRIB.CASH_TO_BOX",
            work["cash_effect"].eq("cash_in_box"),
            "funding_channel",
        ),
        (
            "FUND.CONTRIB.DEBT_LINKED",
            work["support_kind"].eq("debt_linked_support"),
            "funding_channel",
        ),
    ]
    parts: list[pd.DataFrame] = []
    for metric_id, mask, primary_dim in specs:
        sub = work.loc[mask].copy()
        if sub.empty:
            continue
        grouped = (
            sub.groupby(["period", *dims], dropna=False)["support_amount"]
            .sum()
            .reset_index(name="value")
        )
        grouped["metric_id"] = metric_id
        grouped["line_id"] = (
            metric_id
            + "."
            + grouped[primary_dim].astype(str).replace("", "unknown")
            + "."
            + grouped["funding_actor"].astype(str).replace("", "unknown")
            + "."
            + grouped["funding_channel"].astype(str).replace("", "unknown")
            + "."
            + grouped["cash_effect"].astype(str).replace("", "unknown")
            + "."
            + grouped["Currency"].astype(str)
        )
        grouped["source_table"] = "monthly_flow_semantic_split.csv"
        grouped["source_filter"] = (
            f"{FUNDING_SUPPORT_SPECS_VERSION}; metric_id={metric_id}; "
            f"primary_dimension={primary_dim}; explicit semantic support membership"
        )
        grouped["calculation_rule"] = (
            "annual support flow = sum governed semantic measure by "
            "year/Currency/funding dimensions; support membership from "
            f"{FUNDING_SUPPORT_SPECS_VERSION}; ARS/USD never mixed"
        )
        parts.append(grouped)

    if not parts:
        return _legacy._empty(long_cols), empty_wide
    long_df = pd.concat(parts, ignore_index=True)[long_cols]
    long_df = long_df.sort_values(
        ["metric_id", "period", "Currency", "funding_actor", "funding_channel", "cash_effect"]
    ).reset_index(drop=True)
    wide_df = _legacy._annual_wide(
        long_df,
        [
            "metric_id",
            "line_id",
            *dims,
            "source_table",
            "source_filter",
            "calculation_rule",
        ],
        year_columns,
    )
    return long_df, wide_df
