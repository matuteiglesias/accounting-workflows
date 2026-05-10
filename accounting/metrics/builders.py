from __future__ import annotations

from typing import Iterable

import pandas as pd

from .io import MetricsContext, build_metric_frame, concat_metric_frames


def _require_df(ctx: MetricsContext, attr_name: str) -> pd.DataFrame:
    df = getattr(ctx, attr_name)
    if df is None:
        raise ValueError(f"Context is missing required dataframe: {attr_name}")
    return df.copy()


def _periodize_monthly(df: pd.DataFrame, time_col: str = "TimePeriod") -> pd.DataFrame:
    out = df.copy()
    p = pd.PeriodIndex(out[time_col].astype(str), freq="M")
    out["period_q"] = p.year.astype(str) + "Q" + pd.Series(p.quarter, index=out.index).astype(str)
    out["period_y"] = p.year.astype(str)
    return out


def _aggregate_flow_metric(
    df: pd.DataFrame,
    *,
    metric_id: str,
    value_col: str,
    currency_col: str = "Currency",
    run_id: str = "",
    as_of_date: str = "",
    source_layer: str = "",
) -> pd.DataFrame:
    if df.empty:
        return concat_metric_frames([])

    work = _periodize_monthly(df)

    frames: list[pd.DataFrame] = []

    q = (
        work.groupby(["period_q", currency_col], dropna=False)[value_col]
        .sum()
        .reset_index()
    )
    for _, row in q.iterrows():
        frames.append(
            build_metric_frame(
                metric_id=metric_id,
                period_grain="Q",
                period=str(row["period_q"]),
                currency=str(row[currency_col]),
                value=float(row[value_col]),
                run_id=run_id,
                as_of_date=as_of_date,
                source_layer=source_layer,
            )
        )

    y = (
        work.groupby(["period_y", currency_col], dropna=False)[value_col]
        .sum()
        .reset_index()
    )
    for _, row in y.iterrows():
        frames.append(
            build_metric_frame(
                metric_id=metric_id,
                period_grain="Y",
                period=str(row["period_y"]),
                currency=str(row[currency_col]),
                value=float(row[value_col]),
                run_id=run_id,
                as_of_date=as_of_date,
                source_layer=source_layer,
            )
        )

    return concat_metric_frames(frames)


def _aggregate_stock_metric_from_daily(
    df: pd.DataFrame,
    *,
    metric_id: str,
    value_col: str = "balance",
    date_col: str = "Date",
    currency_col: str = "Currency",
    run_id: str = "",
    as_of_date: str = "",
    source_layer: str = "",
) -> pd.DataFrame:
    if df.empty:
        return concat_metric_frames([])

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    work["period_q"] = work[date_col].dt.year.astype(str) + "Q" + work[date_col].dt.quarter.astype(str)
    work["period_y"] = work[date_col].dt.year.astype(str)

    frames: list[pd.DataFrame] = []

    q_last = (
        work.sort_values(date_col)
        .groupby(["period_q", currency_col], dropna=False, as_index=False)
        .tail(1)
    )
    for _, row in q_last.iterrows():
        frames.append(
            build_metric_frame(
                metric_id=metric_id,
                period_grain="Q",
                period=str(row["period_q"]),
                currency=str(row[currency_col]),
                value=float(row[value_col]),
                run_id=run_id,
                as_of_date=as_of_date,
                source_layer=source_layer,
            )
        )

    y_last = (
        work.sort_values(date_col)
        .groupby(["period_y", currency_col], dropna=False, as_index=False)
        .tail(1)
    )
    for _, row in y_last.iterrows():
        frames.append(
            build_metric_frame(
                metric_id=metric_id,
                period_grain="Y",
                period=str(row["period_y"]),
                currency=str(row[currency_col]),
                value=float(row[value_col]),
                run_id=run_id,
                as_of_date=as_of_date,
                source_layer=source_layer,
            )
        )

    return concat_metric_frames(frames)


def _normalize_debt_balance_periodic(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    required = ["period", "currency", "debtor", "creditor", "open_principal", "open_interest", "open_total"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Debt balance view is missing required columns: {missing}")

    for col in ["open_principal", "open_interest", "open_total"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    dedup_cols = ["period", "currency", "debtor", "creditor"]
    # Canonical debt artifacts can carry one row per item_type with repeated totals.
    # Keep a single exposure row per counterparty pair / period.
    work = (
        work.groupby(dedup_cols, dropna=False)[["open_principal", "open_interest", "open_total"]]
        .max()
        .reset_index()
    )
    return work


def _aggregate_debt_metric_from_periodic(
    ctx: MetricsContext,
    *,
    metric_id: str,
    period_grain: str,
    debt_filter,
    value_col: str,
    source_layer: str = "debt_balance",
) -> pd.DataFrame:
    attr = {
        "Q": "debt_balance_quarterly",
        "Y": "debt_balance_yearly",
    }.get(period_grain)
    if not attr:
        raise ValueError(f"Unsupported debt period_grain: {period_grain}")

    df = getattr(ctx, attr)
    if df is None or df.empty:
        return concat_metric_frames([])

    work = _normalize_debt_balance_periodic(df)
    work = work.loc[debt_filter(work)].copy()
    if work.empty:
        return concat_metric_frames([])

    grouped = (
        work.groupby(["period", "currency"], dropna=False)[value_col]
        .sum()
        .reset_index()
    )
    frames: list[pd.DataFrame] = []
    for _, row in grouped.iterrows():
        frames.append(
            build_metric_frame(
                metric_id=metric_id,
                period_grain=period_grain,
                period=str(row["period"]),
                currency=str(row["currency"]),
                value=float(row[value_col]),
                run_id=ctx.run_id,
                as_of_date=ctx.as_of_date,
                source_layer=source_layer,
            )
        )
    return concat_metric_frames(frames)


def _build_debt_metric_qy(
    ctx: MetricsContext,
    *,
    metric_id: str,
    debt_filter,
    value_col: str,
    source_layer: str = "debt_balance",
) -> pd.DataFrame:
    q = _aggregate_debt_metric_from_periodic(
        ctx,
        metric_id=metric_id,
        period_grain="Q",
        debt_filter=debt_filter,
        value_col=value_col,
        source_layer=source_layer,
    )
    y = _aggregate_debt_metric_from_periodic(
        ctx,
        metric_id=metric_id,
        period_grain="Y",
        debt_filter=debt_filter,
        value_col=value_col,
        source_layer=source_layer,
    )
    return concat_metric_frames([q, y])


def build_is_rent_total(ctx: MetricsContext) -> pd.DataFrame:
    df = _require_df(ctx, "per_flow")
    m = df.loc[
        (df["Flujo"].astype(str) == "Cobros") &
        (df["Tipo"].astype(str) == "Renta")
    ]
    return _aggregate_flow_metric(
        m,
        metric_id="IS.RENT.TOTAL",
        value_col="amount",
        run_id=ctx.run_id,
        as_of_date=ctx.as_of_date,
        source_layer="per_flow",
    )


def build_is_contrib_total(ctx: MetricsContext) -> pd.DataFrame:
    df = _require_df(ctx, "v_contributions_monthly")
    return _aggregate_flow_metric(
        df,
        metric_id="IS.CONTRIB.TOTAL",
        value_col="amount",
        run_id=ctx.run_id,
        as_of_date=ctx.as_of_date,
        source_layer="v_contributions_monthly",
    )


def build_is_opex_total(ctx: MetricsContext) -> pd.DataFrame:
    df = _require_df(ctx, "v_opex_category_monthly")
    value_col = "amount_out" if "amount_out" in df.columns else "amount"
    return _aggregate_flow_metric(
        df,
        metric_id="IS.OPEX.TOTAL",
        value_col=value_col,
        run_id=ctx.run_id,
        as_of_date=ctx.as_of_date,
        source_layer="v_opex_category_monthly",
    )


def build_is_draws_personal(ctx: MetricsContext) -> pd.DataFrame:
    df = _require_df(ctx, "ledger")
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    work["TimePeriod"] = work["Date"].dt.to_period("M").astype(str)

    text_cols = [c for c in ["Tipo", "Detalle", "notes", "tag", "Lugar"] if c in work.columns]
    if not text_cols:
        return concat_metric_frames([])

    mask = pd.Series(False, index=work.index)
    for c in text_cols:
        mask = mask | work[c].astype(str).str.contains(
            r"personal|retiro|draw|owner|dividend",
            case=False,
            na=False,
        )

    m = work.loc[mask]
    if m.empty:
        return concat_metric_frames([])

    return _aggregate_flow_metric(
        m,
        metric_id="IS.DRAWS.PERSONAL",
        value_col="amount",
        run_id=ctx.run_id,
        as_of_date=ctx.as_of_date,
        source_layer="ledger",
    )


def build_bs_cash_fb(ctx: MetricsContext) -> pd.DataFrame:
    df = _require_df(ctx, "daily_cash_position")
    m = df.loc[df["Box"].astype(str) == "Family Business"]
    return _aggregate_stock_metric_from_daily(
        m,
        metric_id="BS.CASH.FB",
        run_id=ctx.run_id,
        as_of_date=ctx.as_of_date,
        source_layer="daily_cash_position",
    )


def build_bs_cash_pm(ctx: MetricsContext) -> pd.DataFrame:
    df = _require_df(ctx, "daily_cash_position")
    m = df.loc[df["Box"].astype(str) == "Property Management"]
    return _aggregate_stock_metric_from_daily(
        m,
        metric_id="BS.CASH.PM",
        run_id=ctx.run_id,
        as_of_date=ctx.as_of_date,
        source_layer="daily_cash_position",
    )


def build_bs_debt_pm_to_mi_open(ctx: MetricsContext) -> pd.DataFrame:
    return _build_debt_metric_qy(
        ctx,
        metric_id="BS.DEBT.PM_TO_MI.OPEN",
        debt_filter=lambda df: (df["debtor"].astype(str) == "Property Management")
        & (df["creditor"].astype(str) == "MI"),
        value_col="open_total",
    )


def build_bs_debt_pm_to_primos_open(ctx: MetricsContext) -> pd.DataFrame:
    return _build_debt_metric_qy(
        ctx,
        metric_id="BS.DEBT.PM_TO_PRIMOS.OPEN",
        debt_filter=lambda df: (df["debtor"].astype(str) == "Property Management")
        & (df["creditor"].astype(str) == "Primos"),
        value_col="open_total",
    )


def build_bs_claim_ale_to_pm_open(ctx: MetricsContext) -> pd.DataFrame:
    return _build_debt_metric_qy(
        ctx,
        metric_id="BS.CLAIM.ALE_TO_PM.OPEN",
        debt_filter=lambda df: (df["debtor"].astype(str) == "Alejandro")
        & (df["creditor"].astype(str) == "Property Management"),
        value_col="open_total",
    )


def build_bs_debt_principal_open(ctx: MetricsContext) -> pd.DataFrame:
    return _build_debt_metric_qy(
        ctx,
        metric_id="BS.DEBT.PRINCIPAL.OPEN",
        debt_filter=lambda df: (df["debtor"].astype(str) == "Property Management")
        & (df["creditor"].astype(str).isin(["MI", "Primos"])),
        value_col="open_principal",
    )


def build_bs_debt_interest_open(ctx: MetricsContext) -> pd.DataFrame:
    return _build_debt_metric_qy(
        ctx,
        metric_id="BS.DEBT.INTEREST.OPEN",
        debt_filter=lambda df: (df["debtor"].astype(str) == "Property Management")
        & (df["creditor"].astype(str).isin(["MI", "Primos"])),
        value_col="open_interest",
    )


def build_bs_debt_total_open(ctx: MetricsContext) -> pd.DataFrame:
    return _build_debt_metric_qy(
        ctx,
        metric_id="BS.DEBT.TOTAL.OPEN",
        debt_filter=lambda df: (df["debtor"].astype(str) == "Property Management")
        & (df["creditor"].astype(str).isin(["MI", "Primos"])),
        value_col="open_total",
    )


def build_bs_debt_net_pm_position(ctx: MetricsContext) -> pd.DataFrame:
    liab = build_bs_debt_total_open(ctx)
    claims = build_bs_claim_ale_to_pm_open(ctx)
    if liab.empty and claims.empty:
        return concat_metric_frames([])

    wide = (
        pd.concat([liab, claims], ignore_index=True)
        .pivot_table(
            index=["period_grain", "period", "currency", "run_id", "as_of_date"],
            columns="metric_id",
            values="value",
            aggfunc="first",
        )
        .fillna(0.0)
        .reset_index()
    )
    for needed in ["BS.DEBT.TOTAL.OPEN", "BS.CLAIM.ALE_TO_PM.OPEN"]:
        if needed not in wide.columns:
            wide[needed] = 0.0
    wide["value"] = wide["BS.DEBT.TOTAL.OPEN"] - wide["BS.CLAIM.ALE_TO_PM.OPEN"]

    frames: list[pd.DataFrame] = []
    for _, row in wide.iterrows():
        frames.append(
            build_metric_frame(
                metric_id="BS.DEBT.NET_PM_POSITION",
                period_grain=str(row["period_grain"]),
                period=str(row["period"]),
                currency=str(row["currency"]),
                value=float(row["value"]),
                run_id=str(row["run_id"]),
                as_of_date=str(row["as_of_date"]),
                source_layer="debt_balance",
                build_detail="BS.DEBT.TOTAL.OPEN - BS.CLAIM.ALE_TO_PM.OPEN",
            )
        )
    return concat_metric_frames(frames)


BUILDER_REGISTRY = {
    "build_is_rent_total": build_is_rent_total,
    "build_is_contrib_total": build_is_contrib_total,
    "build_is_opex_total": build_is_opex_total,
    "build_is_draws_personal": build_is_draws_personal,
    "build_bs_cash_fb": build_bs_cash_fb,
    "build_bs_cash_pm": build_bs_cash_pm,
    "build_bs_debt_pm_to_mi_open": build_bs_debt_pm_to_mi_open,
    "build_bs_debt_pm_to_primos_open": build_bs_debt_pm_to_primos_open,
    "build_bs_claim_ale_to_pm_open": build_bs_claim_ale_to_pm_open,
    "build_bs_debt_principal_open": build_bs_debt_principal_open,
    "build_bs_debt_interest_open": build_bs_debt_interest_open,
    "build_bs_debt_total_open": build_bs_debt_total_open,
    "build_bs_debt_net_pm_position": build_bs_debt_net_pm_position,
}


def get_builder(builder_key: str):
    if builder_key not in BUILDER_REGISTRY:
        raise KeyError(f"Unknown builder_key: {builder_key}")
    return BUILDER_REGISTRY[builder_key]


def run_leaf_builders(ctx: MetricsContext, builder_keys: Iterable[str]) -> pd.DataFrame:
    frames = []
    for key in builder_keys:
        fn = get_builder(key)
        frames.append(fn(ctx))
    return concat_metric_frames(frames)
