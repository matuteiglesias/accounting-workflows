"""Annual professional companion tables for dashboard gaps.

The functions in this module are pure pandas builders used by report notebooks.
They keep accounting semantics explicit:
- cash close and debt position are stock metrics selected from the latest month
  available in each year;
- funding and debt activity are flow/support metrics summed by year and
  explicit dimensions;
- ARS and USD remain separate rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_YEAR_COLUMNS = ["2022", "2023", "2024", "2025", "2026"]
MONTH_RE = r"^20\d{2}-(0[1-9]|1[0-2])$"


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _clean_text(series: pd.Series | str, default: str = "") -> pd.Series | str:
    if isinstance(series, pd.Series):
        return series.fillna(default).astype(str).str.strip()
    return str(series).strip() if series is not None else default


def _ensure_columns(df: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def _period_month(df: pd.DataFrame) -> pd.Series:
    if "period" not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    s = df["period"].fillna("").astype(str).str.strip()
    extracted = s.str.extract(r"(20\d{2}-\d{2})", expand=False)
    dt = pd.to_datetime(s, errors="coerce")
    month = extracted.fillna(dt.dt.to_period("M").astype(str))
    return month.where(month.str.match(MONTH_RE, na=False), "")


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["selected_month"] = _period_month(out)
    out["period"] = out["selected_month"].str.slice(0, 4)
    return out[out["period"].str.match(r"^20\d{2}$", na=False)].copy()


def _annual_wide(long_df: pd.DataFrame, id_cols: Sequence[str], year_columns: Sequence[str] = DEFAULT_YEAR_COLUMNS) -> pd.DataFrame:
    if long_df.empty:
        return _empty([*id_cols, *year_columns])
    work = long_df.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce").fillna(0.0)
    wide = (
        work.pivot_table(index=list(id_cols), columns="period", values="value", aggfunc="sum", fill_value=0.0, dropna=True)
        .reset_index()
    )
    for year in year_columns:
        if year not in wide.columns:
            wide[year] = 0.0
    year_cols = [y for y in year_columns if y in wide.columns]
    extra_years = sorted([str(c) for c in wide.columns if str(c).isdigit() and str(c) not in year_cols])
    return wide[list(id_cols) + year_cols + extra_years].sort_values(list(id_cols)).reset_index(drop=True)


def build_annual_cash_close_by_box(cash: pd.DataFrame, year_columns: Sequence[str] = DEFAULT_YEAR_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual cash close by Box/Currency from latest month in each year.

    This is a stock metric. Monthly values are never summed.
    """
    long_cols = ["metric_id", "line_id", "period", "Box", "Currency", "value", "selected_month", "source_table", "source_filter", "calculation_rule"]
    if cash is None or cash.empty:
        return _empty(long_cols), _empty(["metric_id", "line_id", "Box", "Currency", "source_table", "source_filter", "calculation_rule", *year_columns])
    work = _ensure_columns(cash, {"Box": "N/A", "Currency": "N/A", "metric": "cash_close", "value": 0.0})
    work = _add_year(work)
    if work.empty:
        return _empty(long_cols), _empty(["metric_id", "line_id", "Box", "Currency", "source_table", "source_filter", "calculation_rule", *year_columns])
    work["value"] = pd.to_numeric(work["value"], errors="coerce").fillna(0.0)
    work["Box"] = _clean_text(work["Box"], "N/A")
    work["Currency"] = _clean_text(work["Currency"], "N/A")
    work = work.sort_values(["period", "Box", "Currency", "selected_month"])
    latest = work.groupby(["period", "Box", "Currency"], dropna=False).tail(1).copy()
    latest["metric_id"] = "CASH.CLOSE.BY_BOX"
    latest["line_id"] = "CASH.CLOSE.BY_BOX." + latest["Box"].astype(str) + "." + latest["Currency"].astype(str)
    latest["source_table"] = "monthly_cash_close.csv"
    latest["source_filter"] = "latest selected_month in year by Box/Currency; metric=cash_close"
    latest["calculation_rule"] = "annual stock = latest available monthly cash close in year; never sum monthly cash closes"
    long_df = latest[long_cols].sort_values(["period", "Currency", "Box"]).reset_index(drop=True)
    wide_df = _annual_wide(long_df, ["metric_id", "line_id", "Box", "Currency", "source_table", "source_filter", "calculation_rule"], year_columns)
    return long_df, wide_df


def _funding_dimension_defaults(flow: pd.DataFrame) -> pd.DataFrame:
    work = _ensure_columns(flow, {
        "Currency": "N/A", "Box": "", "semantic_bucket": "", "semantic_subbucket": "", "cash_path": "",
        "actor": "", "counterparty": "", "payer": "", "receiver": "", "funding_actor": "", "funding_channel": "",
        "source_box": "", "target_box": "", "beneficiary_box": "", "obligation_box": "", "cash_effect": "", "debt_effect": "",
        "amount_in": 0.0, "amount_out": 0.0, "amount_abs": 0.0, "net_amount": 0.0,
    })
    for c in ["Currency", "Box", "semantic_bucket", "semantic_subbucket", "cash_path", "actor", "counterparty", "payer", "receiver", "funding_actor", "funding_channel", "source_box", "target_box", "beneficiary_box", "obligation_box", "cash_effect", "debt_effect"]:
        work[c] = _clean_text(work[c])
    for c in ["amount_in", "amount_out", "amount_abs", "net_amount"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    original_actor = work["funding_actor"].copy()
    original_channel = work["funding_channel"].copy()
    candidate_blob = (work["semantic_bucket"] + " " + work["semantic_subbucket"] + " " + work["cash_path"] + " " + work["payer"] + " " + work["receiver"]).str.lower()
    work["_is_funding_candidate"] = (
        work["semantic_bucket"].str.lower().eq("funding_contribution")
        | original_actor.ne("")
        | original_channel.ne("")
        | candidate_blob.str.contains(r"fund|funding|aporte|contrib|support|soporte|tenant.*direct|inquil.*direct|debt", regex=True, na=False)
    )

    actor_fallback = work["funding_actor"].where(work["funding_actor"].ne(""), work["actor"])
    actor_fallback = actor_fallback.where(actor_fallback.ne(""), work["counterparty"])
    actor_fallback = actor_fallback.where(actor_fallback.ne(""), work["payer"])
    work["funding_actor"] = actor_fallback.fillna("").astype(str).str.strip()
    tenant_actor = work["funding_actor"].str.lower().isin({"inq", "inquilino", "inquilinos", "tenant", "tenants"}) | work["payer"].str.lower().isin({"inq", "inquilino", "inquilinos", "tenant", "tenants"})
    work.loc[tenant_actor, "funding_actor"] = "Tenants"

    lower_blob = (work["cash_path"] + " " + work["semantic_subbucket"] + " " + work["payer"] + " " + work["receiver"] + " " + work["debt_effect"]).str.lower()
    channel = work["funding_channel"].copy()
    channel = channel.mask(channel.eq("") & lower_blob.str.contains("service|servicio", regex=True, na=False) & lower_blob.str.contains("tenant.*direct|inquil.*direct|direct", regex=True, na=False), "tenant_direct_service_payment")
    channel = channel.mask(channel.eq("") & lower_blob.str.contains("tenant.*direct|inquil.*direct|tax|impuesto", regex=True, na=False), "tenant_direct_tax_payment")
    channel = channel.mask(channel.eq("") & lower_blob.str.contains("tenant|inquil", regex=True, na=False), "tenant_to_box")
    channel = channel.mask(channel.eq("") & lower_blob.str.contains("debt|repay|settlement|deuda", regex=True, na=False), "debt_settlement")
    named_actor = work["funding_actor"].ne("") & ~work["funding_actor"].eq("Tenants")
    channel = channel.mask(channel.eq("") & work["semantic_bucket"].eq("funding_contribution") & named_actor, "named_actor_support")
    channel = channel.mask(channel.eq("") & work["semantic_bucket"].eq("funding_contribution"), "cash_to_box")
    work["funding_channel"] = channel.fillna("").astype(str).str.strip()

    work["target_box"] = work["target_box"].where(work["target_box"].ne(""), work["beneficiary_box"])
    work["target_box"] = work["target_box"].where(work["target_box"].ne(""), work["Box"])
    work["cash_effect"] = work["cash_effect"].where(work["cash_effect"].ne(""), "cash_in_box")
    direct = work["funding_channel"].str.startswith("tenant_direct", na=False)
    work.loc[direct, "cash_effect"] = work.loc[direct, "cash_effect"].where(work.loc[direct, "cash_effect"].ne("cash_in_box"), "no_cash_in_box_direct_payment")
    debt_linked = work["debt_effect"].str.lower().ne("") & ~work["debt_effect"].str.lower().isin({"none", "nan", "n/a"})
    debt_linked |= work["funding_channel"].str.lower().str.contains("debt", na=False)
    work["_is_debt_linked"] = debt_linked

    direct_value = work["amount_abs"].where(work["amount_abs"].ne(0.0), work["amount_in"].abs() + work["amount_out"].abs())
    direct_value = direct_value.where(direct_value.ne(0.0), work["net_amount"].abs())
    cash_value = work["amount_in"].where(work["amount_in"].ne(0.0), work["net_amount"].clip(lower=0.0))
    work["support_value"] = cash_value.where(~direct, direct_value)
    work["_value_rule"] = "cash-to-box uses amount_in; direct obligation support uses amount_abs fallback to abs(net_amount); debt-linked uses the same support_value policy"
    return work


def build_annual_funding_by_actor_channel(flow: pd.DataFrame, year_columns: Sequence[str] = DEFAULT_YEAR_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual funding/support by actor/channel/cash-effect dimensions.

    The output intentionally contains multiple metric_id views over the same
    explicit dimensions so dashboards can fill actor, channel, cash-effect,
    direct-obligation, cash-to-box, target-box, and debt-linked rows without
    renderer-only semantics.
    """
    dims = ["Currency", "funding_actor", "funding_channel", "cash_effect", "target_box", "beneficiary_box", "obligation_box"]
    long_cols = ["metric_id", "line_id", "period", *dims, "value", "source_table", "source_filter", "calculation_rule"]
    empty_wide = _empty(["metric_id", "line_id", *dims, "source_table", "source_filter", "calculation_rule", *year_columns])
    if flow is None or flow.empty:
        return _empty(long_cols), empty_wide
    work = _add_year(_funding_dimension_defaults(flow))
    if work.empty:
        return _empty(long_cols), empty_wide
    work = work[work["_is_funding_candidate"].astype(bool)].copy()
    if work.empty:
        return _empty(long_cols), empty_wide
    work["support_value"] = pd.to_numeric(work["support_value"], errors="coerce").fillna(0.0)

    specs = [
        ("FUND.CONTRIB.BY_FUNDING_ACTOR", pd.Series(True, index=work.index), "funding_actor"),
        ("FUND.CONTRIB.BY_CHANNEL", pd.Series(True, index=work.index), "funding_channel"),
        ("FUND.CONTRIB.BY_CASH_EFFECT", pd.Series(True, index=work.index), "cash_effect"),
        ("FUND.CONTRIB.BY_TARGET_BOX", pd.Series(True, index=work.index), "target_box"),
        ("FUND.CONTRIB.DIRECT_OBLIGATION", work["funding_channel"].str.startswith("tenant_direct", na=False), "funding_channel"),
        ("FUND.CONTRIB.CASH_TO_BOX", work["funding_channel"].isin(["tenant_to_box", "cash_to_box"]) | work["cash_effect"].eq("cash_in_box"), "funding_channel"),
        ("FUND.CONTRIB.DEBT_LINKED", work["_is_debt_linked"].astype(bool), "funding_channel"),
    ]
    parts = []
    for metric_id, mask, primary_dim in specs:
        sub = work[mask].copy()
        if sub.empty:
            continue
        grouped = sub.groupby(["period", *dims], dropna=False)["support_value"].sum().reset_index(name="value")
        grouped["metric_id"] = metric_id
        grouped["line_id"] = (
            metric_id + "."
            + grouped[primary_dim].astype(str).replace("", "unknown") + "."
            + grouped["funding_actor"].astype(str).replace("", "unknown") + "."
            + grouped["funding_channel"].astype(str).replace("", "unknown") + "."
            + grouped["cash_effect"].astype(str).replace("", "unknown") + "."
            + grouped["Currency"].astype(str)
        )
        grouped["source_table"] = "monthly_flow_semantic_split.csv"
        grouped["source_filter"] = f"{metric_id}: semantic_bucket=funding_contribution or explicit funding/support dimensions; primary_dimension={primary_dim}"
        grouped["calculation_rule"] = "annual flow/support = sum support_value by year/Currency/funding dimensions; amount_in for cash-to-box funding; amount_abs fallback to abs(net_amount) for direct obligation support; ARS/USD not mixed"
        parts.append(grouped)
    if not parts:
        return _empty(long_cols), empty_wide
    long_df = pd.concat(parts, ignore_index=True)[long_cols]
    long_df = long_df.sort_values(["metric_id", "period", "Currency", "funding_actor", "funding_channel", "cash_effect"]).reset_index(drop=True)
    wide_df = _annual_wide(long_df, ["metric_id", "line_id", *dims, "source_table", "source_filter", "calculation_rule"], year_columns)
    return long_df, wide_df


def build_annual_debt_stock_by_pair(debt_pos: pd.DataFrame, year_columns: Sequence[str] = DEFAULT_YEAR_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual debt stock by pair from the latest selected close snapshot.

    Debt position is a stock metric.  If monthly_debt_position carries multiple
    snapshots inside the same month, select the latest as_of_date for that
    month/pair before selecting the latest month in the year.  Values are never
    summed across monthly positions or across intra-month snapshots.
    """
    dims = ["Currency", "debtor", "creditor", "pair", "component"]
    long_cols = ["metric_id", "line_id", "period", *dims, "value", "selected_month", "selected_as_of_date", "source_table", "source_filter", "calculation_rule"]
    wide_empty = _empty(["metric_id", "line_id", *dims, "source_table", "source_filter", "calculation_rule", *year_columns])
    if debt_pos is None or debt_pos.empty:
        return _empty(long_cols), wide_empty
    work = _ensure_columns(debt_pos, {"Currency": "N/A", "debtor": "", "creditor": "", "pair": "", "open_principal": 0.0, "open_interest": 0.0, "open_total": 0.0, "as_of_date": ""})
    work = _add_year(work)
    if work.empty:
        return _empty(long_cols), wide_empty
    for c in ["Currency", "debtor", "creditor", "pair", "as_of_date"]:
        work[c] = _clean_text(work[c])
    work["pair"] = work["pair"].where(work["pair"].ne(""), work["debtor"] + " → " + work["creditor"])
    work["selected_as_of_date"] = work["as_of_date"]
    work["__as_of_date"] = pd.to_datetime(work["selected_as_of_date"].replace("", pd.NA), errors="coerce")

    # First collapse multiple snapshots within the same month/pair to the latest
    # selected as_of_date.  Without as_of_date evidence this is stable and keeps
    # the last source row only; it never sums stock snapshots.
    monthly_keys = ["selected_month", "Currency", "debtor", "creditor", "pair"]
    monthly_close = (
        work.sort_values(monthly_keys + ["__as_of_date"], na_position="first")
        .groupby(monthly_keys, dropna=False)
        .tail(1)
        .copy()
    )
    # Then pick the latest selected monthly close in each year/pair.
    yearly_keys = ["period", "Currency", "debtor", "creditor", "pair"]
    latest = (
        monthly_close.sort_values(yearly_keys + ["selected_month", "__as_of_date"], na_position="first")
        .groupby(yearly_keys, dropna=False)
        .tail(1)
        .copy()
    )
    latest["selected_as_of_date"] = latest["selected_as_of_date"].fillna("").astype(str).str.strip()
    latest = latest.drop(columns=["__as_of_date"], errors="ignore")

    parts = []
    for component, metric_id in [("open_principal", "DEBT.STOCK.BY_PAIR.OPEN_PRINCIPAL"), ("open_interest", "DEBT.STOCK.BY_PAIR.OPEN_INTEREST"), ("open_total", "DEBT.STOCK.BY_PAIR.OPEN_TOTAL")]:
        tmp = latest[["period", "Currency", "debtor", "creditor", "pair", "selected_month", "selected_as_of_date", component]].copy()
        tmp = tmp.rename(columns={component: "value"})
        tmp["component"] = component
        tmp["metric_id"] = metric_id
        parts.append(tmp)
    long_df = pd.concat(parts, ignore_index=True) if parts else _empty(long_cols)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0)
    long_df["line_id"] = long_df["metric_id"] + "." + long_df["pair"].astype(str) + "." + long_df["Currency"].astype(str)
    long_df["source_table"] = "monthly_debt_position.csv"
    long_df["source_filter"] = "latest selected_as_of_date within month, then latest selected_month in year by Currency/debtor/creditor/pair"
    long_df["calculation_rule"] = "annual stock = latest selected monthly debt close in year; if multiple snapshots exist in a month use latest as_of_date; never sum monthly positions or snapshots"
    long_df = long_df[long_cols].sort_values(["period", "Currency", "pair", "component"]).reset_index(drop=True)
    wide_df = _annual_wide(long_df, ["metric_id", "line_id", *dims, "source_table", "source_filter", "calculation_rule"], year_columns)
    return long_df, wide_df


def build_annual_debt_activity_by_pair(debt_act: pd.DataFrame, year_columns: Sequence[str] = DEFAULT_YEAR_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual debt activity/repayment flows by pair and activity type."""
    dims = ["Currency", "debtor", "creditor", "pair", "activity_type"]
    long_cols = ["metric_id", "line_id", "period", *dims, "value", "source_table", "source_filter", "calculation_rule"]
    if debt_act is None or debt_act.empty:
        return _empty(long_cols), _empty(["metric_id", "line_id", *dims, "source_table", "source_filter", "calculation_rule", *year_columns])
    work = _ensure_columns(debt_act, {"Currency": "N/A", "debtor": "", "creditor": "", "pair": "", "new_principal": 0.0, "interest_accrued": 0.0, "repayments": 0.0, "settlements": pd.NA, "adjustments": 0.0, "net_change": 0.0})
    work = _add_year(work)
    if work.empty:
        return _empty(long_cols), _empty(["metric_id", "line_id", *dims, "source_table", "source_filter", "calculation_rule", *year_columns])
    for c in ["Currency", "debtor", "creditor", "pair"]:
        work[c] = _clean_text(work[c])
    work["pair"] = work["pair"].where(work["pair"].ne(""), work["debtor"] + " → " + work["creditor"])
    value_cols = ["new_principal", "repayments", "settlements", "net_change", "interest_accrued", "adjustments"]
    for c in value_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    # If the source has no explicit settlements column, expose settlements as a
    # metric view over repayments so dashboards can fill settlement/repayment
    # rows without changing the underlying source value.
    work["settlements"] = work["settlements"].fillna(work["repayments"])
    for c in value_cols:
        work[c] = work[c].fillna(0.0)
    melted = work.melt(id_vars=["period", "Currency", "debtor", "creditor", "pair"], value_vars=value_cols, var_name="activity_type", value_name="value")
    grouped = melted.groupby(["period", *dims], dropna=False)["value"].sum().reset_index()
    metric_map = {
        "repayments": "DEBT.ACTIVITY.REPAYMENT.BY_PAIR",
        "new_principal": "DEBT.ACTIVITY.INCREASE.BY_PAIR",
        "net_change": "DEBT.ACTIVITY.NET_MOVEMENT.BY_PAIR",
        "settlements": "DEBT.ACTIVITY.SETTLEMENT.BY_PAIR",
        "interest_accrued": "DEBT.ACTIVITY.INTEREST.BY_PAIR",
        "adjustments": "DEBT.ACTIVITY.ADJUSTMENT.BY_PAIR",
    }
    grouped["metric_id"] = grouped["activity_type"].map(metric_map).fillna("DEBT.ACTIVITY.BY_PAIR")
    grouped["line_id"] = grouped["metric_id"] + "." + grouped["pair"].astype(str) + "." + grouped["Currency"].astype(str)
    grouped["source_table"] = "monthly_debt_activity.csv"
    grouped["source_filter"] = "monthly_debt_activity.csv activity_type by Currency/debtor/creditor/pair; settlements use explicit settlements column when present, otherwise repayment view"
    grouped["calculation_rule"] = "annual flow = sum monthly debt activity by year/Currency/pair/activity_type; repayments and settlements are flow views, not debt stock"
    long_df = grouped[long_cols].sort_values(["period", "Currency", "pair", "activity_type"]).reset_index(drop=True)
    wide_df = _annual_wide(long_df, ["metric_id", "line_id", *dims, "source_table", "source_filter", "calculation_rule"], year_columns)
    return long_df, wide_df


def write_annual_long_and_wide(long_df: pd.DataFrame, wide_df: pd.DataFrame, tables_dir: Path, stem: str) -> dict[str, Path | None]:
    """Write long/wide annual companion tables when non-empty.

    Empty outputs remove stale files to prevent old public values from lingering.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {}
    for suffix, df in [("long", long_df), ("wide", wide_df)]:
        path = tables_dir / f"{stem}_{suffix}.csv"
        if df is None or df.empty:
            if path.exists():
                path.unlink()
            paths[suffix] = None
        else:
            df.to_csv(path, index=False)
            paths[suffix] = path
    return paths
