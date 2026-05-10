from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from accounting.logging_utils import configure_logging, get_logger

from .drilldown import build_metric_drilldown_artifacts
from .builders import run_leaf_builders
from .derive import derive_default_v1
from .io import MetricsContext, ensure_metric_values_schema
from .registry import default_metric_specs_v1, registry_from_specs, normalize_registry
from .validate import run_basic_validations
from .views import (
    build_draws_discipline_monthly_last6,
    build_flow_rollup_last_n_months,
    build_income_statement_monthly_last6,
    load_ledger,
    parse_noise_floor,
)


METRIC_VALUES_FILENAME = "metric_values.csv"
METRIC_REGISTRY_FILENAME = "metric_registry.csv"

LOG = get_logger("metrics")
VALIDATION_REPORT_FILENAME = "validation_report.csv"
BUILD_MANIFEST_FILENAME = "build_manifest.json"
METRIC_VIEWS_DIRNAME = "metric_views"
REQUIRED_METRIC_VIEW_FILES = [
    "income_statement_monthly_last6.csv",
    "rent_rollup_by_place_m_last6.csv",
    "rent_rollup_by_detail_m_last6.csv",
    "flow_type_rollup_m_last6.csv",
    "draws_discipline_monthly_last6.csv",
    "metric_views_manifest.csv",
]


INCOME_STATEMENT_EXPORT_IDS = [
    "IS.RENT.CABA",
    "IS.RENT.TORCUATO",
    "IS.RENT.TOTAL",
    "IS.CONTRIB.MATIAS",
    "IS.CONTRIB.ALEJANDRO",
    "IS.CONTRIB.INQ_DIR",
    "IS.CONTRIB.INQ_CAJA",
    "IS.CONTRIB.TOTAL",
    "IS.INCOME.TOTAL",
    "IS.OPEX.TAX",
    "IS.OPEX.LEGAL",
    "IS.OPEX.SERVICES",
    "IS.OPEX.MAINTENANCE",
    "IS.OPEX.TOTAL",
    "IS.NET.AFTER_COSTS",
    "IS.DRAWS.PERSONAL",
    "IS.DIVIDENDS",
    "IS.NET.POST_DRAWS",
]

BALANCE_CASH_EXPORT_IDS = [
    "BS.CASH.FB",
    "BS.CASH.PM",
    "BS.CASH.TOTAL",
]

BALANCE_DEBT_EXPORT_IDS = [
    "BS.DEBT.PM_TO_MI.OPEN",
    "BS.DEBT.PM_TO_PRIMOS.OPEN",
    "BS.CLAIM.ALE_TO_PM.OPEN",
    "BS.DEBT.PRINCIPAL.OPEN",
    "BS.DEBT.INTEREST.OPEN",
    "BS.DEBT.TOTAL.OPEN",
    "BS.DEBT.NET_PM_POSITION",
]


def _ordered_metric_filter(df: pd.DataFrame, metric_ids: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    wanted = [metric_id for metric_id in metric_ids if metric_id in set(df["metric_id"].astype(str))]
    if not wanted:
        return df.iloc[0:0].copy()
    out = df.loc[df["metric_id"].astype(str).isin(wanted)].copy()
    out["metric_id"] = pd.Categorical(out["metric_id"], categories=wanted, ordered=True)
    return out.sort_values(["metric_id", "currency", "period"]).reset_index(drop=True)


def find_latest_run_root(base: Path) -> Path:
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {base}")
    candidates = sorted(candidates, key=lambda p: p.name)
    return candidates[-1]


def load_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_optional_csv_candidates(paths: list[Path]) -> Optional[pd.DataFrame]:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return None


DEBT_BALANCE_REQUIRED_COLUMNS = [
    "period",
    "currency",
    "debtor",
    "creditor",
    "open_principal",
    "open_interest",
    "open_total",
]


def _debt_candidate_dirs(run_root: Path) -> list[Path]:
    run_id = run_root.name
    out_root = run_root
    while out_root.name != "out" and out_root.parent != out_root:
        out_root = out_root.parent
    if out_root.name != "out":
        out_root = run_root.parent

    return [
        run_root,
        run_root.parent / "debt_resolution" / run_id,
        run_root.parent.parent / "debt_resolution" / run_id,
        out_root / "debt_resolution" / run_id,
        out_root / "debt_resolution" / "latest",
    ]


def _validate_debt_schema(df: pd.DataFrame, label: str, source_path: Path) -> None:
    missing = [c for c in DEBT_BALANCE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}; source={source_path}"
        )


def _load_debt_artifact(run_root: Path, filename: str) -> Optional[pd.DataFrame]:
    candidate_dirs = _debt_candidate_dirs(run_root)
    candidates = [d / filename for d in candidate_dirs]

    LOG.info("Debt artifact lookup %s candidate_paths=%s", filename, [str(p) for p in candidates])
    for path in candidates:
        exists = path.exists()
        LOG.info("Debt artifact candidate %s exists=%s", path, exists)
        if not exists:
            continue
        df = pd.read_csv(path)
        LOG.info("Debt artifact loaded %s rows=%d", path, len(df))
        _validate_debt_schema(df, filename, path)
        return df

    LOG.warning(
        "Debt artifact not found filename=%s checked_paths=%s",
        filename,
        [str(p) for p in candidates],
    )
    return None


def load_context(run_root: Path, run_id: str, as_of_date: str) -> MetricsContext:
    views_dir = run_root / "views"


    # TODO. metrics.build debt lookup points at a suspicious path

    # Tu load_context() arma:
    # debt_run_dir = run_root.parent.parent / "debt_resolution" / run_root.name
    # Si run_root es out/run/accounting/<run_id>, eso apunta a out/run/debt_resolution/<run_id>, no a out/debt_resolution/<run_id>.
    # Ahí revisaría antes de confiar.

    ledger = load_ledger(run_root)
    per_flow = pd.read_csv(run_root / "per_flow_time_long.freq=M.csv")
    daily_cash_position = pd.read_csv(run_root / "daily_cash_position.csv")
    v_contributions_monthly = pd.read_csv(views_dir / "v_contributions_monthly.csv")
    v_opex_category_monthly = pd.read_csv(views_dir / "v_opex_category_monthly.csv")
    party_balance_detailed = load_optional_csv(views_dir / "party_balance_detailed.csv")
    debt_balance_monthly = _load_debt_artifact(run_root, "debt_balance_monthly.csv")
    debt_balance_quarterly = _load_debt_artifact(run_root, "debt_balance_quarterly.csv")
    debt_balance_yearly = _load_debt_artifact(run_root, "debt_balance_yearly.csv")

    return MetricsContext(
        ledger=ledger,
        per_flow=per_flow,
        daily_cash_position=daily_cash_position,
        v_contributions_monthly=v_contributions_monthly,
        v_opex_category_monthly=v_opex_category_monthly,
        party_balance_detailed=party_balance_detailed,
        debt_balance_monthly=debt_balance_monthly,
        debt_balance_quarterly=debt_balance_quarterly,
        debt_balance_yearly=debt_balance_yearly,
        run_id=run_id,
        as_of_date=as_of_date,
    )


def select_builder_keys(registry_df: pd.DataFrame) -> list[str]:
    reg = normalize_registry(registry_df)
    reg = reg[(reg["is_leaf"]) & (reg["status"] == "active")]
    keys = [x for x in reg["builder_key"].astype(str).tolist() if x.strip()]
    return keys


def build_wide_views(metric_values: pd.DataFrame, out_dir: Path) -> None:
    mv = ensure_metric_values_schema(metric_values)

    for grain in ["Y", "Q"]:
        sub = mv.loc[mv["period_grain"] == grain].copy()
        if sub.empty:
            continue

        wide = (
            sub.assign(metric_key=sub["metric_id"] + " [" + sub["currency"] + "]")
            .pivot_table(
                index="metric_key",
                columns="period",
                values="value",
                aggfunc="first",
            )
            .sort_index()
            .reset_index()
        )
        wide.to_csv(out_dir / f"metric_values_{grain.lower()}_wide.csv", index=False)


def build_statement_views(metric_values: pd.DataFrame, out_dir: Path) -> None:
    mv = ensure_metric_values_schema(metric_values)

    statement_specs = [
        ("income_statement_y.csv", INCOME_STATEMENT_EXPORT_IDS),
        ("balance_cash_y.csv", BALANCE_CASH_EXPORT_IDS),
        ("balance_debt_y.csv", BALANCE_DEBT_EXPORT_IDS),
        ("income_statement_q.csv", INCOME_STATEMENT_EXPORT_IDS),
        ("balance_cash_q.csv", BALANCE_CASH_EXPORT_IDS),
        ("balance_debt_q.csv", BALANCE_DEBT_EXPORT_IDS),
    ]

    for name, metric_ids in statement_specs:
        grain = "Y" if name.endswith("_y.csv") else "Q"
        sub = mv.loc[mv["period_grain"] == grain].copy()
        sub = _ordered_metric_filter(sub, metric_ids)
        if sub.empty:
            continue

        wide = (
            sub.assign(metric_key=sub["metric_id"].astype(str) + " [" + sub["currency"] + "]")
            .pivot_table(
                index="metric_key",
                columns="period",
                values="value",
                aggfunc="first",
                sort=False,
            )
            .reset_index()
        )
        wide.to_csv(out_dir / name, index=False)


def build_debt_metric_views(ctx: MetricsContext, out_dir: Path) -> None:
    views_dir = out_dir / METRIC_VIEWS_DIRNAME
    views_dir.mkdir(parents=True, exist_ok=True)

    monthly = ctx.debt_balance_monthly
    if monthly is None or monthly.empty:
        return

    needed = ["period", "debtor", "creditor", "currency", "open_principal", "open_interest", "open_total"]
    missing = [c for c in needed if c not in monthly.columns]
    if missing:
        return

    work = monthly.copy()
    for col in ["open_principal", "open_interest", "open_total"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    work = (
        work.groupby(["period", "debtor", "creditor", "currency"], dropna=False)[["open_principal", "open_interest", "open_total"]]
        .max()
        .reset_index()
        .sort_values(["period", "debtor", "creditor", "currency"])
    )
    periods = sorted(work["period"].astype(str).unique().tolist())
    keep_periods = set(periods[-12:])
    last12 = work.loc[work["period"].astype(str).isin(keep_periods)].copy()

    last12.to_csv(views_dir / "debt_balance_monthly_last12.csv", index=False)
    last12.to_csv(views_dir / "debt_by_counterparty_m_last12.csv", index=False)

    net = (
        last12.groupby(["period", "currency"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "pm_liabilities_to_mi": g.loc[
                        (g["debtor"].astype(str) == "Property Management")
                        & (g["creditor"].astype(str) == "MI"),
                        "open_total",
                    ].sum(),
                    "pm_liabilities_to_primos": g.loc[
                        (g["debtor"].astype(str) == "Property Management")
                        & (g["creditor"].astype(str) == "Primos"),
                        "open_total",
                    ].sum(),
                    "pm_claims_on_alejandro": g.loc[
                        (g["debtor"].astype(str) == "Alejandro")
                        & (g["creditor"].astype(str) == "Property Management"),
                        "open_total",
                    ].sum(),
                }
            )
        )
        .reset_index()
    )
    if net.empty:
        net = pd.DataFrame(
            columns=[
                "period",
                "currency",
                "pm_liabilities_to_mi",
                "pm_liabilities_to_primos",
                "pm_claims_on_alejandro",
                "pm_net_position",
            ]
        )
    else:
        net["pm_net_position"] = (
            net["pm_liabilities_to_mi"] + net["pm_liabilities_to_primos"] - net["pm_claims_on_alejandro"]
        )
    net.to_csv(views_dir / "debt_net_position_m_last12.csv", index=False)


def _build_cash_position_monthly_last12(ctx: MetricsContext, views_dir: Path) -> None:
    daily = ctx.daily_cash_position
    if daily is None or daily.empty:
        return

    work = daily.copy()
    date_col = next((c for c in ["Date", "date", "fecha", "as_of_date"] if c in work.columns), None)
    box_col = next((c for c in ["Box", "box"] if c in work.columns), None)
    currency_col = next((c for c in ["Currency", "currency"] if c in work.columns), None)
    value_col = next((c for c in ["cash_position", "balance", "amount", "value", "cash"] if c in work.columns), None)
    if not all([date_col, box_col, currency_col, value_col]):
        return

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.loc[work[date_col].notna()].copy()
    if work.empty:
        return

    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[work[value_col].notna()].copy()
    if work.empty:
        return

    work["period"] = work[date_col].dt.to_period("M").astype(str)
    work = work.sort_values([box_col, currency_col, date_col])
    monthly = (
        work.groupby(["period", box_col, currency_col], as_index=False)[value_col]
        .last()
        .rename(columns={box_col: "box", currency_col: "currency", value_col: "cash_position_end"})
    )
    periods = sorted(monthly["period"].astype(str).unique().tolist())
    keep_periods = set(periods[-12:])
    monthly = monthly.loc[monthly["period"].astype(str).isin(keep_periods)].copy()
    monthly.to_csv(views_dir / "cash_position_monthly_last12.csv", index=False)


def _build_contrib_rollup_views(ctx: MetricsContext, views_dir: Path) -> None:
    contrib = ctx.v_contributions_monthly
    if contrib is None or contrib.empty:
        return

    needed = ["TimePeriod", "Currency", "contributor_party", "amount"]
    if any(c not in contrib.columns for c in needed):
        return

    work = contrib.copy()
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)
    work["TimePeriod"] = work["TimePeriod"].astype(str)

    periods = sorted(work["TimePeriod"].unique().tolist())
    keep_periods = set(periods[-12:])
    last12 = work.loc[work["TimePeriod"].isin(keep_periods)].copy()

    monthly = (
        last12.groupby(["contributor_party", "Currency", "TimePeriod"], as_index=False)["amount"]
        .sum()
        .sort_values(["contributor_party", "Currency", "TimePeriod"])
    )
    latest_period = monthly["TimePeriod"].max() if not monthly.empty else ""
    rollup = (
        monthly.groupby(["contributor_party", "Currency"], as_index=False)
        .agg(
            months=("TimePeriod", "nunique"),
            total_12m=("amount", "sum"),
            avg_m=("amount", "mean"),
        )
    )
    if not rollup.empty:
        last_vals = (
            monthly.loc[monthly["TimePeriod"] == latest_period, ["contributor_party", "Currency", "amount"]]
            .rename(columns={"amount": "last_m"})
        )
        rollup = rollup.merge(last_vals, on=["contributor_party", "Currency"], how="left")
        rollup["last_period"] = latest_period
    rollup.to_csv(views_dir / "contrib_rollup_by_party_m_last12.csv", index=False)

    yearly = (
        work.assign(year=work["TimePeriod"].str.slice(0, 4))
        .groupby(["year", "contributor_party", "Currency"], as_index=False)["amount"]
        .sum()
        .rename(columns={"Currency": "currency", "amount": "total_y"})
        .sort_values(["year", "contributor_party", "currency"])
    )
    yearly.to_csv(views_dir / "contrib_rollup_by_party_y.csv", index=False)


def _build_opex_rollup_views(ctx: MetricsContext, views_dir: Path) -> None:
    opex = ctx.v_opex_category_monthly
    if opex is None or opex.empty:
        return

    category_col = "Tipo" if "Tipo" in opex.columns else ("category" if "category" in opex.columns else None)
    amount_col = "amount_out" if "amount_out" in opex.columns else ("amount" if "amount" in opex.columns else None)
    if category_col is None or amount_col is None or "TimePeriod" not in opex.columns or "Currency" not in opex.columns:
        return

    work = opex.copy()
    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce").fillna(0.0)
    work["TimePeriod"] = work["TimePeriod"].astype(str)

    periods = sorted(work["TimePeriod"].unique().tolist())
    keep_periods = set(periods[-12:])
    last12 = work.loc[work["TimePeriod"].isin(keep_periods)].copy()
    monthly = (
        last12.groupby([category_col, "Currency", "TimePeriod"], as_index=False)[amount_col]
        .sum()
        .rename(columns={category_col: "category", "Currency": "currency", amount_col: "amount_out"})
        .sort_values(["category", "currency", "TimePeriod"])
    )
    monthly.to_csv(views_dir / "opex_by_category_m_last12.csv", index=False)

    yearly = (
        work.assign(year=work["TimePeriod"].str.slice(0, 4))
        .groupby(["year", category_col, "Currency"], as_index=False)[amount_col]
        .sum()
        .rename(columns={category_col: "category", "Currency": "currency", amount_col: "amount_out_y"})
        .sort_values(["year", "category", "currency"])
    )
    yearly.to_csv(views_dir / "opex_by_category_y.csv", index=False)


def build_metric_view_exports(
    run_root: Path,
    metric_values: pd.DataFrame,
    registry: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: dict,
    out_dir: Path,
    *,
    months: int,
    rent_place_col: str,
    rent_detail_col: str,
    flow_rollup_groupby: list[str],
    include_statuses: tuple[str, ...],
    noise_floor_by_currency: dict[str, float],
) -> None:
    ledger = load_ledger(run_root)
    views_dir = out_dir / METRIC_VIEWS_DIRNAME
    views_dir.mkdir(parents=True, exist_ok=True)

    build_income_statement_monthly_last6(
        ledger,
        months=months,
        include_statuses=include_statuses,
    ).to_csv(views_dir / "income_statement_monthly_last6.csv", index=False)

    build_flow_rollup_last_n_months(
        ledger,
        flow_filter="Cobros",
        type_filter="Renta",
        groupby_cols=["Box", "Currency", rent_place_col],
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    ).to_csv(views_dir / "rent_rollup_by_place_m_last6.csv", index=False)

    build_flow_rollup_last_n_months(
        ledger,
        flow_filter="Cobros",
        type_filter="Renta",
        groupby_cols=["Box", "Currency", rent_detail_col],
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    ).to_csv(views_dir / "rent_rollup_by_detail_m_last6.csv", index=False)

    build_flow_rollup_last_n_months(
        ledger,
        groupby_cols=list(flow_rollup_groupby),
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
        top_n=12,
    ).to_csv(views_dir / "flow_type_rollup_m_last6.csv", index=False)

    build_draws_discipline_monthly_last6(
        ledger,
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    ).to_csv(views_dir / "draws_discipline_monthly_last6.csv", index=False)

    pd.DataFrame([
        {
            "run_root": str(run_root),
            "months": months,
            "rent_place_col": rent_place_col,
            "rent_detail_col": rent_detail_col,
            "flow_rollup_groupby": ",".join(flow_rollup_groupby),
            "include_statuses": ",".join(include_statuses),
            "noise_floor": json.dumps(noise_floor_by_currency, ensure_ascii=False, sort_keys=True),
            "metric_values_rows": int(len(metric_values)),
            "registry_rows": int(len(registry)),
            "validation_rows": int(len(validation)),
            "source_run_id": manifest.get("run_id", ""),
        }
    ]).to_csv(views_dir / "metric_views_manifest.csv", index=False)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build metric_values from accounting artifacts.")
    parser.add_argument(
        "--run-root",
        default="",
        help="Path to a specific run root, e.g. out/run/accounting/20260127T143301Z",
    )
    parser.add_argument(
        "--runs-base",
        default="out/run/accounting",
        help="Base dir used when --run-root is omitted.",
    )
    parser.add_argument(
        "--out-dir",
        default="out/metrics/latest",
        help="Output directory for metric artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional explicit run_id for metric_values. Defaults to source run dir name.",
    )
    parser.add_argument(
        "--as-of-date",
        default=pd.Timestamp.today().date().isoformat(),
        help="as_of_date string stored in metric_values.",
    )
    parser.add_argument("--months", type=int, default=6, help="Number of monthly periods to surface in metric_views.")
    parser.add_argument("--rent-place-col", default="Lugar", help="Column used for rent rollup by place in metric_views.")
    parser.add_argument("--rent-detail-col", default="Detalle", help="Column used for rent rollup by detail in metric_views.")
    parser.add_argument("--flow-rollup-groupby", default="Flujo,Tipo,Currency", help="Comma-separated groupby columns for generic drilldown metric_views.")
    parser.add_argument("--include-statuses", default="pagado", help="Comma-separated statuses to include in metric_views.")
    parser.add_argument("--noise-floor", default="ARS:5000,USD:10", help="Comma-separated thresholds used in metric_views, e.g. ARS:5000,USD:10")
    args = parser.parse_args()

    run_root = Path(args.run_root) if args.run_root else find_latest_run_root(Path(args.runs_base))
    run_id = args.run_id.strip() or run_root.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Stage start run_root=%s out_dir=%s months=%s", run_root, out_dir, args.months)

    registry = registry_from_specs(default_metric_specs_v1())
    ctx = load_context(run_root=run_root, run_id=run_id, as_of_date=args.as_of_date)

    builder_keys = select_builder_keys(registry)
    leaf = run_leaf_builders(ctx, builder_keys)
    metric_values = derive_default_v1(leaf)
    metric_values = ensure_metric_values_schema(metric_values)

    validation = run_basic_validations(metric_values, registry)

    registry.to_csv(out_dir / METRIC_REGISTRY_FILENAME, index=False)
    metric_values.to_csv(out_dir / METRIC_VALUES_FILENAME, index=False)
    validation.to_csv(out_dir / VALIDATION_REPORT_FILENAME, index=False)

    try:
        metric_values.to_parquet(out_dir / "metric_values.parquet", index=False)
    except Exception as e:
        (out_dir / "parquet_error.txt").write_text(str(e), encoding="utf-8")
        LOG.warning("Optional parquet export failed: %s", e)

    build_wide_views(metric_values, out_dir)
    build_statement_views(metric_values, out_dir)

    include_statuses = tuple(x.strip() for x in args.include_statuses.split(",") if x.strip())
    flow_rollup_groupby = [x.strip() for x in args.flow_rollup_groupby.split(",") if x.strip()]
    noise_floor_by_currency = parse_noise_floor(args.noise_floor)

    manifest = {
        "run_root": str(run_root),
        "run_id": run_id,
        "as_of_date": args.as_of_date,
        "n_registry_rows": int(len(registry)),
        "n_metric_values_rows": int(len(metric_values)),
        "n_validation_rows": int(len(validation)),
        "builder_keys": builder_keys,
    }
    (out_dir / BUILD_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    build_metric_view_exports(
        run_root=run_root,
        metric_values=metric_values,
        registry=registry,
        validation=validation,
        manifest=manifest,
        out_dir=out_dir,
        months=args.months,
        rent_place_col=args.rent_place_col,
        rent_detail_col=args.rent_detail_col,
        flow_rollup_groupby=flow_rollup_groupby,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    )
    build_debt_metric_views(ctx, out_dir)
    views_dir = out_dir / METRIC_VIEWS_DIRNAME
    _build_cash_position_monthly_last12(ctx, views_dir)
    _build_contrib_rollup_views(ctx, views_dir)
    _build_opex_rollup_views(ctx, views_dir)

    build_metric_drilldown_artifacts(
        run_root=run_root,
        out_dir=out_dir,
        run_id=run_id,
        include_statuses=include_statuses,
    )

    if not validation.empty:
        level_counts = validation.get("level", pd.Series(dtype="object")).fillna("UNKNOWN").astype(str).value_counts().to_dict()
        LOG.warning("Validation report has rows=%d levels=%s artifact=%s", len(validation), level_counts, out_dir / VALIDATION_REPORT_FILENAME)
    else:
        LOG.info("Validation report clean artifact=%s", out_dir / VALIDATION_REPORT_FILENAME)

    LOG.info("Stage finish run_id=%s registry_rows=%d metric_values_rows=%d validation_rows=%d manifest=%s", run_id, len(registry), len(metric_values), len(validation), out_dir / BUILD_MANIFEST_FILENAME)


if __name__ == "__main__":
    main()
