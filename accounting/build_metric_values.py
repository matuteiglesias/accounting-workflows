from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .metrics_builders import run_leaf_builders
from .metrics_derive import derive_default_v1
from .metrics_io import MetricsContext, ensure_metric_values_schema, write_table
from .metrics_registry import default_metric_specs_v1, registry_from_specs, normalize_registry
from .metrics_validate import run_basic_validations
from .metrics_views import (
    build_draws_discipline_monthly_last6,
    build_flow_rollup_last_n_months,
    build_income_statement_monthly_last6,
    load_ledger,
    parse_noise_floor,
)


METRIC_VALUES_FILENAME = "metric_values.csv"
METRIC_REGISTRY_FILENAME = "metric_registry.csv"
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


def load_context(run_root: Path, run_id: str, as_of_date: str) -> MetricsContext:
    views_dir = run_root / "views"

    ledger = load_ledger(run_root)
    per_flow = pd.read_csv(run_root / "per_flow_time_long.freq=M.csv")
    daily_cash_position = pd.read_csv(run_root / "daily_cash_position.csv")
    v_contributions_monthly = pd.read_csv(views_dir / "v_contributions_monthly.csv")
    v_opex_category_monthly = pd.read_csv(views_dir / "v_opex_category_monthly.csv")
    party_balance_detailed = load_optional_csv(views_dir / "party_balance_detailed.csv")

    return MetricsContext(
        ledger=ledger,
        per_flow=per_flow,
        daily_cash_position=daily_cash_position,
        v_contributions_monthly=v_contributions_monthly,
        v_opex_category_monthly=v_opex_category_monthly,
        party_balance_detailed=party_balance_detailed,
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

    income_ids = [
        "IS.RENT.TOTAL",
        "IS.CONTRIB.TOTAL",
        "IS.INCOME.TOTAL",
        "IS.OPEX.TOTAL",
        "IS.NET.AFTER_COSTS",
        "IS.DRAWS.PERSONAL",
        "IS.NET.POST_DRAWS",
    ]
    cash_ids = [
        "BS.CASH.FB",
        "BS.CASH.PM",
        "BS.CASH.TOTAL",
    ]

    for name, metric_ids in [
        ("income_statement_y.csv", income_ids),
        ("balance_cash_y.csv", cash_ids),
        ("income_statement_q.csv", income_ids),
        ("balance_cash_q.csv", cash_ids),
    ]:
        grain = "Y" if name.endswith("_y.csv") else "Q"
        sub = mv.loc[(mv["metric_id"].isin(metric_ids)) & (mv["period_grain"] == grain)].copy()
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
        wide.to_csv(out_dir / name, index=False)


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

    print("\n=== BUILD COMPLETE ===")
    print("run_root:", run_root)
    print("run_id:", run_id)
    print("out_dir:", out_dir)
    print("registry rows:", len(registry))
    print("metric_values rows:", len(metric_values))
    print("validation rows:", len(validation))
    if not validation.empty:
        print("\n=== VALIDATION ===")
        print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
