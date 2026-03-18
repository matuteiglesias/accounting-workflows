from __future__ import annotations

import pandas as pd

from .metrics_io import ensure_metric_values_schema


def derive_sum_components(
    metric_values: pd.DataFrame,
    *,
    metric_id: str,
    component_ids: list[str],
    source_layer: str = "derived",
    build_status: str = "ok",
) -> pd.DataFrame:
    mv = ensure_metric_values_schema(metric_values)

    wide = (
        mv.loc[mv["metric_id"].isin(component_ids)]
        .pivot_table(
            index=["period_grain", "period", "currency", "run_id", "as_of_date"],
            columns="metric_id",
            values="value",
            aggfunc="first",
        )
        .fillna(0.0)
        .reset_index()
    )

    if wide.empty:
        return ensure_metric_values_schema(pd.DataFrame(columns=mv.columns))

    for comp in component_ids:
        if comp not in wide.columns:
            wide[comp] = 0.0

    wide["value"] = wide[component_ids].sum(axis=1)
    wide["metric_id"] = metric_id
    wide["source_layer"] = source_layer
    wide["build_status"] = build_status
    wide["build_detail"] = f"sum_components({', '.join(component_ids)})"

    out = wide[
        [
            "metric_id",
            "period_grain",
            "period",
            "currency",
            "value",
            "run_id",
            "as_of_date",
            "source_layer",
            "build_status",
            "build_detail",
        ]
    ]
    return ensure_metric_values_schema(out)


def derive_formula_subtract(
    metric_values: pd.DataFrame,
    *,
    metric_id: str,
    minuend_id: str,
    subtrahend_ids: list[str],
    source_layer: str = "derived",
    build_status: str = "ok",
) -> pd.DataFrame:
    mv = ensure_metric_values_schema(metric_values)
    needed = [minuend_id] + subtrahend_ids

    wide = (
        mv.loc[mv["metric_id"].isin(needed)]
        .pivot_table(
            index=["period_grain", "period", "currency", "run_id", "as_of_date"],
            columns="metric_id",
            values="value",
            aggfunc="first",
        )
        .fillna(0.0)
        .reset_index()
    )

    if wide.empty:
        return ensure_metric_values_schema(pd.DataFrame(columns=mv.columns))

    if minuend_id not in wide.columns:
        wide[minuend_id] = 0.0
    for sid in subtrahend_ids:
        if sid not in wide.columns:
            wide[sid] = 0.0

    wide["value"] = wide[minuend_id] - wide[subtrahend_ids].sum(axis=1)
    wide["metric_id"] = metric_id
    wide["source_layer"] = source_layer
    wide["build_status"] = build_status
    wide["build_detail"] = f"{minuend_id} - ({', '.join(subtrahend_ids)})"

    out = wide[
        [
            "metric_id",
            "period_grain",
            "period",
            "currency",
            "value",
            "run_id",
            "as_of_date",
            "source_layer",
            "build_status",
            "build_detail",
        ]
    ]
    return ensure_metric_values_schema(out)


def _append_and_dedup(current: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([current, new_df], ignore_index=True)
    out = ensure_metric_values_schema(out)
    out = out.drop_duplicates(
        subset=["metric_id", "period_grain", "period", "currency", "run_id", "as_of_date"],
        keep="last",
    ).reset_index(drop=True)
    return out


def derive_default_v1(metric_values: pd.DataFrame) -> pd.DataFrame:
    current = ensure_metric_values_schema(metric_values)

    current = _append_and_dedup(
        current,
        derive_sum_components(
            current,
            metric_id="IS.INCOME.TOTAL",
            component_ids=["IS.RENT.TOTAL", "IS.CONTRIB.TOTAL"],
        ),
    )

    current = _append_and_dedup(
        current,
        derive_formula_subtract(
            current,
            metric_id="IS.NET.AFTER_COSTS",
            minuend_id="IS.INCOME.TOTAL",
            subtrahend_ids=["IS.OPEX.TOTAL"],
        ),
    )

    current = _append_and_dedup(
        current,
        derive_formula_subtract(
            current,
            metric_id="IS.NET.POST_DRAWS",
            minuend_id="IS.NET.AFTER_COSTS",
            subtrahend_ids=["IS.DRAWS.PERSONAL"],
        ),
    )

    current = _append_and_dedup(
        current,
        derive_sum_components(
            current,
            metric_id="BS.CASH.TOTAL",
            component_ids=["BS.CASH.FB", "BS.CASH.PM"],
        ),
    )

    return current
