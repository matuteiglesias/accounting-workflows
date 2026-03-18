from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd

from .metrics_io import ensure_metric_values_schema
from .metrics_registry import normalize_registry


@dataclass
class ValidationIssue:
    level: str
    check_name: str
    message: str
    n_rows: int = 0

    def to_record(self) -> dict:
        return asdict(self)


def _issues_to_df(issues: list[ValidationIssue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["level", "check_name", "message", "n_rows"])
    return pd.DataFrame([x.to_record() for x in issues])


def check_metric_values_unique(metric_values: pd.DataFrame) -> pd.DataFrame:
    mv = ensure_metric_values_schema(metric_values)
    keys = ["metric_id", "period_grain", "period", "currency", "run_id", "as_of_date"]
    dup = mv[mv.duplicated(keys, keep=False)]

    issues: list[ValidationIssue] = []
    if not dup.empty:
        issues.append(
            ValidationIssue(
                level="error",
                check_name="metric_values_unique",
                message="Duplicate metric values found for unique key.",
                n_rows=len(dup),
            )
        )
    return _issues_to_df(issues)


def check_registry_metric_ids_unique(registry_df: pd.DataFrame) -> pd.DataFrame:
    reg = normalize_registry(registry_df)
    dup = reg[reg.duplicated(["metric_id"], keep=False)]

    issues: list[ValidationIssue] = []
    if not dup.empty:
        issues.append(
            ValidationIssue(
                level="error",
                check_name="registry_metric_ids_unique",
                message="Duplicate metric_id found in registry.",
                n_rows=len(dup),
            )
        )
    return _issues_to_df(issues)


def check_leaf_builder_keys_present(registry_df: pd.DataFrame) -> pd.DataFrame:
    reg = normalize_registry(registry_df)
    bad = reg[(reg["is_leaf"]) & (reg["builder_key"].astype(str).str.strip() == "")]

    issues: list[ValidationIssue] = []
    if not bad.empty:
        issues.append(
            ValidationIssue(
                level="error",
                check_name="leaf_builder_keys_present",
                message="Leaf metrics without builder_key found in registry.",
                n_rows=len(bad),
            )
        )
    return _issues_to_df(issues)


def check_metric_ids_known(metric_values: pd.DataFrame, registry_df: pd.DataFrame) -> pd.DataFrame:
    mv = ensure_metric_values_schema(metric_values)
    reg = normalize_registry(registry_df)

    unknown = mv.loc[~mv["metric_id"].isin(reg["metric_id"])]
    issues: list[ValidationIssue] = []
    if not unknown.empty:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="metric_ids_known",
                message="metric_values contains metric_id not present in registry.",
                n_rows=len(unknown),
            )
        )
    return _issues_to_df(issues)


def check_sum_identity(
    metric_values: pd.DataFrame,
    *,
    total_metric_id: str,
    component_ids: list[str],
    tolerance: float = 1e-9,
    check_name: str = "sum_identity",
) -> pd.DataFrame:
    mv = ensure_metric_values_schema(metric_values)
    needed = [total_metric_id] + component_ids

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

    if total_metric_id not in wide.columns:
        return _issues_to_df(
            [ValidationIssue("warning", check_name, f"Total metric missing: {total_metric_id}", 0)]
        )

    for comp in component_ids:
        if comp not in wide.columns:
            wide[comp] = 0.0

    diff = wide[total_metric_id] - wide[component_ids].sum(axis=1)
    bad = wide[diff.abs() > tolerance]

    issues: list[ValidationIssue] = []
    if not bad.empty:
        issues.append(
            ValidationIssue(
                level="error",
                check_name=check_name,
                message=f"{total_metric_id} does not equal sum of components.",
                n_rows=len(bad),
            )
        )
    return _issues_to_df(issues)


def check_formula_subtract_identity(
    metric_values: pd.DataFrame,
    *,
    target_metric_id: str,
    minuend_id: str,
    subtrahend_ids: list[str],
    tolerance: float = 1e-9,
    check_name: str = "formula_subtract_identity",
) -> pd.DataFrame:
    mv = ensure_metric_values_schema(metric_values)
    needed = [target_metric_id, minuend_id] + subtrahend_ids

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

    if target_metric_id not in wide.columns:
        return _issues_to_df(
            [ValidationIssue("warning", check_name, f"Target metric missing: {target_metric_id}", 0)]
        )

    if minuend_id not in wide.columns:
        wide[minuend_id] = 0.0
    for sid in subtrahend_ids:
        if sid not in wide.columns:
            wide[sid] = 0.0

    expected = wide[minuend_id] - wide[subtrahend_ids].sum(axis=1)
    diff = wide[target_metric_id] - expected
    bad = wide[diff.abs() > tolerance]

    issues: list[ValidationIssue] = []
    if not bad.empty:
        issues.append(
            ValidationIssue(
                level="error",
                check_name=check_name,
                message=f"{target_metric_id} does not satisfy subtraction formula.",
                n_rows=len(bad),
            )
        )
    return _issues_to_df(issues)


def run_basic_validations(metric_values: pd.DataFrame, registry_df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        check_metric_values_unique(metric_values),
        check_registry_metric_ids_unique(registry_df),
        check_leaf_builder_keys_present(registry_df),
        check_metric_ids_known(metric_values, registry_df),
        check_sum_identity(metric_values, total_metric_id="IS.RENT.TOTAL", component_ids=["IS.RENT.CABA", "IS.RENT.TORCUATO"], check_name="is_rent_total"),
        check_sum_identity(metric_values, total_metric_id="IS.INCOME.TOTAL", component_ids=["IS.RENT.TOTAL", "IS.CONTRIB.TOTAL"], check_name="is_income_total"),
        check_formula_subtract_identity(metric_values, target_metric_id="IS.NET.AFTER_COSTS", minuend_id="IS.INCOME.TOTAL", subtrahend_ids=["IS.OPEX.TOTAL"], check_name="is_net_after_costs"),
        check_formula_subtract_identity(metric_values, target_metric_id="IS.NET.POST_DRAWS", minuend_id="IS.NET.AFTER_COSTS", subtrahend_ids=["IS.DRAWS.PERSONAL"], check_name="is_net_post_draws"),
        check_sum_identity(metric_values, total_metric_id="BS.CASH.TOTAL", component_ids=["BS.CASH.FB", "BS.CASH.PM"], check_name="bs_cash_total"),
    ]
    out = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    return out.reset_index(drop=True)
