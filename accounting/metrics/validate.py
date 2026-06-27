from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd

from .io import ensure_metric_values_schema
from .registry import normalize_registry

ALLOWED_METRIC_TYPES = {"flow", "stock", "derived", "manual", "qa", "unknown"}
ALLOWED_ECONOMIC_ROLES = {
    "operating",
    "funding",
    "distribution",
    "cash",
    "debt",
    "claim",
    "coverage",
    "qa",
    "unknown",
}
ALLOWED_NAMESPACE_TARGETS = {
    "IS",
    "CF",
    "BS",
    "ID",
    "FUND",
    "DIST",
    "COV",
    "HUMAN",
    "LEGACY",
    "UNKNOWN",
}
ALLOWED_MIGRATION_STATUSES = {
    "keep",
    "alias",
    "split",
    "create",
    "deprecate",
    "investigate",
    "legacy",
}
SEMANTIC_COLUMNS = [
    "metric_type",
    "economic_role",
    "namespace_target",
    "migration_status",
    "legacy_warning",
]

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


def check_registry_semantic_metadata(registry_df: pd.DataFrame) -> pd.DataFrame:
    reg = normalize_registry(registry_df)

    issues: list[ValidationIssue] = []

    missing_cols = [c for c in SEMANTIC_COLUMNS if c not in reg.columns]
    if missing_cols:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="registry_semantic_columns_present",
                message=f"Registry missing semantic metadata columns: {missing_cols}",
                n_rows=len(reg),
            )
        )
        return _issues_to_df(issues)

    active = reg.loc[reg["status"].astype(str).str.lower() == "active"].copy()

    blank_type = active.loc[active["metric_type"].astype(str).str.strip().isin(["", "unknown"])]
    if not blank_type.empty:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="registry_metric_type_present",
                message="Active metrics with empty/unknown metric_type found.",
                n_rows=len(blank_type),
            )
        )

    invalid_type = active.loc[~active["metric_type"].isin(ALLOWED_METRIC_TYPES)]
    if not invalid_type.empty:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="registry_metric_type_allowed",
                message=f"Invalid metric_type found. Allowed={sorted(ALLOWED_METRIC_TYPES)}",
                n_rows=len(invalid_type),
            )
        )

    invalid_role = active.loc[~active["economic_role"].isin(ALLOWED_ECONOMIC_ROLES)]
    if not invalid_role.empty:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="registry_economic_role_allowed",
                message=f"Invalid economic_role found. Allowed={sorted(ALLOWED_ECONOMIC_ROLES)}",
                n_rows=len(invalid_role),
            )
        )

    invalid_namespace = active.loc[~active["namespace_target"].isin(ALLOWED_NAMESPACE_TARGETS)]
    if not invalid_namespace.empty:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="registry_namespace_target_allowed",
                message=f"Invalid namespace_target found. Allowed={sorted(ALLOWED_NAMESPACE_TARGETS)}",
                n_rows=len(invalid_namespace),
            )
        )

    invalid_migration = active.loc[~active["migration_status"].isin(ALLOWED_MIGRATION_STATUSES)]
    if not invalid_migration.empty:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="registry_migration_status_allowed",
                message=f"Invalid migration_status found. Allowed={sorted(ALLOWED_MIGRATION_STATUSES)}",
                n_rows=len(invalid_migration),
            )
        )

    legacy_without_warning = active.loc[
        active["migration_status"].isin(["legacy", "deprecate"])
        & (active["legacy_warning"].astype(str).str.strip() == "")
    ]
    if not legacy_without_warning.empty:
        issues.append(
            ValidationIssue(
                level="warning",
                check_name="registry_legacy_warning_present",
                message="Legacy/deprecate metrics without legacy_warning found.",
                n_rows=len(legacy_without_warning),
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


def check_formula_add_subtract_identity(
    metric_values: pd.DataFrame,
    *,
    target_metric_id: str,
    addend_ids: list[str],
    subtrahend_ids: list[str],
    tolerance: float = 1e-9,
    check_name: str = "formula_add_subtract_identity",
) -> pd.DataFrame:
    mv = ensure_metric_values_schema(metric_values)
    needed = [target_metric_id] + addend_ids + subtrahend_ids

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

    for needed_id in addend_ids + subtrahend_ids:
        if needed_id not in wide.columns:
            wide[needed_id] = 0.0

    expected = wide[addend_ids].sum(axis=1) - wide[subtrahend_ids].sum(axis=1)
    diff = wide[target_metric_id] - expected
    bad = wide[diff.abs() > tolerance]

    issues: list[ValidationIssue] = []
    if not bad.empty:
        issues.append(
            ValidationIssue(
                level="error",
                check_name=check_name,
                message=f"{target_metric_id} does not satisfy add/subtract formula.",
                n_rows=len(bad),
            )
        )
    return _issues_to_df(issues)


def run_basic_validations(metric_values: pd.DataFrame, registry_df: pd.DataFrame) -> pd.DataFrame:
    present_metric_ids = set(ensure_metric_values_schema(metric_values)["metric_id"].astype(str).tolist())

    def _check_if_any_present(required_metric_ids: list[str], check_df: pd.DataFrame) -> pd.DataFrame:
        if present_metric_ids.intersection(required_metric_ids):
            return check_df
        return pd.DataFrame(columns=["level", "check_name", "message", "n_rows"])

    checks = [
        check_metric_values_unique(metric_values),
        check_registry_metric_ids_unique(registry_df),
        check_leaf_builder_keys_present(registry_df),
        check_metric_ids_known(metric_values, registry_df),
        check_registry_semantic_metadata(registry_df),
        check_sum_identity(metric_values, total_metric_id="IS.RENT.TOTAL", component_ids=["IS.RENT.CABA", "IS.RENT.TORCUATO"], check_name="is_rent_total"),
        check_sum_identity(metric_values, total_metric_id="IS.INCOME.TOTAL", component_ids=["IS.RENT.TOTAL", "IS.CONTRIB.TOTAL"], check_name="is_income_total"),
        check_formula_subtract_identity(metric_values, target_metric_id="IS.NET.AFTER_COSTS", minuend_id="IS.INCOME.TOTAL", subtrahend_ids=["IS.OPEX.TOTAL"], check_name="is_net_after_costs"),
        check_formula_subtract_identity(metric_values, target_metric_id="IS.NET.POST_DRAWS", minuend_id="IS.NET.AFTER_COSTS", subtrahend_ids=["IS.DRAWS.PERSONAL"], check_name="is_net_post_draws"),
        check_sum_identity(metric_values, total_metric_id="IS.REVENUE.TOTAL", component_ids=["IS.RENT.TOTAL"], check_name="is_revenue_total_shadow"),
        check_formula_subtract_identity(metric_values, target_metric_id="IS.NET.OPERATING", minuend_id="IS.REVENUE.TOTAL", subtrahend_ids=["IS.OPEX.TOTAL"], check_name="is_net_operating_shadow"),
        check_sum_identity(metric_values, total_metric_id="FUND.CONTRIB.TOTAL", component_ids=["IS.CONTRIB.TOTAL"], check_name="fund_contrib_total_shadow"),
        check_sum_identity(metric_values, total_metric_id="DIST.DRAWS.PERSONAL", component_ids=["IS.DRAWS.PERSONAL"], check_name="dist_draws_personal_shadow"),
        check_formula_add_subtract_identity(
            metric_values,
            target_metric_id="COV.NET.AFTER_DRAWS",
            addend_ids=["IS.NET.OPERATING", "FUND.CONTRIB.TOTAL"],
            subtrahend_ids=["DIST.DRAWS.PERSONAL"],
            check_name="cov_net_after_draws_shadow",
        ),
        check_sum_identity(metric_values, total_metric_id="BS.CASH.TOTAL", component_ids=["BS.CASH.FB", "BS.CASH.PM"], check_name="bs_cash_total"),
        _check_if_any_present(
            ["BS.DEBT.TOTAL.OPEN", "BS.DEBT.PM_TO_MI.OPEN", "BS.DEBT.PM_TO_PRIMOS.OPEN"],
            check_sum_identity(
                metric_values,
                total_metric_id="BS.DEBT.TOTAL.OPEN",
                component_ids=["BS.DEBT.PM_TO_MI.OPEN", "BS.DEBT.PM_TO_PRIMOS.OPEN"],
                check_name="bs_debt_total_open",
            ),
        ),
        _check_if_any_present(
            ["BS.DEBT.NET_PM_POSITION", "BS.DEBT.TOTAL.OPEN", "BS.CLAIM.ALE_TO_PM.OPEN"],
            check_formula_subtract_identity(
                metric_values,
                target_metric_id="BS.DEBT.NET_PM_POSITION",
                minuend_id="BS.DEBT.TOTAL.OPEN",
                subtrahend_ids=["BS.CLAIM.ALE_TO_PM.OPEN"],
                check_name="bs_debt_net_pm_position",
            ),
        ),
        _check_if_any_present(
            ["BS.DEBT.TOTAL.OPEN", "BS.DEBT.PRINCIPAL.OPEN", "BS.DEBT.INTEREST.OPEN"],
            check_sum_identity(
                metric_values,
                total_metric_id="BS.DEBT.TOTAL.OPEN",
                component_ids=["BS.DEBT.PRINCIPAL.OPEN", "BS.DEBT.INTEREST.OPEN"],
                check_name="bs_debt_total_components",
            ),
        ),
    ]
    out = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    return out.reset_index(drop=True)
