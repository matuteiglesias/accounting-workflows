from __future__ import annotations

"""Governed professional views for specialized human reports.

This module is the accounting-facing seam of the specialized-report vertical.
Renderers consume only standardized professional frames returned here and never
classify ledger rows themselves.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from accounting.reports.charts import (
    professional_distribution_view,
    professional_support_view,
    professional_tax_service_payment_view,
)
from accounting.reports.specialized import round2_views as _round2


TOLERANCE = 0.01


@dataclass(frozen=True)
class SpecializedViewResult:
    frame: pd.DataFrame
    metric_id: str
    dimension: str
    table_columns: tuple[tuple[str, str], ...]
    source_paths: tuple[tuple[Path, str], ...]


_SOURCE_LOCATIONS = {
    "semantic_audit": ("run", "classification_audit.csv"),
    "stakeholder_support": ("run", "monthly_stakeholder_support.csv"),
    "semantic_split": ("run", "monthly_flow_semantic_split.csv"),
    "annual_metrics": ("metrics", "annual_balance_dashboard_metrics.csv"),
}

_VIEW_REQUIREMENTS = {
    "pm_tax_by_actor": ("semantic_audit",),
    "pm_services_by_actor": ("semantic_audit",),
    "pm_support_by_actor": ("stakeholder_support",),
    "distributions_by_recipient": ("semantic_audit", "annual_metrics"),
    "rent_by_property": ("annual_metrics",),
    "rent_monthly_evolution": ("semantic_split", "annual_metrics"),
    "opex_by_category": ("annual_metrics",),
    "taxes_by_property": ("semantic_split", "annual_metrics"),
    "services_by_property": ("semantic_split", "annual_metrics"),
    "distributions_vs_rent": ("semantic_audit", "annual_metrics"),
}


def _path_for(source_key: str, run_root: Path, metrics_dir: Path) -> Path:
    root_key, filename = _SOURCE_LOCATIONS[source_key]
    return (run_root if root_key == "run" else metrics_dir) / filename


def source_paths_for_view(
    view_key: str,
    run_root: Path,
    metrics_dir: Path,
) -> tuple[tuple[Path, str], ...]:
    if view_key in _round2.VIEW_REQUIREMENTS:
        return _round2.source_paths_for_view(view_key, run_root, metrics_dir)
    keys = _VIEW_REQUIREMENTS[view_key]
    paths = []
    for key in keys:
        path = _path_for(key, run_root, metrics_dir)
        prefix = "run" if _SOURCE_LOCATIONS[key][0] == "run" else "metrics"
        paths.append((path, f"{prefix}/{path.name}"))
    return tuple(paths)


def view_is_available(
    view_key: str,
    run_root: Path,
    metrics_dir: Path,
    scope: str = "FBPM",
) -> bool:
    """Return true only when the governed view has an actual reportable population.

    Missing files or empty governed populations mean that this optional report is
    not available for the run. Reconciliation or semantic errors are intentionally
    not swallowed: they must fail the bundle rather than silently hiding a bad
    report.
    """
    if not all(path.is_file() for path, _ in source_paths_for_view(view_key, run_root, metrics_dir)):
        return False
    return not build_specialized_view(
        view_key,
        run_root=run_root,
        metrics_dir=metrics_dir,
        scope=scope,
    ).frame.empty


def _read(source_key: str, run_root: Path, metrics_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_path_for(source_key, run_root, metrics_dir))


def _clean_label(series: pd.Series, fallback: str) -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    return out.mask(out.eq("") | out.eq("nan"), fallback)


def _split_base(split: pd.DataFrame) -> pd.DataFrame:
    required = {
        "period", "Currency", "Lugar", "semantic_bucket", "semantic_subbucket",
        "amount_in", "amount_out",
    }
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"monthly_flow_semantic_split missing required columns: {missing}")
    frame = split.copy()
    frame["period"] = frame["period"].astype(str)
    frame["year"] = frame["period"].str[:4]
    frame["property"] = _clean_label(frame["Lugar"], "Sin ubicación")
    frame["amount_in"] = pd.to_numeric(frame["amount_in"], errors="coerce")
    frame["amount_out"] = pd.to_numeric(frame["amount_out"], errors="coerce")
    return frame


def _rows_from_grouped(
    grouped: pd.DataFrame,
    *,
    metric_id: str,
    dimension: str,
    period_basis: str,
    scope: str,
    source_filter: str,
    calculation_rule: str,
) -> pd.DataFrame:
    rows = grouped.copy()
    rows["metric_id"] = metric_id
    rows["scope"] = scope
    rows["period_basis"] = period_basis
    rows["line_id"] = rows.apply(
        lambda r: f"{metric_id}|{r['period']}|{r['Currency']}|{r[dimension]}", axis=1
    )
    rows["source_table"] = "monthly_flow_semantic_split.csv"
    rows["source_filter"] = source_filter
    rows["calculation_rule"] = calculation_rule
    return rows


def _available_annual_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"metric_id", "period", "Currency", "value"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"annual_balance_dashboard_metrics missing required columns: {missing}")
    frame = metrics.copy()
    if "value_status" in frame.columns:
        frame = frame.loc[frame["value_status"].astype(str).eq("available")].copy()
    frame["period"] = frame["period"].astype(str).str.removesuffix(".0")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.loc[frame["value"].notna()].copy()


def _annual_dimension_metric(
    metrics: pd.DataFrame,
    *,
    metric_id: str,
    dimension_name: str,
    output_dimension: str,
    scope: str,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    frame = _available_annual_metrics(metrics)
    if "dimension_name" not in frame.columns or "dimension_value" not in frame.columns:
        raise ValueError(f"annual metric {metric_id} requires dimension_name/dimension_value")
    frame = frame.loc[
        frame["metric_id"].astype(str).eq(metric_id)
        & frame["dimension_name"].fillna("").astype(str).eq(dimension_name)
    ].copy()
    frame[output_dimension] = _clean_label(frame["dimension_value"], "Sin clasificar")
    if label_map:
        frame[output_dimension] = frame[output_dimension].map(label_map).fillna(frame[output_dimension])
    duplicate = frame.duplicated(["period", "Currency", output_dimension], keep=False)
    if duplicate.any():
        raise ValueError(f"annual metric {metric_id} has duplicate governed dimension rows")
    frame["scope"] = scope
    frame["period_basis"] = "annual"
    frame["line_id"] = frame.apply(
        lambda r: f"{metric_id}|{r['period']}|{r['Currency']}|{r[output_dimension]}", axis=1
    )
    if "source_table" not in frame.columns:
        frame["source_table"] = "annual_balance_dashboard_metrics.csv"
    if "source_filter" not in frame.columns:
        frame["source_filter"] = f"metric_id={metric_id}; dimension_name={dimension_name}"
    if "calculation_rule" not in frame.columns:
        frame["calculation_rule"] = "consume governed annual dashboard metric; no report-side reclassification"
    return frame


def _annual_scalar_metric(metrics: pd.DataFrame, metric_id: str, scope: str) -> pd.DataFrame:
    frame = _available_annual_metrics(metrics)
    frame = frame.loc[frame["metric_id"].astype(str).eq(metric_id)].copy()
    if "dimension_name" in frame.columns:
        frame = frame.loc[frame["dimension_name"].fillna("").astype(str).eq("")].copy()
    duplicate = frame.duplicated(["period", "Currency"], keep=False)
    if duplicate.any():
        raise ValueError(f"annual scalar metric {metric_id} is not singular by period/currency")
    frame["scope"] = scope
    frame["period_basis"] = "annual"
    if "source_table" not in frame.columns:
        frame["source_table"] = "annual_balance_dashboard_metrics.csv"
    if "source_filter" not in frame.columns:
        frame["source_filter"] = f"metric_id={metric_id}"
    if "calculation_rule" not in frame.columns:
        frame["calculation_rule"] = "consume governed annual dashboard metric"
    return frame


def _annual_category_scalar(metrics: pd.DataFrame, subbucket: str, scope: str) -> pd.DataFrame:
    frame = _available_annual_metrics(metrics)
    if "dimension_name" not in frame.columns or "dimension_value" not in frame.columns:
        raise ValueError("annual OPEX category authority requires dimension fields")
    frame = frame.loc[
        frame["metric_id"].astype(str).eq("IS.OPEX.BY_CATEGORY")
        & frame["dimension_name"].fillna("").astype(str).eq("semantic_subbucket")
        & frame["dimension_value"].fillna("").astype(str).eq(subbucket)
    ].copy()
    duplicate = frame.duplicated(["period", "Currency"], keep=False)
    if duplicate.any():
        raise ValueError(f"annual OPEX category {subbucket} is not singular by period/currency")
    frame["scope"] = scope
    return frame


def _assert_year_currency_reconciliation(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
    actual_period_is_monthly: bool,
) -> None:
    work = actual[["period", "Currency", "value"]].copy()
    work["period"] = work["period"].astype(str).str[:4] if actual_period_is_monthly else work["period"].astype(str)
    actual_totals = work.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
    expected_totals = expected[["period", "Currency", "value"]].copy()
    expected_totals["period"] = expected_totals["period"].astype(str).str.removesuffix(".0")
    expected_totals = expected_totals.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
    merged = actual_totals.merge(
        expected_totals,
        on=["period", "Currency"],
        how="outer",
        suffixes=("_actual", "_expected"),
    ).fillna(0.0)
    merged["gap"] = pd.to_numeric(merged["value_actual"], errors="coerce").fillna(0) - pd.to_numeric(
        merged["value_expected"], errors="coerce"
    ).fillna(0)
    bad = merged.loc[merged["gap"].abs().gt(TOLERANCE)]
    if not bad.empty:
        detail = bad[["period", "Currency", "value_actual", "value_expected", "gap"]].to_dict("records")
        raise ValueError(f"specialized view does not reconcile to governed annual authority: {label}: {detail}")


def _rent_by_property(metrics: pd.DataFrame, scope: str) -> SpecializedViewResult:
    out = _annual_dimension_metric(
        metrics,
        metric_id="IS.RENT.BY_PROPERTY",
        dimension_name="Lugar",
        output_dimension="property",
        scope=scope,
    )
    return SpecializedViewResult(
        out,
        "IS.RENT.BY_PROPERTY",
        "property",
        (("property", "Inmueble / fuente"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _rent_monthly(split: pd.DataFrame, metrics: pd.DataFrame, scope: str) -> SpecializedViewResult:
    frame = _split_base(split)
    frame = frame.loc[
        frame["semantic_bucket"].astype(str).eq("operating_revenue")
        & frame["semantic_subbucket"].astype(str).eq("rent")
    ].copy()
    grouped = (
        frame.groupby(["period", "Currency"], as_index=False, sort=True)["amount_in"]
        .sum()
        .rename(columns={"amount_in": "value"})
    )
    grouped["month"] = grouped["period"]
    out = _rows_from_grouped(
        grouped,
        metric_id="RENT.MONTHLY",
        dimension="month",
        period_basis="monthly",
        scope=scope,
        source_filter="semantic_bucket=operating_revenue; semantic_subbucket=rent",
        calculation_rule="monthly governed rent amount_in; annual sum reconciled to IS.RENT.TOTAL",
    )
    _assert_year_currency_reconciliation(
        out,
        _annual_scalar_metric(metrics, "IS.RENT.TOTAL", scope),
        label="monthly rent -> IS.RENT.TOTAL",
        actual_period_is_monthly=True,
    )
    return SpecializedViewResult(
        out,
        "RENT.MONTHLY",
        "month",
        (("period", "Mes"), ("value", "Renta"), ("Currency", "Moneda")),
        (),
    )


def _opex_by_category(metrics: pd.DataFrame, scope: str) -> SpecializedViewResult:
    labels = {
        "taxes": "Impuestos",
        "services": "Servicios",
        "maintenance": "Mantenimiento",
        "legal": "Legales",
    }
    out = _annual_dimension_metric(
        metrics,
        metric_id="IS.OPEX.BY_CATEGORY",
        dimension_name="semantic_subbucket",
        output_dimension="category",
        scope=scope,
        label_map=labels,
    )
    return SpecializedViewResult(
        out,
        "IS.OPEX.BY_CATEGORY",
        "category",
        (("category", "Categoría"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _opex_by_property(
    split: pd.DataFrame,
    metrics: pd.DataFrame,
    scope: str,
    subbucket: str,
) -> SpecializedViewResult:
    frame = _split_base(split)
    frame = frame.loc[
        frame["semantic_bucket"].astype(str).eq("property_opex")
        & frame["semantic_subbucket"].astype(str).eq(subbucket)
    ].copy()
    metric_id = "TAXES.BY.PROPERTY" if subbucket == "taxes" else "SERVICES.BY.PROPERTY"
    grouped = (
        frame.groupby(["year", "Currency", "property"], as_index=False, sort=True)["amount_out"]
        .sum()
        .rename(columns={"year": "period", "amount_out": "value"})
    )
    grouped = grouped.loc[grouped["value"].ge(0)]
    out = _rows_from_grouped(
        grouped,
        metric_id=metric_id,
        dimension="property",
        period_basis="annual",
        scope=scope,
        source_filter=f"semantic_bucket=property_opex; semantic_subbucket={subbucket}",
        calculation_rule=f"annual governed {subbucket} amount_out by Lugar; reconciled to IS.OPEX.BY_CATEGORY",
    )
    _assert_year_currency_reconciliation(
        out,
        _annual_category_scalar(metrics, subbucket, scope),
        label=f"{subbucket} by property -> IS.OPEX.BY_CATEGORY",
        actual_period_is_monthly=False,
    )
    return SpecializedViewResult(
        out,
        metric_id,
        "property",
        (("property", "Inmueble / ubicación"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _comparison_distribution_rent(
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    scope: str,
) -> SpecializedViewResult:
    dist = professional_distribution_view(audit, metrics, scope=scope)
    dist_total = (
        dist.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
        .assign(concept="Distribuciones registradas")
    )
    rent_total = _annual_scalar_metric(metrics, "IS.RENT.TOTAL", scope)[["period", "Currency", "value"]].copy()
    rent_total["concept"] = "Renta reconocida"
    frame = pd.concat([rent_total, dist_total], ignore_index=True, sort=False)
    frame["metric_id"] = "DISTRIBUTIONS.VS.RENT"
    frame["scope"] = scope
    frame["period_basis"] = "annual"
    frame["line_id"] = frame.apply(
        lambda r: f"DISTRIBUTIONS.VS.RENT|{r['period']}|{r['Currency']}|{r['concept']}", axis=1
    )
    frame["source_table"] = "annual_balance_dashboard_metrics.csv + classification_audit.csv"
    frame["source_filter"] = "IS.RENT.TOTAL compared with governed distribution membership"
    frame["calculation_rule"] = "present independently governed annual totals side by side; no netting"
    return SpecializedViewResult(
        frame,
        "DISTRIBUTIONS.VS.RENT",
        "concept",
        (("concept", "Concepto"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _pilot_tax_service(audit: pd.DataFrame, scope: str, category: str) -> SpecializedViewResult:
    required = {"semantic_subbucket", "Box"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(f"PM tax/service report source missing fields: {missing}")
    rows = audit.loc[
        audit["semantic_subbucket"].astype(str).eq(category)
        & audit["Box"].astype(str).eq("Property Management")
    ].copy()
    view = professional_tax_service_payment_view(rows, scope=scope)
    metric_id = "PM.TAXES.COVERAGE.BY_ACTOR" if category == "taxes" else "PM.SERVICES.COVERAGE.BY_ACTOR"
    if not view.empty:
        view["metric_id"] = metric_id
        view["source_filter"] = (
            f"Box=Property Management; semantic_subbucket={category}; "
            "cash/payment leg; economic direct mirror excluded"
        )
        view["calculation_rule"] = (
            f"annual PM {category} applications grouped by identified funding actor or paying PM Box"
        )
        view["line_id"] = view.apply(
            lambda r: f"{metric_id}|{r['period']}|{r['Currency']}|{r['funding_actor']}", axis=1
        )
    return SpecializedViewResult(
        view,
        metric_id,
        "funding_actor",
        (("funding_actor", "Actor / fuente"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _pilot_support(support: pd.DataFrame, scope: str) -> SpecializedViewResult:
    view = professional_support_view(support, scope=scope)
    return SpecializedViewResult(
        view,
        "SUPPORT.BY_ACTOR",
        "funding_actor",
        (("funding_actor", "Actor"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _pilot_distribution(audit: pd.DataFrame, metrics: pd.DataFrame, scope: str) -> SpecializedViewResult:
    view = professional_distribution_view(audit, metrics, scope=scope)
    return SpecializedViewResult(
        view,
        "DIST.BY_RECIPIENT",
        "recipient",
        (("recipient", "Receptor"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def build_specialized_view(
    view_key: str,
    *,
    run_root: Path,
    metrics_dir: Path,
    scope: str,
) -> SpecializedViewResult:
    if view_key in _round2.VIEW_REQUIREMENTS:
        result = _round2.build_view(
            view_key,
            run_root=run_root,
            metrics_dir=metrics_dir,
            scope=scope,
        )
        return SpecializedViewResult(
            result.frame,
            result.metric_id,
            result.dimension,
            result.table_columns,
            source_paths_for_view(view_key, run_root, metrics_dir),
        )

    loaders: dict[str, Callable[[], pd.DataFrame]] = {
        key: (lambda source_key=key: _read(source_key, run_root, metrics_dir))
        for key in _VIEW_REQUIREMENTS[view_key]
    }
    frames = {key: loader() for key, loader in loaders.items()}
    sources = source_paths_for_view(view_key, run_root, metrics_dir)

    if view_key == "pm_tax_by_actor":
        result = _pilot_tax_service(frames["semantic_audit"], scope, "taxes")
    elif view_key == "pm_services_by_actor":
        result = _pilot_tax_service(frames["semantic_audit"], scope, "services")
    elif view_key == "pm_support_by_actor":
        result = _pilot_support(frames["stakeholder_support"], scope)
    elif view_key == "distributions_by_recipient":
        result = _pilot_distribution(frames["semantic_audit"], frames["annual_metrics"], scope)
    elif view_key == "rent_by_property":
        result = _rent_by_property(frames["annual_metrics"], scope)
    elif view_key == "rent_monthly_evolution":
        result = _rent_monthly(frames["semantic_split"], frames["annual_metrics"], scope)
    elif view_key == "opex_by_category":
        result = _opex_by_category(frames["annual_metrics"], scope)
    elif view_key == "taxes_by_property":
        result = _opex_by_property(frames["semantic_split"], frames["annual_metrics"], scope, "taxes")
    elif view_key == "services_by_property":
        result = _opex_by_property(frames["semantic_split"], frames["annual_metrics"], scope, "services")
    elif view_key == "distributions_vs_rent":
        result = _comparison_distribution_rent(
            frames["semantic_audit"], frames["annual_metrics"], scope
        )
    else:
        raise KeyError(f"unknown specialized report view: {view_key}")

    return SpecializedViewResult(
        result.frame,
        result.metric_id,
        result.dimension,
        result.table_columns,
        sources,
    )
