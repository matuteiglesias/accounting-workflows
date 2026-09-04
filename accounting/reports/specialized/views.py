from __future__ import annotations

"""Governed professional views for specialized human reports.

This module is the accounting-facing seam of the specialized-report vertical.
Renderers consume only the standardized frames returned here and never classify
ledger rows themselves.
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
    "rent_by_property": ("semantic_split",),
    "rent_monthly_evolution": ("semantic_split",),
    "opex_by_category": ("semantic_split",),
    "taxes_by_property": ("semantic_split",),
    "services_by_property": ("semantic_split",),
    "distributions_vs_rent": ("semantic_audit", "annual_metrics", "semantic_split"),
}


def _path_for(source_key: str, run_root: Path, metrics_dir: Path) -> Path:
    root_key, filename = _SOURCE_LOCATIONS[source_key]
    return (run_root if root_key == "run" else metrics_dir) / filename


def source_paths_for_view(view_key: str, run_root: Path, metrics_dir: Path) -> tuple[tuple[Path, str], ...]:
    keys = _VIEW_REQUIREMENTS[view_key]
    paths = []
    for key in keys:
        path = _path_for(key, run_root, metrics_dir)
        prefix = "run" if _SOURCE_LOCATIONS[key][0] == "run" else "metrics"
        paths.append((path, f"{prefix}/{path.name}"))
    return tuple(paths)


def view_is_available(view_key: str, run_root: Path, metrics_dir: Path) -> bool:
    return all(path.is_file() for path, _ in source_paths_for_view(view_key, run_root, metrics_dir))


def _read(source_key: str, run_root: Path, metrics_dir: Path) -> pd.DataFrame:
    path = _path_for(source_key, run_root, metrics_dir)
    return pd.read_csv(path)


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


def _rent_by_property(split: pd.DataFrame, scope: str) -> SpecializedViewResult:
    frame = _split_base(split)
    frame = frame.loc[
        frame["semantic_bucket"].astype(str).eq("operating_revenue")
        & frame["semantic_subbucket"].astype(str).eq("rent")
    ].copy()
    grouped = (
        frame.groupby(["year", "Currency", "property"], as_index=False, sort=True)["amount_in"]
        .sum()
        .rename(columns={"year": "period", "amount_in": "value"})
    )
    grouped = grouped.loc[grouped["value"].ge(0)]
    out = _rows_from_grouped(
        grouped,
        metric_id="RENT.BY.PROPERTY",
        dimension="property",
        period_basis="annual",
        scope=scope,
        source_filter="semantic_bucket=operating_revenue; semantic_subbucket=rent",
        calculation_rule="annual governed rent amount_in summed by Lugar",
    )
    return SpecializedViewResult(
        out,
        "RENT.BY.PROPERTY",
        "property",
        (("property", "Inmueble / fuente"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _rent_monthly(split: pd.DataFrame, scope: str) -> SpecializedViewResult:
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
        calculation_rule="monthly governed rent amount_in summed across in-scope locations",
    )
    return SpecializedViewResult(
        out,
        "RENT.MONTHLY",
        "month",
        (("period", "Mes"), ("value", "Renta"), ("Currency", "Moneda")),
        (),
    )


def _opex_by_category(split: pd.DataFrame, scope: str) -> SpecializedViewResult:
    frame = _split_base(split)
    frame = frame.loc[frame["semantic_bucket"].astype(str).eq("property_opex")].copy()
    labels = {
        "taxes": "Impuestos",
        "services": "Servicios",
        "maintenance": "Mantenimiento",
        "legal": "Legales",
    }
    frame["category"] = frame["semantic_subbucket"].astype(str).map(labels).fillna(
        frame["semantic_subbucket"].astype(str)
    )
    grouped = (
        frame.groupby(["year", "Currency", "category"], as_index=False, sort=True)["amount_out"]
        .sum()
        .rename(columns={"year": "period", "amount_out": "value"})
    )
    grouped = grouped.loc[grouped["value"].ge(0)]
    out = _rows_from_grouped(
        grouped,
        metric_id="OPEX.BY.CATEGORY",
        dimension="category",
        period_basis="annual",
        scope=scope,
        source_filter="semantic_bucket=property_opex",
        calculation_rule="annual governed property OPEX amount_out summed by semantic_subbucket",
    )
    return SpecializedViewResult(
        out,
        "OPEX.BY.CATEGORY",
        "category",
        (("category", "Categoría"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _opex_by_property(split: pd.DataFrame, scope: str, subbucket: str) -> SpecializedViewResult:
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
        calculation_rule=f"annual governed {subbucket} amount_out summed by Lugar",
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
    split: pd.DataFrame,
    scope: str,
) -> SpecializedViewResult:
    dist = professional_distribution_view(audit, metrics, scope=scope)
    rent = _rent_by_property(split, scope).frame
    dist_total = (
        dist.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
        .assign(concept="Distribuciones registradas")
    )
    rent_total = (
        rent.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
        .assign(concept="Renta reconocida")
    )
    frame = pd.concat([rent_total, dist_total], ignore_index=True, sort=False)
    frame["metric_id"] = "DISTRIBUTIONS.VS.RENT"
    frame["scope"] = scope
    frame["period_basis"] = "annual"
    frame["line_id"] = frame.apply(
        lambda r: f"DISTRIBUTIONS.VS.RENT|{r['period']}|{r['Currency']}|{r['concept']}", axis=1
    )
    frame["source_table"] = "monthly_flow_semantic_split.csv + classification_audit.csv"
    frame["source_filter"] = "governed rent membership compared with governed distribution membership"
    frame["calculation_rule"] = "present independently governed annual totals side by side; no netting"
    return SpecializedViewResult(
        frame,
        "DISTRIBUTIONS.VS.RENT",
        "concept",
        (("concept", "Concepto"), ("value", "Importe"), ("Currency", "Moneda")),
        (),
    )


def _pilot_tax_service(audit: pd.DataFrame, scope: str, category: str) -> SpecializedViewResult:
    rows = audit.loc[audit["semantic_subbucket"].astype(str).eq(category)].copy()
    view = professional_tax_service_payment_view(rows, scope=scope)
    return SpecializedViewResult(
        view,
        "TAX_SERVICE.PAYMENTS.BY_ACTOR",
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
        result = _rent_by_property(frames["semantic_split"], scope)
    elif view_key == "rent_monthly_evolution":
        result = _rent_monthly(frames["semantic_split"], scope)
    elif view_key == "opex_by_category":
        result = _opex_by_category(frames["semantic_split"], scope)
    elif view_key == "taxes_by_property":
        result = _opex_by_property(frames["semantic_split"], scope, "taxes")
    elif view_key == "services_by_property":
        result = _opex_by_property(frames["semantic_split"], scope, "services")
    elif view_key == "distributions_vs_rent":
        result = _comparison_distribution_rent(
            frames["semantic_audit"], frames["annual_metrics"], frames["semantic_split"], scope
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
