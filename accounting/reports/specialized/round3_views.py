from __future__ import annotations

"""Round-3 governed professional views for specialized human reports.

These builders only regroup existing governed artifacts. They do not classify
ledger transactions, infer legal responsibility, or manufacture cash/debt.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from accounting.reports.charts import professional_distribution_view


TOLERANCE = 0.01


@dataclass(frozen=True)
class Round3ViewResult:
    frame: pd.DataFrame
    metric_id: str
    dimension: str
    table_columns: tuple[tuple[str, str], ...]


SOURCE_LOCATIONS = {
    "annual_metrics": ("metrics", "annual_balance_dashboard_metrics.csv"),
    "semantic_split": ("run", "monthly_flow_semantic_split.csv"),
    "semantic_audit": ("run", "classification_audit.csv"),
    "stakeholder_support": ("run", "monthly_stakeholder_support.csv"),
    "treasury_flow": ("run", "monthly_box_treasury_flow.csv"),
    "cash_accountability": ("run", "monthly_cash_accountability.csv"),
}

VIEW_REQUIREMENTS = {
    "rent_annual_comparison": ("annual_metrics",),
    "distributions_by_year": ("semantic_audit", "annual_metrics"),
    "opex_monthly_evolution": ("semantic_split", "annual_metrics"),
    "maintenance_by_property": ("semantic_split", "annual_metrics"),
    "legal_costs_by_property": ("semantic_split", "annual_metrics"),
    "support_by_obligation_category": ("stakeholder_support",),
    "support_by_funding_channel": ("stakeholder_support",),
    "support_by_settlement_nature": ("stakeholder_support",),
    "physical_inflows_by_category": ("treasury_flow", "cash_accountability"),
    "physical_outflows_by_category": ("treasury_flow", "cash_accountability"),
    "cash_residuals": ("cash_accountability",),
}


def _path_for(source_key: str, run_root: Path, metrics_dir: Path) -> Path:
    root_key, filename = SOURCE_LOCATIONS[source_key]
    return (run_root if root_key == "run" else metrics_dir) / filename


def source_paths_for_view(
    view_key: str,
    run_root: Path,
    metrics_dir: Path,
) -> tuple[tuple[Path, str], ...]:
    paths: list[tuple[Path, str]] = []
    for key in VIEW_REQUIREMENTS[view_key]:
        path = _path_for(key, run_root, metrics_dir)
        prefix = "run" if SOURCE_LOCATIONS[key][0] == "run" else "metrics"
        paths.append((path, f"{prefix}/{path.name}"))
    return tuple(paths)


def _read(source_key: str, run_root: Path, metrics_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_path_for(source_key, run_root, metrics_dir))


def _require(frame: pd.DataFrame, columns: set[str], source_name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} missing required round-3 columns: {missing}")


def _text(series: pd.Series, fallback: str) -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    return out.mask(out.eq("") | out.eq("nan"), fallback)


def _scope_boxes(scope: str) -> set[str]:
    if scope == "FBPM":
        return {"Family Business", "Property Management"}
    if scope in {"Family Business", "Property Management"}:
        return {scope}
    raise ValueError(f"unsupported Box scope for specialized report: {scope}")


def _available_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    _require(metrics, {"metric_id", "period", "Currency", "value"}, "annual_balance_dashboard_metrics.csv")
    out = metrics.copy()
    if "value_status" in out.columns:
        out = out.loc[out["value_status"].astype(str).eq("available")].copy()
    out["period"] = out["period"].astype(str).str.removesuffix(".0")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.loc[out["value"].notna()].copy()


def _scalar_metric(metrics: pd.DataFrame, metric_id: str) -> pd.DataFrame:
    frame = _available_metrics(metrics)
    frame = frame.loc[frame["metric_id"].astype(str).eq(metric_id)].copy()
    if "dimension_name" in frame.columns:
        frame = frame.loc[frame["dimension_name"].fillna("").astype(str).eq("")].copy()
    if frame.duplicated(["period", "Currency"], keep=False).any():
        raise ValueError(f"annual scalar metric is not singular by period/currency: {metric_id}")
    return frame


def _category_metric(metrics: pd.DataFrame, category: str) -> pd.DataFrame:
    frame = _available_metrics(metrics)
    _require(frame, {"dimension_name", "dimension_value"}, "annual_balance_dashboard_metrics.csv")
    frame = frame.loc[
        frame["metric_id"].astype(str).eq("IS.OPEX.BY_CATEGORY")
        & frame["dimension_name"].fillna("").astype(str).eq("semantic_subbucket")
        & frame["dimension_value"].fillna("").astype(str).eq(category)
    ].copy()
    if frame.duplicated(["period", "Currency"], keep=False).any():
        raise ValueError(f"annual OPEX category is not singular by period/currency: {category}")
    return frame


def _assert_year_currency(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    a = actual[["period", "Currency", "value"]].copy()
    a["period"] = a["period"].astype(str).str[:4]
    a = a.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
    e = expected[["period", "Currency", "value"]].copy()
    e["period"] = e["period"].astype(str).str[:4]
    e = e.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
    merged = a.merge(e, on=["period", "Currency"], how="outer", suffixes=("_actual", "_expected")).fillna(0.0)
    merged["gap"] = merged["value_actual"] - merged["value_expected"]
    bad = merged.loc[merged["gap"].abs().gt(TOLERANCE)]
    if not bad.empty:
        raise ValueError(f"round-3 reconciliation failed: {label}: {bad.to_dict('records')}")


def _decorate(
    frame: pd.DataFrame,
    *,
    metric_id: str,
    dimension: str,
    scope: str,
    period_basis: str,
    source_table: str,
    source_filter: str,
    calculation_rule: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["metric_id"] = metric_id
    out["scope"] = scope
    out["period_basis"] = period_basis
    out["line_id"] = out.apply(
        lambda row: f"{metric_id}|{row['period']}|{row['Currency']}|{row[dimension]}", axis=1
    )
    out["source_table"] = source_table
    out["source_filter"] = source_filter
    out["calculation_rule"] = calculation_rule
    return out


def _rent_annual(metrics: pd.DataFrame, scope: str) -> Round3ViewResult:
    frame = _scalar_metric(metrics, "IS.RENT.TOTAL")[["period", "Currency", "value"]].copy()
    frame["year"] = frame["period"]
    frame = frame.loc[frame["value"].ge(0)].copy()
    out = _decorate(
        frame,
        metric_id="RENT.ANNUAL.COMPARISON",
        dimension="year",
        scope=scope,
        period_basis="annual",
        source_table="annual_balance_dashboard_metrics.csv",
        source_filter="metric_id=IS.RENT.TOTAL; scalar governed annual metric",
        calculation_rule="present governed annual rent totals by year; native currencies separate",
    )
    return Round3ViewResult(out, "RENT.ANNUAL.COMPARISON", "year", (("year", "Año"), ("value", "Renta"), ("Currency", "Moneda")))


def _distributions_annual(audit: pd.DataFrame, metrics: pd.DataFrame, scope: str) -> Round3ViewResult:
    view = professional_distribution_view(audit, metrics, scope=scope)
    if view.empty:
        return Round3ViewResult(pd.DataFrame(), "DIST.ANNUAL", "year", (("year", "Año"), ("value", "Distribuciones"), ("Currency", "Moneda")))
    grouped = view.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
    grouped["year"] = grouped["period"].astype(str)
    out = _decorate(
        grouped,
        metric_id="DIST.ANNUAL",
        dimension="year",
        scope=scope,
        period_basis="annual",
        source_table="classification_audit.csv + annual_balance_dashboard_metrics.csv",
        source_filter="governed distribution membership from professional_distribution_view",
        calculation_rule="sum governed recipient slices by annual period; no entitlement or netting inference",
    )
    return Round3ViewResult(out, "DIST.ANNUAL", "year", (("year", "Año"), ("value", "Distribuciones"), ("Currency", "Moneda")))


def _split_base(split: pd.DataFrame) -> pd.DataFrame:
    _require(split, {"period", "Currency", "Lugar", "semantic_bucket", "semantic_subbucket", "amount_out"}, "monthly_flow_semantic_split.csv")
    frame = split.copy()
    frame["period"] = frame["period"].astype(str)
    frame["year"] = frame["period"].str[:4]
    frame["property"] = _text(frame["Lugar"], "Sin ubicación")
    frame["amount_out"] = pd.to_numeric(frame["amount_out"], errors="coerce")
    if frame["amount_out"].isna().any():
        raise ValueError("semantic split contains unavailable amount_out")
    return frame


def _opex_monthly(split: pd.DataFrame, metrics: pd.DataFrame, scope: str) -> Round3ViewResult:
    frame = _split_base(split)
    frame = frame.loc[frame["semantic_bucket"].astype(str).eq("property_opex")].copy()
    grouped = frame.groupby(["period", "Currency"], as_index=False, sort=True)["amount_out"].sum().rename(columns={"amount_out": "value"})
    grouped = grouped.loc[grouped["value"].ge(0)].copy()
    grouped["month"] = grouped["period"]
    _assert_year_currency(grouped, _scalar_metric(metrics, "IS.OPEX.PROPERTY"), "monthly OPEX -> IS.OPEX.PROPERTY")
    out = _decorate(
        grouped,
        metric_id="OPEX.MONTHLY",
        dimension="month",
        scope=scope,
        period_basis="monthly",
        source_table="monthly_flow_semantic_split.csv",
        source_filter="semantic_bucket=property_opex",
        calculation_rule="monthly governed property OPEX amount_out; annual sum reconciled to IS.OPEX.PROPERTY",
    )
    return Round3ViewResult(out, "OPEX.MONTHLY", "month", (("period", "Mes"), ("value", "OPEX"), ("Currency", "Moneda")))


def _category_by_property(split: pd.DataFrame, metrics: pd.DataFrame, scope: str, category: str) -> Round3ViewResult:
    frame = _split_base(split)
    frame = frame.loc[
        frame["semantic_bucket"].astype(str).eq("property_opex")
        & frame["semantic_subbucket"].astype(str).eq(category)
    ].copy()
    grouped = frame.groupby(["year", "Currency", "property"], as_index=False, sort=True)["amount_out"].sum().rename(columns={"year": "period", "amount_out": "value"})
    grouped = grouped.loc[grouped["value"].gt(TOLERANCE)].copy()
    expected = _category_metric(metrics, category)
    _assert_year_currency(grouped, expected, f"{category} by property -> IS.OPEX.BY_CATEGORY")
    metric_id = "MAINTENANCE.BY.PROPERTY" if category == "maintenance" else "LEGAL.BY.PROPERTY"
    out = _decorate(
        grouped,
        metric_id=metric_id,
        dimension="property",
        scope=scope,
        period_basis="annual",
        source_table="monthly_flow_semantic_split.csv",
        source_filter=f"semantic_bucket=property_opex; semantic_subbucket={category}",
        calculation_rule=f"annual governed {category} amount_out by Lugar; reconciled to IS.OPEX.BY_CATEGORY",
    )
    label = "Mantenimiento" if category == "maintenance" else "Gastos legales"
    return Round3ViewResult(out, metric_id, "property", (("property", "Inmueble / ubicación"), ("value", label), ("Currency", "Moneda")))


def _support_dimension(support: pd.DataFrame, scope: str, field: str, metric_id: str, column_label: str) -> Round3ViewResult:
    _require(support, {"period", "Currency", "target_box", "recognized_amount", field}, "monthly_stakeholder_support.csv")
    work = support.copy()
    work["target_box"] = _text(work["target_box"], "")
    work = work.loc[work["target_box"].isin(_scope_boxes(scope))].copy()
    work[field] = _text(work[field], "Sin clasificar")
    work["value"] = pd.to_numeric(work["recognized_amount"], errors="coerce")
    if work["value"].isna().any() or work["value"].lt(-TOLERANCE).any():
        raise ValueError(f"support dimension {field} contains unavailable or negative recognized amounts")
    work["period"] = work["period"].astype(str).str[:4]
    grouped = work.groupby(["period", "Currency", field], as_index=False, sort=True)["value"].sum()
    grouped = grouped.loc[grouped["value"].gt(TOLERANCE)].copy()
    out = _decorate(
        grouped,
        metric_id=metric_id,
        dimension=field,
        scope=scope,
        period_basis="annual",
        source_table="monthly_stakeholder_support.csv",
        source_filter=f"target_box in reporting scope; group recognized_amount by {field}",
        calculation_rule=f"annual governed stakeholder support grouped by {field}; no cash/debt/legal inference",
    )
    return Round3ViewResult(out, metric_id, field, ((field, column_label), ("value", "Apoyo reconocido"), ("Currency", "Moneda")))


def _physical_by_category(flow: pd.DataFrame, cash: pd.DataFrame, scope: str, direction: str) -> Round3ViewResult:
    _require(flow, {"period", "Box", "Currency", "movement_basis", "cash_direction", "cash_category", "amount_in", "amount_out"}, "monthly_box_treasury_flow.csv")
    measure = "amount_in" if direction == "in" else "amount_out"
    total_measure = "total_cash_in" if direction == "in" else "total_cash_out"
    work = flow.loc[
        flow["Box"].astype(str).isin(_scope_boxes(scope))
        & flow["movement_basis"].astype(str).eq("actual_cash")
        & flow["cash_direction"].astype(str).eq(direction)
    ].copy()
    work["value"] = pd.to_numeric(work[measure], errors="coerce")
    if work["value"].isna().any() or work["value"].lt(-TOLERANCE).any():
        raise ValueError("treasury category view contains unavailable or negative physical cash values")
    work["period"] = work["period"].astype(str).str[:4]
    work["cash_category"] = _text(work["cash_category"], "unknown")
    grouped = work.groupby(["period", "Currency", "cash_category"], as_index=False, sort=True)["value"].sum()
    grouped = grouped.loc[grouped["value"].gt(TOLERANCE)].copy()

    _require(cash, {"period", "Box", "Currency", total_measure}, "monthly_cash_accountability.csv")
    expected = cash.loc[cash["Box"].astype(str).isin(_scope_boxes(scope)), ["period", "Currency", total_measure]].copy()
    expected["value"] = pd.to_numeric(expected[total_measure], errors="coerce")
    if expected["value"].isna().any():
        raise ValueError("cash accountability contains unavailable physical cash total")
    expected["period"] = expected["period"].astype(str).str[:4]
    expected = expected.groupby(["period", "Currency"], as_index=False, sort=True)["value"].sum()
    _assert_year_currency(grouped, expected, f"physical cash {direction} category -> {total_measure}")

    is_in = direction == "in"
    metric_id = "TREASURY.PHYSICAL.IN.BY_CATEGORY" if is_in else "TREASURY.PHYSICAL.OUT.BY_CATEGORY"
    out = _decorate(
        grouped,
        metric_id=metric_id,
        dimension="cash_category",
        scope=scope,
        period_basis="annual",
        source_table="monthly_box_treasury_flow.csv + monthly_cash_accountability.csv",
        source_filter=f"movement_basis=actual_cash; cash_direction={direction}; Box in reporting scope",
        calculation_rule=f"annual physical cash {direction} grouped by governed cash_category and reconciled to {total_measure}",
    )
    return Round3ViewResult(out, metric_id, "cash_category", (("cash_category", "Categoría de caja"), ("value", "Importe físico"), ("Currency", "Moneda")))


def _cash_residuals(cash: pd.DataFrame, scope: str) -> Round3ViewResult:
    columns = ["other_cash_in", "unknown_cash_in", "other_cash_out", "unknown_cash_out"]
    _require(cash, {"period", "Box", "Currency", *columns}, "monthly_cash_accountability.csv")
    work = cash.loc[cash["Box"].astype(str).isin(_scope_boxes(scope))].copy()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[columns].isna().any().any() or work[columns].lt(-TOLERANCE).any().any():
        raise ValueError("cash residual view contains unavailable or negative residual components")
    work["period"] = work["period"].astype(str).str[:4]
    melted = work.melt(id_vars=["period", "Currency"], value_vars=columns, var_name="residual_type", value_name="value")
    grouped = melted.groupby(["period", "Currency", "residual_type"], as_index=False, sort=True)["value"].sum()
    grouped = grouped.loc[grouped["value"].gt(TOLERANCE)].copy()
    labels = {
        "other_cash_in": "Otras entradas",
        "unknown_cash_in": "Entradas sin clasificar",
        "other_cash_out": "Otras salidas",
        "unknown_cash_out": "Salidas sin clasificar",
    }
    grouped["residual_type"] = grouped["residual_type"].map(labels).fillna(grouped["residual_type"])
    out = _decorate(
        grouped,
        metric_id="TREASURY.CASH.RESIDUALS",
        dimension="residual_type",
        scope=scope,
        period_basis="annual",
        source_table="monthly_cash_accountability.csv",
        source_filter="other_cash_* and unknown_cash_* within reporting scope",
        calculation_rule="annual sum of already-governed residual cash components; diagnostic only, no reclassification",
    )
    return Round3ViewResult(out, "TREASURY.CASH.RESIDUALS", "residual_type", (("residual_type", "Residual"), ("value", "Importe"), ("Currency", "Moneda")))


def build_view(
    view_key: str,
    *,
    run_root: Path,
    metrics_dir: Path,
    scope: str,
) -> Round3ViewResult:
    frames = {key: _read(key, run_root, metrics_dir) for key in VIEW_REQUIREMENTS[view_key]}
    if view_key == "rent_annual_comparison":
        return _rent_annual(frames["annual_metrics"], scope)
    if view_key == "distributions_by_year":
        return _distributions_annual(frames["semantic_audit"], frames["annual_metrics"], scope)
    if view_key == "opex_monthly_evolution":
        return _opex_monthly(frames["semantic_split"], frames["annual_metrics"], scope)
    if view_key == "maintenance_by_property":
        return _category_by_property(frames["semantic_split"], frames["annual_metrics"], scope, "maintenance")
    if view_key == "legal_costs_by_property":
        return _category_by_property(frames["semantic_split"], frames["annual_metrics"], scope, "legal")
    if view_key == "support_by_obligation_category":
        return _support_dimension(frames["stakeholder_support"], scope, "obligation_category", "SUPPORT.BY_OBLIGATION_CATEGORY", "Categoría de obligación")
    if view_key == "support_by_funding_channel":
        return _support_dimension(frames["stakeholder_support"], scope, "funding_channel", "SUPPORT.BY_FUNDING_CHANNEL", "Canal de funding")
    if view_key == "support_by_settlement_nature":
        return _support_dimension(frames["stakeholder_support"], scope, "settlement_nature", "SUPPORT.BY_SETTLEMENT_NATURE", "Naturaleza de aplicación")
    if view_key == "physical_inflows_by_category":
        return _physical_by_category(frames["treasury_flow"], frames["cash_accountability"], scope, "in")
    if view_key == "physical_outflows_by_category":
        return _physical_by_category(frames["treasury_flow"], frames["cash_accountability"], scope, "out")
    if view_key == "cash_residuals":
        return _cash_residuals(frames["cash_accountability"], scope)
    raise KeyError(f"unknown round-3 specialized report view: {view_key}")
