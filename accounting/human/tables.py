from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from accounting.metrics.build import METRIC_VIEWS_DIRNAME
from accounting.metrics.drilldown import DRILLDOWN_DIRNAME, DRILLDOWN_INDEX_FILENAME


# -----------------------
# Public spec model
# -----------------------

@dataclass(frozen=True)
class HumanTableSpec:
    item_id: str
    slug: str
    title: str
    builder_key: str
    group: str = "core"
    notes: str = ""
    enabled_by_default: bool = True


# -----------------------
# Registry of default tables
# -----------------------

def default_human_table_specs_v1() -> List[HumanTableSpec]:
    return [
        HumanTableSpec("1.1", "cash_snapshot", "Snapshot de caja", "cash_snapshot", "liquidity"),
        HumanTableSpec("1.2", "cash_by_box_y", "Caja por box (anual)", "cash_by_box_y", "liquidity"),
        HumanTableSpec("1.3", "cash_by_box_q", "Caja por box (trimestral)", "cash_by_box_q", "liquidity"),
        HumanTableSpec("1.4", "cash_position_monthly_last12", "Posición de caja mensual últimos 12 meses", "cash_position_monthly_last12", "liquidity"),
        HumanTableSpec("1.5", "debt_snapshot", "Snapshot de deuda", "debt_snapshot", "debt"),
        HumanTableSpec("1.6", "debt_principal_vs_interest_snapshot", "Deuda: principal vs interés", "debt_principal_vs_interest_snapshot", "debt"),
        HumanTableSpec("1.7", "cash_vs_debt_snapshot", "Caja vs deuda", "cash_vs_debt_snapshot", "liquidity"),
        HumanTableSpec("1.8", "income_statement_monthly_last6", "P&L mensual últimos 6 meses", "income_statement_monthly_last6", "income"),
        HumanTableSpec("1.9", "income_statement_y", "P&L anual", "income_statement_y", "income"),
        HumanTableSpec("1.10", "income_statement_q", "P&L trimestral", "income_statement_q", "income"),
        HumanTableSpec("1.11", "opex_by_category_m_last12", "Opex por categoría últimos 12 meses", "opex_by_category_m_last12", "income"),
        HumanTableSpec("1.12", "opex_by_category_y", "Opex por categoría (anual)", "opex_by_category_y", "income"),
        HumanTableSpec("1.13", "rent_rollup_by_place_m_last6", "Renta por lugar, caja y moneda", "rent_rollup_by_place_m_last6", "income"),
        HumanTableSpec("1.14", "rent_rollup_by_detail_m_last6", "Renta por detalle, caja y moneda", "rent_rollup_by_detail_m_last6", "income"),
        HumanTableSpec("1.15", "contrib_rollup_by_party_m_last12", "Contribuciones por parte últimos 12 meses", "contrib_rollup_by_party_m_last12", "income"),
        HumanTableSpec("1.16", "contrib_rollup_by_party_y", "Contribuciones por parte (anual)", "contrib_rollup_by_party_y", "income"),
        HumanTableSpec("1.17", "flow_type_rollup_m_last6", "Drilldown por flujo y tipo", "flow_type_rollup_m_last6", "flows"),
        HumanTableSpec("1.18", "draws_discipline_monthly_last6", "Retiros y disciplina, últimos 6 meses", "draws_discipline_monthly_last6", "flows"),
        HumanTableSpec("1.19", "debt_balance_monthly_last12", "Deuda abierta mensual últimos 12 meses", "debt_balance_monthly_last12", "debt"),
        HumanTableSpec("1.20", "debt_by_counterparty_m_last12", "Deuda por contraparte últimos 12 meses", "debt_by_counterparty_m_last12", "debt"),
        HumanTableSpec("1.21", "debt_net_position_m_last12", "Posición neta PM últimos 12 meses", "debt_net_position_m_last12", "debt"),
        HumanTableSpec("1.22", "validation_report_expanded", "Validaciones expandidas", "validation_report_expanded", "qa"),
        HumanTableSpec("1.23", "metric_coverage_registry", "Cobertura del registry", "metric_coverage_registry", "qa"),
        HumanTableSpec("1.24", "drilldown_availability", "Disponibilidad de drilldown", "drilldown_availability", "qa"),
        HumanTableSpec("1.25", "data_quality", "Calidad de datos y cobertura", "data_quality", "qa"),
    ]


# -----------------------
# Context loading
# -----------------------

@dataclass
class HumanTablesContext:
    metrics_dir: Path
    registry: pd.DataFrame
    metric_values: pd.DataFrame
    validation: pd.DataFrame
    manifest: Dict[str, Any]
    metric_views_manifest: Dict[str, str]
    drilldown_index: pd.DataFrame
    metric_contract_frontier: pd.DataFrame
    frontend_metric_series: pd.DataFrame
    metric_views_cache: Dict[str, pd.DataFrame]


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json_required(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def load_human_tables_context(metrics_dir: Path) -> HumanTablesContext:
    metrics_dir = Path(metrics_dir)

    registry = _read_csv_required(metrics_dir / "metric_registry.csv")
    metric_values = _read_csv_required(metrics_dir / "metric_values.csv")
    validation = _read_csv_required(metrics_dir / "validation_report.csv")
    manifest = _read_json_required(metrics_dir / "build_manifest.json")

    metric_views_manifest_path = metrics_dir / METRIC_VIEWS_DIRNAME / "metric_views_manifest.csv"
    metric_views_manifest_df = _read_csv_required(metric_views_manifest_path)
    metric_views_manifest = {}
    if not metric_views_manifest_df.empty:
        metric_views_manifest = {
            str(k): str(v)
            for k, v in metric_views_manifest_df.iloc[0].to_dict().items()
        }

    drilldown_path = metrics_dir / DRILLDOWN_DIRNAME / DRILLDOWN_INDEX_FILENAME
    drilldown_index = pd.read_csv(drilldown_path) if drilldown_path.exists() else pd.DataFrame()
    metric_contract_frontier = pd.read_csv(metrics_dir / "metric_contract_frontier.csv") if (metrics_dir / "metric_contract_frontier.csv").exists() else pd.DataFrame()
    frontend_metric_series = pd.read_csv(metrics_dir / "frontend_metric_series.csv") if (metrics_dir / "frontend_metric_series.csv").exists() else pd.DataFrame()

    return HumanTablesContext(
        metrics_dir=metrics_dir,
        registry=registry,
        metric_values=metric_values,
        validation=validation,
        manifest=manifest,
        metric_views_manifest=metric_views_manifest,
        drilldown_index=drilldown_index,
        metric_contract_frontier=metric_contract_frontier,
        frontend_metric_series=frontend_metric_series,
        metric_views_cache={},
    )


# -----------------------
# Small helpers
# -----------------------

def _latest_period(metric_values: pd.DataFrame, grain: str) -> Optional[str]:
    vals = (
        metric_values.loc[metric_values["period_grain"] == grain, "period"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    vals = sorted(vals)
    return vals[-1] if vals else None


def _prev_y(period_y: Optional[str]) -> Optional[str]:
    if not period_y:
        return None
    try:
        return str(int(period_y) - 1)
    except Exception:
        return None


def _lookup_metric(metric_values: pd.DataFrame, metric_id: str, grain: str, period: Optional[str]) -> pd.DataFrame:
    if not period:
        return metric_values.iloc[0:0].copy()
    return metric_values.loc[
        (metric_values["metric_id"] == metric_id)
        & (metric_values["period_grain"] == grain)
        & (metric_values["period"] == period)
    ].copy()


def _label_map(registry: pd.DataFrame) -> Dict[str, str]:
    if "label" not in registry.columns:
        return {}
    return dict(zip(registry["metric_id"].astype(str), registry["label"].astype(str)))


def _read_metric_view_cached(ctx: HumanTablesContext, filename: str) -> pd.DataFrame:
    if filename in ctx.metric_views_cache:
        return ctx.metric_views_cache[filename].copy()

    path = ctx.metrics_dir / METRIC_VIEWS_DIRNAME / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing metric view: {path}")

    df = pd.read_csv(path)
    ctx.metric_views_cache[filename] = df
    return df.copy()


def _safe_read_metric_view_cached(ctx: HumanTablesContext, filename: str) -> pd.DataFrame:
    path = ctx.metrics_dir / METRIC_VIEWS_DIRNAME / filename
    if not path.exists():
        return pd.DataFrame()
    return _read_metric_view_cached(ctx, filename)


def _build_snapshot_from_metric_ids(
    registry: pd.DataFrame,
    metric_values: pd.DataFrame,
    metric_ids: Sequence[str],
    *,
    grain: str = "Y",
) -> pd.DataFrame:
    current = _latest_period(metric_values, grain)
    prev = _prev_y(current) if grain == "Y" else None
    labels = _label_map(registry)

    rows: List[Dict[str, Any]] = []
    for metric_id in metric_ids:
        cur = _lookup_metric(metric_values, metric_id, grain, current)
        prv = _lookup_metric(metric_values, metric_id, grain, prev) if prev else metric_values.iloc[0:0].copy()
        currencies = sorted(set(cur["currency"].astype(str)) | set(prv["currency"].astype(str))) or [""]
        for currency in currencies:
            c = cur.loc[cur["currency"].astype(str) == currency]
            p = prv.loc[prv["currency"].astype(str) == currency]
            cur_val = c["value"].iloc[0] if not c.empty else pd.NA
            prev_val = p["value"].iloc[0] if not p.empty else pd.NA
            delta = (cur_val - prev_val) if (pd.notna(cur_val) and pd.notna(prev_val)) else pd.NA
            rows.append(
                {
                    "metric_id": metric_id,
                    "label": labels.get(metric_id, metric_id),
                    "currency": currency,
                    "period": current or "",
                    "value": cur_val,
                    "prev_y": prev_val,
                    "delta_vs_prev_y": delta,
                }
            )
    return pd.DataFrame(rows)


def _build_metric_coverage_registry(ctx: HumanTablesContext) -> pd.DataFrame:
    reg = ctx.registry.copy()
    built = set(ctx.metric_values["metric_id"].astype(str).tolist())

    out = pd.DataFrame(
        {
            "metric_id": reg.get("metric_id", pd.Series(dtype=str)).astype(str),
            "label": reg.get("label", pd.Series(dtype=str)).astype(str),
            "statement": reg.get("statement", pd.Series(dtype=str)).astype(str),
            "section": reg.get("section", pd.Series(dtype=str)).astype(str),
            "is_leaf": reg.get("is_leaf", pd.Series(dtype=bool)).astype(bool),
            "builder_key": reg.get("builder_key", pd.Series(dtype=str)).astype(str),
            "status": reg.get("status", pd.Series(dtype=str)).astype(str),
        }
    )
    out["present_in_metric_values"] = out["metric_id"].isin(built)
    return out.sort_values(["statement", "section", "metric_id"]).reset_index(drop=True)


def _build_validation_report_expanded(ctx: HumanTablesContext) -> pd.DataFrame:
    val = ctx.validation.copy()
    if val.empty:
        return pd.DataFrame(columns=["check_name", "level", "ok", "message", "detail"])

    rename_map = {}
    if "status" in val.columns and "ok" not in val.columns:
        rename_map["status"] = "ok"
    if "notes" in val.columns and "detail" not in val.columns:
        rename_map["notes"] = "detail"
    val = val.rename(columns=rename_map)

    keep = [c for c in ["check_name", "level", "ok", "message", "detail"] if c in val.columns]
    if not keep:
        return val
    return val[keep].copy()


def _build_drilldown_availability(ctx: HumanTablesContext) -> pd.DataFrame:
    dd = ctx.drilldown_index.copy()
    if dd.empty:
        return pd.DataFrame(
            columns=[
                "metric_id",
                "period_grain",
                "period",
                "currency",
                "status",
                "matched_rows",
                "target_metric_value",
                "matched_value_sum",
                "difference_vs_target",
            ]
        )

    wanted = [
        "metric_id",
        "period_grain",
        "period",
        "currency",
        "status",
        "matched_rows",
        "target_metric_value",
        "matched_value_sum",
        "difference_vs_target",
    ]
    keep = [c for c in wanted if c in dd.columns]
    return dd[keep].sort_values([c for c in ["metric_id", "period_grain", "period", "currency"] if c in keep]).reset_index(drop=True)


def _build_data_quality(ctx: HumanTablesContext) -> pd.DataFrame:
    registry = ctx.registry
    metric_values = ctx.metric_values
    validation = ctx.validation
    manifest = ctx.manifest

    active_leaf = registry.loc[
        registry.get("is_leaf", pd.Series(False, index=registry.index)).astype(bool)
        & (registry.get("status", pd.Series("active", index=registry.index)).astype(str) == "active")
    ].copy()
    built_metric_ids = set(metric_values["metric_id"].astype(str).tolist())
    missing_leaf = active_leaf.loc[~active_leaf["metric_id"].astype(str).isin(built_metric_ids)]

    errors = validation.loc[validation.get("level", pd.Series("", index=validation.index)).astype(str).str.lower() == "error"]
    warnings = validation.loc[validation.get("level", pd.Series("", index=validation.index)).astype(str).str.lower() == "warning"]

    rows = [
        {"check_name": "registry_rows", "value": int(len(registry)), "status": "ok", "detail": ""},
        {"check_name": "metric_values_rows", "value": int(len(metric_values)), "status": "ok", "detail": ""},
        {"check_name": "validation_errors", "value": int(len(errors)), "status": "error" if len(errors) else "ok", "detail": "; ".join(errors.get("check_name", pd.Series(dtype=str)).astype(str).tolist())},
        {"check_name": "validation_warnings", "value": int(len(warnings)), "status": "warning" if len(warnings) else "ok", "detail": "; ".join(warnings.get("check_name", pd.Series(dtype=str)).astype(str).tolist())},
        {"check_name": "missing_active_leaf_metrics", "value": int(len(missing_leaf)), "status": "warning" if len(missing_leaf) else "ok", "detail": ", ".join(missing_leaf["metric_id"].astype(str).tolist())},
        {"check_name": "source_run_root", "value": ctx.manifest.get("run_root", ""), "status": "ok", "detail": ""},
        {"check_name": "source_run_id", "value": ctx.manifest.get("run_id", ""), "status": "ok", "detail": ""},
        {"check_name": "as_of_date", "value": manifest.get("as_of_date", ""), "status": "ok", "detail": ""},
    ]
    return pd.DataFrame(rows)


# -----------------------
# Human table builders
# -----------------------

def _frontier_contract_row(ctx: HumanTablesContext, metric_id: str) -> pd.Series | None:
    frontier = ctx.metric_contract_frontier
    if frontier.empty or "metric_id" not in frontier.columns:
        return None
    sub = frontier.loc[frontier["metric_id"].astype(str).eq(metric_id)]
    return None if sub.empty else sub.iloc[0]


def _frontend_metric_rows(ctx: HumanTablesContext, metric_id: str) -> pd.DataFrame:
    series = ctx.frontend_metric_series
    if series.empty or "metric_id" not in series.columns:
        return pd.DataFrame()
    return series.loc[series["metric_id"].astype(str).eq(metric_id)].copy()


def build_cash_snapshot_table(ctx: HumanTablesContext) -> pd.DataFrame:
    contract = _frontier_contract_row(ctx, "BS.CASH.TOTAL")
    cash_rows = _frontend_metric_rows(ctx, "BS.CASH.TOTAL")
    if contract is None:
        return pd.DataFrame(columns=["metric_id", "label", "currency", "period", "value", "status", "frontend_suitability", "source_table", "caveat"])
    status = str(contract.get("status", ""))
    suitability = str(contract.get("frontend_suitability", ""))
    caveat = str(contract.get("caveat", ""))
    if cash_rows.empty or status == "unavailable" or suitability == "unavailable":
        return pd.DataFrame([{
            "metric_id": "BS.CASH.TOTAL",
            "label": contract.get("label", "Frontend-safe cash total"),
            "currency": "",
            "period": "",
            "value": pd.NA,
            "status": "unavailable",
            "frontend_suitability": "unavailable",
            "source_table": "monthly_cash_close.csv",
            "caveat": caveat or "No frontend-safe cash rows; cash narrative blocked.",
        }])
    cash_rows["value"] = pd.to_numeric(cash_rows["value"], errors="coerce")
    latest = cash_rows.sort_values(["Currency", "period"]).groupby("Currency", dropna=False).tail(1)
    latest = latest.rename(columns={"Currency": "currency"})
    latest["label"] = contract.get("label", "Frontend-safe cash total")
    latest["status"] = status or "active"
    cols = ["metric_id", "label", "currency", "period", "value", "status", "frontend_suitability", "source_table", "caveat"]
    return latest.reindex(columns=cols).reset_index(drop=True)


def build_debt_snapshot_table(ctx: HumanTablesContext) -> pd.DataFrame:
    metric_ids = [
        "BS.DEBT.PM_TO_MI.OPEN",
        "BS.DEBT.PM_TO_PRIMOS.OPEN",
        "BS.CLAIM.ALE_TO_PM.OPEN",
        "BS.DEBT.TOTAL.OPEN",
        "BS.DEBT.NET_PM_POSITION",
        "BS.DEBT.PRINCIPAL.OPEN",
        "BS.DEBT.INTEREST.OPEN",
    ]
    present_ids = [m for m in metric_ids if m in set(ctx.registry.get("metric_id", pd.Series(dtype=str)).astype(str))]
    return _build_snapshot_from_metric_ids(ctx.registry, ctx.metric_values, present_ids, grain="Y")


def build_debt_principal_vs_interest_snapshot_table(ctx: HumanTablesContext) -> pd.DataFrame:
    metric_ids = [
        "BS.DEBT.PRINCIPAL.OPEN",
        "BS.DEBT.INTEREST.OPEN",
    ]
    present_ids = [m for m in metric_ids if m in set(ctx.registry.get("metric_id", pd.Series(dtype=str)).astype(str))]
    return _build_snapshot_from_metric_ids(ctx.registry, ctx.metric_values, present_ids, grain="Y")


def build_cash_vs_debt_snapshot_table(ctx: HumanTablesContext) -> pd.DataFrame:
    cash = build_cash_snapshot_table(ctx)
    debt = build_debt_snapshot_table(ctx)

    def _pick(df: pd.DataFrame, metric_id: str, currency: str) -> Any:
        sub = df.loc[(df["metric_id"] == metric_id) & (df["currency"].astype(str) == str(currency))]
        if sub.empty:
            return pd.NA
        return sub["value"].iloc[0]

    currencies = sorted(set(cash.get("currency", pd.Series(dtype=str)).astype(str)) | set(debt.get("currency", pd.Series(dtype=str)).astype(str)))
    rows: List[Dict[str, Any]] = []
    for cur in currencies:
        cash_fb = _pick(cash, "BS.CASH.FB", cur)
        cash_pm = _pick(cash, "BS.CASH.PM", cur)
        cash_total = _pick(cash, "BS.CASH.TOTAL", cur)
        pm_to_mi = _pick(debt, "BS.DEBT.PM_TO_MI.OPEN", cur)
        pm_to_primos = _pick(debt, "BS.DEBT.PM_TO_PRIMOS.OPEN", cur)
        claim_ale = _pick(debt, "BS.CLAIM.ALE_TO_PM.OPEN", cur)
        net_pm = _pick(debt, "BS.DEBT.NET_PM_POSITION", cur)
        rows.append(
            {
                "currency": cur,
                "cash_fb": cash_fb,
                "cash_pm": cash_pm,
                "cash_total": cash_total,
                "pm_to_mi_open": pm_to_mi,
                "pm_to_primos_open": pm_to_primos,
                "ale_to_pm_claim_open": claim_ale,
                "pm_net_position": net_pm,
            }
        )
    return pd.DataFrame(rows)


def build_income_statement_monthly_last6_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _read_metric_view_cached(ctx, "income_statement_monthly_last6.csv")


def build_income_statement_y_table(ctx: HumanTablesContext) -> pd.DataFrame:
    path = ctx.metrics_dir / "income_statement_y.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_income_statement_q_table(ctx: HumanTablesContext) -> pd.DataFrame:
    path = ctx.metrics_dir / "income_statement_q.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_cash_by_box_y_table(ctx: HumanTablesContext) -> pd.DataFrame:
    rows = _frontend_metric_rows(ctx, "BS.CASH.CLOSE.BOX")
    if rows.empty:
        return pd.DataFrame(columns=["metric_id", "period", "currency", "box", "value", "source_table", "frontend_suitability"])
    rows = rows.rename(columns={"Currency": "currency", "dimension_value": "box"})
    rows["year"] = rows["period"].astype(str).str.slice(0, 4)
    return rows.reindex(columns=["metric_id", "year", "period", "currency", "box", "value", "source_table", "frontend_suitability"]).sort_values(["year", "currency", "box", "period"]).reset_index(drop=True)


def build_cash_by_box_q_table(ctx: HumanTablesContext) -> pd.DataFrame:
    rows = _frontend_metric_rows(ctx, "BS.CASH.CLOSE.BOX")
    if rows.empty:
        return pd.DataFrame(columns=["metric_id", "period", "currency", "box", "value", "source_table", "frontend_suitability"])
    rows = rows.rename(columns={"Currency": "currency", "dimension_value": "box"})
    rows["quarter"] = pd.to_datetime(rows["period"].astype(str) + "-01", errors="coerce").dt.to_period("Q").astype(str)
    return rows.reindex(columns=["metric_id", "quarter", "period", "currency", "box", "value", "source_table", "frontend_suitability"]).sort_values(["quarter", "currency", "box", "period"]).reset_index(drop=True)


def build_cash_position_monthly_last12_table(ctx: HumanTablesContext) -> pd.DataFrame:
    rows = _frontend_metric_rows(ctx, "BS.CASH.TOTAL")
    if rows.empty:
        return pd.DataFrame(columns=["metric_id", "period", "currency", "cash_position_end", "source_table", "frontend_suitability"])
    rows = rows.rename(columns={"Currency": "currency", "value": "cash_position_end"})
    return rows.reindex(columns=["metric_id", "period", "currency", "cash_position_end", "source_table", "frontend_suitability"]).sort_values(["currency", "period"]).groupby("currency", dropna=False).tail(12).reset_index(drop=True)


def build_rent_rollup_by_place_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _read_metric_view_cached(ctx, "rent_rollup_by_place_m_last6.csv")


def build_rent_rollup_by_detail_table(ctx: HumanTablesContext) -> pd.DataFrame:
    # This file may be "extra" in some runs. Keep tolerant.
    candidates = [
        "rent_rollup_by_detail_m_last6.csv",
        "extra__rent_rollup_by_detail_m_last6.csv",
    ]
    for filename in candidates:
        path = ctx.metrics_dir / METRIC_VIEWS_DIRNAME / filename
        if path.exists():
            return pd.read_csv(path)
    # also tolerate it living outside metric_views when copied manually
    extras = [
        ctx.metrics_dir.parent / "human_reports" / filename for filename in candidates
    ]
    for path in extras:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def build_flow_type_rollup_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _read_metric_view_cached(ctx, "flow_type_rollup_m_last6.csv")


def build_draws_discipline_monthly_last6_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "draws_discipline_monthly_last6.csv")


def build_debt_balance_monthly_last12_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "debt_balance_monthly_last12.csv")


def build_debt_by_counterparty_m_last12_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "debt_by_counterparty_m_last12.csv")


def build_debt_net_position_m_last12_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "debt_net_position_m_last12.csv")


def build_opex_by_category_m_last12_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "opex_by_category_m_last12.csv")


def build_opex_by_category_y_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "opex_by_category_y.csv")


def build_contrib_rollup_by_party_m_last12_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "contrib_rollup_by_party_m_last12.csv")


def build_contrib_rollup_by_party_y_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _safe_read_metric_view_cached(ctx, "contrib_rollup_by_party_y.csv")


def build_validation_report_expanded_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _build_validation_report_expanded(ctx)


def build_metric_coverage_registry_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _build_metric_coverage_registry(ctx)


def build_drilldown_availability_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _build_drilldown_availability(ctx)


def build_data_quality_table(ctx: HumanTablesContext) -> pd.DataFrame:
    return _build_data_quality(ctx)


# -----------------------
# Builder registry
# -----------------------

HUMAN_TABLE_BUILDERS: Dict[str, Callable[[HumanTablesContext], pd.DataFrame]] = {
    "cash_snapshot": build_cash_snapshot_table,
    "cash_by_box_y": build_cash_by_box_y_table,
    "cash_by_box_q": build_cash_by_box_q_table,
    "cash_position_monthly_last12": build_cash_position_monthly_last12_table,
    "debt_snapshot": build_debt_snapshot_table,
    "debt_principal_vs_interest_snapshot": build_debt_principal_vs_interest_snapshot_table,
    "cash_vs_debt_snapshot": build_cash_vs_debt_snapshot_table,
    "income_statement_monthly_last6": build_income_statement_monthly_last6_table,
    "income_statement_y": build_income_statement_y_table,
    "income_statement_q": build_income_statement_q_table,
    "opex_by_category_m_last12": build_opex_by_category_m_last12_table,
    "opex_by_category_y": build_opex_by_category_y_table,
    "rent_rollup_by_place_m_last6": build_rent_rollup_by_place_table,
    "rent_rollup_by_detail_m_last6": build_rent_rollup_by_detail_table,
    "contrib_rollup_by_party_m_last12": build_contrib_rollup_by_party_m_last12_table,
    "contrib_rollup_by_party_y": build_contrib_rollup_by_party_y_table,
    "flow_type_rollup_m_last6": build_flow_type_rollup_table,
    "draws_discipline_monthly_last6": build_draws_discipline_monthly_last6_table,
    "debt_balance_monthly_last12": build_debt_balance_monthly_last12_table,
    "debt_by_counterparty_m_last12": build_debt_by_counterparty_m_last12_table,
    "debt_net_position_m_last12": build_debt_net_position_m_last12_table,
    "validation_report_expanded": build_validation_report_expanded_table,
    "metric_coverage_registry": build_metric_coverage_registry_table,
    "drilldown_availability": build_drilldown_availability_table,
    "data_quality": build_data_quality_table,
}


# -----------------------
# Public orchestration
# -----------------------

def build_human_tables(
    ctx: HumanTablesContext,
    specs: Sequence[HumanTableSpec],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        builder = HUMAN_TABLE_BUILDERS.get(spec.builder_key)
        if builder is None:
            raise KeyError(f"No human table builder registered for builder_key={spec.builder_key!r}")
        out[spec.slug] = builder(ctx)
    return out


def build_human_tables_with_specs(
    ctx: HumanTablesContext,
    specs: Optional[Sequence[HumanTableSpec]] = None,
) -> Tuple[List[HumanTableSpec], Dict[str, pd.DataFrame]]:
    if specs is None:
        specs = [s for s in default_human_table_specs_v1() if s.enabled_by_default]
    specs = list(specs)
    tables = build_human_tables(ctx, specs)
    return specs, tables
