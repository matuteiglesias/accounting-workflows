#!/usr/bin/env python3
from __future__ import annotations

"""
Stub architecture for a front-oriented human balance report builder.

Purpose
- Keep the first implementation in a single file.
- Reuse accounting.human.tables as the source of human-facing tables.
- Compose those tables into narrative blocks and profiles.
- Support a gradual migration away from the legacy balance_humano_v2 monolith.

Notes
- This file is intentionally stubbed.
- Function signatures, dataclasses, orchestration flow, and extension points are defined.
- Rendering and block logic can be filled in later by Codex.
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence

import pandas as pd

from accounting.logging_utils import configure_logging, get_logger
from accounting.metrics.drilldown import (
    DRILLDOWN_DIRNAME,
    DRILLDOWN_INDEX_FILENAME,
    drilldown_lookup,
)
from accounting.metrics.views import parse_noise_floor
from accounting.human.tables import (
    HumanTableSpec,
    build_human_tables_with_specs,
    load_human_tables_context,
)


LOG = get_logger("human_balance_front")

REPORT_ID = "human_balance_front_v1"
DEFAULT_PROFILE = "full_front"
DEFAULT_NOISE_FLOOR = {"ARS": 5000.0, "USD": 10.0}
DEFAULT_INCLUDE_STATUSES = ("pagado",)
DEFAULT_CSS = """
:root {
  --fg: #111;
  --muted: #444;
  --bg: #fff;
  --border: #ddd;
  --stripe: #fafafa;
  --accent: #0b57d0;
  --warn: #8a5a00;
  --err: #8f0000;
  --ok: #1a6b2b;
  --maxw: 1180px;
  --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
html, body { background: var(--bg); color: var(--fg); font-family: var(--font); margin: 0; padding: 0; }
main.report { max-width: var(--maxw); margin: 0 auto; padding: 28px 18px 48px; }
h1, h2, h3 { margin: 22px 0 10px; line-height: 1.2; }
h1 { font-size: 28px; }
h2 { font-size: 21px; border-top: 1px solid var(--border); padding-top: 14px; }
h3 { font-size: 16px; }
p { line-height: 1.45; color: var(--muted); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.small { font-size: 12px; color: var(--muted); }
.warn { color: var(--warn); }
.err { color: var(--err); }
.ok { color: var(--ok); }
.pre { font-family: var(--mono); white-space: pre-wrap; word-break: break-word; background: #f7f7f7; border: 1px solid var(--border); padding: 10px; border-radius: 8px; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 8px 0 18px; }
.kpi { border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
.kpi .label { font-size: 12px; color: var(--muted); }
.kpi .value { font-size: 24px; margin-top: 6px; }
.block { margin: 0 0 28px; }
.block-head { margin-bottom: 10px; }
.block-purpose { font-size: 13px; color: var(--muted); }
.block-key-message { border-left: 3px solid var(--accent); padding-left: 10px; margin: 10px 0 12px; color: var(--fg); }
.block-note { font-size: 12px; color: var(--muted); margin-top: 8px; }
.report-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 10px 0 16px; }
.report-table th, .report-table td { border: 1px solid var(--border); padding: 6px 8px; vertical-align: top; }
.report-table th { background: #f3f3f3; text-align: left; font-weight: 600; }
.report-table tr:nth-child(even) td { background: var(--stripe); }
.callout { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin: 10px 0; }
.callout.warn { border-color: #e6d5a8; background: #fff9ea; color: #6d4b00; }
.callout.err { border-color: #e2b5b5; background: #fff1f1; color: #7a0000; }
.callout.ok { border-color: #b8d8bf; background: #f2fff5; color: #145723; }
.section-divider { margin: 26px 0 14px; border-top: 1px dashed var(--border); }
"""


# =========================================================
# Section 1. Dataclasses / config / profiles
# =========================================================

BlockStatus = Literal["ready", "partial", "hidden"]
ProfileName = Literal["executive", "core_evidence", "prudential", "methodology", "full_front"]
CalloutLevel = Literal["ok", "warn", "err"]


@dataclass
class FrontTableRef:
    slug: str
    title: str
    notes: str = ""
    role: Literal["primary", "support", "appendix"] = "support"
    include_csv_link: bool = False
    include_html_link: bool = False
    include_if_empty: bool = False


@dataclass
class FrontChartSpec:
    chart_id: str
    title: str
    kind: Literal["line", "bar", "waterfall", "none"] = "none"
    source_slug: Optional[str] = None
    notes: str = ""


@dataclass
class FrontCallout:
    level: CalloutLevel
    text: str


@dataclass
class FrontBlock:
    block_id: str
    title: str
    layer: str
    purpose: str
    key_message: str
    status: BlockStatus = "ready"
    audience: str = "general"
    order: int = 0
    narrative_html: str = ""
    table_refs: List[FrontTableRef] = field(default_factory=list)
    chart_specs: List[FrontChartSpec] = field(default_factory=list)
    callouts: List[FrontCallout] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class FrontBuildConfig:
    run_root: Path
    metrics_dir: Path
    write_dir: Path
    profile: ProfileName = DEFAULT_PROFILE  # type: ignore[assignment]
    months: int = 6
    include_statuses: Sequence[str] = DEFAULT_INCLUDE_STATUSES
    noise_floor_by_currency: Dict[str, float] = field(default_factory=lambda: DEFAULT_NOISE_FLOOR.copy())
    rent_place_col: str = "Lugar"
    rent_detail_col: str = "Detalle"
    flow_rollup_groupby: Sequence[str] = field(default_factory=lambda: ["Flujo", "Tipo", "Currency"])
    include_partial_blocks: bool = True
    include_hidden_blocks: bool = False
    generate_standalone_tables: bool = True
    generate_manifest: bool = True
    write_css: bool = True


@dataclass
class FrontRenderPaths:
    base: Path
    tables: Path
    html: Path
    pages: Path
    assets: Path
    drilldown: Path


@dataclass
class FrontDataContext:
    config: FrontBuildConfig
    run_manifest: Dict[str, Any]
    metric_views_manifest: Dict[str, Any]
    drilldown_index: pd.DataFrame
    drilldown_lookup_map: Dict[Any, Dict[str, Any]]
    table_specs: List[HumanTableSpec]
    tables_by_slug: Dict[str, pd.DataFrame]
    table_specs_by_slug: Dict[str, HumanTableSpec]
    generated_at_utc: str


@dataclass
class FrontRenderResult:
    blocks: List[FrontBlock]
    pages_written: List[Path]
    manifest_path: Optional[Path] = None


PROFILE_BLOCKS: Dict[str, List[str]] = {
    "executive": [
        "executive_summary",
        "cash_visibility",
        "recent_performance",
        "draws_discipline",
        "cost_structure",
        "rent_engines",
        "contributions_support",
    ],
    "core_evidence": [
        "cash_visibility",
        "recent_performance",
        "draws_discipline",
        "cost_structure",
        "rent_engines",
        "contributions_support",
        "flow_type_bridge",
    ],
    "prudential": [
        "cash_visibility",
        "prudential_balance",
        "contributions_support",
        "action_guidance",
    ],
    "methodology": [
        "methodology_quality",
    ],
    "full_front": [
        "executive_summary",
        "cash_visibility",
        "recent_performance",
        "draws_discipline",
        "cost_structure",
        "rent_engines",
        "contributions_support",
        "flow_type_bridge",
        "prudential_balance",
        "methodology_quality",
        "action_guidance",
    ],
}


# =========================================================
# Section 2. Small utilities
# =========================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _fmt_num(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def _slug_to_title(slug: str) -> str:
    return slug.replace("_", " ").strip().title()


def _ensure_paths(base: Path) -> FrontRenderPaths:
    paths = FrontRenderPaths(
        base=base,
        tables=base / "tables",
        html=base / "html",
        pages=base / "pages",
        assets=base / "assets",
        drilldown=base / "drilldown",
    )
    for p in (paths.base, paths.tables, paths.html, paths.pages, paths.assets, paths.drilldown):
        p.mkdir(parents=True, exist_ok=True)
    return paths


# =========================================================
# Section 3. Common loaders
# =========================================================


def load_metric_views_manifest(metrics_dir: Path) -> Dict[str, Any]:
    """Load metric views manifest (best-effort, non-fatal)."""
    path = metrics_dir / "metric_views" / "metric_views_manifest.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        LOG.warning("Could not read metric views manifest from %s", path, exc_info=True)
        return {}
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): ("" if pd.isna(v) else v) for k, v in row.items()}


def load_drilldown_index(metrics_dir: Path) -> pd.DataFrame:
    path = metrics_dir / DRILLDOWN_DIRNAME / DRILLDOWN_INDEX_FILENAME
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_front_data_context(config: FrontBuildConfig) -> FrontDataContext:
    """Load all reusable inputs needed by block builders.

    This is the main shared loader for the mega-file architecture.
    """
    human_ctx = load_human_tables_context(config.metrics_dir)
    table_specs, tables = build_human_tables_with_specs(human_ctx)

    drilldown_index = load_drilldown_index(config.metrics_dir)
    dd_lookup = drilldown_lookup(drilldown_index)

    ctx = FrontDataContext(
        config=config,
        run_manifest=getattr(human_ctx, "manifest", {}),
        metric_views_manifest=load_metric_views_manifest(config.metrics_dir),
        drilldown_index=drilldown_index,
        drilldown_lookup_map=dd_lookup,
        table_specs=table_specs,
        tables_by_slug=tables,
        table_specs_by_slug={s.slug: s for s in table_specs},
        generated_at_utc=_now_iso(),
    )
    return ctx


# =========================================================
# Section 4. Table helpers / selectors / drilldown helpers
# =========================================================


def get_table(ctx: FrontDataContext, slug: str) -> pd.DataFrame:
    return ctx.tables_by_slug.get(slug, pd.DataFrame())


def has_non_empty_table(ctx: FrontDataContext, slug: str) -> bool:
    df = get_table(ctx, slug)
    return not df.empty


def build_kpi_cards(ctx: FrontDataContext) -> List[Dict[str, str]]:
    """Stub: select top KPI cards for report header.

    Expected output:
    [{"label": "Caja total [ARS]", "value": "2,876,421"}, ...]
    """
    cash_snapshot = get_table(ctx, "cash_snapshot")
    debt_snapshot = get_table(ctx, "debt_snapshot")
    income_statement_m = get_table(ctx, "income_statement_monthly_last6")
    draws_discipline_m = get_table(ctx, "draws_discipline_monthly_last6")

    def _pick_row(df: pd.DataFrame, metric_id: str, currency: Optional[str] = None) -> Optional[pd.Series]:
        if df.empty or "metric_id" not in df.columns:
            return None
        sub = df.loc[df["metric_id"].astype(str) == str(metric_id)]
        if currency is not None and "currency" in sub.columns:
            sub = sub.loc[sub["currency"].astype(str) == str(currency)]
        return None if sub.empty else sub.iloc[0]


    def _first_present(*rows: Optional[pd.Series]) -> Optional[pd.Series]:
        for row in rows:
            if row is not None:
                return row
        return None
    
    def _pick_metric(metric_id: str, fallback_label: str) -> Dict[str, str]:
        if income_statement_m.empty or "metric_id" not in income_statement_m.columns:
            return {"label": fallback_label, "value": "N/A"}
        sub = income_statement_m.loc[income_statement_m["metric_id"].astype(str) == str(metric_id)]
        if sub.empty:
            return {"label": fallback_label, "value": "N/A"}
        row = sub.iloc[0]
        month_cols = sorted([str(c) for c in sub.columns if str(c).startswith("20")])
        val = row[month_cols[-1]] if month_cols else pd.NA
        label = str(row.get("label", fallback_label))
        return {"label": f"{label} [{row.get('currency','')}]", "value": _fmt_num(val)}

    cards: List[Dict[str, str]] = []


    cash_row = _first_present(
        _pick_row(cash_snapshot, "BS.CASH.TOTAL", "ARS"),
        _pick_row(cash_snapshot, "BS.CASH.TOTAL"),
    )
    
    if cash_row is not None:
        cards.append({"label": f"Caja total [{cash_row.get('currency','')}]", "value": _fmt_num(cash_row.get("value", pd.NA))})

    for metric_id, label in [("IS.NET.AFTER_COSTS", "Neto después de costos"), ("IS.OPEX.TOTAL", "Opex")]:
        item = _pick_metric(metric_id, label)
        if item["value"] != "N/A":
            cards.append(item)



    debt_row = _first_present(
        _pick_row(debt_snapshot, "BS.DEBT.NET_PM_POSITION", "USD"),
        _pick_row(debt_snapshot, "BS.DEBT.NET_PM_POSITION", "ARS"),
        _pick_row(debt_snapshot, "BS.DEBT.NET_PM_POSITION"),
    )


    if debt_row is not None:
        cards.append(
            {
                "label": f"Posición neta PM [{debt_row.get('currency','')}]",
                "value": _fmt_num(debt_row.get("value", pd.NA)),
            }
        )

    if not draws_discipline_m.empty:
        row = draws_discipline_m.iloc[0]
        distress = int(pd.to_numeric(row.get("distress_months", 0), errors="coerce") or 0)
        cards.append({"label": f"Meses en distress [{row.get('currency','')}]", "value": str(distress)})

    return cards[:6]


def maybe_drilldown_link(
    ctx: FrontDataContext,
    metric_id: str,
    period_grain: str,
    period: str,
    currency: str,
    value: Any,
) -> Optional[str]:
    key = (str(metric_id), str(period_grain), str(period), str(currency))
    row = ctx.drilldown_lookup_map.get(key)
    if not row:
        return None
    html_relpath = str(row.get("detail_html_relpath", "")).strip()
    if not html_relpath:
        return None
    href = html_relpath if html_relpath.startswith("../") else f"../{html_relpath.lstrip('./')}"
    return f"<a href='{href}' target='_blank' rel='noopener noreferrer'>{_fmt_num(value)}</a>"


def choose_primary_note_for_table(ctx: FrontDataContext, slug: str) -> str:
    spec = ctx.table_specs_by_slug.get(slug)
    if spec and spec.notes:
        return spec.notes

    mv = ctx.metric_views_manifest
    if slug == "rent_rollup_by_place_m_last6":
        return f"groupby = Box, Currency, {mv.get('rent_place_col', ctx.config.rent_place_col)}"
    if slug == "rent_rollup_by_detail_m_last6":
        return f"groupby = Box, Currency, {mv.get('rent_detail_col', ctx.config.rent_detail_col)}"
    if slug == "flow_type_rollup_m_last6":
        groupby = str(mv.get("flow_rollup_groupby", "")).strip()
        groupby = groupby.replace(",", " , ") if groupby else " , ".join(ctx.config.flow_rollup_groupby)
        return f"groupby = {groupby}"
    if slug == "income_statement_monthly_last6":
        statuses = mv.get("include_statuses", ",".join(ctx.config.include_statuses))
        return f"Ventana mensual reciente. Estados incluidos: {statuses}."
    return ""


# =========================================================
# Section 5. Narrative fragment helpers
# =========================================================


def narrative_executive_summary(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. This block should summarize the case in plain language, "
        "state the central thesis, and explain what the reader is about to see.</p>"
    )


def narrative_cash_visibility(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain visible liquidity, distribution between boxes, "
        "and why visible cash does not automatically equal freely distributable surplus.</p>"
    )


def narrative_recent_performance(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain recent rent, costs, net after costs, and why "
        "the system cannot be described as simply lacking income.</p>"
    )


def narrative_draws_discipline(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain the relationship between recent draws and net, "
        "and why this still matters even if monthly distress count is zero.</p>"
    )


def narrative_cost_structure(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain the major cost categories and separate operating, "
        "conservation, financial, and imputed components as needed.</p>"
    )


def narrative_rent_engines(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain that rent is concentrated in a small number of engines, "
        "which identifies where the main value of the system sits.</p>"
    )


def narrative_contributions_support(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain which actors visibly contributed additional support "
        "and why that matters for fairness and net position.</p>"
    )


def narrative_flow_type_bridge(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain the bridge between rent, transfers, taxes, service, "
        "repayments, contributions, and other visible flow/type components.</p>"
    )


def narrative_prudential_balance(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain prudential reading of cash once debt, claims, and "
        "net position are taken into account.</p>"
    )


def narrative_methodology_quality(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain what is strong, what is partial, and what remains "
        "methodologically limited.</p>"
    )


def narrative_action_guidance(ctx: FrontDataContext) -> str:
    return (
        "<p>Stub narrative. Explain what minimum actions, demands, or decisions follow "
        "from the evidence shown in the report.</p>"
    )


# =========================================================
# Section 6. Block builders
# =========================================================


def build_block_executive_summary(ctx: FrontDataContext) -> FrontBlock:
    return FrontBlock(
        block_id="executive_summary",
        title="Resumen ejecutivo",
        layer="A",
        purpose="Abrir el caso y fijar la lectura general.",
        key_message="Stub key message for executive summary.",
        order=10,
        narrative_html=narrative_executive_summary(ctx),
        table_refs=[],
        callouts=[FrontCallout(level="ok", text="Stub: include 5 verdades contables y distributivas.")],
        tags=["executive", "opening"],
    )


def build_block_cash_visibility(ctx: FrontDataContext) -> FrontBlock:
    primary_df = get_table(ctx, "cash_snapshot")
    status: BlockStatus = "ready" if not primary_df.empty else "partial"
    key_message = "Caja visible disponible para lectura."
    if not primary_df.empty and {"metric_id", "currency", "value"}.issubset(primary_df.columns):
        sub = primary_df.loc[primary_df["metric_id"].astype(str) == "BS.CASH.TOTAL"]
        if not sub.empty:
            row = sub.iloc[0]
            key_message = f"Caja total visible: {_fmt_num(row.get('value', pd.NA))} {row.get('currency', '')}."

    block = FrontBlock(
        block_id="cash_visibility",
        title="Caja visible",
        layer="B",
        purpose="Mostrar liquidez visible actual y su evolución reciente.",
        key_message=key_message,
        order=20,
        narrative_html=narrative_cash_visibility(ctx),
        table_refs=[
            FrontTableRef(slug="cash_snapshot", title="Snapshot de caja", role="primary"),
            FrontTableRef(slug="cash_position_monthly_last12", title="Posición de caja mensual últimos 12 meses", role="support"),
            FrontTableRef(slug="cash_by_box_y", title="Caja por box anual", role="appendix"),
        ],
        chart_specs=[
            FrontChartSpec(chart_id="cash_trend_12m", title="Evolución de caja visible", kind="line", source_slug="cash_position_monthly_last12"),
        ],
        tags=["cash", "balance_sheet"],
    )
    if status != "ready":
        block.status = status
        block.callouts.append(FrontCallout(level="warn", text="Falta la tabla principal de caja; el bloque queda parcial."))
    return block


def build_block_recent_performance(ctx: FrontDataContext) -> FrontBlock:
    df = get_table(ctx, "income_statement_monthly_last6")
    status: BlockStatus = "ready" if not df.empty else "partial"
    key_message = "Lectura reciente de renta, costos y neto disponible."
    if not df.empty and {"metric_id", "currency"}.issubset(df.columns):
        sub = df.loc[df["metric_id"].astype(str) == "IS.NET.AFTER_COSTS"]
        month_cols = sorted([str(c) for c in df.columns if str(c).startswith("20")])
        if not sub.empty and month_cols:
            row = sub.iloc[0]
            key_message = f"Neto después de costos (último mes): {_fmt_num(row.get(month_cols[-1], pd.NA))} {row.get('currency', '')}."

    block = FrontBlock(
        block_id="recent_performance",
        title="Resultado reciente",
        layer="B",
        purpose="Mostrar renta, costos, ingresos y neto en la ventana reciente.",
        key_message=key_message,
        order=30,
        narrative_html=narrative_recent_performance(ctx),
        table_refs=[
            FrontTableRef(slug="income_statement_monthly_last6", title="P&L mensual últimos 6 meses", role="primary"),
        ],
        chart_specs=[
            FrontChartSpec(chart_id="recent_income_vs_opex", title="Renta, costos y neto recientes", kind="bar", source_slug="income_statement_monthly_last6"),
        ],
        tags=["income_statement", "recent"],
    )
    if status != "ready":
        block.status = status
        block.callouts.append(FrontCallout(level="warn", text="Sin P&L mensual reciente; este bloque queda parcial."))
    return block


def build_block_draws_discipline(ctx: FrontDataContext) -> FrontBlock:
    df = get_table(ctx, "draws_discipline_monthly_last6")
    status: BlockStatus = "ready" if not df.empty else "partial"
    key_message = "Disciplina de retiros en seguimiento."
    if not df.empty and "distress_months" in df.columns:
        distress = int(pd.to_numeric(df["distress_months"], errors="coerce").fillna(0).max())
        key_message = f"Meses en distress por retiros en ventana reciente: {distress}."

    block = FrontBlock(
        block_id="draws_discipline",
        title="Retiros y disciplina",
        layer="B",
        purpose="Comparar retiros con neto mensual reciente.",
        key_message=key_message,
        order=40,
        narrative_html=narrative_draws_discipline(ctx),
        table_refs=[
            FrontTableRef(slug="draws_discipline_monthly_last6", title="Retiros y disciplina, últimos 6 meses", role="primary"),
        ],
        chart_specs=[
            FrontChartSpec(chart_id="draws_vs_net", title="Retiros vs neto", kind="bar", source_slug="draws_discipline_monthly_last6"),
        ],
        tags=["draws", "discipline"],
    )
    if status != "ready":
        block.status = status
        block.callouts.append(FrontCallout(level="warn", text="Sin tabla de retiros/disciplinas; el bloque queda parcial."))
    return block


def build_block_cost_structure(ctx: FrontDataContext) -> FrontBlock:
    df = get_table(ctx, "opex_by_category_m_last12")
    status: BlockStatus = "ready" if not df.empty else "partial"
    key_message = "Estructura de costos observada en los últimos meses."
    if not df.empty and "total_6m" in df.columns:
        top = df.sort_values("total_6m", ascending=False).iloc[0]
        cat = str(top.get("Tipo", top.get("category", "categoría principal")))
        key_message = f"Mayor presión reciente: {cat} con total 6m {_fmt_num(top.get('total_6m', pd.NA))}."

    block = FrontBlock(
        block_id="cost_structure",
        title="Costos reales del sistema",
        layer="B",
        purpose="Mostrar rubros de costo relevantes y estructura reciente.",
        key_message=key_message,
        order=50,
        narrative_html=narrative_cost_structure(ctx),
        table_refs=[
            FrontTableRef(slug="opex_by_category_m_last12", title="Opex por categoría últimos 12 meses", role="primary"),
            FrontTableRef(slug="opex_by_category_y", title="Opex por categoría anual", role="support"),
        ],
        chart_specs=[
            FrontChartSpec(chart_id="opex_category_mix", title="Mix de costos", kind="bar", source_slug="opex_by_category_m_last12"),
        ],
        callouts=[FrontCallout(level="warn", text="Stub: separar costos operativos, financieros e imputados cuando corresponda.")],
        tags=["opex", "costs"],
    )
    if status != "ready":
        block.status = status
        block.callouts.append(FrontCallout(level="warn", text="Falta Opex por categoría mensual; bloque parcial."))
    return block


def build_block_rent_engines(ctx: FrontDataContext) -> FrontBlock:
    df = get_table(ctx, "rent_rollup_by_place_m_last6")
    status: BlockStatus = "ready" if not df.empty else "partial"
    key_message = "La renta se concentra en pocos motores."
    if not df.empty and "total_6m" in df.columns:
        top = df.sort_values("total_6m", ascending=False).iloc[0]
        place = top.get(ctx.config.rent_place_col, top.get("Lugar", "N/A"))
        key_message = f"Motor líder en 6m: {place} ({_fmt_num(top.get('total_6m', pd.NA))})."

    block = FrontBlock(
        block_id="rent_engines",
        title="Motores de renta",
        layer="B",
        purpose="Mostrar los principales lugares y detalles que explican la renta.",
        key_message=key_message,
        order=60,
        narrative_html=narrative_rent_engines(ctx),
        table_refs=[
            FrontTableRef(slug="rent_rollup_by_place_m_last6", title="Renta por lugar, caja y moneda", role="primary"),
            FrontTableRef(slug="rent_rollup_by_detail_m_last6", title="Renta por detalle, caja y moneda", role="support"),
        ],
        chart_specs=[
            FrontChartSpec(chart_id="rent_top_places", title="Top motores de renta", kind="bar", source_slug="rent_rollup_by_place_m_last6"),
        ],
        tags=["rent", "engines"],
    )
    if status != "ready":
        block.status = status
        block.callouts.append(FrontCallout(level="warn", text="Sin tabla principal de motores de renta; bloque parcial."))
    return block


def build_block_contributions_support(ctx: FrontDataContext) -> FrontBlock:
    df = get_table(ctx, "contrib_rollup_by_party_m_last12")
    status: BlockStatus = "ready" if not df.empty else "partial"
    key_message = "Las contribuciones visibles se presentan por parte."
    if not df.empty and "total_6m" in df.columns:
        top = df.sort_values("total_6m", ascending=False).iloc[0]
        party = str(top.get("party", top.get("payer", top.get("receiver", "parte principal"))))
        key_message = f"Parte con mayor soporte visible (6m): {party} ({_fmt_num(top.get('total_6m', pd.NA))})."

    block = FrontBlock(
        block_id="contributions_support",
        title="Quién sostuvo el sistema",
        layer="B",
        purpose="Mostrar contribuciones visibles por parte y su historia reciente.",
        key_message=key_message,
        order=70,
        narrative_html=narrative_contributions_support(ctx),
        table_refs=[
            FrontTableRef(slug="contrib_rollup_by_party_m_last12", title="Contribuciones por parte últimos 12 meses", role="primary"),
            FrontTableRef(slug="contrib_rollup_by_party_y", title="Contribuciones por parte anual", role="support"),
        ],
        chart_specs=[
            FrontChartSpec(chart_id="contrib_by_party", title="Contribuciones por parte", kind="bar", source_slug="contrib_rollup_by_party_m_last12"),
        ],
        tags=["contributions", "support"],
    )
    if status != "ready":
        block.status = status
        block.callouts.append(FrontCallout(level="warn", text="Sin tabla principal de contribuciones; bloque parcial."))
    return block


def build_block_flow_type_bridge(ctx: FrontDataContext) -> FrontBlock:
    return FrontBlock(
        block_id="flow_type_bridge",
        title="Puente por flujo y tipo",
        layer="B",
        purpose="Unir renta, gasto, impuestos, servicio, repago, contribuciones y FX en una sola superficie de lectura.",
        key_message="Stub key message for flow/type bridge.",
        order=80,
        narrative_html=narrative_flow_type_bridge(ctx),
        table_refs=[
            FrontTableRef(slug="flow_type_rollup_m_last6", title="Drilldown por flujo y tipo", role="primary"),
        ],
        chart_specs=[],
        tags=["bridge", "flow_type"],
    )


def build_block_prudential_balance(ctx: FrontDataContext) -> FrontBlock:
    debt_snapshot = get_table(ctx, "debt_snapshot")
    cash_vs_debt = get_table(ctx, "cash_vs_debt_snapshot")
    has_primary = (not debt_snapshot.empty) and (not cash_vs_debt.empty)
    block_status: BlockStatus = "partial" if has_primary else "hidden"
    callouts: List[FrontCallout] = []
    if not has_primary:
        callouts.append(
            FrontCallout(level="warn", text="No hay tablas prudenciales de deuda completas; se oculta este bloque.")
        )

    return FrontBlock(
        block_id="prudential_balance",
        title="Lectura prudencial de caja y deuda",
        layer="C",
        purpose="Pasar de caja visible a caja prudencialmente interpretable.",
        key_message="Lectura prudencial disponible cuando existe base de deuda consistente.",
        order=90,
        narrative_html=narrative_prudential_balance(ctx),
        table_refs=[
            FrontTableRef(slug="debt_snapshot", title="Snapshot de deuda", role="primary", include_if_empty=False),
            FrontTableRef(slug="debt_principal_vs_interest_snapshot", title="Deuda: principal vs interés", role="support", include_if_empty=False),
            FrontTableRef(slug="cash_vs_debt_snapshot", title="Caja vs deuda", role="primary", include_if_empty=False),
            FrontTableRef(slug="debt_balance_monthly_last12", title="Deuda abierta mensual últimos 12 meses", role="support", include_if_empty=False),
            FrontTableRef(slug="debt_by_counterparty_m_last12", title="Deuda por contraparte últimos 12 meses", role="support", include_if_empty=False),
            FrontTableRef(slug="debt_net_position_m_last12", title="Posición neta PM últimos 12 meses", role="support", include_if_empty=False),
        ],
        chart_specs=[
            FrontChartSpec(chart_id="cash_to_net_position", title="Caja visible vs posición prudente", kind="waterfall", source_slug="cash_vs_debt_snapshot"),
        ],
        status=block_status,
        callouts=callouts,
        tags=["prudence", "debt", "claims"],
    )


def build_block_methodology_quality(ctx: FrontDataContext) -> FrontBlock:
    return FrontBlock(
        block_id="methodology_quality",
        title="Metodología y calidad",
        layer="D",
        purpose="Mostrar confiabilidad, cobertura y límites del material.",
        key_message="Stub key message for methodology and quality.",
        order=100,
        narrative_html=narrative_methodology_quality(ctx),
        table_refs=[
            FrontTableRef(slug="validation_report_expanded", title="Validaciones expandidas", role="primary"),
            FrontTableRef(slug="metric_coverage_registry", title="Cobertura del registry", role="support"),
            FrontTableRef(slug="drilldown_availability", title="Disponibilidad de drilldown", role="support"),
            FrontTableRef(slug="data_quality", title="Calidad de datos y cobertura", role="primary"),
        ],
        chart_specs=[],
        tags=["qa", "methodology"],
    )


def build_block_action_guidance(ctx: FrontDataContext) -> FrontBlock:
    return FrontBlock(
        block_id="action_guidance",
        title="Guía de acción",
        layer="E",
        purpose="Traducir la evidencia a exigencias mínimas, prudencia distributiva y próximos pasos.",
        key_message="Stub key message for action guidance.",
        order=110,
        narrative_html=narrative_action_guidance(ctx),
        table_refs=[],
        chart_specs=[],
        callouts=[
            FrontCallout(level="ok", text="Stub: include minimum demands, decision gates, and next actions."),
        ],
        tags=["action", "governance"],
    )


def build_all_blocks(ctx: FrontDataContext) -> Dict[str, FrontBlock]:
    blocks = {
        "executive_summary": build_block_executive_summary(ctx),
        "cash_visibility": build_block_cash_visibility(ctx),
        "recent_performance": build_block_recent_performance(ctx),
        "draws_discipline": build_block_draws_discipline(ctx),
        "cost_structure": build_block_cost_structure(ctx),
        "rent_engines": build_block_rent_engines(ctx),
        "contributions_support": build_block_contributions_support(ctx),
        "flow_type_bridge": build_block_flow_type_bridge(ctx),
        "prudential_balance": build_block_prudential_balance(ctx),
        "methodology_quality": build_block_methodology_quality(ctx),
        "action_guidance": build_block_action_guidance(ctx),
    }
    return blocks


# =========================================================
# Section 7. Block selection / filtering / profile composition
# =========================================================


def filter_block_for_profile(block: FrontBlock, config: FrontBuildConfig) -> bool:
    if block.status == "hidden" and not config.include_hidden_blocks:
        return False
    if block.status == "partial" and not config.include_partial_blocks:
        return False
    return True


def select_blocks_for_profile(ctx: FrontDataContext, all_blocks: Dict[str, FrontBlock]) -> List[FrontBlock]:
    block_ids = PROFILE_BLOCKS.get(ctx.config.profile, PROFILE_BLOCKS[DEFAULT_PROFILE])
    selected = [all_blocks[b] for b in block_ids if b in all_blocks]
    selected = [b for b in selected if filter_block_for_profile(b, ctx.config)]
    selected.sort(key=lambda b: b.order)
    return selected


# =========================================================
# Section 8. HTML rendering helpers
# =========================================================


def render_kpi_grid(ctx: FrontDataContext) -> str:
    cards = build_kpi_cards(ctx)
    if not cards:
        return ""
    inner = "\n".join(
        f"<div class='kpi'><div class='label'>{c['label']}</div><div class='value'>{c['value']}</div></div>"
        for c in cards
    )
    return f"<div class='kpi-grid'>{inner}</div>"


def render_df_html(
    df: pd.DataFrame,
    *,
    cell_renderer: Optional[Callable[[str, Any, pd.Series], Optional[str]]] = None,
) -> str:
    html_df = df.copy()
    for c in html_df.columns:
        rendered: List[str] = []
        for _, row in html_df.iterrows():
            value = row[c]
            if cell_renderer is not None:
                custom = cell_renderer(c, value, row)
                if custom is not None:
                    rendered.append(custom)
                    continue
            if pd.api.types.is_numeric_dtype(df[c]):
                rendered.append(_fmt_num(value))
            else:
                rendered.append("" if pd.isna(value) else str(value))
        html_df[c] = rendered
    return html_df.to_html(index=False, classes="report-table", border=0, escape=False)


def render_callouts(callouts: Sequence[FrontCallout]) -> str:
    if not callouts:
        return ""
    return "\n".join(f"<div class='callout {c.level}'>{c.text}</div>" for c in callouts)


def render_table_ref(ctx: FrontDataContext, paths: FrontRenderPaths, ref: FrontTableRef) -> str:
    df = get_table(ctx, ref.slug)
    if df.empty and not ref.include_if_empty:
        return ""

    note = ref.notes or choose_primary_note_for_table(ctx, ref.slug)
    title = ref.title or _slug_to_title(ref.slug)

    def _cell_renderer(col: str, value: Any, row: pd.Series) -> Optional[str]:
        col_s = str(col)
        if ref.slug == "income_statement_monthly_last6" and col_s.startswith("20"):
            return maybe_drilldown_link(
                ctx,
                metric_id=str(row.get("metric_id", "")),
                period_grain="M",
                period=col_s,
                currency=str(row.get("currency", "")),
                value=value,
            )
        if ref.slug == "draws_discipline_monthly_last6" and col_s.startswith("draws_"):
            period = col_s.replace("draws_", "", 1)
            return maybe_drilldown_link(
                ctx,
                metric_id="IS.DRAWS.PERSONAL",
                period_grain="M",
                period=period,
                currency=str(row.get("currency", "")),
                value=value,
            )
        return None

    if df.empty:
        body = "<p class='warn'>Tabla vacía.</p>"
    else:
        body = render_df_html(df, cell_renderer=_cell_renderer)

    links: List[str] = []
    if ref.include_csv_link:
        links.append(f"<a href='../tables/{ref.slug}.csv'>CSV</a>")
    if ref.include_html_link:
        links.append(f"<a href='../html/{ref.slug}.html'>HTML</a>")
    links_html = f"<p class='small'>{' | '.join(links)}</p>" if links else ""
    note_html = f"<p class='small'>{note}</p>" if note else ""

    return (
        f"<h3>{title}</h3>"
        + note_html
        + links_html
        + body
    )


def render_chart_stub(chart: FrontChartSpec) -> str:
    return (
        f"<div class='callout warn'>"
        f"Chart stub: <strong>{chart.title}</strong>"
        f" ({chart.kind}) from <code>{chart.source_slug or ''}</code>."
        f"</div>"
    )


def render_block(ctx: FrontDataContext, paths: FrontRenderPaths, block: FrontBlock) -> str:
    tables_html = "\n".join(render_table_ref(ctx, paths, t) for t in block.table_refs)
    charts_html = "\n".join(render_chart_stub(c) for c in block.chart_specs)
    callouts_html = render_callouts(block.callouts)
    notes_html = "\n".join(f"<p class='block-note'>{n}</p>" for n in block.notes)

    status_badge = f"<span class='small'>status: {block.status}</span>"
    return f"""
<section class='block' id='{block.block_id}'>
  <div class='block-head'>
    <h2>{block.title}</h2>
    <p class='block-purpose'>{block.purpose}</p>
    <p class='small'>{status_badge}</p>
  </div>
  <div class='block-key-message'>{block.key_message}</div>
  {block.narrative_html}
  {callouts_html}
  {charts_html}
  {tables_html}
  {notes_html}
</section>
"""


def render_report_page(ctx: FrontDataContext, paths: FrontRenderPaths, blocks: Sequence[FrontBlock], title: str) -> str:
    meta_html = (
        f"<p>run_id: {ctx.run_manifest.get('run_id', '')}<br>"
        f"as_of_date: {ctx.run_manifest.get('as_of_date', '')}<br>"
        f"run_root: {ctx.config.run_root}<br>"
        f"profile: {ctx.config.profile}<br>"
        f"generated_at_utc: {ctx.generated_at_utc}</p>"
    )

    sections = "\n".join(render_block(ctx, paths, b) for b in blocks)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>{title}</title>
  <style>{DEFAULT_CSS}</style>
</head>
<body>
  <main class='report'>
    <h1>{title}</h1>
    {meta_html}
    {render_kpi_grid(ctx)}
    {sections}
  </main>
</body>
</html>
"""
    return html


# =========================================================
# Section 9. Optional standalone table exports
# =========================================================


def write_standalone_table_exports(ctx: FrontDataContext, paths: FrontRenderPaths) -> None:
    """Stub: write per-table CSV/HTML exports for browsing and reuse.

    This can mimic the legacy behavior while front pages become block-oriented.
    """
    for slug, df in ctx.tables_by_slug.items():
        _write_csv(df, paths.tables / f"{slug}.csv")
        table_html = render_df_html(df) if not df.empty else "<p class='warn'>Tabla vacía.</p>"
        page = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{slug}</title><style>{DEFAULT_CSS}</style></head><body><main class='report'><h1>{_slug_to_title(slug)}</h1>{table_html}</main></body></html>"
        _write_text(page, paths.html / f"{slug}.html")


# =========================================================
# Section 10. Manifest writing
# =========================================================


def build_front_manifest(ctx: FrontDataContext, blocks: Sequence[FrontBlock], pages_written: Sequence[Path]) -> Dict[str, Any]:
    return {
        "report_id": REPORT_ID,
        "generated_at_utc": ctx.generated_at_utc,
        "profile": ctx.config.profile,
        "run_root": str(ctx.config.run_root),
        "metrics_dir": str(ctx.config.metrics_dir),
        "write_dir": str(ctx.config.write_dir),
        "block_count": len(blocks),
        "blocks": [
            {
                "block_id": b.block_id,
                "title": b.title,
                "layer": b.layer,
                "status": b.status,
                "tags": b.tags,
                "table_slugs": [t.slug for t in b.table_refs],
            }
            for b in blocks
        ],
        "pages_written": [str(p) for p in pages_written],
    }


def write_front_manifest(paths: FrontRenderPaths, manifest: Dict[str, Any]) -> Path:
    manifest_path = paths.base / "front_manifest.json"
    _write_text(json.dumps(manifest, indent=2, ensure_ascii=False), manifest_path)
    return manifest_path


# =========================================================
# Section 11. Main orchestration
# =========================================================


def build_front_report(config: FrontBuildConfig) -> FrontRenderResult:
    LOG.info(
        "Front stage start run_root=%s metrics_dir=%s write_dir=%s profile=%s",
        config.run_root,
        config.metrics_dir,
        config.write_dir,
        config.profile,
    )

    paths = _ensure_paths(config.write_dir)
    ctx = load_front_data_context(config)
    all_blocks = build_all_blocks(ctx)
    blocks = select_blocks_for_profile(ctx, all_blocks)

    pages_written: List[Path] = []

    # Primary page for the selected profile.
    page_title = f"Human Balance Front [{config.profile}]"
    main_html = render_report_page(ctx, paths, blocks, page_title)
    main_path = paths.pages / f"{config.profile}.html"
    _write_text(main_html, main_path)
    pages_written.append(main_path)

    # Optional profile-specific companion pages could be added here later.
    # Example:
    # - executive.html
    # - core_evidence.html
    # - prudential.html
    # - methodology.html

    if config.generate_standalone_tables:
        write_standalone_table_exports(ctx, paths)

    manifest_path: Optional[Path] = None
    if config.generate_manifest:
        manifest = build_front_manifest(ctx, blocks, pages_written)
        manifest_path = write_front_manifest(paths, manifest)

    if config.write_css:
        _write_text(DEFAULT_CSS, paths.assets / "report.css")

    LOG.info(
        "Front stage finish main_page=%s manifest=%s blocks=%s",
        main_path,
        manifest_path,
        len(blocks),
    )

    return FrontRenderResult(
        blocks=blocks,
        pages_written=pages_written,
        manifest_path=manifest_path,
    )


# =========================================================
# Section 12. CLI
# =========================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build front-oriented human balance reports.")
    p.add_argument("--run-root", required=True, help="Accounting run root.")
    p.add_argument("--metrics-dir", required=True, help="Directory containing metric artifacts.")
    p.add_argument("--write-dir", required=True, help="Output directory for front reports.")
    p.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=["executive", "core_evidence", "prudential", "methodology", "full_front"],
        help="Front profile to render.",
    )
    p.add_argument("--months", type=int, default=6, help="Number of months to emphasize in recent views.")
    p.add_argument("--rent-place-col", default="Lugar", help="Rent place grouping column.")
    p.add_argument("--rent-detail-col", default="Detalle", help="Rent detail grouping column.")
    p.add_argument("--flow-rollup-groupby", default="Flujo,Tipo,Currency", help="Comma-separated columns.")
    p.add_argument("--include-statuses", default="pagado", help="Comma-separated statuses.")
    p.add_argument("--noise-floor", default="ARS:5000,USD:10", help="Comma-separated thresholds.")
    p.add_argument("--hide-partial", action="store_true", help="Hide partial blocks.")
    p.add_argument("--show-hidden", action="store_true", help="Include hidden blocks.")
    p.add_argument("--no-standalone-tables", action="store_true", help="Do not write per-table exports.")
    p.add_argument("--no-manifest", action="store_true", help="Do not write front manifest.")
    p.add_argument("--no-css", action="store_true", help="Do not write CSS asset.")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> FrontBuildConfig:
    include_statuses = tuple(x.strip() for x in args.include_statuses.split(",") if x.strip())
    flow_rollup_groupby = [x.strip() for x in args.flow_rollup_groupby.split(",") if x.strip()]
    noise_floor_by_currency = parse_noise_floor(args.noise_floor)

    return FrontBuildConfig(
        run_root=Path(args.run_root),
        metrics_dir=Path(args.metrics_dir),
        write_dir=Path(args.write_dir),
        profile=args.profile,
        months=args.months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
        rent_place_col=args.rent_place_col,
        rent_detail_col=args.rent_detail_col,
        flow_rollup_groupby=flow_rollup_groupby,
        include_partial_blocks=not args.hide_partial,
        include_hidden_blocks=args.show_hidden,
        generate_standalone_tables=not args.no_standalone_tables,
        generate_manifest=not args.no_manifest,
        write_css=not args.no_css,
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    config = config_from_args(args)
    build_front_report(config)


if __name__ == "__main__":
    main()
