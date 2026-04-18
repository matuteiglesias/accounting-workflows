#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from accounting.logging_utils import configure_logging, get_logger
from accounting.build_metric_values import METRIC_VIEWS_DIRNAME, REQUIRED_METRIC_VIEW_FILES
from accounting.metric_drilldown import (
    DRILLDOWN_DIRNAME,
    DRILLDOWN_INDEX_FILENAME,
    drilldown_lookup,
)
from accounting.metrics_views import parse_noise_floor
from accounting.human_balance_tables import (
    HumanTableSpec,
    build_human_tables_with_specs,
    default_human_table_specs_v1,
    load_human_tables_context,
)

REPORT_ID = "balance_human_v2"
METRIC_VIEWS_MANIFEST_FILENAME = "metric_views_manifest.csv"

DEFAULT_CSS = """
:root {
  --fg: #111;
  --muted: #444;
  --bg: #fff;
  --border: #ddd;
  --table-stripe: #fafafa;
  --maxw: 1180px;
  --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
html, body { background: var(--bg); color: var(--fg); font-family: var(--font); margin: 0; padding: 0; }
main.report { max-width: var(--maxw); margin: 0 auto; padding: 28px 18px 50px; }
h1, h2, h3 { margin: 22px 0 10px; line-height: 1.2; }
h1 { font-size: 28px; }
h2 { font-size: 20px; border-top: 1px solid var(--border); padding-top: 16px; }
p { line-height: 1.45; color: var(--muted); }
.report-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 10px 0 16px; }
.report-table th, .report-table td { border: 1px solid var(--border); padding: 6px 8px; vertical-align: top; }
.report-table th { background: #f3f3f3; text-align: left; font-weight: 600; }
.report-table tr:nth-child(even) td { background: var(--table-stripe); }
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 8px 0 20px; }
.kpi { border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
.kpi .label { font-size: 12px; color: var(--muted); }
.kpi .value { font-size: 24px; margin-top: 6px; }
.small { font-size: 12px; color: var(--muted); }
.warn { color: #8a5a00; }
.err { color: #8f0000; }
.ok { color: #1a6b2b; }
a { color: #0b57d0; text-decoration: none; }
a:hover { text-decoration: underline; }
.pre { font-family: var(--mono); white-space: pre-wrap; word-break: break-word; background: #f7f7f7; border: 1px solid var(--border); padding: 10px; border-radius: 8px; }
"""

DEFAULT_NOISE_FLOOR = {"ARS": 5000.0, "USD": 10.0}
DEFAULT_INCLUDE_STATUSES = ("pagado",)

LOG = get_logger("human_balance")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs(base: Path) -> Dict[str, Path]:
    tables = base / "tables"
    html = base / "html"
    for d in (base, tables, html):
        d.mkdir(parents=True, exist_ok=True)
    return {"base": base, "tables": tables, "html": html}


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _render_df_html(
    df: pd.DataFrame,
    *,
    cell_renderer: Optional[Callable[[str, Any, pd.Series], Optional[str]]] = None,
) -> str:
    html_df = df.copy()
    for c in html_df.columns:
        rendered = []
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


def _df_to_html_fragment(
    df: pd.DataFrame,
    title: str,
    notes: str = "",
    *,
    cell_renderer: Optional[Callable[[str, Any, pd.Series], Optional[str]]] = None,
) -> str:
    if df.empty:
        body = "<p class='warn'>Tabla vacía.</p>"
    else:
        body = _render_df_html(df, cell_renderer=cell_renderer)
    note_html = f"<p class='small'>{notes}</p>" if notes else ""
    return f"<h2>{title}</h2>\n{note_html}\n{body}\n"


def _manifest_item(spec: HumanTableSpec, csv_path: Path, html_path: Path) -> Dict[str, Any]:
    return {
        "item_id": spec.item_id,
        "kind": "table",
        "slug": spec.slug,
        "title": spec.title,
        "csv": str(csv_path),
        "html": str(html_path),
        "group": spec.group,
        "notes": spec.notes,
    }


def read_metric_views_manifest(metrics_dir: Path) -> Dict[str, str]:
    path = metrics_dir / METRIC_VIEWS_DIRNAME / METRIC_VIEWS_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing metric views manifest: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise FileNotFoundError(f"Metric views manifest is empty: {path}")
    return {k: str(v) for k, v in df.iloc[0].to_dict().items()}


def read_metric_drilldown_index(metrics_dir: Path) -> pd.DataFrame:
    path = metrics_dir / DRILLDOWN_DIRNAME / DRILLDOWN_INDEX_FILENAME
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ensure_required_metric_views(metrics_dir: Path) -> None:
    missing = [
        str(metrics_dir / METRIC_VIEWS_DIRNAME / name)
        for name in REQUIRED_METRIC_VIEW_FILES
        if not (metrics_dir / METRIC_VIEWS_DIRNAME / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required metric view artifacts in metrics_dir: " + ", ".join(missing)
        )


def build_summary_kpis(tables: Dict[str, pd.DataFrame]) -> List[Dict[str, str]]:
    cash_snapshot = tables.get("cash_snapshot", pd.DataFrame())
    debt_snapshot = tables.get("debt_snapshot", pd.DataFrame())
    income_statement_m = tables.get("income_statement_monthly_last6", pd.DataFrame())
    draws_discipline_m = tables.get("draws_discipline_monthly_last6", pd.DataFrame())

    def _pick_row(df: pd.DataFrame, metric_id: str, currency: Optional[str] = None) -> Optional[pd.Series]:
        if df.empty or "metric_id" not in df.columns:
            return None
        sub = df.loc[df["metric_id"] == metric_id]
        if currency is not None and "currency" in sub.columns:
            sub = sub.loc[sub["currency"].astype(str) == str(currency)]
        if sub.empty:
            return None
        return sub.iloc[0]

    def _pick_cash_total() -> Dict[str, str]:
        row = _pick_row(cash_snapshot, "BS.CASH.TOTAL", "ARS")
        if row is None:
            row = _pick_row(cash_snapshot, "BS.CASH.TOTAL")
        if row is None:
            return {"label": "Caja total", "value": "N/A"}
        return {"label": f"Caja total [{row.get('currency','')}]", "value": _fmt_num(row.get("value", pd.NA))}

    def _pick_debt_net() -> Dict[str, str]:
        row = _pick_row(debt_snapshot, "BS.DEBT.NET_PM_POSITION", "USD")
        if row is None:
            row = _pick_row(debt_snapshot, "BS.DEBT.NET_PM_POSITION", "ARS")
        if row is None:
            row = _pick_row(debt_snapshot, "BS.DEBT.NET_PM_POSITION")
        if row is None:
            return {"label": "Posición neta PM", "value": "N/A"}
        return {"label": f"Posición neta PM [{row.get('currency','')}]", "value": _fmt_num(row.get("value", pd.NA))}

    def _pick_metric(metric_id: str, label: str) -> Dict[str, str]:
        if income_statement_m.empty or "metric_id" not in income_statement_m.columns:
            return {"label": label, "value": "N/A"}
        sub = income_statement_m.loc[income_statement_m["metric_id"] == metric_id]
        if sub.empty:
            return {"label": label, "value": "N/A"}
        row = sub.iloc[0]
        last_cols = [c for c in sub.columns if str(c).startswith("20")]
        val = row[last_cols[-1]] if last_cols else pd.NA
        return {"label": f"{label} [{row.get('currency','')}]", "value": _fmt_num(val)}

    kpis = [
        _pick_cash_total(),
        _pick_debt_net(),
        _pick_metric("IS.NET.AFTER_COSTS", "Neto después de costos"),
        _pick_metric("IS.OPEX.TOTAL", "Opex"),
    ]

    if not draws_discipline_m.empty:
        row = draws_discipline_m.iloc[0]
        kpis.append(
            {
                "label": f"Meses en distress [{row.get('currency','')}]",
                "value": str(int(row.get("distress_months", 0))),
            }
        )
    return kpis


def build_human_balance_report(
    metrics_dir: Path,
    write_dir: Path,
    run_root: Path,
    *,
    months: int,
    rent_place_col: str,
    rent_detail_col: str,
    flow_rollup_groupby: Sequence[str],
    include_statuses: Sequence[str],
    noise_floor_by_currency: Dict[str, float],
) -> None:
    ensure_required_metric_views(metrics_dir)
    metric_views_manifest = read_metric_views_manifest(metrics_dir)
    drilldown_index = read_metric_drilldown_index(metrics_dir)
    dd_lookup = drilldown_lookup(drilldown_index)

    ctx = load_human_tables_context(metrics_dir)
    specs, tables = build_human_tables_with_specs(ctx)

    dirs = _ensure_dirs(write_dir)
    drilldown_html_dir = write_dir / "drilldown"
    drilldown_html_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "report_id": REPORT_ID,
        "created_at_utc": _now_iso(),
        "metrics_input": str(metrics_dir),
        "run_root": str(run_root),
        "out_base": str(write_dir),
        "months": months,
        "include_statuses": list(include_statuses),
        "noise_floor_by_currency": noise_floor_by_currency,
        "items": [],
    }

    def _drilldown_link(metric_id: str, period_grain: str, period: str, currency: str, value: Any) -> Optional[str]:
        key = (str(metric_id), str(period_grain), str(period), str(currency))
        row = dd_lookup.get(key)
        if not row:
            return None
        html_relpath = row.get("detail_html_relpath", "")
        if not html_relpath:
            return None
        return f"<a href='{html_relpath}' target='_blank' rel='noopener noreferrer'>{_fmt_num(value)}</a>"

    def _income_statement_renderer(col: str, value: Any, row: pd.Series) -> Optional[str]:
        if str(col).startswith("20"):
            return _drilldown_link(str(row.get("metric_id", "")), "M", str(col), str(row.get("currency", "")), value)
        return None

    def _draws_renderer(col: str, value: Any, row: pd.Series) -> Optional[str]:
        if str(col).startswith("draws_"):
            period = str(col).replace("draws_", "", 1)
            return _drilldown_link("IS.DRAWS.PERSONAL", "M", period, str(row.get("currency", "")), value)
        return None

    for dd_row in drilldown_index.to_dict(orient="records"):
        detail_csv_relpath = str(dd_row.get("detail_csv_relpath", ""))
        detail_csv_path = metrics_dir / detail_csv_relpath
        detail_slug = Path(detail_csv_relpath).stem if detail_csv_relpath else ""
        if not detail_slug:
            continue

        detail_html_relpath = f"drilldown/{detail_slug}.html"
        detail_df = pd.read_csv(detail_csv_path) if detail_csv_path.exists() else pd.DataFrame()
        filter_json = str(dd_row.get("filter_json", ""))

        metadata_html = (
            f"<h1>{dd_row.get('metric_id', '')}</h1>"
            f"<p>run_id: {dd_row.get('run_id', '')}<br>"
            f"period_grain: {dd_row.get('period_grain', '')}<br>"
            f"period: {dd_row.get('period', '')}<br>"
            f"currency: {dd_row.get('currency', '')}<br>"
            f"source_table: {dd_row.get('source_table', '')}<br>"
            f"status: {dd_row.get('status', '')}</p>"
            "<div class='kpi-grid'>"
            f"<div class='kpi'><div class='label'>Target metric value</div><div class='value'>{_fmt_num(dd_row.get('target_metric_value', pd.NA))}</div></div>"
            f"<div class='kpi'><div class='label'>Matched value sum</div><div class='value'>{_fmt_num(dd_row.get('matched_value_sum', pd.NA))}</div></div>"
            f"<div class='kpi'><div class='label'>Difference vs target</div><div class='value'>{_fmt_num(dd_row.get('difference_vs_target', pd.NA))}</div></div>"
            f"<div class='kpi'><div class='label'>Matched rows</div><div class='value'>{_fmt_num(dd_row.get('matched_rows', pd.NA))}</div></div>"
            "</div>"
            f"<h2>Filter spec</h2><div class='pre'>{filter_json}</div>"
            f"<p><a href='../../{detail_csv_relpath}' target='_blank' rel='noopener noreferrer'>Abrir CSV detalle</a></p>"
        )

        detail_html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{dd_row.get('metric_id', '')}</title><style>{DEFAULT_CSS}</style>"
            "</head><body><main class='report'>"
            + metadata_html
            + _df_to_html_fragment(detail_df, "Filas relevantes de ledger_canonical")
            + "</main></body></html>"
        )
        _write_text(detail_html, drilldown_html_dir / f"{detail_slug}.html")
        dd_row["detail_html_relpath"] = detail_html_relpath

    if not drilldown_index.empty:
        updated_index = pd.DataFrame(drilldown_index.to_dict(orient="records"))
        if len(updated_index) == len(drilldown_index):
            updated_index["detail_html_relpath"] = [
                f"drilldown/{Path(str(x)).stem}.html" if str(x) else ""
                for x in updated_index["detail_csv_relpath"]
            ]
            drilldown_index = updated_index
            dd_lookup = drilldown_lookup(drilldown_index)
            drilldown_index.to_csv(metrics_dir / DRILLDOWN_DIRNAME / DRILLDOWN_INDEX_FILENAME, index=False)

    renderer_by_slug: Dict[str, Callable[[str, Any, pd.Series], Optional[str]]] = {
        "income_statement_monthly_last6": _income_statement_renderer,
        "draws_discipline_monthly_last6": _draws_renderer,
    }

    spec_by_slug = {s.slug: s for s in specs}

    for spec in specs:
        df = tables.get(spec.slug, pd.DataFrame())
        base = f"{spec.item_id}__{spec.slug}"
        csv_path = dirs["tables"] / f"{base}.csv"
        html_path = dirs["html"] / f"{base}.html"
        _write_csv(df, csv_path)
        _write_text(
            _df_to_html_fragment(
                df,
                spec.title,
                spec.notes,
                cell_renderer=renderer_by_slug.get(spec.slug),
            ),
            html_path,
        )
        manifest["items"].append(_manifest_item(spec, csv_path, html_path))

    kpis = build_summary_kpis(tables)
    kpi_html = "\n".join(
        f"<div class='kpi'><div class='label'>{x['label']}</div><div class='value'>{x['value']}</div></div>"
        for x in kpis
    )

    standalone_sections = [
        "<h1>Balance humano v2</h1>",
        f"<p>run_id: {ctx.manifest.get('run_id', '')}<br>"
        f"as_of_date: {ctx.manifest.get('as_of_date', '')}<br>"
        f"run_root: {run_root}<br>"
        f"months: {metric_views_manifest.get('months', months)}</p>",
        f"<div class='kpi-grid'>{kpi_html}</div>",
    ]

    for spec in specs:
        df = tables.get(spec.slug, pd.DataFrame())
        note = spec.notes
        if spec.slug == "rent_rollup_by_place_m_last6":
            note = note or f"groupby = Box, Currency, {metric_views_manifest.get('rent_place_col', rent_place_col)}"
        elif spec.slug == "rent_rollup_by_detail_m_last6":
            note = note or f"groupby = Box, Currency, {metric_views_manifest.get('rent_detail_col', rent_detail_col)}"
        elif spec.slug == "flow_type_rollup_m_last6":
            note = note or f"groupby = {metric_views_manifest.get('flow_rollup_groupby', '').replace(',', ' , ') or ' , '.join(flow_rollup_groupby)}"

        standalone_sections.append(
            _df_to_html_fragment(
                df,
                spec.title,
                note,
                cell_renderer=renderer_by_slug.get(spec.slug),
            )
        )

    standalone_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Balance humano v2</title>"
        f"<style>{DEFAULT_CSS}</style>"
        "</head><body><main class='report'>"
        + "\n".join(standalone_sections)
        + "</main></body></html>"
    )

    _write_text(standalone_html, dirs["base"] / "balance_humano_v2.html")
    _write_text(DEFAULT_CSS, dirs["base"] / "report.css")
    _write_text(json.dumps(manifest, indent=2, ensure_ascii=False), dirs["base"] / "story_manifest.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a human-oriented balance report from metrics artifacts and metric_views outputs."
    )
    p.add_argument("--run-root", required=True, help="Accounting run root containing ledger_canonical.csv")
    p.add_argument("--metrics-dir", required=True, help="Directory with metric_registry.csv / metric_values.csv etc.")
    p.add_argument("--write-dir", required=True, help="Output balance report directory.")
    p.add_argument("--months", type=int, default=6, help="Number of monthly periods to surface.")
    p.add_argument("--rent-place-col", default="Lugar", help="Column used for rent rollup by place.")
    p.add_argument("--rent-detail-col", default="Detalle", help="Column used for rent rollup by detail.")
    p.add_argument("--flow-rollup-groupby", default="Flujo,Tipo,Currency", help="Comma-separated groupby columns for generic drilldown.")
    p.add_argument("--include-statuses", default="pagado", help="Comma-separated statuses to include, e.g. pagado or pagado,planeado.")
    p.add_argument("--noise-floor", default="ARS:5000,USD:10", help="Comma-separated thresholds, e.g. ARS:5000,USD:10")
    return p.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    metrics_dir = Path(args.metrics_dir)
    write_dir = Path(args.write_dir)
    run_root = Path(args.run_root)
    include_statuses = tuple(x.strip() for x in args.include_statuses.split(",") if x.strip())
    noise_floor_by_currency = parse_noise_floor(args.noise_floor)
    flow_rollup_groupby = [x.strip() for x in args.flow_rollup_groupby.split(",") if x.strip()]

    LOG.info(
        "Stage start run_root=%s metrics_dir=%s write_dir=%s months=%s",
        run_root,
        metrics_dir,
        write_dir,
        args.months,
    )

    build_human_balance_report(
        metrics_dir=metrics_dir,
        write_dir=write_dir,
        run_root=run_root,
        months=args.months,
        rent_place_col=args.rent_place_col,
        rent_detail_col=args.rent_detail_col,
        flow_rollup_groupby=flow_rollup_groupby,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    )

    LOG.info(
        "Stage finish story_manifest=%s standalone_html=%s",
        write_dir / "story_manifest.json",
        write_dir / "balance_humano_v2.html",
    )


if __name__ == "__main__":
    main()