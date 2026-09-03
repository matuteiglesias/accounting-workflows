#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_EXCLUDE_PATTERNS = [
    # Full normalized row-level exports are useful as source CSVs, but too redundant for a digest.
    r".*_normalized\.csv$",
    r"cash_levels_all_normalized_ts\.csv$",
    r"cash_fb_related_rows_audit\.csv$",
]

PINNED_TABLE_ORDER = [
    # Overview
    "overview_balance_dashboard.csv",
    "overview_funds_bridge_reconciliation_ars.csv",
    "overview_extended_qa.csv",
    "overview_professional_comments.csv",
    # Cash / liquidity
    "cash_annual_box_flow_bridge_wide.csv",
    "cash_scope_comparison_annual.csv",
    "cash_fb_2024_focus_compact.csv",
    "cash_fb_related_grouped_audit.csv",
    "cash_unknown_review_pm_ars_reconciliation.csv",
    "cash_unknown_review_pm_ars_check.csv",
    "cash_validated_levels_from_metrics.csv",
    "cash_liquidity_extended_qa.csv",
    # Income
    "income_operating_statement.csv",
    "income_rent_by_property.csv",
    "income_opex_by_category.csv",
    "income_operations_extended_qa.csv",
    "income_professional_comments.csv",
    # Debt
    "debt_stock_summary.csv",
    "debt_rollforward.csv",
    "debt_open_items_action_list.csv",
    "debt_reconciliation_summary.csv",
    "debt_extended_qa.csv",
    "debt_professional_comments.csv",
    # Monthly tables
    "monthly_tables_operating_statement_matrix_ars.csv",
    "monthly_tables_operating_statement_matrix.csv",
    "monthly_tables_flow_bucket_all_measures.csv",
    "monthly_tables_flow_subbucket_all_measures.csv",
    "monthly_tables_draws_by_box_amount_out.csv",
    "monthly_tables_draws_by_type_amount_out.csv",
    "monthly_tables_fb_bridge_matrix.csv",
    "monthly_tables_pm_stress_matrix.csv",
    "monthly_tables_household_bridge_matrix.csv",
    "monthly_tables_opex_by_type_amount_out.csv",
    "monthly_tables_fx_treasury_compact.csv",
    "monthly_tables_unknown_review_net_matrix.csv",
    "monthly_tables_debt_position_matrix.csv",
    "monthly_tables_debt_activity_matrix.csv",
    "monthly_tables_cash_close_matrix.csv",
    "monthly_tables_diagnostic_box_level_matrix.csv",
    "monthly_tables_index.csv",
    "monthly_tables_professional_comments.csv",
    # Monthly chart notebook outputs, when present.
    "monthly_dynamics_operating_selected_ars.csv",
    "monthly_dynamics_bucket_net_ars.csv",
    "monthly_dynamics_draws_by_box_ars.csv",
    "monthly_dynamics_fb_bridge_ars.csv",
    "monthly_dynamics_pm_stress_bridge_ars.csv",
    "monthly_dynamics_household_bridge_ars.csv",
    "monthly_dynamics_opex_by_type_ars.csv",
    "monthly_dynamics_fx_audit.csv",
    "monthly_dynamics_unknown_review_ars.csv",
    "monthly_dynamics_debt_open_by_pair_usd.csv",
    "monthly_dynamics_debt_activity_total_usd.csv",
    "monthly_dynamics_chart_index.csv",
]


REPORT_ORDER = [
    "01_balance_dashboard_overview",
    "02_cash_and_liquidity",
    "03_income_rent_and_operations",
    "04_debt_open_items_and_reconciliation",
    "06_monthly_dynamics_bar_charts",
    "07_monthly_dynamics_tables",
]


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip())
    return re.sub(r"-+", "-", s).strip("-").lower() or "section"


def esc(x: object) -> str:
    return html.escape("" if x is None else str(x))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def human_title(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"^\d+[_-]?", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip().title()


def table_category(name: str) -> str:
    n = name.lower()
    if n.startswith("overview_"):
        return "01 overview"
    if n.startswith("cash_"):
        return "02 cash and liquidity"
    if n.startswith("income_"):
        return "03 income and operations"
    if n.startswith("debt_"):
        return "04 debt"
    if n.startswith("monthly_tables_"):
        return "07 monthly matrices"
    if n.startswith("monthly_dynamics_"):
        return "06 monthly dynamics"
    if "qa" in n or "reconciliation" in n or "unknown" in n:
        return "90 QA / reconciliation"
    if "normalized" in n or "audit" in n or "detail" in n:
        return "95 detailed audit / appendix"
    return "80 other tables"


def preferred_sort_key(path: Path) -> tuple:
    name = path.name
    try:
        pinned = PINNED_TABLE_ORDER.index(name)
    except ValueError:
        pinned = 9999
    return (pinned, table_category(name), name)


def report_sort_key(path: Path) -> tuple:
    stem = path.stem
    for i, prefix in enumerate(REPORT_ORDER):
        if stem.startswith(prefix):
            return (i, stem)
    m = re.match(r"^(\d+)", stem)
    return (int(m.group(1)) if m else 999, stem)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"<p class='warn'>Could not read {esc(path)}: {esc(exc)}</p>"


def extract_body(html_text: str) -> str:
    body = re.search(r"<body[^>]*>(.*?)</body>", html_text, flags=re.S | re.I)
    if body:
        return body.group(1)
    # Drop a full html/head wrapper if present, but tolerate fragment HTML.
    html_text = re.sub(r"<!doctype[^>]*>", "", html_text, flags=re.I)
    html_text = re.sub(r"<html[^>]*>|</html>", "", html_text, flags=re.I)
    html_text = re.sub(r"<head[^>]*>.*?</head>", "", html_text, flags=re.S | re.I)
    return html_text


def file_row_count(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def format_value(v: object) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
        if abs(x) >= 1000:
            if abs(x - round(x)) < 1e-9:
                return f"{int(round(x)):,}".replace(",", ".")
            return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.2f}".replace(".", ",")
    return str(v)


def df_to_html(df: pd.DataFrame, *, classes: str = "data-table") -> str:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(format_value)
        else:
            out[c] = out[c].fillna("").astype(str)
    return out.to_html(index=False, escape=True, classes=classes, border=0)


def read_csv_preview(
    path: Path,
    *,
    max_full_rows: int,
    max_full_cells: int,
    preview_rows: int,
) -> tuple[pd.DataFrame, dict]:
    rows = file_row_count(path)
    try:
        head = pd.read_csv(path, nrows=preview_rows)
    except Exception as exc:
        return pd.DataFrame({"error": [f"{type(exc).__name__}: {exc}"]}), {
            "rows": rows,
            "cols": 0,
            "mode": "error",
        }

    cols = len(head.columns)
    full = rows <= max_full_rows and rows * max(cols, 1) <= max_full_cells

    if full:
        try:
            df = pd.read_csv(path)
            return df, {"rows": rows, "cols": len(df.columns), "mode": "full"}
        except Exception as exc:
            return pd.DataFrame({"error": [f"{type(exc).__name__}: {exc}"]}), {
                "rows": rows,
                "cols": cols,
                "mode": "error",
            }

    # Preview: head + tail with separator.
    try:
        tail = pd.read_csv(path, skiprows=max(rows - preview_rows + 1, 1)) if rows > preview_rows else pd.DataFrame()
        if not tail.empty:
            # Re-read tail with header if skiprows made weird columns.
            tail = pd.read_csv(path).tail(preview_rows)
        sep = pd.DataFrame([{c: "…" for c in head.columns}])
        df = pd.concat([head, sep, tail], ignore_index=True) if not tail.empty else head
    except Exception:
        df = head
    return df, {"rows": rows, "cols": cols, "mode": "preview"}


def should_exclude_from_main(name: str, exclude_patterns: Iterable[str]) -> bool:
    return any(re.match(pat, name) for pat in exclude_patterns)


def collect_tables(pack: Path, *, include_appendix: bool) -> list[Path]:
    tables_dir = pack / "tables"
    if not tables_dir.exists():
        return []
    files = sorted(tables_dir.glob("*.csv"), key=preferred_sort_key)
    if include_appendix:
        return files
    main = []
    appendix = []
    for p in files:
        if should_exclude_from_main(p.name, DEFAULT_EXCLUDE_PATTERNS):
            appendix.append(p)
        else:
            main.append(p)
    return main + appendix


def build_html_digest(
    pack: Path,
    out_path: Path,
    *,
    project_root: Path | None = None,
    max_full_rows: int = 260,
    max_full_cells: int = 14000,
    preview_rows: int = 30,
    include_appendix: bool = True,
    title: str = "Accounting professional pack digest",
) -> Path:
    pack = pack.resolve()
    root = project_root.resolve() if project_root else pack.parents[2] if len(pack.parents) >= 3 else pack
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_reports = sorted((pack / "html").glob("*.html"), key=report_sort_key) if (pack / "html").exists() else []
    md_reports = sorted((pack / "markdown").glob("*.md"), key=report_sort_key) if (pack / "markdown").exists() else []
    tables = collect_tables(pack, include_appendix=include_appendix)

    # Prefer HTML reports, then include markdown-only reports not already represented.
    html_stems = {p.stem for p in html_reports}
    md_only = [p for p in md_reports if p.stem not in html_stems]

    toc = []
    sections = []

    def add_toc(section_id: str, label: str, level: int = 1) -> None:
        toc.append((section_id, label, level))

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    intro_id = "intro"
    add_toc(intro_id, "Overview", 1)

    # Report section
    reports_id = "reports"
    add_toc(reports_id, "Short reports", 1)
    report_cards = []
    for p in html_reports:
        sid = "report-" + slugify(p.stem)
        add_toc(sid, human_title(p.stem), 2)
        body = extract_body(read_text(p))
        report_cards.append(f"""
        <section class="report-block" id="{sid}">
          <div class="section-kicker">Report · <a href="{esc(rel(p, out_path.parent))}">{esc(rel(p, root))}</a></div>
          <h2>{esc(human_title(p.stem))}</h2>
          <div class="embedded-report">{body}</div>
        </section>
        """)
    for p in md_only:
        sid = "report-" + slugify(p.stem)
        add_toc(sid, human_title(p.stem), 2)
        txt = read_text(p)
        report_cards.append(f"""
        <section class="report-block" id="{sid}">
          <div class="section-kicker">Markdown report · <a href="{esc(rel(p, out_path.parent))}">{esc(rel(p, root))}</a></div>
          <h2>{esc(human_title(p.stem))}</h2>
          <pre class="markdown-pre">{esc(txt)}</pre>
        </section>
        """)

    # Table index metadata
    table_meta = []
    for p in tables:
        rows = file_row_count(p)
        try:
            cols = len(pd.read_csv(p, nrows=1).columns)
        except Exception:
            cols = 0
        excluded = should_exclude_from_main(p.name, DEFAULT_EXCLUDE_PATTERNS)
        table_meta.append({
            "category": table_category(p.name),
            "table": p.name,
            "rows": rows,
            "cols": cols,
            "display": "appendix/preview" if excluded else "full-or-preview",
            "path": rel(p, root),
        })
    table_index_df = pd.DataFrame(table_meta)

    table_index_id = "table-index"
    add_toc(table_index_id, "Table index", 1)

    # Tables by category
    table_sections = []
    current_cat = None
    for p in tables:
        cat = table_category(p.name)
        if cat != current_cat:
            current_cat = cat
            cat_id = "tables-" + slugify(cat)
            add_toc(cat_id, cat, 1)
            table_sections.append(f'<h1 id="{cat_id}">{esc(cat)}</h1>')

        sid = "table-" + slugify(p.name)
        add_toc(sid, p.name, 2)

        is_appendix = should_exclude_from_main(p.name, DEFAULT_EXCLUDE_PATTERNS)
        local_max_rows = min(max_full_rows, 80) if is_appendix else max_full_rows
        local_max_cells = min(max_full_cells, 6000) if is_appendix else max_full_cells

        df, info = read_csv_preview(
            p,
            max_full_rows=local_max_rows,
            max_full_cells=local_max_cells,
            preview_rows=preview_rows,
        )

        mode = info.get("mode", "")
        rows = info.get("rows", 0)
        cols = info.get("cols", 0)
        warning = ""
        if mode == "preview":
            warning = f"<span class='pill warn'>preview: {rows} rows × {cols} cols</span>"
        elif mode == "full":
            warning = f"<span class='pill ok'>full: {rows} rows × {cols} cols</span>"
        else:
            warning = f"<span class='pill err'>{esc(mode)}</span>"

        table_sections.append(f"""
        <section class="table-block" id="{sid}">
          <div class="section-kicker">{esc(cat)} · <a href="{esc(rel(p, out_path.parent))}">{esc(rel(p, root))}</a></div>
          <h2>{esc(p.name)}</h2>
          <div class="meta-line">{warning}</div>
          <div class="table-wrap">
            {df_to_html(df)}
          </div>
        </section>
        """)

    toc_html = "\n".join(
        f'<a class="toc-l{level}" href="#{esc(sid)}">{esc(label)}</a>'
        for sid, label, level in toc
    )

    css = """
    :root {
      --bg:#fff; --fg:#111; --muted:#555; --border:#d8d8d8; --soft:#f7f7f7;
      --accent:#0b57d0; --warn:#8a5a00; --ok:#1a6b2b; --err:#8f0000;
      --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--fg); font-family:var(--font); }
    .layout { display:grid; grid-template-columns: 290px minmax(0, 1fr); gap:0; }
    aside { position:sticky; top:0; height:100vh; overflow:auto; border-right:1px solid var(--border); padding:18px 14px; background:#fbfbfb; }
    main { max-width: none; padding:26px 28px 80px; overflow:hidden; }
    h1 { font-size:26px; margin:32px 0 12px; border-top:2px solid var(--border); padding-top:18px; }
    h2 { font-size:20px; margin:24px 0 10px; }
    h3 { font-size:16px; margin:18px 0 8px; }
    p { color:var(--muted); line-height:1.45; }
    a { color:var(--accent); text-decoration:none; }
    a:hover { text-decoration:underline; }
    .title { font-size:30px; margin:0 0 6px; border:0; padding:0; }
    .subtitle { color:var(--muted); margin:0 0 20px; }
    .toc-title { font-weight:700; margin:0 0 10px; }
    aside a { display:block; color:#222; padding:4px 2px; border-radius:4px; font-size:13px; }
    aside a:hover { background:#eee; text-decoration:none; }
    .toc-l1 { font-weight:700; margin-top:8px; }
    .toc-l2 { padding-left:14px; color:#444; font-size:12px; }
    .meta-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; margin:18px 0 26px; }
    .meta-card { border:1px solid var(--border); background:var(--soft); border-radius:10px; padding:12px; }
    .meta-card .label { font-size:12px; color:var(--muted); }
    .meta-card .value { font-family:var(--mono); font-size:13px; margin-top:5px; word-break:break-word; }
    .section-kicker { font-size:12px; color:var(--muted); font-family:var(--mono); margin-bottom:8px; }
    .report-block, .table-block { margin:0 0 36px; padding-bottom:18px; border-bottom:1px solid #eee; }
    .embedded-report { border:1px solid var(--border); border-radius:10px; padding:14px; overflow:auto; background:#fff; }
    .embedded-report table { width:100%; border-collapse:collapse; font-size:12px; }
    .embedded-report th, .embedded-report td { border:1px solid var(--border); padding:5px 7px; vertical-align:top; }
    .embedded-report th { background:#f0f0f0; }
    .markdown-pre { white-space:pre-wrap; border:1px solid var(--border); border-radius:10px; padding:12px; background:#fafafa; font-family:var(--mono); font-size:12px; }
    .table-wrap { width:100%; max-height:none; overflow:auto; border:1px solid var(--border); border-radius:10px; }
    table.data-table { border-collapse:collapse; width:max-content; min-width:100%; font-size:11px; }
    table.data-table th, table.data-table td { border:1px solid var(--border); padding:4px 6px; vertical-align:top; white-space:nowrap; }
    table.data-table th { position:sticky; top:0; background:#efefef; z-index:2; text-align:left; }
    table.data-table tr:nth-child(even) td { background:#fbfbfb; }
    .meta-line { margin:6px 0 10px; }
    .pill { display:inline-block; font-size:12px; border-radius:999px; padding:3px 8px; border:1px solid var(--border); background:#f7f7f7; }
    .pill.ok { color:var(--ok); border-color:#b8d8bf; background:#f2fff5; }
    .pill.warn { color:var(--warn); border-color:#e6d5a8; background:#fff9ea; }
    .pill.err { color:var(--err); border-color:#e2b5b5; background:#fff1f1; }
    .note { border-left:4px solid var(--accent); padding:8px 12px; background:#f7faff; color:#333; margin:14px 0; }
    @media print {
      .layout { display:block; }
      aside { display:none; }
      main { padding:0; }
      .table-wrap { overflow:visible; }
      table.data-table { font-size:8px; }
      a { color:#000; }
    }
    """

    full_html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{esc(title)}</title>
<style>{css}</style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="toc-title">Índice</div>
    {toc_html}
  </aside>
  <main>
    <section id="{intro_id}">
      <h1 class="title">{esc(title)}</h1>
      <p class="subtitle">Digest de reportes, tablas y matrices mensuales. Generado {esc(generated)}.</p>
      <div class="meta-grid">
        <div class="meta-card"><div class="label">Pack</div><div class="value">{esc(rel(pack, root))}</div></div>
        <div class="meta-card"><div class="label">HTML reports</div><div class="value">{len(html_reports)}</div></div>
        <div class="meta-card"><div class="label">Markdown reports</div><div class="value">{len(md_reports)}</div></div>
        <div class="meta-card"><div class="label">CSV tables</div><div class="value">{len(tables)}</div></div>
      </div>
      <div class="note">
        Este documento es para revisión humana y auditoría visual. No recalcula la contabilidad.
        Las tablas grandes se muestran como preview y quedan enlazadas al CSV fuente.
      </div>
    </section>

    <section id="{reports_id}">
      <h1>Short reports</h1>
      {''.join(report_cards) if report_cards else '<p>No report HTML/Markdown files found.</p>'}
    </section>

    <section id="{table_index_id}">
      <h1>Table index</h1>
      <div class="table-wrap">
        {df_to_html(table_index_df)}
      </div>
    </section>

    {''.join(table_sections)}
  </main>
</div>
</body>
</html>
"""
    out_path.write_text(full_html, encoding="utf-8")

    # Also write a CSV index next to the digest for programmatic checking.
    if not table_index_df.empty:
        table_index_df.to_csv(out_path.with_suffix(".tables.csv"), index=False)

    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build one large scrolling HTML digest from professional_pack/latest.")
    ap.add_argument("--pack", type=Path, default=Path("out/professional_pack/latest"), help="Professional pack directory.")
    ap.add_argument("--out", type=Path, default=None, help="Output HTML path.")
    ap.add_argument("--project-root", type=Path, default=Path("."), help="Repo root for relative links.")
    ap.add_argument("--max-full-rows", type=int, default=260, help="Maximum rows for full inline display.")
    ap.add_argument("--max-full-cells", type=int, default=14000, help="Maximum cells for full inline display.")
    ap.add_argument("--preview-rows", type=int, default=30, help="Rows shown at head and tail for previews.")
    ap.add_argument("--no-appendix", action="store_true", help="Do not include large appendix/normalized CSV previews.")
    ap.add_argument("--title", default="Accounting professional pack digest")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pack = args.pack
    out = args.out or (pack / "digest" / "accounting_professional_pack_digest.html")
    result = build_html_digest(
        pack=pack,
        out_path=out,
        project_root=args.project_root,
        max_full_rows=args.max_full_rows,
        max_full_cells=args.max_full_cells,
        preview_rows=args.preview_rows,
        include_appendix=not args.no_appendix,
        title=args.title,
    )
    print(result)
    print(result.with_suffix(".tables.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
