from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .drilldown import DEFAULT_TOLERANCE, INDEX_FILENAME

MONTH_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")
YEAR_RE = re.compile(r"^20\d{2}$")

CSS = """
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 28px; color: #111; }
h1 { margin-bottom: 4px; }
.note { color: #555; max-width: 1000px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 32px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
th { background: #f4f4f4; position: sticky; top: 0; }
a.drilldown { color: #0b57d0; text-decoration: none; font-weight: 600; }
a.drilldown:hover { text-decoration: underline; }
.section { border-top: 1px solid #ddd; padding-top: 18px; margin-top: 24px; }
.small { font-size: 12px; color: #555; }
"""


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        x = float(value)
    except Exception:
        return html.escape(str(value))
    return f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"


def _period_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if MONTH_RE.match(str(c)) or YEAR_RE.match(str(c))]


def _cell_str(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _lookup(index: pd.DataFrame, tolerance: float) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if index.empty:
        return out
    for row in index.to_dict(orient="records"):
        try:
            residual = abs(float(row.get("residual", 0.0)))
            matched_rows = int(float(row.get("matched_rows", 0)))
        except Exception:
            continue
        if str(row.get("status")) != "ok" or matched_rows <= 0 or residual > tolerance:
            continue
        key = (_cell_str(row.get("table_id", "")), _cell_str(row.get("row_id", "")), _cell_str(row.get("period", "")), _cell_str(row.get("measure", "")))
        out[key] = row
    return out


def _row_id_for(table_id: str, row_idx: int, row: pd.Series) -> str:
    from .drilldown import row_context_id
    return row_context_id(table_id, row_idx, row)


def _render_table(table_id: str, df: pd.DataFrame, lookup: dict[tuple[str, str, str, str], dict[str, Any]], pack_dir: Path) -> str:
    month_cols = set(_period_columns(df))
    rendered = df.copy().astype(object)
    for row_idx, row in df.iterrows():
        row_id = _row_id_for(table_id, row_idx, row)
        for col in month_cols:
            measure = _cell_str(row.get("measure", ""))
            metric = _cell_str(row.get("metric", row.get("line", row.get("statement_line", ""))))
            metric_id = _cell_str(row.get("metric_id", ""))
            candidates = [measure, metric, metric_id]
            for fallback in ["amount_out", "amount_in", "net_amount", "amount_abs"]:
                if fallback not in candidates:
                    candidates.append(fallback)
            link_row = None
            for cand in candidates:
                link_row = lookup.get((table_id, row_id, str(col), cand))
                if link_row:
                    break
            value = row.get(col)
            if link_row:
                href = "../" + str(link_row.get("detail_html_relpath", "")).lstrip("./")
                title = f"matched_rows={link_row.get('matched_rows')}; residual={link_row.get('residual')}"
                rendered.at[row_idx, col] = f"<a class='drilldown' href='{html.escape(href)}' title='{html.escape(title)}'>{_fmt(value)}</a>"
            else:
                rendered.at[row_idx, col] = _fmt(value)
    for col in rendered.columns:
        if col not in month_cols:
            rendered[col] = [html.escape("" if pd.isna(v) else str(v)) for v in rendered[col]]
    return rendered.to_html(index=False, escape=False, border=0)


def build_professional_linked_digest(repo_root: Path, pack_dir: Path, tolerance: float = DEFAULT_TOLERANCE) -> Path:
    repo_root = Path(repo_root)
    pack_dir = Path(pack_dir)
    tables_dir = pack_dir / "tables"
    digest_dir = pack_dir / "digest"
    digest_dir.mkdir(parents=True, exist_ok=True)
    index_path = pack_dir / "drilldown" / INDEX_FILENAME
    index = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
    dd = _lookup(index, tolerance)
    sections: list[str] = []
    table_paths = list(tables_dir.glob("monthly_tables_*.csv"))
    for name in ["overview_balance_dashboard.csv", "income_operating_statement.csv", "cash_annual_box_flow_bridge_wide.csv"]:
        path = tables_dir / name
        if path.exists():
            table_paths.append(path)
    for path in sorted(table_paths):
        table_id = path.stem
        df = pd.read_csv(path)
        csv_rel = path.relative_to(pack_dir).as_posix()
        sections.append(
            "<div class='section'>"
            f"<h2>{html.escape(table_id)}</h2>"
            f"<p class='small'><a href='../{html.escape(csv_rel)}'>Open source CSV</a></p>"
            + _render_table(table_id, df, dd, pack_dir)
            + "</div>"
        )
    output = digest_dir / "accounting_professional_pack_digest_linked.html"
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Professional pack linked digest</title><style>{CSS}</style></head><body>"
        "<h1>Professional pack linked digest</h1>"
        "<p class='note'>Cells are linked only when professional_drilldown_index.csv marks them ok, has matched rows, and residual is within tolerance. The renderer does not recalculate accounting semantics.</p>"
        + "\n".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render professional pack digest with drilldown links.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    print(build_professional_linked_digest(args.repo_root, args.pack, args.tolerance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
