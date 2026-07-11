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
.wide-time-wrap { overflow-x: auto; max-width: 100%; }
table.wide-time { table-layout: fixed; min-width: 1200px; }
table.wide-time th.period-col, table.wide-time td.period-col {
  position: sticky;
  left: 0;
  background: #fff;
  z-index: 1;
  min-width: 90px;
  width: 90px;
}
table.wide-time th.series-col {
  min-width: 120px;
  max-width: 150px;
  white-space: normal;
  word-break: break-word;
  vertical-align: bottom;
}
.series-head {
  display: flex;
  flex-direction: column;
  gap: 2px;
  line-height: 1.15;
}
.series-head .measure { font-weight: 700; }
.series-head .meta { color: #555; font-size: 11px; font-weight: 400; }
.zero-muted { color: #aaa; }

.digest-table {
  table-layout: fixed;
  width: max-content;
  min-width: 100%;
}

.digest-table col.numeric-col {
  width: 3cm;
  min-width: 3cm;
  max-width: 3cm;
}

.digest-table th,
.digest-table td {
  overflow-wrap: anywhere;
}

.digest-table th.numeric-col,
.digest-table td.numeric-col {
  width: 3cm;
  min-width: 3cm;
  max-width: 3cm;
  text-align: right;
  white-space: nowrap;
}

.digest-table th.text-col,
.digest-table td.text-col {
  min-width: 2.5cm;
}

.digest-scroll {
  overflow-x: auto;
  max-width: 100%;
}


"""



from accounting.logging_utils import configure_logging, get_logger
LOG = get_logger("render linked digest")


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        x = float(value)
    except Exception:
        return html.escape(str(value))
    return f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"



def _monthly_period_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if MONTH_RE.match(str(c))]


def _annual_period_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if YEAR_RE.match(str(c))]


def _id_columns_for_time_matrix(df: pd.DataFrame) -> list[str]:
    preferred = [
        "measure",
        "Currency",
        "pair",
        "Box",
        "Lugar",
        "actor",
        "counterparty",
        "semantic_bucket",
        "semantic_subbucket",
        "metric",
        "line",
        "statement_line",
    ]
    return [c for c in preferred if c in df.columns]


def _should_render_as_transposed_monthly_matrix(table_id: str, df: pd.DataFrame) -> bool:
    month_cols = _monthly_period_columns(df)
    id_cols = _id_columns_for_time_matrix(df)

    if table_id in {
        "monthly_tables_debt_activity_matrix",
        "monthly_tables_debt_position_matrix",
    }:
        return True

    # General heuristic: many month columns and a compact row identity.
    return len(month_cols) >= 6 and len(id_cols) >= 2 and len(id_cols) <= 6



def _period_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if MONTH_RE.match(str(c)) or YEAR_RE.match(str(c))]

def _is_numeric_like_value(value: Any) -> bool:
    if pd.isna(value):
        return True

    text = str(value).strip()
    if text == "":
        return True

    # Accept formatted numbers like 1,234.56 or -1,234
    text = text.replace(",", "")

    try:
        float(text)
        return True
    except Exception:
        return False


def _numeric_only_columns(df: pd.DataFrame) -> set[str]:
    numeric_cols: set[str] = set()

    for col in df.columns:
        s = df[col]

        # Empty columns are not useful to classify as numeric.
        non_empty = s.dropna().astype(str).str.strip()
        non_empty = non_empty[non_empty.ne("")]

        if non_empty.empty:
            continue

        if s.map(_is_numeric_like_value).all():
            numeric_cols.add(str(col))

    return numeric_cols


def _add_table_colgroup_and_classes(
    html_table: str,
    *,
    columns: list[str],
    numeric_cols: set[str],
) -> str:
    colgroup = "<colgroup>"
    for col in columns:
        cls = "numeric-col" if str(col) in numeric_cols else "text-col"
        colgroup += f"<col class='{cls}'>"
    colgroup += "</colgroup>"

    html_table = html_table.replace(
        '<table border="0" class="dataframe digest-table">',
        '<table border="0" class="dataframe digest-table">',
        1,
    )

    html_table = html_table.replace(">\n  <thead>", f">\n  {colgroup}\n  <thead>", 1)

    # Pandas to_html does not put per-column classes on th/td.
    # The colgroup handles width; this wrapper handles scrolling.
    return f"<div class='digest-scroll'>{html_table}</div>"

def _num(value: Any) -> float:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return 0.0
    return float(x)


def _dedupe_preserve_order(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = _cell_str(value).strip()
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


def _lookup_by_cell(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for (table_id, row_id, period, _measure), row in lookup.items():
        by_cell.setdefault((table_id, row_id, period), []).append(row)

    return by_cell


def _pick_link_row(
    *,
    table_id: str,
    row_id: str,
    period: str,
    value: Any,
    candidates: list[str],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    tolerance: float,
) -> dict[str, Any] | None:
    # First: strict key lookup.
    for cand in candidates:
        link_row = lookup.get((table_id, row_id, period, cand))
        if link_row:
            return link_row

    # Second: if there is exactly one OK drilldown for this cell, use it.
    rows = by_cell.get((table_id, row_id, period), [])
    if len(rows) == 1:
        return rows[0]

    # Third: if several candidate measures exist, choose the one whose displayed value matches.
    cell_value = _num(value)
    for row in rows:
        if abs(_num(row.get("display_value")) - cell_value) <= tolerance:
            return row

    return None


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


def _candidate_measures_for_row(row: pd.Series) -> list[str]:
    return _dedupe_preserve_order(
        [
            row.get("measure", ""),
            row.get("metric", ""),
            row.get("line", ""),
            row.get("statement_line", ""),
            row.get("metric_id", ""),

            # Derived/professional rows.
            "value",
            "amount",

            # Flow measures.
            "amount_out",
            "amount_in",
            "net_amount",
            "amount_abs",

            # Debt measures / future debt drilldowns.
            "new_principal",
            "interest_accrued",
            "repayments",
            "adjustments",
            "net_change",
            "open_total",
            "open_principal",
            "open_interest",

            "",

            "cash_close",
            "closing_cash",
            "closing_balance",
            "balance",
            "new_principal",
            "interest_accrued",
            "repayments",
            "adjustments",
            "net_change",
            "open_total",
            "open_principal",
            "open_interest",
            "open_amount",

            "diagnostic_box_level",

        ]
    )


def _render_linked_value(
    *,
    table_id: str,
    row_idx: int,
    row: pd.Series,
    period: str,
    value: Any,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    tolerance: float,
) -> str:
    row_id = _row_id_for(table_id, int(row_idx), row)

    link_row = _pick_link_row(
        table_id=table_id,
        row_id=row_id,
        period=str(period),
        value=value,
        candidates=_candidate_measures_for_row(row),
        lookup=lookup,
        by_cell=by_cell,
        tolerance=tolerance,
    )

    if link_row:
        href = "../" + str(link_row.get("detail_html_relpath", "")).lstrip("./")
        title = (
            f"measure={link_row.get('measure')}; "
            f"matched_rows={link_row.get('matched_rows')}; "
            f"residual={link_row.get('residual')}"
        )
        return (
            f"<a class='drilldown' href='{html.escape(href)}' "
            f"title='{html.escape(title)}'>{_fmt(value)}</a>"
        )

    text = _fmt(value)
    if abs(_num(value)) <= tolerance:
        return f"<span class='zero-muted'>{text}</span>"
    return text


def _series_header_html(row: pd.Series, id_cols: list[str]) -> str:
    measure = html.escape(_cell_str(row.get("measure", "")))
    currency = html.escape(_cell_str(row.get("Currency", "")))

    meta_parts = []
    for col in id_cols:
        if col in {"measure", "Currency"}:
            continue
        value = _cell_str(row.get(col, "")).strip()
        if value:
            meta_parts.append(f"{html.escape(col)}: {html.escape(value)}")

    chunks = ["<div class='series-head'>"]

    if measure:
        chunks.append(f"<span class='measure'>{measure}</span>")

    if currency:
        chunks.append(f"<span class='meta'>{currency}</span>")

    for part in meta_parts:
        chunks.append(f"<span class='meta'>{part}</span>")

    chunks.append("</div>")
    return "".join(chunks)


def _render_transposed_monthly_matrix(
    table_id: str,
    df: pd.DataFrame,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    pack_dir: Path,
    tolerance: float = DEFAULT_TOLERANCE,
) -> str:
    month_cols = _monthly_period_columns(df)
    id_cols = _id_columns_for_time_matrix(df)
    by_cell = _lookup_by_cell(lookup)

    # Drop series whose full monthly path is zero.
    values = (
        df[month_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    keep_series = values.abs().sum(axis=1).gt(tolerance)
    work = df.loc[keep_series].copy()

    LOG.info(
        "[linked-digest] transposed monthly matrix table_id=%s original_series=%s kept_series=%s months=%s id_cols=%s",
        table_id,
        len(df),
        len(work),
        len(month_cols),
        id_cols,
    )

    if work.empty:
        return "<p class='small'>All monthly series are zero; table omitted.</p>"

    # Preserve original row indexes for drilldown lookup.
    original_items = list(work.iterrows())

    html_parts: list[str] = []
    html_parts.append("<div class='wide-time-wrap'>")

    html_parts.append("<div class='wide-time-wrap digest-scroll'>")


    # html_parts.append("<table class='wide-time'>")
    html_parts.append("<table class='wide-time digest-table'>")


    html_parts.append("<colgroup>")
    html_parts.append("<col class='text-col'>")  # period column
    for _row_idx, _row in original_items:
        html_parts.append("<col class='numeric-col'>")
    html_parts.append("</colgroup>")


    # Header
    html_parts.append("<thead><tr>")
    html_parts.append("<th class='period-col'>period</th>")
    for _row_idx, row in original_items:
        html_parts.append(f"<th class='series-col'>{_series_header_html(row, id_cols)}</th>")
    html_parts.append("</tr></thead>")

    # Body
    html_parts.append("<tbody>")
    linked_cells = 0
    nonzero_cells = 0

    for period in month_cols:
        html_parts.append("<tr>")
        html_parts.append(f"<td class='period-col'>{html.escape(str(period))}</td>")

        for row_idx, row in original_items:
            value = row.get(period)

            if abs(_num(value)) > tolerance:
                nonzero_cells += 1

            cell_html = _render_linked_value(
                table_id=table_id,
                row_idx=int(row_idx),
                row=row,
                period=str(period),
                value=value,
                lookup=lookup,
                by_cell=by_cell,
                tolerance=tolerance,
            )

            if "class='drilldown'" in cell_html:
                linked_cells += 1

            html_parts.append(f"<td>{cell_html}</td>")

        html_parts.append("</tr>")

    html_parts.append("</tbody></table></div>")

    LOG.info(
        "[linked-digest] transposed link stats table_id=%s nonzero_cells=%s linked_cells=%s",
        table_id,
        nonzero_cells,
        linked_cells,
    )

    return "\n".join(html_parts)


def _render_table(
    table_id: str,
    df: pd.DataFrame,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    pack_dir: Path,
    tolerance: float = DEFAULT_TOLERANCE,
) -> str:
    period_cols = set(_period_columns(df))
    by_cell = _lookup_by_cell(lookup)

    rendered = df.copy().astype(object)

    linked_cells = 0
    nonzero_period_cells = 0

    for row_idx, row in df.iterrows():
        row_id = _row_id_for(table_id, int(row_idx), row)

        for col in period_cols:
            value = row.get(col)

            if abs(_num(value)) > tolerance:
                nonzero_period_cells += 1

            cell_html = _render_linked_value(
                table_id=table_id,
                row_idx=int(row_idx),
                row=row,
                period=str(col),
                value=value,
                lookup=lookup,
                by_cell=by_cell,
                tolerance=tolerance,
            )

            if "class='drilldown'" in cell_html:
                linked_cells += 1

            rendered.at[row_idx, col] = cell_html



    #         candidates = _dedupe_preserve_order(
    #             [
    #                 row.get("measure", ""),
    #                 row.get("metric", ""),
    #                 row.get("line", ""),
    #                 row.get("statement_line", ""),
    #                 row.get("metric_id", ""),

    #                 # Derived annual/professional tables.
    #                 "value",

    #                 # Monthly statement tables.
    #                 "amount",

    #                 # Canonical semantic split measures.
    #                 "amount_out",
    #                 "amount_in",
    #                 "net_amount",
    #                 "amount_abs",

    #                 # Allow index rows whose measure is blank.
    #                 "",
    #             ]
    #         )

    #         link_row = _pick_link_row(
    #             table_id=table_id,
    #             row_id=row_id,
    #             period=str(col),
    #             value=value,
    #             candidates=candidates,
    #             lookup=lookup,
    #             by_cell=by_cell,
    #             tolerance=tolerance,
    #         )

    #         if link_row:
    #             linked_cells += 1
    #             href = "../" + str(link_row.get("detail_html_relpath", "")).lstrip("./")
    #             title = (
    #                 f"measure={link_row.get('measure')}; "
    #                 f"matched_rows={link_row.get('matched_rows')}; "
    #                 f"residual={link_row.get('residual')}"
    #             )
    #             rendered.at[row_idx, col] = (
    #                 f"<a class='drilldown' href='{html.escape(href)}' "
    #                 f"title='{html.escape(title)}'>{_fmt(value)}</a>"
    #             )
    #         else:
    #             rendered.at[row_idx, col] = _fmt(value)

    # for col in rendered.columns:
    #     if col not in period_cols:
    #         rendered[col] = [
    #             html.escape("" if pd.isna(v) else str(v))
    #             for v in rendered[col]
    #         ]



    LOG.info(
        "[linked-digest] table link stats table_id=%s nonzero_period_cells=%s linked_cells=%s",
        table_id,
        nonzero_period_cells,
        linked_cells,
    )

    # return rendered.to_html(index=False, escape=False, border=0)
    numeric_cols = _numeric_only_columns(df)

    # Period columns should always behave as numeric display columns.
    numeric_cols |= {str(c) for c in period_cols}

    html_table = rendered.to_html(
        index=False,
        escape=False,
        border=0,
        classes="digest-table",
    )

    return _add_table_colgroup_and_classes(
        html_table,
        columns=[str(c) for c in rendered.columns],
        numeric_cols=numeric_cols,
    )



# def build_professional_linked_digest(repo_root: Path, pack_dir: Path, tolerance: float = DEFAULT_TOLERANCE) -> Path:
#     repo_root = Path(repo_root)
#     pack_dir = Path(pack_dir)
#     tables_dir = pack_dir / "tables"
#     digest_dir = pack_dir / "digest"
#     digest_dir.mkdir(parents=True, exist_ok=True)
#     index_path = pack_dir / "drilldown" / INDEX_FILENAME
#     index = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
#     dd = _lookup(index, tolerance)
#     sections: list[str] = []
#     table_paths = list(tables_dir.glob("monthly_tables_*.csv"))
#     for name in ["overview_balance_dashboard.csv", "income_operating_statement.csv", "cash_annual_box_flow_bridge_wide.csv"]:
#         path = tables_dir / name
#         if path.exists():
#             table_paths.append(path)
#     for path in sorted(table_paths):
#         table_id = path.stem
#         df = pd.read_csv(path)
#         csv_rel = path.relative_to(pack_dir).as_posix()
#         sections.append(
#             "<div class='section'>"
#             f"<h2>{html.escape(table_id)}</h2>"
#             f"<p class='small'><a href='../{html.escape(csv_rel)}'>Open source CSV</a></p>"
#             + _render_table(table_id, df, dd, pack_dir)
#             + "</div>"
#         )
#     output = digest_dir / "accounting_professional_pack_digest_linked.html"
#     output.write_text(
#         "<!doctype html><html><head><meta charset='utf-8'>"
#         f"<title>Professional pack linked digest</title><style>{CSS}</style></head><body>"
#         "<h1>Professional pack linked digest</h1>"
#         "<p class='note'>Cells are linked only when professional_drilldown_index.csv marks them ok, has matched rows, and residual is within tolerance. The renderer does not recalculate accounting semantics.</p>"
#         + "\n".join(sections)
#         + "</body></html>",
#         encoding="utf-8",
#     )
#     return output


import time


def build_professional_linked_digest(
    repo_root: Path,
    pack_dir: Path,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Path:
    t0 = time.perf_counter()

    repo_root = Path(repo_root)
    pack_dir = Path(pack_dir)
    tables_dir = pack_dir / "tables"
    digest_dir = pack_dir / "digest"

    LOG.info(
        "[linked-digest] start build_professional_linked_digest "
        "repo_root=%s pack_dir=%s tables_dir=%s tolerance=%s",
        repo_root,
        pack_dir,
        tables_dir,
        tolerance,
    )

    digest_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("[linked-digest] digest_dir ready: %s", digest_dir)

    index_path = pack_dir / "drilldown" / INDEX_FILENAME

    if index_path.exists():
        LOG.info("[linked-digest] loading drilldown index: %s", index_path)
        read_t0 = time.perf_counter()
        index = pd.read_csv(index_path)
        LOG.info(
            "[linked-digest] loaded drilldown index rows=%s cols=%s elapsed=%.2fs",
            len(index),
            len(index.columns),
            time.perf_counter() - read_t0,
        )
    else:
        LOG.warning(
            "[linked-digest] drilldown index missing; digest will render without linked cells: %s",
            index_path,
        )
        index = pd.DataFrame()

    LOG.info("[linked-digest] building drilldown lookup from index")
    lookup_t0 = time.perf_counter()
    dd = _lookup(index, tolerance)
    LOG.info(
        "[linked-digest] drilldown lookup ready type=%s size=%s elapsed=%.2fs",
        type(dd).__name__,
        len(dd) if hasattr(dd, "__len__") else "unknown",
        time.perf_counter() - lookup_t0,
    )

    sections: list[str] = []

    LOG.info("[linked-digest] discovering table CSVs in %s", tables_dir)

    table_paths = list(tables_dir.glob("monthly_tables_*.csv"))
    LOG.info(
        "[linked-digest] discovered monthly table files count=%s",
        len(table_paths),
    )

    extra_names = [
        "overview_balance_dashboard.csv",
        "income_operating_statement.csv",
        "cash_annual_box_flow_bridge_wide.csv",
    ]

    for name in extra_names:
        path = tables_dir / name
        if path.exists():
            table_paths.append(path)
            LOG.info("[linked-digest] added extra table: %s", path)
        else:
            LOG.debug("[linked-digest] optional extra table missing: %s", path)

    table_paths = sorted(table_paths)

    if not table_paths:
        LOG.warning(
            "[linked-digest] no table CSVs found; output will contain only header/note. tables_dir=%s",
            tables_dir,
        )
    else:
        LOG.info(
            "[linked-digest] rendering %s table sections",
            len(table_paths),
        )

    for table_i, path in enumerate(table_paths, start=1):
        table_t0 = time.perf_counter()
        table_id = path.stem

        LOG.info(
            "[linked-digest] table start %s/%s table_id=%s path=%s",
            table_i,
            len(table_paths),
            table_id,
            path,
        )

        try:
            file_size = path.stat().st_size
            LOG.debug(
                "[linked-digest] table file metadata table_id=%s size_bytes=%s",
                table_id,
                file_size,
            )

            read_t0 = time.perf_counter()
            print(path)
            print(path)
            print(path)
            df = pd.read_csv(path)

            if (len(df) < 100) and (len(df.T) <100):
                LOG.info(
                    "[linked-digest] table loaded table_id=%s rows=%s cols=%s elapsed=%.2fs",
                    table_id,
                    len(df),
                    len(df.columns),
                    time.perf_counter() - read_t0,
                )

                csv_rel = path.relative_to(pack_dir).as_posix()

                LOG.info(
                    "[linked-digest] rendering HTML table table_id=%s rows=%s cols=%s",
                    table_id,
                    len(df),
                    len(df.columns),
                )

                render_t0 = time.perf_counter()



                # rendered_table = _render_table(table_id, df, dd, pack_dir)
                # rendered_table = _render_table(table_id, df, dd, pack_dir, tolerance=tolerance)

                if _should_render_as_transposed_monthly_matrix(table_id, df):
                    rendered_table = _render_transposed_monthly_matrix(
                        table_id,
                        df,
                        dd,
                        pack_dir,
                        tolerance=tolerance,
                    )
                else:
                    rendered_table = _render_table(
                        table_id,
                        df,
                        dd,
                        pack_dir,
                        tolerance=tolerance,
                    )



                LOG.info(
                    "[linked-digest] rendered HTML table table_id=%s html_chars=%s elapsed=%.2fs",
                    table_id,
                    len(rendered_table),
                    time.perf_counter() - render_t0,
                )

                section = (
                    "<div class='section'>"
                    f"<h2>{html.escape(table_id)}</h2>"
                    f"<p class='small'><a href='../{html.escape(csv_rel)}'>Open source CSV</a></p>"
                    + rendered_table
                    + "</div>"
                )

                sections.append(section)

                LOG.info(
                "[linked-digest] table done table_id=%s section_chars=%s sections_total=%s elapsed=%.2fs",
                table_id,
                len(section),
                len(sections),
                time.perf_counter() - table_t0,
            )

        except Exception:
            LOG.exception(
                "[linked-digest] table failed table_id=%s path=%s table=%s/%s elapsed=%.2fs",
                table_id,
                path,
                table_i,
                len(table_paths),
                time.perf_counter() - table_t0,
            )
            raise

    output = digest_dir / "accounting_professional_pack_digest_linked.html"

    LOG.info(
        "[linked-digest] assembling final HTML sections=%s output=%s",
        len(sections),
        output,
    )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Professional pack linked digest</title><style>{CSS}</style></head><body>"
        "<h1>Professional pack linked digest</h1>"
        "<p class='note'>Cells are linked only when professional_drilldown_index.csv marks them ok, has matched rows, and residual is within tolerance. The renderer does not recalculate accounting semantics.</p>"
        + "\n".join(sections)
        + "</body></html>"
    )

    LOG.info(
        "[linked-digest] final HTML assembled chars=%s sections=%s",
        len(html_doc),
        len(sections),
    )

    write_t0 = time.perf_counter()
    output.write_text(html_doc, encoding="utf-8")

    LOG.info(
        "[linked-digest] wrote linked digest output=%s size_bytes=%s elapsed_write=%.2fs elapsed_total=%.2fs",
        output,
        output.stat().st_size if output.exists() else "unknown",
        time.perf_counter() - write_t0,
        time.perf_counter() - t0,
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
