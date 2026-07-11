from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from accounting.logging_utils import configure_logging, get_logger
LOG = get_logger("professional issue digest")

DEFAULT_TOLERANCE = 1e-6
INDEX_FILENAME = "professional_drilldown_index.csv"
DEFAULT_STATUSES = ("residual_warning", "unsupported", "error")
ISSUES_FILENAME = "professional_drilldown_issues.csv"
SUMMARY_FILENAME = "professional_drilldown_issue_summary.csv"
HTML_FILENAME = "accounting_professional_drilldown_issues.html"

ISSUE_COLUMNS = [
    "issue_id",
    "status",
    "severity",
    "table_id",
    "drilldown_id",
    "period",
    "Currency",
    "measure",
    "display_value",
    "matched_value_sum",
    "residual",
    "residual_abs",
    "residual_pct_of_display",
    "matched_rows",
    "source_artifact",
    "lineage_level",
    "detail_csv_relpath",
    "detail_html_relpath",
    "filter_reason",
    "filter_note",
    "row_label",
    "row_context_compact",
    "next_action_hint",
    "detail_csv_exists",
    "detail_html_exists",
    "detail_amount_in_sum",
    "detail_amount_out_sum",
    "detail_net_amount_sum",
    "detail_amount_abs_sum",
    "detail_n_tx_sum",
    "semantic_bucket_counts",
    "semantic_subbucket_counts",
    "rule_id_counts",
    "rule_ids_counts",
]

SUMMARY_COLUMNS = [
    "status",
    "severity",
    "table_id",
    "filter_reason",
    "next_action_hint",
    "n",
    "residual_abs_sum",
    "display_value_abs_sum",
]

ROW_CONTEXT_KEYS = [
    "line",
    "metric",
    "metric_id",
    "statement_line",
    "Box",
    "Currency",
    "semantic_bucket",
    "semantic_subbucket",
    "cash_path",
    "Lugar",
    "dimension_name",
    "dimension_value",
]


NUMERIC_DETAIL_COLUMNS = {
    "amount_in": "detail_amount_in_sum",
    "amount_out": "detail_amount_out_sum",
    "net_amount": "detail_net_amount_sum",
    "amount_abs": "detail_amount_abs_sum",
    "n_tx": "detail_n_tx_sum",
}

COUNT_DETAIL_COLUMNS = {
    "semantic_bucket": "semantic_bucket_counts",
    "semantic_subbucket": "semantic_subbucket_counts",
    "rule_id": "rule_id_counts",
    "rule_ids": "rule_ids_counts",
}

CSS = """
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 28px; color: #111; }
h1 { margin-bottom: 4px; }
.note { color: #555; max-width: 1000px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 28px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
th { background: #f4f4f4; text-align: left; position: sticky; top: 0; }
.kpis { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }
.kpi { border: 1px solid #ddd; border-radius: 10px; padding: 10px 14px; min-width: 130px; }
.kpi .label { color: #555; font-size: 12px; }
.kpi .value { font-size: 22px; font-weight: 650; }
.high { background: #fff0f0; }
.medium { background: #fff8e6; }
.low { background: #f6fbff; }
.links a { margin-right: 8px; }
.small { color: #555; font-size: 12px; }
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(raw: Any) -> dict[str, Any]:
    text = "" if raw is None else str(raw).strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _num(value: Any) -> float:
    try:
        x = float(str(value).replace(",", ""))
    except Exception:
        return 0.0
    return 0.0 if math.isnan(x) else x


def _fmt_num(value: Any) -> str:
    x = _num(value)
    return f"{x:.10g}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({col: _text(row.get(col, "")) for col in columns} for row in rows)


def _relpath_exists(pack: Path, relpath: str) -> bool:
    return bool(relpath) and (pack / relpath).exists()


def _compact_counts(counter: Counter[str], limit: int = 8) -> str:
    parts = []
    for key, n in counter.most_common(limit):
        if key:
            parts.append(f"{key}={n}")
    return "; ".join(parts)


def _enrich_detail_csv(pack: Path, relpath: str) -> dict[str, str]:
    enrichment = {out_col: "" for out_col in NUMERIC_DETAIL_COLUMNS.values()}
    enrichment.update({out_col: "" for out_col in COUNT_DETAIL_COLUMNS.values()})
    if not relpath:
        return enrichment
    path = pack / relpath
    if not path.exists():
        return enrichment
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            numeric_sums = defaultdict(float)
            counters: dict[str, Counter[str]] = {col: Counter() for col in COUNT_DETAIL_COLUMNS}
            for row in reader:
                for col in NUMERIC_DETAIL_COLUMNS:
                    if col in row:
                        numeric_sums[col] += _num(row.get(col))
                for col in COUNT_DETAIL_COLUMNS:
                    value = _text(row.get(col, "")).strip()
                    if value:
                        counters[col][value] += 1
    except Exception as exc:
        LOG.warning("failed reading detail csv %s: %s", path, exc)
        return enrichment
    for col, out_col in NUMERIC_DETAIL_COLUMNS.items():
        if col in numeric_sums:
            enrichment[out_col] = _fmt_num(numeric_sums[col])
    for col, out_col in COUNT_DETAIL_COLUMNS.items():
        enrichment[out_col] = _compact_counts(counters[col])
    return enrichment


def _row_label(row: dict[str, str], row_context: dict[str, Any], filter_data: dict[str, Any]) -> str:
    bucket_pair = "/".join(x for x in [_text(row_context.get("semantic_bucket")).strip(), _text(row_context.get("semantic_subbucket")).strip()] if x)
    candidates = [
        row_context.get("line"),
        row_context.get("metric"),
        row_context.get("metric_id"),
        row_context.get("statement_line"),
        filter_data.get("statement_line"),
        filter_data.get("metric_id"),
        bucket_pair,
        row.get("drilldown_id"),
    ]
    return next((_text(x).strip() for x in candidates if _text(x).strip()), "")


def _compact_context(row_context: dict[str, Any]) -> str:
    return "; ".join(f"{key}={_text(row_context.get(key)).strip()}" for key in ROW_CONTEXT_KEYS if _text(row_context.get(key)).strip())


def _severity(status: str, display_value: float, residual: float, tolerance: float) -> str:
    if status == "error" or (status == "residual_warning" and abs(residual) >= 100000) or (status == "unsupported" and abs(display_value) > tolerance):
        return "high"
    if (status == "residual_warning" and abs(residual) > tolerance) or (status == "unsupported" and abs(display_value) > tolerance):
        return "medium"
    return "low"


def _next_action_hint(status: str, filter_reason: str, matched_rows: int, detail_csv_exists: bool, detail_enrichment: dict[str, str]) -> str:
    reason = filter_reason.casefold()
    if status == "unsupported":
        if "unsupported cash bridge line" in reason:
            return "Add/adjust _cash_bridge_line_spec mapping for this row_label."
        if "missing currency" in reason:
            return "Missing Currency prevents safe drilldown; inspect source table row."
        if "stock/cash metric is not a flow drilldown" in reason:
            return "Expected unsupported stock/cash diagnostic; likely no action unless this should become stock lineage."
        return "Unsupported cell; inspect filter_json and row_context_json."
    if status == "residual_warning":
        has_detail_sums = any(detail_enrichment.get(col, "") != "" for col in NUMERIC_DETAIL_COLUMNS.values())
        if matched_rows == 0:
            return "Filter found no rows; inspect semantic mapping or source artifact freshness."
        if matched_rows > 0 and has_detail_sums:
            return "Rows found but sum does not match displayed value; compare source table vs drilldown source artifacts."
        if not detail_csv_exists:
            return "Detail CSV missing; regenerate professional drilldowns cleanly."
        return "Residual warning; inspect detail rows and source artifacts."
    if status == "error":
        return "Builder error or missing source; inspect source_artifact and manifest."
    return "Inspect drilldown issue."


def build_issue_rows(pack: Path, statuses: set[str], tolerance: float) -> list[dict[str, Any]]:
    index_path = pack / "drilldown" / INDEX_FILENAME
    index_rows = _read_csv(index_path)
    LOG.info("loaded index rows: %s", len(index_rows))
    issue_source_rows = [row for row in index_rows if _text(row.get("status")).strip() in statuses]
    LOG.info("filtered issue rows: %s", len(issue_source_rows))

    issues: list[dict[str, Any]] = []
    enriched_count = 0
    for n, row in enumerate(issue_source_rows, start=1):
        status = _text(row.get("status")).strip()
        filter_data = _safe_json(row.get("filter_json"))
        row_context = _safe_json(row.get("row_context_json"))
        detail_csv_relpath = _text(row.get("detail_csv_relpath") or row.get("detail_csv_path")).strip()
        detail_html_relpath = _text(row.get("detail_html_relpath") or row.get("detail_html_path")).strip()
        detail_csv_exists = _relpath_exists(pack, detail_csv_relpath)
        detail_html_exists = _relpath_exists(pack, detail_html_relpath)
        detail_enrichment = _enrich_detail_csv(pack, detail_csv_relpath)
        if detail_csv_exists:
            enriched_count += 1
        display_value = _num(row.get("display_value"))
        matched_value_sum = _num(row.get("matched_value_sum"))
        residual = _num(row.get("residual", display_value - matched_value_sum))
        matched_rows = int(_num(row.get("matched_rows")))
        filter_reason = _text(filter_data.get("reason") or row.get("filter_reason") or row.get("reason")).strip()
        filter_note = _text(filter_data.get("filter_note") or row.get("filter_note")).strip()
        if not filter_reason and _text(filter_data.get("unsupported")).strip():
            filter_reason = _text(filter_data.get("unsupported")).strip()
        currency = _text(row.get("Currency") or row_context.get("Currency")).strip()
        issue = {
            "issue_id": f"issue_{n:05d}",
            "status": status,
            "severity": _severity(status, display_value, residual, tolerance),
            "table_id": row.get("table_id", ""),
            "drilldown_id": row.get("drilldown_id", ""),
            "period": row.get("period", ""),
            "Currency": currency,
            "measure": row.get("measure", ""),
            "display_value": _fmt_num(display_value),
            "matched_value_sum": _fmt_num(matched_value_sum),
            "residual": _fmt_num(residual),
            "residual_abs": _fmt_num(abs(residual)),
            "residual_pct_of_display": "" if abs(display_value) <= tolerance else _fmt_num(residual / display_value),
            "matched_rows": str(matched_rows),
            "source_artifact": row.get("source_artifact") or filter_data.get("source_table", ""),
            "lineage_level": row.get("lineage_level", ""),
            "detail_csv_relpath": detail_csv_relpath,
            "detail_html_relpath": detail_html_relpath,
            "filter_reason": filter_reason,
            "filter_note": filter_note,
            "row_label": _row_label(row, row_context, filter_data),
            "row_context_compact": _compact_context(row_context),
            "detail_csv_exists": str(detail_csv_exists),
            "detail_html_exists": str(detail_html_exists),
            **detail_enrichment,
        }
        issue["next_action_hint"] = _next_action_hint(status, filter_reason, matched_rows, detail_csv_exists, detail_enrichment)
        issues.append(issue)
    LOG.info("detail csv enrichment count: %s", enriched_count)
    return issues


def build_summary_rows(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for issue in issues:
        key = tuple(_text(issue.get(col)) for col in ["status", "severity", "table_id", "filter_reason", "next_action_hint"])
        rec = grouped.setdefault(key, dict(zip(["status", "severity", "table_id", "filter_reason", "next_action_hint"], key), n=0, residual_abs_sum=0.0, display_value_abs_sum=0.0))
        rec["n"] += 1
        rec["residual_abs_sum"] += abs(_num(issue.get("residual")))
        rec["display_value_abs_sum"] += abs(_num(issue.get("display_value")))
    rows = list(grouped.values())
    rows.sort(key=lambda r: (-int(r["n"]), r["status"], r["table_id"], r["filter_reason"]))
    for row in rows:
        row["residual_abs_sum"] = _fmt_num(row["residual_abs_sum"])
        row["display_value_abs_sum"] = _fmt_num(row["display_value_abs_sum"])
    LOG.info("summary counts: %s groups", len(rows))
    return rows


def _counts_table(rows: list[dict[str, Any]], key_cols: list[str], value_col: str = "n") -> list[dict[str, Any]]:
    counter: Counter[tuple[str, ...]] = Counter(tuple(_text(row.get(col)) for col in key_cols) for row in rows)
    out = [dict(zip(key_cols, key), **{value_col: n}) for key, n in counter.items()]
    out.sort(key=lambda r: (-int(r[value_col]), *[_text(r.get(col)) for col in key_cols]))
    return out


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        cls = html.escape(_text(row.get("severity", "")))
        cells = "".join(f"<td>{html.escape(_text(row.get(col, '')))}</td>" for col in columns)
        body.append(f"<tr class=\"{cls}\">{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _issue_html_row(issue: dict[str, Any]) -> str:
    detail_html = _text(issue.get("detail_html_relpath")).strip()
    detail_csv = _text(issue.get("detail_csv_relpath")).strip()
    links = []
    if detail_html:
        links.append(f'<a href="../{html.escape(detail_html)}">detail HTML</a>')
    if detail_csv:
        links.append(f'<a href="../{html.escape(detail_csv)}">detail CSV</a>')
    columns = ["status", "severity", "table_id", "row_label", "period", "Currency", "measure", "display_value", "matched_value_sum", "residual", "matched_rows", "filter_reason", "next_action_hint"]
    cells = "".join(f"<td>{html.escape(_text(issue.get(col, '')))}</td>" for col in columns)
    cells += f'<td class="links">{" ".join(links)}</td>'
    return f'<tr class="{html.escape(_text(issue.get("severity")))}">{cells}</tr>'


def write_html(issues: list[dict[str, Any]], summary: list[dict[str, Any]], output_path: Path) -> None:
    status_counts = _counts_table(issues, ["status"])
    status_table_counts = _counts_table(issues, ["status", "table_id"])
    reason_counts = _counts_table(issues, ["status", "filter_reason", "next_action_hint"])
    issue_cols = ["status", "severity", "table_id", "row_label", "period", "Currency", "measure", "display_value", "matched_value_sum", "residual", "matched_rows", "filter_reason", "next_action_hint", "links"]
    issue_head = "".join(f"<th>{html.escape(col)}</th>" for col in issue_cols)
    issue_rows = "".join(_issue_html_row(issue) for issue in issues)
    html_text = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Professional drilldown issue digest</title><style>{CSS}</style></head>
<body>
<h1>Professional drilldown issue digest</h1>
<p class="note">Generated at {html.escape(_now_iso())}. This lightweight report lists non-OK drilldown cells for iterative hardening.</p>
<p><a href="../drilldown/{ISSUES_FILENAME}">Download professional_drilldown_issues.csv</a> · <a href="../drilldown/{SUMMARY_FILENAME}">Download professional_drilldown_issue_summary.csv</a></p>
<div class="kpis"><div class="kpi"><div class="label">Issues</div><div class="value">{len(issues)}</div></div><div class="kpi"><div class="label">Summary groups</div><div class="value">{len(summary)}</div></div></div>
<h2>Status counts</h2>{_html_table(status_counts, ["status", "n"])}
<h2>Status × table summary</h2>{_html_table(status_table_counts, ["status", "table_id", "n"])}
<h2>Reason summary</h2>{_html_table(reason_counts, ["status", "filter_reason", "next_action_hint", "n"])}
<h2>Full issues table</h2>
<table><thead><tr>{issue_head}</tr></thead><tbody>{issue_rows}</tbody></table>
</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a professional drilldown issue digest from non-OK drilldown index rows.")
    parser.add_argument("--pack", required=True, type=Path, help="Professional pack directory, e.g. out/professional_pack/latest")
    parser.add_argument("--statuses", default=",".join(DEFAULT_STATUSES), help="Comma-separated statuses to include")
    parser.add_argument("--max-detail-preview-rows", type=int, default=5, help="Reserved for future lightweight detail previews")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    pack = args.pack
    statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    issues = build_issue_rows(pack, statuses, args.tolerance)
    summary = build_summary_rows(issues)

    issues_path = pack / "drilldown" / ISSUES_FILENAME
    summary_path = pack / "drilldown" / SUMMARY_FILENAME
    html_path = pack / "digest" / HTML_FILENAME
    _write_csv(issues_path, issues, ISSUE_COLUMNS)
    _write_csv(summary_path, summary, SUMMARY_COLUMNS)
    write_html(issues, summary, html_path)
    LOG.info("output paths: %s; %s; %s", issues_path, summary_path, html_path)


if __name__ == "__main__":
    main()
