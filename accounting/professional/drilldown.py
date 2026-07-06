from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

INDEX_FILENAME = "professional_drilldown_index.csv"
MANIFEST_FILENAME = "professional_drilldown_manifest.json"
QA_FILENAME = "professional_drilldown_qa.csv"
DETAILS_DIRNAME = "details"
DEFAULT_TOLERANCE = 1e-6
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_RESIDUAL_WARNING = "residual_warning"
STATUS_UNSUPPORTED = "unsupported"
STATUS_ERROR = "error"

MONTH_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")
YEAR_RE = re.compile(r"^20\d{2}$")
DERIVED_TABLE_IDS = (
    "monthly_tables_operating_statement_matrix",
    "monthly_tables_operating_statement_matrix_ars",
    "overview_balance_dashboard",
    "income_operating_statement",
    "cash_annual_box_flow_bridge_wide",
)
SUPPORTED_TABLE_IDS = (
    "monthly_tables_flow_bucket_all_measures",
    "monthly_tables_flow_subbucket_all_measures",
    "monthly_tables_draws_by_box_amount_out",
    "monthly_tables_draws_by_type_amount_out",
    "monthly_tables_fb_bridge_matrix",
    "monthly_tables_pm_stress_matrix",
    "monthly_tables_household_bridge_matrix",
    "monthly_tables_opex_by_type_amount_out",
    "monthly_tables_fx_treasury_compact",
    "monthly_tables_unknown_review_net_matrix",
    *DERIVED_TABLE_IDS,
)

CSS = """
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 28px; color: #111; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 24px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
th { background: #f4f4f4; text-align: left; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; max-width: 1100px; }
.kpi { border: 1px solid #ddd; border-radius: 10px; padding: 10px; }
.kpi .label { color: #555; font-size: 12px; }
.kpi .value { font-size: 22px; margin-top: 4px; }
pre { white-space: pre-wrap; word-break: break-word; background: #f7f7f7; border: 1px solid #ddd; padding: 10px; border-radius: 8px; }
.warn { color: #8a5a00; }
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _norm(value: Any) -> str:
    return _as_str(value).strip()


def _num(value: Any) -> float:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return 0.0
    return float(x)


def _fmt_num(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return html.escape(_as_str(value))
    if math.isnan(x):
        return ""
    return f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"


def _month_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if MONTH_RE.match(str(c))]


def _period_columns(table_id: str, df: pd.DataFrame) -> list[str]:
    if table_id in {"overview_balance_dashboard", "income_operating_statement", "cash_annual_box_flow_bridge_wide"}:
        return [str(c) for c in df.columns if YEAR_RE.match(str(c))]
    return _month_columns(df)


def _safe_id(*parts: Any) -> str:
    raw = "__".join(_norm(p) for p in parts if _norm(p) != "")
    raw = re.sub(r"[^A-Za-z0-9_.=-]+", "_", raw).strip("_")
    return raw[:180] or "drilldown"


def _contains_any(series: pd.Series, *needles: str) -> pd.Series:
    text = series.fillna("").astype(str)
    mask = pd.Series(False, index=series.index)
    for needle in needles:
        mask |= text.str.contains(needle, case=False, na=False, regex=False)
    return mask


def _regex_any(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(pattern, case=False, na=False, regex=True)


def _eq_col(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    if col not in df.columns or _norm(value) == "":
        return pd.Series(True, index=df.index)
    return df[col].astype(str).fillna("").eq(_as_str(value))


def _first_present(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and _norm(row.get(name)):
            return row.get(name)
    return ""


def _metric_name(row: pd.Series) -> str:
    return _norm(_first_present(row, "metric", "line", "statement_line", "measure", "row", "label"))


def _bucket_eq(df: pd.DataFrame, bucket: str) -> pd.Series:
    return df.get("semantic_bucket", pd.Series("", index=df.index)).astype(str).eq(bucket)


def _bucket_contains(df: pd.DataFrame, pattern: str) -> pd.Series:
    return _regex_any(df.get("semantic_bucket", pd.Series("", index=df.index)), pattern)


def _subbucket_contains(df: pd.DataFrame, pattern: str) -> pd.Series:
    return _regex_any(df.get("semantic_subbucket", pd.Series("", index=df.index)), pattern)


def _fx_mask(df: pd.DataFrame) -> pd.Series:
    mask = _bucket_contains(df, r"treasury_fx|\bfx\b") | _subbucket_contains(df, r"\bfx\b")
    for col in ["cash_path", "payer", "receiver", "counterparty"]:
        if col in df.columns:
            if col == "cash_path":
                mask |= _contains_any(df[col], "FX", "Cambio")
            else:
                mask |= df[col].fillna("").astype(str).str.casefold().eq("fx")
    return mask


def _unknown_mask(df: pd.DataFrame) -> pd.Series:
    return _bucket_contains(df, r"unknown|review") | _subbucket_contains(df, r"unknown|review")


def _fb_mask(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["Box", "payer", "receiver", "actor", "counterparty"] if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask |= _contains_any(df[col], "Family Business", "FB")
    return mask


def _detail_from_audit(audit: pd.DataFrame, semantic_subset: pd.DataFrame, filter_func: Callable[[pd.DataFrame], pd.Series]) -> tuple[pd.DataFrame, str]:
    if audit.empty:
        return semantic_subset.copy(), "semantic_only"
    try:
        subset = audit.loc[filter_func(audit)].copy()
    except Exception:
        subset = pd.DataFrame()
    tx_ids: set[str] = set()
    if "source_tx_ids_sample" in semantic_subset.columns:
        for item in semantic_subset["source_tx_ids_sample"].dropna().astype(str):
            tx_ids.update(x for x in item.split(";") if x)
    if tx_ids and "tx_id" in audit.columns:
        by_tx = audit.loc[audit["tx_id"].astype(str).isin(tx_ids)].copy()
        if not by_tx.empty:
            subset = by_tx
    return subset, "classification_audit" if not subset.empty else "classification_audit_empty"


def _measure_sum(df: pd.DataFrame, measure: str) -> float:
    if measure not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[measure], errors="coerce").fillna(0.0).sum())


def _row_context(table_id: str, row: pd.Series) -> dict[str, Any]:
    skip = set(_month_columns(pd.DataFrame(columns=row.index)))
    return {str(k): ("" if pd.isna(v) else v) for k, v in row.to_dict().items() if str(k) not in skip}


def row_context_id(table_id: str, row_index: int, row: pd.Series) -> str:
    """Stable renderer/builder key for a professional table row."""
    return _safe_id(table_id, row_index, json.dumps(_row_context(table_id, row), sort_keys=True, default=str))


@dataclass(frozen=True)
class CellSpec:
    table_id: str
    measure: str
    filter_func: Callable[[pd.DataFrame, pd.Series], pd.Series]
    caveat_func: Callable[[pd.Series], str] = lambda row: ""
    unsupported_if: Callable[[pd.Series], bool] = lambda row: False


def _spec_for_cell(table_id: str, row: pd.Series) -> CellSpec | None:
    measure = _norm(row.get("measure"))

    def base_period_currency(df: pd.DataFrame, r: pd.Series) -> pd.Series:
        return _eq_col(df, "Currency", r.get("Currency"))

    if table_id == "monthly_tables_flow_bucket_all_measures":
        return CellSpec(table_id, measure, lambda df, r: base_period_currency(df, r) & _eq_col(df, "Box", r.get("Box")) & _eq_col(df, "semantic_bucket", r.get("semantic_bucket")))
    if table_id == "monthly_tables_flow_subbucket_all_measures":
        return CellSpec(table_id, measure, lambda df, r: base_period_currency(df, r) & _eq_col(df, "Box", r.get("Box")) & _eq_col(df, "semantic_bucket", r.get("semantic_bucket")) & _eq_col(df, "semantic_subbucket", r.get("semantic_subbucket")))
    if table_id == "monthly_tables_draws_by_box_amount_out":
        return CellSpec(table_id, "amount_out", lambda df, r: base_period_currency(df, r) & _bucket_eq(df, "family_withdrawal_candidate") & _eq_col(df, "Box", r.get("Box")))
    if table_id == "monthly_tables_draws_by_type_amount_out":
        return CellSpec(table_id, "amount_out", lambda df, r: base_period_currency(df, r) & _bucket_eq(df, "family_withdrawal_candidate") & _eq_col(df, "semantic_subbucket", r.get("semantic_subbucket")))
    if table_id == "monthly_tables_opex_by_type_amount_out":
        return CellSpec(table_id, "amount_out", lambda df, r: base_period_currency(df, r) & _bucket_eq(df, "property_opex") & _eq_col(df, "Box", r.get("Box")) & _eq_col(df, "semantic_subbucket", r.get("semantic_subbucket")))
    if table_id == "monthly_tables_unknown_review_net_matrix":
        return CellSpec(table_id, "net_amount", lambda df, r: base_period_currency(df, r) & _unknown_mask(df))
    if table_id == "monthly_tables_fx_treasury_compact":
        return CellSpec(table_id, "net_amount", lambda df, r: base_period_currency(df, r) & _fx_mask(df))
    if table_id == "monthly_tables_fb_bridge_matrix":
        metric = _metric_name(row)
        mapping = {
            "rent_or_revenue_in": ("amount_in", lambda df: _bucket_eq(df, "operating_revenue")),
            "withdrawals_out": ("amount_out", lambda df: _bucket_eq(df, "family_withdrawal_candidate")),
            "funding_in": ("amount_in", lambda df: _bucket_eq(df, "funding_contribution")),
            "fx_or_treasury_net": ("net_amount", _fx_mask),
            "net_flow": ("net_amount", lambda df: pd.Series(True, index=df.index)),
        }
        if metric not in mapping:
            return CellSpec(table_id, measure or metric, lambda df, r: pd.Series(False, index=df.index), unsupported_if=lambda r: True)
        value_col, mask_fn = mapping[metric]
        return CellSpec(table_id, value_col, lambda df, r: base_period_currency(df, r) & _fb_mask(df) & mask_fn(df), caveat_func=lambda r: "FB-related no equivale siempre a Box=Family Business.")
    if table_id in {"monthly_tables_pm_stress_matrix", "monthly_tables_household_bridge_matrix"}:
        box = "Property Management" if table_id == "monthly_tables_pm_stress_matrix" else "Household"
        metric = _metric_name(row)
        mapping: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
            "revenue_in": ("amount_in", lambda df: _bucket_eq(df, "operating_revenue")),
            "property_opex_out": ("amount_out", lambda df: _bucket_eq(df, "property_opex")),
            "opex_out": ("amount_out", lambda df: _bucket_eq(df, "property_opex")),
            "withdrawals_out": ("amount_out", lambda df: _bucket_eq(df, "family_withdrawal_candidate")),
            "funding_in": ("amount_in", lambda df: _bucket_eq(df, "funding_contribution")),
            "debt_net": ("net_amount", lambda df: _bucket_contains(df, "debt")),
            "unknown_net": ("net_amount", _unknown_mask),
            "fx_or_treasury_net": ("net_amount", _fx_mask),
            "net_flow": ("net_amount", lambda df: pd.Series(True, index=df.index)),
        }
        if metric not in mapping:
            return CellSpec(table_id, measure or metric, lambda df, r: pd.Series(False, index=df.index), unsupported_if=lambda r: True)
        value_col, mask_fn = mapping[metric]
        return CellSpec(table_id, value_col, lambda df, r: base_period_currency(df, r) & df.get("Box", pd.Series("", index=df.index)).astype(str).eq(box) & mask_fn(df))
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _candidate_run_roots(repo_root: Path, pack_dir: Path, run_root: Path | None) -> Iterable[Path]:
    if run_root is not None:
        yield run_root
    yield pack_dir
    yield pack_dir.parent
    yield repo_root / "out" / "run" / "accounting" / "latest"


def _find_source(repo_root: Path, pack_dir: Path, run_root: Path | None, filename: str) -> Path | None:
    direct_roots = list(_candidate_run_roots(repo_root, pack_dir, run_root))
    for root in direct_roots:
        path = root / filename
        if path.exists():
            return path
    extra = [
        repo_root / "out" / "metrics" / "latest" / filename,
        repo_root / "public" / "accounting" / "latest" / "canonical_dashboard" / filename,
        repo_root / "public" / "accounting" / "latest" / filename,
        pack_dir / "source" / filename,
        pack_dir / "tables" / filename,
    ]
    for path in extra:
        if path.exists():
            return path
    return None


def _write_detail_html(path: Path, index_row: dict[str, Any], detail_df: pd.DataFrame, sections: list[tuple[str, pd.DataFrame]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filters = json.dumps(json.loads(index_row.get("filter_json") or "{}"), indent=2, ensure_ascii=False)
    caveat = _norm(index_row.get("caveat"))
    if sections:
        body = "".join(f"<h2>{html.escape(title)}</h2>" + (df.to_html(index=False, escape=True, classes="detail") if not df.empty else "<p class='warn'>No rows.</p>") for title, df in sections)
    else:
        body = detail_df.to_html(index=False, escape=True, classes="detail") if not detail_df.empty else "<p class='warn'>No matching detail rows.</p>"
    kpis = "".join(
        f"<div class='kpi'><div class='label'>{html.escape(label)}</div><div class='value'>{_fmt_num(index_row.get(key))}</div></div>"
        for label, key in [("Displayed value", "display_value"), ("Matched sum", "matched_value_sum"), ("Residual", "residual"), ("Matched rows", "matched_rows")]
    )
    text = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(_as_str(index_row.get('drilldown_id')))}</title><style>{CSS}</style></head><body>
<h1>{html.escape(_as_str(index_row.get('table_id')))}</h1>
<p><strong>Drilldown:</strong> {html.escape(_as_str(index_row.get('drilldown_id')))}<br>
<strong>Status:</strong> {html.escape(_as_str(index_row.get('status')))}<br>
<strong>Period:</strong> {html.escape(_as_str(index_row.get('period')))} | <strong>Currency:</strong> {html.escape(_as_str(index_row.get('Currency')))} | <strong>Measure:</strong> {html.escape(_as_str(index_row.get('measure')))}<br>
<strong>Source artifact:</strong> {html.escape(_as_str(index_row.get('source_artifact')))} | <strong>Lineage:</strong> {html.escape(_as_str(index_row.get('lineage_level')))}</p>
{f"<p class='warn'><strong>Caveat:</strong> {html.escape(caveat)}</p>" if caveat else ""}
<div class='kpis'>{kpis}</div>
<h2>Filters</h2><pre>{html.escape(filters)}</pre>
<h2>Row context</h2><pre>{html.escape(json.dumps(json.loads(index_row.get('row_context_json') or '{}'), indent=2, ensure_ascii=False, default=str))}</pre>
<p><a href='{html.escape(Path(_as_str(index_row.get('detail_csv_relpath'))).name)}'>Open detail CSV</a></p>
{"" if sections else "<h2>Relevant rows</h2>"}{body}
</body></html>"""
    path.write_text(text, encoding="utf-8")



def _year_mask(df: pd.DataFrame, period: str) -> pd.Series:
    if "period" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["period"].astype(str).str.startswith(str(period)) if YEAR_RE.match(str(period)) else df["period"].astype(str).eq(str(period))


def _statement_line(row: pd.Series) -> str:
    return _metric_name(row)


def _semantic_filter_for_statement_line(line: str) -> tuple[str, Callable[[pd.DataFrame], pd.Series]] | None:
    mapping: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
        "operating_revenue": ("amount_in", lambda df: _bucket_eq(df, "operating_revenue")),
        "rent_revenue": ("amount_in", lambda df: _bucket_eq(df, "operating_revenue") & _eq_col(df, "semantic_subbucket", "rent")),
        "property_opex_true": ("amount_out", lambda df: _bucket_eq(df, "property_opex")),
        "funding_contributions": ("amount_in", lambda df: _bucket_eq(df, "funding_contribution")),
        "family_draws_or_distributions": ("amount_out", lambda df: _bucket_eq(df, "family_withdrawal_candidate")),
        "unknown_or_ambiguous_outflows": ("amount_abs", _unknown_mask),
        "treasury_fx_conversion_in": ("amount_in", lambda df: _bucket_eq(df, "treasury_fx") & _eq_col(df, "semantic_subbucket", "fx_conversion_proceeds")),
        "treasury_fx_conversion_out": ("amount_out", lambda df: _bucket_eq(df, "treasury_fx") & _eq_col(df, "semantic_subbucket", "fx_conversion_outflow")),
        "treasury_fx_cost": ("amount_out", lambda df: _bucket_eq(df, "treasury_fx") & _eq_col(df, "semantic_subbucket", "fx_cost_or_spread")),
        "treasury_fx_net": ("net_amount", lambda df: _bucket_eq(df, "treasury_fx")),
    }
    return mapping.get(line)


def _statement_components(stmt: pd.DataFrame, period: str, currency: str, component_lines: list[str]) -> pd.DataFrame:
    if stmt.empty:
        return pd.DataFrame()
    mask = _year_mask(stmt, period) & _eq_col(stmt, "Currency", currency)
    if "statement_line" in stmt.columns:
        mask &= stmt["statement_line"].astype(str).isin(component_lines)
    return stmt.loc[mask].copy()


def _annual_source_rows(annual: pd.DataFrame, row: pd.Series, period: str) -> pd.DataFrame:
    if annual.empty:
        return pd.DataFrame()
    metric_id = _norm(_first_present(row, "metric_id", "metric", "line"))
    mask = annual.get("period", pd.Series("", index=annual.index)).astype(str).eq(str(period)) & _eq_col(annual, "Currency", row.get("Currency"))
    if metric_id and "metric_id" in annual.columns:
        mask &= annual["metric_id"].astype(str).eq(metric_id)
    for dim_col in ["dimension_name", "dimension_value", "section", "dashboard_section"]:
        if dim_col in row.index and dim_col in annual.columns and _norm(row.get(dim_col)):
            mask &= annual[dim_col].astype(str).eq(_as_str(row.get(dim_col)))
    return annual.loc[mask].copy()


def _cash_bridge_semantic_rows(split: pd.DataFrame, row: pd.Series, period: str) -> tuple[pd.DataFrame, str, str]:
    measure = _norm(_first_present(row, "measure", "value_col"))
    line = _metric_name(row).casefold()
    if not measure:
        if "in" in line:
            measure = "amount_in"
        elif "out" in line:
            measure = "amount_out"
        else:
            measure = "net_amount"
    mask = _year_mask(split, period) & _eq_col(split, "Currency", row.get("Currency"))
    for col in ["Box", "semantic_bucket", "semantic_subbucket", "cash_path"]:
        if col in row.index and _norm(row.get(col)):
            mask &= _eq_col(split, col, row.get(col))
    return split.loc[mask].copy(), measure, "year/Currency/Box/semantic/cash_path flow bridge"


def _build_derived_cell(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    display_value: float,
    split: pd.DataFrame,
    audit: pd.DataFrame,
    stmt: pd.DataFrame,
    annual: pd.DataFrame,
    tolerance: float,
) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    currency = _norm(row.get("Currency"))
    if not currency and table_id != "overview_balance_dashboard":
        return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "", {"unsupported": True, "reason": "missing Currency would risk cross-currency aggregation"}, "", pd.DataFrame(), []

    line = _statement_line(row)
    sections: list[tuple[str, pd.DataFrame]] = []
    caveat = "Derived drilldown: explanation page, not necessarily raw ledger rows."

    if table_id in {"monthly_tables_operating_statement_matrix", "monthly_tables_operating_statement_matrix_ars"}:
        if stmt.empty:
            return STATUS_ERROR, 0.0, -display_value, "missing_source", "monthly_operating_statement.csv", {"error": "missing monthly_operating_statement.csv"}, caveat, pd.DataFrame(), []
        source = stmt.loc[_year_mask(stmt, period) & _eq_col(stmt, "Currency", currency) & _eq_col(stmt, "statement_line", line)].copy()
        matched = _measure_sum(source, "amount")
        if line == "net_operating":
            components = _statement_components(stmt, period, currency, ["operating_revenue", "property_opex_true"])
            sections.append(("Formula", pd.DataFrame([{"formula": "operating_revenue - property_opex_true", "displayed_value": display_value, "source_sum": matched}])))
            sections.append(("Component rows", components))
        elif line == "coverage_after_draws":
            components = _statement_components(stmt, period, currency, ["net_operating", "funding_contributions", "family_draws_or_distributions"])
            sections.append(("Formula", pd.DataFrame([{"formula": "net_operating + funding_contributions - family_draws_or_distributions", "displayed_value": display_value, "source_sum": matched}])))
            sections.append(("Component rows", components))
        sem_spec = _semantic_filter_for_statement_line(line)
        semantic_rows = pd.DataFrame()
        detail_rows = source
        if sem_spec is not None:
            measure, sem_filter = sem_spec
            semantic_rows = split.loc[_year_mask(split, period) & _eq_col(split, "Currency", currency) & sem_filter(split)].copy() if not split.empty else pd.DataFrame()
            detail_rows, lineage = _detail_from_audit(audit, semantic_rows, lambda df, p=period, sf=sem_filter: _year_mask(df, p) & _eq_col(df, "Currency", currency) & sf(df))
            sections.append(("Semantic rows", semantic_rows))
            sections.append(("Classification rows", detail_rows))
        else:
            lineage = "monthly_operating_statement"
        sections.insert(0, ("Source statement rows", source))
        residual = matched - display_value
        status = STATUS_EMPTY if source.empty else (STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING)
        filters = {"period": period, "Currency": currency, "statement_line": line, "source": "monthly_operating_statement.csv"}
        return status, matched, residual, lineage, "monthly_operating_statement.csv", filters, caveat, detail_rows if not detail_rows.empty else source, sections

    if table_id in {"overview_balance_dashboard", "income_operating_statement"}:
        annual_rows = _annual_source_rows(annual, row, period)
        if annual.empty or annual_rows.empty:
            return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "annual_balance_dashboard_metrics.csv", {"unsupported": True, "reason": "no matching annual metric row", "period": period}, "Annual source row unavailable.", pd.DataFrame(), []
        source_table = _norm(annual_rows.iloc[0].get("source_table"))
        flow_type = _norm(annual_rows.iloc[0].get("flow_type"))
        calc_rule = _norm(annual_rows.iloc[0].get("calculation_rule"))
        if flow_type == "stock" or source_table in {"monthly_cash_close.csv", "monthly_debt_position.csv"}:
            return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "annual_balance_dashboard_metrics.csv", {"unsupported": True, "reason": "stock/cash metric is not a flow drilldown", "source_table": source_table}, "Stock/cash metrics are not treated as flow ledger drilldowns.", annual_rows, [("Annual metric row", annual_rows)]
        matched = _measure_sum(annual_rows, "value")
        sections.append(("Annual metric row", annual_rows))
        monthly_rows = pd.DataFrame()
        semantic_rows = pd.DataFrame()
        detail_rows = annual_rows
        lineage = "annual_balance_dashboard_metrics"
        if source_table == "monthly_operating_statement.csv" and not stmt.empty:
            source_filter = _norm(annual_rows.iloc[0].get("source_filter"))
            stmt_line = source_filter.split("statement_line=", 1)[1].split(";", 1)[0].strip() if "statement_line=" in source_filter else ""
            monthly_rows = stmt.loc[_year_mask(stmt, period) & _eq_col(stmt, "Currency", currency) & (_eq_col(stmt, "statement_line", stmt_line) if stmt_line else pd.Series(True, index=stmt.index))].copy()
            sections.append(("Monthly source rows", monthly_rows))
            lineage = "annual_to_monthly_statement"
        if source_table == "monthly_flow_semantic_split.csv" and not split.empty:
            metric_id = _norm(annual_rows.iloc[0].get("metric_id"))
            dim_name = _norm(annual_rows.iloc[0].get("dimension_name"))
            dim_value = _norm(annual_rows.iloc[0].get("dimension_value"))
            sem_mask = _year_mask(split, period) & _eq_col(split, "Currency", currency)
            if metric_id in {"IS.RENT.TOTAL", "IS.RENT.BY_PROPERTY"}:
                sem_mask &= _bucket_eq(split, "operating_revenue") & _eq_col(split, "semantic_subbucket", "rent")
            elif metric_id in {"IS.OPEX.BY_CATEGORY"}:
                sem_mask &= _bucket_eq(split, "property_opex")
            elif metric_id in {"FUND.CONTRIB.BY_ACTOR"}:
                sem_mask &= _bucket_eq(split, "funding_contribution")
            elif metric_id in {"DIST.DRAWS.BY_TYPE"}:
                sem_mask &= _bucket_eq(split, "family_withdrawal_candidate")
            if dim_name and dim_value and dim_name in split.columns:
                sem_mask &= _eq_col(split, dim_name, dim_value)
            semantic_rows = split.loc[sem_mask].copy()
            sections.append(("Semantic rows", semantic_rows))
            detail_rows, lineage = _detail_from_audit(audit, semantic_rows, lambda df, p=period: _year_mask(df, p) & _eq_col(df, "Currency", currency))
            sections.append(("Classification rows", detail_rows))
        residual = matched - display_value
        status = STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
        filters = {"period": period, "Currency": currency, "source_table": source_table, "calculation_rule": calc_rule, "row_context": _row_context(table_id, row)}
        return status, matched, residual, lineage, "annual_balance_dashboard_metrics.csv", filters, caveat, detail_rows, sections

    if table_id == "cash_annual_box_flow_bridge_wide":
        line_lower = line.casefold()
        if any(token in line_lower for token in ["validated cash", "cash close", "diagnostic box balance"]):
            return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "monthly_flow_semantic_split.csv", {"unsupported": True, "reason": "cash/stock diagnostic is not a flow drilldown"}, "Cash levels and diagnostic balances are not flow ledger drilldowns.", pd.DataFrame(), []
        if split.empty:
            return STATUS_ERROR, 0.0, -display_value, "missing_source", "monthly_flow_semantic_split.csv", {"error": "missing monthly_flow_semantic_split.csv"}, caveat, pd.DataFrame(), []
        semantic_rows, measure, filter_note = _cash_bridge_semantic_rows(split, row, period)
        matched = _measure_sum(semantic_rows, measure)
        residual = matched - display_value
        detail_rows, lineage = _detail_from_audit(audit, semantic_rows, lambda df, p=period: _year_mask(df, p) & _eq_col(df, "Currency", currency))
        sections = [("Flow bridge semantic rows", semantic_rows), ("Classification rows", detail_rows)]
        filters = {"year": period, "Currency": currency, "measure": measure, "filter_note": filter_note, "row_context": _row_context(table_id, row)}
        status = STATUS_EMPTY if semantic_rows.empty else (STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING)
        return status, matched, residual, lineage, "monthly_flow_semantic_split.csv", filters, caveat, detail_rows if not detail_rows.empty else semantic_rows, sections

    return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "", {"unsupported": True}, "", pd.DataFrame(), []

def build_professional_flow_drilldowns(repo_root: Path, pack_dir: Path, run_root: Path | None = None, tables_dir: Path | None = None, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Path]:
    repo_root = Path(repo_root)
    pack_dir = Path(pack_dir)
    tables_dir = Path(tables_dir) if tables_dir is not None else pack_dir / "tables"
    drill_dir = pack_dir / "drilldown"
    details_dir = drill_dir / DETAILS_DIRNAME
    details_dir.mkdir(parents=True, exist_ok=True)

    split_path = _find_source(repo_root, pack_dir, run_root, "monthly_flow_semantic_split.csv")
    audit_path = _find_source(repo_root, pack_dir, run_root, "classification_audit.csv")
    stmt_path = _find_source(repo_root, pack_dir, run_root, "monthly_operating_statement.csv")
    annual_path = _find_source(repo_root, pack_dir, run_root, "annual_balance_dashboard_metrics.csv")
    split = _read_csv(split_path) if split_path else pd.DataFrame()
    audit = _read_csv(audit_path) if audit_path else pd.DataFrame()
    stmt = _read_csv(stmt_path) if stmt_path else pd.DataFrame()
    annual = _read_csv(annual_path) if annual_path else pd.DataFrame()
    for df in [split, audit, stmt, annual]:
        if not df.empty:
            for col in ["amount_in", "amount_out", "net_amount", "amount_abs", "amount", "value"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    scope = list(SUPPORTED_TABLE_IDS)
    index_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    for table_id in scope:
        table_path = tables_dir / f"{table_id}.csv"
        if not table_path.exists():
            qa_rows.append({"table_id": table_id, "check": "table_exists", "status": "warning", "detail": f"Missing table: {table_path}"})
            continue
        table = pd.read_csv(table_path)
        months = _period_columns(table_id, table)
        if not months:
            qa_rows.append({"table_id": table_id, "check": "month_columns", "status": "warning", "detail": "No YYYY-MM columns detected"})
            continue
        for row_idx, row in table.iterrows():
            spec = _spec_for_cell(table_id, row)
            context = _row_context(table_id, row)
            row_id = row_context_id(table_id, int(row_idx), row)
            for period in months:
                display_value = _num(row.get(period))
                measure = spec.measure if spec else _norm(row.get("measure"))
                missing_currency = _norm(row.get("Currency")) == ""
                drilldown_id = _safe_id(table_id, row_idx, period, row.get("Currency"), measure)
                detail_csv_rel = f"drilldown/{DETAILS_DIRNAME}/{drilldown_id}.csv"
                detail_html_rel = f"drilldown/{DETAILS_DIRNAME}/{drilldown_id}.html"
                base = {
                    "drilldown_id": drilldown_id, "table_id": table_id, "row_id": row_id, "period": period,
                    "Currency": _norm(row.get("Currency")), "measure": measure, "source_artifact": "monthly_flow_semantic_split.csv",
                    "detail_csv_relpath": detail_csv_rel, "detail_html_relpath": detail_html_rel,
                    "display_value": display_value, "filter_json": "{}", "row_context_json": json.dumps(context, ensure_ascii=False, sort_keys=True, default=str),
                    "lineage_level": "", "caveat": "",
                }
                try:
                    if table_id in DERIVED_TABLE_IDS:
                        status, matched, residual, lineage, source_artifact, filters, caveat, detail_df, sections = _build_derived_cell(
                            table_id=table_id, row=row, period=period, display_value=display_value,
                            split=split, audit=audit, stmt=stmt, annual=annual, tolerance=tolerance,
                        )
                        out_row = {**base, "source_artifact": source_artifact, "matched_rows": int(len(detail_df)), "matched_value_sum": matched, "residual": residual, "status": status, "filter_json": json.dumps(filters, ensure_ascii=False, sort_keys=True, default=str), "lineage_level": lineage, "caveat": caveat}
                        (pack_dir / detail_csv_rel).parent.mkdir(parents=True, exist_ok=True)
                        detail_df.to_csv(pack_dir / detail_csv_rel, index=False)
                        _write_detail_html(pack_dir / detail_html_rel, out_row, detail_df, sections=sections)
                        index_rows.append(out_row)
                        qa_status = "pass" if status == STATUS_OK else ("warning" if status in {STATUS_EMPTY, STATUS_RESIDUAL_WARNING, STATUS_UNSUPPORTED} else "fail")
                        qa_rows.append({"table_id": table_id, "drilldown_id": drilldown_id, "check": "cell_reconciliation", "status": qa_status, "detail": f"status={status}; residual={residual}; matched_rows={len(detail_df)}"})
                        continue

                    if split.empty:
                        status, semantic_subset, matched, residual, lineage, filters, caveat = STATUS_ERROR, pd.DataFrame(), 0.0, -display_value, "missing_source", {"error": "missing monthly_flow_semantic_split.csv"}, ""
                    elif missing_currency:
                        status, semantic_subset, matched, residual, lineage, filters, caveat = STATUS_UNSUPPORTED, pd.DataFrame(), 0.0, -display_value, "unsupported", {"unsupported": True, "reason": "missing Currency would risk cross-currency aggregation", "measure": measure}, ""
                    elif spec is None or spec.unsupported_if(row) or measure not in split.columns:
                        status, semantic_subset, matched, residual, lineage, filters, caveat = STATUS_UNSUPPORTED, pd.DataFrame(), 0.0, -display_value, "unsupported", {"unsupported": True, "measure": measure}, ""
                    else:
                        period_mask = split.get("period", pd.Series("", index=split.index)).astype(str).eq(period)
                        semantic_subset = split.loc[period_mask & spec.filter_func(split, row)].copy()
                        matched = _measure_sum(semantic_subset, spec.measure)
                        residual = matched - display_value
                        if semantic_subset.empty:
                            status = STATUS_EMPTY
                        elif abs(residual) <= tolerance:
                            status = STATUS_OK
                        else:
                            status = STATUS_RESIDUAL_WARNING
                        filters = {"period": period, "Currency": _norm(row.get("Currency")), "measure": spec.measure, "row_context": context}
                        caveat = spec.caveat_func(row)
                        detail_df, lineage = _detail_from_audit(audit, semantic_subset, lambda df, r=row, p=period, s=spec: df.get("period", pd.Series("", index=df.index)).astype(str).eq(p) & s.filter_func(df, r))
                        if detail_df.empty and lineage != "semantic_only":
                            detail_df = semantic_subset
                    if 'detail_df' not in locals() or (split.empty or spec is None or (spec and spec.unsupported_if(row))):
                        detail_df = semantic_subset.copy() if 'semantic_subset' in locals() else pd.DataFrame()
                    matched_rows = int(len(detail_df)) if not detail_df.empty else int(len(semantic_subset)) if 'semantic_subset' in locals() else 0
                    out_row = {**base, "matched_rows": matched_rows, "matched_value_sum": matched, "residual": residual, "status": status, "filter_json": json.dumps(filters, ensure_ascii=False, sort_keys=True, default=str), "lineage_level": lineage, "caveat": caveat}
                    (pack_dir / detail_csv_rel).parent.mkdir(parents=True, exist_ok=True)
                    detail_df.to_csv(pack_dir / detail_csv_rel, index=False)
                    _write_detail_html(pack_dir / detail_html_rel, out_row, detail_df)
                    index_rows.append(out_row)
                    qa_status = "pass" if status == STATUS_OK else ("warning" if status in {STATUS_EMPTY, STATUS_RESIDUAL_WARNING, STATUS_UNSUPPORTED} else "fail")
                    qa_rows.append({"table_id": table_id, "drilldown_id": drilldown_id, "check": "cell_reconciliation", "status": qa_status, "detail": f"status={status}; residual={residual}; matched_rows={matched_rows}"})
                finally:
                    if 'detail_df' in locals():
                        del detail_df

    columns = ["drilldown_id", "table_id", "row_id", "period", "Currency", "measure", "source_artifact", "detail_csv_relpath", "detail_html_relpath", "matched_rows", "matched_value_sum", "display_value", "residual", "status", "filter_json", "row_context_json", "lineage_level", "caveat"]
    index = pd.DataFrame(index_rows, columns=columns)
    qa = pd.DataFrame(qa_rows, columns=["table_id", "drilldown_id", "check", "status", "detail"])
    index_path = drill_dir / INDEX_FILENAME
    manifest_path = drill_dir / MANIFEST_FILENAME
    qa_path = drill_dir / QA_FILENAME
    index.to_csv(index_path, index=False)
    qa.to_csv(qa_path, index=False)
    manifest = {"created_at_utc": _now_iso(), "repo_root": str(repo_root), "pack_dir": str(pack_dir), "tables_dir": str(tables_dir), "run_root": str(run_root or ""), "monthly_flow_semantic_split": str(split_path or ""), "classification_audit": str(audit_path or ""), "monthly_operating_statement": str(stmt_path or ""), "annual_balance_dashboard_metrics": str(annual_path or ""), "tolerance": tolerance, "index_rows": int(len(index)), "qa_rows": int(len(qa)), "status_counts": index["status"].value_counts().to_dict() if not index.empty else {}}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"index": index_path, "manifest": manifest_path, "qa": qa_path, "details_dir": details_dir}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build professional flow drilldown artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    paths = build_professional_flow_drilldowns(args.repo_root, args.pack, args.run_root, args.tables_dir, args.tolerance)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
