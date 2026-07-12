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
FAST_TABLE_CELL_LIMIT = 100
MAX_TABLE_CELL_LIMIT = 500
TABLE_TOO_LARGE_WARNING = "Table has too many cells to afford triggering drilldowns; skipping drilldown build for this table."
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
    "monthly_tables_cash_close_matrix",
    "monthly_tables_debt_activity_matrix",
    "monthly_tables_diagnostic_box_level_matrix",
    "monthly_tables_debt_position_matrix",
    "annual_cash_close_by_box_wide",
    "annual_funding_by_actor_channel_wide",
    "annual_debt_stock_by_pair_wide",
    "annual_debt_activity_by_pair_wide",
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
    "monthly_tables_draws_by_type_net_amount",
    "monthly_tables_fx_treasury_all_measures",
    "monthly_tables_fx_treasury_amount_in",
    "monthly_tables_fx_treasury_amount_out",
    "monthly_tables_fx_treasury_net_amount",
    "monthly_tables_fx_treasury_compact",

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



from accounting.logging_utils import configure_logging, get_logger
from accounting.professional.table_contracts import enrich_professional_table_contracts
LOG = get_logger("prof drilldown")



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _norm(value: Any) -> str:
    return _as_str(value).strip()


def _norm_period_key(value: Any) -> str:
    s = _norm(value)
    if not s:
        return ""
    # pandas may read annual year keys as floats (for example 2023.0).
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


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
    if table_id in {
        "overview_balance_dashboard",
        "income_operating_statement",
        "cash_annual_box_flow_bridge_wide",
        "annual_cash_close_by_box_wide",
        "annual_funding_by_actor_channel_wide",
        "annual_debt_stock_by_pair_wide",
        "annual_debt_activity_by_pair_wide",
    }:
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


# def _row_context(table_id: str, row: pd.Series) -> dict[str, Any]:
#     skip = set(_month_columns(pd.DataFrame(columns=row.index)))
#     return {str(k): ("" if pd.isna(v) else v) for k, v in row.to_dict().items() if str(k) not in skip}

def _row_context(table_id: str, row: pd.Series) -> dict[str, Any]:
    period_cols = set(_period_columns(table_id, pd.DataFrame(columns=row.index)))
    return {
        str(k): ("" if pd.isna(v) else v)
        for k, v in row.to_dict().items()
        if str(k) not in period_cols
    }

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




FX_TREASURY_TABLE_IDS = {
    "monthly_tables_fx_treasury_all_measures",
    "monthly_tables_fx_treasury_amount_in",
    "monthly_tables_fx_treasury_amount_out",
    "monthly_tables_fx_treasury_net_amount",
    "monthly_tables_fx_treasury_compact",
}

FX_MEASURES = {"amount_in", "amount_out", "net_amount", "amount_abs"}


def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    # Prefer explicit row grain. This is mandatory for compact/all_measures.
    for col in ["measure", "metric"]:
        measure = _norm(row.get(col))
        if measure in FX_MEASURES:
            return measure

    # Only infer from table name for single-measure tables.
    if table_id == "monthly_tables_fx_treasury_amount_in":
        return "amount_in"

    if table_id == "monthly_tables_fx_treasury_amount_out":
        return "amount_out"

    if table_id == "monthly_tables_fx_treasury_net_amount":
        return "net_amount"

    # Compact/all_measures must not default to net_amount.
    return ""



FX_TREASURY_TABLE_IDS = {
    "monthly_tables_fx_treasury_all_measures",
    "monthly_tables_fx_treasury_amount_in",
    "monthly_tables_fx_treasury_amount_out",
    "monthly_tables_fx_treasury_net_amount",
    "monthly_tables_fx_treasury_compact",
}

FX_MEASURES = {"amount_in", "amount_out", "net_amount", "amount_abs"}


def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    measure = _norm(row.get("measure"))

    if measure in FX_MEASURES:
        return measure

    if table_id == "monthly_tables_fx_treasury_amount_in":
        return "amount_in"

    if table_id == "monthly_tables_fx_treasury_amount_out":
        return "amount_out"

    if table_id == "monthly_tables_fx_treasury_net_amount":
        return "net_amount"

    metric = _metric_name(row).casefold().strip()

    mapping = {
        "fx_conversion_proceeds": "amount_in",
        "fx_conversion_outflow": "amount_out",
        "fx_cost_or_spread": "amount_out",
        "fx_net": "net_amount",
        "net_amount": "net_amount",
        "amount_in": "amount_in",
        "amount_out": "amount_out",
        "amount_abs": "amount_abs",
    }

    if metric in {"", "monthly_tables_fx_treasury_compact"} and table_id == "monthly_tables_fx_treasury_compact":
        return "net_amount"

    return mapping.get(metric, "")




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
    

    if table_id == "monthly_tables_draws_by_type_net_amount":
        return CellSpec(
            table_id,
            "net_amount",
            lambda df, r: (
                base_period_currency(df, r)
                & _bucket_eq(df, "family_withdrawal_candidate")
                & _eq_col(df, "Box", r.get("Box"))
                & _eq_col(df, "semantic_subbucket", r.get("semantic_subbucket"))
            ),
        )


    if table_id in FX_TREASURY_TABLE_IDS:
        fx_measure = _fx_treasury_measure_for_row(table_id, row)

        if fx_measure not in FX_MEASURES:
            return CellSpec(
                table_id,
                measure or _metric_name(row),
                lambda df, r: pd.Series(False, index=df.index),
                unsupported_if=lambda r: True,
            )

        return CellSpec(
            table_id,
            fx_measure,
            lambda df, r: (
                base_period_currency(df, r)
                & _optional_strict_eq_col(df, "Box", r.get("Box"))
                & _strict_eq_col(df, "semantic_bucket", "treasury_fx")
                & _optional_strict_eq_col(df, "semantic_subbucket", r.get("semantic_subbucket"))
            ),
        )

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



MAX_DETAIL_HTML_ROWS = 100
MAX_DETAIL_HTML_COLS = 100


def _should_render_full_detail_table(df: pd.DataFrame) -> bool:
    return len(df) <= MAX_DETAIL_HTML_ROWS and len(df.columns) <= MAX_DETAIL_HTML_COLS


# def _numeric_sum_row_df(df: pd.DataFrame) -> pd.DataFrame:
#     """Return numeric column sums as a one-row dataframe for HTML comparison."""
#     if df.empty:
#         return pd.DataFrame([{"note": "empty detail dataframe"}])

#     numeric = df.select_dtypes(include=["number"]).copy()
#     if numeric.empty:
#         return pd.DataFrame([{"note": "no numeric columns"}])

#     return pd.DataFrame([numeric.sum(numeric_only=True).to_dict()])


def _reconciliation_row_df(index_row: dict[str, Any], detail_df: pd.DataFrame) -> pd.DataFrame:
    """One-row reconciliation summary for the clicked cell."""
    measure = _as_str(index_row.get("measure"))
    detail_measure_sum = ""

    if measure and measure in detail_df.columns:
        detail_measure_sum = float(
            pd.to_numeric(detail_df[measure], errors="coerce").fillna(0.0).sum()
        )

    return pd.DataFrame(
        [
            {
                "table_id": index_row.get("table_id", ""),
                "period": index_row.get("period", ""),
                "Currency": index_row.get("Currency", ""),
                "measure": measure,
                "display_value": index_row.get("display_value", 0.0),
                "matched_value_sum": index_row.get("matched_value_sum", 0.0),
                "detail_measure_sum": detail_measure_sum,
                "residual": index_row.get("residual", 0.0),
                "matched_rows": index_row.get("matched_rows", 0),
                "status": index_row.get("status", ""),
            }
        ]
    )


DETAIL_SUM_COLUMNS = [
    "amount_in",
    "amount_out",
    "net_amount",
    "amount_abs",
    "n_tx",
    "amount",
    "value",
    "cash_close",
    "closing_cash",
    "diagnostic_box_level",
    "closing_balance",
    "balance",
    "new_principal",
    "interest_accrued",
    "repayments",
    "adjustments",
    "opening_total",
    "closing_total",
    "net_change",
    "open_amount",
    "open_principal",
    "open_interest",
    "open_total",
]


def _detail_sum_row_df(df: pd.DataFrame) -> pd.DataFrame:
    """One-row sum table for known accounting amount columns."""
    if df.empty:
        return pd.DataFrame([{"note": "empty dataframe"}])

    row: dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
    }

    found = False
    for col in DETAIL_SUM_COLUMNS:
        if col in df.columns:
            row[col] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
            found = True

    if not found:
        row["note"] = "none of expected sum columns found"

    return pd.DataFrame([row])

def _render_df_section(
    title: str,
    df: pd.DataFrame,
    *,
    empty_message: str = "No rows.",
    include_sums: bool = True,
) -> str:
    title_html = html.escape(title)

    if df.empty:
        return f"<h2>{title_html}</h2><p class='warn'>{html.escape(empty_message)}</p>"

    chunks: list[str] = [f"<h2>{title_html}</h2>"]

    if include_sums:
        sums_df = _detail_sum_row_df(df)
        chunks.append("<h3>Column sums</h3>")
        chunks.append(
            sums_df.to_html(
                index=False,
                escape=True,
                classes="detail sum-row",
                border=0,
            )
        )

    if _should_render_full_detail_table(df):
        chunks.append(
            df.to_html(
                index=False,
                escape=True,
                classes="detail",
                border=0,
            )
        )
    else:
        rows = len(df)
        cols = len(df.columns)
        preview = df.iloc[:MAX_DETAIL_HTML_ROWS, :MAX_DETAIL_HTML_COLS].copy()

        chunks.append(
            "<p class='warn'>"
            f"Full table omitted from HTML because it is large: "
            f"{rows} rows × {cols} columns. "
            "Open the CSV for the complete detail."
            "</p>"
        )
        chunks.append(
            f"<h3>Preview: first {MAX_DETAIL_HTML_ROWS} rows × first {MAX_DETAIL_HTML_COLS} columns</h3>"
        )
        chunks.append(
            preview.to_html(
                index=False,
                escape=True,
                classes="detail",
                border=0,
            )
        )

    return "\n".join(chunks)

def _write_detail_html(
    path: Path,
    index_row: dict[str, Any],
    detail_df: pd.DataFrame,
    sections: list[tuple[str, pd.DataFrame]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    filters = json.dumps(
        json.loads(index_row.get("filter_json") or "{}"),
        indent=2,
        ensure_ascii=False,
    )
    row_context = json.dumps(
        json.loads(index_row.get("row_context_json") or "{}"),
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    caveat = _norm(index_row.get("caveat"))

    body_parts: list[str] = []

    reconciliation_df = _reconciliation_row_df(index_row, detail_df)
    # numeric_sums_df = _numeric_sum_row_df(detail_df)
    numeric_sums_df = _detail_sum_row_df(detail_df)

    body_parts.append(
        _render_df_section(
            "Reconciliation",
            reconciliation_df,
            empty_message="No reconciliation row.",
        )
    )

    body_parts.append(
        _render_df_section(
            "Drilldown numeric sums",
            numeric_sums_df,
            empty_message="No numeric sums.",
        )
    )

    if sections:
        for title, df in sections:
            body_parts.append(_render_df_section(title, df))
    else:
        body_parts.append(
            _render_df_section(
                "Relevant rows",
                detail_df,
                empty_message="No matching detail rows.",
            )
        )

    body = "\n".join(body_parts)

    kpis = "".join(
        f"<div class='kpi'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{_fmt_num(index_row.get(key))}</div></div>"
        for label, key in [
            ("Displayed value", "display_value"),
            ("Matched sum", "matched_value_sum"),
            ("Residual", "residual"),
            ("Matched rows", "matched_rows"),
        ]
    )

    detail_csv_name = Path(_as_str(index_row.get("detail_csv_relpath"))).name

    text = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>{html.escape(_as_str(index_row.get('drilldown_id')))}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{html.escape(_as_str(index_row.get('table_id')))}</h1>

<p>
<strong>Drilldown:</strong> {html.escape(_as_str(index_row.get('drilldown_id')))}<br>
<strong>Status:</strong> {html.escape(_as_str(index_row.get('status')))}<br>
<strong>Period:</strong> {html.escape(_as_str(index_row.get('period')))}
|
<strong>Currency:</strong> {html.escape(_as_str(index_row.get('Currency')))}
|
<strong>Measure:</strong> {html.escape(_as_str(index_row.get('measure')))}<br>
<strong>Source artifact:</strong> {html.escape(_as_str(index_row.get('source_artifact')))}
|
<strong>Lineage:</strong> {html.escape(_as_str(index_row.get('lineage_level')))}
</p>

{f"<p class='warn'><strong>Caveat:</strong> {html.escape(caveat)}</p>" if caveat else ""}

<div class='kpis'>{kpis}</div>

<h2>Filters</h2>
<pre>{html.escape(filters)}</pre>

<h2>Row context</h2>
<pre>{html.escape(row_context)}</pre>

<p><a href='{html.escape(detail_csv_name)}'>Open detail CSV</a></p>

{body}
</body>
</html>
"""

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


def _funding_metric_semantic_mask(split: pd.DataFrame, metric_id: str) -> pd.Series:
    if split.empty:
        return pd.Series(False, index=split.index)
    bucket = split.get("semantic_bucket", pd.Series("", index=split.index)).fillna("").astype(str)
    channel = split.get("funding_channel", pd.Series("", index=split.index)).fillna("").astype(str).str.strip()
    cash_effect = split.get("cash_effect", pd.Series("", index=split.index)).fillna("").astype(str).str.strip()
    debt_effect = split.get("debt_effect", pd.Series("none", index=split.index)).fillna("none").astype(str).str.strip()
    support = bucket.eq("funding_contribution") | channel.ne("") | debt_effect.ne("none")
    if metric_id in {
        "FUND.CONTRIB.BY_FUNDING_ACTOR",
        "FUND.CONTRIB.BY_CHANNEL",
        "FUND.CONTRIB.BY_CASH_EFFECT",
        "FUND.CONTRIB.BY_TARGET_BOX",
    }:
        return support
    if metric_id == "FUND.CONTRIB.DIRECT_OBLIGATION":
        return support & cash_effect.eq("no_cash_in_box_direct_payment")
    if metric_id == "FUND.CONTRIB.CASH_TO_BOX":
        return support & cash_effect.eq("cash_in_box")
    if metric_id == "FUND.CONTRIB.DEBT_LINKED":
        return support & debt_effect.ne("none")
    return pd.Series(False, index=split.index)




ANNUAL_PRESENTATION_METRIC_IDS = {
    "funding / aportes": "FUND.CONTRIB.TOTAL",
    "aportes": "FUND.CONTRIB.TOTAL",
    "retiros / gasto personal": "DIST.DRAWS.PERSONAL",
    "gasto personal": "DIST.DRAWS.PERSONAL",
    "dividendos": "DIST.DIVIDENDS",
    "cobertura después de funding y retiros": "COV.NET.AFTER_DRAWS",
    "cobertura despues de funding y retiros": "COV.NET.AFTER_DRAWS",
    "retiros / resultado operativo": "RATIO.DRAWS_TO_OPERATING_RESULT",
    "deuda total abierta": "ID.DEBT.TOTAL.OPEN",
    "principal abierto": "ID.DEBT.PRINCIPAL.OPEN",
    "interés abierto": "ID.DEBT.INTEREST.OPEN",
    "interes abierto": "ID.DEBT.INTEREST.OPEN",
}

ANNUAL_METRIC_ID_ALIASES = {
    "ID.DEBT.TOTAL.OPEN": ("BS.DEBT.TOTAL.OPEN",),
    "ID.DEBT.PRINCIPAL.OPEN": ("BS.DEBT.PRINCIPAL.OPEN",),
    "ID.DEBT.INTEREST.OPEN": ("BS.DEBT.INTEREST.OPEN",),
    "BS.DEBT.TOTAL.OPEN": ("ID.DEBT.TOTAL.OPEN",),
    "BS.DEBT.PRINCIPAL.OPEN": ("ID.DEBT.PRINCIPAL.OPEN",),
    "BS.DEBT.INTEREST.OPEN": ("ID.DEBT.INTEREST.OPEN",),
}

def _annual_metric_id_for_row(row: pd.Series) -> str:
    """Stable annual metric id from row metadata, falling back from Spanish labels."""
    candidates = _annual_metric_id_candidates_for_row(row)
    return candidates[0] if candidates else ""


def _annual_metric_id_candidates_for_row(row: pd.Series) -> list[str]:
    """Candidate metric IDs, including compatibility aliases for renamed contracts."""
    explicit = _norm(row.get("metric_id"))
    label = _metric_name(row)
    # Curated professional presentation labels win over injected metadata;
    # table-contract enrichment can otherwise stamp a generic metric_id that
    # conflicts with the human row object (notably stock/debt labels).
    primary = ANNUAL_PRESENTATION_METRIC_IDS.get(label.casefold()) or explicit or label
    if not primary:
        return []
    out = [primary]
    out.extend(ANNUAL_METRIC_ID_ALIASES.get(primary, ()))
    return list(dict.fromkeys(out))

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

    # Match on stable metadata first. Human-facing Spanish labels such as
    # "Funding / aportes" intentionally do not have to share tokens with the
    # internal metric contract (for example FUND.CONTRIB.TOTAL). Normalize
    # period keys because pandas can deserialize annual years as 2023.0 while
    # professional presentation table columns are strings such as "2023".
    metric_ids = _annual_metric_id_candidates_for_row(row)
    if not metric_ids or "metric_id" not in annual.columns:
        return pd.DataFrame()

    metric_id_set = set(metric_ids)
    mask = annual["metric_id"].map(_norm).isin(metric_id_set)
    mask &= annual.get("period", pd.Series("", index=annual.index)).map(_norm_period_key).eq(_norm_period_key(period))
    mask &= annual.get("Currency", pd.Series("", index=annual.index)).map(_norm).eq(_norm(row.get("Currency")))

    # Blank presentation dimensions mean total-row matching and should not
    # force annual blank equality. Non-empty dimensions are applied when the
    # corresponding annual columns exist.
    for dim_col in ["dimension_name", "dimension_value", "section", "dashboard_section"]:
        dim_value = _norm(row.get(dim_col)) if dim_col in row.index else ""
        if dim_value and dim_col in annual.columns:
            mask &= annual[dim_col].map(_norm).eq(dim_value)
    return annual.loc[mask].copy()


def _col_text(df: pd.DataFrame, col: str) -> pd.Series:
    return df.get(col, pd.Series("", index=df.index)).fillna("").astype(str)


def _rule_id_mask(df: pd.DataFrame, *rule_ids: str) -> pd.Series:
    if "rule_id" not in df.columns:
        return pd.Series(False, index=df.index)
    return _col_text(df, "rule_id").isin(rule_ids)


def _semantic_bucket_subbucket_mask(
    df: pd.DataFrame,
    *,
    bucket: str | None = None,
    subbucket: str | None = None,
    rule_ids: tuple[str, ...] = (),
) -> pd.Series:
    idx = df.index
    semantic_mask = pd.Series(True, index=idx)

    if bucket is not None:
        semantic_mask &= _eq_col(df, "semantic_bucket", bucket)

    if subbucket is not None:
        semantic_mask &= _eq_col(df, "semantic_subbucket", subbucket)

    rule_mask = _rule_id_mask(df, *rule_ids) if rule_ids else pd.Series(False, index=idx)

    if rule_ids:
        return semantic_mask | rule_mask

    return semantic_mask


def _amount_nonzero_mask(df: pd.DataFrame, col: str, tolerance: float = DEFAULT_TOLERANCE) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)

    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs().gt(tolerance)


def _any_flow_amount_mask(df: pd.DataFrame, tolerance: float = DEFAULT_TOLERANCE) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    for col in ["amount_in", "amount_out", "net_amount", "amount_abs"]:
        if col in df.columns:
            mask |= pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs().gt(tolerance)

    return mask

def _cash_bridge_line_spec(
    line: str,
) -> tuple[str, Callable[[pd.DataFrame], pd.Series], str] | None:
    line_n = _norm(line).casefold()

    if (
        "ingresos operativos" in line_n
        or "renta" in line_n
        or "rent" in line_n
    ):
        return (
            "amount_in",
            lambda df: _semantic_mask(
                df,
                bucket="operating_revenue",
                subbucket="rent",
                rule_ids=("R001_rent_collections",),
            ),
            "line=rent => operating_revenue/rent/R001",
        )

    if (
        "funding" in line_n
        or "contribucion" in line_n
        or "contribuciones" in line_n
        or "contribution" in line_n
        or "contributions" in line_n
    ):
        return (
            "amount_in",
            lambda df: _semantic_mask(
                df,
                bucket="funding_contribution",
                rule_ids=("R006_contribution",),
            ),
            "line=funding_contribuciones => funding_contribution/R006",
        )

    if (
        "opex propiedad" in line_n
        or "property opex" in line_n
        or "property_opex" in line_n
        or "opex patrimonial" in line_n
        or ("opex" in line_n and ("propiedad" in line_n or "property" in line_n))
    ):
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df,
                bucket="property_opex",
                rule_ids=(
                    "R002_property_taxes",
                    "R003_property_services",
                    "R004_property_maintenance",
                    "R005_property_legal",
                ),
            ),
            "line=opex_propiedad => property_opex/R002-R005",
        )

    if "impuesto" in line_n or "tax" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df,
                bucket="property_opex",
                subbucket="taxes",
                rule_ids=("R002_property_taxes",),
            ),
            "line=taxes => property_opex/taxes/R002",
        )

    if "servicio" in line_n or "service" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df,
                bucket="property_opex",
                subbucket="services",
                rule_ids=("R003_property_services",),
            ),
            "line=services => property_opex/services/R003",
        )

    if "mantenimiento" in line_n or "maintenance" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df,
                bucket="property_opex",
                subbucket="maintenance",
                rule_ids=("R004_property_maintenance",),
            ),
            "line=maintenance => property_opex/maintenance/R004",
        )

    if "legal" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df,
                bucket="property_opex",
                subbucket="legal",
                rule_ids=("R005_property_legal",),
            ),
            "line=legal => property_opex/legal/R005",
        )

    if (
        "movimiento neto de deuda" in line_n
        or "debt net" in line_n
        or "net debt" in line_n
        or ("deuda" in line_n and "neto" in line_n)
        or ("debt" in line_n and "net" in line_n)
    ):
        return (
            "net_amount",
            lambda df: _semantic_mask(
                df,
                bucket="debt_movement",
                rule_ids=(
                    "R007_debt_principal",
                    "R008_debt_repayment",
                    "R009_debt_interest",
                ),
            ),
            "line=movimiento_neto_deuda => debt_movement/R007-R009 net_amount",
        )

    if (
        "personal" in line_n
        or "retiro" in line_n
        or "retiros" in line_n
        or "withdrawal" in line_n
        or "withdrawals" in line_n
        or "draw" in line_n
        or "draws" in line_n
        or "distribucion" in line_n
        or "distribución" in line_n
        or "distribution" in line_n
    ):
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df,
                bucket="family_withdrawal_candidate",
                rule_ids=(
                    "R010_dividend",
                    "R011_personal_expense_text",
                    "R012_transfer_expense",
                ),
            ),
            "line=retiros_gasto_familiar => family_withdrawal_candidate/R010-R012",
        )

    # Aggregate cash bridge rows.
    if (
        "total entradas" in line_n
        or "total inflows" in line_n
        or "total income" in line_n
    ):
        return (
            "amount_in",
            lambda df: _amount_nonzero_mask(df, "amount_in"),
            "line=total_entradas => amount_in + rows with amount_in != 0",
        )

    if (
        "total salidas" in line_n
        or "total outflows" in line_n
        or "total expenses" in line_n
    ):
        return (
            "amount_out",
            lambda df: _amount_nonzero_mask(df, "amount_out"),
            "line=total_salidas => amount_out + rows with amount_out != 0",
        )

    if (
        "flujo neto observado" in line_n
        or "net flow observed" in line_n
        or "flujo neto" in line_n
        or "net flow" in line_n
    ):
        return (
            "net_amount",
            _any_flow_amount_mask,
            "line=flujo_neto_observado => net_amount + all nonzero flow rows",
        )

    if "fx" in line_n or "cambio" in line_n or "treasury" in line_n:
        return (
            "net_amount",
            _fx_mask,
            "line=fx/treasury => treasury/fx mask",
        )

    if "unknown" in line_n or "review" in line_n or "revis" in line_n:
        return (
            "net_amount",
            _unknown_mask,
            "line=unknown/review => unknown mask",
        )

    return None

def _strict_eq_col(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    """Fail-closed equality for semantic filters."""
    if col not in df.columns or _norm(value) == "":
        return pd.Series(False, index=df.index)
    return df[col].astype(str).fillna("").eq(_as_str(value))



def _optional_strict_eq_col(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    """
    If the row value is blank, do not filter on this column.
    If the row value is present, the source column must exist and match.
    """
    value_n = _norm(value)
    if value_n == "":
        return pd.Series(True, index=df.index)

    if col not in df.columns:
        return pd.Series(False, index=df.index)

    return df[col].fillna("").astype(str).str.strip().eq(value_n)


def _rule_token_mask(df: pd.DataFrame, *rule_ids: str) -> pd.Series:
    """Match rule_id or rule_ids columns, including semicolon-separated rule lists."""
    if not rule_ids:
        return pd.Series(False, index=df.index)

    wanted = set(rule_ids)
    mask = pd.Series(False, index=df.index)

    for col in ["rule_id", "rule_ids"]:
        if col not in df.columns:
            continue

        s = df[col].fillna("").astype(str)

        for rid in wanted:
            mask |= s.eq(rid)
            # mask |= s.str.contains(rf"(^|;){re.escape(rid)}(;|$)", regex=True, na=False)
            mask |= s.str.contains(rf"(?:^|%3b){re.escape(rid)}(?:$|%3b)", regex=True, na=False)


    return mask


def _semantic_mask(
    df: pd.DataFrame,
    *,
    bucket: str | None = None,
    subbucket: str | None = None,
    rule_ids: tuple[str, ...] = (),
) -> pd.Series:
    """
    Semantic mask for professional drilldowns.

    Important: semantic columns fail closed. Missing semantic_bucket should not mean "all rows".
    Rule IDs are allowed as an OR path because split/audit may expose rule_id/rule_ids differently.
    """
    idx = df.index

    sem = pd.Series(True, index=idx)

    if bucket is not None:
        sem &= _strict_eq_col(df, "semantic_bucket", bucket)

    if subbucket is not None:
        sem &= _strict_eq_col(df, "semantic_subbucket", subbucket)

    rules = _rule_token_mask(df, *rule_ids)

    if rule_ids:
        return sem | rules

    return sem


def _extract_after_marker(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].strip()




@dataclass(frozen=True)
class AnnualFormulaSpec:
    formula_id: str
    label: str
    component_metric_ids: tuple[str, ...]
    formula: str


def _annual_formula_spec(row: pd.Series) -> AnnualFormulaSpec | None:
    """Formula/ratio presentation rows that should not look for one raw row."""
    label = _metric_name(row).casefold().strip()
    if label == "margen operativo":
        return AnnualFormulaSpec(
            "operating_margin",
            "Margen operativo",
            ("IS.NET.OPERATING", "IS.REVENUE.OPERATING"),
            "IS.NET.OPERATING / IS.REVENUE.OPERATING",
        )
    if label == "opex / renta":
        return AnnualFormulaSpec(
            "opex_to_rent",
            "OPEX / renta",
            ("IS.OPEX.PROPERTY", "IS.REVENUE.OPERATING"),
            "IS.OPEX.PROPERTY / IS.REVENUE.OPERATING",
        )
    if label == "retiros / resultado operativo":
        return AnnualFormulaSpec(
            "draws_to_operating_result",
            "Retiros / resultado operativo",
            ("DIST.DRAWS.PERSONAL", "IS.NET.OPERATING"),
            "DIST.DRAWS.PERSONAL / IS.NET.OPERATING",
        )
    if label == "cobertura después de funding y retiros":
        return AnnualFormulaSpec(
            "coverage_after_funding_and_draws",
            "Cobertura después de funding y retiros",
            ("COV.NET.AFTER_DRAWS", "IS.NET.OPERATING", "FUND.CONTRIB.TOTAL", "DIST.DRAWS.PERSONAL"),
            "COV.NET.AFTER_DRAWS or IS.NET.OPERATING + FUND.CONTRIB.TOTAL - DIST.DRAWS.PERSONAL",
        )
    return None


def _annual_component_rows(annual: pd.DataFrame, period: str, currency: str, metric_ids: tuple[str, ...]) -> pd.DataFrame:
    if annual.empty:
        return pd.DataFrame()
    mask = annual.get("period", pd.Series("", index=annual.index)).map(_norm_period_key).eq(_norm_period_key(period))
    mask &= annual.get("Currency", pd.Series("", index=annual.index)).map(_norm).eq(_norm(currency))
    if "metric_id" in annual.columns:
        mask &= annual["metric_id"].map(_norm).isin(metric_ids)
    return annual.loc[mask].copy()


def _value_by_metric(component_rows: pd.DataFrame) -> dict[str, float]:
    if component_rows.empty or "metric_id" not in component_rows.columns:
        return {}
    values: dict[str, float] = {}
    for metric_id, group in component_rows.groupby(component_rows["metric_id"].astype(str), dropna=False):
        values[str(metric_id)] = _measure_sum(group, "value")
    return values


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > DEFAULT_TOLERANCE else 0.0


def _build_annual_formula_cell(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    currency: str,
    display_value: float,
    annual: pd.DataFrame,
    tolerance: float,
) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]] | None:
    spec = _annual_formula_spec(row)
    if spec is None:
        return None

    component_rows = _annual_component_rows(annual, period, currency, spec.component_metric_ids)
    values = _value_by_metric(component_rows)

    if spec.formula_id == "operating_margin":
        matched = _safe_div(values.get("IS.NET.OPERATING", 0.0), values.get("IS.REVENUE.OPERATING", 0.0))
    elif spec.formula_id == "opex_to_rent":
        matched = _safe_div(values.get("IS.OPEX.PROPERTY", 0.0), values.get("IS.REVENUE.OPERATING", 0.0))
    elif spec.formula_id == "draws_to_operating_result":
        matched = _safe_div(values.get("DIST.DRAWS.PERSONAL", 0.0), values.get("IS.NET.OPERATING", 0.0))
    elif spec.formula_id == "coverage_after_funding_and_draws":
        matched = values.get(
            "COV.NET.AFTER_DRAWS",
            values.get("IS.NET.OPERATING", 0.0) + values.get("FUND.CONTRIB.TOTAL", 0.0) - values.get("DIST.DRAWS.PERSONAL", 0.0),
        )
    else:
        matched = 0.0

    residual = matched - display_value
    status = STATUS_EMPTY if component_rows.empty else STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
    formula_rows = pd.DataFrame([
        {
            "formula_id": spec.formula_id,
            "formula": spec.formula,
            "displayed_value": display_value,
            "matched_value": matched,
            "residual": residual,
            **values,
        }
    ])
    sections = [("Formula", formula_rows), ("Component annual rows", component_rows)]
    filters = {
        "period": period,
        "Currency": currency,
        "formula_id": spec.formula_id,
        "formula": spec.formula,
        "component_metric_ids": list(spec.component_metric_ids),
        "row_context": _row_context(table_id, row),
    }
    return (
        status,
        matched,
        residual,
        "annual_formula_components",
        "annual_balance_dashboard_metrics.csv",
        filters,
        "Formula/ratio drilldown: components are annual metric rows, not one raw ledger row.",
        component_rows if not component_rows.empty else formula_rows,
        sections,
    )

def _annual_professional_line_spec(
    row: pd.Series,
) -> tuple[str, Callable[[pd.DataFrame], pd.Series], str] | None:
    """
    Direct semantic contract for annual professional presentation rows.

    Returns:
      measure, filter_fn, note

    This is used when annual_balance_dashboard_metrics.csv has no exact metric_id
    match for the human-facing row label.
    """
    line_raw = _norm(row.get("line")) or _norm(row.get("label")) or _metric_name(row)
    line_n = line_raw.casefold()

    if "renta total" in line_n or (
        "ingresos operativos" in line_n and "renta" in line_n
    ):
        return (
            "amount_in",
            lambda df: _semantic_mask(
                df,
                bucket="operating_revenue",
                subbucket="rent",
                rule_ids=("R001_rent_collections",),
            ),
            "annual professional line=renta_total => operating_revenue/rent/R001",
        )

    if "renta por propiedad" in line_n or "rent by property" in line_n:
        lugar = ""

        # Example: "Renta por propiedad — Lugar: CABA"
        if "lugar:" in line_n:
            # Preserve original case/value from raw line.
            lugar = _extract_after_marker(line_raw, "Lugar:")
        elif "Lugar" in row.index and _norm(row.get("Lugar")):
            lugar = _norm(row.get("Lugar"))

        def _rent_property_mask(df: pd.DataFrame, lugar=lugar) -> pd.Series:
            mask = _semantic_mask(
                df,
                bucket="operating_revenue",
                subbucket="rent",
                rule_ids=("R001_rent_collections",),
            )
            if lugar:
                mask &= _strict_eq_col(df, "Lugar", lugar)
            return mask

        return (
            "amount_in",
            _rent_property_mask,
            f"annual professional line=renta_por_propiedad => operating_revenue/rent/R001; Lugar={lugar}",
        )

    if "opex propiedad" in line_n or "opex patrimonial" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df,
                bucket="property_opex",
                rule_ids=(
                    "R002_property_taxes",
                    "R003_property_services",
                    "R004_property_maintenance",
                    "R005_property_legal",
                ),
            ),
            "annual professional line=opex_propiedad => property_opex/R002-R005",
        )

    if "opex por categoría" in line_n or "opex por categoria" in line_n:
        subbucket = ""

        if "impuesto" in line_n or "tax" in line_n:
            subbucket = "taxes"
            rule_ids = ("R002_property_taxes",)
        elif "servicio" in line_n or "service" in line_n:
            subbucket = "services"
            rule_ids = ("R003_property_services",)
        elif "mantenimiento" in line_n or "maintenance" in line_n:
            subbucket = "maintenance"
            rule_ids = ("R004_property_maintenance",)
        elif "legal" in line_n:
            subbucket = "legal"
            rule_ids = ("R005_property_legal",)
        else:
            return None

        return (
            "amount_out",
            lambda df, subbucket=subbucket, rule_ids=rule_ids: _semantic_mask(
                df,
                bucket="property_opex",
                subbucket=subbucket,
                rule_ids=rule_ids,
            ),
            f"annual professional line=opex_categoria => property_opex/{subbucket}",
        )

    if "resultado operativo neto" in line_n or "net operating" in line_n:
        return (
            "net_operating_formula",
            lambda df: pd.Series(True, index=df.index),
            "annual professional line=resultado_operativo_neto => rent amount_in - property_opex amount_out",
        )

    return None


def _is_cash_bridge_debt_movement_line(row: pd.Series) -> bool:
    line = _metric_name(row).casefold()
    return (
        "movimiento neto de deuda" in line
        or "debt net" in line
        or "net debt" in line
        or ("deuda" in line and "neto" in line)
        or ("debt" in line and "net" in line)
    )


def _cash_bridge_semantic_rows(
    split: pd.DataFrame,
    row: pd.Series,
    period: str,
) -> tuple[
    pd.DataFrame,
    str,
    str,
    Callable[[pd.DataFrame], pd.Series] | None,
    bool,
]:
    metric_id = _norm(row.get("metric_id"))
    if metric_id.startswith("FUND.CONTRIB.") and not _is_cash_bridge_debt_movement_line(row):
        mask = _year_mask(split, period) & _eq_col(split, "Currency", row.get("Currency"))
        mask &= _funding_metric_semantic_mask(split, metric_id)

        # Stable contract dimensions win over human bridge labels.  These
        # filters are intentionally explicit so funding/support rows are
        # separated by actor, channel, target/obligation box, cash effect, and
        # debt effect instead of inferred from display text.
        dim_name = _norm(row.get("dimension_name"))
        dim_value = _norm(row.get("dimension_value"))
        if dim_name and dim_value and dim_name in split.columns:
            mask &= _eq_col(split, dim_name, dim_value)

        for col in (
            "Box",
            "cash_path",
            "funding_channel",
            "funding_actor",
            "cash_effect",
            "target_box",
            "beneficiary_box",
            "obligation_box",
            "source_box",
            "debt_effect",
        ):
            if col in row.index and _norm(row.get(col)):
                mask &= _eq_col(split, col, row.get(col))

        cash_effect = _norm(row.get("cash_effect")) or (
            dim_value if dim_name == "cash_effect" else ""
        )
        funding_channel = _norm(row.get("funding_channel")) or (
            dim_value if dim_name == "funding_channel" else ""
        )
        if metric_id == "FUND.CONTRIB.DIRECT_OBLIGATION" or cash_effect == "no_cash_in_box_direct_payment" or funding_channel in {"tenant_direct_tax_payment", "tenant_direct_service_payment"}:
            measure = "amount_out"
        elif metric_id == "FUND.CONTRIB.DEBT_LINKED":
            measure = "amount_abs"
        else:
            measure = "amount_in"

        def contract_filter(df: pd.DataFrame) -> pd.Series:
            out = _funding_metric_semantic_mask(df, metric_id)
            if dim_name and dim_value and dim_name in df.columns:
                out &= _eq_col(df, dim_name, dim_value)
            for col in (
                "Box",
                "cash_path",
                "funding_channel",
                "funding_actor",
                "cash_effect",
                "target_box",
                "beneficiary_box",
                "obligation_box",
                "source_box",
                "debt_effect",
            ):
                if col in row.index and _norm(row.get(col)):
                    out &= _eq_col(df, col, row.get(col))
            return out

        note_bits = [f"metric_id={metric_id}"]
        if dim_name and dim_value:
            note_bits.append(f"{dim_name}={dim_value}")
        return (
            split.loc[mask].copy(),
            measure,
            "year/Currency + stable funding contract (" + "; ".join(note_bits) + ")",
            contract_filter,
            True,
        )

    line = _metric_name(row)
    line_spec = _cash_bridge_line_spec(line)

    if line_spec is None:
        return (
            pd.DataFrame(),
            "",
            f"unsupported cash bridge line: {line}",
            None,
            False,
        )

    measure, line_filter, line_note = line_spec

    mask = _year_mask(split, period) & _eq_col(split, "Currency", row.get("Currency"))

    # Optional dimensional filters from the row.
    # These can remain fail-open because they are optional row dimensions.
    for col in ["Box", "cash_path"]:
        if col in row.index and _norm(row.get(col)):
            mask &= _eq_col(split, col, row.get(col))

    # Core semantic filter. This must be applied.
    mask &= line_filter(split)

    return (
        split.loc[mask].copy(),
        measure,
        f"year/Currency/Box + {line_note}",
        line_filter,
        True,
    )


def _build_annual_professional_fallback(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    currency: str,
    display_value: float,
    split: pd.DataFrame,
    audit: pd.DataFrame,
    tolerance: float,
) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]] | None:
    spec = _annual_professional_line_spec(row)
    if spec is None:
        return None

    measure, line_filter, note = spec

    if split.empty:
        return (
            STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_flow_semantic_split.csv",
            {
                "error": "missing monthly_flow_semantic_split.csv",
                "period": period,
                "Currency": currency,
                "measure": measure,
            },
            "Annual professional fallback requires monthly_flow_semantic_split.csv.",
            pd.DataFrame(),
            [],
        )

    base_mask = _year_mask(split, period) & _eq_col(split, "Currency", currency)

    sections: list[tuple[str, pd.DataFrame]] = []
    caveat = "Annual professional drilldown rebuilt directly from monthly semantic split."

    if measure == "net_operating_formula":
        revenue_filter = lambda df: _semantic_mask(
            df,
            bucket="operating_revenue",
            subbucket="rent",
            rule_ids=("R001_rent_collections",),
        )
        opex_filter = lambda df: _semantic_mask(
            df,
            bucket="property_opex",
            rule_ids=(
                "R002_property_taxes",
                "R003_property_services",
                "R004_property_maintenance",
                "R005_property_legal",
            ),
        )

        revenue_rows = split.loc[base_mask & revenue_filter(split)].copy()
        opex_rows = split.loc[base_mask & opex_filter(split)].copy()

        revenue = _measure_sum(revenue_rows, "amount_in")
        opex = _measure_sum(opex_rows, "amount_out")
        matched = revenue - opex
        residual = matched - display_value

        semantic_rows = pd.concat([revenue_rows, opex_rows], ignore_index=True)

        def audit_filter(df: pd.DataFrame) -> pd.Series:
            return (
                _year_mask(df, period)
                & _eq_col(df, "Currency", currency)
                & (revenue_filter(df) | opex_filter(df))
            )

        detail_rows, lineage = _detail_from_audit(audit, semantic_rows, audit_filter)

        sections.append(
            (
                "Formula",
                pd.DataFrame(
                    [
                        {
                            "formula": "operating_revenue/rent amount_in - property_opex amount_out",
                            "revenue_amount_in": revenue,
                            "property_opex_amount_out": opex,
                            "matched_value": matched,
                            "displayed_value": display_value,
                            "residual": residual,
                        }
                    ]
                ),
            )
        )
        sections.append(("Semantic rows", semantic_rows))
        sections.append(("Classification rows", detail_rows))

        status = (
            STATUS_EMPTY
            if semantic_rows.empty
            else STATUS_OK
            if abs(residual) <= tolerance
            else STATUS_RESIDUAL_WARNING
        )

        filters = {
            "period": period,
            "Currency": currency,
            "measure": "net_amount",
            "source": "monthly_flow_semantic_split.csv",
            "line": _norm(row.get("line")),
            "semantic_contract": note,
            "formula": "operating_revenue/rent amount_in - property_opex amount_out",
        }

        return (
            status,
            matched,
            residual,
            lineage,
            "monthly_flow_semantic_split.csv",
            filters,
            caveat,
            detail_rows if not detail_rows.empty else semantic_rows,
            sections,
        )

    semantic_rows = split.loc[base_mask & line_filter(split)].copy()
    matched = _measure_sum(semantic_rows, measure)
    residual = matched - display_value

    def audit_filter(df: pd.DataFrame) -> pd.Series:
        return _year_mask(df, period) & _eq_col(df, "Currency", currency) & line_filter(df)

    detail_rows, lineage = _detail_from_audit(audit, semantic_rows, audit_filter)

    sections.append(("Semantic rows", semantic_rows))
    sections.append(("Classification rows", detail_rows))

    status = (
        STATUS_EMPTY
        if semantic_rows.empty
        else STATUS_OK
        if abs(residual) <= tolerance
        else STATUS_RESIDUAL_WARNING
    )

    filters = {
        "period": period,
        "Currency": currency,
        "measure": measure,
        "source": "monthly_flow_semantic_split.csv",
        "line": _norm(row.get("line")),
        "semantic_contract": note,
    }

    return (
        status,
        matched,
        residual,
        lineage,
        "monthly_flow_semantic_split.csv",
        filters,
        caveat,
        detail_rows if not detail_rows.empty else semantic_rows,
        sections,
    )



def _pair_parts(pair: Any) -> tuple[str, str]:
    text = _norm(pair)
    for sep in ["→", "->", "=>", " to "]:
        if sep in text:
            left, right = text.split(sep, 1)
            return left.strip(), right.strip()
    return "", ""


def _source_filter_eq(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    """Strict equality for source mart filters."""
    if col not in df.columns or _norm(value) == "":
        return pd.Series(False, index=df.index)
    return df[col].astype(str).fillna("").str.strip().eq(_norm(value))


def _period_eq(df: pd.DataFrame, period: str) -> pd.Series:
    if "period" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["period"].astype(str).eq(str(period))



def _prev_month_period(period: str) -> str:
    ts = pd.Period(str(period), freq="M")
    return str(ts - 1)


def _is_box_level_cash_row(df: pd.DataFrame) -> pd.Series:
    """
    Rows that represent the box-level cash/control balance.

    Excludes party-level internal balances such as Alejandro, MI, Servicios,
    Impuestos, etc. Those can be useful evidence, but they are not the
    reconciliation grain for monthly_tables_cash_close_matrix.
    """
    mask = pd.Series(False, index=df.index)

    if "source_table" in df.columns:
        mask |= df["source_table"].fillna("").astype(str).str.strip().eq(
            "box_balance_time_long.freq=M.csv"
        )

    if "source_type" in df.columns:
        mask |= df["source_type"].fillna("").astype(str).str.strip().eq(
            "inferred_box_motor"
        )

    if "position_type" in df.columns:
        mask |= df["position_type"].fillna("").astype(str).str.strip().eq(
            "inferred_box_motor"
        )

    # Secondary fallback: explicit box row with no party.
    # Keep this after the stronger source flags.
    if "party" in df.columns:
        party_blank = df["party"].isna() | df["party"].astype(str).str.strip().isin(["", "nan", "NaN"])
        if party_blank.any():
            mask |= party_blank

    return mask


def _cash_close_box_rows(
    cash_close: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
) -> pd.DataFrame:
    base = _cash_close_rows(
        cash_close,
        period=period,
        currency=currency,
        box=box,
    )

    if base.empty:
        return base

    box_mask = _is_box_level_cash_row(base)
    box_rows = base.loc[box_mask].copy()

    return box_rows


def _cash_close_box_rows_with_base_fallback(
    cash_close: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
) -> pd.DataFrame:
    box_rows = _cash_close_box_rows(
        cash_close,
        period=period,
        currency=currency,
        box=box,
    )
    if not box_rows.empty:
        return box_rows

    # The diagnostic table is produced from normalized box-balance sources. In
    # some historical months the authoritative prior close row lacks the modern
    # box-level provenance flags used by monthly cash-close drilldowns. For
    # month-over-month diagnostics, use the same period/currency/box row rather
    # than silently treating an existing prior month as zero.
    fallback = _cash_close_rows(
        cash_close,
        period=period,
        currency=currency,
        box=box,
    ).copy()
    if not fallback.empty:
        fallback["box_level_fallback_reason"] = (
            "no flagged box-level row; using period/currency/box close row"
        )
    return fallback


def _cash_close_party_rows(
    cash_close: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
) -> pd.DataFrame:
    base = _cash_close_rows(
        cash_close,
        period=period,
        currency=currency,
        box=box,
    )

    if base.empty:
        return base

    box_mask = _is_box_level_cash_row(base)
    return base.loc[~box_mask].copy()

def _cash_close_rows(
    cash_close: pd.DataFrame,
    *,
    period: str,
    currency: str,
    box: str,
) -> pd.DataFrame:
    if cash_close.empty:
        return pd.DataFrame()

    mask = (
        _period_eq(cash_close, period)
        & _source_filter_eq(cash_close, "Currency", currency)
        & _source_filter_eq(cash_close, "Box", box)
    )

    return cash_close.loc[mask].copy()


def _close_amount_sum(df: pd.DataFrame) -> float:
    if df.empty or "close_amount" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["close_amount"], errors="coerce").fillna(0.0).sum())



def _first_existing_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


# def _build_cash_close_cell(
#     *,
#     row: pd.Series,
#     period: str,
#     display_value: float,
#     cash_close: pd.DataFrame,
#     tolerance: float,
# ) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
#     currency = _norm(row.get("Currency"))
#     box = _norm(row.get("Box"))
#     metric = _norm(row.get("metric")) or _norm(row.get("measure")) or "cash_close"

#     if cash_close.empty:
#         return (
#             STATUS_ERROR,
#             0.0,
#             -display_value,
#             "missing_source",
#             "monthly_cash_close.csv",
#             {"error": "missing monthly_cash_close.csv", "period": period, "Currency": currency, "Box": box},
#             "Cash close drilldown requires monthly_cash_close.csv.",
#             pd.DataFrame(),
#             [],
#         )

#     value_col = _first_existing_col(
#         cash_close,
#         (
#             metric,
#             "cash_close",
#             "closing_cash",
#             "closing_balance",
#             "close_balance",
#             "balance",
#             "amount",
#             "value",
#         ),
#     )

#     if not value_col:
#         return (
#             STATUS_ERROR,
#             0.0,
#             -display_value,
#             "missing_value_column",
#             "monthly_cash_close.csv",
#             {
#                 "error": "no recognized cash close value column",
#                 "period": period,
#                 "Currency": currency,
#                 "Box": box,
#                 "available_columns": list(cash_close.columns),
#             },
#             "Cash close source exists but has no recognized value column.",
#             pd.DataFrame(),
#             [],
#         )

#     mask = (
#         _period_eq(cash_close, period)
#         & _source_filter_eq(cash_close, "Currency", currency)
#         & _source_filter_eq(cash_close, "Box", box)
#     )

#     source = cash_close.loc[mask].copy()
#     matched = _measure_sum(source, value_col)
#     residual = matched - display_value

#     status = (
#         STATUS_EMPTY
#         if source.empty
#         else STATUS_OK
#         if abs(residual) <= tolerance
#         else STATUS_RESIDUAL_WARNING
#     )

#     filters = {
#         "period": period,
#         "Currency": currency,
#         "Box": box,
#         "measure": value_col,
#         "source": "monthly_cash_close.csv",
#         "metric": metric,
#     }

#     sections = [("Cash close rows", source)]

#     return (
#         status,
#         matched,
#         residual,
#         "monthly_cash_close",
#         "monthly_cash_close.csv",
#         filters,
#         "Cash close is a stock/control mart drilldown, not a semantic flow drilldown.",
#         source,
#         sections,
#     )



def _first_existing_col_ci(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Case-insensitive first existing column lookup; returns actual column name."""
    by_norm = {str(c).strip().casefold(): c for c in df.columns}
    for c in candidates:
        key = str(c).strip().casefold()
        if key in by_norm:
            return str(by_norm[key])
    return ""


def _cash_metric_value_column(cash_close: pd.DataFrame, metric: str) -> str:
    metric_n = _norm(metric).casefold()

    # Metric-specific physical column candidates.
    if metric_n == "cash_close":
        candidates = (
            "cash_close",
            "cash_close_level",
            "cash_level",
            "cash_balance",
            "cash_balance_level",
            "validated_cash",
            "validated_cash_level",
            "closing_cash",
            "closing_cash_level",
            "closing_balance",
            "closing_balance_level",
            "ending_cash",
            "ending_balance",
            "balance",
            "value",
            "metric_value",
            "amount",
            "metric_amount",
        )
    elif metric_n == "diagnostic_box_level":
        candidates = (
            "diagnostic_box_level",
            "box_level_diagnostic",
            "box_balance_m_diagnostic",
            "diagnostic_level",
            "cash_delta",
            "cash_residual",
            "close_residual",
            "residual",
            "value",
            "metric_value",
            "amount",
            "metric_amount",
        )
    else:
        candidates = (
            metric,
            "value",
            "metric_value",
            "amount",
            "metric_amount",
            "balance",
        )

    return _first_existing_col_ci(cash_close, candidates)




def _build_cash_control_cell(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    source_df: pd.DataFrame,
    source_name: str,
    default_metric: str,
    tolerance: float,
) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    currency = _norm(row.get("Currency"))
    box = _norm(row.get("Box"))
    metric = _norm(row.get("metric")) or _norm(row.get("measure")) or default_metric

    if source_df.empty:
        return (
            STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            source_name,
            {
                "error": f"missing {source_name}",
                "period": period,
                "Currency": currency,
                "Box": box,
                "metric": metric,
            },
            f"Cash/control drilldown requires {source_name}.",
            pd.DataFrame(),
            [],
        )

    if "close_amount" not in source_df.columns:
        return (
            STATUS_ERROR,
            0.0,
            -display_value,
            "missing_value_column",
            source_name,
            {
                "error": "monthly_cash_close.csv has no close_amount column",
                "period": period,
                "Currency": currency,
                "Box": box,
                "metric": metric,
                "source_name": source_name,
                "available_columns": list(source_df.columns),
            },
            "Cash/control source exists but has no close_amount column.",
            pd.DataFrame(),
            [],
        )

    current_rows = _cash_close_rows(
        source_df,
        period=period,
        currency=currency,
        box=box,
    )

    current_close = _close_amount_sum(current_rows)

    sections: list[tuple[str, pd.DataFrame]] = []


    if metric == "cash_close":
        box_rows = _cash_close_box_rows(
            source_df,
            period=period,
            currency=currency,
            box=box,
        )

        party_rows = _cash_close_party_rows(
            source_df,
            period=period,
            currency=currency,
            box=box,
        )

        detail_df = box_rows.copy()

        if not detail_df.empty:
            detail_df["cash_close"] = pd.to_numeric(
                detail_df["close_amount"],
                errors="coerce",
            ).fillna(0.0)

        matched = float(detail_df["cash_close"].sum()) if "cash_close" in detail_df.columns else 0.0
        residual = matched - display_value

        status = (
            STATUS_EMPTY
            if detail_df.empty
            else STATUS_OK
            if abs(residual) <= tolerance
            else STATUS_RESIDUAL_WARNING
        )

        sections.append(("Box-level cash close row used for reconciliation", detail_df))

        if not party_rows.empty:
            party_rows = party_rows.copy()
            party_rows["excluded_from_reconciliation"] = True
            sections.append(("Party-level internal balance rows excluded", party_rows))

        filters = {
            "period": period,
            "Currency": currency,
            "Box": box,
            "metric": metric,
            "measure": "cash_close",
            "source_measure": "close_amount",
            "source": source_name,
            "calculation_rule": (
                "box-level close_amount only; excludes party-level internal_balance rows"
            ),
            "box_level_row_filter": (
                "source_table=box_balance_time_long.freq=M.csv "
                "OR source_type=inferred_box_motor "
                "OR position_type=inferred_box_motor "
                "OR blank party fallback"
            ),
        }

        return (
            status,
            matched,
            residual,
            source_name.removesuffix(".csv"),
            source_name,
            filters,
            "Cash close is reconciled at box-level grain; party-level internal balances are excluded from the matched sum.",
            detail_df,
            sections,
        )




    if metric == "diagnostic_box_level":
        prev_period = _prev_month_period(period)

        current_rows = _cash_close_box_rows(
            source_df,
            period=period,
            currency=currency,
            box=box,
        )

        previous_rows = _cash_close_box_rows_with_base_fallback(
            source_df,
            period=prev_period,
            currency=currency,
            box=box,
        )

        current_close = _close_amount_sum(current_rows)
        previous_close = _close_amount_sum(previous_rows)

        matched = current_close - previous_close
        residual = matched - display_value

        formula_df = pd.DataFrame(
            [
                {
                    "period": period,
                    "previous_period": prev_period,
                    "Currency": currency,
                    "Box": box,
                    "current_cash_close": current_close,
                    "previous_cash_close": previous_close,
                    "diagnostic_box_level": matched,
                    "display_value": display_value,
                    "residual": residual,
                    "calculation_rule": (
                        "current box-level close_amount - previous month close_amount; "
                        "previous month falls back to period/currency/box close rows "
                        "when box-level provenance flags are absent"
                    ),
                }
            ]
        )

        current_detail = current_rows.copy()
        previous_detail = previous_rows.copy()

        if not current_detail.empty:
            current_detail["period_role"] = "current"

        if not previous_detail.empty:
            previous_detail["period_role"] = "previous"

        sections.append(("Diagnostic formula", formula_df))
        sections.append(("Current box-level cash close row", current_detail))
        sections.append(("Previous box-level cash close row", previous_detail))

        status = (
            STATUS_EMPTY
            if current_rows.empty and previous_rows.empty
            else STATUS_OK
            if abs(residual) <= tolerance
            else STATUS_RESIDUAL_WARNING
        )

        filters = {
            "period": period,
            "previous_period": prev_period,
            "Currency": currency,
            "Box": box,
            "metric": metric,
            "measure": "diagnostic_box_level",
            "source_measure": "close_amount",
            "source": source_name,
            "calculation_rule": (
                "box-level current close_amount - previous month close_amount; "
                "previous month falls back to period/currency/box close rows when "
                "box-level provenance flags are absent"
            ),
        }

        return (
            status,
            matched,
            residual,
            source_name.removesuffix(".csv"),
            source_name,
            filters,
            "Diagnostic box level is a month-over-month delta of box-level cash close.",
            formula_df,
            sections,
        )




    return (
        STATUS_UNSUPPORTED,
        0.0,
        -display_value,
        "unsupported",
        source_name,
        {
            "unsupported": True,
            "reason": f"unsupported cash/control metric: {metric}",
            "period": period,
            "Currency": currency,
            "Box": box,
            "metric": metric,
            "available_columns": list(source_df.columns),
        },
        "Cash/control metric is not mapped.",
        pd.DataFrame(),
        [],
    )

DEBT_ACTIVITY_TYPE_FOR_MEASURE = {
    "new_principal": "new_claim",
    "interest_accrued": "interest_accrual",
    "repayments": "repayment",
    "adjustments": "adjustment",
    "net_change": "net_change",
}


def _build_debt_activity_cell(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_activity: pd.DataFrame,
    tolerance: float,
) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    measure = _norm(row.get("measure"))
    currency = _norm(row.get("Currency"))
    pair = _norm(row.get("pair"))
    debtor, creditor = _pair_parts(pair)

    if debt_activity.empty:
        return (
            STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_debt_activity.csv",
            {"error": "missing monthly_debt_activity.csv", "period": period, "Currency": currency, "pair": pair, "measure": measure},
            "Debt activity drilldown requires monthly_debt_activity.csv.",
            pd.DataFrame(),
            [],
        )

    if measure not in debt_activity.columns:
        return (
            STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_debt_activity.csv",
            {
                "unsupported": True,
                "reason": "measure not present in monthly_debt_activity.csv",
                "period": period,
                "Currency": currency,
                "pair": pair,
                "measure": measure,
                "available_columns": list(debt_activity.columns),
            },
            "Debt activity measure has no source column.",
            pd.DataFrame(),
            [],
        )

    mask = (
        _period_eq(debt_activity, period)
        & _source_filter_eq(debt_activity, "Currency", currency)
        & _source_filter_eq(debt_activity, "debtor", debtor)
        & _source_filter_eq(debt_activity, "creditor", creditor)
    )

    activity_type = DEBT_ACTIVITY_TYPE_FOR_MEASURE.get(measure)
    if activity_type and "activity_type" in debt_activity.columns:
        mask &= _source_filter_eq(debt_activity, "activity_type", activity_type)

    source = debt_activity.loc[mask].copy()
    matched = _measure_sum(source, measure)
    residual = matched - display_value

    status = (
        STATUS_EMPTY
        if source.empty
        else STATUS_OK
        if abs(residual) <= tolerance
        else STATUS_RESIDUAL_WARNING
    )

    filters = {
        "period": period,
        "Currency": currency,
        "pair": pair,
        "debtor": debtor,
        "creditor": creditor,
        "measure": measure,
        "activity_type": activity_type or "",
        "source": "monthly_debt_activity.csv",
    }

    sections = [("Debt activity rows", source)]

    return (
        status,
        matched,
        residual,
        "monthly_debt_activity",
        "monthly_debt_activity.csv",
        filters,
        "Debt activity is resolved-debt evidence, not semantic operating-flow evidence.",
        source,
        sections,
    )




DEBT_COMPONENT_FOR_MEASURE = {
    "open_total": "total",
    "open_principal": "principal",
    "open_interest": "interest",
}


def _select_monthly_debt_position_snapshot(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the selected monthly debt stock snapshot for a filtered cell.

    Debt position rows are stock snapshots, not flows.  When more than one
    snapshot exists for the same period / debtor / creditor / currency /
    component, reconciliation must use the latest as_of_date in the period
    instead of summing all snapshots.
    """
    if rows.empty or "as_of_date" not in rows.columns:
        return rows.copy()

    out = rows.copy()
    out["__as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    valid = out["__as_of_date"].notna()
    if not valid.any():
        return out.drop(columns=["__as_of_date"])

    selected = (
        out.loc[valid]
        .sort_values(["__as_of_date"], na_position="first")
        .tail(1)
        .copy()
    )
    return selected.drop(columns=["__as_of_date"])


def _build_debt_position_cell(
    *,
    row: pd.Series,
    period: str,
    display_value: float,
    debt_position: pd.DataFrame,
    tolerance: float,
) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    measure = _norm(row.get("measure"))
    currency = _norm(row.get("Currency"))
    pair = _norm(row.get("pair"))
    debtor, creditor = _pair_parts(pair)
    component = DEBT_COMPONENT_FOR_MEASURE.get(measure, "")

    if debt_position.empty:
        return (
            STATUS_ERROR,
            0.0,
            -display_value,
            "missing_source",
            "monthly_debt_position.csv",
            {"error": "missing monthly_debt_position.csv", "period": period, "Currency": currency, "pair": pair, "measure": measure},
            "Debt position drilldown requires monthly_debt_position.csv.",
            pd.DataFrame(),
            [],
        )

    if measure not in debt_position.columns:
        return (
            STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "unsupported",
            "monthly_debt_position.csv",
            {
                "unsupported": True,
                "reason": "measure not present in monthly_debt_position.csv",
                "period": period,
                "Currency": currency,
                "pair": pair,
                "measure": measure,
                "available_columns": list(debt_position.columns),
            },
            "Debt position measure has no source column.",
            pd.DataFrame(),
            [],
        )

    mask = (
        _period_eq(debt_position, period)
        & _source_filter_eq(debt_position, "Currency", currency)
        & _source_filter_eq(debt_position, "debtor", debtor)
        & _source_filter_eq(debt_position, "creditor", creditor)
    )

    if component and "component" in debt_position.columns:
        mask &= _source_filter_eq(debt_position, "component", component)

    candidates = debt_position.loc[mask].copy()
    source = _select_monthly_debt_position_snapshot(candidates)
    matched = _num(source.iloc[0].get(measure)) if not source.empty else 0.0
    residual = matched - display_value

    status = (
        STATUS_EMPTY
        if source.empty
        else STATUS_OK
        if abs(residual) <= tolerance
        else STATUS_RESIDUAL_WARNING
    )

    filters = {
        "period": period,
        "Currency": currency,
        "pair": pair,
        "debtor": debtor,
        "creditor": creditor,
        "measure": measure,
        "component": component,
        "source": "monthly_debt_position.csv",
    }

    sections = [
        ("Selected monthly close snapshot", source),
        ("All candidate snapshots in period", candidates),
    ]

    return (
        status,
        matched,
        residual,
        "monthly_debt_position",
        "monthly_debt_position.csv",
        filters,
        "Debt position is a stock balance drilldown, not a semantic flow drilldown.",
        source,
        sections,
    )




def _filter_optional(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    if col not in df.columns or _norm(value) == "":
        return pd.Series(True, index=df.index)
    return df[col].astype(str).fillna("").str.strip().eq(_norm(value))


def _latest_period_rows(df: pd.DataFrame, period_col: str = "period") -> pd.DataFrame:
    if df.empty or period_col not in df.columns:
        return df.copy()
    periods = sorted(df[period_col].astype(str).dropna().unique().tolist())
    if not periods:
        return df.iloc[0:0].copy()
    return df[df[period_col].astype(str).eq(periods[-1])].copy()


def _annual_companion_long_row(row: pd.Series, period: str, display_value: float) -> pd.DataFrame:
    out = row.to_frame().T.copy()
    out["period"] = period
    out["value"] = display_value
    return out


def _build_annual_cash_close_companion_cell(*, row: pd.Series, period: str, display_value: float, cash_close: pd.DataFrame, tolerance: float):
    currency = _norm(row.get("Currency"))
    box = _norm(row.get("Box"))
    if cash_close.empty:
        return STATUS_ERROR, 0.0, -display_value, "missing_source", "monthly_cash_close.csv", {"error": "missing monthly_cash_close.csv", "period": period, "Currency": currency, "Box": box}, "Annual cash close drilldown requires monthly_cash_close.csv.", pd.DataFrame(), []
    value_col = "close_amount" if "close_amount" in cash_close.columns else ("value" if "value" in cash_close.columns else "balance")
    if value_col not in cash_close.columns:
        return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "monthly_cash_close.csv", {"reason": "no cash close value column", "available_columns": list(cash_close.columns)}, "Cash close source has no supported value column.", pd.DataFrame(), []
    candidates = cash_close.loc[_year_mask(cash_close, period) & _filter_optional(cash_close, "Currency", currency) & _filter_optional(cash_close, "Box", box)].copy()
    source = _latest_period_rows(candidates)
    matched = _measure_sum(source, value_col)
    residual = matched - display_value
    status = STATUS_EMPTY if source.empty else STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
    filters = {"period": period, "Currency": currency, "Box": box, "measure": "cash_close", "source_measure": value_col, "calculation_rule": "annual stock = latest monthly cash close in year; not sum of monthly closes"}
    sections = [("Annual companion row", _annual_companion_long_row(row, period, display_value)), ("Selected monthly_cash_close rows", source), ("Candidate monthly_cash_close rows in year", candidates)]
    return status, matched, residual, "annual_cash_close_to_monthly_cash_close", "monthly_cash_close.csv", filters, "Cash close is stock lineage; selected latest month in year, not annual sum.", source, sections


def _build_annual_funding_companion_cell(*, row: pd.Series, period: str, display_value: float, split: pd.DataFrame, audit: pd.DataFrame, tolerance: float):
    currency = _norm(row.get("Currency"))
    if split.empty:
        return STATUS_ERROR, 0.0, -display_value, "missing_source", "monthly_flow_semantic_split.csv", {"error": "missing monthly_flow_semantic_split.csv", "period": period, "Currency": currency}, "Annual funding drilldown requires monthly_flow_semantic_split.csv.", pd.DataFrame(), []
    source = split.loc[_year_mask(split, period) & _filter_optional(split, "Currency", currency)].copy()
    for col in ["funding_actor", "funding_channel", "cash_effect", "target_box", "beneficiary_box", "obligation_box"]:
        source = source.loc[_filter_optional(source, col, row.get(col))].copy()
    direct = _norm(row.get("funding_channel")).startswith("tenant_direct") or _norm(row.get("cash_effect")) == "no_cash_in_box_direct_payment"
    if direct:
        value_col = "amount_abs" if "amount_abs" in source.columns else "net_amount"
        matched = _measure_sum(source, value_col) if value_col == "amount_abs" else float(source.get("net_amount", pd.Series(dtype=float)).abs().sum())
    else:
        value_col = "amount_in" if "amount_in" in source.columns else "net_amount"
        matched = _measure_sum(source, value_col)
    residual = matched - display_value
    status = STATUS_EMPTY if source.empty else STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
    audit_rows = audit.loc[_year_mask(audit, period) & _filter_optional(audit, "Currency", currency)].copy() if not audit.empty else pd.DataFrame()
    filters = {"period": period, "Currency": currency, "funding_actor": _norm(row.get("funding_actor")), "funding_channel": _norm(row.get("funding_channel")), "cash_effect": _norm(row.get("cash_effect")), "value_col": value_col, "calculation_rule": "annual flow/support = sum source funding rows by year and explicit funding dimensions"}
    sections = [("Annual companion row", _annual_companion_long_row(row, period, display_value)), ("Matched monthly_flow_semantic_split rows", source)]
    if not audit_rows.empty:
        sections.append(("Classification audit rows for year/currency", audit_rows))
    return status, matched, residual, "annual_funding_to_monthly_flow_semantic_split", "monthly_flow_semantic_split.csv", filters, "Funding is flow/support lineage; direct obligations are not PM/FB cash inflow.", source, sections


def _build_annual_debt_stock_companion_cell(*, row: pd.Series, period: str, display_value: float, debt_position: pd.DataFrame, tolerance: float):
    currency = _norm(row.get("Currency")); pair = _norm(row.get("pair")); debtor = _norm(row.get("debtor")); creditor = _norm(row.get("creditor")); component = _norm(row.get("component"))
    measure = component if component in {"open_principal", "open_interest", "open_total"} else "open_total"
    if debt_position.empty:
        return STATUS_ERROR, 0.0, -display_value, "missing_source", "monthly_debt_position.csv", {"error": "missing monthly_debt_position.csv", "period": period, "pair": pair}, "Annual debt stock drilldown requires monthly_debt_position.csv.", pd.DataFrame(), []
    candidates = debt_position.loc[_year_mask(debt_position, period) & _filter_optional(debt_position, "Currency", currency) & _filter_optional(debt_position, "debtor", debtor) & _filter_optional(debt_position, "creditor", creditor)].copy()
    if "pair" in candidates.columns and pair:
        candidates = candidates[candidates["pair"].astype(str).str.strip().eq(pair)].copy()
    month_rows = _latest_period_rows(candidates)
    source = _select_monthly_debt_position_snapshot(month_rows)
    matched = _num(source.iloc[0].get(measure)) if not source.empty and measure in source.columns else 0.0
    residual = matched - display_value
    status = STATUS_EMPTY if source.empty else STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
    filters = {"period": period, "Currency": currency, "pair": pair, "debtor": debtor, "creditor": creditor, "component": component, "measure": measure, "calculation_rule": "annual stock = latest selected monthly close in year; latest as_of_date within selected month; not a sum"}
    sections = [("Annual companion row", _annual_companion_long_row(row, period, display_value)), ("Selected annual close row", source), ("Candidate debt position rows in year", candidates)]
    return status, matched, residual, "annual_debt_stock_to_monthly_debt_position", "monthly_debt_position.csv", filters, "Debt stock is stock lineage; selected close snapshot, not annual flow.", source, sections


def _build_annual_debt_activity_companion_cell(*, row: pd.Series, period: str, display_value: float, debt_activity: pd.DataFrame, tolerance: float):
    currency = _norm(row.get("Currency")); pair = _norm(row.get("pair")); debtor = _norm(row.get("debtor")); creditor = _norm(row.get("creditor")); activity_type = _norm(row.get("activity_type"))
    measure = {"settlements": "settlements", "repayments": "repayments", "new_principal": "new_principal", "net_change": "net_change"}.get(activity_type, activity_type)
    if measure == "settlements" and "settlements" not in debt_activity.columns:
        measure = "repayments"
    if debt_activity.empty:
        return STATUS_ERROR, 0.0, -display_value, "missing_source", "monthly_debt_activity.csv", {"error": "missing monthly_debt_activity.csv", "period": period, "pair": pair}, "Annual debt activity drilldown requires monthly_debt_activity.csv.", pd.DataFrame(), []
    if measure not in debt_activity.columns:
        return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "monthly_debt_activity.csv", {"reason": "activity measure missing", "measure": measure, "available_columns": list(debt_activity.columns)}, "Debt activity source has no matching measure column.", pd.DataFrame(), []
    source = debt_activity.loc[_year_mask(debt_activity, period) & _filter_optional(debt_activity, "Currency", currency) & _filter_optional(debt_activity, "debtor", debtor) & _filter_optional(debt_activity, "creditor", creditor)].copy()
    if "pair" in source.columns and pair:
        source = source[source["pair"].astype(str).str.strip().eq(pair)].copy()
    matched = _measure_sum(source, measure)
    residual = matched - display_value
    status = STATUS_EMPTY if source.empty else STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
    filters = {"period": period, "Currency": currency, "pair": pair, "debtor": debtor, "creditor": creditor, "activity_type": activity_type, "measure": measure, "calculation_rule": "annual flow = sum monthly debt activity by year/Currency/pair/activity_type"}
    sections = [("Annual companion row", _annual_companion_long_row(row, period, display_value)), ("Matched monthly_debt_activity rows", source)]
    return status, matched, residual, "annual_debt_activity_to_monthly_debt_activity", "monthly_debt_activity.csv", filters, "Debt activity is flow lineage; repayments/settlements/increases/net movements are annual sums.", source, sections

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
    cash_close: pd.DataFrame,
    debt_activity: pd.DataFrame,
    debt_position: pd.DataFrame,
    tolerance: float,
) -> tuple[str, float, float, str, str, dict[str, Any], str, pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    currency = _norm(row.get("Currency"))
    if not currency and table_id != "overview_balance_dashboard":
        return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "", {"unsupported": True, "reason": "missing Currency would risk cross-currency aggregation"}, "", pd.DataFrame(), []

    line = _statement_line(row)
    sections: list[tuple[str, pd.DataFrame]] = []
    caveat = "Derived drilldown: explanation page, not necessarily raw ledger rows."

    if table_id == "annual_cash_close_by_box_wide":
        return _build_annual_cash_close_companion_cell(row=row, period=period, display_value=display_value, cash_close=cash_close, tolerance=tolerance)
    if table_id == "annual_funding_by_actor_channel_wide":
        return _build_annual_funding_companion_cell(row=row, period=period, display_value=display_value, split=split, audit=audit, tolerance=tolerance)
    if table_id == "annual_debt_stock_by_pair_wide":
        return _build_annual_debt_stock_companion_cell(row=row, period=period, display_value=display_value, debt_position=debt_position, tolerance=tolerance)
    if table_id == "annual_debt_activity_by_pair_wide":
        return _build_annual_debt_activity_companion_cell(row=row, period=period, display_value=display_value, debt_activity=debt_activity, tolerance=tolerance)


    # if table_id == "monthly_tables_cash_close_matrix":
    #     return _build_cash_close_cell(
    #         row=row,
    #         period=period,
    #         display_value=display_value,
    #         cash_close=cash_close,
    #         tolerance=tolerance,
    #     )

    if table_id == "monthly_tables_debt_activity_matrix":
        return _build_debt_activity_cell(
            row=row,
            period=period,
            display_value=display_value,
            debt_activity=debt_activity,
            tolerance=tolerance,
        )

    # if table_id == "monthly_tables_debt_position_matrix":
    #     return _build_debt_position_cell(
    #         row=row,
    #         period=period,
    #         display_value=display_value,
    #         debt_position=debt_position,
    #         tolerance=tolerance,
    #     )



    if table_id in {
        "monthly_tables_cash_close_matrix",
        "monthly_tables_diagnostic_box_level_matrix",
    }:
        default_metric = (
            "diagnostic_box_level"
            if table_id == "monthly_tables_diagnostic_box_level_matrix"
            else "cash_close"
        )

        return _build_cash_control_cell(
            row=row,
            period=period,
            display_value=display_value,
            source_df=cash_close,
            source_name="monthly_cash_close.csv",
            default_metric=default_metric,
            tolerance=tolerance,
        )


    if table_id == "monthly_tables_debt_position_matrix":
        return _build_debt_position_cell(
            row=row,
            period=period,
            display_value=display_value,
            debt_position=debt_position,
            tolerance=tolerance,
        )


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
        formula_cell = _build_annual_formula_cell(
            table_id=table_id,
            row=row,
            period=period,
            currency=currency,
            display_value=display_value,
            annual=annual,
            tolerance=tolerance,
        )
        if formula_cell is not None:
            return formula_cell

        annual_rows = _annual_source_rows(annual, row, period)

        if annual.empty or annual_rows.empty:
            fallback = _build_annual_professional_fallback(
                table_id=table_id,
                row=row,
                period=period,
                currency=currency,
                display_value=display_value,
                split=split,
                audit=audit,
                tolerance=tolerance,
            )
            if fallback is not None:
                return fallback

            return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "annual_balance_dashboard_metrics.csv", {"unsupported": True, "reason": "no matching annual metric row", "period": period}, "Annual source row unavailable.", pd.DataFrame(), []
        
        source_table = _norm(annual_rows.iloc[0].get("source_table"))
        flow_type = _norm(annual_rows.iloc[0].get("flow_type"))
        calc_rule = _norm(annual_rows.iloc[0].get("calculation_rule"))
        if source_table == "monthly_debt_position.csv":
            matched = _measure_sum(annual_rows, "value")
            metric_id = _norm(annual_rows.iloc[0].get("metric_id"))
            debt_rows = debt_position.loc[_year_mask(debt_position, period) & _eq_col(debt_position, "Currency", currency)].copy() if not debt_position.empty else pd.DataFrame()
            component = ""
            if metric_id in {"ID.DEBT.TOTAL.OPEN", "BS.DEBT.TOTAL.OPEN"}:
                component = "total"
            elif metric_id in {"ID.DEBT.PRINCIPAL.OPEN", "BS.DEBT.PRINCIPAL.OPEN"}:
                component = "principal"
            elif metric_id in {"ID.DEBT.INTEREST.OPEN", "BS.DEBT.INTEREST.OPEN"}:
                component = "interest"
            if component and not debt_rows.empty and "component" in debt_rows.columns:
                debt_rows = debt_rows.loc[_source_filter_eq(debt_rows, "component", component)].copy()
            residual = matched - display_value
            status = STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
            sections = [("Annual metric row", annual_rows), ("Debt position rows", debt_rows)]
            filters = {"period": period, "Currency": currency, "metric_id": metric_id, "source_table": source_table, "component": component, "row_context": _row_context(table_id, row)}
            return status, matched, residual, "annual_to_monthly_debt_position", "annual_balance_dashboard_metrics.csv", filters, "Debt stock lineage uses monthly_debt_position.csv, not flow split rows.", debt_rows if not debt_rows.empty else annual_rows, sections

        if flow_type == "stock" or source_table == "monthly_cash_close.csv":
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
            elif metric_id in {"FUND.CONTRIB.TOTAL", "FUND.CONTRIB.BY_ACTOR"}:
                sem_mask &= _bucket_eq(split, "funding_contribution")
            elif metric_id.startswith("FUND.CONTRIB."):
                sem_mask &= _funding_metric_semantic_mask(split, metric_id)
            elif metric_id in {"DIST.DRAWS.PERSONAL", "DIST.DRAWS.BY_TYPE", "DIST.DIVIDENDS"}:
                sem_mask &= _bucket_eq(split, "family_withdrawal_candidate")
                if metric_id == "DIST.DIVIDENDS" and "semantic_subbucket" in split.columns:
                    sem_mask &= _regex_any(split["semantic_subbucket"], r"dividend|dividendo")
            if dim_name and dim_value and dim_name in split.columns:
                sem_mask &= _eq_col(split, dim_name, dim_value)
            semantic_rows = split.loc[sem_mask].copy()
            sections.append(("Semantic rows", semantic_rows))
            detail_rows, lineage = _detail_from_audit(audit, semantic_rows, lambda df, p=period: _year_mask(df, p) & _eq_col(df, "Currency", currency))
            sections.append(("Classification rows", detail_rows))
            if metric_id == "FUND.CONTRIB.DEBT_LINKED":
                debt_activity_rows = debt_activity.loc[_year_mask(debt_activity, period) & _eq_col(debt_activity, "Currency", currency)].copy() if not debt_activity.empty else pd.DataFrame()
                debt_position_rows = debt_position.loc[_year_mask(debt_position, period) & _eq_col(debt_position, "Currency", currency)].copy() if not debt_position.empty else pd.DataFrame()
                sections.append(("Debt activity rows", debt_activity_rows))
                sections.append(("Debt position rows", debt_position_rows))
                if not debt_activity_rows.empty or not debt_position_rows.empty:
                    lineage = "debt_linked_support_with_debt_evidence"
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
        


        semantic_rows, measure, filter_note, line_filter, is_supported_line = _cash_bridge_semantic_rows(
            split,
            row,
            period,
        )


        if (not measure) or (not is_supported_line):
            return (
                STATUS_UNSUPPORTED,
                0.0,
                -display_value,
                "unsupported",
                "monthly_flow_semantic_split.csv",
                {
                    "unsupported": True,
                    "reason": filter_note,
                    "year": period,
                    "Currency": currency,
                    "row_context": _row_context(table_id, row),
                },
                "Cash bridge line has no explicit semantic mapping; no fallback drilldown was generated.",
                pd.DataFrame(),
                [],
            )


        matched = _measure_sum(semantic_rows, measure)
        residual = matched - display_value

        def audit_filter(df: pd.DataFrame, p: str = period) -> pd.Series:
            mask = _year_mask(df, p) & _eq_col(df, "Currency", currency)

            if "Box" in row.index and _norm(row.get("Box")):
                mask &= _eq_col(df, "Box", row.get("Box"))

            if line_filter is not None:
                mask &= line_filter(df)

            return mask


        detail_rows, lineage = _detail_from_audit(audit, semantic_rows, audit_filter)

        sections = [
            ("Flow bridge semantic rows", semantic_rows),
            ("Classification rows", detail_rows),
        ]

        filters = {
            "year": period,
            "Currency": currency,
            "measure": measure,
            "filter_note": filter_note,
            "row_context": _row_context(table_id, row),
        }

        status = STATUS_EMPTY if semantic_rows.empty else (
            STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
        )

        
        # return status, matched, residual, lineage, "monthly_flow_semantic_split.csv", filters, caveat, detail_rows if not detail_rows.empty else semantic_rows, sections
        return (
            status,
            matched,
            residual,
            lineage,
            "monthly_flow_semantic_split.csv",
            filters,
            caveat,
            detail_rows if not detail_rows.empty else semantic_rows,
            sections,
        )


    return STATUS_UNSUPPORTED, 0.0, -display_value, "unsupported", "", {"unsupported": True}, "", pd.DataFrame(), []


import time


def build_professional_flow_drilldowns(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None = None,
    tables_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    fast: bool = False,
) -> dict[str, Path]:
    t0 = time.perf_counter()

    repo_root = Path(repo_root)
    pack_dir = Path(pack_dir)
    tables_dir = Path(tables_dir) if tables_dir is not None else pack_dir / "tables"
    drill_dir = pack_dir / "drilldown"
    details_dir = drill_dir / DETAILS_DIRNAME

    LOG.info(
        "[drilldown] start build_professional_flow_drilldowns "
        "repo_root=%s pack_dir=%s run_root=%s tables_dir=%s tolerance=%s fast=%s",
        repo_root,
        pack_dir,
        run_root,
        tables_dir,
        tolerance,
        fast,
    )

    details_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("[drilldown] details_dir ready: %s", details_dir)
    enriched_tables = enrich_professional_table_contracts(tables_dir)
    if enriched_tables:
        LOG.info("[drilldown] enriched professional table contract columns files=%s", len(enriched_tables))

    LOG.info("[drilldown] locating source artifacts")
    split_path = _find_source(repo_root, pack_dir, run_root, "monthly_flow_semantic_split.csv")
    audit_path = _find_source(repo_root, pack_dir, run_root, "classification_audit.csv")
    stmt_path = _find_source(repo_root, pack_dir, run_root, "monthly_operating_statement.csv")
    annual_path = _find_source(repo_root, pack_dir, run_root, "annual_balance_dashboard_metrics.csv")

    cash_close_path = _find_source(repo_root, pack_dir, run_root, "monthly_cash_close.csv")
    debt_activity_path = _find_source(repo_root, pack_dir, run_root, "monthly_debt_activity.csv")
    debt_position_path = _find_source(repo_root, pack_dir, run_root, "monthly_debt_position.csv")

    LOG.info(
        "[drilldown] source paths found: split=%s audit=%s stmt=%s annual=%s "
        "cash_close=%s debt_activity=%s debt_position=%s",
        split_path,
        audit_path,
        stmt_path,
        annual_path,
        cash_close_path,
        debt_activity_path,
        debt_position_path,
    )

    LOG.info("[drilldown] reading source artifacts")

    read_t0 = time.perf_counter()
    split = _read_csv(split_path) if split_path else pd.DataFrame()
    LOG.info(
        "[drilldown] loaded split rows=%s cols=%s path=%s elapsed=%.2fs",
        len(split),
        len(split.columns),
        split_path,
        time.perf_counter() - read_t0,
    )

    read_t0 = time.perf_counter()
    audit = _read_csv(audit_path) if audit_path else pd.DataFrame()
    LOG.info(
        "[drilldown] loaded audit rows=%s cols=%s path=%s elapsed=%.2fs",
        len(audit),
        len(audit.columns),
        audit_path,
        time.perf_counter() - read_t0,
    )

    read_t0 = time.perf_counter()
    stmt = _read_csv(stmt_path) if stmt_path else pd.DataFrame()
    LOG.info(
        "[drilldown] loaded stmt rows=%s cols=%s path=%s elapsed=%.2fs",
        len(stmt),
        len(stmt.columns),
        stmt_path,
        time.perf_counter() - read_t0,
    )

    read_t0 = time.perf_counter()
    annual = _read_csv(annual_path) if annual_path else pd.DataFrame()
    LOG.info(
        "[drilldown] loaded annual rows=%s cols=%s path=%s elapsed=%.2fs",
        len(annual),
        len(annual.columns),
        annual_path,
        time.perf_counter() - read_t0,
    )


    cash_close = _read_csv(cash_close_path) if cash_close_path else pd.DataFrame()
    # LOG.info(


    debt_activity = _read_csv(debt_activity_path) if debt_activity_path else pd.DataFrame()
    
    
    debt_position = _read_csv(debt_position_path) if debt_position_path else pd.DataFrame()




    LOG.info("[drilldown] coercing numeric columns")
    for name, df in [
        ("split", split),
        ("audit", audit),
        ("stmt", stmt),
        ("annual", annual),
        ("cash_close", cash_close),
        ("debt_activity", debt_activity),
        ("debt_position", debt_position),
    ]:
        if df.empty:
            LOG.info("[drilldown] numeric coercion skipped: %s is empty", name)
            continue

        coerced_cols = []
        for col in [
            "amount_in", "amount_out", "net_amount", "amount_abs", "amount", "value",
            "cash_close", "closing_cash", "closing_balance", "balance", "diagnostic_box_level",
            "new_principal", "interest_accrued", "repayments", "adjustments",
            "opening_total", "closing_total", "net_change",
            "open_amount", "open_principal", "open_interest", "open_total",
            ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                coerced_cols.append(col)

        LOG.info(
            "[drilldown] numeric coercion done: source=%s rows=%s coerced_cols=%s",
            name,
            len(df),
            coerced_cols,
        )

    scope = list(SUPPORTED_TABLE_IDS)
    index_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    table_cell_limit = FAST_TABLE_CELL_LIMIT if fast else MAX_TABLE_CELL_LIMIT
    LOG.info(
        "[drilldown] processing %s supported tables fast=%s table_cell_limit=%s",
        len(scope),
        fast,
        table_cell_limit,
    )

    overall_cell_i = 0
    overall_table_i = 0
    progress_every_cells = 100
    progress_every_rows = 25

    for table_id in scope:
        overall_table_i += 1
        table_t0 = time.perf_counter()
        table_path = tables_dir / f"{table_id}.csv"

        LOG.info(
            "[drilldown] table start %s/%s table_id=%s path=%s",
            overall_table_i,
            len(scope),
            table_id,
            table_path,
        )

        if not table_path.exists():
            LOG.warning("[drilldown] missing table: table_id=%s path=%s", table_id, table_path)
            qa_rows.append(
                {
                    "table_id": table_id,
                    "check": "table_exists",
                    "status": "warning",
                    "detail": f"Missing table: {table_path}",
                }
            )
            continue

        read_t0 = time.perf_counter()
        table = pd.read_csv(table_path)
        LOG.info(
            "[drilldown] table loaded table_id=%s rows=%s cols=%s elapsed=%.2fs",
            table_id,
            len(table),
            len(table.columns),
            time.perf_counter() - read_t0,
        )

        months = _period_columns(table_id, table)
        if not months:
            LOG.warning(
                "[drilldown] no month columns detected table_id=%s rows=%s cols=%s",
                table_id,
                len(table),
                len(table.columns),
            )
            qa_rows.append(
                {
                    "table_id": table_id,
                    "check": "month_columns",
                    "status": "warning",
                    "detail": "No YYYY-MM columns detected",
                }
            )
            continue

        table_total_cells = len(table) * len(months)
        table_cell_i = 0
        table_status_counts: dict[str, int] = {}

        LOG.info(
            "[drilldown] table periods detected table_id=%s months=%s first_month=%s last_month=%s total_cells=%s",
            table_id,
            len(months),
            months[0],
            months[-1],
            table_total_cells,
        )

        if table_total_cells > table_cell_limit:
            detail = (
                f"{TABLE_TOO_LARGE_WARNING} table_id={table_id}; "
                f"cells={table_total_cells}; rows={len(table)}; periods={len(months)}; "
                f"limit={table_cell_limit}; fast={fast}"
            )
            LOG.warning("[drilldown] %s", detail)
            qa_rows.append(
                {
                    "table_id": table_id,
                    "drilldown_id": "",
                    "check": "table_cell_limit",
                    "status": "warning",
                    "detail": detail,
                }
            )
            table_status_counts["skipped_table_too_large"] = 1
            LOG.info(
                "[drilldown] table skipped table_id=%s cells=%s limit=%s fast=%s elapsed=%.2fs",
                table_id,
                table_total_cells,
                table_cell_limit,
                fast,
                time.perf_counter() - table_t0,
            )
            continue

        for row_pos, (row_idx, row) in enumerate(table.iterrows(), start=1):
            row_t0 = time.perf_counter()

            spec = _spec_for_cell(table_id, row)
            context = _row_context(table_id, row)
            row_id = row_context_id(table_id, int(row_idx), row)

            row_measure = spec.measure if spec else _norm(row.get("measure"))
            row_currency = _norm(row.get("Currency"))
            row_label = (
                _norm(row.get("label"))
                or _norm(row.get("row_label"))
                or _norm(row.get("account"))
                or _norm(row.get("category"))
                or _norm(row.get("subcategory"))
                or row_id
            )

            if (
                row_pos == 1
                or row_pos % progress_every_rows == 0
                or row_pos == len(table)
            ):
                LOG.info(
                    "[drilldown] row progress table_id=%s row=%s/%s row_idx=%s row_id=%s "
                    "currency=%s measure=%s label=%s elapsed_table=%.2fs",
                    table_id,
                    row_pos,
                    len(table),
                    row_idx,
                    row_id,
                    row_currency,
                    row_measure,
                    row_label,
                    time.perf_counter() - table_t0,
                )

            for period in months:
                table_cell_i += 1
                overall_cell_i += 1
                cell_t0 = time.perf_counter()

                display_value = _num(row.get(period))
                measure = spec.measure if spec else _norm(row.get("measure"))
                missing_currency = _norm(row.get("Currency")) == ""
                drilldown_id = _safe_id(table_id, row_idx, period, row.get("Currency"), measure)
                detail_csv_rel = f"drilldown/{DETAILS_DIRNAME}/{drilldown_id}.csv"
                detail_html_rel = f"drilldown/{DETAILS_DIRNAME}/{drilldown_id}.html"

                if (
                    table_cell_i == 1
                    or table_cell_i % progress_every_cells == 0
                    or table_cell_i == table_total_cells
                ):
                    LOG.info(
                        "[drilldown] cell progress table_id=%s cell=%s/%s overall_cell=%s "
                        "row_idx=%s period=%s currency=%s measure=%s display_value=%s "
                        "elapsed_table=%.2fs",
                        table_id,
                        table_cell_i,
                        table_total_cells,
                        overall_cell_i,
                        row_idx,
                        period,
                        _norm(row.get("Currency")),
                        measure,
                        display_value,
                        time.perf_counter() - table_t0,
                    )

                LOG.debug(
                    "[drilldown] cell start table_id=%s drilldown_id=%s row_idx=%s period=%s "
                    "currency=%s measure=%s display_value=%s derived=%s",
                    table_id,
                    drilldown_id,
                    row_idx,
                    period,
                    _norm(row.get("Currency")),
                    measure,
                    display_value,
                    table_id in DERIVED_TABLE_IDS,
                )

                base = {
                    "drilldown_id": drilldown_id,
                    "table_id": table_id,
                    "row_id": row_id,
                    "period": period,
                    "Currency": _norm(row.get("Currency")),
                    "measure": measure,
                    "source_artifact": "monthly_flow_semantic_split.csv",
                    "detail_csv_relpath": detail_csv_rel,
                    "detail_html_relpath": detail_html_rel,
                    "display_value": display_value,
                    "filter_json": "{}",
                    "row_context_json": json.dumps(
                        context,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    "lineage_level": "",
                    "caveat": "",
                }



                if abs(display_value) <= tolerance:
                    table_status_counts["skipped_zero"] = table_status_counts.get("skipped_zero", 0) + 1

                    LOG.debug(
                        "[drilldown] zero cell skipped without index row table_id=%s "
                        "drilldown_id=%s row_idx=%s period=%s currency=%s measure=%s",
                        table_id,
                        drilldown_id,
                        row_idx,
                        period,
                        _norm(row.get("Currency")),
                        measure,
                    )

                    continue


                try:
                    if table_id in DERIVED_TABLE_IDS:
                        LOG.debug(
                            "[drilldown] derived build start table_id=%s drilldown_id=%s period=%s display_value=%s",
                            table_id,
                            drilldown_id,
                            period,
                            display_value,
                        )

                        derived_t0 = time.perf_counter()
                        (
                            status,
                            matched,
                            residual,
                            lineage,
                            source_artifact,
                            filters,
                            caveat,
                            detail_df,
                            sections,
                        ) = _build_derived_cell(
                            table_id=table_id,
                            row=row,
                            period=period,
                            display_value=display_value,
                            split=split,
                            audit=audit,
                            stmt=stmt,
                            annual=annual,
                            cash_close=cash_close,
                            debt_activity=debt_activity,
                            debt_position=debt_position,
                            tolerance=tolerance,
                        )

                        LOG.debug(
                            "[drilldown] derived build done table_id=%s drilldown_id=%s "
                            "status=%s matched=%s residual=%s rows=%s lineage=%s elapsed=%.2fs",
                            table_id,
                            drilldown_id,
                            status,
                            matched,
                            residual,
                            len(detail_df),
                            lineage,
                            time.perf_counter() - derived_t0,
                        )

                        derived_measure = _norm(filters.get("measure")) or measure

                        out_row = {
                            **base,
                            "measure": derived_measure,
                            "source_artifact": source_artifact,
                            "matched_rows": int(len(detail_df)),
                            "matched_value_sum": matched,
                            "residual": residual,
                            "status": status,
                            "filter_json": json.dumps(
                                filters,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            "lineage_level": lineage,
                            "caveat": caveat,
                        }

                        write_t0 = time.perf_counter()
                        (pack_dir / detail_csv_rel).parent.mkdir(parents=True, exist_ok=True)
                        detail_df.to_csv(pack_dir / detail_csv_rel, index=False)
                        _write_detail_html(pack_dir / detail_html_rel, out_row, detail_df, sections=sections)

                        LOG.debug(
                            "[drilldown] derived detail written table_id=%s drilldown_id=%s "
                            "csv=%s html=%s rows=%s elapsed=%.2fs",
                            table_id,
                            drilldown_id,
                            pack_dir / detail_csv_rel,
                            pack_dir / detail_html_rel,
                            len(detail_df),
                            time.perf_counter() - write_t0,
                        )

                        index_rows.append(out_row)

                        qa_status = (
                            "pass"
                            if status == STATUS_OK
                            else (
                                "warning"
                                if status in {STATUS_EMPTY, STATUS_RESIDUAL_WARNING, STATUS_UNSUPPORTED}
                                else "fail"
                            )
                        )

                        qa_rows.append(
                            {
                                "table_id": table_id,
                                "drilldown_id": drilldown_id,
                                "check": "cell_reconciliation",
                                "status": qa_status,
                                "detail": f"status={status}; residual={residual}; matched_rows={len(detail_df)}",
                            }
                        )

                        table_status_counts[status] = table_status_counts.get(status, 0) + 1

                        LOG.debug(
                            "[drilldown] cell done table_id=%s drilldown_id=%s status=%s "
                            "qa_status=%s elapsed=%.2fs",
                            table_id,
                            drilldown_id,
                            status,
                            qa_status,
                            time.perf_counter() - cell_t0,
                        )

                        continue

                    if split.empty:
                        LOG.warning(
                            "[drilldown] direct cell has missing split source table_id=%s drilldown_id=%s",
                            table_id,
                            drilldown_id,
                        )

                        status = STATUS_ERROR
                        semantic_subset = pd.DataFrame()
                        matched = 0.0
                        residual = -display_value
                        lineage = "missing_source"
                        filters = {"error": "missing monthly_flow_semantic_split.csv"}
                        caveat = ""

                    elif missing_currency:
                        LOG.debug(
                            "[drilldown] unsupported cell due to missing currency table_id=%s drilldown_id=%s measure=%s",
                            table_id,
                            drilldown_id,
                            measure,
                        )

                        status = STATUS_UNSUPPORTED
                        semantic_subset = pd.DataFrame()
                        matched = 0.0
                        residual = -display_value
                        lineage = "unsupported"
                        filters = {
                            "unsupported": True,
                            "reason": "missing Currency would risk cross-currency aggregation",
                            "measure": measure,
                        }
                        caveat = ""

                    elif spec is None or spec.unsupported_if(row) or measure not in split.columns:
                        LOG.debug(
                            "[drilldown] unsupported cell due to spec/measure table_id=%s drilldown_id=%s "
                            "spec_is_none=%s measure=%s measure_in_split=%s",
                            table_id,
                            drilldown_id,
                            spec is None,
                            measure,
                            measure in split.columns,
                        )

                        status = STATUS_UNSUPPORTED
                        semantic_subset = pd.DataFrame()
                        matched = 0.0
                        residual = -display_value
                        lineage = "unsupported"
                        filters = {"unsupported": True, "measure": measure}
                        caveat = ""

                    else:
                        LOG.debug(
                            "[drilldown] semantic filter start table_id=%s drilldown_id=%s period=%s measure=%s",
                            table_id,
                            drilldown_id,
                            period,
                            spec.measure,
                        )

                        filter_t0 = time.perf_counter()
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

                        filters = {
                            "period": period,
                            "Currency": _norm(row.get("Currency")),
                            "measure": spec.measure,
                            "row_context": context,
                        }
                        caveat = spec.caveat_func(row)

                        LOG.debug(
                            "[drilldown] semantic filter done table_id=%s drilldown_id=%s "
                            "semantic_rows=%s matched=%s display_value=%s residual=%s status=%s elapsed=%.2fs",
                            table_id,
                            drilldown_id,
                            len(semantic_subset),
                            matched,
                            display_value,
                            residual,
                            status,
                            time.perf_counter() - filter_t0,
                        )

                        LOG.debug(
                            "[drilldown] audit detail expansion start table_id=%s drilldown_id=%s audit_rows=%s",
                            table_id,
                            drilldown_id,
                            len(audit),
                        )

                        audit_t0 = time.perf_counter()
                        detail_df, lineage = _detail_from_audit(
                            audit,
                            semantic_subset,
                            lambda df, r=row, p=period, s=spec: (
                                df.get("period", pd.Series("", index=df.index)).astype(str).eq(p)
                                & s.filter_func(df, r)
                            ),
                        )

                        LOG.debug(
                            "[drilldown] audit detail expansion done table_id=%s drilldown_id=%s "
                            "detail_rows=%s lineage=%s elapsed=%.2fs",
                            table_id,
                            drilldown_id,
                            len(detail_df),
                            lineage,
                            time.perf_counter() - audit_t0,
                        )

                        if detail_df.empty and lineage != "semantic_only":
                            LOG.debug(
                                "[drilldown] audit detail empty; falling back to semantic_subset "
                                "table_id=%s drilldown_id=%s semantic_rows=%s lineage=%s",
                                table_id,
                                drilldown_id,
                                len(semantic_subset),
                                lineage,
                            )
                            detail_df = semantic_subset

                    if "detail_df" not in locals() or (
                        split.empty or spec is None or (spec and spec.unsupported_if(row))
                    ):
                        detail_df = semantic_subset.copy() if "semantic_subset" in locals() else pd.DataFrame()

                    matched_rows = (
                        int(len(detail_df))
                        if not detail_df.empty
                        else int(len(semantic_subset))
                        if "semantic_subset" in locals()
                        else 0
                    )

                    out_row = {
                        **base,
                        "matched_rows": matched_rows,
                        "matched_value_sum": matched,
                        "residual": residual,
                        "status": status,
                        "filter_json": json.dumps(
                            filters,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                        "lineage_level": lineage,
                        "caveat": caveat,
                    }

                    write_t0 = time.perf_counter()
                    (pack_dir / detail_csv_rel).parent.mkdir(parents=True, exist_ok=True)
                    detail_df.to_csv(pack_dir / detail_csv_rel, index=False)
                    _write_detail_html(pack_dir / detail_html_rel, out_row, detail_df)

                    LOG.debug(
                        "[drilldown] detail written table_id=%s drilldown_id=%s csv=%s html=%s "
                        "rows=%s elapsed=%.2fs",
                        table_id,
                        drilldown_id,
                        pack_dir / detail_csv_rel,
                        pack_dir / detail_html_rel,
                        len(detail_df),
                        time.perf_counter() - write_t0,
                    )

                    index_rows.append(out_row)

                    qa_status = (
                        "pass"
                        if status == STATUS_OK
                        else (
                            "warning"
                            if status in {STATUS_EMPTY, STATUS_RESIDUAL_WARNING, STATUS_UNSUPPORTED}
                            else "fail"
                        )
                    )

                    qa_rows.append(
                        {
                            "table_id": table_id,
                            "drilldown_id": drilldown_id,
                            "check": "cell_reconciliation",
                            "status": qa_status,
                            "detail": f"status={status}; residual={residual}; matched_rows={matched_rows}",
                        }
                    )

                    table_status_counts[status] = table_status_counts.get(status, 0) + 1

                    LOG.debug(
                        "[drilldown] cell done table_id=%s drilldown_id=%s status=%s qa_status=%s "
                        "matched_rows=%s residual=%s elapsed=%.2fs",
                        table_id,
                        drilldown_id,
                        status,
                        qa_status,
                        matched_rows,
                        residual,
                        time.perf_counter() - cell_t0,
                    )

                except Exception:
                    LOG.exception(
                        "[drilldown] cell failed table_id=%s drilldown_id=%s row_idx=%s "
                        "period=%s currency=%s measure=%s display_value=%s "
                        "table_cell=%s/%s overall_cell=%s",
                        table_id,
                        drilldown_id,
                        row_idx,
                        period,
                        _norm(row.get("Currency")),
                        measure,
                        display_value,
                        table_cell_i,
                        table_total_cells,
                        overall_cell_i,
                    )
                    raise

                finally:
                    if "detail_df" in locals():
                        del detail_df

            LOG.debug(
                "[drilldown] row done table_id=%s row=%s/%s row_idx=%s row_id=%s elapsed=%.2fs",
                table_id,
                row_pos,
                len(table),
                row_idx,
                row_id,
                time.perf_counter() - row_t0,
            )

        LOG.info(
            "[drilldown] table done table_id=%s cells=%s index_rows_total=%s qa_rows_total=%s "
            "status_counts=%s elapsed=%.2fs",
            table_id,
            table_cell_i,
            len(index_rows),
            len(qa_rows),
            table_status_counts,
            time.perf_counter() - table_t0,
        )

    columns = [
        "drilldown_id",
        "table_id",
        "row_id",
        "period",
        "Currency",
        "measure",
        "source_artifact",
        "detail_csv_relpath",
        "detail_html_relpath",
        "matched_rows",
        "matched_value_sum",
        "display_value",
        "residual",
        "status",
        "filter_json",
        "row_context_json",
        "lineage_level",
        "caveat",
    ]

    LOG.info(
        "[drilldown] building final index/qa dataframes index_rows=%s qa_rows=%s",
        len(index_rows),
        len(qa_rows),
    )

    index = pd.DataFrame(index_rows, columns=columns)
    qa = pd.DataFrame(
        qa_rows,
        columns=["table_id", "drilldown_id", "check", "status", "detail"],
    )

    index_path = drill_dir / INDEX_FILENAME
    manifest_path = drill_dir / MANIFEST_FILENAME
    qa_path = drill_dir / QA_FILENAME

    LOG.info(
        "[drilldown] writing final artifacts index=%s qa=%s manifest=%s",
        index_path,
        qa_path,
        manifest_path,
    )

    write_t0 = time.perf_counter()
    index.to_csv(index_path, index=False)
    LOG.info(
        "[drilldown] wrote index rows=%s path=%s elapsed=%.2fs",
        len(index),
        index_path,
        time.perf_counter() - write_t0,
    )

    write_t0 = time.perf_counter()
    qa.to_csv(qa_path, index=False)
    LOG.info(
        "[drilldown] wrote qa rows=%s path=%s elapsed=%.2fs",
        len(qa),
        qa_path,
        time.perf_counter() - write_t0,
    )

    manifest = {
        "created_at_utc": _now_iso(),
        "repo_root": str(repo_root),
        "pack_dir": str(pack_dir),
        "tables_dir": str(tables_dir),
        "run_root": str(run_root or ""),
        "monthly_flow_semantic_split": str(split_path or ""),
        "classification_audit": str(audit_path or ""),
        "monthly_operating_statement": str(stmt_path or ""),
        "annual_balance_dashboard_metrics": str(annual_path or ""),
        "monthly_cash_close": str(cash_close_path or ""),
        "monthly_debt_activity": str(debt_activity_path or ""),
        "monthly_debt_position": str(debt_position_path or ""),
        "tolerance": tolerance,
        "fast": bool(fast),
        "table_cell_limit": int(table_cell_limit),
        "max_table_cell_limit": int(MAX_TABLE_CELL_LIMIT),
        "fast_table_cell_limit": int(FAST_TABLE_CELL_LIMIT),
        "index_rows": int(len(index)),
        "qa_rows": int(len(qa)),
        "status_counts": index["status"].value_counts().to_dict() if not index.empty else {},
    }

    write_t0 = time.perf_counter()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    LOG.info(
        "[drilldown] wrote manifest path=%s elapsed=%.2fs status_counts=%s",
        manifest_path,
        time.perf_counter() - write_t0,
        manifest["status_counts"],
    )

    LOG.info(
        "[drilldown] done build_professional_flow_drilldowns index_rows=%s qa_rows=%s "
        "overall_cells=%s elapsed=%.2fs",
        len(index),
        len(qa),
        overall_cell_i,
        time.perf_counter() - t0,
    )

    return {
        "index": index_path,
        "manifest": manifest_path,
        "qa": qa_path,
        "details_dir": details_dir,
    }



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build professional flow drilldown artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip professional drilldown builds for tables with more than 100 cells.",
    )
    args = parser.parse_args(argv)
    paths = build_professional_flow_drilldowns(
        args.repo_root,
        args.pack,
        args.run_root,
        args.tables_dir,
        args.tolerance,
        fast=args.fast,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
