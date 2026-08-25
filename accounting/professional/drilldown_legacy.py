from __future__ import annotations

"""Bounded compatibility runtime for professional drilldowns.

Current governed semantic execution lives in :mod:`accounting.professional.drilldown`
and its typed executors. This module retains only:

* stable index/detail rendering and source-discovery orchestration;
* a small set of presentation/diagnostic routes that do not yet have a typed
  current executor;
* utility helpers still consumed by the governed debt/annual executors.

Historical cash selection, debt position/activity selection, governed annual
rent/OPEX/draw reconstruction, direct FX measure defaults, and diagnostic
Box-level presentation routing have been retired.
"""

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

from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.logging_utils import configure_logging, get_logger
from accounting.professional.table_contracts import enrich_professional_table_contracts
from accounting.scope import assert_frame_within_scope, load_run_scope_if_present

INDEX_FILENAME = "professional_drilldown_index.csv"
MANIFEST_FILENAME = "professional_drilldown_manifest.json"
QA_FILENAME = "professional_drilldown_qa.csv"
DETAILS_DIRNAME = "details"
DEFAULT_TOLERANCE = 1e-6
FAST_TABLE_CELL_LIMIT = 100
MAX_TABLE_CELL_LIMIT = 500
TABLE_TOO_LARGE_WARNING = (
    "Table has too many cells to afford triggering drilldowns; "
    "skipping drilldown build for this table."
)

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
    "monthly_tables_unknown_review_net_matrix",
    "monthly_tables_draws_by_type_net_amount",
    "monthly_tables_fx_treasury_all_measures",
    "monthly_tables_fx_treasury_amount_in",
    "monthly_tables_fx_treasury_amount_out",
    "monthly_tables_fx_treasury_net_amount",
    "monthly_tables_fx_treasury_compact",
    *DERIVED_TABLE_IDS,
)

# Patched by the current public module so direct FX tables always use the
# single typed authority. The compatibility module does not implement FX
# measure selection.
FX_TREASURY_TABLE_IDS: set[str] | frozenset[str] = set()
FX_MEASURES: set[str] | frozenset[str] = set()

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

LOG = get_logger("prof drilldown")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _norm(value: Any) -> str:
    return _as_str(value).strip()


def _norm_period_key(value: Any) -> str:
    text = _norm(value)
    if not text:
        return ""
    try:
        value_float = float(text)
        if value_float.is_integer():
            return str(int(value_float))
    except Exception:
        pass
    return text


def _num(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(number) else float(number)


def _fmt_num(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return html.escape(_as_str(value))
    if math.isnan(number):
        return ""
    return f"{number:,.2f}" if abs(number) < 1000 else f"{number:,.0f}"


def _month_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if MONTH_RE.match(str(column))]


def _period_columns(table_id: str, df: pd.DataFrame) -> list[str]:
    annual_tables = {
        "overview_balance_dashboard",
        "income_operating_statement",
        "cash_annual_box_flow_bridge_wide",
        "annual_cash_close_by_box_wide",
        "annual_funding_by_actor_channel_wide",
        "annual_debt_stock_by_pair_wide",
        "annual_debt_activity_by_pair_wide",
    }
    matcher = YEAR_RE if table_id in annual_tables else MONTH_RE
    return [str(column) for column in df.columns if matcher.match(str(column))]


def _safe_id(*parts: Any) -> str:
    raw = "__".join(_norm(part) for part in parts if _norm(part))
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", raw).strip("_")[:180] or "drilldown"


def _first_present(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and _norm(row.get(name)):
            return row.get(name)
    return ""


def _metric_name(row: pd.Series) -> str:
    return _norm(
        _first_present(
            row,
            "metric",
            "line",
            "statement_line",
            "measure",
            "row",
            "label",
        )
    )


def _row_context(table_id: str, row: pd.Series) -> dict[str, Any]:
    periods = set(_period_columns(table_id, pd.DataFrame(columns=row.index)))
    return {
        str(key): ("" if pd.isna(value) else value)
        for key, value in row.to_dict().items()
        if str(key) not in periods
    }


def row_context_id(table_id: str, row_index: int, row: pd.Series) -> str:
    return _safe_id(
        table_id,
        row_index,
        json.dumps(_row_context(table_id, row), sort_keys=True, default=str),
    )


@dataclass(frozen=True)
class CellSpec:
    table_id: str
    measure: str
    filter_func: Callable[[pd.DataFrame, pd.Series], pd.Series]
    caveat_func: Callable[[pd.Series], str] = lambda row: ""
    unsupported_if: Callable[[pd.Series], bool] = lambda row: False


def _eq_col(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    if col not in df.columns or not _norm(value):
        return pd.Series(True, index=df.index)
    return df[col].fillna("").astype(str).str.strip().eq(_norm(value))


def _strict_eq_col(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    if col not in df.columns or not _norm(value):
        return pd.Series(False, index=df.index)
    return df[col].fillna("").astype(str).str.strip().eq(_norm(value))


def _source_filter_eq(df: pd.DataFrame, col: str, value: Any) -> pd.Series:
    return _strict_eq_col(df, col, value)


def _period_eq(df: pd.DataFrame, period: str) -> pd.Series:
    if "period" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["period"].fillna("").astype(str).str.strip().eq(str(period))


def _year_mask(df: pd.DataFrame, year: str) -> pd.Series:
    if "period" not in df.columns:
        return pd.Series(False, index=df.index)
    values = df["period"].fillna("").astype(str).str.strip()
    if MONTH_RE.match(str(year)):
        return values.eq(str(year))
    return values.str.slice(0, 4).eq(str(year))


def _pair_parts(pair: str) -> tuple[str, str]:
    text = _norm(pair)
    for separator in ("→", "->", "=>", "|"):
        if separator in text:
            debtor, creditor = text.split(separator, 1)
            return debtor.strip(), creditor.strip()
    return "", ""


def _measure_sum(df: pd.DataFrame, measure: str) -> float:
    if measure not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[measure], errors="coerce").fillna(0.0).sum())


def _annual_companion_long_row(
    row: pd.Series,
    period: str,
    display_value: float,
) -> pd.DataFrame:
    out = row.to_frame().T.copy()
    out["period"] = period
    out["value"] = display_value
    return out


def _contains_any(series: pd.Series, *needles: str) -> pd.Series:
    text = series.fillna("").astype(str)
    mask = pd.Series(False, index=series.index)
    for needle in needles:
        mask |= text.str.contains(needle, case=False, na=False, regex=False)
    return mask


def _regex_any(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(
        pattern, case=False, na=False, regex=True
    )


def _bucket_eq(df: pd.DataFrame, bucket: str) -> pd.Series:
    return df.get("semantic_bucket", pd.Series("", index=df.index)).fillna("").astype(str).eq(bucket)


def _bucket_contains(df: pd.DataFrame, pattern: str) -> pd.Series:
    return _regex_any(
        df.get("semantic_bucket", pd.Series("", index=df.index)), pattern
    )


def _subbucket_contains(df: pd.DataFrame, pattern: str) -> pd.Series:
    return _regex_any(
        df.get("semantic_subbucket", pd.Series("", index=df.index)), pattern
    )


def _unknown_mask(df: pd.DataFrame) -> pd.Series:
    return _bucket_contains(df, r"unknown|review") | _subbucket_contains(
        df, r"unknown|review"
    )


def _fb_mask(df: pd.DataFrame) -> pd.Series:
    columns = [
        column
        for column in ["Box", "payer", "receiver", "actor", "counterparty"]
        if column in df.columns
    ]
    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask |= _contains_any(df[column], "Family Business", "FB")
    return mask


def _rule_token_mask(df: pd.DataFrame, *rule_ids: str) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in ("rule_id", "rule_ids"):
        if column not in df.columns:
            continue
        text = df[column].fillna("").astype(str)
        for rule_id in rule_ids:
            mask |= text.eq(rule_id)
            mask |= text.str.contains(
                rf"(?:^|;){re.escape(rule_id)}(?:$|;)",
                regex=True,
                na=False,
            )
    return mask


def _semantic_mask(
    df: pd.DataFrame,
    *,
    bucket: str | None = None,
    subbucket: str | None = None,
    rule_ids: tuple[str, ...] = (),
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if bucket is not None:
        mask &= _strict_eq_col(df, "semantic_bucket", bucket)
    if subbucket is not None:
        mask &= _strict_eq_col(df, "semantic_subbucket", subbucket)
    if rule_ids:
        mask |= _rule_token_mask(df, *rule_ids)
    return mask


def _contract_measure(bucket: str, subbucket: str = "") -> str:
    return resolve_semantic_measure(bucket, subbucket) or ""


def _detail_from_audit(
    audit: pd.DataFrame,
    semantic_subset: pd.DataFrame,
    filter_func: Callable[[pd.DataFrame], pd.Series],
) -> tuple[pd.DataFrame, str]:
    if audit.empty:
        return semantic_subset.copy(), "semantic_only"

    try:
        subset = audit.loc[filter_func(audit)].copy()
    except Exception:
        subset = pd.DataFrame()

    tx_ids: set[str] = set()
    if "source_tx_ids_sample" in semantic_subset.columns:
        for item in semantic_subset["source_tx_ids_sample"].dropna().astype(str):
            tx_ids.update(token for token in item.split(";") if token)

    if tx_ids and "tx_id" in audit.columns:
        by_tx = audit.loc[audit["tx_id"].astype(str).isin(tx_ids)].copy()
        if not by_tx.empty:
            subset = by_tx

    return (
        subset,
        "classification_audit" if not subset.empty else "classification_audit_empty",
    )


# ---------------------------------------------------------------------------
# Remaining compatibility direct routes
# ---------------------------------------------------------------------------

def _spec_for_cell(table_id: str, row: pd.Series) -> CellSpec | None:
    """Resolve only compatibility families that lack a typed current identity."""

    measure = _norm(row.get("measure"))

    def period_currency(df: pd.DataFrame, r: pd.Series) -> pd.Series:
        return _eq_col(df, "Currency", r.get("Currency"))

    if table_id == "monthly_tables_flow_bucket_all_measures":
        return CellSpec(
            table_id,
            measure,
            lambda df, r: (
                period_currency(df, r)
                & _eq_col(df, "Box", r.get("Box"))
                & _eq_col(df, "semantic_bucket", r.get("semantic_bucket"))
            ),
            caveat_func=lambda _row: (
                "Generic bucket matrix is diagnostic compatibility; current "
                "semantic identities should use drilldown_cell_id."
            ),
        )

    if table_id == "monthly_tables_flow_subbucket_all_measures":
        return CellSpec(
            table_id,
            measure,
            lambda df, r: (
                period_currency(df, r)
                & _eq_col(df, "Box", r.get("Box"))
                & _eq_col(df, "semantic_bucket", r.get("semantic_bucket"))
                & _eq_col(df, "semantic_subbucket", r.get("semantic_subbucket"))
            ),
            caveat_func=lambda _row: (
                "Generic subbucket matrix is diagnostic compatibility; current "
                "semantic identities should use drilldown_cell_id."
            ),
        )

    if table_id == "monthly_tables_unknown_review_net_matrix":
        return CellSpec(
            table_id,
            "net_amount",
            lambda df, r: period_currency(df, r) & _unknown_mask(df),
        )

    if table_id == "monthly_tables_fb_bridge_matrix":
        metric = _metric_name(row)
        mapping: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
            "rent_or_revenue_in": (
                _contract_measure("operating_revenue"),
                lambda df: _bucket_eq(df, "operating_revenue"),
            ),
            "withdrawals_out": (
                _contract_measure("family_withdrawal_candidate"),
                lambda df: _bucket_eq(df, "family_withdrawal_candidate"),
            ),
            "funding_in": (
                _contract_measure("funding_contribution"),
                lambda df: _bucket_eq(df, "funding_contribution"),
            ),
            "net_flow": ("net_amount", lambda df: pd.Series(True, index=df.index)),
        }
        if metric not in mapping:
            return CellSpec(
                table_id,
                measure or metric,
                lambda df, _r: pd.Series(False, index=df.index),
                unsupported_if=lambda _r: True,
            )
        value_col, member_mask = mapping[metric]
        return CellSpec(
            table_id,
            value_col,
            lambda df, r: (
                period_currency(df, r) & _fb_mask(df) & member_mask(df)
            ),
            caveat_func=lambda _row: (
                "FB-related is a presentation bridge and is not identical to "
                "Box=Family Business."
            ),
        )

    if table_id in {
        "monthly_tables_pm_stress_matrix",
        "monthly_tables_household_bridge_matrix",
    }:
        box = (
            "Property Management"
            if table_id == "monthly_tables_pm_stress_matrix"
            else "Household"
        )
        metric = _metric_name(row)
        mapping = {
            "revenue_in": (
                _contract_measure("operating_revenue"),
                lambda df: _bucket_eq(df, "operating_revenue"),
            ),
            "property_opex_out": (
                _contract_measure("property_opex"),
                lambda df: _bucket_eq(df, "property_opex"),
            ),
            "opex_out": (
                _contract_measure("property_opex"),
                lambda df: _bucket_eq(df, "property_opex"),
            ),
            "withdrawals_out": (
                _contract_measure("family_withdrawal_candidate"),
                lambda df: _bucket_eq(df, "family_withdrawal_candidate"),
            ),
            "funding_in": (
                _contract_measure("funding_contribution"),
                lambda df: _bucket_eq(df, "funding_contribution"),
            ),
            "debt_net": (
                "net_amount",
                lambda df: _bucket_contains(df, "debt"),
            ),
            "unknown_net": ("net_amount", _unknown_mask),
            "net_flow": ("net_amount", lambda df: pd.Series(True, index=df.index)),
        }
        if metric not in mapping:
            return CellSpec(
                table_id,
                measure or metric,
                lambda df, _r: pd.Series(False, index=df.index),
                unsupported_if=lambda _r: True,
            )
        value_col, member_mask = mapping[metric]
        return CellSpec(
            table_id,
            value_col,
            lambda df, r: (
                period_currency(df, r)
                & _strict_eq_col(df, "Box", box)
                & member_mask(df)
            ),
        )

    if table_id == "monthly_tables_draws_by_type_net_amount":
        return CellSpec(
            table_id,
            "net_amount",
            lambda df, r: (
                period_currency(df, r)
                & _bucket_eq(df, "family_withdrawal_candidate")
                & _eq_col(df, "Box", r.get("Box"))
                & _eq_col(
                    df, "semantic_subbucket", r.get("semantic_subbucket")
                )
            ),
        )

    return None


# Replaced at import time by accounting.professional.drilldown.
def _fx_treasury_measure_for_row(table_id: str, row: pd.Series) -> str:
    return ""


# ---------------------------------------------------------------------------
# Remaining compatibility derived routes
# ---------------------------------------------------------------------------

def _semantic_filter_for_statement_line(
    line: str,
) -> tuple[str, Callable[[pd.DataFrame], pd.Series]] | None:
    mapping = {
        "operating_revenue": (
            _contract_measure("operating_revenue"),
            lambda df: _bucket_eq(df, "operating_revenue"),
        ),
        "property_opex_true": (
            _contract_measure("property_opex"),
            lambda df: _bucket_eq(df, "property_opex"),
        ),
        "funding_contributions": (
            _contract_measure("funding_contribution"),
            lambda df: _bucket_eq(df, "funding_contribution"),
        ),
        "family_draws_or_distributions": (
            _contract_measure("family_withdrawal_candidate"),
            lambda df: _bucket_eq(df, "family_withdrawal_candidate"),
        ),
    }
    return mapping.get(_norm(line))


@dataclass(frozen=True)
class AnnualFormulaSpec:
    formula_id: str
    label: str
    component_metric_ids: tuple[str, ...]
    formula: str


def _annual_formula_spec(row: pd.Series) -> AnnualFormulaSpec | None:
    label = _metric_name(row).casefold().strip()
    specs = {
        "margen operativo": AnnualFormulaSpec(
            "operating_margin",
            "Margen operativo",
            ("IS.NET.OPERATING", "IS.REVENUE.OPERATING"),
            "IS.NET.OPERATING / IS.REVENUE.OPERATING",
        ),
        "opex / renta": AnnualFormulaSpec(
            "opex_to_rent",
            "OPEX / renta",
            ("IS.OPEX.PROPERTY", "IS.REVENUE.OPERATING"),
            "IS.OPEX.PROPERTY / IS.REVENUE.OPERATING",
        ),
        "retiros / resultado operativo": AnnualFormulaSpec(
            "draws_to_operating_result",
            "Retiros / resultado operativo",
            ("DIST.DRAWS.PERSONAL", "IS.NET.OPERATING"),
            "DIST.DRAWS.PERSONAL / IS.NET.OPERATING",
        ),
        "cobertura después de funding y retiros": AnnualFormulaSpec(
            "coverage_after_funding_and_draws",
            "Cobertura después de funding y retiros",
            (
                "COV.NET.AFTER_DRAWS",
                "IS.NET.OPERATING",
                "FUND.CONTRIB.TOTAL",
                "DIST.DRAWS.PERSONAL",
            ),
            "COV.NET.AFTER_DRAWS",
        ),
        "cobertura despues de funding y retiros": AnnualFormulaSpec(
            "coverage_after_funding_and_draws",
            "Cobertura después de funding y retiros",
            (
                "COV.NET.AFTER_DRAWS",
                "IS.NET.OPERATING",
                "FUND.CONTRIB.TOTAL",
                "DIST.DRAWS.PERSONAL",
            ),
            "COV.NET.AFTER_DRAWS",
        ),
    }
    return specs.get(label)


def _safe_div(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= DEFAULT_TOLERANCE:
        return None
    return numerator / denominator


def _annual_metric_rows(
    annual: pd.DataFrame,
    *,
    metric_id: str,
    period: str,
    currency: str,
) -> pd.DataFrame:
    if annual.empty or "metric_id" not in annual.columns:
        return pd.DataFrame()
    mask = annual["metric_id"].fillna("").astype(str).str.strip().eq(metric_id)
    mask &= annual.get("period", pd.Series("", index=annual.index)).map(
        _norm_period_key
    ).eq(_norm_period_key(period))
    mask &= annual.get("Currency", pd.Series("", index=annual.index)).map(
        _norm
    ).eq(currency)
    if "dimension_name" in annual.columns:
        mask &= annual["dimension_name"].fillna("").astype(str).str.strip().eq("")
    return annual.loc[mask].copy()


def _build_annual_formula_cell(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    currency: str,
    display_value: float,
    annual: pd.DataFrame,
    tolerance: float,
):
    spec = _annual_formula_spec(row)
    if spec is None:
        return None

    components: list[pd.DataFrame] = []
    values: dict[str, float] = {}
    for metric_id in spec.component_metric_ids:
        rows = _annual_metric_rows(
            annual,
            metric_id=metric_id,
            period=period,
            currency=currency,
        )
        if not rows.empty:
            values[metric_id] = _measure_sum(rows, "value")
            components.append(rows)

    if spec.formula_id == "operating_margin":
        value = _safe_div(
            values.get("IS.NET.OPERATING", 0.0),
            values.get("IS.REVENUE.OPERATING", 0.0),
        )
    elif spec.formula_id == "opex_to_rent":
        value = _safe_div(
            values.get("IS.OPEX.PROPERTY", 0.0),
            values.get("IS.REVENUE.OPERATING", 0.0),
        )
    elif spec.formula_id == "draws_to_operating_result":
        value = _safe_div(
            values.get("DIST.DRAWS.PERSONAL", 0.0),
            values.get("IS.NET.OPERATING", 0.0),
        )
    else:
        value = values.get("COV.NET.AFTER_DRAWS")

    if value is None:
        return (
            STATUS_UNSUPPORTED,
            0.0,
            -display_value,
            "legacy_formula_compatibility",
            "annual_balance_dashboard_metrics.csv",
            {
                "unsupported": True,
                "reason": "formula components unavailable or denominator zero",
                "formula": spec.formula,
            },
            "Historical formula compatibility; current supported rows use DerivedMetricSpec.",
            pd.DataFrame(),
            [],
        )

    matched = float(value)
    residual = matched - display_value
    status = (
        STATUS_OK if abs(residual) <= tolerance else STATUS_RESIDUAL_WARNING
    )
    component_rows = (
        pd.concat(components, ignore_index=True)
        if components
        else pd.DataFrame()
    )
    formula_rows = pd.DataFrame(
        [
            {
                "formula": spec.formula,
                "matched_value": matched,
                "displayed_value": display_value,
                "residual": residual,
            }
        ]
    )
    return (
        status,
        matched,
        residual,
        "legacy_formula_compatibility",
        "annual_balance_dashboard_metrics.csv",
        {
            "formula_id": spec.formula_id,
            "formula": spec.formula,
            "period": period,
            "Currency": currency,
        },
        "Historical formula compatibility; current supported rows use DerivedMetricSpec.",
        component_rows,
        [("Formula", formula_rows), ("Component rows", component_rows)],
    )


def _amount_nonzero_mask(
    df: pd.DataFrame,
    col: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(0.0)
        .abs()
        .gt(tolerance)
    )


def _any_flow_amount_mask(
    df: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in ("amount_in", "amount_out", "net_amount", "amount_abs"):
        mask |= _amount_nonzero_mask(df, col, tolerance)
    return mask


def _cash_bridge_line_spec(
    line: str,
) -> tuple[str, Callable[[pd.DataFrame], pd.Series], str] | None:
    """Compatibility mapping for non-FX annual cash-bridge flow lines.

    FX is intentionally absent: current FX cells require explicit measure and
    grain through the single FX authority.
    """

    line_n = _norm(line).casefold()
    if "renta" in line_n or "rent" in line_n or "ingresos operativos" in line_n:
        return (
            "amount_in",
            lambda df: _semantic_mask(
                df,
                bucket="operating_revenue",
                subbucket="rent",
                rule_ids=("R001_rent_collections",),
            ),
            "rent => operating_revenue/rent",
        )
    if "funding" in line_n or "contrib" in line_n:
        return (
            "amount_in",
            lambda df: _semantic_mask(
                df,
                bucket="funding_contribution",
                rule_ids=("R006_contribution",),
            ),
            "funding => narrow funding_contribution",
        )
    if "opex" in line_n and (
        "propiedad" in line_n
        or "property" in line_n
        or "patrimonial" in line_n
    ):
        return (
            "amount_out",
            lambda df: _semantic_mask(df, bucket="property_opex"),
            "property OPEX",
        )
    if "impuesto" in line_n or "tax" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df, bucket="property_opex", subbucket="taxes"
            ),
            "property OPEX/taxes",
        )
    if "servicio" in line_n or "service" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df, bucket="property_opex", subbucket="services"
            ),
            "property OPEX/services",
        )
    if "mantenimiento" in line_n or "maintenance" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df, bucket="property_opex", subbucket="maintenance"
            ),
            "property OPEX/maintenance",
        )
    if "legal" in line_n:
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df, bucket="property_opex", subbucket="legal"
            ),
            "property OPEX/legal",
        )
    if "debt net" in line_n or "net debt" in line_n or (
        "deuda" in line_n and "neto" in line_n
    ):
        return (
            "net_amount",
            lambda df: _semantic_mask(df, bucket="debt_movement"),
            "debt movement net",
        )
    if any(
        token in line_n
        for token in (
            "personal",
            "retiro",
            "withdrawal",
            "draw",
            "distribucion",
            "distribución",
            "distribution",
        )
    ):
        return (
            "amount_out",
            lambda df: _semantic_mask(
                df, bucket="family_withdrawal_candidate"
            ),
            "family withdrawal",
        )
    if "total entradas" in line_n or "total inflows" in line_n:
        return (
            "amount_in",
            lambda df: _amount_nonzero_mask(df, "amount_in"),
            "all nonzero inflows",
        )
    if "total salidas" in line_n or "total outflows" in line_n:
        return (
            "amount_out",
            lambda df: _amount_nonzero_mask(df, "amount_out"),
            "all nonzero outflows",
        )
    if "flujo neto" in line_n or "net flow" in line_n:
        return ("net_amount", _any_flow_amount_mask, "all nonzero flows")
    if "unknown" in line_n or "review" in line_n or "revis" in line_n:
        return ("net_amount", _unknown_mask, "unknown/review")
    return None


ANNUAL_PRESENTATION_METRIC_IDS = {
    "ingresos operativos": "IS.REVENUE.OPERATING",
    "renta": "IS.RENT.TOTAL",
    "opex propiedad": "IS.OPEX.PROPERTY",
    "funding / aportes": "FUND.CONTRIB.TOTAL",
    "aportes": "FUND.CONTRIB.TOTAL",
    "retiros / gasto personal": "DIST.DRAWS.PERSONAL",
    "gasto personal": "DIST.DRAWS.PERSONAL",
    "dividendos": "DIST.DIVIDENDS",
    "deuda total abierta": "ID.DEBT.TOTAL.OPEN",
    "principal abierto": "ID.DEBT.PRINCIPAL.OPEN",
    "interés abierto": "ID.DEBT.INTEREST.OPEN",
    "interes abierto": "ID.DEBT.INTEREST.OPEN",
}

ANNUAL_METRIC_ID_ALIASES = {
    "ID.DEBT.TOTAL.OPEN": ("BS.DEBT.TOTAL.OPEN",),
    "ID.DEBT.PRINCIPAL.OPEN": ("BS.DEBT.PRINCIPAL.OPEN",),
    "ID.DEBT.INTEREST.OPEN": ("BS.DEBT.INTEREST.OPEN",),
}


def _annual_metric_id_candidates_for_row(row: pd.Series) -> list[str]:
    explicit = _norm(row.get("metric_id"))
    label = _metric_name(row).casefold()
    primary = ANNUAL_PRESENTATION_METRIC_IDS.get(label) or explicit
    if not primary:
        return []
    return list(
        dict.fromkeys([primary, *ANNUAL_METRIC_ID_ALIASES.get(primary, ())])
    )


def _annual_source_rows(
    annual: pd.DataFrame,
    row: pd.Series,
    period: str,
) -> pd.DataFrame:
    if annual.empty or "metric_id" not in annual.columns:
        return pd.DataFrame()
    candidates = _annual_metric_id_candidates_for_row(row)
    if not candidates:
        return pd.DataFrame()

    mask = annual["metric_id"].map(_norm).isin(set(candidates))
    mask &= annual.get("period", pd.Series("", index=annual.index)).map(
        _norm_period_key
    ).eq(_norm_period_key(period))
    mask &= annual.get("Currency", pd.Series("", index=annual.index)).map(
        _norm
    ).eq(_norm(row.get("Currency")))

    for dim_col in ("dimension_name", "dimension_value"):
        value = _norm(row.get(dim_col))
        if value and dim_col in annual.columns:
            mask &= annual[dim_col].map(_norm).eq(value)
    return annual.loc[mask].copy()


def _statement_line(row: pd.Series) -> str:
    return _norm(
        _first_present(row, "statement_line", "line", "metric", "label")
    )


def _statement_components(
    stmt: pd.DataFrame,
    period: str,
    currency: str,
    component_lines: list[str],
) -> pd.DataFrame:
    if stmt.empty:
        return pd.DataFrame()
    mask = _year_mask(stmt, period) & _eq_col(stmt, "Currency", currency)
    if "statement_line" in stmt.columns:
        mask &= stmt["statement_line"].astype(str).isin(component_lines)
    return stmt.loc[mask].copy()


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
):
    """Remaining compatibility dispatcher.

    Governed cash, debt, annual additive flows, funding support, derived metrics
    and direct FX never enter this function from the public dispatcher.
    """

    currency = _norm(row.get("Currency"))
    line = _statement_line(row)
    caveat = (
        "Compatibility presentation route; displayed accounting values remain "
        "owned by governed upstream artifacts."
    )

    if table_id in {
        "monthly_tables_operating_statement_matrix",
        "monthly_tables_operating_statement_matrix_ars",
    }:
        if stmt.empty:
            return (
                STATUS_ERROR,
                0.0,
                -display_value,
                "missing_source",
                "monthly_operating_statement.csv",
                {"error": "missing monthly_operating_statement.csv"},
                caveat,
                pd.DataFrame(),
                [],
            )
        source = stmt.loc[
            _year_mask(stmt, period)
            & _eq_col(stmt, "Currency", currency)
            & _eq_col(stmt, "statement_line", line)
        ].copy()
        matched = _measure_sum(source, "amount")
        residual = matched - display_value
        status = (
            STATUS_EMPTY
            if source.empty
            else STATUS_OK
            if abs(residual) <= tolerance
            else STATUS_RESIDUAL_WARNING
        )
        sections: list[tuple[str, pd.DataFrame]] = [
            ("Source statement rows", source)
        ]
        if line == "net_operating":
            components = _statement_components(
                stmt,
                period,
                currency,
                ["operating_revenue", "property_opex_true"],
            )
            sections.extend(
                [
                    (
                        "Formula",
                        pd.DataFrame(
                            [
                                {
                                    "formula": "operating_revenue - property_opex_true",
                                    "displayed_value": display_value,
                                    "source_sum": matched,
                                }
                            ]
                        ),
                    ),
                    ("Component rows", components),
                ]
            )
        elif line == "coverage_after_draws":
            components = _statement_components(
                stmt,
                period,
                currency,
                [
                    "net_operating",
                    "funding_contributions",
                    "family_draws_or_distributions",
                ],
            )
            sections.extend(
                [
                    (
                        "Formula",
                        pd.DataFrame(
                            [
                                {
                                    "formula": (
                                        "net_operating + funding_contributions "
                                        "- family_draws_or_distributions"
                                    ),
                                    "displayed_value": display_value,
                                    "source_sum": matched,
                                }
                            ]
                        ),
                    ),
                    ("Component rows", components),
                ]
            )
        return (
            status,
            matched,
            residual,
            "monthly_operating_statement",
            "monthly_operating_statement.csv",
            {
                "period": period,
                "Currency": currency,
                "statement_line": line,
            },
            caveat,
            source,
            sections,
        )

    if table_id in {"overview_balance_dashboard", "income_operating_statement"}:
        formula = _build_annual_formula_cell(
            table_id=table_id,
            row=row,
            period=period,
            currency=currency,
            display_value=display_value,
            annual=annual,
            tolerance=tolerance,
        )
        if formula is not None:
            return formula

        annual_rows = _annual_source_rows(annual, row, period)
        if annual_rows.empty:
            return (
                STATUS_UNSUPPORTED,
                0.0,
                -display_value,
                "unsupported",
                "annual_balance_dashboard_metrics.csv",
                {
                    "unsupported": True,
                    "reason": "no matching governed annual metric row",
                },
                "Historical report-layer recomputation has been retired.",
                pd.DataFrame(),
                [],
            )

        matched = _measure_sum(annual_rows, "value")
        residual = matched - display_value
        status = (
            STATUS_OK
            if abs(residual) <= tolerance
            else STATUS_RESIDUAL_WARNING
        )
        source_table = _norm(annual_rows.iloc[0].get("source_table"))
        sections: list[tuple[str, pd.DataFrame]] = [
            ("Annual metric row", annual_rows)
        ]
        detail = annual_rows
        lineage = "annual_balance_dashboard_metrics"

        if source_table == "monthly_operating_statement.csv" and not stmt.empty:
            source_filter = _norm(annual_rows.iloc[0].get("source_filter"))
            statement_line = ""
            if "statement_line=" in source_filter:
                statement_line = (
                    source_filter.split("statement_line=", 1)[1]
                    .split(";", 1)[0]
                    .strip()
                )
            monthly_rows = stmt.loc[
                _year_mask(stmt, period)
                & _eq_col(stmt, "Currency", currency)
                & (
                    _eq_col(stmt, "statement_line", statement_line)
                    if statement_line
                    else pd.Series(True, index=stmt.index)
                )
            ].copy()
            sections.append(("Monthly source rows", monthly_rows))
            detail = monthly_rows if not monthly_rows.empty else annual_rows
            lineage = "annual_to_monthly_statement"

        elif source_table == "monthly_debt_position.csv":
            debt_rows = (
                debt_position.loc[
                    _year_mask(debt_position, period)
                    & _eq_col(debt_position, "Currency", currency)
                ].copy()
                if not debt_position.empty
                else pd.DataFrame()
            )
            sections.append(("Debt position rows", debt_rows))
            detail = debt_rows if not debt_rows.empty else annual_rows
            lineage = "annual_to_governed_debt_position"

        elif source_table == "monthly_cash_close.csv":
            return (
                STATUS_UNSUPPORTED,
                0.0,
                -display_value,
                "unsupported",
                "annual_balance_dashboard_metrics.csv",
                {
                    "unsupported": True,
                    "reason": (
                        "cash headline requires the typed cash-position route; "
                        "generic annual compatibility cannot select cash"
                    ),
                },
                "Cash stock is not re-selected by report compatibility.",
                annual_rows,
                sections,
            )

        return (
            status,
            matched,
            residual,
            lineage,
            "annual_balance_dashboard_metrics.csv",
            {
                "period": period,
                "Currency": currency,
                "source_table": source_table,
                "row_context": _row_context(table_id, row),
            },
            caveat,
            detail,
            sections,
        )

    if table_id == "cash_annual_box_flow_bridge_wide":
        line_lower = line.casefold()
        if any(
            token in line_lower
            for token in ("validated cash", "cash close", "diagnostic box balance")
        ):
            return (
                STATUS_UNSUPPORTED,
                0.0,
                -display_value,
                "unsupported",
                "monthly_flow_semantic_split.csv",
                {"unsupported": True, "reason": "stock/diagnostic line"},
                "Cash levels are not flow drilldowns.",
                pd.DataFrame(),
                [],
            )
        if split.empty:
            return (
                STATUS_ERROR,
                0.0,
                -display_value,
                "missing_source",
                "monthly_flow_semantic_split.csv",
                {"error": "missing monthly_flow_semantic_split.csv"},
                caveat,
                pd.DataFrame(),
                [],
            )

        line_spec = _cash_bridge_line_spec(line)
        if line_spec is None:
            return (
                STATUS_UNSUPPORTED,
                0.0,
                -display_value,
                "unsupported",
                "monthly_flow_semantic_split.csv",
                {
                    "unsupported": True,
                    "reason": "cash bridge line has no compatibility mapping",
                },
                "Unmapped cash-bridge presentation lines fail closed.",
                pd.DataFrame(),
                [],
            )
        measure, member_mask, note = line_spec
        semantic_rows = split.loc[
            _year_mask(split, period)
            & _eq_col(split, "Currency", currency)
            & (
                _eq_col(split, "Box", row.get("Box"))
                if _norm(row.get("Box"))
                else pd.Series(True, index=split.index)
            )
            & member_mask(split)
        ].copy()
        matched = _measure_sum(semantic_rows, measure)
        residual = matched - display_value
        status = (
            STATUS_EMPTY
            if semantic_rows.empty
            else STATUS_OK
            if abs(residual) <= tolerance
            else STATUS_RESIDUAL_WARNING
        )
        detail, lineage = _detail_from_audit(
            audit,
            semantic_rows,
            lambda df: (
                _year_mask(df, period)
                & _eq_col(df, "Currency", currency)
                & (
                    _eq_col(df, "Box", row.get("Box"))
                    if _norm(row.get("Box"))
                    else pd.Series(True, index=df.index)
                )
                & member_mask(df)
            ),
        )
        if detail.empty and lineage != "semantic_only":
            detail = semantic_rows
        return (
            status,
            matched,
            residual,
            lineage,
            "monthly_flow_semantic_split.csv",
            {
                "year": period,
                "Currency": currency,
                "Box": _norm(row.get("Box")),
                "measure": measure,
                "compatibility_mapping": note,
            },
            caveat,
            detail if not detail.empty else semantic_rows,
            [
                ("Flow bridge semantic rows", semantic_rows),
                ("Classification rows", detail),
            ],
        )

    return (
        STATUS_UNSUPPORTED,
        0.0,
        -display_value,
        "unsupported",
        "",
        {"unsupported": True},
        "",
        pd.DataFrame(),
        [],
    )


# ---------------------------------------------------------------------------
# Source discovery and detail rendering
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _candidate_run_roots(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None,
) -> Iterable[Path]:
    if run_root is not None:
        yield run_root
    yield pack_dir
    yield pack_dir.parent
    yield repo_root / "out" / "run" / "accounting" / "latest_FBPM"


def _find_source(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None,
    filename: str,
) -> Path | None:
    for root in _candidate_run_roots(repo_root, pack_dir, run_root):
        path = root / filename
        if path.exists():
            return path

    run_scope = load_run_scope_if_present(run_root) if run_root is not None else None
    scope_tag = run_scope.tag if run_scope is not None else "FBPM"
    candidates = [
        repo_root / "out" / "metrics" / f"latest_{scope_tag}" / filename,
        repo_root
        / "public"
        / "accounting"
        / f"latest_{scope_tag}"
        / "canonical_dashboard"
        / filename,
        repo_root / "public" / "accounting" / f"latest_{scope_tag}" / filename,
        pack_dir / "source" / filename,
        pack_dir / "tables" / filename,
    ]
    return next((path for path in candidates if path.exists()), None)


DETAIL_SUM_COLUMNS = [
    "amount_in",
    "amount_out",
    "net_amount",
    "amount_abs",
    "amount",
    "value",
    "close_amount",
    "new_principal",
    "interest_accrued",
    "repayments",
    "adjustments",
    "net_change",
    "open_amount",
    "open_principal",
    "open_interest",
    "open_total",
]


def _detail_sum_row_df(df: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, Any] = {"rows": int(len(df)), "cols": int(len(df.columns))}
    for column in DETAIL_SUM_COLUMNS:
        if column in df.columns:
            row[column] = float(
                pd.to_numeric(df[column], errors="coerce").fillna(0.0).sum()
            )
    return pd.DataFrame([row])


def _render_df_section(title: str, df: pd.DataFrame) -> str:
    title_html = html.escape(title)
    if df.empty:
        return f"<h2>{title_html}</h2><p class='warn'>No rows.</p>"
    return (
        f"<h2>{title_html}</h2>"
        + df.to_html(index=False, escape=True, classes="detail", border=0)
    )


def _write_detail_html(
    path: Path,
    index_row: dict[str, Any],
    detail_df: pd.DataFrame,
    sections: list[tuple[str, pd.DataFrame]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filter_json = json.dumps(
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
    reconciliation = pd.DataFrame(
        [
            {
                "Displayed value": index_row.get("display_value", 0.0),
                "Matched sum": index_row.get("matched_value_sum", 0.0),
                "Residual": index_row.get("residual", 0.0),
                "Matched rows": index_row.get("matched_rows", 0),
                "Status": index_row.get("status", ""),
            }
        ]
    )
    body = [
        _render_df_section("Reconciliation", reconciliation),
        _render_df_section("Drilldown numeric sums", _detail_sum_row_df(detail_df)),
    ]
    if sections:
        body.extend(_render_df_section(title, df) for title, df in sections)
    else:
        body.append(_render_df_section("Relevant rows", detail_df))

    detail_csv_name = Path(
        _as_str(index_row.get("detail_csv_relpath"))
    ).name
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>
<h1>{html.escape(_as_str(index_row.get('table_id')))}</h1>
<p><strong>Displayed value:</strong> {_fmt_num(index_row.get('display_value'))}<br>
<strong>Matched sum:</strong> {_fmt_num(index_row.get('matched_value_sum'))}<br>
<strong>Residual:</strong> {_fmt_num(index_row.get('residual'))}<br>
<strong>Source artifact:</strong> {html.escape(_as_str(index_row.get('source_artifact')))}<br>
<a href='{html.escape(detail_csv_name)}'>Open detail CSV</a></p>
<h2>Filters</h2><pre>{html.escape(filter_json)}</pre>
<h2>Row context</h2><pre>{html.escape(row_context)}</pre>
{''.join(body)}
</body></html>""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Stable orchestration
# ---------------------------------------------------------------------------

def build_professional_flow_drilldowns(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None = None,
    tables_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    fast: bool = False,
) -> dict[str, Path]:
    repo_root = Path(repo_root)
    pack_dir = Path(pack_dir)
    tables_dir = Path(tables_dir) if tables_dir is not None else pack_dir / "tables"
    drill_dir = pack_dir / "drilldown"
    details_dir = drill_dir / DETAILS_DIRNAME
    details_dir.mkdir(parents=True, exist_ok=True)

    enrich_professional_table_contracts(tables_dir)

    source_names = {
        "split": "monthly_flow_semantic_split.csv",
        "audit": "classification_audit.csv",
        "stmt": "monthly_operating_statement.csv",
        "annual": "annual_balance_dashboard_metrics.csv",
        "cash_close": "monthly_cash_close.csv",
        "debt_activity": "monthly_debt_activity.csv",
        "debt_position": "monthly_debt_position.csv",
    }
    paths = {
        key: _find_source(repo_root, pack_dir, run_root, filename)
        for key, filename in source_names.items()
    }
    frames = {
        key: _read_csv(path) if path is not None else pd.DataFrame()
        for key, path in paths.items()
    }

    run_scope = load_run_scope_if_present(run_root) if run_root is not None else None
    if run_scope is not None:
        for key, source_name in (
            ("split", "monthly_flow_semantic_split.csv"),
            ("audit", "classification_audit.csv"),
        ):
            assert_frame_within_scope(frames[key], run_scope, source=source_name)

    numeric_columns = [
        "amount_in",
        "amount_out",
        "net_amount",
        "amount_abs",
        "amount",
        "value",
        "close_amount",
        "new_principal",
        "interest_accrued",
        "repayments",
        "adjustments",
        "net_change",
        "open_amount",
        "open_principal",
        "open_interest",
        "open_total",
    ]
    for frame in frames.values():
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                ).fillna(0.0)

    index_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    cell_limit = FAST_TABLE_CELL_LIMIT if fast else MAX_TABLE_CELL_LIMIT

    for table_id in SUPPORTED_TABLE_IDS:
        table_path = tables_dir / f"{table_id}.csv"
        if not table_path.exists():
            qa_rows.append(
                {
                    "table_id": table_id,
                    "drilldown_id": "",
                    "check": "table_exists",
                    "status": "warning",
                    "detail": f"Missing table: {table_path}",
                }
            )
            continue

        table = pd.read_csv(table_path)
        periods = _period_columns(table_id, table)
        if not periods:
            qa_rows.append(
                {
                    "table_id": table_id,
                    "drilldown_id": "",
                    "check": "period_columns",
                    "status": "warning",
                    "detail": "No period columns detected",
                }
            )
            continue

        total_cells = len(table) * len(periods)
        if total_cells > cell_limit:
            qa_rows.append(
                {
                    "table_id": table_id,
                    "drilldown_id": "",
                    "check": "table_cell_limit",
                    "status": "warning",
                    "detail": (
                        f"{TABLE_TOO_LARGE_WARNING} table_id={table_id}; "
                        f"cells={total_cells}; limit={cell_limit}; fast={fast}"
                    ),
                }
            )
            continue

        for row_idx, row in table.iterrows():
            spec = _spec_for_cell(table_id, row)
            context = _row_context(table_id, row)
            row_id = row_context_id(table_id, int(row_idx), row)

            for period in periods:
                display_value = _num(row.get(period))
                if abs(display_value) <= tolerance:
                    continue

                measure = spec.measure if spec else _norm(row.get("measure"))
                drilldown_id = _safe_id(
                    table_id, row_idx, period, row.get("Currency"), measure
                )
                detail_csv_rel = (
                    f"drilldown/{DETAILS_DIRNAME}/{drilldown_id}.csv"
                )
                detail_html_rel = (
                    f"drilldown/{DETAILS_DIRNAME}/{drilldown_id}.html"
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

                if table_id in DERIVED_TABLE_IDS:
                    result = _build_derived_cell(
                        table_id=table_id,
                        row=row,
                        period=period,
                        display_value=display_value,
                        split=frames["split"],
                        audit=frames["audit"],
                        stmt=frames["stmt"],
                        annual=frames["annual"],
                        cash_close=frames["cash_close"],
                        debt_activity=frames["debt_activity"],
                        debt_position=frames["debt_position"],
                        tolerance=tolerance,
                    )
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
                    ) = result
                else:
                    split = frames["split"]
                    if split.empty:
                        status, matched, residual = (
                            STATUS_ERROR,
                            0.0,
                            -display_value,
                        )
                        lineage = "missing_source"
                        source_artifact = "monthly_flow_semantic_split.csv"
                        filters = {
                            "error": "missing monthly_flow_semantic_split.csv"
                        }
                        caveat = ""
                        detail_df = pd.DataFrame()
                        sections = []
                    elif not _norm(row.get("Currency")):
                        status, matched, residual = (
                            STATUS_UNSUPPORTED,
                            0.0,
                            -display_value,
                        )
                        lineage = "unsupported"
                        source_artifact = "monthly_flow_semantic_split.csv"
                        filters = {
                            "unsupported": True,
                            "reason": (
                                "missing Currency would risk cross-currency "
                                "aggregation"
                            ),
                            "measure": measure,
                        }
                        caveat = ""
                        detail_df = pd.DataFrame()
                        sections = []
                    elif (
                        spec is None
                        or spec.unsupported_if(row)
                        or measure not in split.columns
                    ):
                        status, matched, residual = (
                            STATUS_UNSUPPORTED,
                            0.0,
                            -display_value,
                        )
                        lineage = "unsupported"
                        source_artifact = "monthly_flow_semantic_split.csv"
                        filters = {
                            "unsupported": True,
                            "measure": measure,
                        }
                        caveat = spec.caveat_func(row) if spec else ""
                        detail_df = pd.DataFrame()
                        sections = []
                    else:
                        semantic_subset = split.loc[
                            _period_eq(split, period)
                            & spec.filter_func(split, row)
                        ].copy()
                        matched = _measure_sum(semantic_subset, spec.measure)
                        residual = matched - display_value
                        status = (
                            STATUS_EMPTY
                            if semantic_subset.empty
                            else STATUS_OK
                            if abs(residual) <= tolerance
                            else STATUS_RESIDUAL_WARNING
                        )
                        detail_df, lineage = _detail_from_audit(
                            frames["audit"],
                            semantic_subset,
                            lambda df, r=row, p=period, s=spec: (
                                _period_eq(df, p) & s.filter_func(df, r)
                            ),
                        )
                        if detail_df.empty and lineage != "semantic_only":
                            detail_df = semantic_subset
                        source_artifact = "monthly_flow_semantic_split.csv"
                        filters = {
                            "period": period,
                            "Currency": _norm(row.get("Currency")),
                            "measure": spec.measure,
                            "row_context": context,
                        }
                        caveat = spec.caveat_func(row)
                        sections = []

                out_row = {
                    **base,
                    "measure": _norm(filters.get("measure")) or measure,
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
                detail_path = pack_dir / detail_csv_rel
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                detail_df.to_csv(detail_path, index=False)
                _write_detail_html(
                    pack_dir / detail_html_rel,
                    out_row,
                    detail_df,
                    sections=sections,
                )
                index_rows.append(out_row)
                qa_rows.append(
                    {
                        "table_id": table_id,
                        "drilldown_id": drilldown_id,
                        "check": "cell_reconciliation",
                        "status": (
                            "pass"
                            if status == STATUS_OK
                            else "warning"
                            if status
                            in {
                                STATUS_EMPTY,
                                STATUS_RESIDUAL_WARNING,
                                STATUS_UNSUPPORTED,
                                "unavailable",
                            }
                            else "fail"
                        ),
                        "detail": (
                            f"status={status}; residual={residual}; "
                            f"matched_rows={len(detail_df)}"
                        ),
                    }
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
    index = pd.DataFrame(index_rows, columns=columns)
    qa = pd.DataFrame(
        qa_rows,
        columns=[
            "table_id",
            "drilldown_id",
            "check",
            "status",
            "detail",
        ],
    )

    index_path = drill_dir / INDEX_FILENAME
    manifest_path = drill_dir / MANIFEST_FILENAME
    qa_path = drill_dir / QA_FILENAME
    drill_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)
    qa.to_csv(qa_path, index=False)
    manifest = {
        "created_at_utc": _now_iso(),
        "repo_root": str(repo_root),
        "pack_dir": str(pack_dir),
        "tables_dir": str(tables_dir),
        "run_root": str(run_root or ""),
        **{
            source_names[key].replace(".csv", ""): str(paths[key] or "")
            for key in source_names
        },
        "tolerance": tolerance,
        "fast": bool(fast),
        "table_cell_limit": int(cell_limit),
        "index_rows": int(len(index)),
        "qa_rows": int(len(qa)),
        "status_counts": (
            index["status"].value_counts().to_dict()
            if not index.empty
            else {}
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "index": index_path,
        "manifest": manifest_path,
        "qa": qa_path,
        "details_dir": details_dir,
    }


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Build professional flow drilldown artifacts."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--fast", action="store_true")
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
