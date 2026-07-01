
"""
Shared helpers for governed accounting report notebooks.

This module is intentionally report-layer code, not backend core.

Principles:
- Read contractual/public artifacts.
- Normalize metrics for reporting.
- Build human-readable tables.
- Expose extended QA observations.
- Export clean report outputs.
- Do not classify ledger rows.
- Do not infer cash.
- Do not sum ARS + USD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Iterable, Any
import re
import html
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------

DEFAULT_REQUIRED_ARTIFACTS = {
    "annual_dashboard_metrics": "public/accounting/latest/canonical_dashboard/annual_balance_dashboard_metrics.csv",
    "annual_dashboard_qa": "public/accounting/latest/canonical_dashboard/annual_balance_dashboard_qa.csv",
    "metric_contract_frontier": "public/accounting/latest/public_contract/metric_contract_frontier.csv",
    "public_manifest": "public/accounting/latest/manifest.csv",
}

DEFAULT_OPTIONAL_ARTIFACTS = {
    "artifact_contracts_public": "public/accounting/latest/public_contract/artifact_contracts.csv",
    "publish_contract_qa": "public/accounting/latest/qa/publish_contract_qa.csv",
    "validation_report": "out/metrics/latest/validation_report.csv",
    "monthly_operating_statement": "out/run/accounting/latest/monthly_operating_statement.csv",
    "monthly_flow_semantic_split": "out/run/accounting/latest/monthly_flow_semantic_split.csv",
    "monthly_cash_close": "out/run/accounting/latest/monthly_cash_close.csv",
    "monthly_debt_position": "out/run/accounting/latest/monthly_debt_position.csv",
    "monthly_debt_activity": "out/run/accounting/latest/monthly_debt_activity.csv",
    "debt_status_reconciliation": "out/debt_resolution/latest/debt_status_reconciliation.csv",
    "debt_open_items": "out/debt_resolution/latest/debt_open_items.csv",
    "debt_repayment_events": "out/debt_resolution/latest/debt_repayment_events.csv",
    "debt_allocations": "out/debt_resolution/latest/debt_allocations.csv",
    "debt_resolution_timeline": "out/debt_resolution/latest/debt_resolution_timeline.csv",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def find_repo_root(start: str | Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Makefile").exists() and (candidate / "accounting").is_dir():
            return candidate
        if (candidate / "public" / "accounting").exists() and (candidate / "accounting").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find accounting-backend repo root from {here}")


def professional_pack_dir(repo_root: str | Path) -> Path:
    out = Path(repo_root) / "out" / "professional_pack" / "latest"
    for sub in ["", "tables", "qa", "html", "markdown"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def artifact_path(repo_root: str | Path, artifact_key: str) -> Path:
    registry = {**DEFAULT_REQUIRED_ARTIFACTS, **DEFAULT_OPTIONAL_ARTIFACTS}
    if artifact_key not in registry:
        raise KeyError(f"Unknown artifact key: {artifact_key}")
    return Path(repo_root) / registry[artifact_key]


def read_csv_optional(repo_root: str | Path, artifact_key: str, **kwargs) -> pd.DataFrame | None:
    path = artifact_path(repo_root, artifact_key)
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


def safe_read_csv(path: str | Path) -> tuple[pd.DataFrame | None, str]:
    path = Path(path)
    if not path.exists():
        return None, ""
    try:
        return pd.read_csv(path), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def inspect_artifacts(
    repo_root: str | Path,
    required: Mapping[str, str] | None = None,
    optional: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    repo_root = Path(repo_root)
    required = dict(required or DEFAULT_REQUIRED_ARTIFACTS)
    optional = dict(optional or DEFAULT_OPTIONAL_ARTIFACTS)

    rows = []
    for required_flag, artifacts in [(True, required), (False, optional)]:
        for key, rel in artifacts.items():
            path = repo_root / rel
            df, err = safe_read_csv(path)
            rows.append({
                "artifact_key": key,
                "required": required_flag,
                "expected_path": rel,
                "exists": path.exists(),
                "rows": np.nan if df is None else len(df),
                "cols": np.nan if df is None else len(df.columns),
                "columns": "" if df is None else ", ".join(map(str, df.columns[:50])),
                "error": err,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_year_like(x: Any) -> str:
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def detect_value_column(df: pd.DataFrame) -> str:
    candidates = ["value", "metric_value", "amount", "annual_value", "close_amount", "net_amount"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Could not find value column. Available columns: " + ", ".join(map(str, df.columns)))


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]

    value_col = detect_value_column(x)
    if value_col != "value":
        x["value"] = x[value_col]
    x["value"] = pd.to_numeric(x["value"], errors="coerce")

    if "period" not in x.columns:
        raise ValueError("Expected annual metrics to include column 'period'")
    x["period"] = x["period"].map(_normalize_year_like)

    if "Currency" not in x.columns:
        x["Currency"] = "N/A"
    x["Currency"] = x["Currency"].fillna("N/A").astype(str).str.strip()

    defaults = {
        "metric_id": "",
        "dashboard_section": "",
        "dimension_name": "",
        "dimension_value": "",
        "value_status": "",
        "caveat": "",
        "source_table": "",
        "format_hint": "",
        "metric_nature": "",
        "aggregation_policy": "",
    }
    for col, default in defaults.items():
        if col not in x.columns:
            x[col] = default
        x[col] = x[col].fillna(default)

    return x


def load_annual_dashboard_metrics(repo_root: str | Path) -> pd.DataFrame:
    path = artifact_path(repo_root, "annual_dashboard_metrics")
    if not path.exists():
        raise FileNotFoundError(f"Missing annual metrics: {path}")
    return normalize_metrics(pd.read_csv(path))


def load_annual_dashboard_qa(repo_root: str | Path) -> pd.DataFrame:
    path = artifact_path(repo_root, "annual_dashboard_qa")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def available_years(metrics: pd.DataFrame) -> list[str]:
    return sorted(
        metrics.loc[metrics["period"].astype(str).str.match(r"^\d{4}$", na=False), "period"]
        .astype(str).unique().tolist()
    )


def available_currencies(metrics: pd.DataFrame) -> list[str]:
    return sorted(metrics["Currency"].fillna("N/A").astype(str).unique().tolist())


# ---------------------------------------------------------------------------
# Labels and number formatting
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "IS.REVENUE.OPERATING": "Ingresos operativos",
    "IS.RENT.TOTAL": "Renta total",
    "IS.RENT.BY_PROPERTY": "Renta por propiedad",
    "IS.OPEX.PROPERTY": "OPEX propiedad",
    "IS.OPEX.BY_CATEGORY": "OPEX por categoría",
    "IS.NET.OPERATING": "Resultado operativo neto",
    "FUND.CONTRIB.TOTAL": "Funding / aportes",
    "FUND.CONTRIB.BY_ACTOR": "Funding por actor",
    "DIST.DRAWS.PERSONAL": "Retiros y distribuciones totales",
    "DIST.DIVIDENDS": "Dividendos",
    "DIST.DRAWS.BY_TYPE": "Retiros / distribuciones por tipo",
    "COV.NET.AFTER_DRAWS": "Cobertura después de funding y retiros",
    "COV.SAVINGS_RATE": "Savings / coverage rate",
    "BS.CASH.TOTAL": "Caja validada total",
    "BS.CASH.CLOSE.BOX": "Caja validada por box",
    "BS.SECURITY_DEPOSITS.HELD": "Depósitos en garantía retenidos",
    "ID.DEBT.TOTAL.OPEN": "Deuda total abierta",
    "ID.DEBT.PRINCIPAL.OPEN": "Principal abierto",
    "ID.DEBT.INTEREST.OPEN": "Interés abierto",
    "ID.DEBT.OPEN.BY_COUNTERPARTY": "Saldos abiertos por contraparte",
    "ID.DEBT.ACTIVITY.NEW_CLAIMS": "Nuevos claims / adelantos",
    "ID.DEBT.ACTIVITY.INTEREST_ACCRUED": "Intereses devengados",
    "ID.DEBT.ACTIVITY.REPAYMENTS": "Repagos de deuda interna",
    "ID.DEBT.ACTIVITY.ADJUSTMENTS": "Ajustes residuales",
    "ID.DEBT.ACTIVITY.NET_CHANGE": "Cambio neto de deuda",
    "ID.DEBT.NET_PM_POSITION": "Posición neta PM",
    "DQ.CLASSIFICATION.COVERAGE": "Cobertura de clasificación",
    "DQ.UNKNOWN.AMOUNT": "Importe unknown / review-required",
    "DQ.OPEX.LEAKAGE.AMOUNT": "OPEX leakage detectado",
    "DQ.CASH.FRONTEND_SAFE": "Cash frontend-safe",
    "DQ.DEBT.ACTIVITY.RECONCILIATION": "Reconciliación actividad de deuda",
}

DIMENSION_LABELS = {
    "taxes": "Impuestos",
    "services": "Servicios",
    "maintenance": "Mantenimiento",
    "legal": "Legal",
    "other": "Otros",
    "unknown": "Unknown",
    "review_required": "Review required",
    "personal_expense": "Gasto personal",
    "dividend": "Dividendos",
    "transfer_to_family_expense": "Transferencias a gasto familiar",
    "Property Management": "Property Management",
    "Family Business": "Family Business",
    "Household": "Household",
    "MI": "MI",
    "PM": "PM",
    "FB": "FB",
    "Alejandro": "Alejandro",
    "Primos": "Primos",
    "Hector": "Héctor",
    "Inq": "Inquilinos",
}


def humanize_label(value: Any) -> str:
    s = str(value or "").strip()
    if s.lower() in {"nan", "none", ""}:
        return ""
    return DIMENSION_LABELS.get(s, s)



def format_number_es(value, *, format_hint: str = "", metric_id: str = "", missing: str = "s/d") -> str:
    if pd.isna(value):
        return missing

    try:
        v = float(value)
    except Exception:
        return str(value)

    fmt = str(format_hint or "").lower()
    mid = str(metric_id or "").lower()

    if fmt in {"percent_fraction", "ratio_fraction", "rate_fraction"}:
        txt = f"{v * 100:,.1f}%"
    elif fmt in {"percent_points", "ratio_points", "rate_points"}:
        txt = f"{v:,.1f}%"
    elif fmt in {"percent", "ratio", "rate"}:
        # Backward-compatible default: dashboard ratios should preferably migrate
        # to explicit percent_fraction / percent_points.
        txt = f"{v * 100:,.1f}%"
    else:
        txt = f"{v:,.0f}"

    return txt.replace(",", "__TH__").replace(".", ",").replace("__TH__", ".")


def infer_percent(format_hint: Any, metric_id: Any = "") -> bool:
    fmt = str(format_hint or "").lower()
    mid = str(metric_id or "").lower()
    return fmt in {"percent", "ratio", "rate"} or mid.endswith("rate") or "coverage" in mid and "amount" not in mid


# ---------------------------------------------------------------------------
# Inventories and extended QA
# ---------------------------------------------------------------------------

def metric_inventory(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["metric_id", "dashboard_section", "Currency", "dimension_name", "dimension_value"]
    group_cols = [c for c in group_cols if c in metrics.columns]
    out = (
        metrics
        .groupby(group_cols, dropna=False)
        .agg(
            n_rows=("value", "size"),
            n_values=("value", lambda s: s.notna().sum()),
            min_value=("value", "min"),
            max_value=("value", "max"),
            periods=("period", lambda s: ", ".join(sorted(pd.Series(s).dropna().astype(str).unique()))),
            value_status=("value_status", lambda s: ", ".join(sorted({str(x) for x in s if str(x) and str(x) != "nan"}))[:240]),
            source_table=("source_table", lambda s: ", ".join(sorted({str(x) for x in s if str(x) and str(x) != "nan"}))[:240]),
            caveat=("caveat", lambda s: " | ".join([str(x) for x in s if str(x) and str(x) != "nan"][:3])),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    out["metric_label"] = out["metric_id"].map(METRIC_LABELS).fillna(out["metric_id"])
    return out


def dimension_inventory(metrics: pd.DataFrame) -> pd.DataFrame:
    if not {"dimension_name", "dimension_value"}.issubset(metrics.columns):
        return pd.DataFrame()
    return (
        metrics
        .groupby(["dimension_name", "dimension_value"], dropna=False)
        .agg(
            n_metric_ids=("metric_id", lambda s: s.dropna().astype(str).nunique()),
            metric_ids=("metric_id", lambda s: ", ".join(sorted(s.dropna().astype(str).unique())[:30])),
            n_rows=("metric_id", "size"),
            currencies=("Currency", lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))),
            periods=("period", lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))),
        )
        .reset_index()
        .sort_values(["dimension_name", "dimension_value"])
    )


def extended_qa_findings(
    metrics: pd.DataFrame,
    artifact_inventory: pd.DataFrame | None = None,
    debt_status_reconciliation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    findings = []

    def add(severity, area, check, detail, n=None):
        findings.append({"severity": severity, "area": area, "check": check, "n": n, "detail": detail})

    if artifact_inventory is not None and not artifact_inventory.empty:
        missing_required = artifact_inventory[
            artifact_inventory["required"].eq(True) & ~artifact_inventory["exists"].eq(True)
        ]
        add(
            "fail" if len(missing_required) else "ok",
            "artifacts",
            "required_artifacts_exist",
            "All required artifacts exist" if missing_required.empty else "Missing: " + ", ".join(missing_required["artifact_key"].astype(str)),
            len(missing_required),
        )

    available_nan = metrics[
        metrics["value_status"].astype(str).str.lower().eq("available")
        & metrics["value"].isna()
    ]
    add(
        "fail" if len(available_nan) else "ok",
        "metrics",
        "available_metric_has_value",
        "No available metrics with NaN values" if available_nan.empty else f"{len(available_nan)} rows are available but value is NaN",
        len(available_nan),
    )

    suspicious_currency = metrics[
        metrics["Currency"].astype(str).isin(["ALL", "Mixed", "ARS+USD", "MULTI", ""])
    ]
    add(
        "warning" if len(suspicious_currency) else "ok",
        "currency",
        "no_cross_currency_totals",
        "No suspicious cross-currency Currency labels found" if suspicious_currency.empty else f"{len(suspicious_currency)} suspicious currency rows",
        len(suspicious_currency),
    )

    cash_rows = metrics[metrics["metric_id"].astype(str).str.contains(r"BS\.CASH|DQ\.CASH", regex=True, na=False)]
    cash_available = cash_rows[
        cash_rows["value_status"].astype(str).str.lower().eq("available")
        & cash_rows["value"].notna()
    ]
    add(
        "warning" if cash_available.empty else "ok",
        "cash",
        "cash_frontend_safe_available",
        "No available validated cash metric; cash should remain s/d" if cash_available.empty else f"{len(cash_available)} available cash rows",
        len(cash_available),
    )

    # Hidden metric collisions that made early tables look duplicated.
    key_cols = ["dashboard_section", "dimension_name", "dimension_value", "Currency", "period"]
    key_cols = [c for c in key_cols if c in metrics.columns]
    if key_cols:
        collisions = (
            metrics.groupby(key_cols, dropna=False)
            .agg(n_metric_ids=("metric_id", lambda s: s.dropna().astype(str).nunique()))
            .reset_index()
        )
        n = int((collisions["n_metric_ids"] > 1).sum())
        add(
            "warning" if n else "ok",
            "display",
            "hidden_metric_id_collisions",
            "No obvious hidden metric collisions" if n == 0 else f"{n} display key groups map to multiple metric_id values",
            n,
        )

    # Rows marked unavailable should remain visible as caveats, not zero.
    unavailable = metrics[metrics["value_status"].astype(str).str.lower().isin(["unavailable", "blocked", "missing_metric"])]
    add(
        "warning" if len(unavailable) else "ok",
        "metrics",
        "unavailable_visible",
        "No unavailable rows found" if unavailable.empty else f"{len(unavailable)} unavailable/blocked rows should be surfaced as s/d, not zero",
        len(unavailable),
    )

    if debt_status_reconciliation is not None and not debt_status_reconciliation.empty:
        cols = set(debt_status_reconciliation.columns)
        if {"ledger_status", "engine_status", "open_amount"}.issubset(cols):
            closed_open = debt_status_reconciliation[
                debt_status_reconciliation["ledger_status"].astype(str).str.lower().eq("abierto")
                & debt_status_reconciliation["engine_status"].astype(str).str.lower().eq("closed")
                & pd.to_numeric(debt_status_reconciliation["open_amount"], errors="coerce").fillna(1).eq(0)
            ]
            add(
                "warning" if len(closed_open) else "ok",
                "debt",
                "engine_closed_but_ledger_open",
                "No engine-closed/ledger-open rows" if closed_open.empty else f"{len(closed_open)} rows engine closed but ledger still open",
                len(closed_open),
            )
        if {"opened_at", "closed_at"}.issubset(cols):
            d = debt_status_reconciliation.copy()
            d["opened_at_"] = pd.to_datetime(d["opened_at"], errors="coerce")
            d["closed_at_"] = pd.to_datetime(d["closed_at"], errors="coerce")
            closed_before_open = d[d["closed_at_"].notna() & d["opened_at_"].notna() & (d["closed_at_"] < d["opened_at_"])]
            add(
                "fail" if len(closed_before_open) else "ok",
                "debt",
                "closed_before_opened",
                "No closed_at < opened_at rows" if closed_before_open.empty else f"{len(closed_before_open)} rows have closed_at before opened_at",
                len(closed_before_open),
            )

    return pd.DataFrame(findings, columns=["severity", "area", "check", "n", "detail"])


# ---------------------------------------------------------------------------
# Statement builders
# ---------------------------------------------------------------------------

def agg_value(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    # Annual dashboard rows should already be annual values; sum only after grouping
    # equivalent fragments for the same display line/year/currency.
    return float(vals.sum())


def spec(
    statement: str,
    section: str,
    line: str,
    metric_id: str,
    order: int,
    *,
    dimension_name: str | None = None,
    dimension_value: str | None = None,
    append_dimension: bool = True,
    format_hint: str = "number",
    professional_comment: str = "",
    caveat: str = "",
    sign: int = 1,
) -> dict:
    return {
        "statement": statement,
        "section": section,
        "line": line,
        "metric_id": metric_id,
        "order": order,
        "dimension_name": dimension_name,
        "dimension_value": dimension_value,
        "append_dimension": append_dimension,
        "format_hint": format_hint,
        "professional_comment": professional_comment,
        "caveat": caveat,
        "sign": sign,
    }


def build_statement_long(df: pd.DataFrame, specs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for sp in specs:
        x = df.copy()
        metric_id = sp.get("metric_id")
        if metric_id:
            x = x[x["metric_id"].astype(str).eq(str(metric_id))]

        dim_name = sp.get("dimension_name")
        dim_value = sp.get("dimension_value")
        if dim_name is not None:
            x = x[x["dimension_name"].astype(str).eq(str(dim_name))]
        if dim_value is not None:
            x = x[x["dimension_value"].astype(str).eq(str(dim_value))]

        if x.empty:
            rows.append({
                "statement": sp.get("statement", ""),
                "section": sp.get("section", ""),
                "line_order": sp.get("order", 9999),
                "line": sp.get("line", metric_id or "missing metric"),
                "metric_id": metric_id or "",
                "Currency": "s/d",
                "period": "",
                "value": np.nan,
                "value_status": "missing_metric",
                "format_hint": sp.get("format_hint", "number"),
                "professional_comment": sp.get("professional_comment", ""),
                "caveat": sp.get("caveat", ""),
                "source_table": "",
                "dimension_name": dim_name or "",
                "dimension_value": dim_value or "",
            })
            continue

        for _, r in x.iterrows():
            dimension_name = humanize_label(r.get("dimension_name", ""))
            dimension_value = humanize_label(r.get("dimension_value", ""))
            line = sp.get("line", METRIC_LABELS.get(str(metric_id), str(metric_id)))

            if sp.get("append_dimension", True) and dimension_name and dimension_value:
                if dimension_value not in line:
                    line = f"{line} — {dimension_name}: {dimension_value}"

            val = pd.to_numeric(r.get("value", np.nan), errors="coerce")
            sign = sp.get("sign", 1)
            if pd.notna(val):
                val = val * sign

            rows.append({
                "statement": sp.get("statement", ""),
                "section": sp.get("section", ""),
                "line_order": sp.get("order", 9999),
                "line": line,
                "metric_id": metric_id or r.get("metric_id", ""),
                "Currency": r.get("Currency", "N/A"),
                "period": _normalize_year_like(r.get("period", "")),
                "value": val,
                "value_status": r.get("value_status", ""),
                "format_hint": sp.get("format_hint", r.get("format_hint", "number")),
                "professional_comment": sp.get("professional_comment", ""),
                "caveat": r.get("caveat", sp.get("caveat", "")),
                "source_table": r.get("source_table", ""),
                "dimension_name": r.get("dimension_name", ""),
                "dimension_value": r.get("dimension_value", ""),
            })

    return pd.DataFrame(rows)


def build_statement_table(
    df: pd.DataFrame,
    specs: Sequence[Mapping[str, Any]],
    years: Sequence[str] | None = None,
    *,
    include_debug_cols: bool = False,
    drop_all_empty_years: bool = False,
) -> pd.DataFrame:
    years = list(years or available_years(df))
    years = [_normalize_year_like(y) for y in years]

    long = build_statement_long(df, specs)
    if "period" in long.columns:
        long["period"] = long["period"].map(_normalize_year_like)

    id_cols = [
        "statement",
        "section",
        "line_order",
        "line",
        "metric_id",
        "Currency",
        "format_hint",
        "professional_comment",
    ]

    value_long = long[long["period"].astype(str).ne("")].copy()
    if value_long.empty:
        wide = long[id_cols + ["value_status", "caveat", "source_table"]].drop_duplicates()
        for y in years:
            wide[y] = np.nan
    else:
        grouped = (
            value_long
            .groupby(id_cols + ["period"], dropna=False, as_index=False)
            .agg(value=("value", agg_value))
        )
        # Important: no pivot_table(dropna=False), because that can fabricate
        # unobserved cartesian combinations.
        wide = (
            grouped
            .set_index(id_cols + ["period"])["value"]
            .unstack("period")
            .reset_index()
        )
        for y in years:
            if y not in wide.columns:
                wide[y] = np.nan

    meta = (
        long.groupby(id_cols, dropna=False)
        .agg(
            value_status=("value_status", lambda s: ", ".join(sorted({str(x) for x in s if str(x) and str(x) != "nan"}))[:240]),
            caveat=("caveat", lambda s: " | ".join([str(x) for x in s if str(x) and str(x) != "nan"][:3])),
            source_table=("source_table", lambda s: ", ".join(sorted({str(x) for x in s if str(x) and str(x) != "nan"}))[:240]),
            dimension_name=("dimension_name", lambda s: ", ".join(sorted({str(x) for x in s if str(x) and str(x) != "nan"}))[:240]),
            dimension_value=("dimension_value", lambda s: ", ".join(sorted({str(x) for x in s if str(x) and str(x) != "nan"}))[:240]),
        )
        .reset_index()
    )

    wide = wide.merge(meta, on=id_cols, how="left", validate="one_to_one")
    wide = wide.sort_values(["statement", "line_order", "section", "line", "Currency", "metric_id"])

    year_cols = [y for y in years if y in wide.columns]
    if drop_all_empty_years and year_cols:
        keep = ~wide[year_cols].isna().all(axis=1)
        wide = wide[keep].copy()

    display_cols = ["section", "line", "Currency", *years, "value_status", "professional_comment", "caveat", "format_hint"]
    if include_debug_cols:
        display_cols = [
            "section",
            "line",
            "metric_id",
            "dimension_name",
            "dimension_value",
            "Currency",
            *years,
            "value_status",
            "professional_comment",
            "caveat",
            "source_table",
            "format_hint",
        ]
    return wide[[c for c in display_cols if c in wide.columns]].copy()


def add_ratio_row(
    table: pd.DataFrame,
    *,
    section: str,
    line: str,
    numerator_line_contains: str,
    denominator_line_contains: str,
    years: Sequence[str],
    currency: str = "ARS",
    order_after: bool = True,
    professional_comment: str = "",
) -> pd.DataFrame:
    """
    Presentation-layer ratio. It does not write back to metrics.
    Useful for operating margin, OPEX/rent, draws/result.
    """
    out = table.copy()
    mask_num = out["line"].astype(str).str.contains(numerator_line_contains, regex=False, na=False) & out["Currency"].astype(str).eq(currency)
    mask_den = out["line"].astype(str).str.contains(denominator_line_contains, regex=False, na=False) & out["Currency"].astype(str).eq(currency)
    if not mask_num.any() or not mask_den.any():
        return out

    num = out.loc[mask_num, list(years)].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    den = out.loc[mask_den, list(years)].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    vals = num / den.replace(0, np.nan)

    row = {
        "section": section,
        "line": line,
        "Currency": currency,
        **{y: vals.get(y, np.nan) for y in years},
        "value_status": "derived_in_notebook",
        "professional_comment": professional_comment,
        "caveat": "Ratio de presentación calculado desde métricas anuales; no se exporta como métrica core.",
        "format_hint": "percent_fraction",
    }
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)


# ---------------------------------------------------------------------------
# Display and export
# ---------------------------------------------------------------------------

def display_statement(df: pd.DataFrame, title: str, subtitle: str | None = None, *, debug: bool = False):
    try:
        from IPython.display import display, Markdown
        display(Markdown(f"## {title}"))
        if subtitle:
            display(Markdown(subtitle))
    except Exception:
        print(f"\n## {title}\n")
        if subtitle:
            print(subtitle)

    out = df.copy()
    year_cols = [c for c in out.columns if str(c).replace(".0", "").isdigit()]
    rename_years = {c: _normalize_year_like(c) for c in year_cols}
    out = out.rename(columns=rename_years)
    year_cols = [rename_years[c] for c in year_cols]

    for y in year_cols:
        out[y] = out[y].astype("object")

    for idx, row in out.iterrows():
        for y in year_cols:
            # out.at[idx, y] = format_number_es(
            #     row[y],
            #     # percent=infer_percent(row.get("format_hint", ""), row.get("metric_id", "")),
            # )



            out.at[idx, y] = format_number_es(
                row[y],
                format_hint=row.get("format_hint", ""),
                metric_id=row.get("metric_id", ""),
            )



    if not debug:
        drop_cols = {"format_hint"}
        out = out[[c for c in out.columns if c not in drop_cols]]

    try:
        from IPython.display import display
        display(out.style.hide(axis="index"))
    except Exception:
        print(out.to_string(index=False))

    return out


def export_table(df: pd.DataFrame, repo_root: str | Path, filename: str) -> Path:
    out = professional_pack_dir(repo_root) / "tables" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    x = df.copy()
    if max_rows is not None:
        x = x.head(max_rows)
    try:
        return x.to_markdown(index=False)
    except Exception:
        return x.to_csv(index=False)


def write_markdown_report(repo_root: str | Path, filename: str, title: str, sections: Sequence[tuple[str, str | pd.DataFrame]]) -> Path:
    path = professional_pack_dir(repo_root) / "markdown" / filename
    parts = [f"# {title}\n"]
    for heading, content in sections:
        parts.append(f"\n## {heading}\n")
        if isinstance(content, pd.DataFrame):
            parts.append(dataframe_to_markdown(content))
        else:
            parts.append(str(content))
    path.write_text("\n\n".join(parts), encoding="utf-8")
    return path


def write_html_report(repo_root: str | Path, filename: str, title: str, sections: Sequence[tuple[str, str | pd.DataFrame]]) -> Path:
    path = professional_pack_dir(repo_root) / "html" / filename
    css = """
    body { font-family: sans-serif; margin: 32px; line-height: 1.45; }
    table { border-collapse: collapse; margin: 16px 0 28px 0; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
    th { background: #f4f4f4; }
    .note { color: #444; max-width: 900px; }
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>"]
    parts.append(f"<h1>{html.escape(title)}</h1>")
    for heading, content in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        if isinstance(content, pd.DataFrame):
            parts.append(content.to_html(index=False, escape=True))
        else:
            parts.append(f"<div class='note'>{html.escape(str(content)).replace(chr(10), '<br>')}</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def short_status_table(qa: pd.DataFrame) -> pd.DataFrame:
    if qa is None or qa.empty:
        return pd.DataFrame(columns=["severity", "area", "check", "detail"])
    return qa.sort_values(["severity", "area", "check"]).copy()


# ---------------------------------------------------------------------------
# Debt-specific helpers
# ---------------------------------------------------------------------------

def load_debt_status_reconciliation(repo_root: str | Path) -> pd.DataFrame:
    df = read_csv_optional(repo_root, "debt_status_reconciliation")
    return pd.DataFrame() if df is None else df


def debt_action_list(debt_status_reconciliation: pd.DataFrame) -> pd.DataFrame:
    if debt_status_reconciliation is None or debt_status_reconciliation.empty:
        return pd.DataFrame()

    d = debt_status_reconciliation.copy()
    if "open_amount" in d.columns:
        d["open_amount_num"] = pd.to_numeric(d["open_amount"], errors="coerce")
    else:
        d["open_amount_num"] = np.nan

    for c in ["opened_at", "closed_at"]:
        if c in d.columns:
            d[c + "_dt"] = pd.to_datetime(d[c], errors="coerce")

    conditions = []
    if {"ledger_status", "engine_status"}.issubset(d.columns):
        conditions.append(d["ledger_status"].astype(str).str.lower().eq("abierto"))
        conditions.append(d["engine_status"].astype(str).str.lower().eq("closed"))
    if "open_amount_num" in d.columns:
        conditions.append(d["open_amount_num"].fillna(1).eq(0))

    if conditions:
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask & cond
    else:
        mask = pd.Series(False, index=d.index)

    out = d[mask].copy()

    if {"opened_at_dt", "closed_at_dt"}.issubset(out.columns):
        out["chronology_status"] = np.where(out["closed_at_dt"] < out["opened_at_dt"], "investigate_closed_before_opened", "safe_to_mark_closed")
    else:
        out["chronology_status"] = "unknown"

    out["recommended_status"] = np.where(out["chronology_status"].eq("safe_to_mark_closed"), "cerrado", "review_required")
    out["recommendation_reason"] = np.where(
        out["chronology_status"].eq("safe_to_mark_closed"),
        "Engine closed this item with open_amount=0 and chronology is valid.",
        "Engine closed this item but closed_at is before opened_at or chronology is unavailable; do not bulk-edit ledger without review.",
    )

    preferred_cols = [
        "debt_id", "source_tx_id", "opened_at", "closed_at", "debtor", "creditor", "currency",
        "item_type", "original_amount", "open_amount", "ledger_status", "engine_status",
        "chronology_status", "recommended_status", "recommendation_reason",
    ]
    return out[[c for c in preferred_cols if c in out.columns]].copy()


def summarize_debt_reconciliation(debt_status_reconciliation: pd.DataFrame) -> pd.DataFrame:
    if debt_status_reconciliation is None or debt_status_reconciliation.empty:
        return pd.DataFrame(columns=["issue", "n_rows"])

    d = debt_status_reconciliation.copy()
    rows = []

    if {"ledger_status", "engine_status", "open_amount"}.issubset(d.columns):
        open_amount = pd.to_numeric(d["open_amount"], errors="coerce")
        mask = (
            d["ledger_status"].astype(str).str.lower().eq("abierto")
            & d["engine_status"].astype(str).str.lower().eq("closed")
            & open_amount.fillna(1).eq(0)
        )
        rows.append({"issue": "engine_closed_but_ledger_open", "n_rows": int(mask.sum())})

    if {"opened_at", "closed_at"}.issubset(d.columns):
        opened = pd.to_datetime(d["opened_at"], errors="coerce")
        closed = pd.to_datetime(d["closed_at"], errors="coerce")
        mask = opened.notna() & closed.notna() & (closed < opened)
        rows.append({"issue": "closed_at_before_opened_at", "n_rows": int(mask.sum())})

    if "reconciliation_note" in d.columns:
        by_note = d["reconciliation_note"].fillna("").astype(str).value_counts().reset_index()
        by_note.columns = ["issue", "n_rows"]
        rows.extend(by_note.to_dict("records"))

    return pd.DataFrame(rows).drop_duplicates().sort_values(["n_rows", "issue"], ascending=[False, True])
