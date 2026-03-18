#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


REPORT_ID = "balance_human_v2"


@dataclass(frozen=True)
class ItemSpec:
    item_id: str
    kind: str
    slug: str
    title: str
    notes: str = ""


ITEMS: List[ItemSpec] = [
    ItemSpec("1.1", "table", "cash_snapshot", "Snapshot de caja"),
    ItemSpec("1.2", "table", "income_statement_monthly_last6", "P&L mensual últimos 6 meses"),
    ItemSpec("1.3", "table", "rent_rollup_by_place_m_last6", "Renta por lugar, caja y moneda"),
    ItemSpec("1.4", "table", "flow_type_rollup_m_last6", "Drilldown por flujo y tipo"),
    ItemSpec("1.5", "table", "data_quality", "Calidad de datos y cobertura"),
]

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
"""

DEFAULT_NOISE_FLOOR = {"ARS": 5000.0, "USD": 10.0}
DEFAULT_INCLUDE_STATUSES = ("pagado",)


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


def _df_to_html_fragment(df: pd.DataFrame, title: str, notes: str = "") -> str:
    if df.empty:
        body = "<p class='warn'>Tabla vacía.</p>"
    else:
        html_df = df.copy()
        for c in html_df.columns:
            if pd.api.types.is_numeric_dtype(html_df[c]):
                html_df[c] = html_df[c].map(_fmt_num)
        body = html_df.to_html(index=False, classes="report-table", border=0, escape=False)
    note_html = f"<p class='small'>{notes}</p>" if notes else ""
    return f"<h2>{title}</h2>\n{note_html}\n{body}\n"


def _manifest_item(spec: ItemSpec, csv_path: Path, html_path: Path) -> Dict[str, Any]:
    return {
        "item_id": spec.item_id,
        "kind": spec.kind,
        "slug": spec.slug,
        "title": spec.title,
        "csv": str(csv_path),
        "html": str(html_path),
    }


def _parse_noise_floor(text: str) -> Dict[str, float]:
    if not text.strip():
        return dict(DEFAULT_NOISE_FLOOR)
    out: Dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        curr, val = part.split(":", 1)
        out[curr.strip().upper()] = float(val.strip())
    return out


# def ensure_metrics_exist(metrics_dir: Path, run_root: Optional[Path], as_of_date: str) -> None:
#     required = [
#         metrics_dir / "metric_registry.csv",
#         metrics_dir / "metric_values.csv",
#         metrics_dir / "validation_report.csv",
#         metrics_dir / "build_manifest.json",
#     ]
#     if all(p.exists() for p in required):
#         return

#     if run_root is None:
#         missing = [str(p) for p in required if not p.exists()]
#         raise FileNotFoundError(
#             "Metrics artifacts missing and no --run-root provided to bootstrap them. Missing: "
#             + ", ".join(missing)
#         )

#     cmd = [
#         "python3",
#         "-m",
#         "accounting.build_metric_values",
#         "--run-root",
#         str(run_root),
#         "--out-dir",
#         str(metrics_dir),
#         "--as-of-date",
#         as_of_date,
#     ]
#     subprocess.run(cmd, check=True)


# def read_metrics_artifacts(metrics_dir: Path) -> Dict[str, Any]:
#     return {
#         "registry": pd.read_csv(metrics_dir / "metric_registry.csv"),
#         "metric_values": pd.read_csv(metrics_dir / "metric_values.csv"),
#         "validation": pd.read_csv(metrics_dir / "validation_report.csv"),
#         "manifest": json.loads((metrics_dir / "build_manifest.json").read_text(encoding="utf-8")),
#     }


def read_metrics_artifacts(metrics_dir: Path) -> Dict[str, Any]:
    required = {
        "registry": metrics_dir / "metric_registry.csv",
        "metric_values": metrics_dir / "metric_values.csv",
        "validation": metrics_dir / "validation_report.csv",
        "manifest": metrics_dir / "build_manifest.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required metrics artifacts in metrics_dir: " + ", ".join(missing)
        )

    return {
        "registry": pd.read_csv(required["registry"]),
        "metric_values": pd.read_csv(required["metric_values"]),
        "validation": pd.read_csv(required["validation"]),
        "manifest": json.loads(required["manifest"].read_text(encoding="utf-8")),
    }

def _infer_run_root(manifest: Dict[str, Any], explicit_run_root: Optional[Path]) -> Path:
    if explicit_run_root is not None:
        return explicit_run_root
    run_root = manifest.get("run_root", "")
    if not run_root:
        raise FileNotFoundError("Could not infer run_root from manifest and none was provided.")
    return Path(run_root)

def load_ledger(run_root: Path) -> pd.DataFrame:
    path = run_root / "ledger_canonical.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ledger file: {path}")

    df = pd.read_csv(path)

    date_col = _resolve_col(df, "Date", ["date", "posted_date"])
    amount_col = _resolve_amount_col(df)
    currency_col = _resolve_col(df, "Currency", ["currency"])
    status_col = _resolve_col(df, "status", ["Status"])
    flujo_col = _resolve_col(df, "Flujo", ["flujo"])
    tipo_col = _resolve_col(df, "Tipo", ["tipo"])

    rename_map = {
        date_col: "Date",
        amount_col: "amount",
        currency_col: "Currency",
        status_col: "status",
        flujo_col: "Flujo",
        tipo_col: "Tipo",
    }

    for optional in ["Box", "Lugar", "Detalle", "payer", "receiver", "medio", "tag"]:
        if optional in df.columns:
            rename_map[optional] = optional

    df = df.rename(columns=rename_map).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["status"] = df["status"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=["Date", "amount", "Currency"]).copy()
    df["period_m"] = df["Date"].dt.to_period("M").astype(str)
    return df
    
    

def _latest_period(metric_values: pd.DataFrame, grain: str) -> Optional[str]:
    vals = sorted(metric_values.loc[metric_values["period_grain"] == grain, "period"].dropna().astype(str).unique().tolist())
    return vals[-1] if vals else None


def _prev_y(period_y: Optional[str]) -> Optional[str]:
    if not period_y:
        return None
    try:
        return str(int(period_y) - 1)
    except Exception:
        return None


def _lookup_metric(metric_values: pd.DataFrame, metric_id: str, grain: str, period: Optional[str]) -> pd.DataFrame:
    if not period:
        return metric_values.iloc[0:0].copy()
    return metric_values.loc[
        (metric_values["metric_id"] == metric_id)
        & (metric_values["period_grain"] == grain)
        & (metric_values["period"] == period)
    ].copy()


def _label_map(registry: pd.DataFrame) -> Dict[str, str]:
    if "label" not in registry.columns:
        return {}
    return dict(zip(registry["metric_id"].astype(str), registry["label"].astype(str)))


def _last_n_months(df: pd.DataFrame, months: int) -> List[str]:
    periods = sorted(df["period_m"].dropna().astype(str).unique().tolist())
    return periods[-months:]


def _apply_noise_floor_rows(df: pd.DataFrame, total_col: str, currency_col: str, noise_floor_by_currency: Dict[str, float]) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    def keep_row(row: pd.Series) -> bool:
        curr = str(row.get(currency_col, "")).upper()
        thr = noise_floor_by_currency.get(curr, None)
        if thr is None:
            return True
        val = row.get(total_col, pd.NA)
        if pd.isna(val):
            return True
        return abs(float(val)) >= thr
    mask = work.apply(keep_row, axis=1)
    return work.loc[mask].reset_index(drop=True)


def build_cash_snapshot(registry: pd.DataFrame, metric_values: pd.DataFrame) -> pd.DataFrame:
    metric_ids = ["BS.CASH.FB", "BS.CASH.PM", "BS.CASH.TOTAL"]
    current_y = _latest_period(metric_values, "Y")
    prev_y = _prev_y(current_y)
    label_map = _label_map(registry)

    rows = []
    for metric_id in metric_ids:
        cur = _lookup_metric(metric_values, metric_id, "Y", current_y)
        prv = _lookup_metric(metric_values, metric_id, "Y", prev_y)
        currencies = sorted(set(cur["currency"].astype(str)) | set(prv["currency"].astype(str))) or [""]
        for currency in currencies:
            c = cur.loc[cur["currency"].astype(str) == currency]
            p = prv.loc[prv["currency"].astype(str) == currency]
            cur_val = c["value"].iloc[0] if not c.empty else pd.NA
            prev_val = p["value"].iloc[0] if not p.empty else pd.NA
            delta = (cur_val - prev_val) if (pd.notna(cur_val) and pd.notna(prev_val)) else pd.NA
            rows.append({
                "metric_id": metric_id,
                "label": label_map.get(metric_id, metric_id),
                "currency": currency,
                "period": current_y or "",
                "value": cur_val,
                "prev_y": prev_val,
                "delta_vs_prev_y": delta,
            })
    return pd.DataFrame(rows)


def build_data_quality(registry: pd.DataFrame, metric_values: pd.DataFrame, validation: pd.DataFrame, manifest: Dict[str, Any]) -> pd.DataFrame:
    active_leaf = registry.loc[
        registry.get("is_leaf", pd.Series(False, index=registry.index)).astype(bool)
        & (registry.get("status", pd.Series("active", index=registry.index)).astype(str) == "active")
    ].copy()
    built_metric_ids = set(metric_values["metric_id"].astype(str).tolist())
    missing_leaf = active_leaf.loc[~active_leaf["metric_id"].astype(str).isin(built_metric_ids)]

    errors = validation.loc[validation.get("level", pd.Series("", index=validation.index)).astype(str).str.lower() == "error"]
    warnings = validation.loc[validation.get("level", pd.Series("", index=validation.index)).astype(str).str.lower() == "warning"]

    rows = [
        {"check_name": "registry_rows", "value": int(len(registry)), "status": "ok", "detail": ""},
        {"check_name": "metric_values_rows", "value": int(len(metric_values)), "status": "ok", "detail": ""},
        {"check_name": "validation_errors", "value": int(len(errors)), "status": "error" if len(errors) else "ok", "detail": "; ".join(errors.get("check_name", pd.Series(dtype=str)).astype(str).tolist())},
        {"check_name": "validation_warnings", "value": int(len(warnings)), "status": "warning" if len(warnings) else "ok", "detail": "; ".join(warnings.get("check_name", pd.Series(dtype=str)).astype(str).tolist())},
        {"check_name": "missing_active_leaf_metrics", "value": int(len(missing_leaf)), "status": "warning" if len(missing_leaf) else "ok", "detail": ", ".join(missing_leaf["metric_id"].astype(str).tolist())},
        {"check_name": "source_run_root", "value": manifest.get("run_root", ""), "status": "ok", "detail": ""},
        {"check_name": "source_run_id", "value": manifest.get("run_id", ""), "status": "ok", "detail": ""},
        {"check_name": "as_of_date", "value": manifest.get("as_of_date", ""), "status": "ok", "detail": ""},
    ]
    return pd.DataFrame(rows)


def build_flow_rollup_last_n_months(
    ledger: pd.DataFrame,
    *,
    flow_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    groupby_cols: Sequence[str],
    months: int = 6,
    include_statuses: Sequence[str] = DEFAULT_INCLUDE_STATUSES,
    amount_col: str = "amount",
    currency_col: str = "Currency",
    status_col: str = "status",
    noise_floor_by_currency: Optional[Dict[str, float]] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    work = ledger.copy()

    if flow_filter is not None:
        work = work.loc[work["Flujo"].astype(str) == flow_filter].copy()
    if type_filter is not None:
        work = work.loc[work["Tipo"].astype(str) == type_filter].copy()
    if include_statuses:
        allowed = {str(x).strip().lower() for x in include_statuses}
        work = work.loc[work[status_col].astype(str).str.strip().str.lower().isin(allowed)].copy()

    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
    work = work.dropna(subset=[amount_col, currency_col]).copy()

    months_list = _last_n_months(work, months)
    if months_list:
        work = work.loc[work["period_m"].isin(months_list)].copy()

    missing_cols = [c for c in groupby_cols if c not in work.columns]
    if missing_cols:
        raise ValueError(f"Missing groupby columns in ledger: {missing_cols}")

    work[groupby_cols] = work[groupby_cols].fillna("").astype(str)

    base_group_cols = []
    for c in list(groupby_cols):
        if c not in base_group_cols:
            base_group_cols.append(c)

    if currency_col not in base_group_cols:
        base_group_cols.append(currency_col)

    group_cols = base_group_cols + ["period_m"]

    grouped = (
        work.groupby(group_cols, dropna=False)[amount_col]
        .sum()
        .reset_index()
    )

    if grouped.empty:
        cols = list(groupby_cols) + [currency_col] + months_list + ["total_6m", "avg_m", "last_m", "delta_last_vs_prev"]
        return pd.DataFrame(columns=cols)

    wide = (
        grouped.pivot_table(
            index=base_group_cols,
            columns="period_m",
            values=amount_col,
            aggfunc="sum",
            fill_value=0.0,
            )
            .reset_index()
        )

    for m in months_list:
        if m not in wide.columns:
            wide[m] = 0.0

    wide["total_6m"] = wide[months_list].sum(axis=1) if months_list else 0.0
    wide["avg_m"] = wide["total_6m"] / max(len(months_list), 1)
    wide["last_m"] = wide[months_list[-1]] if months_list else 0.0
    wide["delta_last_vs_prev"] = (
        wide[months_list[-1]] - wide[months_list[-2]]
        if len(months_list) >= 2
        else pd.NA
    )

    if noise_floor_by_currency:
        wide = _apply_noise_floor_rows(wide, "total_6m", currency_col, noise_floor_by_currency)

    sort_cols = ["total_6m"]
    wide = wide.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    if top_n is not None and len(wide) > top_n:
        wide = wide.head(top_n).reset_index(drop=True)

    # ordered_cols = list(groupby_cols) + [currency_col] + months_list + ["total_6m", "avg_m", "last_m", "delta_last_vs_prev"]
    ordered_cols = base_group_cols + months_list + ["total_6m", "avg_m", "last_m", "delta_last_vs_prev"]
    return wide[ordered_cols]


def build_income_statement_monthly_last6(
    ledger: pd.DataFrame,
    *,
    months: int,
    include_statuses: Sequence[str],
) -> pd.DataFrame:
    work = ledger.copy()
    allowed = {str(x).strip().lower() for x in include_statuses}
    work = work.loc[work["status"].astype(str).str.strip().str.lower().isin(allowed)].copy()
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work = work.dropna(subset=["amount", "Currency"]).copy()

    months_list = _last_n_months(work, months)
    work = work.loc[work["period_m"].isin(months_list)].copy()

    specs = [
        ("IS.RENT.TOTAL", "Renta total", (work["Flujo"].astype(str) == "Cobros") & (work["Tipo"].astype(str) == "Renta")),
        ("IS.CONTRIB.TOTAL", "Contribuciones totales", work["Flujo"].astype(str) == "Contribucion"),
        ("IS.OPEX.TOTAL", "Costos operativos totales", work["Flujo"].astype(str) == "Pagos"),
    ]

    rows = []
    for metric_id, label, mask in specs:
        sub = work.loc[mask].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby(["Currency", "period_m"], dropna=False)["amount"]
            .sum()
            .reset_index()
        )
        wide = (
            agg.pivot_table(index=["Currency"], columns="period_m", values="amount", aggfunc="sum", fill_value=0.0)
            .reset_index()
        )
        for m in months_list:
            if m not in wide.columns:
                wide[m] = 0.0
        for _, row in wide.iterrows():
            out = {
                "metric_id": metric_id,
                "label": label,
                "currency": row["Currency"],
            }
            vals = [row[m] for m in months_list]
            for m in months_list:
                out[m] = row[m]
            out["total_6m"] = sum(vals)
            out["avg_m"] = out["total_6m"] / max(len(months_list), 1)
            rows.append(out)

    df = pd.DataFrame(rows)

    # derived rows
    if not df.empty:
        derived_rows = []
        for currency in sorted(df["currency"].astype(str).unique().tolist()):
            sub = df.loc[df["currency"].astype(str) == currency].set_index("metric_id")
            rent = sub.loc["IS.RENT.TOTAL"] if "IS.RENT.TOTAL" in sub.index else None
            contrib = sub.loc["IS.CONTRIB.TOTAL"] if "IS.CONTRIB.TOTAL" in sub.index else None
            opex = sub.loc["IS.OPEX.TOTAL"] if "IS.OPEX.TOTAL" in sub.index else None

            if rent is not None or contrib is not None:
                out = {"metric_id": "IS.INCOME.TOTAL", "label": "Ingresos totales", "currency": currency}
                for m in months_list:
                    out[m] = (rent[m] if rent is not None else 0.0) + (contrib[m] if contrib is not None else 0.0)
                out["total_6m"] = sum(out[m] for m in months_list)
                out["avg_m"] = out["total_6m"] / max(len(months_list), 1)
                derived_rows.append(out)

            if (rent is not None or contrib is not None) and opex is not None:
                base = derived_rows[-1] if derived_rows else None
                if base and base["metric_id"] == "IS.INCOME.TOTAL":
                    out = {"metric_id": "IS.NET.AFTER_COSTS", "label": "Neto después de costos", "currency": currency}
                    for m in months_list:
                        out[m] = base[m] - opex[m]
                    out["total_6m"] = sum(out[m] for m in months_list)
                    out["avg_m"] = out["total_6m"] / max(len(months_list), 1)
                    derived_rows.append(out)

        if derived_rows:
            df = pd.concat([df, pd.DataFrame(derived_rows)], ignore_index=True)

    ordered_metric_ids = [
        "IS.RENT.TOTAL",
        "IS.CONTRIB.TOTAL",
        "IS.INCOME.TOTAL",
        "IS.OPEX.TOTAL",
        "IS.NET.AFTER_COSTS",
    ]
    df["__sort"] = df["metric_id"].map({k: i for i, k in enumerate(ordered_metric_ids)})
    cols = ["metric_id", "label", "currency"] + months_list + ["total_6m", "avg_m"]
    return df.sort_values(["currency", "__sort"]).reset_index(drop=True)[cols]


def build_draws_discipline_monthly_last6(
    ledger: pd.DataFrame,
    *,
    months: int,
    include_statuses: Sequence[str],
    noise_floor_by_currency: Dict[str, float],
) -> pd.DataFrame:
    work = ledger.copy()
    allowed = {str(x).strip().lower() for x in include_statuses}
    work = work.loc[work["status"].astype(str).str.strip().str.lower().isin(allowed)].copy()
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work = work.dropna(subset=["amount", "Currency"]).copy()
    months_list = _last_n_months(work, months)
    work = work.loc[work["period_m"].isin(months_list)].copy()

    text_cols = [c for c in ["Tipo", "Detalle", "tag", "Lugar"] if c in work.columns]
    mask = pd.Series(False, index=work.index)
    for c in text_cols:
        mask = mask | work[c].astype(str).str.contains(r"personal|retiro|draw|owner|dividend", case=False, na=False)

    draws = (
        work.loc[mask]
        .groupby(["Currency", "period_m"], dropna=False)["amount"]
        .sum()
        .reset_index()
    )
    net = build_income_statement_monthly_last6(work, months=months, include_statuses=include_statuses)
    net = net.loc[net["metric_id"] == "IS.NET.AFTER_COSTS"].copy()

    rows = []
    for currency in sorted(set(draws["Currency"].astype(str)) | set(net["currency"].astype(str))):
        dsub = draws.loc[draws["Currency"].astype(str) == currency]
        nsub = net.loc[net["currency"].astype(str) == currency]
        row = {"currency": currency}
        total_draws = 0.0
        distress = 0
        for m in months_list:
            d = dsub.loc[dsub["period_m"].astype(str) == m, "amount"]
            n = nsub[m].iloc[0] if (not nsub.empty and m in nsub.columns) else pd.NA
            draw_val = float(d.iloc[0]) if not d.empty else 0.0
            row[f"draws_{m}"] = draw_val
            row[f"net_{m}"] = n
            if pd.notna(n) and float(n) <= 0 and draw_val > 0:
                distress += 1
            total_draws += draw_val
        row["draws_total_6m"] = total_draws
        row["distress_months"] = distress
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _apply_noise_floor_rows(df, "draws_total_6m", "currency", noise_floor_by_currency)
    return df


def build_summary_kpis(cash_snapshot: pd.DataFrame, income_statement_m: pd.DataFrame, draws_discipline_m: pd.DataFrame) -> List[Dict[str, str]]:
    def _pick_cash() -> Dict[str, str]:
        sub = cash_snapshot.loc[cash_snapshot["metric_id"] == "BS.CASH.TOTAL"]
        if sub.empty:
            return {"label": "Caja total", "value": "N/A"}
        row = sub.iloc[0]
        return {"label": f"Caja total [{row.get('currency','')}]", "value": _fmt_num(row.get("value", pd.NA))}

    def _pick_metric(metric_id: str, label: str) -> Dict[str, str]:
        sub = income_statement_m.loc[income_statement_m["metric_id"] == metric_id]
        if sub.empty:
            return {"label": label, "value": "N/A"}
        row = sub.iloc[0]
        last_cols = [c for c in sub.columns if c.startswith("20")]
        val = row[last_cols[-1]] if last_cols else pd.NA
        return {"label": f"{label} [{row.get('currency','')}]", "value": _fmt_num(val)}

    kpis = [
        _pick_cash(),
        _pick_metric("IS.NET.AFTER_COSTS", "Neto después de costos"),
        _pick_metric("IS.OPEX.TOTAL", "Opex"),
    ]
    if not draws_discipline_m.empty:
        row = draws_discipline_m.iloc[0]
        kpis.append({"label": f"Meses en distress [{row.get('currency','')}]", "value": str(int(row.get("distress_months", 0)))})
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
    arts = read_metrics_artifacts(metrics_dir)
    registry = arts["registry"]
    metric_values = arts["metric_values"]
    validation = arts["validation"]
    manifest_in = arts["manifest"]
    ledger = load_ledger(run_root)

    cash_snapshot = build_cash_snapshot(registry, metric_values)
    income_statement_monthly_last6 = build_income_statement_monthly_last6(
        ledger, months=months, include_statuses=include_statuses
    )
    rent_rollup_by_place = build_flow_rollup_last_n_months(
        ledger,
        flow_filter="Cobros",
        type_filter="Renta",
        groupby_cols=["Box", "Currency", rent_place_col],
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    )
    flow_type_rollup = build_flow_rollup_last_n_months(
        ledger,
        groupby_cols=list(flow_rollup_groupby),
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
        top_n=12,
    )
    data_quality = build_data_quality(registry, metric_values, validation, manifest_in)
    draws_discipline_monthly = build_draws_discipline_monthly_last6(
        ledger,
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    )

    tables = {
        "cash_snapshot": cash_snapshot,
        "income_statement_monthly_last6": income_statement_monthly_last6,
        "rent_rollup_by_place_m_last6": rent_rollup_by_place,
        "flow_type_rollup_m_last6": flow_type_rollup,
        "data_quality": data_quality,
    }
    item_by_slug = {x.slug: x for x in ITEMS}

    dirs = _ensure_dirs(write_dir)
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

    for slug, df in tables.items():
        spec = item_by_slug[slug]
        base = f"{spec.item_id}__{spec.slug}"
        csv_path = dirs["tables"] / f"{base}.csv"
        html_path = dirs["html"] / f"{base}.html"
        _write_csv(df, csv_path)
        _write_text(_df_to_html_fragment(df, spec.title, spec.notes), html_path)
        manifest["items"].append(_manifest_item(spec, csv_path, html_path))

    kpis = build_summary_kpis(cash_snapshot, income_statement_monthly_last6, draws_discipline_monthly)
    kpi_html = "\n".join(
        f"<div class='kpi'><div class='label'>{x['label']}</div><div class='value'>{x['value']}</div></div>"
        for x in kpis
    )

    standalone_sections = [
        "<h1>Balance humano v2</h1>",
        f"<p>run_id: {manifest_in.get('run_id', '')}<br>as_of_date: {manifest_in.get('as_of_date', '')}<br>run_root: {run_root}<br>months: {months}</p>",
        f"<div class='kpi-grid'>{kpi_html}</div>",
        _df_to_html_fragment(cash_snapshot, "Snapshot de caja"),
        _df_to_html_fragment(income_statement_monthly_last6, f"P&L mensual últimos {months} meses"),
        _df_to_html_fragment(rent_rollup_by_place, "Renta por lugar, caja y moneda", f"groupby = Box, Currency, {rent_place_col}"),
        _df_to_html_fragment(flow_type_rollup, "Drilldown por flujo y tipo", f"groupby = {', '.join(flow_rollup_groupby)}"),
        _df_to_html_fragment(draws_discipline_monthly, "Retiros y disciplina, últimos 6 meses"),
        _df_to_html_fragment(data_quality, "Calidad de datos y cobertura"),
    ]
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

    # useful extra drilldown not in manifest
    rent_rollup_by_detail = build_flow_rollup_last_n_months(
        ledger,
        flow_filter="Cobros",
        type_filter="Renta",
        groupby_cols=["Box", "Currency", rent_detail_col],
        months=months,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    )
    _write_csv(rent_rollup_by_detail, dirs["tables"] / "extra__rent_rollup_by_detail_m_last6.csv")
    _write_text(
        _df_to_html_fragment(
            rent_rollup_by_detail,
            "Renta por detalle, caja y moneda",
            f"groupby = Box, Currency, {rent_detail_col}",
        ),
        dirs["html"] / "extra__rent_rollup_by_detail_m_last6.html",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a human-oriented balance storypack v2 from metrics artifacts and ledger.")
    p.add_argument("--run-root", required=True, help="Accounting run root containing ledger_canonical.csv")
    p.add_argument("--metrics-dir", required=True, help="Directory with metric_registry.csv / metric_values.csv etc.")
    p.add_argument("--write-dir", required=True, help="Output storypack directory.")
    # p.add_argument("--as-of-date", default=pd.Timestamp.today().date().isoformat(), help="Used if metrics bootstrap is needed.")
    p.add_argument("--months", type=int, default=6, help="Number of monthly periods to surface.")
    p.add_argument("--rent-place-col", default="Lugar", help="Column used for rent rollup by place.")
    p.add_argument("--rent-detail-col", default="Detalle", help="Column used for rent rollup by detail.")
    p.add_argument("--flow-rollup-groupby", default="Flujo,Tipo,Currency", help="Comma-separated groupby columns for generic drilldown.")
    p.add_argument("--include-statuses", default="pagado", help="Comma-separated statuses to include, e.g. pagado or pagado,planeado.")
    p.add_argument("--noise-floor", default="ARS:5000,USD:10", help="Comma-separated thresholds, e.g. ARS:5000,USD:10")
    return p.parse_args()



def _resolve_amount_col(df: pd.DataFrame) -> str:
    candidates = ["amount", "monto", "Amount", "Debit", "Credit"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No amount-like column found. Available columns: {list(df.columns)}")


def _resolve_col(df: pd.DataFrame, preferred: str, aliases: list[str]) -> str:
    if preferred in df.columns:
        return preferred
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(f"Missing required column '{preferred}'. Available columns: {list(df.columns)}")
    
    
def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    write_dir = Path(args.write_dir)
    run_root = Path(args.run_root) if args.run_root else None
    include_statuses = tuple(x.strip() for x in args.include_statuses.split(",") if x.strip())
    noise_floor_by_currency = _parse_noise_floor(args.noise_floor)
    flow_rollup_groupby = [x.strip() for x in args.flow_rollup_groupby.split(",") if x.strip()]

    # ensure_metrics_exist(metrics_dir=metrics_dir, run_root=run_root, as_of_date=args.as_of_date)
    arts = read_metrics_artifacts(metrics_dir)
    # inferred_run_root = _infer_run_root(arts["manifest"], run_root)
    inferred_run_root = Path(args.run_root)

    build_human_balance_report(
        metrics_dir=metrics_dir,
        write_dir=write_dir,
        run_root=inferred_run_root,
        months=args.months,
        rent_place_col=args.rent_place_col,
        rent_detail_col=args.rent_detail_col,
        flow_rollup_groupby=flow_rollup_groupby,
        include_statuses=include_statuses,
        noise_floor_by_currency=noise_floor_by_currency,
    )

    print("=== BUILD COMPLETE ===")
    print("metrics_dir:", metrics_dir)
    print("run_root:", inferred_run_root)
    print("write_dir:", write_dir)
    print("story_manifest:", write_dir / "story_manifest.json")
    print("standalone_html:", write_dir / "balance_humano_v2.html")


if __name__ == "__main__":
    main()
