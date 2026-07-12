from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

CONTRACT_COLUMNS = [
    "metric_id",
    "line_id",
    "dimension_name",
    "dimension_value",
    "funding_channel",
    "funding_actor",
    "cash_effect",
]

YEAR_RE = re.compile(r"^20\d{2}$")
MONTH_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")
    return token[:120] or "line"


def _row_blob(row: pd.Series) -> str:
    fields = [
        "metric_id", "line_id", "metric", "line", "label", "statement_line", "measure",
        "section", "dashboard_section", "dimension_name", "dimension_value", "funding_channel",
        "funding_actor", "cash_effect",
    ]
    return " | ".join(_text(row.get(c)) for c in fields if c in row.index).casefold()


def _first(row: pd.Series, *cols: str) -> str:
    for col in cols:
        value = _text(row.get(col))
        if value:
            return value
    return ""


def _infer_metric_contract(table_id: str, row: pd.Series) -> dict[str, str]:
    existing_metric = _text(row.get("metric_id"))
    existing_dim_name = _text(row.get("dimension_name"))
    existing_dim_value = _text(row.get("dimension_value"))
    existing_channel = _text(row.get("funding_channel"))
    existing_actor = _text(row.get("funding_actor"))
    existing_cash = _text(row.get("cash_effect"))

    blob = _row_blob(row)
    metric = existing_metric
    dim_name = existing_dim_name
    dim_value = existing_dim_value
    channel = existing_channel
    actor = existing_actor
    cash = existing_cash

    if not metric:
        if "funding" in blob or "aporte" in blob or "contrib" in blob or _text(row.get("metric")) == "funding_in":
            metric = "FUND.CONTRIB.TOTAL"
        if "inquil" in blob or re.search(r"\binq\b", blob):
            actor = actor or "Inquilino"
            if "impuesto" in blob or "tax" in blob:
                metric = "FUND.CONTRIB.BY_CHANNEL"
                dim_name = dim_name or "funding_channel"
                dim_value = dim_value or "tenant_direct_tax_payment"
                channel = channel or "tenant_direct_tax_payment"
                cash = cash or "no_cash_in_box_direct_payment"
            elif "servicio" in blob or "service" in blob:
                metric = "FUND.CONTRIB.BY_CHANNEL"
                dim_name = dim_name or "funding_channel"
                dim_value = dim_value or "tenant_direct_service_payment"
                channel = channel or "tenant_direct_service_payment"
                cash = cash or "no_cash_in_box_direct_payment"
            elif "caja" in blob or "box" in blob:
                metric = "FUND.CONTRIB.BY_CHANNEL"
                dim_name = dim_name or "funding_channel"
                dim_value = dim_value or "tenant_to_box"
                channel = channel or "tenant_to_box"
                cash = cash or "cash_in_box"
        for name, canonical in [
            ("mat", "Matías"),
            ("alejandro", "Alejandro"),
            ("alen", "Alejandro"),
            ("primos", "Primos"),
            ("héctor", "Héctor"),
            ("hector", "Héctor"),
        ]:
            if name in blob:
                metric = "FUND.CONTRIB.BY_FUNDING_ACTOR"
                dim_name = dim_name or "funding_actor"
                dim_value = dim_value or canonical
                actor = actor or canonical
                break
        if "household" in blob or re.search(r"\bhh\b", blob):
            metric = "FUND.CONTRIB.BY_CHANNEL"
            dim_name = dim_name or "funding_channel"
            dim_value = dim_value or "household_to_pm"
            channel = channel or "household_to_pm"
            actor = actor or "Household"
        if "direct" in blob and ("obligation" in blob or "impuesto" in blob or "tax" in blob or "servicio" in blob or "service" in blob):
            metric = metric or "FUND.CONTRIB.DIRECT_OBLIGATION"
            cash = cash or "no_cash_in_box_direct_payment"
        if "debt" in blob or "deuda" in blob:
            metric = metric or "FUND.CONTRIB.DEBT_LINKED"

    line_seed = _first(row, "line_id", "metric_id", "metric", "line", "label", "statement_line", "measure") or table_id
    line_id = _text(row.get("line_id")) or _safe_token(f"{table_id}:{line_seed}")

    if metric == "FUND.CONTRIB.BY_CHANNEL" and not dim_name and channel:
        dim_name, dim_value = "funding_channel", channel
    if metric == "FUND.CONTRIB.BY_FUNDING_ACTOR" and not dim_name and actor:
        dim_name, dim_value = "funding_actor", actor
    if metric == "FUND.CONTRIB.BY_CASH_EFFECT" and not dim_name and cash:
        dim_name, dim_value = "cash_effect", cash

    return {
        "metric_id": metric,
        "line_id": line_id,
        "dimension_name": dim_name,
        "dimension_value": dim_value,
        "funding_channel": channel,
        "funding_actor": actor,
        "cash_effect": cash,
    }


def enrich_professional_table(df: pd.DataFrame, table_id: str) -> pd.DataFrame:
    out = df.copy()
    for col in CONTRACT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    if out.empty:
        return out
    inferred = out.apply(lambda row: _infer_metric_contract(table_id, row), axis=1, result_type="expand")
    for col in CONTRACT_COLUMNS:
        current = out[col].fillna("").astype(str).str.strip()
        out[col] = out[col].where(current.ne(""), inferred[col])
    period_cols = [c for c in out.columns if YEAR_RE.match(str(c)) or MONTH_RE.match(str(c))]
    front = [c for c in CONTRACT_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in front and c not in period_cols]
    return out[front + rest + period_cols]


def enrich_professional_table_contracts(tables_dir: Path) -> list[Path]:
    tables_dir = Path(tables_dir)
    if not tables_dir.exists():
        return []
    written: list[Path] = []
    for path in sorted(tables_dir.glob("*.csv")):
        df = pd.read_csv(path)
        enriched = enrich_professional_table(df, path.stem)
        if list(enriched.columns) != list(df.columns) or not enriched.equals(df):
            enriched.to_csv(path, index=False)
            written.append(path)
    return written
