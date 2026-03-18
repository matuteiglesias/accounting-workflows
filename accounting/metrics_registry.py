from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import pandas as pd


REGISTRY_COLUMNS = [
    "metric_id",
    "statement",
    "section",
    "label",
    "agg_rule",
    "is_leaf",
    "source_layer",
    "builder_key",
    "parent_metric_id",
    "display_code",
    "sort_key",
    "currency_mode",
    "status",
    "notes",
]


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    statement: str
    section: str
    label: str
    agg_rule: str
    is_leaf: bool
    source_layer: str = ""
    builder_key: str = ""
    parent_metric_id: str = ""
    display_code: str = ""
    sort_key: int = 0
    currency_mode: str = "by_currency"
    status: str = "active"
    notes: str = ""

    def to_record(self) -> dict:
        return asdict(self)


def registry_from_specs(specs: Iterable[MetricSpec]) -> pd.DataFrame:
    rows = [s.to_record() for s in specs]
    df = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    return normalize_registry(df)


def normalize_registry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in REGISTRY_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out["metric_id"] = out["metric_id"].astype(str).str.strip()
    out["statement"] = out["statement"].astype(str).str.strip()
    out["section"] = out["section"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    out["agg_rule"] = out["agg_rule"].astype(str).str.strip()
    out["source_layer"] = out["source_layer"].astype(str).str.strip()
    out["builder_key"] = out["builder_key"].astype(str).str.strip()
    out["parent_metric_id"] = out["parent_metric_id"].astype(str).str.strip()
    out["display_code"] = out["display_code"].astype(str).str.strip()
    out["currency_mode"] = out["currency_mode"].astype(str).str.strip().replace("", "by_currency")
    out["status"] = out["status"].astype(str).str.strip().replace("", "active")
    out["notes"] = out["notes"].astype(str)

    if "is_leaf" in out.columns:
        out["is_leaf"] = out["is_leaf"].astype(bool)
    else:
        out["is_leaf"] = False

    out["sort_key"] = pd.to_numeric(out["sort_key"], errors="coerce").fillna(0).astype(int)

    out = out[REGISTRY_COLUMNS].sort_values(["sort_key", "metric_id"]).reset_index(drop=True)
    return out


def load_registry(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported registry format for path: {path}")
    return normalize_registry(df)


def save_registry(df: pd.DataFrame, path: str) -> None:
    df = normalize_registry(df)
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported registry format for path: {path}")


def get_metric_spec(registry_df: pd.DataFrame, metric_id: str) -> pd.Series:
    m = registry_df.loc[registry_df["metric_id"] == metric_id]
    if m.empty:
        raise KeyError(f"metric_id not found in registry: {metric_id}")
    if len(m) > 1:
        raise ValueError(f"metric_id appears multiple times in registry: {metric_id}")
    return m.iloc[0]


def list_metric_ids(
    registry_df: pd.DataFrame,
    *,
    statement: Optional[str] = None,
    is_leaf: Optional[bool] = None,
    status: str = "active",
) -> list[str]:
    df = normalize_registry(registry_df)

    if statement is not None:
        df = df.loc[df["statement"] == statement]
    if is_leaf is not None:
        df = df.loc[df["is_leaf"] == is_leaf]
    if status:
        df = df.loc[df["status"] == status]

    return df["metric_id"].tolist()


def default_metric_specs_v1() -> list[MetricSpec]:
    return [
        MetricSpec("IS.RENT.TOTAL", "IS", "RENT", "Renta total", "sum_components", False, "", "", "IS.INCOME.TOTAL", "IS.1", 100),
        MetricSpec("IS.CONTRIB.TOTAL", "IS", "CONTRIB", "Contribuciones totales", "sum", True, "v_contributions_monthly", "build_is_contrib_total", "IS.INCOME.TOTAL", "IS.2", 200),
        MetricSpec("IS.INCOME.TOTAL", "IS", "INCOME", "Ingresos totales", "sum_components", False, "", "", "", "IS.3", 300),
        MetricSpec("IS.OPEX.TOTAL", "IS", "OPEX", "Costos operativos totales", "sum", True, "v_opex_category_monthly", "build_is_opex_total", "IS.NET.AFTER_COSTS", "IS.4", 400),
        MetricSpec("IS.NET.AFTER_COSTS", "IS", "RESULT", "Neto después de costos", "formula", False, "", "", "", "IS.5", 500),
        MetricSpec("IS.DRAWS.PERSONAL", "IS", "DRAWS", "Retiros personales", "sum", True, "ledger", "build_is_draws_personal", "IS.NET.POST_DRAWS", "IS.6", 600),
        MetricSpec("IS.NET.POST_DRAWS", "IS", "RESULT", "Neto después de retiros", "formula", False, "", "", "", "IS.7", 700),
        MetricSpec("BS.CASH.FB", "BS", "CASH", "Fondos FB al cierre", "last", True, "daily_cash_position", "build_bs_cash_fb", "BS.CASH.TOTAL", "BS.1.1", 810),
        MetricSpec("BS.CASH.PM", "BS", "CASH", "Fondos PM al cierre", "last", True, "daily_cash_position", "build_bs_cash_pm", "BS.CASH.TOTAL", "BS.1.2", 820),
        MetricSpec("BS.CASH.TOTAL", "BS", "CASH", "Activos líquidos totales", "sum_components", False, "", "", "BS.ASSETS.TOTAL", "BS.1", 800),
    ]
