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
    "metric_type",
    "economic_role",
    "namespace_target",
    "migration_status",
    "legacy_warning",
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
    metric_type: str = "unknown"
    economic_role: str = "unknown"
    namespace_target: str = "UNKNOWN"
    migration_status: str = "investigate"
    legacy_warning: str = ""

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
    out["metric_type"] = out["metric_type"].astype(str).str.strip().str.lower().replace("", "unknown")
    out["economic_role"] = out["economic_role"].astype(str).str.strip().str.lower().replace("", "unknown")
    out["namespace_target"] = out["namespace_target"].astype(str).str.strip().str.upper().replace("", "UNKNOWN")
    out["migration_status"] = out["migration_status"].astype(str).str.strip().str.lower().replace("", "investigate")
    out["legacy_warning"] = out["legacy_warning"].astype(str)

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
        # MetricSpec("IS.RENT.CABA", "IS", "RENT", "Renta CABA", "sum", True, "per_flow", "build_is_rent_caba", "IS.RENT.TOTAL", "IS.1.1", 110),
        # MetricSpec("IS.RENT.TORCUATO", "IS", "RENT", "Renta Torcuato", "sum", True, "per_flow", "build_is_rent_torcuato", "IS.RENT.TOTAL", "IS.1.2", 120),
        MetricSpec("IS.RENT.TOTAL", "IS", "RENT", "Renta total", "sum_components", False, "", "", "IS.INCOME.TOTAL", "IS.1", 100, metric_type="flow", economic_role="operating", namespace_target="IS", migration_status="keep"),
        MetricSpec("IS.REVENUE.TOTAL", "IS", "REVENUE", "Ingresos operativos totales", "sum_components", False, "", "", "IS.NET.OPERATING", "IS.1.5", 150, metric_type="derived", economic_role="operating", namespace_target="IS", migration_status="create", notes="Shadow metric: clean operating revenue currently aliases IS.RENT.TOTAL."),
        MetricSpec("IS.CONTRIB.TOTAL", "IS", "CONTRIB", "Contribuciones totales", "sum", True, "v_contributions_monthly", "build_is_contrib_total", "IS.INCOME.TOTAL", "IS.2", 200, metric_type="flow", economic_role="funding", namespace_target="FUND", migration_status="alias", legacy_warning="Currently lives under IS but semantically represents funding/contributions."),
        MetricSpec("FUND.CONTRIB.TOTAL", "FUND", "CONTRIB", "Contribuciones familiares totales", "sum_components", False, "", "", "COV.NET.AFTER_DRAWS", "FUND.1", 210, metric_type="derived", economic_role="funding", namespace_target="FUND", migration_status="create", notes="Shadow metric alias of legacy IS.CONTRIB.TOTAL."),
        MetricSpec("IS.INCOME.TOTAL", "IS", "INCOME", "Ingresos totales", "sum_components", False, "", "", "", "IS.3", 300, metric_type="derived", economic_role="coverage", namespace_target="COV", migration_status="legacy", legacy_warning="Mixes operating rent with family contributions."),
        MetricSpec("IS.OPEX.TOTAL", "IS", "OPEX", "Costos operativos totales", "sum", True, "v_opex_category_monthly", "build_is_opex_total", "IS.NET.AFTER_COSTS", "IS.4", 400, metric_type="flow", economic_role="operating", namespace_target="IS", migration_status="keep"),
        MetricSpec("IS.NET.OPERATING", "IS", "RESULT", "Resultado operativo neto", "formula", False, "", "", "COV.NET.AFTER_DRAWS", "IS.4.5", 450, metric_type="derived", economic_role="operating", namespace_target="IS", migration_status="create", notes="Shadow metric: IS.REVENUE.TOTAL - IS.OPEX.TOTAL."),
        MetricSpec("IS.NET.AFTER_COSTS", "IS", "RESULT", "Neto después de costos", "formula", False, "", "", "", "IS.5", 500, metric_type="derived", economic_role="coverage", namespace_target="COV", migration_status="legacy", legacy_warning="Currently depends on IS.INCOME.TOTAL, which includes contributions."),
        MetricSpec("IS.DRAWS.PERSONAL", "IS", "DRAWS", "Retiros personales", "sum", True, "ledger", "build_is_draws_personal", "IS.NET.POST_DRAWS", "IS.6", 600, metric_type="flow", economic_role="distribution", namespace_target="DIST", migration_status="alias", legacy_warning="Currently lives under IS but semantically represents draws/distributions."),
        MetricSpec("DIST.DRAWS.PERSONAL", "DIST", "DRAWS", "Retiros personales", "sum_components", False, "", "", "COV.NET.AFTER_DRAWS", "DIST.1", 610, metric_type="derived", economic_role="distribution", namespace_target="DIST", migration_status="create", notes="Shadow metric alias of legacy IS.DRAWS.PERSONAL."),
        MetricSpec("IS.NET.POST_DRAWS", "IS", "RESULT", "Neto después de retiros", "formula", False, "", "", "", "IS.7", 700, metric_type="derived", economic_role="coverage", namespace_target="COV", migration_status="legacy", legacy_warning="Legacy coverage metric after draws; depends on funding-inclusive IS.NET.AFTER_COSTS."),
        MetricSpec("COV.NET.AFTER_DRAWS", "COV", "COVERAGE", "Cobertura neta después de aportes y retiros", "formula", False, "", "", "", "COV.1", 710, metric_type="derived", economic_role="coverage", namespace_target="COV", migration_status="create", notes="Shadow coverage metric: IS.NET.OPERATING + FUND.CONTRIB.TOTAL - DIST.DRAWS.PERSONAL."),
        MetricSpec("BS.CASH.FB", "BS", "CASH", "Fondos FB al cierre", "last", True, "daily_cash_position", "build_bs_cash_fb", "BS.CASH.TOTAL", "BS.1.1", 810, metric_type="stock", economic_role="cash", namespace_target="BS", migration_status="keep"),
        MetricSpec("BS.CASH.PM", "BS", "CASH", "Fondos PM al cierre", "last", True, "daily_cash_position", "build_bs_cash_pm", "BS.CASH.TOTAL", "BS.1.2", 820, metric_type="stock", economic_role="cash", namespace_target="BS", migration_status="keep"),
        MetricSpec("BS.CASH.TOTAL", "BS", "CASH", "Activos líquidos totales", "sum_components", False, "", "", "BS.ASSETS.TOTAL", "BS.1", 800, metric_type="stock", economic_role="cash", namespace_target="BS", migration_status="keep"),
        MetricSpec("BS.DEBT.PM_TO_MI.OPEN", "BS", "DEBT", "Deuda PM con MI (abierta)", "last", True, "debt_balance", "build_bs_debt_pm_to_mi_open", "BS.DEBT.TOTAL.OPEN", "BS.2.1", 910, metric_type="stock", economic_role="debt", namespace_target="ID", migration_status="alias", notes="Internal debt exposure also belongs in the future ID namespace."),
        MetricSpec("BS.DEBT.PM_TO_PRIMOS.OPEN", "BS", "DEBT", "Deuda PM con Primos (abierta)", "last", True, "debt_balance", "build_bs_debt_pm_to_primos_open", "BS.DEBT.TOTAL.OPEN", "BS.2.2", 920, metric_type="stock", economic_role="debt", namespace_target="ID", migration_status="alias", notes="Internal debt exposure also belongs in the future ID namespace."),
        MetricSpec("BS.CLAIM.ALE_TO_PM.OPEN", "BS", "CLAIM", "Crédito PM contra Alejandro (abierto)", "last", True, "debt_balance", "build_bs_claim_ale_to_pm_open", "BS.DEBT.NET_PM_POSITION", "BS.2.3", 930, metric_type="stock", economic_role="claim", namespace_target="BS", migration_status="alias", notes="Likely future BS receivable alias; confirm naming with stakeholders."),
        MetricSpec("BS.DEBT.PRINCIPAL.OPEN", "BS", "DEBT", "Principal deuda PM (abierto)", "last", True, "debt_balance", "build_bs_debt_principal_open", "BS.DEBT.TOTAL.OPEN", "BS.2.4", 940, metric_type="stock", economic_role="debt", namespace_target="ID", migration_status="alias"),
        MetricSpec("BS.DEBT.INTEREST.OPEN", "BS", "DEBT", "Interés deuda PM (abierto)", "last", True, "debt_balance", "build_bs_debt_interest_open", "BS.DEBT.TOTAL.OPEN", "BS.2.5", 950, metric_type="stock", economic_role="debt", namespace_target="ID", migration_status="alias"),
        MetricSpec("BS.DEBT.TOTAL.OPEN", "BS", "DEBT", "Deuda total PM (abierta)", "sum", True, "debt_balance", "build_bs_debt_total_open", "BS.LIAB.TOTAL", "BS.2", 900, metric_type="stock", economic_role="debt", namespace_target="ID", migration_status="alias", notes="Stock metric; current agg_rule=sum means sum across counterparties/components at period close, not across time."),
        MetricSpec("BS.DEBT.NET_PM_POSITION", "BS", "DEBT", "Posición neta PM frente a deuda", "formula", True, "debt_balance", "build_bs_debt_net_pm_position", "BS.NET.TOTAL", "BS.2.6", 960, metric_type="stock", economic_role="debt", namespace_target="ID", migration_status="alias", notes="Formula-like stock metric currently implemented by a builder."),
    ]
