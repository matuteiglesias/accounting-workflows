from __future__ import annotations

"""Attach stable Wave 5 derived-metric identities to professional tables.

This is a compatibility metadata adapter, not a formula authority. Exact
presentation mappings are centralized here so the executor itself never
branches on human labels. Explicit IDs always win and conflicts fail closed.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from accounting.contracts.derived_metrics import resolve_derived_metric_spec


DERIVED_METRIC_ID_COLUMN = "derived_metric_id"
DERIVED_ID_SOURCE_COLUMN = "derived_metric_id_source"

_METRIC_ID_MAP = {
    "IS.NET.OPERATING": "derived.net_operating",
    "COV.NET.AFTER_DRAWS": "derived.coverage_after_draws",
    "COV.SAVINGS_RATE": "derived.savings_rate",
    "RATIO.OPERATING_MARGIN": "derived.operating_margin",
    "RATIO.OPEX_TO_RENT": "derived.opex_to_rent",
    "RATIO.DRAWS_TO_OPERATING_RESULT": "derived.draws_to_operating_result",
}

_OVERVIEW_LABEL_MAP = {
    "margen operativo": "derived.operating_margin",
    "opex / renta": "derived.opex_to_rent",
    "retiros / resultado operativo": "derived.draws_to_operating_result",
    "cobertura después de funding y retiros": "derived.coverage_after_draws",
    "cobertura despues de funding y retiros": "derived.coverage_after_draws",
    "tasa de ahorro": "derived.savings_rate",
    "savings rate": "derived.savings_rate",
}

_INCOME_LABEL_MAP = {
    "resultado operativo neto": "derived.net_operating",
    "net operating": "derived.net_operating",
    "net operating result": "derived.net_operating",
    "cobertura después de funding y retiros": "derived.coverage_after_draws",
    "cobertura despues de funding y retiros": "derived.coverage_after_draws",
}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _first(row: pd.Series, *names: str) -> str:
    for name in names:
        value = _text(row.get(name))
        if value:
            return value
    return ""


def _validate(spec_id: str) -> str:
    if not spec_id:
        return ""
    if resolve_derived_metric_spec(spec_id) is None:
        raise ValueError(f"Unknown derived_metric_id metadata: {spec_id!r}")
    return spec_id


def _candidate(table_id: str, row: pd.Series) -> tuple[str, str]:
    metric_id = _text(row.get("metric_id"))
    if metric_id in _METRIC_ID_MAP:
        return _METRIC_ID_MAP[metric_id], "stable_metric_id"

    label = _first(row, "metric", "line", "label", "statement_line").casefold()
    if table_id == "overview_balance_dashboard" and label in _OVERVIEW_LABEL_MAP:
        return _OVERVIEW_LABEL_MAP[label], "compatibility_presentation_mapping"
    if table_id == "income_operating_statement" and label in _INCOME_LABEL_MAP:
        return _INCOME_LABEL_MAP[label], "compatibility_presentation_mapping"
    if table_id == "monthly_tables_diagnostic_box_level_matrix":
        metric = _first(row, "metric", "measure", "line", "label").casefold()
        if metric == "diagnostic_box_level":
            return "derived.diagnostic_box_level", "stable_table_metric"
    return "", ""


def enrich_derived_metric_table(df: pd.DataFrame, table_id: str) -> pd.DataFrame:
    out = df.copy()
    if DERIVED_METRIC_ID_COLUMN not in out.columns:
        out[DERIVED_METRIC_ID_COLUMN] = ""
    if DERIVED_ID_SOURCE_COLUMN not in out.columns:
        out[DERIVED_ID_SOURCE_COLUMN] = ""

    if out.empty:
        return out

    for idx, row in out.iterrows():
        existing = _text(row.get(DERIVED_METRIC_ID_COLUMN))
        candidate, source = _candidate(table_id, row)
        if existing:
            _validate(existing)
            if candidate and candidate != existing:
                raise ValueError(
                    "Explicit derived_metric_id conflicts with stable/compatibility metadata: "
                    f"table_id={table_id!r}; explicit={existing!r}; candidate={candidate!r}"
                )
            out.at[idx, DERIVED_METRIC_ID_COLUMN] = existing
            out.at[idx, DERIVED_ID_SOURCE_COLUMN] = _text(row.get(DERIVED_ID_SOURCE_COLUMN)) or "explicit"
        elif candidate:
            out.at[idx, DERIVED_METRIC_ID_COLUMN] = _validate(candidate)
            out.at[idx, DERIVED_ID_SOURCE_COLUMN] = source

    period_cols = [c for c in out.columns if str(c).startswith("20")]
    front = [DERIVED_METRIC_ID_COLUMN, DERIVED_ID_SOURCE_COLUMN]
    rest = [c for c in out.columns if c not in front and c not in period_cols]
    return out[front + rest + period_cols]


def enrich_derived_metric_tables(tables_dir: Path) -> list[Path]:
    tables_dir = Path(tables_dir)
    if not tables_dir.exists():
        return []
    relevant = {
        "overview_balance_dashboard",
        "income_operating_statement",
        "monthly_tables_diagnostic_box_level_matrix",
    }
    written: list[Path] = []
    for table_id in sorted(relevant):
        path = tables_dir / f"{table_id}.csv"
        if not path.exists():
            continue
        before = pd.read_csv(path)
        after = enrich_derived_metric_table(before, table_id)
        if list(after.columns) != list(before.columns) or not after.equals(before):
            after.to_csv(path, index=False)
            written.append(path)
    return written
