from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

import pandas as pd

from .metrics_views import load_ledger


DRILLDOWN_DIRNAME = "metric_drilldown"
DRILLDOWN_INDEX_FILENAME = "metric_drilldown_index.csv"
DRILLDOWN_MANIFEST_FILENAME = "metric_drilldown_manifest.json"
DRILLDOWN_DETAILS_DIRNAME = "details"
SUPPORTED_DRILLDOWN_METRICS = (
    "IS.RENT.TOTAL",
    "IS.OPEX.TOTAL",
    "IS.DRAWS.PERSONAL",
)


def _slugify_drilldown(metric_id: str, period_grain: str, period: str, currency: str) -> str:
    return f"{metric_id}__{period_grain}__{period}__{currency}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {value!r}")


def _draws_mask(work: pd.DataFrame) -> pd.Series:
    text_cols = [c for c in ["Tipo", "Detalle", "notes", "tag", "Lugar"] if c in work.columns]
    if not text_cols:
        return pd.Series(False, index=work.index)

    mask = pd.Series(False, index=work.index)
    for col in text_cols:
        mask = mask | work[col].astype(str).str.contains(
            r"personal|retiro|draw|owner|dividend",
            case=False,
            na=False,
        )
    return mask


def _official_monthly_targets(run_root: Path) -> pd.DataFrame:
    views_dir = run_root / "views"
    per_flow = pd.read_csv(run_root / "per_flow_time_long.freq=M.csv")
    opex = pd.read_csv(views_dir / "v_opex_category_monthly.csv")
    ledger = load_ledger(run_root)

    frames: list[pd.DataFrame] = []

    rent = per_flow.loc[
        (per_flow["Flujo"].astype(str) == "Cobros")
        & (per_flow["Tipo"].astype(str) == "Renta")
    ].copy()
    if not rent.empty:
        rent_agg = (
            rent.groupby(["TimePeriod", "Currency"], dropna=False)["amount"]
            .sum()
            .reset_index()
            .rename(columns={"TimePeriod": "period", "amount": "target_metric_value", "Currency": "currency"})
        )
        rent_agg["metric_id"] = "IS.RENT.TOTAL"
        rent_agg["period_grain"] = "M"
        frames.append(rent_agg[["metric_id", "period_grain", "period", "currency", "target_metric_value"]])

    if not opex.empty:
        value_col = "amount_out" if "amount_out" in opex.columns else "amount"
        opex_agg = (
            opex.groupby(["TimePeriod", "Currency"], dropna=False)[value_col]
            .sum()
            .reset_index()
            .rename(columns={"TimePeriod": "period", value_col: "target_metric_value", "Currency": "currency"})
        )
        opex_agg["metric_id"] = "IS.OPEX.TOTAL"
        opex_agg["period_grain"] = "M"
        frames.append(opex_agg[["metric_id", "period_grain", "period", "currency", "target_metric_value"]])

    draws = ledger.copy()
    draws["TimePeriod"] = draws["Date"].dt.to_period("M").astype(str)
    draws = draws.loc[_draws_mask(draws)].copy()
    if not draws.empty:
        draws_agg = (
            draws.groupby(["TimePeriod", "Currency"], dropna=False)["amount"]
            .sum()
            .reset_index()
            .rename(columns={"TimePeriod": "period", "amount": "target_metric_value", "Currency": "currency"})
        )
        draws_agg["metric_id"] = "IS.DRAWS.PERSONAL"
        draws_agg["period_grain"] = "M"
        frames.append(draws_agg[["metric_id", "period_grain", "period", "currency", "target_metric_value"]])

    if not frames:
        return pd.DataFrame(columns=["metric_id", "period_grain", "period", "currency", "target_metric_value"])
    out = pd.concat(frames, ignore_index=True)
    out["period"] = out["period"].astype(str)
    out["currency"] = out["currency"].astype(str)
    out["target_metric_value"] = pd.to_numeric(out["target_metric_value"], errors="coerce").fillna(0.0)
    return out


def _filter_spec_factory(include_statuses: Sequence[str]) -> dict[str, Callable[[pd.DataFrame, str, str], tuple[pd.DataFrame, dict[str, Any]]]]:
    normalized_statuses = [str(x).strip().lower() for x in include_statuses if str(x).strip()]

    def _base_filtered(work: pd.DataFrame, period: str, currency: str) -> pd.DataFrame:
        mask = (
            (work["period_m"].astype(str) == str(period))
            & (work["Currency"].astype(str) == str(currency))
        )
        if normalized_statuses:
            mask = mask & work["status"].astype(str).str.strip().str.lower().isin(normalized_statuses)
        return work.loc[mask].copy()

    def rent(work: pd.DataFrame, period: str, currency: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        subset = _base_filtered(work, period, currency)
        subset = subset.loc[
            (subset["Flujo"].astype(str) == "Cobros")
            & (subset["Tipo"].astype(str) == "Renta")
        ].copy()
        return subset, {
            "period_m": period,
            "currency": currency,
            "include_statuses": normalized_statuses,
            "conditions": [
                {"column": "Flujo", "op": "eq", "value": "Cobros"},
                {"column": "Tipo", "op": "eq", "value": "Renta"},
            ],
        }

    def opex(work: pd.DataFrame, period: str, currency: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        subset = _base_filtered(work, period, currency)
        subset = subset.loc[subset["Flujo"].astype(str) == "Pagos"].copy()
        return subset, {
            "period_m": period,
            "currency": currency,
            "include_statuses": normalized_statuses,
            "conditions": [
                {"column": "Flujo", "op": "eq", "value": "Pagos"},
            ],
            "note": "Ledger-level approximation aligned to the current monthly P&L meaning of opex.",
        }

    def draws(work: pd.DataFrame, period: str, currency: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        subset = _base_filtered(work, period, currency)
        subset = subset.loc[_draws_mask(subset)].copy()
        return subset, {
            "period_m": period,
            "currency": currency,
            "include_statuses": normalized_statuses,
            "regex": r"personal|retiro|draw|owner|dividend",
            "text_columns": [c for c in ["Tipo", "Detalle", "notes", "tag", "Lugar"] if c in subset.columns],
        }

    return {
        "IS.RENT.TOTAL": rent,
        "IS.OPEX.TOTAL": opex,
        "IS.DRAWS.PERSONAL": draws,
    }


def _status_for_difference(diff: float, matched_rows: int) -> str:
    if matched_rows == 0:
        return "empty"
    if math.isclose(diff, 0.0, abs_tol=1e-9):
        return "exact"
    return "approx_match"


def build_metric_drilldown_artifacts(
    *,
    run_root: Path,
    out_dir: Path,
    run_id: str,
    include_statuses: Sequence[str],
) -> pd.DataFrame:
    base_dir = out_dir / DRILLDOWN_DIRNAME
    details_dir = base_dir / DRILLDOWN_DETAILS_DIRNAME
    details_dir.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger(run_root)
    target_values = _official_monthly_targets(run_root)
    filters = _filter_spec_factory(include_statuses)

    index_rows: list[dict[str, Any]] = []
    for row in target_values.itertuples(index=False):
        metric_id = str(row.metric_id)
        period = str(row.period)
        currency = str(row.currency)
        target_metric_value = float(row.target_metric_value)
        slug = _slugify_drilldown(metric_id, "M", period, currency)

        if metric_id not in filters:
            index_rows.append(
                {
                    "run_id": run_id,
                    "metric_id": metric_id,
                    "period_grain": "M",
                    "period": period,
                    "currency": currency,
                    "source_table": "ledger_canonical.csv",
                    "filter_json": json.dumps({"unsupported": True}, ensure_ascii=False, sort_keys=True),
                    "detail_csv_relpath": f"{DRILLDOWN_DIRNAME}/{DRILLDOWN_DETAILS_DIRNAME}/{slug}.csv",
                    "detail_html_relpath": "",
                    "matched_rows": 0,
                    "matched_value_sum": 0.0,
                    "target_metric_value": target_metric_value,
                    "difference_vs_target": -target_metric_value,
                    "status": "unsupported",
                }
            )
            continue

        subset, filter_spec = filters[metric_id](ledger, period, currency)
        matched_value_sum = float(pd.to_numeric(subset.get("amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        matched_rows = int(len(subset))
        difference_vs_target = matched_value_sum - target_metric_value
        status = _status_for_difference(difference_vs_target, matched_rows)

        detail_relpath = f"{DRILLDOWN_DIRNAME}/{DRILLDOWN_DETAILS_DIRNAME}/{slug}.csv"
        subset.to_csv(out_dir / detail_relpath, index=False)

        index_rows.append(
            {
                "run_id": run_id,
                "metric_id": metric_id,
                "period_grain": "M",
                "period": period,
                "currency": currency,
                "source_table": "ledger_canonical.csv",
                "filter_json": json.dumps(filter_spec, ensure_ascii=False, sort_keys=True, default=_json_default),
                "detail_csv_relpath": detail_relpath,
                "detail_html_relpath": "",
                "matched_rows": matched_rows,
                "matched_value_sum": matched_value_sum,
                "target_metric_value": target_metric_value,
                "difference_vs_target": difference_vs_target,
                "status": status,
            }
        )

    index_df = pd.DataFrame(index_rows)
    if index_df.empty:
        index_df = pd.DataFrame(
            columns=[
                "run_id",
                "metric_id",
                "period_grain",
                "period",
                "currency",
                "source_table",
                "filter_json",
                "detail_csv_relpath",
                "detail_html_relpath",
                "matched_rows",
                "matched_value_sum",
                "target_metric_value",
                "difference_vs_target",
                "status",
            ]
        )
    index_df.to_csv(base_dir / DRILLDOWN_INDEX_FILENAME, index=False)

    manifest = {
        "run_id": run_id,
        "run_root": str(run_root),
        "source_table": "ledger_canonical.csv",
        "supported_metrics": list(SUPPORTED_DRILLDOWN_METRICS),
        "include_statuses": list(include_statuses),
        "rows": int(len(index_df)),
    }
    (base_dir / DRILLDOWN_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return index_df


def drilldown_lookup(index_df: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if index_df.empty:
        return {}
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in index_df.to_dict(orient="records"):
        key = (
            str(row.get("metric_id", "")),
            str(row.get("period_grain", "")),
            str(row.get("period", "")),
            str(row.get("currency", "")),
        )
        lookup[key] = row
    return lookup


def supported_metric_ids() -> Iterable[str]:
    return SUPPORTED_DRILLDOWN_METRICS
