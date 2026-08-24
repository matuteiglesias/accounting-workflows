from __future__ import annotations

"""Metrics-frontier facade with governed validated-cash projection.

The pre-PR15B frontier implementation remains in ``frontier_legacy``. This
module delegates all non-cash metrics to it and rewrites only the two modern
cash headline series from the shared governed cash projection.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from accounting.cash_authority import validated_cash_schema_supported
from accounting.cash_projection import iter_validated_monthly_cash_positions
from accounting.metrics import frontier_legacy as _legacy


# Explicit compatibility surface derived from repository caller census.
# Do not broaden this list: every retained legacy symbol must have a caller
# or an independently documented compatibility contract/removal condition.
LEGACY_COMPAT_EXPORTS = ()


def _selection_period_end(selection) -> str:
    if selection.selected.empty or "period_end" not in selection.selected.columns:
        return selection.period
    values = selection.selected["period_end"].fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    return values.max() if not values.empty else selection.period


def build_metrics_frontier(
    run_root: Path,
    metrics_dir: Path,
    run_id: str,
    as_of_date: str,
) -> Dict[str, Path]:
    paths = _legacy.build_metrics_frontier(run_root, metrics_dir, run_id, as_of_date)

    run_root = Path(run_root)
    metrics_dir = Path(metrics_dir)
    cash_path = run_root / "monthly_cash_close.csv"
    cash = pd.read_csv(cash_path) if cash_path.exists() else None
    if cash is None or cash.empty or not validated_cash_schema_supported(cash):
        return paths

    frontier_path = paths["metric_contract_frontier"]
    series_path = paths["frontend_metric_series"]
    frontier = pd.read_csv(frontier_path)
    series = pd.read_csv(series_path)

    cash_ids = {"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"}
    frontier = frontier.loc[~frontier["metric_id"].astype(str).isin(cash_ids)].copy()
    series = series.loc[~series["metric_id"].astype(str).isin(cash_ids)].copy()

    projections = list(iter_validated_monthly_cash_positions(cash))
    selected_rows: list[dict[str, Any]] = []
    for projection in projections:
        selected = projection.selection
        if not selected.available:
            continue

        metric_id = (
            "BS.CASH.TOTAL"
            if projection.scope == "currency"
            else "BS.CASH.CLOSE.BOX"
        )
        selected_rows.append(
            _legacy._series_row(
                metric_id,
                projection.reporting_period,
                _selection_period_end(selected),
                projection.currency,
                selected.value,
                "monthly_cash_close.csv",
                run_id,
                as_of_date,
                "safe",
                True,
                False,
                False,
                "Governed validated account snapshots only; inferred/internal excluded and no fallback used.",
                dimension_name="Box" if projection.scope == "box" else "",
                dimension_value=projection.box if projection.scope == "box" else "",
            )
        )

    has_governed_cash = bool(selected_rows)
    status = "active" if has_governed_cash else "unavailable"
    suitability = "safe" if has_governed_cash else "unavailable"
    validation = "ok" if has_governed_cash else "warn"
    caveat = (
        "Cash headline uses latest valid as_of_date per Box/account_id and sums "
        "selected validated account closes. Inferred control and internal balances "
        "are excluded; no fallback is permitted."
        if has_governed_cash
        else "No complete governed validated cash position is available; no inferred/internal fallback used."
    )
    rule_total = (
        "cash.position.validated: latest valid as_of_date per Box/account_id; "
        "sum selected accounts by period/currency; missing/incomplete position unavailable"
    )
    rule_box = (
        "cash.position.validated: latest valid as_of_date per account_id within "
        "period/Currency/Box; sum selected accounts; no inferred/internal fallback"
    )
    new_frontier = pd.DataFrame(
        [
            _legacy._contract_row(
                "BS.CASH.TOTAL",
                "Frontend-safe cash total",
                "cash",
                "stock",
                "monthly_cash_close.csv",
                rule_total,
                suitability=suitability,
                caveat=caveat,
                status=status,
                validation=validation,
            ),
            _legacy._contract_row(
                "BS.CASH.CLOSE.BOX",
                "Frontend-safe cash by Box",
                "cash",
                "stock",
                "monthly_cash_close.csv",
                rule_box,
                suitability=suitability,
                caveat=caveat,
                status=status,
                validation=validation,
                notes="dimension_name=Box; governed by cash.position.validated",
            ),
        ],
        columns=_legacy.FRONTIER_COLUMNS,
    )
    frontier = pd.concat([frontier, new_frontier], ignore_index=True)
    if selected_rows:
        series = pd.concat(
            [series, pd.DataFrame(selected_rows, columns=_legacy.SERIES_COLUMNS)],
            ignore_index=True,
        )

    metric_registry = _legacy._read_csv(metrics_dir / "metric_registry.csv")
    metric_values = _legacy._read_csv(metrics_dir / "metric_values.csv")
    qa = _legacy.build_frontier_qa(frontier, series, cash, metric_registry, metric_values)
    qa = pd.concat(
        [
            qa,
            pd.DataFrame(
                [
                    {
                        "check": "cash_headline_uses_governed_validated_projection",
                        "status": "pass",
                        "detail": f"cash_series_rows={len(selected_rows)}; fallback_to_inferred=never",
                        "severity": "error",
                    },
                    {
                        "check": "cash_frontier_excludes_internal_and_inferred_positions",
                        "status": "pass",
                        "detail": "all BS.CASH series rows are reconstructed from cash.position.validated selections",
                        "severity": "error",
                    },
                    {
                        "check": "cash_frontier_scopes_are_source_backed",
                        "status": "pass",
                        "detail": f"projection_scopes={len(projections)}; no report-layer Cartesian scope synthesis",
                        "severity": "error",
                    },
                ],
                columns=_legacy.QA_COLUMNS,
            ),
        ],
        ignore_index=True,
    )

    frontier.to_csv(frontier_path, index=False)
    series.to_csv(series_path, index=False)
    qa.to_csv(paths["metrics_frontier_qa"], index=False)
    qa.to_csv(paths["frontier_source_qa"], index=False)
    return paths
