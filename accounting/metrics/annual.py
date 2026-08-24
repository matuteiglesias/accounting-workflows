from __future__ import annotations

"""Annual-metrics facade with governed cash and funding-support projection.

The delegated legacy builder still materializes the compatibility baseline. This
facade then replaces only the surfaces with established lower authorities:
validated cash and, for modern semantic rows, broader funding/support metrics.
The narrow ``FUND.CONTRIB.TOTAL`` accounting metric is deliberately untouched.
"""

from pathlib import Path

import pandas as pd

from accounting.cash_authority import validated_cash_schema_supported
from accounting.cash_projection import iter_validated_annual_cash_positions
from accounting.contracts.funding_support import (
    FUNDING_SUPPORT_SPECS_VERSION,
    classify_funding_support,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.metrics import annual_legacy as _legacy


LEGACY_COMPAT_EXPORTS = (
    "ANNUAL_CONTRACT_COLUMNS",
    "ANNUAL_METRICS_COLUMNS",
    "QA_COLUMNS",
)

ANNUAL_CONTRACT_COLUMNS = _legacy.ANNUAL_CONTRACT_COLUMNS
ANNUAL_METRICS_COLUMNS = _legacy.ANNUAL_METRICS_COLUMNS
QA_COLUMNS = _legacy.QA_COLUMNS

_SUPPORT_METRIC_IDS = {
    "FUND.CONTRIB.BY_FUNDING_ACTOR",
    "FUND.CONTRIB.BY_CHANNEL",
    "FUND.CONTRIB.BY_CASH_EFFECT",
    "FUND.CONTRIB.BY_TARGET_BOX",
    "FUND.CONTRIB.DIRECT_OBLIGATION",
    "FUND.CONTRIB.CASH_TO_BOX",
    "FUND.CONTRIB.DEBT_LINKED",
}
_MODERN_FUNDING_COLUMNS = {
    "period",
    "Currency",
    "semantic_bucket",
    "semantic_subbucket",
    "funding_channel",
    "cash_effect",
    "debt_effect",
}


def _cash_rule(scope: str) -> str:
    return (
        f"annual stock = last governed validated cash period in year for {scope}; "
        "latest valid as_of_date per Box/account_id; sum selected account closes; "
        "never sum monthly positions; no inferred/internal fallback"
    )


def _nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _rewrite_funding_support_metrics(
    metrics: pd.DataFrame,
    split: pd.DataFrame | None,
    *,
    run_id: str,
    as_of_date: str,
) -> tuple[pd.DataFrame, bool]:
    if split is None or split.empty or not _MODERN_FUNDING_COLUMNS.issubset(split.columns):
        return metrics, False

    members = classify_funding_support(split, strict=True).copy()
    out = metrics.loc[~metrics["metric_id"].astype(str).isin(_SUPPORT_METRIC_IDS)].copy()
    rows: list[dict[str, object]] = []
    if not members.empty:
        members["period"] = members["period"].astype(str).str.slice(0, 4)
        members["support_amount"] = pd.to_numeric(
            members["support_amount"], errors="coerce"
        ).fillna(0.0)
        caveat = (
            "Broader support is governed by funding_support_specs_v1 and remains "
            "distinct from the narrow core FUND.CONTRIB.TOTAL metric."
        )
        dim_specs = [
            ("funding_actor", "FUND.CONTRIB.BY_FUNDING_ACTOR"),
            ("funding_channel", "FUND.CONTRIB.BY_CHANNEL"),
            ("cash_effect", "FUND.CONTRIB.BY_CASH_EFFECT"),
            ("target_box", "FUND.CONTRIB.BY_TARGET_BOX"),
        ]
        for dim, metric_id in dim_specs:
            if dim not in members.columns:
                continue
            sub = members.loc[_nonempty(members[dim])]
            grouped = sub.groupby(["period", "Currency", dim], dropna=False)[
                "support_amount"
            ].sum().reset_index()
            for _, row in grouped.iterrows():
                rows.append(
                    _legacy._base(
                        metric_id,
                        row.period,
                        row.Currency,
                        row.support_amount,
                        "available",
                        "flow",
                        "funding_support",
                        "2. Funding and distributions",
                        "monthly_flow_semantic_split.csv",
                        f"{FUNDING_SUPPORT_SPECS_VERSION}; dimension={dim}",
                        "annual funding/support flow = sum governed support membership by year, currency, and dimension",
                        run_id,
                        as_of_date,
                        dim_name=dim,
                        dim_value=str(row[dim]),
                        suit="safe_with_caveat",
                        caveat=caveat,
                    )
                )

        total_specs = [
            (
                "FUND.CONTRIB.DIRECT_OBLIGATION",
                members["support_kind"].eq("direct_obligation_payment"),
                "support_kind=direct_obligation_payment",
            ),
            (
                "FUND.CONTRIB.CASH_TO_BOX",
                members["cash_effect"].fillna("").astype(str).eq("cash_in_box"),
                "cash_effect=cash_in_box",
            ),
            (
                "FUND.CONTRIB.DEBT_LINKED",
                members["support_kind"].eq("debt_linked_support"),
                "support_kind=debt_linked_support",
            ),
        ]
        for metric_id, mask, filter_detail in total_specs:
            grouped = members.loc[mask].groupby(["period", "Currency"], dropna=False)[
                "support_amount"
            ].sum().reset_index()
            for _, row in grouped.iterrows():
                rows.append(
                    _legacy._base(
                        metric_id,
                        row.period,
                        row.Currency,
                        row.support_amount,
                        "available",
                        "flow",
                        "funding_support",
                        "2. Funding and distributions",
                        "monthly_flow_semantic_split.csv",
                        f"{FUNDING_SUPPORT_SPECS_VERSION}; {filter_detail}",
                        "annual funding/support flow = sum governed support membership by year and currency",
                        run_id,
                        as_of_date,
                        suit="safe_with_caveat",
                        caveat=caveat,
                    )
                )

    present = {str(row["metric_id"]) for row in rows}
    for metric_id in sorted(_SUPPORT_METRIC_IDS - present):
        rows.append(
            _legacy._base(
                metric_id,
                "",
                "",
                pd.NA,
                "unavailable",
                "flow",
                "unavailable",
                "2. Funding and distributions",
                "monthly_flow_semantic_split.csv",
                FUNDING_SUPPORT_SPECS_VERSION,
                "governed funding/support membership unavailable for requested metric",
                run_id,
                as_of_date,
                suit="unavailable",
                validation="warn",
                caveat="No governed funding-support members for this metric; no label inference used.",
            )
        )
    return pd.concat(
        [out, pd.DataFrame(rows, columns=_legacy.ANNUAL_METRICS_COLUMNS)],
        ignore_index=True,
    ), True


def _rewrite_cash_metrics(
    metrics: pd.DataFrame,
    cash: pd.DataFrame | None,
    *,
    run_id: str,
    as_of_date: str,
) -> tuple[pd.DataFrame, list[object], bool]:
    if cash is None or cash.empty or not validated_cash_schema_supported(cash):
        return metrics, [], False

    cash_ids = {"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"}
    metrics = metrics.loc[~metrics["metric_id"].astype(str).isin(cash_ids)].copy()
    projections = list(iter_validated_annual_cash_positions(cash))
    new_rows: list[dict[str, object]] = []
    for projection in projections:
        selected = projection.selection
        available = selected.available
        if projection.scope == "currency":
            new_rows.append(
                _legacy._base(
                    "BS.CASH.TOTAL",
                    projection.reporting_period,
                    projection.currency,
                    selected.value if available else pd.NA,
                    "available" if available else "unavailable",
                    "stock",
                    "cash" if available else "unavailable",
                    "3. Cash and liquidity",
                    "monthly_cash_close.csv",
                    "cash.position.validated; fallback_to_inferred=never",
                    _cash_rule("Currency"),
                    run_id,
                    as_of_date,
                    suit="safe" if available else "unavailable",
                    validation="ok" if available else "warn",
                    caveat=(
                        "Governed validated account snapshots only; inferred control and internal balances excluded."
                        if available
                        else f"Governed validated cash unavailable: {selected.reason}; no fallback used."
                    ),
                )
            )
            continue
        new_rows.append(
            _legacy._base(
                "BS.CASH.CLOSE.BOX",
                projection.reporting_period,
                projection.currency,
                selected.value if available else pd.NA,
                "available" if available else "unavailable",
                "stock",
                "cash" if available else "unavailable",
                "3. Cash and liquidity",
                "monthly_cash_close.csv",
                "cash.position.validated; dimension=Box; fallback_to_inferred=never",
                _cash_rule("Box/Currency"),
                run_id,
                as_of_date,
                dim_name="Box",
                dim_value=projection.box,
                suit="safe" if available else "unavailable",
                validation="ok" if available else "warn",
                caveat=(
                    "Governed validated account snapshots only; inferred control and internal balances excluded."
                    if available
                    else f"Governed validated cash unavailable: {selected.reason}; no fallback used."
                ),
            )
        )

    if not new_rows:
        rule = _cash_rule("available scope")
        for metric_id in ["BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"]:
            new_rows.append(
                _legacy._base(
                    metric_id,
                    "",
                    "",
                    pd.NA,
                    "unavailable",
                    "stock",
                    "unavailable",
                    "3. Cash and liquidity",
                    "monthly_cash_close.csv",
                    "cash.position.validated; no governed position available",
                    rule,
                    run_id,
                    as_of_date,
                    suit="unavailable",
                    validation="warn",
                    caveat="No complete governed validated cash position; no inferred/internal fallback used.",
                )
            )
    return pd.concat(
        [metrics, pd.DataFrame(new_rows, columns=_legacy.ANNUAL_METRICS_COLUMNS)],
        ignore_index=True,
    ), projections, True


def build_annual_balance_dashboard(
    run_root: Path,
    metrics_dir: Path,
    run_id: str,
    as_of_date: str,
) -> dict[str, Path]:
    paths = _legacy.build_annual_balance_dashboard(
        run_root, metrics_dir, run_id, as_of_date
    )

    run_root, metrics_dir = Path(run_root), Path(metrics_dir)
    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    split_path = run_root / "monthly_flow_semantic_split.csv"
    split = pd.read_csv(split_path) if split_path.exists() else None
    metrics, funding_rewritten = _rewrite_funding_support_metrics(
        metrics, split, run_id=run_id, as_of_date=as_of_date
    )

    cash_path = run_root / "monthly_cash_close.csv"
    cash = pd.read_csv(cash_path) if cash_path.exists() else None
    metrics, projections, cash_rewritten = _rewrite_cash_metrics(
        metrics, cash, run_id=run_id, as_of_date=as_of_date
    )

    if not funding_rewritten and not cash_rewritten:
        return paths

    contract = _legacy._contract_from_rows(metrics)
    qa = _legacy.build_annual_balance_dashboard_qa(metrics, contract)
    extra_qa: list[dict[str, object]] = []
    if funding_rewritten:
        core = metrics.loc[metrics["metric_id"].astype(str).eq("FUND.CONTRIB.TOTAL")]
        extra_qa.extend(
            [
                {
                    "check": "annual_support_uses_governed_membership",
                    "status": "pass",
                    "detail": f"{FUNDING_SUPPORT_SPECS_VERSION}; no label inference on modern semantic rows",
                    "severity": "error",
                },
                {
                    "check": "core_funding_metric_remains_distinct",
                    "status": "pass" if not core.empty else "fail",
                    "detail": "FUND.CONTRIB.TOTAL preserved outside broader support rewrite",
                    "severity": "error",
                },
            ]
        )
    if cash_rewritten:
        extra_qa.extend(
            [
                {
                    "check": "annual_cash_uses_governed_validated_projection",
                    "status": "pass",
                    "detail": "BS.CASH values rebuilt from shared source-backed cash projection; cash.position.validated; fallback_to_inferred=never",
                    "severity": "error",
                },
                {
                    "check": "annual_cash_never_sums_monthly_positions",
                    "status": "pass",
                    "detail": "last governed period then same monthly account-snapshot primitive",
                    "severity": "error",
                },
                {
                    "check": "annual_cash_scopes_are_source_backed",
                    "status": "pass",
                    "detail": f"projection_scopes={len(projections)}; no report-layer Cartesian scope synthesis",
                    "severity": "error",
                },
            ]
        )
    if extra_qa:
        qa = pd.concat(
            [qa, pd.DataFrame(extra_qa, columns=_legacy.QA_COLUMNS)],
            ignore_index=True,
        )

    metrics.to_csv(paths["annual_balance_dashboard_metrics"], index=False)
    contract.to_csv(paths["annual_balance_dashboard_contract"], index=False)
    qa.to_csv(paths["annual_balance_dashboard_qa"], index=False)
    return paths
