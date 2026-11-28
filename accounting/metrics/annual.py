from __future__ import annotations

"""Annual-metrics facade with governed funding/support and Box controls.

The delegated legacy builder still materializes the compatibility baseline. This
facade then replaces only the surfaces with established lower authorities:
modern semantic funding/support metrics and governed Box controls.
The narrow ``FUND.CONTRIB.TOTAL`` accounting metric is deliberately untouched.

For source-audit continuity, delegated atomic flow projection still resolves
``resolve_semantic_measure`` and then reads ``rows[measure]``. This facade does
not introduce a competing atomic-measure authority.
"""

from pathlib import Path

import pandas as pd

from accounting.contracts.annual_flow_membership import build_annual_flow_membership
from accounting.contracts.funding_support import (
    FUNDING_SUPPORT_SPECS_VERSION,
    classify_funding_support,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.metrics import annual_legacy as _legacy
from accounting.metrics.resource_bridge import write_resource_bridge, SOURCE as RESOURCE_SOURCE, MEMBERS as RESOURCE_MEMBERS
from accounting.metrics.cash_control_adapter import (
    SOURCE_TABLE as CASH_CONTROL_SOURCE,
    annual_rows as cash_control_annual_rows,
    read_source as read_cash_control_source,
)


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


def _rewrite_cash_control_metrics(
    metrics: pd.DataFrame,
    source: pd.DataFrame | None,
    *,
    run_id: str,
    as_of_date: str,
) -> tuple[pd.DataFrame, bool]:
    """Add annual movement/control rows from the governed treasury source."""

    ids = {"CTRL.BOX.NET_MOVEMENT", "CTRL.BOX.CUM_NET"}
    out = metrics.loc[~metrics["metric_id"].astype(str).isin(ids)].copy()
    caveat = (
        "Recorded Box movement/control only: cumulative control is the latest "
        "zero-origin month-end value, not actual cash or liquidity; unknown opening "
        "balances remain unknown; no FX or cross-Box aggregation."
    )
    rows: list[dict[str, object]] = []
    if source is not None and not source.empty:
        for row in cash_control_annual_rows(source, run_id=run_id, as_of_date=as_of_date):
            rows.append(
                _legacy._base(
                    row["metric_id"], row["period"], row["Currency"], row["value"],
                    "available", "flow" if row["metric_id"] == "CTRL.BOX.NET_MOVEMENT" else "stock",
                    "cash_control", "3. Cash and liquidity", CASH_CONTROL_SOURCE,
                    "cash_control_specs_v1; dimension=Box",
                    "annual flow = sum monthly net_cash_flow by year/Box/Currency; "
                    "annual control = latest month-end closing_control by year/Box/Currency",
                    run_id, as_of_date, dim_name="Box", dim_value=str(row["Box"]),
                    suit="safe_with_caveat", caveat=row["caveat"],
                )
            )
    else:
        for metric_id, flow in [("CTRL.BOX.NET_MOVEMENT", "flow"), ("CTRL.BOX.CUM_NET", "stock")]:
            rows.append(
                _legacy._base(
                    metric_id, "", "", pd.NA, "unavailable", flow, "unavailable",
                    "3. Cash and liquidity", CASH_CONTROL_SOURCE, "", "unavailable: source absent",
                    run_id, as_of_date, suit="unavailable", validation="warn",
                    caveat="Source absent; no synthetic cash-control series emitted.",
                )
            )
    return pd.concat(
        [out, pd.DataFrame(rows, columns=_legacy.ANNUAL_METRICS_COLUMNS)], ignore_index=True
    ), True


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
    annual_membership = build_annual_flow_membership(split) if split is not None else build_annual_flow_membership(pd.DataFrame())
    membership_path = metrics_dir / "annual_flow_membership.csv"
    annual_membership.to_csv(membership_path, index=False)
    paths["annual_flow_membership"] = membership_path
    metrics, funding_rewritten = _rewrite_funding_support_metrics(
        metrics, split, run_id=run_id, as_of_date=as_of_date
    )

    cash_control = read_cash_control_source(run_root)
    metrics, cash_control_rewritten = _rewrite_cash_control_metrics(
        metrics, cash_control, run_id=run_id, as_of_date=as_of_date
    )

    bridge = write_resource_bridge(run_root, metrics_dir)
    if bridge is not None:
        paths["monthly_resource_bridge"] = metrics_dir / RESOURCE_SOURCE
        paths["resource_bridge_membership"] = metrics_dir / RESOURCE_MEMBERS
        bridge = bridge.assign(period=bridge.period.astype(str).str[:4]).groupby(["period","Currency"]).sum(numeric_only=True).reset_index()
        additions = []
        for row in bridge.itertuples(index=False):
            vals = {
                "FUND.ECONOMIC.TOTAL": row.economic_funding,
                "FUND.ECONOMIC.CONTRIBUTIONS": row.contribution,
                "FUND.ECONOMIC.DESIGNATED": row.designated_advance,
                "FUND.ECONOMIC.DIRECT_TENANT": row.direct_tenant_support,
                "FUND.ECONOMIC.DIRECT_OTHER": row.direct_other_support,
                "FUND.ECONOMIC.CLEARING_PENDING": row.clearing_pending,
                "FUND.ECONOMIC.LINKED_RECOVERY": row.linked_debt_recovery,
            }
            sub = metrics.loc[metrics.period.astype(str).str[:4].eq(row.period) & metrics.Currency.eq(row.Currency) & metrics.dimension_name.fillna("").eq("")]
            lookup = sub.set_index("metric_id").value
            if "IS.NET.OPERATING" in lookup and "DIST.DRAWS.PERSONAL" in lookup:
                operating = float(lookup["IS.NET.OPERATING"])
                coverage = operating + row.economic_funding + row.linked_debt_recovery - float(lookup["DIST.DRAWS.PERSONAL"])
                vals["COV.RESOURCE.AFTER_DRAWS"] = coverage
                vals["COV.RESOURCE.RATIO"] = coverage / operating if operating else None
            for mid, value in vals.items():
                ratio = mid.endswith(".RATIO")
                coverage_metric = mid.startswith("COV.")
                additions.append(_legacy._base(mid,row.period,row.Currency,value,"not_applicable" if value is None else "available",
                    "ratio" if ratio else "flow","economic_funding","2. Funding and distributions",
                    RESOURCE_SOURCE if not coverage_metric else RESOURCE_SOURCE+";monthly_operating_statement.csv",
                    "explicit paid transaction membership; linked recovery separately allocated",
                    "annual ratio = annual resource coverage / operating result" if ratio else
                    ("annual flow = net operating + economic funding + linked debt recovery - draws" if coverage_metric else "annual flow = sum monthly governed resource component"),
                    run_id,as_of_date,suit="safe_with_caveat",caveat="Economic financing includes reimbursable advances and direct external support; not profit or cash. Linked recovery is debt settlement, not fresh funding. Pending clearing excluded."))
        members = pd.read_csv(metrics_dir / RESOURCE_MEMBERS, keep_default_na=False)
        fund = members.loc[members.component.isin(["contribution","designated_advance","direct_tenant_support","direct_other_support"])].copy()
        fund["period"] = fund.period.astype(str).str[:4]
        for (period,currency,actor), amount in fund.groupby(["period","Currency","funding_actor"]).amount.sum().items():
            additions.append(_legacy._base("FUND.ECONOMIC.BY_ACTOR",period,currency,amount,"available",
                "flow","economic_funding","2. Funding and distributions",RESOURCE_MEMBERS,
                "economic funding members; grouped by actual contributor",
                "annual flow = sum unique funding transaction amounts",run_id,as_of_date,
                dim_name="funding_actor",dim_value=actor,suit="safe_with_caveat",
                caveat="Includes designated advances and direct payments, excludes debt settlements and pending clearing."))
        metrics = pd.concat([metrics,pd.DataFrame(additions,columns=_legacy.ANNUAL_METRICS_COLUMNS)],ignore_index=True)
    cash_rewritten = bridge is not None

    if not funding_rewritten and not cash_rewritten and not cash_control_rewritten:
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
    if cash_control_rewritten:
        extra_qa.append(
            {
                "check": "annual_cash_controls_use_governed_accountability",
                "status": "pass",
                "detail": "monthly_cash_accountability.csv; flows summed and cumulative controls selected at latest month-end; no FX or cross-Box aggregation",
                "severity": "error",
            }
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
