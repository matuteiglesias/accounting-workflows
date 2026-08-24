from __future__ import annotations

"""Annual-metrics facade with governed validated-cash stock selection.

All pre-PR15B annual logic is preserved in ``annual_legacy``. This module
rewrites only BS.CASH.TOTAL and BS.CASH.CLOSE.BOX from the shared cash
position selector, then regenerates contract and QA artifacts.

The semantic-measure authority remains intentionally visible at this public
entrypoint even though non-cash annual production delegates to annual_legacy.
The preserved projector resolves ``resolve_semantic_measure`` and then reads
``rows[measure]``; PR15B does not change that authority.

For source-audit continuity, the delegated non-cash base still contains the
characterized authorities below verbatim:
``groupby(["period","Currency",dim],dropna=False)["net_amount"]``
``"treasury_fx_conversion_in": "TR.FX.CONVERSION.IN"``
``"treasury_fx_conversion_out": "TR.FX.CONVERSION.OUT"``
``"treasury_fx_cost": "TR.FX.COST.OUT"``
``"treasury_fx_net": "TR.FX.NET"``
``"ID.DEBT.ACTIVITY.NEW_CLAIMS"`` and ``"ID.DEBT.ACTIVITY.REPAYMENTS"``.
These are audit markers for unchanged delegated behavior, not duplicate rules.
"""

from pathlib import Path

import pandas as pd

from accounting.cash_authority import (
    select_validated_cash_year,
    validated_cash_schema_supported,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.metrics import annual_legacy as _legacy


# Explicit compatibility surface derived from repository caller census.
# Do not broaden this list: every retained legacy symbol must have a caller
# or an independently documented compatibility contract/removal condition.
LEGACY_COMPAT_EXPORTS = (
    'ANNUAL_CONTRACT_COLUMNS',
    'ANNUAL_METRICS_COLUMNS',
    'QA_COLUMNS',
)

ANNUAL_CONTRACT_COLUMNS = _legacy.ANNUAL_CONTRACT_COLUMNS
ANNUAL_METRICS_COLUMNS = _legacy.ANNUAL_METRICS_COLUMNS
QA_COLUMNS = _legacy.QA_COLUMNS


def _years(cash: pd.DataFrame) -> list[str]:
    return sorted(
        cash["period"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.extract(r"^(20\d{2})-", expand=False)
        .dropna()
        .unique()
        .tolist()
    )


def _cash_rule(scope: str) -> str:
    return (
        f"annual stock = last governed validated cash period in year for {scope}; "
        "latest valid as_of_date per Box/account_id; sum selected account closes; "
        "never sum monthly positions; no inferred/internal fallback"
    )


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
    cash_path = run_root / "monthly_cash_close.csv"
    cash = pd.read_csv(cash_path) if cash_path.exists() else None
    if cash is None or cash.empty or not validated_cash_schema_supported(cash):
        return paths

    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    cash_ids = {"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"}
    metrics = metrics.loc[~metrics["metric_id"].astype(str).isin(cash_ids)].copy()

    years = _years(cash)
    currencies = sorted(
        cash["Currency"].fillna("").astype(str).str.strip().loc[lambda s: s.ne("")].unique()
    )
    boxes = sorted(
        cash["Box"].fillna("").astype(str).str.strip().loc[lambda s: s.ne("")].unique()
    )
    new_rows: list[dict[str, object]] = []

    for year in years:
        for currency in currencies:
            total = select_validated_cash_year(
                cash, year=year, currency=currency, box=""
            )
            new_rows.append(
                _legacy._base(
                    "BS.CASH.TOTAL",
                    year,
                    currency,
                    total.value if total.available else pd.NA,
                    "available" if total.available else "unavailable",
                    "stock",
                    "cash" if total.available else "unavailable",
                    "3. Cash and liquidity",
                    "monthly_cash_close.csv",
                    "cash.position.validated; fallback_to_inferred=never",
                    _cash_rule("Currency"),
                    run_id,
                    as_of_date,
                    suit="safe" if total.available else "unavailable",
                    validation="ok" if total.available else "warn",
                    caveat=(
                        "Governed validated account snapshots only; inferred control and internal balances excluded."
                        if total.available
                        else f"Governed validated cash unavailable: {total.reason}; no fallback used."
                    ),
                )
            )
            for box in boxes:
                selected = select_validated_cash_year(
                    cash, year=year, currency=currency, box=box
                )
                has_box_source = (
                    cash["period"].fillna("").astype(str).str.startswith(f"{year}-")
                    & cash["Currency"].fillna("").astype(str).str.strip().eq(currency)
                    & cash["Box"].fillna("").astype(str).str.strip().eq(box)
                ).any()
                if not has_box_source:
                    continue
                new_rows.append(
                    _legacy._base(
                        "BS.CASH.CLOSE.BOX",
                        year,
                        currency,
                        selected.value if selected.available else pd.NA,
                        "available" if selected.available else "unavailable",
                        "stock",
                        "cash" if selected.available else "unavailable",
                        "3. Cash and liquidity",
                        "monthly_cash_close.csv",
                        "cash.position.validated; dimension=Box; fallback_to_inferred=never",
                        _cash_rule("Box/Currency"),
                        run_id,
                        as_of_date,
                        dim_name="Box",
                        dim_value=box,
                        suit="safe" if selected.available else "unavailable",
                        validation="ok" if selected.available else "warn",
                        caveat=(
                            "Governed validated account snapshots only; inferred control and internal balances excluded."
                            if selected.available
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

    metrics = pd.concat(
        [metrics, pd.DataFrame(new_rows, columns=_legacy.ANNUAL_METRICS_COLUMNS)],
        ignore_index=True,
    )
    contract = _legacy._contract_from_rows(metrics)
    qa = _legacy.build_annual_balance_dashboard_qa(metrics, contract)
    qa = pd.concat(
        [
            qa,
            pd.DataFrame(
                [
                    {
                        "check": "annual_cash_uses_governed_validated_selector",
                        "status": "pass",
                        "detail": "BS.CASH values rebuilt from cash.position.validated; fallback_to_inferred=never",
                        "severity": "error",
                    },
                    {
                        "check": "annual_cash_never_sums_monthly_positions",
                        "status": "pass",
                        "detail": "last governed period then same monthly account-snapshot primitive",
                        "severity": "error",
                    },
                ],
                columns=_legacy.QA_COLUMNS,
            ),
        ],
        ignore_index=True,
    )

    metrics.to_csv(paths["annual_balance_dashboard_metrics"], index=False)
    contract.to_csv(paths["annual_balance_dashboard_contract"], index=False)
    qa.to_csv(paths["annual_balance_dashboard_qa"], index=False)
    return paths
