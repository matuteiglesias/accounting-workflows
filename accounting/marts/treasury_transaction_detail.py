from __future__ import annotations

"""Transaction-grain projection of already-governed Box treasury cash.

This module never decides whether a transaction is cash. It receives the same
classified pre-groupby frame used by ``build_monthly_box_treasury_flow`` after
that frame has already been assigned ``movement_basis``, ``cash_direction`` and
cash arithmetic. It only persists the actual-cash population and reconciles it
back to the canonical monthly treasury mart.
"""

from pathlib import Path
from typing import Any

import pandas as pd


TOLERANCE = 0.01
DETAIL_COLUMNS = [
    "tx_id",
    "Date",
    "period",
    "period_end",
    "Box",
    "Currency",
    "movement_basis",
    "cash_direction",
    "cash_category",
    "amount_in",
    "amount_out",
    "net_amount",
    "payer",
    "receiver",
    "Lugar",
    "Detalle",
    "semantic_bucket",
    "semantic_subbucket",
    "direction_source",
    "cash_effect",
    "debt_effect",
    "classification_status",
    "classification_confidence",
    "review_required",
    "rule_id",
    "source_file",
    "source_row",
]
QA_COLUMNS = [
    "check",
    "period",
    "Box",
    "Currency",
    "amount_in_gap",
    "amount_out_gap",
    "net_gap",
    "status",
    "severity",
    "detail",
]


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[list(columns)].isna().any().any():
        raise ValueError(f"atomic treasury detail has non-numeric columns: {columns}")
    return out


def _build_detail(work: pd.DataFrame) -> pd.DataFrame:
    required = {
        "tx_id",
        "Date",
        "period",
        "period_end",
        "Box",
        "Currency",
        "movement_basis",
        "cash_direction",
        "cash_category",
        "amount_in",
        "amount_out",
        "net_amount",
        "direction_source",
    }
    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(f"pre-groupby treasury frame missing atomic detail columns: {missing}")

    detail = work.loc[
        work["movement_basis"].astype(str).eq("actual_cash")
        & work["cash_direction"].astype(str).isin(["in", "out"])
    ].copy()
    for column in DETAIL_COLUMNS:
        if column not in detail.columns:
            detail[column] = ""
    detail = _numeric(detail, ("amount_in", "amount_out", "net_amount"))
    detail = detail[DETAIL_COLUMNS].sort_values(
        ["Box", "Currency", "Date", "tx_id"]
    ).reset_index(drop=True)
    return detail


def _qa(detail: pd.DataFrame, monthly: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        check: str,
        *,
        period: str = "",
        box: str = "",
        currency: str = "",
        in_gap: float = 0.0,
        out_gap: float = 0.0,
        net_gap: float = 0.0,
        detail_text: str,
    ) -> None:
        gap = max(abs(float(in_gap)), abs(float(out_gap)), abs(float(net_gap)))
        rows.append(
            {
                "check": check,
                "period": period,
                "Box": box,
                "Currency": currency,
                "amount_in_gap": in_gap,
                "amount_out_gap": out_gap,
                "net_gap": net_gap,
                "status": "pass" if gap <= tolerance else "fail",
                "severity": "error",
                "detail": detail_text,
            }
        )

    arithmetic_gap = detail["amount_in"] - detail["amount_out"] - detail["net_amount"]
    max_gap = float(arithmetic_gap.abs().max()) if not detail.empty else 0.0
    add(
        "atomic_in_minus_out_equals_net",
        net_gap=max_gap,
        detail_text=f"max_abs_gap={max_gap}",
    )

    bad_source = detail.loc[~detail["direction_source"].astype(str).eq("box_party_match")]
    add(
        "atomic_actual_cash_requires_box_party_match",
        net_gap=float(len(bad_source)),
        detail_text=f"bad_rows={len(bad_source)}",
    )

    actual_monthly = monthly.loc[
        monthly["movement_basis"].astype(str).eq("actual_cash")
    ].copy()
    for column in ("amount_in", "amount_out", "net_amount"):
        actual_monthly[column] = pd.to_numeric(actual_monthly[column], errors="coerce").fillna(0.0)
    expected = (
        actual_monthly.groupby(["period", "Box", "Currency"], dropna=False, as_index=False)
        .agg(
            expected_in=("amount_in", "sum"),
            expected_out=("amount_out", "sum"),
            expected_net=("net_amount", "sum"),
        )
    )
    atomic = (
        detail.groupby(["period", "Box", "Currency"], dropna=False, as_index=False)
        .agg(
            atomic_in=("amount_in", "sum"),
            atomic_out=("amount_out", "sum"),
            atomic_net=("net_amount", "sum"),
        )
    )
    keys = pd.concat(
        [
            expected[["period", "Box", "Currency"]],
            atomic[["period", "Box", "Currency"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    merged = (
        keys.merge(expected, on=["period", "Box", "Currency"], how="left")
        .merge(atomic, on=["period", "Box", "Currency"], how="left")
        .fillna(0.0)
    )
    for _, row in merged.iterrows():
        in_gap = float(row["atomic_in"]) - float(row["expected_in"])
        out_gap = float(row["atomic_out"]) - float(row["expected_out"])
        net_gap = float(row["atomic_net"]) - float(row["expected_net"])
        add(
            "atomic_detail_matches_monthly_treasury",
            period=str(row["period"]),
            box=str(row["Box"]),
            currency=str(row["Currency"]),
            in_gap=in_gap,
            out_gap=out_gap,
            net_gap=net_gap,
            detail_text=(
                f"atomic=({row['atomic_in']},{row['atomic_out']},{row['atomic_net']}); "
                f"monthly=({row['expected_in']},{row['expected_out']},{row['expected_net']})"
            ),
        )

    qa = pd.DataFrame(rows, columns=QA_COLUMNS)
    failures = qa.loc[qa["severity"].eq("error") & qa["status"].eq("fail")]
    if not failures.empty:
        bad = failures[["check", "period", "Box", "Currency", "net_gap"]].to_dict("records")
        raise ValueError(f"atomic treasury transaction detail reconciliation failed: {bad[:10]}")
    return qa


def write_treasury_transaction_detail(
    *,
    work: pd.DataFrame,
    monthly: pd.DataFrame,
    out_dir: Path,
    tolerance: float = TOLERANCE,
) -> dict[str, Path]:
    """Persist actual physical Box cash at transaction grain and hard-reconcile it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = _build_detail(work)
    qa = _qa(detail, monthly, tolerance)
    detail_path = out_dir / "box_treasury_transaction_detail.csv"
    qa_path = out_dir / "box_treasury_transaction_detail_qa.csv"
    detail.to_csv(detail_path, index=False)
    qa.to_csv(qa_path, index=False)
    return {
        "box_treasury_transaction_detail": detail_path,
        "box_treasury_transaction_detail_qa": qa_path,
    }
