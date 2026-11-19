from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.marts.debt import build_monthly_debt_position


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_debt_fixture(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    debt_dir = tmp_path / "debt"
    out_dir = tmp_path / "debt_out"
    _write(
        debt_dir / "debt_balance_monthly.csv",
        [
            {"as_of_date": "2025-02-28", "period": "2025-02", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 1000, "open_interest": 50, "open_total": 1050},
            {"as_of_date": "2025-03-15", "period": "2025-03", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 900, "open_interest": 30, "open_total": 930},
            {"as_of_date": "2025-03-31", "period": "2025-03", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 850, "open_interest": 20, "open_total": 870},
            {"as_of_date": "not-a-date-a", "period": "2025-04", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 800, "open_interest": 0, "open_total": 800},
            {"as_of_date": "not-a-date-z", "period": "2025-04", "debtor": "PM", "creditor": "MI", "currency": "usd", "open_principal": 700, "open_interest": 0, "open_total": 700},
        ],
    )
    _write(
        debt_dir / "debt_open_items.csv",
        [
            {"opened_at": "2025-02-01", "debtor": "PM", "creditor": "MI", "currency": "usd", "item_type": "Prestamo", "original_amount": 1000},
            {"opened_at": "2025-02-02", "debtor": "PM", "creditor": "MI", "currency": "usd", "item_type": "Interes", "original_amount": 50},
        ],
    )
    _write(
        debt_dir / "debt_repayment_events.csv",
        [
            {"repayment_date": "2025-03-20", "debtor": "PM", "creditor": "MI", "currency": "usd", "allocated_amount": 180},
            {"repayment_date": "2025-04-20", "debtor": "PM", "creditor": "MI", "currency": "usd", "allocated_amount": 170},
        ],
    )
    paths = build_monthly_debt_position(debt_dir, out_dir)
    return (
        pd.read_csv(paths["monthly_debt_position"]),
        pd.read_csv(paths["monthly_debt_activity"]),
    )


def test_debt_position_builder_selects_latest_valid_snapshot_and_exposes_component_contract(
    tmp_path: Path,
) -> None:
    position, _ = _build_debt_fixture(tmp_path)

    march = position[position["period"].eq("2025-03")]
    assert set(march["as_of_date"]) == {"2025-03-31"}
    assert set(march["component"]) == {"principal", "interest", "total"}

    expected = {"principal": 850.0, "interest": 20.0, "total": 870.0}
    for component, amount in expected.items():
        row = march[march["component"].eq(component)].iloc[0]
        assert row["open_amount"] == amount
        assert row["open_principal"] == 850.0
        assert row["open_interest"] == 20.0
        assert row["open_total"] == 870.0


def test_all_invalid_debt_as_of_dates_fail_closed_without_lexical_or_prior_fallback(
    tmp_path: Path,
) -> None:
    position, _ = _build_debt_fixture(tmp_path)
    april = position[position["period"].eq("2025-04")]

    assert set(april["position_status"]) == {"unavailable"}
    assert set(april["valid_as_of_rows"]) == {0}
    assert april["as_of_date"].fillna("").astype(str).eq("").all()
    assert april["open_amount"].isna().all()
    assert april["open_total"].isna().all()
    assert april["selection_reason"].astype(str).str.contains("no valid as_of_date").all()


def test_debt_activity_preserves_sparse_flow_identity_and_known_repayments(
    tmp_path: Path,
) -> None:
    _, activity = _build_debt_fixture(tmp_path)

    assert set(activity["activity_type"]) == {
        "opening_balance",
        "new_claim",
        "interest_accrual",
        "repayment",
        "adjustment",
        "closing_balance",
        "net_change",
    }
    nonzero_specs = {
        "new_principal": "new_claim",
        "interest_accrued": "interest_accrual",
        "repayments": "repayment",
        "adjustments": "adjustment",
        "net_change": "net_change",
    }
    for measure, activity_type in nonzero_specs.items():
        nonzero = activity[
            pd.to_numeric(activity[measure], errors="coerce")
            .fillna(0)
            .abs()
            .gt(1e-9)
        ]
        assert set(nonzero["activity_type"]) <= {activity_type}

    march = activity[
        activity["period"].eq("2025-03")
        & activity["activity_type"].eq("repayment")
    ]
    assert float(march["repayments"].sum()) == 180.0

    april = activity[
        activity["period"].eq("2025-04")
        & activity["activity_type"].eq("repayment")
    ]
    assert float(april["repayments"].sum()) == 170.0
    assert set(april["reconciliation_status"]) == {"unavailable_position"}

    repayments = activity[activity["activity_type"].eq("repayment")]
    assert float(repayments["repayments"].sum()) == 350.0
