from __future__ import annotations

import pandas as pd

from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.metrics.frontier import build_metrics_frontier


def _cash_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "period": "2026-01", "period_end": "2026-01-31", "as_of_date": "2026-01-31",
            "Box": "Property Management", "party": "", "account_id": "bank-a",
            "Currency": "ARS", "close_amount": 100.0, "position_type": "cash_close",
            "source_type": "bank_statement", "cash_suitability": "frontend_safe",
            "is_frontend_safe": True, "validation_status": "validated", "validated_by": "controller",
        }
    ])


def test_frontier_and_annual_metrics_do_not_recreate_retired_cash_headlines(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    cash = _cash_fixture()
    cash.to_csv(run_root / "monthly_cash_close.csv", index=False)

    frontier = pd.read_csv(
        build_metrics_frontier(run_root, tmp_path / "frontier", "fixture", "2026-01-31")["frontend_metric_series"]
    )
    assert not frontier["metric_id"].isin({"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"}).any()

    annual = pd.read_csv(
        build_annual_balance_dashboard(run_root, tmp_path / "annual", "fixture", "2026-01-31")["annual_balance_dashboard_metrics"]
    )
    assert not annual["metric_id"].isin({"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX", "DQ.CASH.FRONTEND_SAFE"}).any()
