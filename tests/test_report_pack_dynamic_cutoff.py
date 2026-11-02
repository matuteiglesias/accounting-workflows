"""Regression for monthly report validation after the original August snapshot."""

from pathlib import Path

import pandas as pd

from accounting.reports.build import _build_pack_validation


def _result(tmp_path: Path, *, cutoff: str = "2026-10-31", period: str = "2026-10", stamped: str = "2026-11-02", scope: str = "FBPM", run_id: str = "20261102T180104Z_FBPM") -> dict[str, str]:
    annual = tmp_path / "annual_balance_dashboard_metrics.csv"
    position = tmp_path / "monthly_debt_position.csv"
    cycles = tmp_path / "family_business_accountability_cycles.csv"
    report = tmp_path / "report.html"
    pd.DataFrame([{"metric_id": "ID.DEBT.TOTAL.OPEN", "period": "2026.0", "Currency": "USD", "value": 100.0, "as_of_date": stamped}]).to_csv(annual, index=False)
    pd.DataFrame([
        {"period": "2026-10", "as_of_date": "2026-10-31", "debtor": "PM", "creditor": "MI", "component": "total", "open_amount": 100.0},
        {"period": "2026-10", "as_of_date": "2026-10-31", "debtor": "PM", "creditor": "Primos", "component": "total", "open_amount": 0.0},
    ]).to_csv(position, index=False)
    pd.DataFrame([{"period": period, "cycle_start": "2026-09-01", "as_of_date": stamped, "Box": "Family Business"}]).to_csv(cycles, index=False)
    report.write_text("<h1>Approved report</h1>")
    frame = _build_pack_validation(run_id=run_id, scope_tag=scope, as_of_date=cutoff, annual_metrics=annual, debt_position=position, source_paths=[annual, cycles, position], report_html=[report])
    return dict(zip(frame["check"], frame["status"]))


def test_october_close_accepts_november_generation_stamp(tmp_path: Path) -> None:
    assert set(_result(tmp_path).values()) == {"pass"}


def test_future_accounting_period_still_fails(tmp_path: Path) -> None:
    assert _result(tmp_path, period="2026-11")["no_period_after_cutoff"] == "fail"


def test_stamp_after_run_day_still_fails(tmp_path: Path) -> None:
    assert _result(tmp_path, stamped="2026-11-03")["no_period_after_cutoff"] == "fail"


def test_close_must_match_latest_debt_month(tmp_path: Path) -> None:
    assert _result(tmp_path, cutoff="2026-08-31")["exact_release_identity"] == "fail"


def test_scope_must_match_run_identity(tmp_path: Path) -> None:
    assert _result(tmp_path, scope="OTHER")["exact_release_identity"] == "fail"
