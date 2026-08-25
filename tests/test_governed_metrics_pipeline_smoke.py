from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from accounting.metrics.build import RETIRED_OUTPUT_DIRS, RETIRED_OUTPUTS
from accounting.metrics.frontier import CANONICAL_FRONTIER_SOURCES


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "fixtures" / "ledger_fixture.csv"


def _run(*args: str) -> None:
    subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _statement_year_currency(
    stmt: pd.DataFrame, line: str
) -> dict[tuple[str, str], float]:
    sub = stmt.loc[stmt["statement_line"].astype(str).eq(line)].copy()
    if sub.empty:
        return {}
    sub["year"] = sub["period"].astype(str).str.slice(0, 4)
    sub["amount"] = pd.to_numeric(sub["amount"], errors="coerce").fillna(0.0)
    grouped = sub.groupby(["year", "Currency"], dropna=False)["amount"].sum()
    return {
        (str(year), str(currency)): float(value)
        for (year, currency), value in grouped.items()
    }


def _annual_year_currency(
    metrics: pd.DataFrame, metric_id: str
) -> dict[tuple[str, str], float]:
    sub = metrics.loc[
        metrics["metric_id"].astype(str).eq(metric_id)
        & metrics["value_status"].astype(str).eq("available")
    ].copy()
    if sub.empty:
        return {}
    sub = sub.loc[sub["period"].astype(str).str.fullmatch(r"\d{4}")].copy()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce").fillna(0.0)
    grouped = sub.groupby(["period", "Currency"], dropna=False)["value"].sum()
    return {
        (str(year), str(currency)): float(value)
        for (year, currency), value in grouped.items()
    }


def _assert_close_dict(
    actual: dict[tuple[str, str], float],
    expected: dict[tuple[str, str], float],
) -> None:
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        assert abs(actual[key] - expected_value) < 1e-6, (
            key,
            actual[key],
            expected_value,
        )


def test_smoke_pipeline_builds_governed_metrics_without_legacy_views(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"

    _run(
        "-m",
        "accounting.ledger.ingest",
        "--mode",
        "smoke",
        "--fixture",
        str(FIXTURE),
        "--out-dir",
        str(run_root),
        "--run-id",
        "smoke-governed-metrics",
    )
    _run(
        "-m",
        "accounting.stage_d.materialize",
        "--out-dir",
        str(run_root),
        "--freq",
        "M",
        "--force",
        "1",
        "--mode",
        "smoke",
        "--run-id",
        "smoke-governed-metrics",
    )
    _run(
        "-m",
        "accounting.metrics.build",
        "--run-root",
        str(run_root),
        "--out-dir",
        str(metrics_dir),
        "--run-id",
        "smoke-governed-metrics",
    )

    required = {
        "metric_contract_frontier.csv",
        "frontend_metric_series.csv",
        "metrics_frontier_qa.csv",
        "annual_balance_dashboard_metrics.csv",
        "annual_balance_dashboard_contract.csv",
        "annual_balance_dashboard_qa.csv",
        "annual_flow_membership.csv",
        "artifact_contracts.csv",
        "source_contract_qa.csv",
        "build_manifest.json",
    }
    assert all((metrics_dir / name).is_file() for name in required)
    assert all(not (metrics_dir / name).exists() for name in RETIRED_OUTPUTS)
    assert all(not (metrics_dir / name).exists() for name in RETIRED_OUTPUT_DIRS)
    assert not (run_root / "views").exists()

    frontier = pd.read_csv(metrics_dir / "metric_contract_frontier.csv")
    series = pd.read_csv(metrics_dir / "frontend_metric_series.csv")
    assert not frontier["legacy_flag"].astype(str).str.lower().isin(
        {"true", "1", "yes", "y"}
    ).any()
    assert not series["legacy_flag"].astype(str).str.lower().isin(
        {"true", "1", "yes", "y"}
    ).any()
    assert set(series["source_table"].dropna().astype(str)).issubset(
        CANONICAL_FRONTIER_SOURCES
    )
    assert series["Currency"].fillna("").astype(str).str.strip().ne("").all()

    stmt = pd.read_csv(run_root / "monthly_operating_statement.csv")
    annual = pd.read_csv(metrics_dir / "annual_balance_dashboard_metrics.csv")
    line_map = {
        "IS.REVENUE.OPERATING": "operating_revenue",
        "IS.OPEX.PROPERTY": "property_opex_true",
        "IS.NET.OPERATING": "net_operating",
        "FUND.CONTRIB.TOTAL": "funding_contributions",
        "DIST.DRAWS.PERSONAL": "family_draws_or_distributions",
        "COV.NET.AFTER_DRAWS": "coverage_after_draws",
    }
    for metric_id, statement_line in line_map.items():
        _assert_close_dict(
            _annual_year_currency(annual, metric_id),
            _statement_year_currency(stmt, statement_line),
        )

    available_cash = annual.loc[
        annual["metric_id"].astype(str).str.startswith("BS.CASH")
        & annual["value_status"].astype(str).eq("available")
    ]
    if not available_cash.empty:
        assert available_cash["source_table"].astype(str).eq(
            "monthly_cash_close.csv"
        ).all()
        assert available_cash["calculation_rule"].astype(str).str.contains(
            "never sum monthly positions", case=False, regex=False
        ).all()
        assert available_cash["source_filter"].astype(str).str.contains(
            "fallback_to_inferred=never", case=False, regex=False
        ).all()
