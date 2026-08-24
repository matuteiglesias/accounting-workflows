from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.metrics.annual import build_annual_balance_dashboard


def test_annual_savings_rate_zero_denominator_is_not_applicable(tmp_path: Path) -> None:
    """Annual ratio semantics are governed independently from legacy drilldown formulas."""

    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"
    run_root.mkdir()
    pd.DataFrame(
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "statement_line": "net_operating",
                "amount": 0.0,
            },
            {
                "period": "2026-01",
                "Currency": "ARS",
                "statement_line": "coverage_after_draws",
                "amount": 100.0,
            },
        ]
    ).to_csv(run_root / "monthly_operating_statement.csv", index=False)

    paths = build_annual_balance_dashboard(
        run_root, metrics_dir, "fixture", "2026-01-31"
    )
    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    savings = metrics[
        metrics["metric_id"].eq("COV.SAVINGS_RATE")
        & metrics["period"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .eq("2026")
        & metrics["Currency"].eq("ARS")
    ]

    assert len(savings) == 1
    assert savings.iloc[0]["value_status"] == "not_applicable"
    assert pd.isna(savings.iloc[0]["value"])
