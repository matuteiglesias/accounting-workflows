from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.reports.annual_management.render import render_report


def _contract(metric_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_id": metric_id,
                "flow_or_stock": "flow",
                "frontend_suitability": "safe",
                "legacy_flag": False,
                "validation_status": "ok",
            }
            for metric_id in metric_ids
        ]
    )


def test_annual_report_renders_exact_governed_cells_without_semantic_leakage(
    tmp_path: Path,
) -> None:
    rows = []

    def add(metric_id: str, value: float | None, currency: str = "ARS", status: str = "available") -> None:
        rows.append(
            {
                "metric_id": metric_id,
                "period": "2026",
                "Currency": currency,
                "dimension_name": "",
                "dimension_value": "",
                "value": value,
                "value_status": status,
                "source_table": "synthetic.csv",
                "run_id": "synthetic_FBPM",
                "as_of_date": "2026-08-25",
            }
        )

    add("IS.REVENUE.OPERATING", 100.0)
    add("IS.RENT.TOTAL", 100.0)
    add("IS.OPEX.PROPERTY", 20.0)
    add("IS.NET.OPERATING", 80.0)
    add("FUND.CONTRIB.TOTAL", 0.0)
    add("DIST.DRAWS.PERSONAL", 70.0)
    add("DIST.DIVIDENDS", 10.0)
    add("COV.NET.AFTER_DRAWS", 10.0)
    add("COV.SAVINGS_RATE", 0.10)
    add("BS.CASH.TOTAL", None, status="unavailable")
    add("IS.RENT.TOTAL", 25.0, currency="USD")
    add("IS.NET.OPERATING", 25.0, currency="USD")
    add("DIST.DIVIDENDS", 5.0, currency="USD")
    add("COV.NET.AFTER_DRAWS", 20.0, currency="USD")

    metrics = pd.DataFrame(rows)
    contract = _contract(sorted(metrics["metric_id"].unique()))
    qa = pd.DataFrame(
        [{"check": "annual_flows_sum_monthly_flows", "status": "pass", "severity": "error", "detail": "synthetic"}]
    )

    metrics_path = tmp_path / "annual_balance_dashboard_metrics.csv"
    contract_path = tmp_path / "annual_balance_dashboard_contract.csv"
    qa_path = tmp_path / "annual_balance_dashboard_qa.csv"
    metrics.to_csv(metrics_path, index=False)
    contract.to_csv(contract_path, index=False)
    qa.to_csv(qa_path, index=False)

    outputs = render_report(
        metrics_path=metrics_path,
        contract_path=contract_path,
        qa_path=qa_path,
        out_dir=tmp_path / "report",
    )

    html = outputs["html"].read_text(encoding="utf-8")
    cells = pd.read_csv(outputs["cells"])
    validation = pd.read_csv(outputs["validation"])

    assert html.count('class="report-page"') == 6
    assert "2026 YTD" in html
    assert "Operación en USD" in html
    assert "No disponible" in html
    assert not (validation["status"] == "fail").any()
    assert not (
        (cells["page_id"] == "summary")
        & (cells["metric_id"] == "ID.DEBT.TOTAL.OPEN")
    ).any()
    assert (
        (cells["metric_id"] == "IS.RENT.BY_PROPERTY")
        & (cells["dimension_value"] == "CABA")
        & (cells["value_status"] == "structural_zero")
    ).any()
    assert (
        (cells["metric_id"] == "BS.CASH.TOTAL")
        & (cells["Currency"] == "ARS")
        & (cells["display_value"] == "No disponible")
    ).any()


def test_annual_report_fails_on_duplicate_metric_grain(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "metric_id": "IS.RENT.TOTAL",
                "period": "2026",
                "Currency": "ARS",
                "dimension_name": "",
                "dimension_value": "",
                "value": 10.0,
                "value_status": "available",
            },
            {
                "metric_id": "IS.RENT.TOTAL",
                "period": "2026",
                "Currency": "ARS",
                "dimension_name": "",
                "dimension_value": "",
                "value": 11.0,
                "value_status": "available",
            },
        ]
    )
    contract = _contract(["IS.RENT.TOTAL"])
    metrics_path = tmp_path / "metrics.csv"
    contract_path = tmp_path / "contract.csv"
    metrics.to_csv(metrics_path, index=False)
    contract.to_csv(contract_path, index=False)

    try:
        render_report(
            metrics_path=metrics_path,
            contract_path=contract_path,
            qa_path=None,
            out_dir=tmp_path / "report",
        )
    except ValueError as exc:
        assert "validation failed" in str(exc)
    else:
        raise AssertionError("duplicate annual metric grain must fail closed")
