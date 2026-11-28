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
    repayment_detail_path = tmp_path / "monthly_debt_repayment_detail.csv"
    metrics.to_csv(metrics_path, index=False)
    contract.to_csv(contract_path, index=False)
    qa.to_csv(qa_path, index=False)
    pd.DataFrame([
        {
            "period": "2026-01", "repayment_tx_id": "repay-1",
            "repayment_date": "2026-01-12", "debtor": "PM", "creditor": "MI",
            "Currency": "USD", "repayment_amount": 200.0, "allocated_amount": 200.0,
            "leftover_amount": 0.0, "allocation_status": "resolved",
            "target_debt_id": "prestamo::debt-1", "target_source_tx_id": "debt-1",
            "target_item_type": "Prestamo", "target_opened_at": "2025-12-01",
            "target_detail": "Synthetic obligation", "balance_before": 200.0,
            "balance_after": 0.0,
        }
    ]).to_csv(repayment_detail_path, index=False)

    outputs = render_report(
        metrics_path=metrics_path,
        contract_path=contract_path,
        qa_path=qa_path,
        out_dir=tmp_path / "report",
        repayment_detail_path=repayment_detail_path,
    )

    html = outputs["html"].read_text(encoding="utf-8")
    cells = pd.read_csv(outputs["cells"], keep_default_na=False)
    validation = pd.read_csv(outputs["validation"])

    assert html.count('class="report-page"') == 6
    assert "2026 YTD" in html
    assert "Operación en USD" in html
    # Executive five-card story: rent, property costs, result, draws, after-draws.
    # Dividend distributions are a subset of draws, not an additional outflow.
    assert '<div class="kpi-label">OPEX DE PROPIEDAD</div>' in html
    assert '<div class="kpi-label">DISTRIBUCIONES REGISTRADAS</div>' not in html
    assert "Las distribuciones registradas forman parte de los retiros" in html
    assert "Para profundizar en el Viewer" in html
    assert "N/A" in html
    assert "Trazabilidad detallada" in html
    assert "Posición y movimientos de deuda" in html
    assert "repay-1" not in html
    assert "prestamo::debt-1" not in html
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
    assert "Control acumulado de movimientos" in html
    assert "origen cero" in html
    assert "Caja validada" not in html
    assert not cells["metric_id"].isin(
        ["BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX", "DQ.CASH.FRONTEND_SAFE"]
    ).any()
    assert (
        (cells["metric_id"] == "CTRL.BOX.CUM_NET")
        & (cells["value_status"] == "unavailable")
        & (cells["display_value"] == "N/A")
    ).any()


def test_annual_report_renders_box_controls_by_native_currency_without_total(
    tmp_path: Path,
) -> None:
    rows = []

    def add(
        metric_id: str,
        value: float,
        currency: str = "ARS",
        dimension_value: str = "",
    ) -> None:
        rows.append(
            {
                "metric_id": metric_id,
                "period": "2026",
                "Currency": currency,
                "dimension_name": "Box" if dimension_value else "",
                "dimension_value": dimension_value,
                "value": value,
                "value_status": "available",
                "source_table": "annual_balance_dashboard_metrics.csv",
                "run_id": "synthetic_FBPM",
                "as_of_date": "2026-12-31",
            }
        )

    for metric_id, value in [
        ("IS.REVENUE.OPERATING", 100),
        ("IS.RENT.TOTAL", 100),
        ("IS.OPEX.PROPERTY", 20),
        ("IS.NET.OPERATING", 80),
        ("FUND.CONTRIB.TOTAL", 0),
        ("DIST.DRAWS.PERSONAL", 70),
        ("DIST.DIVIDENDS", 10),
        ("COV.NET.AFTER_DRAWS", 10),
        ("COV.SAVINGS_RATE", 0.1),
    ]:
        add(metric_id, value)
    for metric_id, value in [
        ("IS.RENT.TOTAL", 25),
        ("IS.NET.OPERATING", 25),
        ("DIST.DIVIDENDS", 5),
        ("COV.NET.AFTER_DRAWS", 20),
    ]:
        add(metric_id, value, "USD")
    for box, ars_flow, ars_cum, usd_flow, usd_cum in [
        ("Family Business", 100, 500, 10, 40),
        ("Property Management", -25, 75, -2, 8),
    ]:
        add("CTRL.BOX.NET_MOVEMENT", ars_flow, dimension_value=box)
        add("CTRL.BOX.CUM_NET", ars_cum, dimension_value=box)
        add("CTRL.BOX.NET_MOVEMENT", usd_flow, "USD", box)
        add("CTRL.BOX.CUM_NET", usd_cum, "USD", box)

    metrics_path = tmp_path / "metrics.csv"
    contract_path = tmp_path / "contract.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    _contract(sorted(pd.DataFrame(rows)["metric_id"].unique())).to_csv(
        contract_path, index=False
    )

    outputs = render_report(
        metrics_path=metrics_path,
        contract_path=contract_path,
        qa_path=None,
        out_dir=tmp_path / "report",
    )
    html = outputs["html"].read_text(encoding="utf-8")
    cells = pd.read_csv(outputs["cells"], keep_default_na=False)

    assert "Movimiento neto registrado" in html
    assert "Control acumulado de movimientos" in html
    assert "No es saldo bancario" in html or "no es saldo bancario" in html
    control_cells = cells.loc[cells["metric_id"].fillna("").str.startswith("CTRL.BOX")]
    assert set(control_cells["Currency"]) == {"ARS", "USD"}
    assert set(control_cells["dimension_value"]) == {
        "Family Business", "Property Management"
    }
    assert not cells["metric_id"].isin({"BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX"}).any()
    assert "TOTAL ARS" not in html and "TOTAL USD" not in html


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
