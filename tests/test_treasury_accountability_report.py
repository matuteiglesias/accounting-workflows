from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from accounting.reports.treasury_accountability.render import render_report


def _row(
    *,
    period: str,
    box: str,
    currency: str,
    opening: float,
    rent: float,
    funding: float,
    taxes: float,
    dividends: float,
    repayment: float,
) -> dict[str, object]:
    total_in = rent + funding
    total_out = taxes + dividends + repayment
    net = total_in - total_out
    closing = opening + net
    return {
        "period": period,
        "period_end": f"{period}-28",
        "control_as_of_date": f"{period}-28",
        "Box": box,
        "Currency": currency,
        "opening_control": opening,
        "rent_in": rent,
        "funding_cash_in": funding,
        "taxes_out": taxes,
        "dividends_out": dividends,
        "debt_repayments_out": repayment,
        "direct_tax_support_non_cash": 0.0,
        "direct_service_support_non_cash": 0.0,
        "other_non_cash_support": 0.0,
        "total_cash_in": total_in,
        "total_cash_out": total_out,
        "net_cash_flow": net,
        "closing_control": closing,
        "box_motor_net": net,
        "box_flow_net": net,
        "reconciliation_gap": 0.0,
        "reconciliation_status": "reconciled",
        "validated_cash_close": None,
        "validated_cash_status": "unavailable",
        "validated_cash_reason": "no_validated_cash_candidates",
        "validated_as_of_date": "",
        "validated_account_count": 0,
        "n_tx": 3,
        "n_review_required": 0,
        "unknown_cash_in": 0.0,
        "unknown_cash_out": 0.0,
    }


def test_treasury_report_renders_three_native_currency_series(tmp_path: Path) -> None:
    rows = [
        _row(period="2026-01", box="Family Business", currency="ARS", opening=0, rent=100, funding=0, taxes=0, dividends=0, repayment=0),
        _row(period="2026-02", box="Family Business", currency="ARS", opening=100, rent=50, funding=0, taxes=0, dividends=0, repayment=0),
        _row(period="2026-01", box="Property Management", currency="ARS", opening=0, rent=200, funding=20, taxes=50, dividends=10, repayment=0),
        _row(period="2026-02", box="Property Management", currency="ARS", opening=160, rent=100, funding=0, taxes=20, dividends=0, repayment=40),
        _row(period="2026-01", box="Property Management", currency="USD", opening=0, rent=380, funding=0, taxes=0, dividends=100, repayment=0),
        _row(period="2026-02", box="Property Management", currency="USD", opening=280, rent=380, funding=0, taxes=0, dividends=0, repayment=500),
    ]
    accountability = pd.DataFrame(rows)
    qa = pd.DataFrame(
        [
            {
                "check": "cash_components_reconcile_to_box_motor",
                "period": "",
                "Box": "",
                "Currency": "",
                "amount": 0.0,
                "status": "pass",
                "severity": "error",
                "detail": "synthetic",
            }
        ]
    )
    source_path = tmp_path / "monthly_cash_accountability.csv"
    qa_path = tmp_path / "monthly_cash_accountability_qa.csv"
    accountability.to_csv(source_path, index=False)
    qa.to_csv(qa_path, index=False)

    outputs = render_report(
        accountability_path=source_path,
        qa_path=qa_path,
        out_dir=tmp_path / "report",
        start_period="2026-01",
    )

    html = outputs["html"].read_text(encoding="utf-8")
    validation = pd.read_csv(outputs["validation"])
    summary = pd.read_csv(outputs["summary"])

    assert "FAMILY BUSINESS" in html
    assert "PROPERTY MANAGEMENT" in html
    assert "Control acumulado y flujo neto mensual · USD" in html
    assert "TOTAL 2026 YTD" in html
    assert "TOTAL PERÍODO" in html
    assert "Movimiento registrado desde origen cero" in html
    assert "No constituyen saldos bancarios" in html
    assert not (validation["status"] == "fail").any()
    assert set(zip(summary["Box"], summary["Currency"])) == {
        ("Family Business", "ARS"),
        ("Property Management", "ARS"),
        ("Property Management", "USD"),
    }
    assert summary["reconciled_rows"].sum() == len(rows)


def test_treasury_report_does_not_require_legacy_validation_columns(tmp_path: Path) -> None:
    row = _row(
        period="2026-01", box="Property Management", currency="ARS", opening=0,
        rent=100, funding=0, taxes=0, dividends=0, repayment=0,
    )
    source_path = tmp_path / "monthly_cash_accountability.csv"
    pd.DataFrame([row]).drop(
        columns=[
            "validated_cash_close", "validated_cash_status", "validated_cash_reason",
            "validated_as_of_date", "validated_account_count",
        ]
    ).to_csv(source_path, index=False)
    outputs = render_report(accountability_path=source_path, qa_path=None, out_dir=tmp_path / "report")
    html = outputs["html"].read_text(encoding="utf-8")
    assert "Caja validada" not in html
    assert "Movimiento registrado desde origen cero" in html


def test_treasury_report_rejects_non_reconciled_source(tmp_path: Path) -> None:
    row = _row(
        period="2026-01",
        box="Property Management",
        currency="ARS",
        opening=0,
        rent=100,
        funding=0,
        taxes=0,
        dividends=0,
        repayment=0,
    )
    row["reconciliation_status"] = "mismatch"
    source_path = tmp_path / "monthly_cash_accountability.csv"
    pd.DataFrame([row]).to_csv(source_path, index=False)

    with pytest.raises(ValueError, match="validation failed"):
        render_report(
            accountability_path=source_path,
            qa_path=None,
            out_dir=tmp_path / "report",
            start_period="2026-01",
        )


def test_future_months_are_grey_in_same_treasury_table(tmp_path: Path) -> None:
    data = pd.DataFrame([
        _row(period='2026-10', box='Property Management', currency='ARS', opening=0,
             rent=100, funding=0, taxes=0, dividends=0, repayment=0),
        _row(period='2026-12', box='Property Management', currency='ARS', opening=100,
             rent=200, funding=0, taxes=0, dividends=0, repayment=0),
    ])
    source = tmp_path / 'monthly_cash_accountability.csv'
    data.to_csv(source, index=False)
    result = render_report(
        accountability_path=source, qa_path=None, out_dir=tmp_path/'report',
        start_period='2026-10', projection_as_of_date='2026-11-19',
        projection_through_date='2027-03-31',
    )
    html = result['html'].read_text(encoding='utf-8')
    assert 'Último cierre observado: 31/10/2026' in html
    assert 'horizonte proyectado: 2027-03-31' in html
    assert 'class="future"><td class="month">2026-11</td>' in html
    assert 'class="future"><td class="month">2026-12</td>' in html
    assert 'class="future"><td class="month">2027-03</td>' in html
    assert 'class="future"><td class="month">2026-10</td>' not in html
    assert 'tr.future td' in html
    assert not (pd.read_csv(result['validation']).status == 'fail').any()


def test_treasury_projection_cli_runs_and_marks_future_as_simulation(tmp_path: Path) -> None:
    """Regression: the renderer's main() must expose both projection flags."""
    data = pd.DataFrame([
        _row(period="2026-10", box="Property Management", currency="ARS", opening=0,
             rent=100, funding=0, taxes=0, dividends=0, repayment=0),
        _row(period="2026-12", box="Property Management", currency="ARS", opening=100,
             rent=200, funding=0, taxes=0, dividends=0, repayment=0),
    ])
    source = tmp_path / "cash.csv"
    data.to_csv(source, index=False)
    command = [
        sys.executable, "-m", "accounting.reports.treasury_accountability.render",
        "--accountability", str(source),
        "--out-dir", str(tmp_path / "future"),
        "--start-period", "2026-10",
        "--projection-as-of-date", "2026-11-19",
        "--projection-through-date", "2027-03-31",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    html = (tmp_path / "future" / "report.html").read_text()
    assert "Último cierre observado: 31/10/2026" in html
    assert "horizonte proyectado: 2027-03-31" in html
    assert 'class="future"><td class="month">2026-11</td>' in html
    assert 'class="future"><td class="month">2027-03</td>' in html
    assert 'class="future"><td class="month">2026-10</td>' not in html
    assert "No constituyen saldos bancarios" in html
    assert not pd.read_csv(tmp_path / "future" / "report_validation.csv")["status"].eq("fail").any()

    # Historical report remains a separate explicit mode (no projection flags).
    observed_command = command[:command.index("--projection-as-of-date")]
    observed_command[observed_command.index("--out-dir") + 1] = str(tmp_path / "historical")
    historical = subprocess.run(observed_command, text=True, capture_output=True, check=False)
    assert historical.returncode == 0, historical.stderr
    observed = (tmp_path / "historical" / "report.html").read_text()
    assert "horizonte proyectado" not in observed
    assert 'class="future"' not in observed
