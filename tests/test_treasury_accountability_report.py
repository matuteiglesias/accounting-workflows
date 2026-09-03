from __future__ import annotations

from pathlib import Path

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
    assert "No equivale a caja validada" in html
    assert "no disponible" in html
    assert not (validation["status"] == "fail").any()
    assert set(zip(summary["Box"], summary["Currency"])) == {
        ("Family Business", "ARS"),
        ("Property Management", "ARS"),
        ("Property Management", "USD"),
    }
    assert summary["validated_cash_available_rows"].sum() == 0


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
