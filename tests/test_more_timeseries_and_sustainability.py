from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.annual_management.distribution_time_series import (
    operating_vs_distributions_chart,
)
from accounting.reports.specialized.time_series_inputs import monthly_rent_chart
from accounting.reports.treasury_accountability.time_series_inputs import cash_inout_chart
from accounting.reports.short_term_sustainability.render import render_report
from accounting.reports.time_series import TimeSeriesSpec, render_timeseries_svg


def test_ts03_rent_excludes_partial_month_and_never_mixes_currency():
    d = pd.DataFrame([
        {"period": p, "Currency": "ARS", "value": v, "scope": "FBPM",
         "period_basis": "monthly", "source_table": "governed.csv"}
        for p, v in [("2026-01", 100), ("2026-02", 200), ("2026-03", 300)]
    ])
    html, trace = monthly_rent_chart(d, currency="ARS", as_of_date="2026-03-19")
    assert "ts03_monthly_rent_ars" in html
    assert trace.period.tolist() == ["2026-01", "2026-02"]
    assert trace.value.tolist() == [100, 200]


def test_ts04_separate_box_inflows_and_outflows():
    d = pd.DataFrame([
        {"period": p, "Box": "Property Management", "Currency": "ARS",
         "total_cash_in": i, "total_cash_out": o, "reconciliation_status": "reconciled"}
        for p, i, o in [
            ("2026-01", 100, 30), ("2026-02", 110, 40),
            ("2026-03", 150, 60),
        ]
    ])
    html, trace = cash_inout_chart(d, as_of_date="2026-03-15")
    assert "ts04_cash_inout_property_management_ars" in html
    assert trace.period.unique().tolist() == ["2026-01", "2026-02"]
    assert trace.loc[trace.series_id.eq("cash_out"), "value"].sum() == 70
    d.loc[0, "Currency"] = "USD"
    with pytest.raises(ValueError, match="mix"):
        cash_inout_chart(d, as_of_date="2026-03-15")


def test_ts06_uses_exact_governed_operating_results_and_registered_draws():
    d = pd.DataFrame([
        {"period": month, "Currency": "ARS", "statement_line": category,
         "amount": value, "classification_coverage_ratio": 1.0,
         "unknown_amount": 0, "review_required_amount": 0}
        for month, net, drawn in [
            ("2026-01", 80, 5), ("2026-02", -10, 0), ("2026-03", 90, 15)
        ]
        for category, value in [
            ("net_operating", net), ("family_draws_or_distributions", drawn)
        ]
    ])
    html, trace = operating_vs_distributions_chart(d, as_of_date="2026-04-12")
    assert "ts06_operating_vs_distributions" in html
    assert trace.loc[trace.series_id.eq("net_operating"), "value"].tolist() == [80, -10, 90]
    assert trace.loc[trace.series_id.eq("family_draws_or_distributions"), "value"].sum() == 20


def _make_sustainability_fixture(tmp_path: Path):
    base, policy, out = (tmp_path / x for x in ("source-run", "policy", "output"))
    base.mkdir(); policy.mkdir()
    raw = b"pinned all-status canonical snapshot\n"
    (base / "ledger_canonical_all_status.csv").write_bytes(raw)
    months = pd.period_range("2026-05", "2027-04", freq="M")
    rows = []
    for box in ("Property Management", "Family Business"):
        for i, month in enumerate(months):
            now, future = i == 6, i > 6
            present = i < 11
            rows.append({
                "period": str(month), "Box": box, "Currency": "ARS",
                "temporal_status": "partial_month" if now else "projected" if future else "observed",
                "materialized_month": present,
                "total_cash_in": 100 if present else None,
                "total_cash_out": (200 if i == 8 and box == "Property Management" else 20) if present else None,
                "net_cash_flow": (-100 if i == 8 and box == "Property Management" else 80) if present else None,
                "closing_control": 1000 + i * 80 if present else None,
                "planned_cash_operating_cost_due_this_month": (200 if i == 8 else 20) if present else None,
                "actual_or_projected_operating_costs_out": 20 if present else None,
            })
    pd.DataFrame(rows).to_csv(policy / "monthly_policy_inputs.csv", index=False)
    (policy / "policy_inputs_manifest.json").write_text(json.dumps({
        "schema": "accounting.policy_inputs.v1",
        "base_run_id": base.name,
        "base_all_status_sha256": hashlib.sha256(raw).hexdigest(),
        "as_of_date": "2026-11-19",
        "through_date": "2027-04-30",
    }))
    return base, policy, out


def test_ts09_november_starts_forward_window_and_april_is_missing(tmp_path):
    base, policy, out = _make_sustainability_fixture(tmp_path)
    outputs = render_report(
        run_root=base, policy_root=policy, out_dir=out,
        as_of_date="2026-11-19", require_pdf=False,
    )
    data = pd.read_csv(outputs["table"])
    assert data.period.unique().tolist() == list(
        pd.period_range("2026-05", "2027-04", freq="M").astype(str)
    )
    pm = data.query("Box == 'Property Management'")
    assert pm.query("period == '2026-10'").coverage.iloc[0] == "observado"
    assert pm.query("period == '2026-11'").coverage.iloc[0] == "actual parcial"
    assert pm.query("period == '2027-04'").coverage.iloc[0] == "sin plan"
    assert pd.isna(pm.query("period == '2027-04'").cash_in.iloc[0])
    assert pd.isna(pm.query("period == '2027-04'").opex_due.iloc[0])
    assert pm.query("period == '2027-01'").opex_due.iloc[0] == 200
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["historical_months"] == [
        "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"
    ]
    assert manifest["forward_months"] == [
        "2026-11", "2026-12", "2027-01", "2027-02", "2027-03", "2027-04"
    ]
    assert manifest["gap_checks"][0]["minimum_dec_through_last_planned"] == 20
    assert manifest["gap_checks"][0]["missing_plan_months"] == ["2027-04"]
    assert not manifest["financial_sustainability_confirmed"]
    document = outputs["html"].read_text()
    assert document.count('class="governed-timeseries"') == 6
    assert "Facturas vencidas pendientes" in document
    assert "OPEX caja" in document
    trace = pd.read_csv(outputs["trace"])
    assert not trace.period.eq("2027-04").any()
    assert trace.loc[trace.period.eq("2026-11"), "phase"].eq("current").all()


def test_ts09_rejects_mismatched_or_incomplete_pinned_inputs(tmp_path):
    base, policy, out = _make_sustainability_fixture(tmp_path)
    m = json.loads((policy / "policy_inputs_manifest.json").read_text())
    m["through_date"] = "2027-03-31"
    (policy / "policy_inputs_manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="six-month window"):
        render_report(run_root=base, policy_root=policy, out_dir=out,
                      as_of_date="2026-11-19", require_pdf=False)
    m["through_date"] = "2027-04-30"
    m["base_all_status_sha256"] = "bad"
    (policy / "policy_inputs_manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="pinned"):
        render_report(run_root=base, policy_root=policy, out_dir=out,
                      as_of_date="2026-11-19", require_pdf=False)


def test_renderer_calendar_domain_shows_april_without_making_a_zero_point():
    spec = TimeSeriesSpec(
        chart_id="missing_april", title="Forecast coverage",
        frequency="M", mark="line", unit="ARS", scope="FBPM",
        series_ids=("net",), series_labels=("Net",),
        period_domain=("2027-02", "2027-03", "2027-04"),
    )
    points = pd.DataFrame([
        {"period": p, "series_id": "net", "value": v, "unit": "ARS",
         "scope": "FBPM", "phase": "projected", "source_table": "policy.csv"}
        for p, v in [("2027-02", 100), ("2027-03", -40)]
    ])
    svg, trace = render_timeseries_svg(spec, points)
    assert "27/04" in svg
    assert "2027-04" not in set(trace.period)
