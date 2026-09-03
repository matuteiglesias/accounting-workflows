from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.build import build_report_bundle
from accounting.reports.pdf import render_pdf
from accounting.reports.publish import publish_report_bundle
from accounting.support.latest import assert_latest_targets_exist, update_scoped_latest


def _write_sources(run_root: Path, metrics_dir: Path) -> None:
    run_root.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "metric_id": "IS.RENT.TOTAL",
                "period": "2026",
                "Currency": "ARS",
                "dimension_name": "",
                "dimension_value": "",
                "value": 10.0,
                "value_status": "available",
                "run_id": run_root.name,
                "as_of_date": "2026-08-25",
            }
        ]
    ).to_csv(metrics_dir / "annual_balance_dashboard_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric_id": "IS.RENT.TOTAL",
                "flow_or_stock": "flow",
                "frontend_suitability": "safe",
                "legacy_flag": False,
                "validation_status": "ok",
            }
        ]
    ).to_csv(metrics_dir / "annual_balance_dashboard_contract.csv", index=False)
    pd.DataFrame(
        [{"check": "annual", "status": "pass", "severity": "error", "detail": "synthetic"}]
    ).to_csv(metrics_dir / "annual_balance_dashboard_qa.csv", index=False)
    pd.DataFrame(
        [
            {
                "period": "2026-08",
                "Box": "Property Management",
                "Currency": "ARS",
                "opening_control": 0.0,
                "total_cash_in": 10.0,
                "total_cash_out": 0.0,
                "net_cash_flow": 10.0,
                "closing_control": 10.0,
                "reconciliation_gap": 0.0,
                "reconciliation_status": "reconciled",
                "validated_cash_status": "unavailable",
                "n_review_required": 0,
                "unknown_cash_in": 0.0,
                "unknown_cash_out": 0.0,
            }
        ]
    ).to_csv(run_root / "monthly_cash_accountability.csv", index=False)
    pd.DataFrame(
        [{"check": "treasury", "status": "pass", "severity": "error", "detail": "synthetic"}]
    ).to_csv(run_root / "monthly_cash_accountability_qa.csv", index=False)
    pd.DataFrame(
        columns=[
            "cycle_id", "cycle_start", "cycle_end", "view_type", "as_of_date",
            "Box", "Currency", "opening_accountability_balance",
            "accountable_receipts", "documented_distributions", "supported_uses",
            "documented_transfers_out", "closing_accountability_balance",
            "validated_cash", "validated_cash_status", "validated_cash_as_of_date",
            "other_documented_custody", "accountability_gap",
            "accountability_gap_status", "n_months", "n_tx", "source_table", "policy_id",
        ]
    ).to_csv(run_root / "family_business_accountability_cycles.csv", index=False)
    pd.DataFrame(
        columns=[
            "period", "repayment_tx_id", "repayment_date", "debtor", "creditor",
            "Currency", "repayment_amount", "allocated_amount", "leftover_amount",
            "allocation_status", "target_debt_id", "target_source_tx_id",
            "target_item_type", "target_opened_at", "target_detail",
            "balance_before", "balance_after",
        ]
    ).to_csv(run_root / "monthly_debt_repayment_detail.csv", index=False)
    pd.DataFrame([{"period": "2026-08", "as_of_date": "2026-08-31"}]).to_csv(run_root / "monthly_debt_position.csv", index=False)
    for name in ("monthly_debt_activity.csv", "cost_allocation_gaps.csv"):
        pd.DataFrame([{"synthetic": True}]).to_csv(run_root / name, index=False)
    for name in ("monthly_debt_position_qa.csv", "monthly_debt_activity_qa.csv", "cost_allocation_gaps_qa.csv"):
        pd.DataFrame([{"check": name, "status": "pass", "severity": "error", "detail": "synthetic"}]).to_csv(run_root / name, index=False)


def _fake_reporter(kind: str):
    def render(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        html = out_dir / "report.html"
        validation = out_dir / "report_validation.csv"
        html.write_text(f"<html>{kind}</html>", encoding="utf-8")
        pd.DataFrame(
            [{"check": kind, "status": "pass", "severity": "error", "detail": "synthetic"}]
        ).to_csv(validation, index=False)
        outputs = {"html": html, "validation": validation}
        if kind == "annual":
            cells = out_dir / "report_cells.csv"
            pd.DataFrame([{"metric_id": "IS.RENT.TOTAL", "raw_value": 10.0}]).to_csv(cells, index=False)
            outputs["cells"] = cells
        else:
            summary = out_dir / "report_series_summary.csv"
            pd.DataFrame([{"Box": "Property Management", "Currency": "ARS"}]).to_csv(summary, index=False)
            outputs["summary"] = summary
        return outputs

    return render


def _fake_pdf(html_path: Path, pdf_path: Path, **_kwargs) -> Path:
    Path(pdf_path).write_bytes(b"%PDF-1.4\nsynthetic\n")
    return Path(pdf_path)


def test_report_bundle_builds_catalog_manifests_and_pdf_from_one_run(tmp_path: Path, monkeypatch) -> None:
    run_id = "20260825T190000Z_FBPM"
    run_root = tmp_path / "out" / "run" / "accounting" / run_id
    metrics_dir = tmp_path / "out" / "metrics" / run_id
    out_dir = tmp_path / "out" / "reports" / run_id
    _write_sources(run_root, metrics_dir)

    monkeypatch.setattr("accounting.reports.build.render_annual", _fake_reporter("annual"))
    monkeypatch.setattr("accounting.reports.build.render_treasury", _fake_reporter("treasury"))
    monkeypatch.setattr("accounting.reports.build.render_debt", _fake_reporter("debt"))
    monkeypatch.setattr("accounting.reports.build.render_pdf", _fake_pdf)

    outputs = build_report_bundle(
        run_root=run_root,
        metrics_dir=metrics_dir,
        out_dir=out_dir,
        scope_tag="FBPM",
        generated_at_utc="2026-08-25T19:00:00Z",
    )

    catalog = json.loads(outputs["catalog"].read_text(encoding="utf-8"))
    assert [row["report_id"] for row in catalog["reports"]] == [
        "annual_management",
        "treasury_accountability",
        "debt_accountability",
    ]
    assert all(row["pdf"].endswith("report.pdf") for row in catalog["reports"])
    assert all(row["manifest"] is None for row in catalog["reports"])

    annual_manifest = json.loads(outputs["annual_manifest"].read_text(encoding="utf-8"))
    assert annual_manifest["source_run_id"] == run_id
    assert all(not Path(source["path"]).is_absolute() for source in annual_manifest["sources"])
    assert {source["path"] for source in annual_manifest["sources"]} == {
        "metrics/annual_balance_dashboard_metrics.csv",
        "metrics/annual_balance_dashboard_contract.csv",
            "metrics/annual_balance_dashboard_qa.csv",
            "run/monthly_debt_repayment_detail.csv",
        }
    treasury_manifest = json.loads(outputs["treasury_manifest"].read_text(encoding="utf-8"))
    assert "run/family_business_accountability_cycles.csv" in {
        source["path"] for source in treasury_manifest["sources"]
    }
    debt_manifest = json.loads(outputs["debt_manifest"].read_text(encoding="utf-8"))
    assert len(debt_manifest["sources"]) == 7
    assert Path(outputs["annual_pdf"]).read_bytes().startswith(b"%PDF-")


def test_report_bundle_rejects_mixed_run_directories(tmp_path: Path) -> None:
    run_root = tmp_path / "out" / "run" / "accounting" / "run_A"
    metrics_dir = tmp_path / "out" / "metrics" / "run_B"
    run_root.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="mix accounting runs"):
        build_report_bundle(
            run_root=run_root,
            metrics_dir=metrics_dir,
            out_dir=tmp_path / "reports",
            scope_tag="FBPM",
            require_pdf=False,
        )


def test_report_bundle_rejects_metrics_with_wrong_embedded_run_id(tmp_path: Path) -> None:
    run_id = "20260825T190000Z_FBPM"
    run_root = tmp_path / "out" / "run" / "accounting" / run_id
    metrics_dir = tmp_path / "out" / "metrics" / run_id
    _write_sources(run_root, metrics_dir)
    metrics = pd.read_csv(metrics_dir / "annual_balance_dashboard_metrics.csv")
    metrics["run_id"] = "different_run"
    metrics.to_csv(metrics_dir / "annual_balance_dashboard_metrics.csv", index=False)

    with pytest.raises(ValueError, match="run_id does not match"):
        build_report_bundle(
            run_root=run_root,
            metrics_dir=metrics_dir,
            out_dir=tmp_path / "reports",
            scope_tag="FBPM",
            require_pdf=False,
        )


def test_pdf_adapter_accepts_explicit_fake_browser(tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    html_path.write_text("<html>report</html>", encoding="utf-8")
    browser = tmp_path / "fake-browser"
    browser.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "arg = next(a for a in sys.argv if a.startswith('--print-to-pdf='))\n"
        "pathlib.Path(arg.split('=',1)[1]).write_bytes(b'%PDF-1.4\\nsynthetic\\n')\n",
        encoding="utf-8",
    )
    browser.chmod(0o755)
    pdf = render_pdf(html_path, tmp_path / "report.pdf", browser_bin=browser)
    assert pdf.read_bytes().startswith(b"%PDF-")


def test_report_publication_copies_only_catalog_html_and_pdf(tmp_path: Path) -> None:
    project = tmp_path
    reports = project / "out" / "reports" / "run_FBPM"
    annual = reports / "annual_management"
    treasury = reports / "treasury_accountability"
    annual.mkdir(parents=True)
    treasury.mkdir(parents=True)
    for directory in (annual, treasury):
        (directory / "report.html").write_text("html", encoding="utf-8")
        (directory / "report.pdf").write_bytes(b"%PDF-1.4\n")
        (directory / "report_manifest.json").write_text("{}\n", encoding="utf-8")
        (directory / "report_validation.csv").write_text("check,status\nx,pass\n", encoding="utf-8")
    (annual / "report_cells.csv").write_text("metric_id,value\nX,1\n", encoding="utf-8")
    (reports / "source_accounting.csv").write_text("secret,value\nx,1\n", encoding="utf-8")
    catalog = {
        "schema": "accounting_report_catalog.v1",
        "source_run_id": "run_FBPM",
        "scope_tag": "FBPM",
        "as_of_date": "2026-08-25",
        "generated_at_utc": "2026-08-25T19:00:00Z",
        "reports": [
            {
                "report_id": "annual_management",
                "title": "Annual",
                "description": "",
                "period_label": "2026",
                "sort_order": 10,
                "html": "annual_management/report.html",
                "pdf": "annual_management/report.pdf",
                "manifest": None,
            },
            {
                "report_id": "treasury_accountability",
                "title": "Treasury",
                "description": "",
                "period_label": "2026",
                "sort_order": 20,
                "html": "treasury_accountability/report.html",
                "pdf": "treasury_accountability/report.pdf",
                "manifest": None,
            },
        ],
    }
    (reports / "report_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    target = publish_report_bundle(
        project_root=project,
        scope_tag="FBPM",
        reports_root=reports,
    )
    published = sorted(
        str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()
    )
    assert published == [
        "annual_management/report.html",
        "annual_management/report.pdf",
        "report_catalog.json",
        "treasury_accountability/report.html",
        "treasury_accountability/report.pdf",
    ]
    assert not any(path.endswith(".csv") for path in published)
    assert not any(path.endswith("report_manifest.json") for path in published)


def test_latest_preflight_leaves_existing_pointer_unchanged_when_later_target_missing(tmp_path: Path) -> None:
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    (base_a / "old").mkdir(parents=True)
    (base_a / "new").mkdir()
    (base_b / "old").mkdir(parents=True)
    update_scoped_latest(base_a, "old", "FBPM")
    update_scoped_latest(base_b, "old", "FBPM")

    with pytest.raises(FileNotFoundError):
        assert_latest_targets_exist([base_a, base_b], "new")

    assert (base_a / "latest_FBPM").resolve() == base_a / "old"
    assert (base_b / "latest_FBPM").resolve() == base_b / "old"


def test_makefile_wires_reports_into_one_atomic_latest_update() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    assert "RUN_REPORTS_DIR := $(RUN_REPORTS_BASE)/$(RUN_ID)" in makefile
    assert ".PHONY: run-reports _run_reports_action" in makefile
    assert "run-reports: _run_reports_action" in makefile
    assert "publish-reports:" in makefile
    assert "run-full: run-canonical" in makefile

    full = makefile.split("run-full: run-canonical\n", 1)[1].split("\n\n\n#", 1)[0]
    for stage in ["run-debt", "run-metrics", "run-reports", "_update_latest", "publish-latest", "publish-reports", "release-check"]:
        assert f"$(MAKE) {stage}" in full

    latest = makefile.split("_update_latest:\n", 1)[1].split("\n\n.PHONY: publish-latest", 1)[0]
    assert '--base "$(RUN_REPORTS_BASE)"' in latest
    assert "_update_latest_core" not in makefile
    assert "update-latest-light" not in makefile
