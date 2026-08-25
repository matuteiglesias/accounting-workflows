from __future__ import annotations

import json
from pathlib import Path

import pytest

from accounting.reports import REPORT_CATALOG_SCHEMA, REPORT_MANIFEST_SCHEMA
from accounting.reports.catalog import ReportCatalogItem, build_report_catalog, validate_catalog_files
from accounting.reports.common import ensure_relative_bundle_path, sha256_file
from accounting.reports.manifest import ReportOutput, ReportSource, build_report_manifest


def test_report_catalog_orders_items_and_rejects_duplicates() -> None:
    catalog = build_report_catalog(
        source_run_id="run_FBPM",
        scope_tag="FBPM",
        as_of_date="2026-08-25",
        generated_at_utc="2026-08-25T19:00:00Z",
        reports=[
            ReportCatalogItem("treasury", "Treasury", "", "2022-2026", 20, "treasury/report.html"),
            ReportCatalogItem("annual", "Annual", "", "2022-2026", 10, "annual/report.html"),
        ],
    )
    assert catalog["schema"] == REPORT_CATALOG_SCHEMA
    assert [row["report_id"] for row in catalog["reports"]] == ["annual", "treasury"]

    with pytest.raises(ValueError, match="duplicate report_id"):
        build_report_catalog(
            source_run_id="run_FBPM",
            scope_tag="FBPM",
            as_of_date="2026-08-25",
            generated_at_utc="2026-08-25T19:00:00Z",
            reports=[
                ReportCatalogItem("annual", "A", "", "", 10, "a/report.html"),
                ReportCatalogItem("annual", "B", "", "", 20, "b/report.html"),
            ],
        )


def test_bundle_paths_fail_closed() -> None:
    assert ensure_relative_bundle_path("annual/report.html") == "annual/report.html"
    for value in ("", ".", "../secret.csv", "/tmp/report.html"):
        with pytest.raises(ValueError):
            ensure_relative_bundle_path(value)


def test_catalog_validation_requires_referenced_files(tmp_path: Path) -> None:
    (tmp_path / "annual").mkdir()
    (tmp_path / "annual" / "report.html").write_text("ok", encoding="utf-8")
    catalog = build_report_catalog(
        source_run_id="run_FBPM",
        scope_tag="FBPM",
        as_of_date="2026-08-25",
        generated_at_utc="2026-08-25T19:00:00Z",
        reports=[ReportCatalogItem("annual", "Annual", "", "2026", 10, "annual/report.html")],
    )
    validate_catalog_files(catalog, bundle_root=tmp_path)
    catalog["reports"][0]["pdf"] = "annual/report.pdf"
    with pytest.raises(FileNotFoundError):
        validate_catalog_files(catalog, bundle_root=tmp_path)


def test_report_manifest_captures_source_and_output_hashes(tmp_path: Path) -> None:
    source_path = tmp_path / "metrics.csv"
    source_path.write_text("metric_id,value\nIS.RENT.TOTAL,10\n", encoding="utf-8")
    report_dir = tmp_path / "annual"
    report_dir.mkdir()
    html_path = report_dir / "report.html"
    html_path.write_text("<html>10</html>", encoding="utf-8")

    source = ReportSource.from_file(source_path, rows=1)
    output = ReportOutput.from_file(html_path, bundle_root=tmp_path)
    manifest = build_report_manifest(
        report_id="annual_management",
        renderer_version="1",
        source_run_id="run_FBPM",
        scope_tag="FBPM",
        as_of_date="2026-08-25",
        sources=[source],
        outputs={"html": output},
        validation_status="pass",
    )

    assert manifest["schema"] == REPORT_MANIFEST_SCHEMA
    assert manifest["sources"][0]["sha256"] == sha256_file(source_path)
    assert manifest["outputs"]["html"]["path"] == "annual/report.html"
    assert manifest["outputs"]["html"]["sha256"] == sha256_file(html_path)
    json.dumps(manifest)
