from __future__ import annotations

import json
from pathlib import Path

from accounting.reports.publish import publish_report_bundle


def test_publication_strips_internal_manifest_even_when_exact_run_catalog_references_it(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "out" / "reports" / "run_FBPM"
    annual = reports / "annual_management"
    annual.mkdir(parents=True)
    (annual / "report.html").write_text("<html>annual</html>", encoding="utf-8")
    (annual / "report.pdf").write_bytes(b"%PDF-1.4\n")
    (annual / "report_manifest.json").write_text('{"internal": true}\n', encoding="utf-8")

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
                "description": "Synthetic",
                "period_label": "2026",
                "sort_order": 10,
                "html": "annual_management/report.html",
                "pdf": "annual_management/report.pdf",
                "manifest": "annual_management/report_manifest.json",
            }
        ],
    }
    (reports / "report_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    target = publish_report_bundle(
        project_root=tmp_path,
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
    ]
    public_catalog = json.loads((target / "report_catalog.json").read_text(encoding="utf-8"))
    assert public_catalog["reports"][0]["manifest"] is None
