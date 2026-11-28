from __future__ import annotations

import json
from pathlib import Path

import pytest

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

@pytest.mark.parametrize("private_entry", [
    {"report_id": "pm_payment_evidence", "attachments": []},
    {"report_id": "annual_management", "attachments": [
        {"path": "annual_management/evidence/secret.pdf", "sha256": "a" * 64}
    ]},
])
def test_public_publisher_rejects_private_evidence_before_touching_public_target(
    tmp_path: Path, private_entry: dict
) -> None:
    reports = tmp_path / "out/reports/RUN_FBPM"
    folder = reports / "annual_management"
    folder.mkdir(parents=True)
    (folder / "report.html").write_text("<html>Test</html>", encoding="utf-8")
    (folder / "report.pdf").write_bytes(b"%PDF-test")
    catalog = {
        "schema": "accounting_report_catalog.v1", "source_run_id": "RUN_FBPM",
        "scope_tag": "FBPM", "as_of_date": "2026-10-26", "generated_at_utc": "2026-11-19T21:00:00Z",
        "reports": [{
            "title": "Test", "description": "Test", "period_label": "2026",
            "sort_order": 10, "html": "annual_management/report.html",
            "pdf": "annual_management/report.pdf", "manifest": None,
            **private_entry
        }],
    }
    (reports / "report_catalog.json").write_text(json.dumps(catalog))
    existing = tmp_path / "public/reports/latest_FBPM"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("existing public state")
    with pytest.raises(ValueError, match="private evidence"):
        publish_report_bundle(project_root=tmp_path, scope_tag="FBPM", reports_root=reports)
    assert (existing / "keep.txt").read_text() == "existing public state"


def test_personal_scope_cannot_be_published(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="Personal accounting scope is private"):
        publish_report_bundle(project_root=tmp_path, scope_tag="PERSONAL")
