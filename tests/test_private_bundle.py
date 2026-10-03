from __future__ import annotations

import hashlib
import json
from pathlib import Path

from accounting.reports.private_bundle import compose_private_report_bundle


def _catalog(run_id: str, report_id: str, attachment: str | None = None) -> dict:
    report = {
        "report_id": report_id,
        "title": report_id,
        "description": "synthetic",
        "period_label": "synthetic",
        "sort_order": 10,
        "html": f"{report_id}/report.html",
        "pdf": f"{report_id}/report.pdf",
        "manifest": None,
    }
    if attachment:
        report["attachments"] = [{"path": attachment, "sha256": ""}]
    return {"schema": "accounting_report_catalog.v1", "source_run_id": run_id, "scope_tag": "FBPM", "as_of_date": "2026-08-31", "generated_at_utc": "2026-10-01T00:00:00Z", "reports": [report]}


def test_compose_private_bundle_keeps_allowlisted_attachment(tmp_path: Path) -> None:
    normal = tmp_path / "normal"
    evidence = tmp_path / "evidence"
    out = tmp_path / "private"
    for root, report_id in ((normal, "annual_management"), (evidence, "pm_payment_evidence")):
        (root / report_id).mkdir(parents=True)
        (root / report_id / "report.html").write_text("<!doctype html><html></html>", encoding="utf-8")
        (root / report_id / "report.pdf").write_bytes(b"%PDF-synthetic")
    attachment = evidence / "pm_payment_evidence" / "evidence" / ("a" * 64 + ".pdf")
    attachment.parent.mkdir()
    attachment.write_bytes(b"%PDF-approved")
    catalog = _catalog("RUN_FBPM", "annual_management")
    (normal / "report_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    evidence_catalog = _catalog("RUN_FBPM", "pm_payment_evidence", "pm_payment_evidence/evidence/" + attachment.name)
    evidence_catalog["reports"][0]["attachments"][0]["sha256"] = hashlib.sha256(attachment.read_bytes()).hexdigest()
    (evidence / "report_catalog.json").write_text(json.dumps(evidence_catalog), encoding="utf-8")

    compose_private_report_bundle(normal_reports_root=normal, evidence_reports_root=evidence, out_root=out)
    combined = json.loads((out / "report_catalog.json").read_text(encoding="utf-8"))
    assert [item["report_id"] for item in combined["reports"]] == ["annual_management", "pm_payment_evidence"]
    assert (out / "pm_payment_evidence" / "evidence" / attachment.name).read_bytes() == attachment.read_bytes()
