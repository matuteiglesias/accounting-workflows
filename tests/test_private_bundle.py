from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from accounting.reports.private_bundle import compose_private_report_bundle
from scripts.check_private_evidence_source import validate_source


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
    evidence_catalog["scope_tag"] = "PM_EVIDENCE"
    evidence_catalog["reports"][0]["attachments"][0]["sha256"] = hashlib.sha256(attachment.read_bytes()).hexdigest()
    (evidence / "report_catalog.json").write_text(json.dumps(evidence_catalog), encoding="utf-8")

    compose_private_report_bundle(normal_reports_root=normal, evidence_reports_root=evidence, out_root=out)
    combined = json.loads((out / "report_catalog.json").read_text(encoding="utf-8"))
    assert [item["report_id"] for item in combined["reports"]] == ["annual_management", "pm_payment_evidence"]
    assert (out / "pm_payment_evidence" / "evidence" / attachment.name).read_bytes() == attachment.read_bytes()

def test_private_bundle_refuses_cross_run_or_wrong_scope_before_creating_target(tmp_path: Path) -> None:
    normal = tmp_path / "normal"
    evidence = tmp_path / "evidence"
    for root, rid, scope in (
        (normal, "annual_management", "FBPM"),
        (evidence, "pm_payment_evidence", "PM_EVIDENCE"),
    ):
        folder = root / rid
        folder.mkdir(parents=True)
        (folder / "report.html").write_text("<html></html>", encoding="utf-8")
        (folder / "report.pdf").write_bytes(b"%PDF-test")
        catalog = _catalog("RUN_FBPM", rid)
        catalog["scope_tag"] = scope
        (root / "report_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    target = tmp_path / "private"
    evidence_path = evidence / "report_catalog.json"
    catalog = json.loads(evidence_path.read_text(encoding="utf-8"))
    catalog["source_run_id"] = "OLDER_FBPM"
    evidence_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="same source run"):
        compose_private_report_bundle(normal_reports_root=normal, evidence_reports_root=evidence, out_root=target)
    assert not target.exists()

    catalog["source_run_id"] = "RUN_FBPM"
    catalog["scope_tag"] = "HH"
    evidence_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="PM_EVIDENCE scope"):
        compose_private_report_bundle(normal_reports_root=normal, evidence_reports_root=evidence, out_root=target)
    assert not target.exists()

def test_private_viewer_preflight_rejects_stale_source_and_publication_flag(tmp_path: Path) -> None:
    run = tmp_path / "out/run/accounting/20261119T210655Z_FBPM"
    run.mkdir(parents=True)
    ledger = run / "ledger_canonical.csv"
    ledger.write_text("tx_id,amount\na,10\n")
    snap = tmp_path / "snapshot"
    snap.mkdir()
    doc = {
        "artifact": "acct.transaction-evidence@1",
        "source_run_id": run.name,
        "publication_allowed": False,
        "source_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    }
    manifest = snap / "manifest.json"
    manifest.write_text(json.dumps(doc))
    validate_source(run, snap)
    doc["source_run_id"] = "OLD_FBPM"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="source run"):
        validate_source(run, snap)
    doc["source_run_id"] = run.name
    doc["publication_allowed"] = True
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="publication_allowed"):
        validate_source(run, snap)
    doc["publication_allowed"] = False
    doc["source_ledger_sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="SHA256"):
        validate_source(run, snap)
