from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from accounting.diagnostics.report_product_ownership import audit_report_products


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _file(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run" / "R1_FBPM"
    metrics = tmp_path / "metrics" / "R1_FBPM"
    reports = tmp_path / "reports" / "R1_FBPM"
    rsource = _file(run / "a.csv", b"a\n1\n")
    usource = _file(run / "b.csv", b"b\n2\n")
    msource = _file(metrics / "c.csv", b"c\n3\n")
    output = _file(reports / "first" / "report.html", b"<html>report</html>")
    _write_json(run / "meta" / "stage_D_materialize.json", {"stage": "D.materialize"})
    (run / "meta" / "artifacts.jsonl").write_text(json.dumps({
        "role": "derived", "stage": "D.materialize", "relpath": "a.csv",
    }) + "\n", encoding="utf-8")
    _write_json(metrics / "build_manifest.json", {
        "run_id": "R1_FBPM",
        "frontier_outputs": {"a": str(metrics / "c.csv")},
        "annual_dashboard_outputs": {},
        "source_contract_outputs": {},
    })
    _write_json(reports / "report_catalog.json", {
        "source_run_id": "R1_FBPM",
        "scope_tag": "FBPM",
        "as_of_date": "2026-11-01",
        "reports": [{"report_id": "first", "html": "first/report.html", "pdf": None}],
    })
    _write_json(reports / "first" / "report_manifest.json", {
        "report_id": "first",
        "source_run_id": "R1_FBPM",
        "renderer_version": "first.v1",
        "scope_tag": "FBPM",
        "as_of_date": "2026-11-01",
        "validation_status": "pass",
        "sources": [
            {"path": "run/a.csv", "sha256": rsource["sha256"]},
            {"path": "run/b.csv", "sha256": usource["sha256"]},
            {"path": "metrics/c.csv", "sha256": msource["sha256"]},
        ],
        "outputs": {"html": {"path": "first/report.html", "sha256": output["sha256"]}},
    })
    return run, metrics, reports


def test_inventory_tracks_producers_consumers_and_unattributed_sources(tmp_path: Path) -> None:
    run, metrics, reports = _fixture(tmp_path)
    actual = audit_report_products(run, metrics, reports)
    assert actual["report_count"] == 1
    assert actual["source_count"] == 3
    assert actual["output_count"] == 1
    assert actual["source_owner_gaps"] == ["run/b.csv"]
    inputs = {r["artifact"]: r for r in actual["inputs"]}
    assert inputs["run/a.csv"]["producer"] == "D.materialize"
    assert inputs["metrics/c.csv"]["producer"] == "accounting.metrics.build"
    assert inputs["run/b.csv"]["producer"] is None
    assert all(i["consumers"] == ["first"] for i in inputs.values())


def test_rejects_duplicate_report_ids(tmp_path: Path) -> None:
    run, metrics, reports = _fixture(tmp_path)
    path = reports / "report_catalog.json"
    catalog = json.loads(path.read_text())
    catalog["reports"].append(catalog["reports"][0])
    _write_json(path, catalog)
    with pytest.raises(ValueError, match="duplicate report ID"):
        audit_report_products(run, metrics, reports)


def test_rejects_mixed_execution_id(tmp_path: Path) -> None:
    run, metrics, reports = _fixture(tmp_path)
    manifest_path = reports / "first" / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_run_id"] = "OTHER_RUN"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_report_products(run, metrics, reports)


def test_rejects_checksum_drift(tmp_path: Path) -> None:
    run, metrics, reports = _fixture(tmp_path)
    (run / "a.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_report_products(run, metrics, reports)


def test_rejects_catalog_output_not_in_manifest(tmp_path: Path) -> None:
    run, metrics, reports = _fixture(tmp_path)
    path = reports / "first" / "report_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"]["html"]["path"] = "first/other.html"
    _write_json(path, manifest)
    with pytest.raises((FileNotFoundError, ValueError)):
        audit_report_products(run, metrics, reports)


def test_rejects_path_escape(tmp_path: Path) -> None:
    run, metrics, reports = _fixture(tmp_path)
    path = reports / "first" / "report_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sources"][0]["path"] = "run/../../private.csv"
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="invalid artifact path"):
        audit_report_products(run, metrics, reports)


def test_rejects_latest_symlink_ambiguity(tmp_path: Path) -> None:
    run, metrics, reports = _fixture(tmp_path)
    latest = run.parent / "latest_FBPM"
    latest.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="exact run"):
        audit_report_products(latest, metrics, reports)
