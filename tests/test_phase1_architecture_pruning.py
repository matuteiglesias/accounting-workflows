from __future__ import annotations

from pathlib import Path

from accounting.publish.manifest import SCHEMA_NAME, build_public_bundle_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_retired_empty_and_alternate_modules_are_absent() -> None:
    for rel in [
        "accounting/publish/snapshot.py",
        "accounting/debt/models.py",
        "accounting/debt/rules.py",
        "accounting/config.py",
        "accounting/contracts/models.py",
        "accounting/human",
        "accounting/viz",
    ]:
        assert not (REPO_ROOT / rel).exists(), rel


def test_makefile_has_no_live_human_or_front_report_pipeline() -> None:
    make = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for retired in [
        "accounting.human",
        "human_reports",
        "run-human",
        "human-report",
        "front-report",
        "build-report",
        "build-front",
        "_run_human_balance_action",
    ]:
        assert retired not in make
    assert "run-dashboard: run-metrics" in make
    assert "professional-drilldowns:" in make
    assert "professional-linked-digest:" in make


def test_publication_is_metrics_debt_bundle_not_human_report_dependency() -> None:
    src = (REPO_ROOT / "accounting/publish/latest.py").read_text(encoding="utf-8")
    assert "human_latest" not in src
    assert "balance_human_v2" not in src
    assert "story_manifest" not in src
    assert "publish_report" not in src
    assert "publish_presentation" not in src
    assert SCHEMA_NAME == "accounting_public_bundle.v1"
    manifest = build_public_bundle_manifest(
        source_run_id="run-1",
        status="ok",
        source_paths={"metrics_latest": "m", "debt_latest": "d"},
        files=[],
        metrics={},
        debt={},
    )
    assert manifest["schema_name"] == "accounting_public_bundle.v1"
    assert "reports" not in manifest
    assert "navigation" not in manifest


def test_no_flask_runtime_is_owned_by_accounting_source() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (REPO_ROOT / "accounting").rglob("*.py")
    ).lower()
    assert "from flask" not in source
    assert "import flask" not in source
