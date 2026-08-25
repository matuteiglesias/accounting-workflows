from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PUBLIC_TARGETS = {
    "help",
    "run-env",
    "doctor",
    "validate",
    "clean-derived",
    "publish-latest",
    "publish-reports",
    "release-check",
    "smoke-ingest",
    "smoke-materialize",
    "smoke-core",
    "smoke-full",
    "smoke-usd-ccl-valuation",
    "smoke-usd-ccl-management-flows",
    "run-usd-ccl-valuation",
    "run-usd-ccl-management-flows",
    "run-ingest",
    "run-materialize",
    "run-canonical",
    "run-debt",
    "run-metrics",
    "run-reports",
    "run-full",
    "professional-drilldowns",
    "professional-linked-digest",
}

RETIRED_TARGETS = {
    "ledger",
    "materialize",
    "debt",
    "debt-views",
    "metrics",
    "publish",
    "build-all",
    "run",
    "run-all",
    "run-accounting",
    "run-accounting-full",
    "run-debt-views",
    "run-debt-balance",
    "run-metrics-live",
    "run-dashboard",
    "metrics-from-run",
    "reports-from-run",
    "run-downstream-from-ledger",
    "run-live-light",
    "assert-live-light-no-debt",
    "update-latest-light",
    "smoke",
    "smoke-accounting",
    "smoke-env",
}


def _public_targets(makefile: str) -> set[str]:
    targets = set(re.findall(r"^([A-Za-z0-9][A-Za-z0-9_-]*):", makefile, re.MULTILINE))
    return {target for target in targets if not target.startswith("_")}


def test_makefile_public_surface_is_exact_and_alias_free() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert _public_targets(makefile) == EXPECTED_PUBLIC_TARGETS
    assert RETIRED_TARGETS.isdisjoint(_public_targets(makefile))
    assert ".DEFAULT_GOAL := help" in makefile


def test_stage_targets_are_replayable_and_live_composites_are_explicit() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "run-materialize: _run_materialize_action" in makefile
    assert "run-debt: _run_debt_resolution_action" in makefile
    assert "run-metrics: _run_metrics_action" in makefile
    assert "run-reports: _run_reports_action" in makefile
    assert "run-canonical: run-ingest" in makefile
    assert "run-full: run-canonical" in makefile

    for retired in RETIRED_TARGETS:
        assert f"{retired}:" not in makefile


def test_readme_names_only_current_command_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for command in ["run-canonical", "run-debt", "run-metrics", "run-reports", "run-full"]:
        assert f"make {command}" in readme
    for retired in [
        "run-accounting",
        "run-accounting-full",
        "run-debt-views",
        "run-dashboard",
        "metrics-from-run",
        "reports-from-run",
        "build-all",
    ]:
        assert f"make {retired}" not in readme
