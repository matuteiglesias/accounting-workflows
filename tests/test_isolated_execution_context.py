from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from accounting.support.isolated_execution import isolated_accounting_execution


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    return repo


def test_isolated_workspace_disappears_and_is_outside_checkout(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with isolated_accounting_execution(repo, temp_parent=tmp_path) as execution:
        workspace = execution.workspace
        assert not workspace.is_relative_to(repo)
        assert execution.run_id.endswith("_FBPM")
        assert execution.canonical_root.is_relative_to(workspace)
        assert execution.reports_root.is_relative_to(workspace)
        (execution.out_root / "test").write_text("temporary")
    assert not workspace.exists()


def test_exception_still_removes_workspace(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    workspace = None
    with pytest.raises(RuntimeError, match="synthetic"):
        with isolated_accounting_execution(repo, temp_parent=tmp_path) as execution:
            workspace = execution.workspace
            raise RuntimeError("synthetic failure")
    assert workspace is not None and not workspace.exists()


@pytest.mark.parametrize("forbidden", [
    "run-full", "run-canonical", "run-ingest", "publish-latest",
    "publish-reports", "_update_latest", "professional-drilldowns", "clean-derived",
])
def test_live_and_publication_targets_are_rejected(tmp_path: Path, forbidden: str) -> None:
    repo = _repo(tmp_path)
    with isolated_accounting_execution(repo, temp_parent=tmp_path) as execution:
        with pytest.raises(ValueError, match="not authorized"):
            execution.make_command(forbidden)


def test_stage_command_uses_temp_out_and_explicit_run_identity(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    calls = []
    def fake_run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr("accounting.support.isolated_execution.subprocess.run", fake_run)
    with isolated_accounting_execution(repo, temp_parent=tmp_path) as execution:
        execution.run_stage("run-debt")
        cmd, opts = calls[0]
        assert "run-debt" in cmd
        assert f"OUT={execution.out_root}" in cmd
        assert f"RUN_ID={execution.run_id}" in cmd
        assert opts["cwd"] == repo
        assert opts["check"] is True
        assert "run-full" not in cmd and "publish-latest" not in cmd


def test_rejects_unknown_box_scope(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with isolated_accounting_execution(repo, temp_parent=tmp_path) as execution:
        with pytest.raises(ValueError, match="FBPM only"):
            execution.make_command("run-reports", boxes="Household")
        with pytest.raises(ValueError, match="FBPM only"):
            execution.make_command("run-reports", boxes='FBPM"; echo unsafe')


def test_export_is_explicit_and_survives_cleanup(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    dest = tmp_path / "report.html"
    with isolated_accounting_execution(repo, temp_parent=tmp_path) as execution:
        execution.reports_root.mkdir(parents=True)
        source = execution.reports_root / "report.html"
        source.write_text("<html>simulated</html>", encoding="utf-8")
        execution.export(source, dest)
        with pytest.raises(FileExistsError, match="refusing overwrite"):
            execution.export(source, dest)
        with pytest.raises(ValueError, match="outlive"):
            execution.export(source, execution.workspace / "transient.html")
        with pytest.raises(ValueError, match="production"):
            (repo / "out").mkdir()
            execution.export(source, repo / "out" / "never.html")
    assert dest.read_text(encoding="utf-8") == "<html>simulated</html>"


def test_workspace_cannot_live_inside_repo(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="inside the repository"):
        with isolated_accounting_execution(repo, temp_parent=repo):
            pass
    assert not list(repo.glob("accounting-isolated-*"))
