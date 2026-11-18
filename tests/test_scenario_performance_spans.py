from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from accounting.scenarios import runner


def test_scenario_stage_timings_are_opt_in(tmp_path: Path, monkeypatch) -> None:
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "STAGES", ("run-debt",))
    execution = SimpleNamespace(
        workspace=tmp_path,
        repo_root=tmp_path,
        make_command=lambda stage: ["make", stage],
    )
    ordinary = runner.run_stages(execution)
    assert ordinary is None
    assert calls[0][0] == ["make", "run-debt"]
    assert calls[0][1]["env"] is None

    measurements: list[dict] = []
    runner.run_stages(execution, measurements=measurements)
    assert measurements[0]["kind"] == "make_stage"
    assert measurements[0]["stage"] == "run-debt"
    assert measurements[0]["elapsed_seconds"] >= 0
    assert calls[1][0][-1] == "PY=python3 -m accounting.diagnostics.timed_python"
    assert calls[1][1]["env"]["ACCOUNTING_TIMINGS_JSONL"] == str(tmp_path / "python_timings.jsonl")


def test_scenario_stage_timings_include_wrapped_subprocess_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "STAGES", ("run-debt",))
    def fake_run(command, **kwargs):
        log = Path(kwargs["env"]["ACCOUNTING_TIMINGS_JSONL"])
        log.write_text('{"kind":"python_subprocess","command":["-m","accounting.debt.balance_views"],'
                       '"elapsed_seconds":8.5,"exit_code":0}\n')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    execution = SimpleNamespace(
        workspace=tmp_path, repo_root=tmp_path,
        make_command=lambda stage: ["make", stage],
    )
    measurements: list[dict] = []
    runner.run_stages(execution, measurements=measurements)
    assert [record["kind"] for record in measurements] == ["make_stage", "python_subprocess"]
    assert measurements[1]["elapsed_seconds"] == 8.5
