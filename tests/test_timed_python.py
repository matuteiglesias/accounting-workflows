from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from accounting.diagnostics import timed_python


def test_timed_python_records_success_and_exit_code(tmp_path: Path) -> None:
    timings = tmp_path / "timings.jsonl"
    env = {**os.environ, "ACCOUNTING_TIMINGS_JSONL": str(timings)}
    for code, expected in [("print('ok')", 0), ("import sys; sys.exit(7)", 7)]:
        run = subprocess.run(
            [sys.executable, "-m", "accounting.diagnostics.timed_python", "-c", code],
            cwd=Path(__file__).resolve().parents[1], env=env,
            text=True, capture_output=True,
        )
        assert run.returncode == expected
    records = [json.loads(line) for line in timings.read_text().splitlines()]
    assert len(records) == 2
    assert [record["exit_code"] for record in records] == [0, 7]
    assert all(record["elapsed_seconds"] >= 0 for record in records)
    assert all(record["command"][0] == "-c" for record in records)


def test_timed_python_does_not_write_without_optin(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ACCOUNTING_TIMINGS_JSONL", raising=False)
    monkeypatch.setattr(sys, "argv", ["timed_python", "-c", "print('ok')"])
    assert timed_python.main() == 0
    assert list(tmp_path.iterdir()) == []
