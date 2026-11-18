from __future__ import annotations

import json
from pathlib import Path

import pytest

from accounting.artifacts.producer_provenance import register_mart_outputs


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "run" / "20261118T200115Z_FBPM"
    root.mkdir(parents=True)
    return root


def _output(root: Path, name: str, content: str = "value\n1\n") -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def _log(root: Path) -> list[dict]:
    return [json.loads(x) for x in (root / "meta" / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]


def test_records_producer_in_canonical_artifact_manifest_and_retries_idempotently(tmp_path: Path) -> None:
    root = _root(tmp_path)
    a, b = _output(root, "monthly_debt_position.csv"), _output(root, "monthly_debt_activity.csv")
    products = {"monthly_debt_position": a, "monthly_debt_activity": b}
    assert len(register_mart_outputs(root, products, stage="E.debt_position")) == 2
    assert register_mart_outputs(root, products, stage="E.debt_position") == []
    log = _log(root)
    assert len(log) == 2
    assert {x["relpath"] for x in log} == {a.name, b.name}
    assert all(x["run_id"] == root.name and x["stage"] == "E.debt_position" for x in log)
    assert all(x["sha256"] and x["artifact_role"] for x in log)


def test_rejects_different_producer_for_same_artifact(tmp_path: Path) -> None:
    root = _root(tmp_path)
    a = _output(root, "monthly_cash_accountability.csv")
    register_mart_outputs(root, {"monthly_cash_accountability": a}, stage="E.treasury_accountability")
    with pytest.raises(ValueError, match="conflicting producer"):
        register_mart_outputs(root, {"monthly_cash_accountability": a}, stage="D.materialize")
    assert len(_log(root)) == 1


def test_same_producer_replay_records_changed_output_hash(tmp_path: Path) -> None:
    root = _root(tmp_path)
    a = _output(root, "monthly_debt_position.csv")
    register_mart_outputs(root, {"monthly_debt_position": a}, stage="E.debt_position")
    a.write_text("value\n2\n")
    assert len(register_mart_outputs(root, {"monthly_debt_position": a}, stage="E.debt_position")) == 1
    assert len(_log(root)) == 2
    assert len({record["sha256"] for record in _log(root)}) == 2
    assert register_mart_outputs(root, {"monthly_debt_position": a}, stage="E.debt_position") == []
    a.write_text("value\n1\n", encoding="utf-8")
    assert len(register_mart_outputs(root, {"monthly_debt_position": a}, stage="E.debt_position")) == 1
    assert len(_log(root)) == 3  # Reverting to a prior hash is still a new revision.


def test_preflight_prevents_partial_artifact_registration(tmp_path: Path) -> None:
    root = _root(tmp_path)
    a = _output(root, "monthly_debt_position.csv")
    with pytest.raises(FileNotFoundError):
        register_mart_outputs(
            root, {"monthly_debt_position": a, "other": root / "missing.csv"},
            stage="E.debt_position",
        )
    assert not (root / "meta" / "artifacts.jsonl").exists()


def test_rejects_duplicate_path_with_different_names(tmp_path: Path) -> None:
    root = _root(tmp_path)
    a = _output(root, "a.csv")
    with pytest.raises(ValueError, match="multiple output names"):
        register_mart_outputs(root, {"a": a, "alias_a": a}, stage="E.debt_position")


def test_rejects_latest_alias(tmp_path: Path) -> None:
    root = tmp_path / "run" / "latest_FBPM"
    root.mkdir(parents=True)
    a = _output(root, "a.csv")
    with pytest.raises(ValueError, match="exact run"):
        register_mart_outputs(root, {"a": a}, stage="E.debt_position")
