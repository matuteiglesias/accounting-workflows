from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.support.io import atomic_write_df
from accounting.support.partitions import load_partitions_json, save_partitions_json


def test_stage_d_delegates_generic_infrastructure_to_shared_support() -> None:
    source = Path("accounting/stage_d/materialize.py").read_text(encoding="utf-8")
    assert "from accounting.support.hashing import sha256_file" in source
    assert "from accounting.support.io import atomic_write_df" in source
    assert "from accounting.support.partitions import load_partitions_json, save_partitions_json" in source
    for forbidden in [
        "def _atomic_write_csv",
        "def _sha256_file",
        "def load_partitions_json",
        "def save_partitions_json",
        "def _write_manifest",
        "import hashlib",
        "import json",
    ]:
        assert forbidden not in source


def test_shared_csv_writer_preserves_stage_d_index_false_shape(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    frame = pd.DataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    atomic_write_df(frame, path, index=False)
    reread = pd.read_csv(path)
    assert list(reread.columns) == ["a", "b"]
    assert reread.to_dict("records") == frame.to_dict("records")


def test_shared_partition_writer_round_trips_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "partitions.json"
    payload = {"freq": "M", "outputs": {"x.csv": {"rows": 3}}}
    save_partitions_json(path, payload)
    assert load_partitions_json(path) == payload
    assert not path.with_suffix(path.suffix + ".tmp").exists()
