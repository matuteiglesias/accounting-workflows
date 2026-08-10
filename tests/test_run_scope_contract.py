import json

import pandas as pd
import pytest

from accounting.scope import (
    assert_frame_within_scope,
    load_run_scope,
    scope_metadata,
)


def _write_scope(run_root, boxes):
    meta = run_root / "meta"
    meta.mkdir(parents=True)
    manifest = {"stage": "A.ingest", "params": scope_metadata(set(boxes))}
    (meta / "stage_A_ingest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_run_scope_is_immutable_and_validates_foreign_boxes(tmp_path):
    _write_scope(tmp_path, {"Family Business", "Property Management"})
    scope = load_run_scope(tmp_path)

    assert scope.boxes == ("Family Business", "Property Management")
    assert scope.codes == ("FB", "PM")
    assert scope.tag == "FBPM"

    assert_frame_within_scope(
        pd.DataFrame({"Box": ["Family Business", "Property Management"]}),
        scope,
        source="valid.csv",
        require_box=True,
    )
    with pytest.raises(ValueError, match="outside run scope FBPM.*Household"):
        assert_frame_within_scope(
            pd.DataFrame({"Box": ["Property Management", "Household"]}),
            scope,
            source="leaking.csv",
            require_box=True,
        )


def test_run_scope_rejects_tampered_stage_a_metadata(tmp_path):
    _write_scope(tmp_path, {"Household"})
    path = tmp_path / "meta" / "stage_A_ingest.json"
    manifest = json.loads(path.read_text())
    manifest["params"]["scope_tag"] = "FBPM"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Invalid Stage A scope metadata scope_tag"):
        load_run_scope(tmp_path)
