import json
import subprocess
import sys

import pandas as pd
import pytest

from accounting.scope import (
    assert_frame_within_scope,
    box_scope_mask,
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


def test_scope_tag_cli_does_not_import_pandas():
    # Make evaluates this in every invocation; keep its cold-start cheap.
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            "from accounting.scope import canonical_scope_tag, parse_box_scope; "
            "assert canonical_scope_tag(parse_box_scope('Family Business,Property Management')) == 'FBPM'; "
            "assert canonical_scope_tag(parse_box_scope('Household')) == 'HH'; "
            "assert 'pandas' not in sys.modules"
        )],
        capture_output=True, text=True, check=True,
    )
    assert result.returncode == 0


def test_box_scope_mask_legacy_fallback_still_works():
    frame = pd.DataFrame({"amount": [1, 2]})
    result = box_scope_mask(frame, {"Family Business"})
    assert result.tolist() == [True, True]
    assert result.index.equals(frame.index)
