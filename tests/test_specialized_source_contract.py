from __future__ import annotations

from pathlib import Path

import pytest

from accounting.reports.specialized import round2_views, round3_views, views
from accounting.reports.specialized.source_contract import (
    ROUND1_VIEW_KEYS,
    ROUND2_VIEW_KEYS,
    ROUND3_VIEW_KEYS,
    SOURCE_LOCATIONS,
    UNPUBLISHED_VIEW_KEYS,
    VIEW_REQUIREMENTS,
    source_paths_for_view,
)
from accounting.reports.specialized.spec import REPORT_SPECS


def test_every_supported_report_has_exactly_one_declared_view_contract() -> None:
    keys = [report.view_key for report in REPORT_SPECS]
    assert len(keys) == len(set(keys))
    assert UNPUBLISHED_VIEW_KEYS == {"rent_annual_comparison", "distributions_by_year"}
    assert set(keys) == set(VIEW_REQUIREMENTS) - UNPUBLISHED_VIEW_KEYS
    assert not ROUND1_VIEW_KEYS.intersection(ROUND2_VIEW_KEYS)
    assert not ROUND1_VIEW_KEYS.intersection(ROUND3_VIEW_KEYS)
    assert not ROUND2_VIEW_KEYS.intersection(ROUND3_VIEW_KEYS)
    assert ROUND1_VIEW_KEYS | ROUND2_VIEW_KEYS | ROUND3_VIEW_KEYS == set(VIEW_REQUIREMENTS)
    assert set(views._VIEW_REQUIREMENTS) == ROUND1_VIEW_KEYS
    assert set(round2_views.VIEW_REQUIREMENTS) == ROUND2_VIEW_KEYS
    assert set(round3_views.VIEW_REQUIREMENTS) == ROUND3_VIEW_KEYS


def test_all_input_keys_have_a_real_location_and_a_consistent_namespace(tmp_path: Path) -> None:
    run = tmp_path / "run"
    metrics = tmp_path / "metrics"
    for key, (namespace, filename) in SOURCE_LOCATIONS.items():
        assert namespace in {"run", "metrics"}, key
        assert filename and Path(filename).name == filename, key
        assert ".." not in filename, key
    for view_key, required_keys in VIEW_REQUIREMENTS.items():
        assert required_keys
        assert len(required_keys) == len(set(required_keys))
        assert set(required_keys) <= SOURCE_LOCATIONS.keys(), view_key
        actual = source_paths_for_view(view_key, run, metrics)
        expected = tuple(
            ((run if SOURCE_LOCATIONS[key][0] == "run" else metrics) / SOURCE_LOCATIONS[key][1],
             f"{SOURCE_LOCATIONS[key][0]}/{SOURCE_LOCATIONS[key][1]}")
            for key in required_keys
        )
        assert actual == expected
        assert views.source_paths_for_view(view_key, run, metrics) == expected
        if view_key in ROUND2_VIEW_KEYS:
            assert round2_views.source_paths_for_view(view_key, run, metrics) == expected
        if view_key in ROUND3_VIEW_KEYS:
            assert round3_views.source_paths_for_view(view_key, run, metrics) == expected


def test_no_silent_unknown_specialized_view(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        source_paths_for_view("does_not_exist", tmp_path, tmp_path)
