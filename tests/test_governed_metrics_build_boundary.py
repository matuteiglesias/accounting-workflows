from __future__ import annotations

from pathlib import Path

from accounting.metrics import build
from accounting.metrics import frontier


def test_governed_metrics_build_removes_retired_outputs(tmp_path: Path) -> None:
    for name in build.RETIRED_OUTPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")
    for name in build.RETIRED_OUTPUT_DIRS:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "stale.csv").write_text("stale\n", encoding="utf-8")

    removed = set(build.remove_retired_outputs(tmp_path))

    assert set(build.RETIRED_OUTPUTS).issubset(removed)
    assert {f"{name}/" for name in build.RETIRED_OUTPUT_DIRS}.issubset(removed)
    assert all(not (tmp_path / name).exists() for name in build.RETIRED_OUTPUTS)
    assert all(not (tmp_path / name).exists() for name in build.RETIRED_OUTPUT_DIRS)


def test_governed_metrics_sources_exclude_retired_universe() -> None:
    canonical = set(build.CANONICAL_SOURCE_NAMES)
    assert canonical == frontier.CANONICAL_FRONTIER_SOURCES
    assert "metric_values.csv" not in canonical
    assert "metric_registry.csv" not in canonical
    assert "daily_cash_position.csv" not in canonical
    assert "views/v_contributions_monthly.csv" not in canonical
    assert "views/v_opex_category_monthly.csv" not in canonical
