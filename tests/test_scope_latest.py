from pathlib import Path

import pytest

from accounting.publish.latest import resolve_paths
from accounting.support.latest import update_scoped_latest


@pytest.mark.parametrize("order", [("FBPM", "HH"), ("HH", "FBPM")])
def test_scoped_latest_is_isolated_in_both_orders(tmp_path, order):
    base = tmp_path / "out" / "run" / "accounting"
    targets = {"FBPM": "run_FBPM", "HH": "run_HH"}
    for target in targets.values():
        (base / target).mkdir(parents=True)
    for tag in order:
        update_scoped_latest(base, targets[tag], tag)

    assert (base / "latest_FBPM").resolve() == base / "run_FBPM"
    assert (base / "latest_HH").resolve() == base / "run_HH"
    assert (base / "latest").resolve() == base / "run_FBPM"


def test_publish_resolves_one_scope_and_rejects_mixed_runs(tmp_path):
    out = tmp_path / "out"
    for producer in ["metrics", "debt_resolution"]:
        base = out / producer
        (base / "run_FBPM").mkdir(parents=True)
        update_scoped_latest(base, "run_FBPM", "FBPM")
    run_base = out / "run" / "accounting"
    (run_base / "run_FBPM").mkdir(parents=True)
    update_scoped_latest(run_base, "run_FBPM", "FBPM")

    paths = resolve_paths(tmp_path, Path("public/accounting"), scope_tag="FBPM")
    assert {paths.metrics_latest.name, paths.debt_latest.name, paths.run_latest.name} == {"run_FBPM"}
    assert paths.public_root == tmp_path / "public/accounting/latest_FBPM"

    metrics = out / "metrics"
    (metrics / "other_FBPM").mkdir()
    (metrics / "latest_FBPM").unlink()
    (metrics / "latest_FBPM").symlink_to("other_FBPM", target_is_directory=True)
    with pytest.raises(ValueError, match="mix accounting runs"):
        resolve_paths(tmp_path, Path("public/accounting"), scope_tag="FBPM")
