from __future__ import annotations

"""Compatibility reachability for legacy facade modules.

These assertions do not define accounting meaning. They prevent cleanup from
physically deleting compatibility modules while current facades still import
them for supported historical/minimal-schema paths.

Removal condition: delete this module and then the corresponding legacy modules
when supported producers/artifacts no longer require these compatibility paths
and a repository/notebook/caller usage census is zero.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_facades_remain_explicit_dependencies_while_compatibility_is_supported() -> None:
    drilldown = (ROOT / "accounting" / "professional" / "drilldown.py").read_text(
        encoding="utf-8"
    )
    flow_base = (
        ROOT / "accounting" / "professional" / "drilldown_wave4_base.py"
    ).read_text(encoding="utf-8")
    annual = (ROOT / "accounting" / "metrics" / "annual.py").read_text(
        encoding="utf-8"
    )
    companion = (
        ROOT / "accounting" / "professional" / "annual_dashboard_tables.py"
    ).read_text(encoding="utf-8")

    assert "drilldown_wave4_base as _base" in drilldown
    assert "drilldown_legacy as _legacy" in flow_base
    assert "annual_legacy as _legacy" in annual
    assert "annual_dashboard_tables_legacy as _legacy" in companion
