from pathlib import Path

TEST = r'''from __future__ import annotations

import csv
import importlib
from pathlib import Path


FACADES = {
    "accounting.metrics.annual": "accounting.metrics.annual_legacy",
    "accounting.metrics.frontier": "accounting.metrics.frontier_legacy",
    "accounting.professional.annual_dashboard_tables": "accounting.professional.annual_dashboard_tables_legacy",
    "accounting.professional.drilldown_wave4_base": "accounting.professional.drilldown_legacy",
    "accounting.professional.drilldown": "accounting.professional.drilldown_wave4_base",
}


def _inventory():
    path = Path("notes/accounting_simplification_phase4_legacy_export_inventory_20260824.csv")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_facades_have_no_broad_dynamic_reexport():
    paths = [
        Path("accounting/metrics/annual.py"),
        Path("accounting/metrics/frontier.py"),
        Path("accounting/professional/annual_dashboard_tables.py"),
        Path("accounting/professional/drilldown_wave4_base.py"),
        Path("accounting/professional/drilldown.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "for _name in dir(" not in text, path
        assert "globals()[_name]" not in text, path
        assert "LEGACY_COMPAT_EXPORTS" in text, path


def test_explicit_compat_exports_are_exactly_caller_inventory():
    rows = _inventory()
    for facade_name, delegate_name in FACADES.items():
        facade = importlib.import_module(facade_name)
        delegate = importlib.import_module(delegate_name)
        expected = {
            row["symbol"]
            for row in rows
            if row["facade"] == facade_name and row["ownership"] == "explicit_compat_export"
        }
        assert set(facade.LEGACY_COMPAT_EXPORTS) == expected
        for symbol in expected:
            assert hasattr(facade, symbol), (facade_name, symbol)
            assert getattr(facade, symbol) is getattr(delegate, symbol), (facade_name, symbol)


def test_all_caller_inventory_symbols_remain_importable():
    for row in _inventory():
        facade = importlib.import_module(row["facade"])
        assert hasattr(facade, row["symbol"]), row
        assert int(row["caller_count"]) >= 1
        assert row["callers"]


def test_drilldown_deletion_map_has_explicit_blockers_and_removal_conditions():
    path = Path("notes/accounting_simplification_phase4_drilldown_deletion_map_20260824.csv")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 10
    for row in rows:
        assert row["legacy_route_family"]
        assert row["governed_replacement"]
        assert row["legacy_reachable"]
        assert row["blocker"]
        assert row["removal_condition"]
    blockers = " ".join(row["blocker"] for row in rows)
    assert "FundingSupportSpec" in blockers
    assert "FX grain" in blockers
    assert "annual lineage" in blockers
'''

Path("tests/test_phase4_explicit_legacy_facades.py").write_text(TEST, encoding="utf-8")

state = Path("notes/current_state_map.md")
text = state.read_text(encoding="utf-8")
marker = "## Phase 4 facade ownership (2026-08-24)"
if marker not in text:
    text += f'''\n\n{marker}\n\n- Modern migration facades expose only an explicit repository-caller compatibility surface; broad `dir(delegate) -> globals()` re-exports are forbidden.\n- `accounting.professional.drilldown_legacy` remains a compatibility implementation, not current semantic authority. Its remaining route families and removal blockers are tracked in `notes/accounting_simplification_phase4_drilldown_deletion_map_20260824.csv`.\n- New governed consumers must import modern contracts/executors directly rather than creating new dependencies on `*_legacy` symbols.\n'''
    state.write_text(text, encoding="utf-8")
