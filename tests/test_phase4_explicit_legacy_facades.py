from __future__ import annotations

import csv
import importlib
from pathlib import Path


REMAINING_FACADES = {
    "accounting.metrics.annual": "accounting.metrics.annual_legacy",
}


def _inventory():
    path = Path("notes/accounting_simplification_phase4_legacy_export_inventory_20260824.csv")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_remaining_migration_facade_has_no_broad_dynamic_reexport() -> None:
    path = Path("accounting/metrics/annual.py")
    text = path.read_text(encoding="utf-8")
    assert "for _name in dir(" not in text
    assert "globals()[_name]" not in text
    assert "LEGACY_COMPAT_EXPORTS" in text


def test_annual_companion_has_internal_fallback_but_no_public_legacy_exports() -> None:
    path = Path("accounting/professional/annual_dashboard_tables.py")
    text = path.read_text(encoding="utf-8")
    assert "annual_dashboard_tables_legacy as _legacy" in text
    assert "LEGACY_COMPAT_EXPORTS" not in text
    assert "build_annual_debt_stock_by_pair =" not in text
    assert "build_annual_debt_activity_by_pair =" not in text
    assert "write_annual_long_and_wide =" not in text


def test_professional_drilldown_is_reunified_without_wave_facade() -> None:
    drilldown = Path("accounting/professional/drilldown.py")
    text = drilldown.read_text(encoding="utf-8")

    assert not Path("accounting/professional/drilldown_wave4_base.py").exists()
    assert "drilldown_wave4_base" not in text
    assert "drilldown_legacy as _legacy" in text
    assert "resolve_annual_flow_membership_spec" in text
    assert "execute_annual_funding_support" in text
    assert "execute_monthly_debt_position" in text
    assert "execute_monthly_debt_activity" in text
    assert "execute_monthly_cash_position" in text
    assert "resolve_fx_drilldown" in text

    module = importlib.import_module("accounting.professional.drilldown")
    assert set(module.LEGACY_COMPAT_EXPORTS) == {
        "DEFAULT_TOLERANCE",
        "INDEX_FILENAME",
        "STATUS_OK",
        "STATUS_UNSUPPORTED",
        "_annual_formula_spec",
        "_build_annual_formula_cell",
        "_cash_bridge_line_spec",
        "_safe_div",
        "_semantic_filter_for_statement_line",
        "row_context_id",
    }


def test_reunified_frontier_has_no_legacy_delegate() -> None:
    frontier = Path("accounting/metrics/frontier.py")
    text = frontier.read_text(encoding="utf-8")

    assert not Path("accounting/metrics/frontier_legacy.py").exists()
    assert "frontier_legacy" not in text
    assert "LEGACY_COMPAT_EXPORTS" not in text
    assert "def build_metrics_frontier(" in text
    assert "iter_validated_monthly_cash_positions" in text


def test_remaining_facade_exports_are_exactly_caller_inventory() -> None:
    rows = _inventory()
    for facade_name, delegate_name in REMAINING_FACADES.items():
        facade = importlib.import_module(facade_name)
        delegate = importlib.import_module(delegate_name)
        expected = {
            row["symbol"]
            for row in rows
            if row["facade"] == facade_name
            and row["ownership"] == "explicit_compat_export"
        }
        assert set(facade.LEGACY_COMPAT_EXPORTS) == expected
        for symbol in expected:
            assert hasattr(facade, symbol), (facade_name, symbol)
            assert getattr(facade, symbol) is getattr(delegate, symbol), (
                facade_name,
                symbol,
            )
