from __future__ import annotations

"""Architecture checks for the intentionally bounded compatibility remainder.

Compatibility is not accounting authority. The professional Wave-4 facade is
retired; the remaining ``drilldown_legacy`` module is allowed only as the stable
orchestration/rendering seam plus explicitly documented presentation routes.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_professional_wave_facade_is_gone_and_legacy_remainder_is_bounded() -> None:
    drilldown_path = ROOT / "accounting" / "professional" / "drilldown.py"
    legacy_path = ROOT / "accounting" / "professional" / "drilldown_legacy.py"
    wave_path = ROOT / "accounting" / "professional" / "drilldown_wave4_base.py"

    drilldown = drilldown_path.read_text(encoding="utf-8")
    legacy = legacy_path.read_text(encoding="utf-8")

    assert not wave_path.exists()
    assert "drilldown_wave4_base" not in drilldown
    assert "drilldown_legacy as _legacy" in drilldown
    assert "Bounded compatibility runtime" in legacy

    for retired in [
        "def _build_cash_control_cell(",
        "def _build_annual_cash_close_companion_cell(",
        "def _build_debt_position_cell(",
        "def _build_debt_activity_cell(",
        "def _build_annual_debt_stock_companion_cell(",
        "def _build_annual_debt_activity_companion_cell(",
        "monthly_tables_diagnostic_box_level_matrix",
    ]:
        assert retired not in legacy


def test_remaining_legacy_delegates_are_not_automatically_public_compatibility() -> None:
    annual = (ROOT / "accounting" / "metrics" / "annual.py").read_text(
        encoding="utf-8"
    )
    companion = (
        ROOT / "accounting" / "professional" / "annual_dashboard_tables.py"
    ).read_text(encoding="utf-8")

    assert "annual_legacy as _legacy" in annual
    assert "LEGACY_COMPAT_EXPORTS" in annual
    assert "annual_dashboard_tables_legacy as _legacy" in companion
    assert "LEGACY_COMPAT_EXPORTS" not in companion
