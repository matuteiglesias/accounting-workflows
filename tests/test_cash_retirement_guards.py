from pathlib import Path


ROOT = Path(__file__).parents[1]


def _live_python() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "accounting").rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_live_code_has_no_retired_cash_authorities_or_fallback() -> None:
    source = _live_python()
    assert "cash_authority" not in source
    assert "cash_projection" not in source
    assert "select_validated_cash_period" not in source
    assert "select_validated_cash_year" not in source
    assert "select_inferred_box_control_period" not in source
    diagnostic = (ROOT / "accounting/professional/derived_metric_executor.py").read_text(encoding="utf-8")
    assert "cash_close" not in diagnostic
    assert "validated_cash_fallback" not in diagnostic
