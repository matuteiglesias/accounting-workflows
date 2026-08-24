from pathlib import Path

path = Path('tests/test_position_activity_marts.py')
text = path.read_text()
old = '''def test_known_defect_all_invalid_debt_as_of_dates_fail_open_lexically(
    tmp_path: Path,
) -> None:
    """P0 fixture: preserve the reproduced mart defect until the authority repair lands."""

    position, _ = _build_debt_fixture(tmp_path)
    april = position[position["period"].eq("2025-04")]

    # This assertion is intentionally the current wrong behavior. Do not delete
    # it as cleanup: the repair must invert it to unavailable/no lexical close.
    assert set(april["as_of_date"]) == {"not-a-date-z"}
    assert april[april["component"].eq("total")].iloc[0]["open_amount"] == 700.0
'''
new = '''def test_all_invalid_debt_as_of_dates_fail_closed_without_lexical_or_prior_fallback(
    tmp_path: Path,
) -> None:
    position, _ = _build_debt_fixture(tmp_path)
    april = position[position["period"].eq("2025-04")]

    assert set(april["position_status"]) == {"unavailable"}
    assert set(april["valid_as_of_rows"]) == {0}
    assert april["as_of_date"].fillna("").astype(str).eq("").all()
    assert april["open_amount"].isna().all()
    assert april["open_total"].isna().all()
    assert april["selection_reason"].astype(str).str.contains("no valid as_of_date").all()
'''
if old not in text:
    raise SystemExit('known-defect test block not found')
text = text.replace(old, new)
old = '''    repayments = activity[activity["activity_type"].eq("repayment")]
    assert float(repayments["repayments"].sum()) == 350.0
'''
new = '''    april = activity[
        activity["period"].eq("2025-04")
        & activity["activity_type"].eq("repayment")
    ]
    assert float(april["repayments"].sum()) == 170.0
    assert set(april["reconciliation_status"]) == {"unavailable_position"}

    repayments = activity[activity["activity_type"].eq("repayment")]
    assert float(repayments["repayments"].sum()) == 350.0
'''
if old not in text:
    raise SystemExit('activity assertion marker not found')
text = text.replace(old, new)
path.write_text(text)
