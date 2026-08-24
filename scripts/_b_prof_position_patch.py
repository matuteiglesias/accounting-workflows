from pathlib import Path

path = Path('accounting/professional/debt_position_executor.py')
text = path.read_text()
text = text.replace(
    'import pandas as pd\n\nfrom accounting.contracts.debt_position_activity import (\n',
    'import pandas as pd\n\nfrom accounting.debt.position_authority import (\n    select_debt_position,\n    selected_debt_position_rows,\n)\nfrom accounting.contracts.debt_position_activity import (\n',
)
start = text.index('\ndef _latest_period_rows(')
end = text.index('\ndef execute_monthly_debt_position(', start)
text = text[:start] + '\n' + text[end:]
old = '''    selected, valid_count = _latest_valid_as_of(candidates, spec)
    sections = [
        ("Selected monthly close snapshot", selected),
        ("All candidate snapshots in period", candidates),
    ]
    if selected.empty:
        return _unavailable(
            display_value=display_value,
            filters=filters,
            candidates=candidates,
            reason="no valid as_of_date in selected monthly debt-position candidates",
            sections=sections,
        )

    matched = _legacy._num(selected.iloc[0].get(spec.value_ref))
'''
new = '''    selection = select_debt_position(
        candidates,
        period=period,
        annual=False,
        as_of_field=spec.as_of_field,
    )
    selected = selected_debt_position_rows(candidates, selection)
    valid_count = selection.valid_as_of_rows
    sections = [
        ("Selected monthly close snapshot", selected),
        ("All candidate snapshots in period", candidates),
    ]
    if not selection.available or selected.empty:
        return _unavailable(
            display_value=display_value,
            filters=filters,
            candidates=candidates,
            reason=selection.reason,
            sections=sections,
        )

    matched = _legacy._num(selected.iloc[0].get(spec.value_ref))
'''
if old not in text:
    raise SystemExit('monthly selector block not found')
text = text.replace(old, new)
text = text.replace(
    '        "selected_as_of_date": _legacy._norm(selected.iloc[0].get(spec.as_of_field)),\n',
    '        "selected_as_of_date": selection.selected_as_of_date,\n',
    1,
)
old = '''    month_candidates, selected_period = _latest_period_rows(year_candidates)
    filters = {**base_filters, "selected_period": selected_period}
    selected, valid_count = _latest_valid_as_of(month_candidates, spec)
    sections = [
        ("Annual companion row", _legacy._annual_companion_long_row(row, period, display_value)),
        ("Selected annual close row", selected),
        ("Candidates in selected closing period", month_candidates),
        ("Candidate debt position rows in year", year_candidates),
    ]
    if selected.empty:
        return _unavailable(
            display_value=display_value,
            filters=filters,
            candidates=month_candidates,
            reason="latest debt-position period has no valid as_of_date; prior periods are not substituted",
            sections=sections,
        )

    matched = _legacy._num(selected.iloc[0].get(spec.value_ref))
'''
new = '''    selection = select_debt_position(
        year_candidates,
        period=period,
        annual=True,
        as_of_field=spec.as_of_field,
    )
    selected_period = selection.selected_period
    month_candidates = year_candidates.loc[
        year_candidates["period"].astype(str).eq(selected_period)
    ].copy()
    filters = {**base_filters, "selected_period": selected_period}
    selected = selected_debt_position_rows(year_candidates, selection)
    valid_count = selection.valid_as_of_rows
    sections = [
        ("Annual companion row", _legacy._annual_companion_long_row(row, period, display_value)),
        ("Selected annual close row", selected),
        ("Candidates in selected closing period", month_candidates),
        ("Candidate debt position rows in year", year_candidates),
    ]
    if not selection.available or selected.empty:
        return _unavailable(
            display_value=display_value,
            filters=filters,
            candidates=month_candidates,
            reason=selection.reason,
            sections=sections,
        )

    matched = _legacy._num(selected.iloc[0].get(spec.value_ref))
'''
if old not in text:
    raise SystemExit('annual selector block not found')
text = text.replace(old, new)
old_asof = '        "selected_as_of_date": _legacy._norm(selected.iloc[0].get(spec.as_of_field)),\n'
if old_asof not in text:
    raise SystemExit('annual selected asof marker not found')
text = text.replace(old_asof, '        "selected_as_of_date": selection.selected_as_of_date,\n', 1)
path.write_text(text)
