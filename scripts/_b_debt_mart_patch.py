from pathlib import Path

path = Path('accounting/marts/debt.py')
text = path.read_text()
text = text.replace(
    'from accounting.debt.resolve import RULE_VERSION\n',
    'from accounting.debt.position_authority import (\n    select_debt_position,\n    selected_debt_position_rows,\n)\nfrom accounting.debt.resolve import RULE_VERSION\n',
)
text = text.replace(
    '    "component",\n    "open_amount",\n',
    '    "component",\n    "position_status",\n    "selection_reason",\n    "candidate_rows",\n    "valid_as_of_rows",\n    "open_amount",\n',
)
old = '''    unique = base.drop_duplicates(
        ["period", "debtor", "creditor", "Currency", "as_of_date"]
    )[
        [
            "period",
            "period_end",
            "as_of_date",
            "debtor",
            "creditor",
            "Currency",
            "open_principal",
            "open_interest",
            "open_total",
        ]
    ].copy()
    unique["__as_of_date"] = pd.to_datetime(unique["as_of_date"], errors="coerce")
    unique = (
        unique.sort_values(["period", "debtor", "creditor", "Currency", "__as_of_date", "as_of_date"], na_position="first")
        .groupby(["period", "debtor", "creditor", "Currency"], dropna=False, as_index=False)
        .tail(1)
        .drop(columns=["__as_of_date"])
        .reset_index(drop=True)
    )
'''
new = '''    candidates = base.drop_duplicates(
        ["period", "debtor", "creditor", "Currency", "as_of_date"]
    )[
        [
            "period",
            "period_end",
            "as_of_date",
            "debtor",
            "creditor",
            "Currency",
            "open_principal",
            "open_interest",
            "open_total",
        ]
    ].copy()
    selected_groups: list[pd.DataFrame] = []
    for _, group in candidates.groupby(
        ["period", "debtor", "creditor", "Currency"],
        dropna=False,
        sort=False,
    ):
        period = str(group.iloc[0]["period"])
        selection = select_debt_position(group, period=period, annual=False)
        if selection.available:
            chosen = selected_debt_position_rows(group, selection).tail(1).copy()
            chosen["position_status"] = "available"
            chosen["selection_reason"] = selection.reason
            chosen["candidate_rows"] = selection.candidate_rows
            chosen["valid_as_of_rows"] = selection.valid_as_of_rows
            selected_groups.append(chosen)
            continue
        unavailable = group.iloc[[0]].copy()
        unavailable["as_of_date"] = ""
        unavailable[["open_principal", "open_interest", "open_total"]] = pd.NA
        unavailable["position_status"] = "unavailable"
        unavailable["selection_reason"] = selection.reason
        unavailable["candidate_rows"] = selection.candidate_rows
        unavailable["valid_as_of_rows"] = selection.valid_as_of_rows
        selected_groups.append(unavailable)
    unique = (
        pd.concat(selected_groups, ignore_index=True)
        if selected_groups
        else pd.DataFrame(columns=[
            "period", "period_end", "as_of_date", "debtor", "creditor", "Currency",
            "open_principal", "open_interest", "open_total", "position_status",
            "selection_reason", "candidate_rows", "valid_as_of_rows",
        ])
    )
'''
if old not in text:
    raise SystemExit('monthly selection block not found')
text = text.replace(old, new)
old = '''        for component, amount_col in [
            ("principal", "open_principal"),
            ("interest", "open_interest"),
            ("total", "open_total"),
        ]:
            rows.append(
                {
                    "period": row["period"],
                    "period_end": row["period_end"],
                    "as_of_date": row["as_of_date"],
                    "debtor": row["debtor"],
                    "creditor": row["creditor"],
                    "Currency": row["Currency"],
                    "component": component,
                    "open_amount": float(row[amount_col]),
                    "open_principal": float(row["open_principal"]),
                    "open_interest": float(row["open_interest"]),
                    "open_total": float(row["open_total"]),
                    "source_table": "debt_balance_monthly.csv",
                    "source_rule_version": RULE_VERSION,
                    "n_open_items": n_open_items,
                    "caveat": "Consumption wrapper over resolved debt balances; debt engine logic is unchanged.",
                    "frontend_suitability": "safe_with_caveat",
                }
            )
'''
new = '''        available = str(row.get("position_status", "available")) == "available"
        for component, amount_col in [
            ("principal", "open_principal"),
            ("interest", "open_interest"),
            ("total", "open_total"),
        ]:
            rows.append(
                {
                    "period": row["period"],
                    "period_end": row["period_end"],
                    "as_of_date": row["as_of_date"],
                    "debtor": row["debtor"],
                    "creditor": row["creditor"],
                    "Currency": row["Currency"],
                    "component": component,
                    "position_status": row.get("position_status", "available"),
                    "selection_reason": row.get("selection_reason", ""),
                    "candidate_rows": int(row.get("candidate_rows", 0)),
                    "valid_as_of_rows": int(row.get("valid_as_of_rows", 0)),
                    "open_amount": float(row[amount_col]) if available else pd.NA,
                    "open_principal": float(row["open_principal"]) if available else pd.NA,
                    "open_interest": float(row["open_interest"]) if available else pd.NA,
                    "open_total": float(row["open_total"]) if available else pd.NA,
                    "source_table": "debt_balance_monthly.csv",
                    "source_rule_version": RULE_VERSION,
                    "n_open_items": n_open_items,
                    "caveat": (
                        "Consumption wrapper over resolved debt balances; latest valid as_of authority applied."
                        if available
                        else f"Governed debt position unavailable: {row.get('selection_reason', '')}; no lexical or prior-period fallback."
                    ),
                    "frontend_suitability": "safe_with_caveat" if available else "unavailable",
                }
            )
'''
if old not in text:
    raise SystemExit('position row block not found')
text = text.replace(old, new)
text = text.replace(
    '    source_total = float(unique["open_total"].sum())\n',
    '    source_total = float(pd.to_numeric(unique["open_total"], errors="coerce").sum())\n',
)
marker = '''        {
            "check": "no_cross_currency_debt_total",
'''
addition = '''        {
            "check": "invalid_as_of_fails_closed",
            "status": (
                "pass"
                if out.empty
                or out.loc[out["position_status"].astype(str).eq("unavailable"), "open_amount"].isna().all()
                else "fail"
            ),
            "detail": f"unavailable_rows={int(out['position_status'].astype(str).eq('unavailable').sum()) if not out.empty else 0}; lexical_fallback=never",
            "severity": "error",
        },
'''
if marker not in text:
    raise SystemExit('QA insertion marker not found')
text = text.replace(marker, addition + marker, 1)
text = text.replace(
    '''    totals["closing_total"] = pd.to_numeric(
        totals["open_amount"], errors="coerce"
    ).fillna(0.0)
''',
    '''    totals["closing_total"] = pd.to_numeric(
        totals["open_amount"], errors="coerce"
    )
''',
)
old = '''    keys["closing_total"] = pd.to_numeric(
        keys["closing_total"], errors="coerce"
    ).fillna(0.0)
    keys = keys.sort_values(["debtor", "creditor", "Currency", "period"]).reset_index(
        drop=True
    )
    keys["opening_total"] = (
        keys.groupby(["debtor", "creditor", "Currency"], dropna=False)["closing_total"]
        .shift(1)
        .fillna(0.0)
    )
    keys["net_change"] = keys["closing_total"] - keys["opening_total"]
'''
new = '''    keys["closing_total"] = pd.to_numeric(
        keys["closing_total"], errors="coerce"
    )
    keys = keys.sort_values(["debtor", "creditor", "Currency", "period"]).reset_index(
        drop=True
    )
    keys["opening_total"] = keys.groupby(
        ["debtor", "creditor", "Currency"], dropna=False
    )["closing_total"].shift(1)
    first_in_pair = keys.groupby(
        ["debtor", "creditor", "Currency"], dropna=False
    ).cumcount().eq(0)
    keys.loc[first_in_pair & keys["opening_total"].isna(), "opening_total"] = 0.0
    keys["net_change"] = keys["closing_total"] - keys["opening_total"]
'''
if old not in text:
    raise SystemExit('activity opening/closing block not found')
text = text.replace(old, new)
old = '''    activity_base["reconciliation_status"] = (
        activity_base["adjustments"]
        .abs()
        .le(0.01)
        .map({True: "reconciled", False: "residual_adjustment_visible"})
    )
'''
new = '''    reconcilable = activity_base[["opening_total", "closing_total"]].notna().all(axis=1)
    activity_base["reconciliation_status"] = "unavailable_position"
    activity_base.loc[reconcilable, "reconciliation_status"] = (
        activity_base.loc[reconcilable, "adjustments"]
        .abs()
        .le(0.01)
        .map({True: "reconciled", False: "residual_adjustment_visible"})
    )
'''
if old not in text:
    raise SystemExit('activity reconciliation block not found')
text = text.replace(old, new)
old = '''        {
            "check": "opening_closing_present",
            "status": (
                "pass"
                if out.empty
                or out[["opening_total", "closing_total"]].notna().all().all()
                else "fail"
            ),
            "detail": f"rows={len(out)}",
            "severity": "error",
        },
'''
new = '''        {
            "check": "opening_closing_present_or_position_unavailable",
            "status": (
                "pass"
                if out.empty
                or (
                    out[["opening_total", "closing_total"]].notna().all(axis=1)
                    | out["reconciliation_status"].astype(str).eq("unavailable_position")
                ).all()
                else "fail"
            ),
            "detail": f"rows={len(out)}; unavailable_position_rows={int(out['reconciliation_status'].astype(str).eq('unavailable_position').sum()) if not out.empty else 0}",
            "severity": "error",
        },
'''
if old not in text:
    raise SystemExit('opening/closing QA block not found')
text = text.replace(old, new)
old = '''        {
            "check": "activity_reconciles_to_position",
            "status": "pass" if recon.abs().le(0.01).all() else "fail",
            "detail": f"max_diff={float(recon.abs().max()) if len(recon) else 0.0}",
            "severity": "error",
        },
'''
new = '''        {
            "check": "activity_reconciles_to_position",
            "status": (
                "pass"
                if recon.loc[activity_base["reconciliation_status"].ne("unavailable_position")].abs().le(0.01).all()
                else "fail"
            ),
            "detail": (
                f"max_diff={float(recon.loc[activity_base['reconciliation_status'].ne('unavailable_position')].abs().max()) if activity_base['reconciliation_status'].ne('unavailable_position').any() else 0.0}; "
                f"unavailable_position_periods={int(activity_base['reconciliation_status'].eq('unavailable_position').sum())}"
            ),
            "severity": "error",
        },
'''
if old not in text:
    raise SystemExit('activity QA reconciliation block not found')
text = text.replace(old, new)
path.write_text(text)
