from pathlib import Path

path = Path('accounting/metrics/annual_legacy.py')
text = path.read_text()
text = text.replace(
    'import pandas as pd\n\nfrom accounting.contracts.semantic_measures import resolve_semantic_measure\n',
    'import pandas as pd\n\nfrom accounting.debt.position_authority import (\n    select_debt_position,\n    selected_debt_position_rows,\n)\nfrom accounting.contracts.semantic_measures import resolve_semantic_measure\n',
)
old = '''    if debt is not None and not debt.empty:
        d=_year(debt); d["open_amount"]=pd.to_numeric(d.get("open_amount",0),errors="coerce").fillna(0.0); last=d.sort_values("period_end").groupby(["period","Currency","debtor","creditor","component"],dropna=False).tail(1)
        comp={"total":"ID.DEBT.TOTAL.OPEN","principal":"ID.DEBT.PRINCIPAL.OPEN","interest":"ID.DEBT.INTEREST.OPEN"}
        for component,mid in comp.items():
            for _,r in last[last.component.astype(str).eq(component)].groupby(["period","Currency"],dropna=False)["open_amount"].sum().reset_index().iterrows(): rows.append(_base(mid,r.period,r.Currency,r.open_amount,"available","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv",f"component={component}","annual debt stock = last monthly close in year; sum counterparties by currency",run_id,as_of_date,suit="safe_with_caveat",caveat="Debt is stock, not flow; not mixed into operating result."))
        for _,r in last[last.component.astype(str).eq("total")].iterrows(): rows.append(_base("ID.DEBT.OPEN.BY_COUNTERPARTY",r.period,r.Currency,r.open_amount,"available","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv","component=total; dimension=debtor_creditor","annual stock = last monthly close by year, debtor, creditor, currency",run_id,as_of_date,dim_name="debtor_creditor",dim_value=f"{r.debtor} -> {r.creditor}",suit="safe_with_caveat"))
        for _,g in last[last.component.astype(str).eq("total")].groupby(["period","Currency"],dropna=False):
            val=g.loc[g.creditor.astype(str).str.contains("Property Management",case=False,na=False),"open_amount"].sum()-g.loc[g.debtor.astype(str).str.contains("Property Management",case=False,na=False),"open_amount"].sum(); rows.append(_base("ID.DEBT.NET_PM_POSITION",_[0],_[1],val,"available","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv","component=total; PM creditor less PM debtor","annual stock = last monthly close net PM position by currency",run_id,as_of_date,suit="safe_with_caveat"))
    else:
        for mid in ["ID.DEBT.TOTAL.OPEN","ID.DEBT.OPEN.BY_COUNTERPARTY","ID.DEBT.PRINCIPAL.OPEN","ID.DEBT.INTEREST.OPEN","ID.DEBT.NET_PM_POSITION"]: unavailable(mid,"monthly_debt_position.csv","4. Internal debt and claims","stock")
'''
new = '''    if debt is not None and not debt.empty:
        d=debt.copy(); d["period"]=d["period"].astype(str); d["year"]=d["period"].str.slice(0,4); d["open_amount"]=pd.to_numeric(d.get("open_amount",pd.NA),errors="coerce")
        selected_records=[]
        for (year,currency,debtor,creditor,component), group in d.groupby(["year","Currency","debtor","creditor","component"],dropna=False,sort=False):
            selection=select_debt_position(group,period=str(year),annual=True)
            if selection.available:
                picked=selected_debt_position_rows(group,selection).tail(1).iloc[0].to_dict()
                picked["value_status"]="available" if pd.notna(picked.get("open_amount")) else "unavailable"
            else:
                closing=group.loc[group["period"].astype(str).eq(selection.selected_period)]
                picked=(closing.iloc[0] if not closing.empty else group.iloc[0]).to_dict()
                picked["as_of_date"]=""
                picked["open_amount"]=pd.NA
                picked["value_status"]="unavailable"
            picked["period"]=str(year)
            picked["selection_reason"]=selection.reason
            picked["selected_period"]=selection.selected_period
            picked["valid_as_of_rows"]=selection.valid_as_of_rows
            selected_records.append(picked)
        last=pd.DataFrame(selected_records)
        comp={"total":"ID.DEBT.TOTAL.OPEN","principal":"ID.DEBT.PRINCIPAL.OPEN","interest":"ID.DEBT.INTEREST.OPEN"}
        for component,mid in comp.items():
            sub=last[last.component.astype(str).eq(component)]
            for (year,currency),g in sub.groupby(["period","Currency"],dropna=False):
                complete=g["value_status"].astype(str).eq("available").all() and g["open_amount"].notna().all()
                value=float(g["open_amount"].sum()) if complete else pd.NA
                caveat="Debt is stock, not flow; latest period then latest valid as_of_date; no prior-period or lexical fallback."
                if not complete: caveat += " At least one closing counterparty position is unavailable."
                rows.append(_base(mid,year,currency,value,"available" if complete else "unavailable","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv",f"component={component}","annual debt stock = latest period then latest valid as_of_date per counterparty; sum only when all closing positions are available",run_id,as_of_date,suit="safe_with_caveat" if complete else "unavailable",validation="ok" if complete else "warn",caveat=caveat))
        totals=last[last.component.astype(str).eq("total")]
        for _,r in totals.iterrows():
            available_row=str(r.get("value_status","unavailable"))=="available" and pd.notna(r.get("open_amount"))
            rows.append(_base("ID.DEBT.OPEN.BY_COUNTERPARTY",r.period,r.Currency,float(r.open_amount) if available_row else pd.NA,"available" if available_row else "unavailable","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv","component=total; dimension=debtor_creditor","annual stock = latest period then latest valid as_of_date by debtor, creditor, currency",run_id,as_of_date,dim_name="debtor_creditor",dim_value=f"{r.debtor} -> {r.creditor}",suit="safe_with_caveat" if available_row else "unavailable",validation="ok" if available_row else "warn",caveat="Debt stock selection is governed; prior periods are never substituted for an invalid closing period."))
        for (year,currency),g in totals.groupby(["period","Currency"],dropna=False):
            complete=g["value_status"].astype(str).eq("available").all() and g["open_amount"].notna().all()
            if complete:
                value=g.loc[g.creditor.astype(str).str.contains("Property Management",case=False,na=False),"open_amount"].sum()-g.loc[g.debtor.astype(str).str.contains("Property Management",case=False,na=False),"open_amount"].sum()
            else: value=pd.NA
            rows.append(_base("ID.DEBT.NET_PM_POSITION",year,currency,value,"available" if complete else "unavailable","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv","component=total; PM creditor less PM debtor","annual stock = latest period then latest valid as_of_date; net PM position only when all closing counterparties are available",run_id,as_of_date,suit="safe_with_caveat" if complete else "unavailable",validation="ok" if complete else "warn",caveat="No prior-period or lexical fallback for unavailable closing debt positions."))
    else:
        for mid in ["ID.DEBT.TOTAL.OPEN","ID.DEBT.OPEN.BY_COUNTERPARTY","ID.DEBT.PRINCIPAL.OPEN","ID.DEBT.INTEREST.OPEN","ID.DEBT.NET_PM_POSITION"]: unavailable(mid,"monthly_debt_position.csv","4. Internal debt and claims","stock")
'''
if old not in text:
    raise SystemExit('annual debt block not found')
text = text.replace(old, new)
path.write_text(text)
