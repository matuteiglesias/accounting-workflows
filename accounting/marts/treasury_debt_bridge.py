"""Reconcile repayment events by source transaction, not by Box/debtor name.

Cash and debt currency lanes are distinct. Allocation-grain input is collapsed
once per repayment event, never by summing repeated gross event amounts.
"""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd

COLUMNS = ["tx_id","period","Box","cash_direction","cash_currency","cash_amount",
           "debt_currency","debt_event_amount","principal_allocated","interest_allocated",
           "unallocated","ccl_rate","matched_native_cash","status","note"]
CENT = Decimal(".01")
def money(v):
    n = Decimal(str(v))
    if not n.is_finite():
        raise ValueError("nonfinite repayment amount")
    return n.quantize(CENT, rounding=ROUND_HALF_UP)

def load_bridge(root: Path):
    cash_path = root / "box_treasury_transaction_detail.csv"
    debt_path = root / "monthly_debt_repayment_detail.csv"
    if not cash_path.exists() or not debt_path.exists():
        return None
    cash = pd.read_csv(cash_path, dtype=str, keep_default_na=False)
    cash = cash.loc[cash.cash_category.eq("debt_repayment") & cash.movement_basis.eq("actual_cash")]
    if cash.tx_id.eq("").any() or cash.tx_id.duplicated().any():
        raise ValueError("duplicate or missing treasury repayment transaction identity")
    detail = pd.read_csv(debt_path, dtype=str, keep_default_na=False)
    source_path = root / "ledger_canonical.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False) if source_path.exists() else pd.DataFrame()
    if not source.empty:
        if source.tx_id.duplicated().any():
            raise ValueError("duplicate canonical transaction identity")
        source = source.set_index("tx_id")
    events = {}
    if detail.repayment_tx_id.eq("").any():
        raise ValueError("missing debt repayment identity")
    for tid, group in detail.groupby("repayment_tx_id", sort=False):
        fixed = ["period","repayment_date","debtor","creditor","Currency","repayment_amount","leftover_amount"]
        if any(group[col].nunique(dropna=False)!=1 for col in fixed):
            raise ValueError("inconsistent repeated debt event metadata")
        first = group.iloc[0]
        gross, leftover = money(first.repayment_amount), money(first.leftover_amount)
        if group.loc[group.target_debt_id.ne(""),"target_debt_id"].duplicated().any():
            raise ValueError("duplicate allocation target within repayment event")
        principal = sum((money(v) for v in group.loc[group.target_item_type.eq("Prestamo"),"allocated_amount"]), Decimal(0))
        interest = sum((money(v) for v in group.loc[group.target_item_type.eq("Interes"),"allocated_amount"]), Decimal(0))
        if min(gross,leftover,principal,interest)<0 or sum((money(v) for v in group.allocated_amount), Decimal(0)) != principal+interest or principal+interest+leftover!=gross:
            raise ValueError("debt repayment allocations do not reconcile")
        events[tid] = (first,gross,principal,interest,leftover)
    cash = cash.set_index("tx_id")
    rows = []
    for tid in sorted(set(cash.index)|set(events)):
        c = cash.loc[tid] if tid in cash.index else None
        e = events.get(tid)
        s = source.loc[tid] if not source.empty and tid in source.index else None
        row = dict.fromkeys(COLUMNS, "")
        row["tx_id"] = tid
        if e:
            d,g,p,i,l = e
            row.update(period=d.period,debt_currency=d.Currency,debt_event_amount=g,principal_allocated=p,interest_allocated=i,unallocated=l)
        if c is None:
            row.update(Box=s.Box if s is not None else "",status="debt_only",note="No actual Box cash detail; review settlement basis, not a missing cash inference.")
        else:
            direction = c.cash_direction
            if direction not in ("in","out"):
                raise ValueError("invalid actual cash direction")
            amount=money(c.amount_in if direction=="in" else c.amount_out)
            row.update(period=c.period,Box=c.Box,cash_direction=direction,cash_currency=c.Currency,cash_amount=amount,matched_native_cash=Decimal(0))
            if not e:
                row.update(status="cash_only",note="No corresponding debt-engine event")
            else:
                expected = amount
                if c.Currency != d.Currency:
                    if c.Currency=="ARS" and d.Currency=="USD" and s is not None and "repay_usd_at_ccl" in str(s.get("tag","")).split(";"):
                        rate=Decimal(str(s.get("CCL","")))
                        if not rate.is_finite() or rate<=0:
                            raise ValueError("invalid explicit CCL bridge")
                        expected=money(amount/rate)
                        row["ccl_rate"]=rate
                    else:
                        expected=None
                if expected is None or expected!=g or c.period!=d.period or str(c.Date)[:10]!=str(d.repayment_date)[:10] or c.payer!=d.debtor or c.receiver!=d.creditor:
                    row.update(status="mismatch",note="Currency, amount, date or counterparties disagree")
                else:
                    row.update(matched_native_cash=amount,status="partial" if l else "matched",
                               note="Event linked; unapplied debt amount retained" if l else "Event linked and fully allocated")
        rows.append(row)
    return pd.DataFrame(rows,columns=COLUMNS)
