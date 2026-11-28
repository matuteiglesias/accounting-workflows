"""Reconcile repayment events by source transaction, not by Box/debtor name.

Cash and debt currency lanes are distinct. Allocation-grain input is collapsed
once per repayment event, never by summing repeated gross event amounts.
"""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd

COLUMNS = ["tx_id","period","Box","cash_direction","cash_category","cash_currency","cash_amount",
           "debt_currency","debt_event_amount","principal_allocated","interest_allocated",
           "unallocated","ccl_rate","matched_native_cash","linked_draw_tx_id","status","note"]
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
    cash = cash.loc[cash.movement_basis.eq("actual_cash")].copy()
    if cash.tx_id.eq("").any() or cash.tx_id.duplicated().any():
        raise ValueError("duplicate or missing treasury repayment transaction identity")
    detail = pd.read_csv(debt_path, dtype=str, keep_default_na=False)
    source_path = root / "ledger_canonical.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False) if source_path.exists() else pd.DataFrame()
    if not source.empty:
        if source.tx_id.duplicated().any():
            raise ValueError("duplicate canonical transaction identity")
        source = source.set_index("tx_id")
    # Only explicit directed draws may link a non-repayment cash category.
    directed = set()
    if not source.empty and "tag" in source:
        directed = set(source.index[source.tag.str.contains(r"(?:^|;)settlement_draw_for:[^;]+(?:;|$)", regex=True)])
    cash = cash.loc[cash.cash_category.eq("debt_repayment") | cash.tx_id.isin(directed)]
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
            # A separate debtor-to-creditor settlement can be fulfilled by an
            # explicitly linked draw. Never repeat its cash in this event.
            tags = str(s.get("tag","")).split(";") if s is not None else []
            links = [v.split(":",1)[1] for v in tags if v.startswith("settlement_via_draw:")]
            if links:
                if len(links)!=1 or "transaction_id" not in source:
                    raise ValueError("invalid settlement draw identity")
                candidates = source.loc[source.transaction_id.str.replace(r"\.0+$","",regex=True).eq(links[0])]
                if len(candidates)!=1:
                    raise ValueError("settlement draw identity not unique")
                draw_id = candidates.index[0]
                if draw_id not in directed or draw_id not in cash.index or draw_id not in events:
                    raise ValueError("linked draw missing from cash/debt events")
                draw = cash.loc[draw_id]
                draw_event = events[draw_id]
                if (draw.cash_category not in {"personal_draws","dividends"} or draw.cash_direction!="out"
                    or draw.receiver!=d.creditor or draw.Currency!=d.Currency or draw.period!=d.period
                    or str(draw.Date)[:10]!=str(d.repayment_date)[:10]
                    or money(draw.amount_out)!=g or draw_event[1]!=g):
                    raise ValueError("linked draw settlement metadata mismatch")
                row.update(linked_draw_tx_id=draw_id,status="constructive_linked" if not l else "constructive_partial",
                           note="Debt settlement via linked draw; no additional Box cash.")

        else:
            direction = c.cash_direction
            if direction not in ("in","out"):
                raise ValueError("invalid actual cash direction")
            amount=money(c.amount_in if direction=="in" else c.amount_out)
            row.update(period=c.period,Box=c.Box,cash_direction=direction,cash_category=c.cash_category,cash_currency=c.Currency,cash_amount=amount,matched_native_cash=Decimal(0))
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
                directed_creditor = [v.split(":",1)[1] for v in str(s.get("tag","")).split(";") if v.startswith("settlement_draw_for:")] if s is not None else []
                receiver_ok = c.receiver == d.creditor or (tid in directed and directed_creditor == [d.creditor] and direction == "out" and c.cash_category in {"personal_draws","dividends"})
                if expected is None or expected!=g or c.period!=d.period or str(c.Date)[:10]!=str(d.repayment_date)[:10] or c.payer!=d.debtor or not receiver_ok:
                    row.update(status="mismatch",note="Currency, amount, date or counterparties disagree")
                else:
                    row.update(matched_native_cash=amount,status="partial" if l else "matched",
                               note=("Directed draw; " if tid in directed else "") + ("event linked; unapplied debt amount retained" if l else "event linked and fully allocated"))
        rows.append(row)
    result = pd.DataFrame(rows,columns=COLUMNS)
    linked = result.loc[result.linked_draw_tx_id.ne(""),"linked_draw_tx_id"]
    if linked.duplicated().any():
        raise ValueError("draw reused by multiple constructive settlements")
    return result
