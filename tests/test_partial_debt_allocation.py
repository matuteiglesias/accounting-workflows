import pandas as pd

from accounting.debt.balance_views import _last_snapshot_by_period, build_debt_balance_daily
from accounting.debt.resolve import OpenItem, resolve_repayments


def item(debt_id, amount, *, kind="Prestamo", opened="2026-01-01", debtor="A", creditor="PM", currency="USD"):
    return OpenItem(debt_id, debt_id, opened, debtor, creditor, currency, kind, amount, amount, debt_id, "", "", "abierto", "open")


def repayments(*rows):
    return pd.DataFrame([{"tx_id":tx,"Date":date,"debtor":d,"creditor":c,"Currency":cur,"repayment_amount":amount,"status":"pagado"} for tx,date,d,c,cur,amount in rows])


def test_partial_interest_then_principal_and_subsequent_repayment():
    items=[item("i",50,kind="Interes"),item("p",100)]
    _, alloc, events, _, _=resolve_repayments(items,repayments(("r1","2026-02-01","A","PM","USD",30),("r2","2026-03-01","A","PM","USD",50)))
    assert list(alloc["target_debt_id"]) == ["i","i","p"]
    assert list(alloc["allocated_amount"]) == [30,20,30]
    assert list(alloc["balance_after"]) == [20,0,70]
    assert events["leftover_amount"].sum() == 0


def test_partial_principal_exact_close_and_overpayment_remainder():
    items=[item("p",100)]
    open_df, alloc, events, _, _=resolve_repayments(items,repayments(("r1","2026-02-01","A","PM","USD",40),("r2","2026-03-01","A","PM","USD",80)))
    assert list(alloc["balance_after"]) == [60,0]
    assert events.iloc[-1]["leftover_amount"] == 20
    assert open_df.iloc[0]["engine_status"] == "closed"


def test_allocation_never_crosses_future_pair_currency_or_cost_gap():
    items=[item("future",10,opened="2026-04-01"),item("pair",10,debtor="B"),item("ars",10,currency="ARS")]
    _, alloc, events, _, _=resolve_repayments(items,repayments(("r","2026-03-01","A","PM","USD",10)))
    assert alloc.empty and events.iloc[0]["leftover_amount"] == 10


def test_partial_allocation_drives_daily_monthly_yearly_stock_and_reopening():
    items=[item("p1",100),item("p2",25,opened="2026-04-01")]
    open_df, alloc, _, _, _=resolve_repayments(items,repayments(("r1","2026-02-01","A","PM","USD",40),("r2","2026-03-01","A","PM","USD",60)))
    daily=build_debt_balance_daily(open_df,allocations=alloc,end_date="2026-12-31")
    assert daily.loc[daily.as_of_date.eq("2026-02-28"),"open_total"].iloc[0] == 60
    assert daily.loc[daily.as_of_date.eq("2026-03-31"),"open_total"].iloc[0] == 0
    assert daily.loc[daily.as_of_date.eq("2026-04-30"),"open_total"].iloc[0] == 25
    assert _last_snapshot_by_period(daily,"M","M").loc[lambda d:d.period.eq("2026-02"),"open_total"].iloc[0] == 60
    assert _last_snapshot_by_period(daily,"Y","Y")["open_total"].iloc[-1] == 25
