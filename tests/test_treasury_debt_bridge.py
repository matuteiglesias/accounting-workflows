import pandas as pd
import pytest
from accounting.marts.treasury_debt_bridge import load_bridge

def test_bridge_event_grain_native_cash_and_debt_allocations(tmp_path):
    cash = []
    for tid,currency,amount,direction,payer,receiver in [
        ("fx","ARS",10000,"out","PM","MI"),
        ("partial","USD",980,"out","PM","Primos"),
        ("incoming","USD",190,"in","Alejandro","PM"),
        ("missing","USD",20,"out","PM","MI"),
    ]:
        cash.append(dict(tx_id=tid,Date="2026-08-20",period="2026-08",Box="Property Management",
            Currency=currency,movement_basis="actual_cash",cash_category="debt_repayment",
            cash_direction=direction,amount_in=amount if direction=="in" else 0,
            amount_out=amount if direction=="out" else 0,payer=payer,receiver=receiver))
    pd.DataFrame(cash).to_csv(tmp_path/"box_treasury_transaction_detail.csv",index=False)
    pd.DataFrame([dict(tx_id="fx",Box="Property Management",tag="repay_usd_at_ccl",CCL=1000),
        dict(tx_id="noncash",Box="Family Business",tag="",CCL="")]).to_csv(tmp_path/"ledger_canonical.csv",index=False)
    details=[]
    for tid,payer,receiver,gross,leftover,parts in [
        ("fx","PM","MI",10,0,[("p1","Prestamo",8),("i1","Interes",2)]),
        ("partial","PM","Primos",980,1.46,[("p2","Prestamo",895.50),("i2","Interes",83.04)]),
        ("incoming","Alejandro","PM",190,0,[("p3","Prestamo",190)]),
        ("noncash","Hector","MI",15,0,[("i3","Interes",15)]),
    ]:
        for target,kind,allocated in parts:
            details.append(dict(repayment_tx_id=tid,period="2026-08",repayment_date="2026-08-20",
                debtor=payer,creditor=receiver,Currency="USD",repayment_amount=gross,leftover_amount=leftover,
                target_debt_id=target,target_item_type=kind,allocated_amount=allocated))
    pd.DataFrame(details).to_csv(tmp_path/"monthly_debt_repayment_detail.csv",index=False)
    d=load_bridge(tmp_path).set_index("tx_id")
    assert len(d)==5
    assert d.loc["fx","matched_native_cash"]==10000 and d.loc["fx","debt_event_amount"]==10
    assert d.loc["partial","status"]=="partial" and float(d.loc["partial","unallocated"])==1.46
    assert float(d.loc["partial","principal_allocated"])==895.5
    assert d.loc["incoming","status"]=="matched"
    assert d.loc["noncash","status"]=="debt_only"
    assert d.loc["missing","status"]=="cash_only"
    details.append(details[0])
    pd.DataFrame(details).to_csv(tmp_path/"monthly_debt_repayment_detail.csv",index=False)
    with pytest.raises(ValueError,match="duplicate allocation"):
        load_bridge(tmp_path)

def test_missing_lineage_is_unavailable(tmp_path):
    assert load_bridge(tmp_path) is None


def test_directed_draw_links_to_debt_without_reclassifying_cash(tmp_path):
    pd.DataFrame([dict(tx_id="draw",Date="2026-08-20",period="2026-08",Box="Property Management",
      Currency="USD",movement_basis="actual_cash",cash_category="personal_draws",
      cash_direction="out",amount_in=0,amount_out=490,payer="PM",receiver="MI")]).to_csv(
          tmp_path/"box_treasury_transaction_detail.csv",index=False)
    pd.DataFrame([dict(tx_id="draw",Box="Property Management",tag="settlement_draw_for:Primos",CCL="")]).to_csv(
        tmp_path/"ledger_canonical.csv",index=False)
    pd.DataFrame([dict(repayment_tx_id="draw",period="2026-08",repayment_date="2026-08-20",
        debtor="PM",creditor="Primos",Currency="USD",repayment_amount=490,allocated_amount=478.54,
        leftover_amount=11.46,target_debt_id="p",target_item_type="Prestamo")]).to_csv(
        tmp_path/"monthly_debt_repayment_detail.csv",index=False)
    r=load_bridge(tmp_path).iloc[0]
    assert r.cash_category=="personal_draws"
    assert r.status=="partial" and float(r.unallocated)==11.46 and float(r.matched_native_cash)==490


def test_linked_constructive_settlement_does_not_repeat_draw_cash(tmp_path):
    pd.DataFrame([dict(tx_id="draw",Date="2026-08-20",period="2026-08",Box="Property Management",
        Currency="USD",movement_basis="actual_cash",cash_category="personal_draws",
        cash_direction="out",amount_in=0,amount_out=490,payer="PM",receiver="MI")]).to_csv(
        tmp_path/"box_treasury_transaction_detail.csv",index=False)
    pd.DataFrame([
        dict(tx_id="draw",transaction_id="1670.0",Box="Property Management",tag="settlement_draw_for:Primos"),
        dict(tx_id="hector",transaction_id="1448",Box="Property Management",tag="settlement_via_draw:1670"),
    ]).to_csv(tmp_path/"ledger_canonical.csv",index=False)
    pd.DataFrame([dict(repayment_tx_id=tid,period="2026-08",repayment_date="2026-08-20",
        debtor=debtor,creditor=creditor,Currency="USD",repayment_amount=490,allocated_amount=490,
        leftover_amount=0,target_debt_id=tid+"-p",target_item_type="Prestamo")
        for tid,debtor,creditor in [("draw","PM","Primos"),("hector","Hector","MI")]]).to_csv(
        tmp_path/"monthly_debt_repayment_detail.csv",index=False)
    d=load_bridge(tmp_path).set_index("tx_id")
    assert d.loc["hector","status"]=="constructive_linked"
    assert d.loc["hector","linked_draw_tx_id"]=="draw"
    assert d.loc["hector","cash_amount"]==""
    assert d.loc["draw","cash_amount"]==490
