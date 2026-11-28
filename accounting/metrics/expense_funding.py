"""Explicit funding-to-expense attribution, distinct from payment execution."""
from decimal import Decimal
import pandas as pd
import re

ALIASES={"MI":"Matías","INQ":"Inquilinos","INQUILINO":"Inquilinos","HECTOR":"Héctor / Silvina","HÉCTOR":"Héctor / Silvina","HH":"Household","FERNANDO":"Primos"}
def actor(v):return ALIASES.get(str(v).upper(),str(v) or "Sin identificar")
def cents(v):return int((Decimal(str(v))*100).quantize(Decimal("1")))
def partition(audit,ledger,funding):
    """Each explicit funding leg used at most once; residual remains payment circuit.

    This is an attribution of recorded resources, not ultimate personal incidence.
    """
    a=audit.fillna("").copy()
    if a.tx_id.duplicated().any() or funding.tx_id.duplicated().any():raise ValueError("duplicate funding or audit identity")
    ids=ledger.assign(transaction_id=ledger.transaction_id.astype(str))
    ids=ids[ids.transaction_id.ne("")]
    byid=ids.groupby("transaction_id").tx_id.agg(list).to_dict()
    bytx=a.set_index("tx_id")
    fm=funding[funding.component.isin(["contribution","designated_advance"])].set_index("tx_id")
    rows=[];used=set()
    for r in a[a.semantic_bucket.eq("property_opex")&a.direction.eq("out")].itertuples(index=False):
        tags=str(getattr(r,"tag",""))
        src=re.findall(r"(?:^|;)expense_source:([^;]+)",tags)
        circuit=actor(src[0] if src else r.payer)
        links=re.findall(r"(?:^|;)expense_funding_ids:([^;]+)",tags)
        if len(links)>1:raise ValueError("duplicate expense funding marker")
        remaining=cents(r.amount)
        for lid in links[0].split(",") if links else []:
            candidates=byid.get(lid,[])
            if len(candidates)!=1:raise ValueError("missing/ambiguous funding transaction "+lid)
            tid=candidates[0]
            if tid in used or tid not in fm.index or tid not in bytx.index:raise ValueError("unavailable or reused funding "+lid)
            v=bytx.loc[tid]; fund=fm.loc[tid]
            if (v.Box,v.Currency,v.Lugar,v.semantic_subbucket)!=(r.Box,r.Currency,r.Lugar,r.semantic_subbucket):
                # Contributions use designated_expense, not tax/service subbucket.
                if (v.Box,v.Currency,v.Lugar,v.Tipo)!=(r.Box,r.Currency,r.Lugar,r.Tipo):
                    raise ValueError("funding scope/type mismatch "+lid)
            n=cents(fund.amount)
            if n<=0 or n>remaining:raise ValueError("funding exceeds expense "+lid)
            rows.append((r.tx_id,r.period,r.Currency,r.Box,circuit,actor(fund.funding_actor),n/100,tid))
            remaining-=n;used.add(tid)
        if remaining:
            rows.append((r.tx_id,r.period,r.Currency,r.Box,circuit,circuit,remaining/100,""))
    out=pd.DataFrame(rows,columns=["tx_id","period","Currency","Box","circuit","funding_source","amount","funding_tx_id"])
    expected=a[a.semantic_bucket.eq("property_opex")&a.direction.eq("out")].set_index("tx_id").amount.map(cents)
    actual=out.groupby("tx_id").amount.sum().map(cents).reindex(expected.index,fill_value=0)
    if not expected.eq(actual).all():raise ValueError("expense funding conservation failed")
    return out
