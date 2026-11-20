"""HH1 v3: 35-month accounting bridge, funding, categories and preview privacy."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.household_v3 import build_household_v3, _cost_label
from accounting.scope import scope_metadata


def fixture_run(tmp_path: Path) -> Path:
    r=tmp_path/"20261120T220100Z_HH"
    (r/"meta").mkdir(parents=True)
    (r/"meta/stage_A_ingest.json").write_text(json.dumps({
        "params":{**scope_metadata({"Household"}),"cutoff_date":"2026-11-20"}
    }))
    cash=[];semantic=[];ledger=[]
    opening=-4179.
    for period in pd.period_range("2024-01","2026-11",freq="M"):
        ym=str(period);yr=period.year
        incoming=100.;out=200.
        def add(tid: str, amount: float, *, payer: str, direction: str,
                subbucket: str, bucket: str, issuer: str = "Edenor", detail: str="", tag: str=""):
            row={
                "tx_id":tid,"Date":ym+"-10","Currency":"ARS","Box":"Household",
                "amount":amount,"payer":payer,"receiver":"HH" if direction=="in" else "Servicios",
                "semantic_bucket":bucket,"semantic_subbucket":subbucket,
                "direction":direction,"Detalle":detail,"tag":tag,
            }
            semantic.append(row)
            ledger.append({**row,"issuer":issuer,"status":"pagado"})
        add(ym+"-in",100.,payer="MI",direction="in",subbucket="family_or_tenant_contribution",
            bucket="funding_contribution",issuer="")
        add(ym+"-out",200.,payer="HH",direction="out",subbucket="services",
            bucket="property_opex",issuer="Edenor")
        if ym=="2024-05":
            out+=250
            add(ym+"-makro",250,payer="HH",direction="out",subbucket="maintenance",
                bucket="property_opex",issuer="",detail="Makro")
        if ym=="2025-03":
            out+=500
            add(ym+"-abl",500,payer="HH",direction="out",subbucket="taxes",
                bucket="property_opex",issuer="ABL")
        if ym=="2026-11":
            incoming+=856;out+=906
            add(ym+"-dog-cande",706,payer="Cande",direction="in",subbucket="maintenance",
                bucket="property_opex",issuer="",detail="Perro",tag="HH_PET_DOG_202611")
            add(ym+"-dog-alen",150,payer="Alen",direction="in",subbucket="maintenance",
                bucket="property_opex",issuer="",detail="Perro",tag="HH_PET_DOG_202611")
            add(ym+"-dog-exp",906,payer="HH",direction="out",subbucket="maintenance",
                bucket="property_opex",issuer="",detail="Perro",tag="HH_PET_DOG_202611")
        closing=opening+incoming-out
        cash.append({
            "Box":"Household","Currency":"ARS","period":ym,
            "opening_control":opening,"total_cash_in":incoming,
            "total_cash_out":out,"net_cash_flow":incoming-out,
            "closing_control":closing,"reconciliation_status":"reconciled",
        })
        opening=closing
    pd.DataFrame(semantic).to_csv(r/"classification_audit.csv",index=False)
    pd.DataFrame(ledger).to_csv(r/"ledger_canonical.csv",index=False)
    pd.DataFrame(cash).to_csv(r/"monthly_cash_accountability.csv",index=False)
    return r


def test_v3_reconciled_35_months_five_reports(tmp_path):
    r=fixture_run(tmp_path)
    catalog=build_household_v3(r,tmp_path/"reports")
    d=json.loads(catalog.read_text())
    assert d["schema"]=="household_poc_catalog.v3"
    assert d["data_as_of_date"]=="2026-11-20" and d["as_of_date"]=="2026-11-30"
    assert d["closing_status"]=="PROVISIONAL_AS_OF"
    assert d["data_qa"]["cash_months_since_2024"]==35
    assert d["data_qa"]["virtual_opening_2024_ars"]==-4179.
    assert d["data_qa"]["unapproved_evidence_links"]==0
    assert [r["report_id"] for r in d["reports"]]==[
        "virtual_balance","contributions","outflows","monthly_costs","evidence"]
    assert len(d["skipped_existing_reports"])==32
    assert set(p.suffix for p in catalog.parent.rglob("*") if p.is_file())=={".json",".svg",".html"}
    for entry in d["reports"]:
        assert (catalog.parent/entry["chart"]).is_file()
        assert (catalog.parent/entry["html"]).is_file()
    svg=(catalog.parent/"charts/virtual_balance.svg").read_text()
    assert "Entradas +" in svg and "Salidas −" in svg and "Neto ±" in svg
    assert 'href="../charts/virtual_balance.svg"' in (catalog.parent/"reports/virtual_balance.html").read_text()
    assert "Saldo acumulado" in svg and 'stroke="#325d88"' in svg
    assert "2024" in svg and "2026" in svg
    contrib=(catalog.parent/"reports/contributions.html").read_text()
    assert "Candela" in contrib and "Alén" in contrib
    assert "706" in contrib or "856" in contrib
    cost=(catalog.parent/"reports/outflows.html").read_text()
    assert "ABL" in cost and "Makro" in cost and "Perro" in cost
    assert "porcentajes" not in cost
    exp=(catalog.parent/"reports/monthly_costs.html").read_text()
    assert "PROMEDIO 2024 / 12" in exp and "PROMEDIO 2026 / 12" in exp
    evidence=(catalog.parent/"reports/evidence.html").read_text()
    assert "Sin documento vinculado" in evidence
    assert "<a href=\"https://" not in evidence


@pytest.mark.parametrize("field,value,pattern",[
    ("scope","Family Business","foreign-Box"),
    ("cutoff","2026-11-19","cutoff"),
    ("future_date","2026-11-24","after observed cutoff"),
    ("unreconciled","review","unreconciled"),
    ("incomplete","missing","every ARS month"),
    ("nonbridging",123.,"monthly net"),
    ("unknown_payer","UNKNOWN","Unmapped HH contributors"),
])
def test_invalid_sources_rejected(tmp_path,field,value,pattern):
    r=fixture_run(tmp_path)
    if field=="scope":
        s=pd.read_csv(r/"classification_audit.csv")
        s.loc[0,"Box"]=value
        s.to_csv(r/"classification_audit.csv",index=False)
    elif field=="cutoff":
        m=json.loads((r/"meta/stage_A_ingest.json").read_text())
        m["params"]["cutoff_date"]=value
        (r/"meta/stage_A_ingest.json").write_text(json.dumps(m))
    elif field=="future_date":
        s=pd.read_csv(r/"classification_audit.csv")
        s.loc[s.index[-1],"Date"]=value
        s.to_csv(r/"classification_audit.csv",index=False)
    elif field=="unreconciled":
        c=pd.read_csv(r/"monthly_cash_accountability.csv")
        c.loc[0,"reconciliation_status"]=value
        c.to_csv(r/"monthly_cash_accountability.csv",index=False)
    elif field=="incomplete":
        c=pd.read_csv(r/"monthly_cash_accountability.csv")
        c=c.iloc[:-1]
        c.to_csv(r/"monthly_cash_accountability.csv",index=False)
    elif field=="nonbridging":
        c=pd.read_csv(r/"monthly_cash_accountability.csv")
        c.loc[4,"closing_control"]=value
        c.to_csv(r/"monthly_cash_accountability.csv",index=False)
    elif field=="unknown_payer":
        s=pd.read_csv(r/"classification_audit.csv")
        s.loc[0,"payer"]=value
        s.to_csv(r/"classification_audit.csv",index=False)
    with pytest.raises(ValueError,match=pattern):
        build_household_v3(r,tmp_path/"report")


def test_cost_classification_uses_issuer_not_guesswork():
    row=pd.Series({"semantic_subbucket":"services","issuer":"Metrogas","Detalle":"","tag":""})
    assert _cost_label(row)=="Gas"
    row.issuer="unknown";assert _cost_label(row)=="Servicios sin identificar"
    row.semantic_subbucket="maintenance";row.Detalle="Perro — consulta"
    assert _cost_label(row)=="Mantenimiento · Perro"
    row.Detalle="Makro";assert _cost_label(row)=="Mantenimiento · Makro"
    row.semantic_subbucket="taxes";row.issuer="ABL"
    assert _cost_label(row)=="ABL"
