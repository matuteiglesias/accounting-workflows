"""Isolated Household proof report: compatibility, source scope, and chart safety."""
import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.household_poc import build_household_poc
from accounting.scope import scope_metadata


def fixture_run(tmp_path: Path) -> Path:
    r=tmp_path/"20261119T211126Z_HH"
    (r/"meta").mkdir(parents=True)
    (r/"meta"/"stage_A_ingest.json").write_text(
        json.dumps({"params":scope_metadata({"Household"})}),encoding="utf-8")
    rows=[]
    for i in range(1,11):
        row={"period":f"2026-{i:02d}","Currency":"ARS","Box":"Household",
             "total_cash_in":float(100*i),"total_cash_out":float(200*i),
             "funding_cash_in":float(100*i),"reconciliation_status":"reconciled"}
        for col in ("services_out","maintenance_out","legal_out","taxes_out","debt_repayments_out",
                    "debt_principal_out","personal_draws_out","internal_transfer_out","other_cash_out","unknown_cash_out"):
            row[col]=float(200*i) if col=="services_out" else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(r/"monthly_cash_accountability.csv",index=False)
    pd.DataFrame([
        {"tx_id":"HH_1","Date":"2026-01-05","Currency":"ARS","Box":"Household",
         "semantic_bucket":"property_opex","semantic_subbucket":"services",
         "direction":"out","amount":400.},
        {"tx_id":"HH_2","Date":"2026-05-07","Currency":"ARS","Box":"Household",
         "semantic_bucket":"funding_contribution","semantic_subbucket":"family_or_tenant_contribution",
         "direction":"in","amount":250.},
    ]).to_csv(r/"classification_audit.csv",index=False)
    return r


def test_hh1_five_reports_and_explicit_skips(tmp_path):
    r=fixture_run(tmp_path)
    path=build_household_poc(r,tmp_path/"out")
    d=json.loads(path.read_text())
    assert d["scope_tag"]=="HH" and d["scope_boxes"]==["Household"]
    assert len(d["reports"])==5
    assert len(d["skipped_existing_reports"])==32
    assert all(x["status"]=="SKIPPED_INCOMPATIBLE" for x in d["skipped_existing_reports"])
    assert d["data_qa"]["annual_cash_in_ars"]==5500
    assert d["data_qa"]["annual_cash_out_ars"]==11000
    import hashlib
    for x in d["reports"]:
        f=path.parent/x["chart"]
        assert hashlib.sha256(f.read_bytes()).hexdigest()==x["sha256"]
        assert f.read_text().startswith("<svg")
    assert not list(path.parent.rglob("*.csv"))


def test_hh1_refuses_foreign_box(tmp_path):
    r=fixture_run(tmp_path)
    a=pd.read_csv(r/"classification_audit.csv");a.loc[0,"Box"]="Family Business"
    a.to_csv(r/"classification_audit.csv",index=False)
    with pytest.raises(ValueError,match="Cross-scope"):
        build_household_poc(r,tmp_path/"out")


def test_hh1_refuses_wrong_universe(tmp_path):
    r=fixture_run(tmp_path)
    (r/"meta"/"stage_A_ingest.json").write_text(
        json.dumps({"params":scope_metadata({"Family Business","Property Management"})}),encoding="utf-8")
    with pytest.raises(ValueError,match="rejects"):
        build_household_poc(r,tmp_path/"out")


def test_hh1_refuses_unreconciled_month(tmp_path):
    r=fixture_run(tmp_path)
    c=pd.read_csv(r/"monthly_cash_accountability.csv")
    c.loc[0,"reconciliation_status"]="review"
    c.to_csv(r/"monthly_cash_accountability.csv",index=False)
    with pytest.raises(ValueError,match="reconcile"):
        build_household_poc(r,tmp_path/"out")


def test_hh1_refuses_future_source_and_incomplete_period(tmp_path):
    r=fixture_run(tmp_path)
    a=pd.read_csv(r/"classification_audit.csv")
    a.loc[0,"Date"]="2026-11-01"
    a.to_csv(r/"classification_audit.csv",index=False)
    with pytest.raises(ValueError,match="past"):
        build_household_poc(r,tmp_path/"out")

    r=fixture_run(tmp_path/"other")
    a=pd.read_csv(r/"monthly_cash_accountability.csv")
    a=a.iloc[:-1]
    a.to_csv(r/"monthly_cash_accountability.csv",index=False)
    with pytest.raises(ValueError,match="incomplete"):
        build_household_poc(r,tmp_path/"out2")
