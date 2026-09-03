from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.debt_accountability.render import build_validation, render_report


def _sources(root: Path) -> dict[str, Path]:
    position = pd.DataFrame([
        {"period":"2026-08","period_end":"2026-08-31","as_of_date":"2026-08-31","debtor":"A","creditor":"PM","Currency":"USD","component":"principal","position_status":"available","open_amount":90,"open_principal":90,"open_interest":10,"open_total":100},
        {"period":"2026-08","period_end":"2026-08-31","as_of_date":"2026-08-31","debtor":"A","creditor":"PM","Currency":"USD","component":"interest","position_status":"available","open_amount":10,"open_principal":90,"open_interest":10,"open_total":100},
        {"period":"2026-08","period_end":"2026-08-31","as_of_date":"2026-08-31","debtor":"A","creditor":"PM","Currency":"USD","component":"total","position_status":"available","open_amount":100,"open_principal":90,"open_interest":10,"open_total":100},
    ])
    activity=[]
    for kind, values in [("opening_balance",{}),("new_claim",{"new_principal":150}),("interest_accrual",{"interest_accrued":10}),("repayment",{"repayments":60}),("adjustment",{"adjustments":0}),("closing_balance",{}),("net_change",{})]:
        activity.append({"period":"2026-08","period_end":"2026-08-31","Currency":"USD","debtor":"A","creditor":"PM","activity_type":kind,"new_principal":0,"interest_accrued":0,"repayments":0,"adjustments":0,"opening_total":0,"closing_total":100,"net_change":100,**values})
    detail=pd.DataFrame([{"period":"2026-08","repayment_tx_id":"r1","repayment_date":"2026-08-20","debtor":"A","creditor":"PM","Currency":"USD","repayment_amount":60,"allocated_amount":60,"leftover_amount":0,"allocation_status":"resolved","target_debt_id":"d1","target_source_tx_id":"s1","target_opened_at":"2026-08-01","target_detail":"Obligación","balance_before":60,"balance_after":0}])
    gaps=pd.DataFrame([{"source_tx_id":"g1","Date":"2023-01-01","period":"2023-01","Currency":"USD","amount":20,"Lugar":"CABA","description":"Costo","status":"abierto","economic_scope":"Property Management","accounting_nature":"unresolved_cost_allocation","debt_effect":"none","allocation_status":"unresolved","asserted_bearer":""}])
    paths={}
    for name, frame in [("position",position),("activity",pd.DataFrame(activity)),("detail",detail),("gaps",gaps)]:
        paths[name]=root/f"{name}.csv"; frame.to_csv(paths[name],index=False)
    qa=pd.DataFrame([{"check":"source","status":"pass","severity":"error","detail":"ok"}])
    for name in ("position_qa","activity_qa","gaps_qa"):
        paths[name]=root/f"{name}.csv"; qa.to_csv(paths[name],index=False)
    return paths


def test_debt_report_renders_governed_sections_and_details(tmp_path: Path) -> None:
    p=_sources(tmp_path)
    outputs=render_report(position_path=p["position"],position_qa_path=p["position_qa"],activity_path=p["activity"],activity_qa_path=p["activity_qa"],repayment_detail_path=p["detail"],gaps_path=p["gaps"],gaps_qa_path=p["gaps_qa"],out_dir=tmp_path/"out",as_of_date="2026-08-31")
    document=outputs["html"].read_text()
    assert "Posición actual" in document and "Cómo cambió" in document
    assert "<details>" in document and "Fuera de deuda" in document
    assert pd.read_csv(outputs["validation"])["status"].eq("pass").all()


def test_debt_report_rejects_future_target_and_material_adjustment(tmp_path: Path) -> None:
    p=_sources(tmp_path)
    detail=pd.read_csv(p["detail"]); detail["target_opened_at"]="2026-09-01"; detail.to_csv(p["detail"],index=False)
    activity=pd.read_csv(p["activity"]); activity.loc[activity.activity_type.eq("adjustment"),"adjustments"]=5; activity.to_csv(p["activity"],index=False)
    with pytest.raises(ValueError,match="validation failed"):
        render_report(position_path=p["position"],position_qa_path=p["position_qa"],activity_path=p["activity"],activity_qa_path=p["activity_qa"],repayment_detail_path=p["detail"],gaps_path=p["gaps"],gaps_qa_path=p["gaps_qa"],out_dir=tmp_path/"bad",as_of_date="2026-08-31")
