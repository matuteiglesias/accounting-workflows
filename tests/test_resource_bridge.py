from pathlib import Path
import pandas as pd
import pytest
from accounting.marts.semantic import build_semantic_outputs, build_monthly_operating_statement
from accounting.metrics.resource_bridge import build_resource_bridge

def test_resource_bridge_economic_funding_not_cash_or_debt(tmp_path: Path):
    def row(id, payer, receiver, flow, kind, amount, cur="ARS", detail=""):
        return dict(tx_id=id,Date="2026-01-02",amount=amount,Currency=cur,Box="Property Management",Lugar="A",
                    payer=payer,receiver=receiver,Flujo=flow,Tipo=kind,Detalle=detail)
    raw=pd.DataFrame([
        row("advance","MI","PM","Contribucion","Impuestos",100),
        row("paid","PM","Impuestos","Pagos","Impuestos",100,detail="Matias"),
        row("direct","Inq","Impuestos","Pagos","Impuestos",30,detail="Matias"),
        row("tenant","Inq","PM","Contribucion","Contribuciones",20),
        row("hh","Alejandro","HH","Transfer","Contribuciones",40),
        row("cross","FB","PM","Contribucion","Contribuciones",50),
        row("clear","Inq","PM","Contribucion","Repago",25),
        row("debt","PM","MI","Transfer","Prestamo",10,"USD"),
        row("draw","PM","Alejandro","Transfer","Dividendo",19,"USD"),
        row("return","Alejandro","PM","Repago","Repago",19,"USD"),
        row("fx","FX","PM","Cambio","FX",200),
    ])
    paths=build_semantic_outputs(raw,tmp_path)
    audit=pd.read_csv(paths["classification_audit"],keep_default_na=False)
    by=audit.set_index("tx_id")
    assert by.loc["advance","semantic_subbucket"]=="designated_expense"
    assert by.loc["paid","funding_actor"]==""
    assert by.loc["direct","funding_actor"]=="Inquilino"
    assert by.loc["hh","semantic_bucket"]=="internal_transfer"
    assert by.loc["cross","semantic_bucket"]=="internal_transfer"
    pairs=pd.DataFrame([{"distribution_tx_id":"draw","recovery_tx_id":"return"}])
    allocations=pd.DataFrame([{"repayment_tx_id":"return","allocated_amount":10},{"repayment_tx_id":"return","allocated_amount":9}])
    monthly,members=build_resource_bridge(audit,pairs,allocations)
    ars=monthly.set_index("Currency").loc["ARS"]
    assert ars.economic_funding==150
    assert ars.designated_advance==100
    assert ars.direct_tenant_support==30
    assert ars.clearing_pending==25
    assert monthly.set_index("Currency").loc["USD","linked_debt_recovery"]==19
    assert set(members.tx_id)=={"advance","direct","tenant","clear","return"}
    build_monthly_operating_statement(tmp_path)
    stmt=pd.read_csv(tmp_path/"monthly_operating_statement.csv")
    assert stmt.loc[stmt.Currency.eq("ARS")&stmt.statement_line.eq("property_opex_true"),"amount"].sum()==130
    with pytest.raises(ValueError,match="not fully allocated"):
        build_resource_bridge(audit,pairs,allocations.iloc[:1])

def test_legacy_without_pairs_does_not_guess(tmp_path):
    a=pd.DataFrame([dict(tx_id="repay",period="2026-01",Currency="USD",Box="Property Management",
        payer="Alejandro",receiver="PM",Flujo="Repago",Tipo="Repago",semantic_bucket="debt_movement",
        semantic_subbucket="repayment",direction="in",amount=19)])
    monthly,members=build_resource_bridge(a)
    assert members.empty
    assert monthly.linked_debt_recovery.sum()==0

def test_recovery_registry_is_pinned_and_cutoff_scoped(tmp_path,monkeypatch):
    import json
    from accounting.scope import scope_metadata
    from accounting.cutoff import cutoff_metadata
    from accounting.metrics.resource_bridge import load_recovery_pairs
    run=tmp_path/"run"
    (run/"meta").mkdir(parents=True)
    (run/"meta"/"stage_A_ingest.json").write_text(json.dumps({"params":{
        **scope_metadata({"Property Management"}),**cutoff_metadata("2026-06-30")}}))
    registry=tmp_path/"registry.csv"
    pd.DataFrame([
        {"Box":"Property Management","recovery_date":"2026-05-10","distribution_tx_id":"a","recovery_tx_id":"b"},
        {"Box":"Property Management","recovery_date":"2026-07-10","distribution_tx_id":"c","recovery_tx_id":"d"},
        {"Box":"Family Business","recovery_date":"2026-05-10","distribution_tx_id":"e","recovery_tx_id":"f"},
    ]).to_csv(registry,index=False)
    monkeypatch.setenv("RESOURCE_RECOVERY_REGISTRY",str(registry))
    assert load_recovery_pairs(run).to_dict("records")==[{"distribution_tx_id":"a","recovery_tx_id":"b"}]
    registry.unlink()
    assert load_recovery_pairs(run).recovery_tx_id.tolist()==["b"]
