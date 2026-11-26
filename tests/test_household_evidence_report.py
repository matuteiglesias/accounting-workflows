"""HH reuses PM documentary contract but cannot import PM/FB evidence or publish."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
import pytest
from accounting.reports import payment_evidence

def digest(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def fixture(tmp:Path):
 run=tmp/"run/20260101T000000Z_HH"
 snap=tmp/"snap"
 run.mkdir(parents=True);(snap/"evidence").mkdir(parents=True)
 rows=[{"tx_id":"aa","transaction_id":"17","Date":"2026-01-01","amount":"101.35",
        "amount_cents":"10135","Currency":"ARS","Box":"Household","payer":"HH",
        "Tipo":"Servicio","receiver":"Servicios","status":"pagado","source_row":"10"}]
 pd.DataFrame(rows).to_csv(run/"ledger_canonical.csv",index=False)
 pd.DataFrame(rows).to_csv(snap/"required_transactions.csv",index=False)
 evidence=(snap/"evidence/original.pdf");evidence.write_bytes(b"%PDF-1.4\ntest\n")
 eid=digest(evidence)
 pd.DataFrame([{"evidence_id":eid,"content_sha256":eid,"media_type":"application/pdf",
                "display_name":"Factura servicio","href":"evidence/original.pdf"}]).to_csv(snap/"evidence_documents.csv",index=False)
 pd.DataFrame([{"tx_id":"aa","evidence_id":eid,"relation":"liability_source","status":"candidate"}]).to_csv(snap/"transaction_evidence.csv",index=False)
 m={"artifact":"acct.transaction-evidence@1","scope_tag":"HH_EVIDENCE",
    "source_ledger_sha256":digest(run/"ledger_canonical.csv"),
    "required_transaction_count":1,
    "required_tx_ids_sha256":payment_evidence._required_hash(["aa"]),
    "snapshot_id":"HH_TEST","review_preview":True,"publication_allowed":False}
 (snap/"manifest.json").write_text(json.dumps(m))
 return run,snap

def test_household_builds_private_links_with_shared_renderer(tmp_path:Path,monkeypatch):
 run,snap=fixture(tmp_path)
 def fake_pdf(html,target,*,browser_bin=None):
  target.write_bytes(b"%PDF-1.4\ntest");return target
 monkeypatch.setattr(payment_evidence,"render_pdf",fake_pdf)
 res=payment_evidence.build_payment_evidence_report(
  run_root=run,snapshot_dir=snap,out_root=tmp_path/"out",
  report_id="hh_payment_evidence",report_title="HH Documentos",
  report_description="Only Household",scope_tag="HH_EVIDENCE")
 html=Path(res["report_html"]).read_text()
 assert 'HH Documentos' in html
 assert 'Ver boleta (no acredita pago)' in html
 assert 'Comprobante aprobado' not in html
 assert '<tr data-status="bill_only"' in html
 assert 'Property Management' not in html
 manifest=json.loads(Path(res["report_manifest"]).read_text())
 catalog=json.loads(Path(res["catalog"]).read_text())
 assert manifest["publication_allowed"] is False
 assert catalog["scope_tag"]=="HH_EVIDENCE" and catalog["source_run_id"]==run.name
 assert len(manifest["attachments"])==1
 assert (Path(res["report_html"]).parent/"evidence").is_dir()

@pytest.mark.parametrize("breach",["bad_source_run","crossbox","wrong_snapshot_scope","amount_tamper","rogue_report_id"])
def test_hh_scope_guards_fail_closed(tmp_path:Path,breach):
 run,snap=fixture(tmp_path)
 args=dict(run_root=run,snapshot_dir=snap,out_root=tmp_path/"out",
  report_id="hh_payment_evidence",report_title="HH",
  scope_tag="HH_EVIDENCE")
 if breach=="bad_source_run":
  target=run.parent/"20260101T000000Z_FBPM";run.rename(target);args["run_root"]=target
 elif breach=="crossbox":
  file=snap/"required_transactions.csv";r=pd.read_csv(file,dtype=str);r.loc[0,"Box"]="Property Management";r.to_csv(file,index=False)
 elif breach=="wrong_snapshot_scope":
  file=snap/"manifest.json";m=json.loads(file.read_text());m["scope_tag"]="PM_EVIDENCE";file.write_text(json.dumps(m))
 elif breach=="amount_tamper":
  file=snap/"required_transactions.csv";r=pd.read_csv(file,dtype=str);r.loc[0,"amount_cents"]="999999";r.to_csv(file,index=False)
 elif breach=="rogue_report_id":
  args["report_id"]="pm_payment_evidence"
 with pytest.raises(ValueError,match="HH|cross-Box|scope|identity|Unsupported"):
  payment_evidence.build_payment_evidence_report(**args)
 assert not (tmp_path/"out").exists()
