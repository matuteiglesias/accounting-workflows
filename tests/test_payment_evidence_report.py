from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.reports import payment_evidence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("population_size", [183, 214])
@pytest.mark.parametrize("extension,media_type", [
    ("pdf", "application/pdf"),
    ("jpeg", "image/jpeg"),
    ("png", "image/png"),
])
def test_payment_evidence_report_rebuilds_frozen_rows_and_only_links_approved_payment_proof(
    tmp_path: Path, monkeypatch, extension: str, media_type: str, population_size: int
) -> None:
    run = tmp_path / "run" / "RUN_FBPM"
    snapshot = tmp_path / "snapshot" / "PM"
    run.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    ledger = pd.DataFrame(
        [
            {
                "tx_id": f"tx-{index:03d}",
                "Date": "2026-08-01",
                "amount": "100.00",
                "Currency": "ARS",
                "payer": "PM",
                "Tipo": "Impuestos",
                "Box": "Property Management",
                "receiver": "Impuestos",
                "Detalle": "synthetic",
                "notes": "",
            }
            for index in range(population_size)
        ]
    )
    ledger.to_csv(run / "ledger_canonical.csv", index=False)
    required = ledger.copy()
    required["status"] = "pagado"
    required.to_csv(snapshot / "required_transactions.csv", index=False)
    source = snapshot / "evidence" / ("a" * 64 + "." + extension)
    source.parent.mkdir()
    source.write_bytes(b"%PDF-approved")
    evidence_id = _sha(source)
    docs = pd.DataFrame(
        [{
            "evidence_id": evidence_id,
            "content_sha256": evidence_id,
            "media_type": media_type,
            "display_name": "approved." + extension,
            "href": f"evidence/{source.name}",
        }]
    )
    docs.to_csv(snapshot / "evidence_documents.csv", index=False)
    relations = pd.DataFrame(
        [
            {"tx_id": "tx-000", "evidence_id": evidence_id, "relation": "payment_proof", "status": "approved"},
            {"tx_id": "tx-001", "evidence_id": evidence_id, "relation": "payment_proof", "status": "candidate"},
        ]
    )
    relations.to_csv(snapshot / "transaction_evidence.csv", index=False)
    manifest = {
        "artifact": "acct.transaction-evidence@1",
        "snapshot_id": "PM",
        "source_ledger_sha256": _sha(run / "ledger_canonical.csv"),
        "required_transaction_count": population_size,
        "required_tx_ids_sha256": payment_evidence._required_hash(required.tx_id.tolist()),
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fake_pdf(_html: Path, target: Path, *, browser_bin=None) -> Path:
        target.write_bytes(b"%PDF-synthetic")
        return target

    monkeypatch.setattr(payment_evidence, "render_pdf", fake_pdf)
    result = payment_evidence.build_payment_evidence_report(
        run_root=run, snapshot_dir=snapshot, out_root=tmp_path / "bundle"
    )
    report = Path(result["report_html"]).read_text(encoding="utf-8")
    assert report.count('<tr data-status=') == population_size
    assert report.count('<span class="status status-approved">') == 1
    assert report.count('<span class="status status-candidate">') == 1
    assert report.count('href="evidence/') == 2
    assert len(list((tmp_path / "bundle" / "pm_payment_evidence" / "evidence").glob("*." + extension))) == 1
    assert f"evidence/{evidence_id}.{extension}" in report



def test_bill_only_is_not_mislabeled_payment_candidate_and_preview_is_explicit() -> None:
    required = pd.DataFrame([
        {"tx_id": "tx-bill", "Date": "2026-10-10", "amount": "149216", "Currency": "ARS",
         "payer": "PM", "Tipo": "Impuestos", "Box": "Property Management",
         "receiver": "Tigre", "evidence_review_note": "Falta recibo de pago"},
        {"tx_id": "tx-candidate", "Date": "2026-10-13", "amount": "84996", "Currency": "ARS",
         "payer": "PM", "Tipo": "Impuestos", "Box": "Property Management",
         "receiver": "ARBA", "evidence_review_note": "Fecha del comprobante diferente"},
    ])
    eid = "a" * 64
    docs = pd.DataFrame([{"evidence_id": eid, "content_sha256": eid,
                          "media_type": "application/pdf", "display_name": "liquidacion.pdf",
                          "href": f"evidence/{eid}.pdf"}])
    relations = pd.DataFrame([
        {"tx_id": "tx-bill", "evidence_id": eid, "relation": "liability_source", "status": "candidate"},
        {"tx_id": "tx-candidate", "evidence_id": eid, "relation": "payment_proof", "status": "candidate"},
    ])
    html = payment_evidence._html(required, docs, relations,
                                  source_run_id="RUN_FBPM", cutoff="2026-10-13",
                                  snapshot_id="PRIVATE_REVIEW", review_preview=True)
    assert html.count('<tr data-status="bill_only"') == 1
    assert html.count('<tr data-status="candidate"') == 1
    assert "Solo boleta; falta prueba de pago" in html
    assert "Ver boleta (no acredita pago)" in html
    assert "Revisar comprobante (no aprobado)" in html
    assert "REPORTE PRIVADO EN REVISIÓN" in html
    assert "Falta recibo de pago" in html



def test_counterpart_does_not_count_as_payment_candidate_or_approved() -> None:
    required = pd.DataFrame([
       {"tx_id":"approved-tx","transaction_id":"100","Date":"2024-10-07","amount":"100",
        "Currency":"ARS","payer":"PM","Tipo":"Impuestos","Box":"Property Management",
        "receiver":"ARBA", "evidence_review_group":""},
       {"tx_id":"mirror-tx","transaction_id":"101","Date":"2024-10-07","amount":"100",
        "Currency":"ARS","payer":"MI","Tipo":"Impuestos","Box":"Property Management",
        "receiver":"PM", "evidence_review_group":"COUNTERPART_OF_ALREADY_APPROVED_PAYMENT",
        "evidence_review_note":"Evento aprobado en operación 100, no otro pago"},
    ])
    h="a"*64
    docs=pd.DataFrame([{"evidence_id":h,"content_sha256":h,"media_type":"application/pdf",
                        "display_name":"example.pdf","href":f"evidence/{h}.pdf"}])
    relations=pd.DataFrame([
      {"tx_id":"approved-tx","evidence_id":h,"relation":"payment_proof","status":"approved"},
      {"tx_id":"mirror-tx","evidence_id":h,"relation":"other_support","status":"candidate"},
    ])
    body=payment_evidence._html(required,docs,relations,source_run_id="RUN_FBPM",
                                cutoff="2024-10-07",snapshot_id="PRIVATE_REVIEW",
                                review_preview=True)
    assert body.count('<tr data-status="approved"')==1
    assert body.count('<tr data-status="counterpart"')==1
    assert body.count('<tr data-status="candidate"')==0
    assert "Recibo de la contrapartida" in body
    assert "Contrapartidas (no pagos adicionales)" in body


def test_statement_column_optional_separates_bills_from_payment_proofs() -> None:
    required = pd.DataFrame([
        {"transaction_id":"1","tx_id":"tx-only-statement","Date":"2024-09-07",
         "payer":"PM","Tipo":"Impuestos","amount":"62640","Currency":"ARS","receiver":"AGIP"},
        {"transaction_id":"2","tx_id":"tx-proof-and-statement","Date":"2024-10-07",
         "payer":"PM","Tipo":"Impuestos","amount":"119602","Currency":"ARS","receiver":"AGIP"},
        {"transaction_id":"3","tx_id":"tx-without-statement","Date":"2024-11-07",
         "payer":"PM","Tipo":"Impuestos","amount":"200","Currency":"ARS","receiver":"AGIP"},
    ])
    docs=pd.DataFrame([
        {"evidence_id":"a"*64,"content_sha256":"a"*64,"media_type":"application/pdf",
         "display_name":"bank_statement.pdf","href":"evidence/"+"a"*64+".pdf"},
        {"evidence_id":"b"*64,"content_sha256":"b"*64,"media_type":"application/pdf",
         "display_name":"tax_bill.pdf","href":"evidence/"+"b"*64+".pdf"},
    ])
    relations=pd.DataFrame([
        {"tx_id":"tx-only-statement","evidence_id":"a"*64,
         "relation":"statement_context","status":"candidate"},
        {"tx_id":"tx-proof-and-statement","evidence_id":"a"*64,
         "relation":"payment_proof","status":"approved"},
        {"tx_id":"tx-proof-and-statement","evidence_id":"a"*64,
         "relation":"statement_context","status":"candidate"},
        {"tx_id":"tx-without-statement","evidence_id":"b"*64,
         "relation":"payment_proof","status":"approved"},
    ])
    out=payment_evidence._html(required,docs,relations,source_run_id="RUN",
                               cutoff="2024-11-07",snapshot_id="PRIVATE",
                               review_preview=True)
    assert "Statement / boleta (opcional)" in out
    assert out.count('<tr data-status="statement_only"')==1
    assert out.count('<tr data-status="approved"')==2
    assert out.count('<tr data-status="missing"')==0
    assert "Ver estado de cuenta (no es recibo)" in out
    assert "Solo estado de cuenta; pago no validado" in out
    # Two columns: credit-card proof is also traceable to its source statement.
    assert out.count("evidence/"+("a"*64)+".pdf")==3
    assert out.count("evidence/"+("b"*64)+".pdf")==1
    assert "Solo boleta; falta prueba de pago" not in out
