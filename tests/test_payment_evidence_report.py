from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from accounting.reports import payment_evidence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_payment_evidence_report_rebuilds_frozen_rows_and_only_links_approved_payment_proof(
    tmp_path: Path, monkeypatch
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
            for index in range(183)
        ]
    )
    ledger.to_csv(run / "ledger_canonical.csv", index=False)
    required = ledger.copy()
    required["status"] = "pagado"
    required.to_csv(snapshot / "required_transactions.csv", index=False)
    source = snapshot / "evidence" / ("a" * 64 + ".pdf")
    source.parent.mkdir()
    source.write_bytes(b"%PDF-approved")
    evidence_id = _sha(source)
    docs = pd.DataFrame(
        [{
            "evidence_id": evidence_id,
            "content_sha256": evidence_id,
            "media_type": "application/pdf",
            "display_name": "approved.pdf",
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
        "required_transaction_count": 183,
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
    assert report.count('<tr data-status=') == 183
    assert report.count('<span class="status status-approved">') == 1
    assert report.count('<span class="status status-candidate">') == 1
    assert report.count('href="evidence/') == 1
    assert len(list((tmp_path / "bundle" / "pm_payment_evidence" / "evidence").glob("*.pdf"))) == 1
