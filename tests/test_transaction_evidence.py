from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.evidence.relations import (
    EvidenceContractError,
    TransactionEvidenceIndex,
    prepare_evidence_html_frame,
)
from accounting.professional.drilldown import build_professional_flow_drilldowns
from accounting.professional.evidence_enrichment import (
    enrich_professional_drilldowns_with_evidence,
    evidence_coverage,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_id": "ev-pdf",
                "content_sha256": "a" * 64,
                "media_type": "application/pdf",
                "display_name": "synthetic-payment-proof.pdf",
                "href": "evidence/synthetic-payment-proof.pdf",
            },
            {
                "evidence_id": "ev-image",
                "content_sha256": "b" * 64,
                "media_type": "image/png",
                "display_name": "synthetic-screenshot.png",
                "href": "evidence/synthetic-screenshot.png",
            },
        ]
    )


def _relations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tx_id": "tx-approved",
                "evidence_id": "ev-pdf",
                "relation": "payment_proof",
                "status": "approved",
            },
            {
                "tx_id": "tx-review",
                "evidence_id": "ev-image",
                "relation": "payment_proof",
                "status": "candidate",
            },
        ]
    )


def test_transaction_evidence_is_optional_and_supports_pdf_and_image() -> None:
    index = TransactionEvidenceIndex.from_frames(_documents(), _relations())
    assert index.document_count == 2
    assert index.relation_count == 2
    assert index.approved_relation_count == 1
    assert index.links_for_tx("tx-approved")[0].short_label == "PDF"
    assert index.has_candidate_for_tx("tx-review") is True

    source = pd.DataFrame(
        [
            {"tx_id": "tx-approved", "amount": 10},
            {"tx_id": "tx-review", "amount": 20},
            {"tx_id": "tx-missing", "amount": 30},
        ]
    )
    coverage = evidence_coverage(source, index)
    assert coverage.transaction_rows == 3
    assert coverage.linked_rows == 1
    assert coverage.review_rows == 1
    assert coverage.missing_rows == 1
    assert coverage.coverage_pct == 33.3

    frame, replacements = prepare_evidence_html_frame(source, index)
    assert frame.columns[-1] == "Evidence"
    rendered = frame.to_html(index=False, escape=True)
    for token, replacement in replacements.items():
        rendered = rendered.replace(token, replacement)
    assert "synthetic-payment-proof.pdf" in rendered
    assert ">PDF</a>" in rendered
    assert "Review" in rendered
    assert "—" in rendered


def test_transaction_evidence_rejects_unsafe_or_partial_contracts(tmp_path: Path) -> None:
    bad_documents = _documents()
    bad_documents.loc[0, "href"] = "javascript:alert(1)"
    with pytest.raises(EvidenceContractError, match="unsupported scheme"):
        TransactionEvidenceIndex.from_frames(bad_documents, _relations())

    run = tmp_path / "run"
    run.mkdir(parents=True)
    _documents().to_csv(run / "evidence_documents.csv", index=False)
    pack = tmp_path / "pack"
    with pytest.raises(EvidenceContractError, match="incomplete"):
        enrich_professional_drilldowns_with_evidence(pack_dir=pack, run_root=run)


def test_professional_tax_drilldown_exposes_link_review_missing_and_coverage_without_changing_accounting_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
                "amount_in": 0,
                "amount_out": 600,
                "net_amount": -600,
                "amount_abs": 600,
                "n_tx": 3,
                "source_tx_ids_sample": "tx-approved;tx-review;tx-missing",
            }
        ],
    )
    _write(
        run / "classification_audit.csv",
        [
            {
                "tx_id": "tx-approved",
                "period": "2026-01",
                "Currency": "ARS",
                "amount": 100,
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
            },
            {
                "tx_id": "tx-review",
                "period": "2026-01",
                "Currency": "ARS",
                "amount": 200,
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
            },
            {
                "tx_id": "tx-missing",
                "period": "2026-01",
                "Currency": "ARS",
                "amount": 300,
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
            },
        ],
    )
    _write(
        tables / "monthly_tables_flow_subbucket_all_measures.csv",
        [
            {
                "measure": "amount_out",
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "taxes",
                "2026-01": 600,
            }
        ],
    )
    _documents().to_csv(run / "evidence_documents.csv", index=False)
    _relations().to_csv(run / "transaction_evidence.csv", index=False)

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index_before = pd.read_csv(paths["index"])
    row = index_before.iloc[0]
    detail_csv = pack / row["detail_csv_relpath"]
    detail_before = detail_csv.read_bytes()

    result = enrich_professional_drilldowns_with_evidence(
        pack_dir=pack,
        run_root=run,
    )
    assert result["evidence_loaded"] is True
    assert result["approved_relations"] == 1
    assert result["transaction_rows"] == 3
    assert result["linked_transaction_rows"] == 1
    assert result["review_transaction_rows"] == 1
    assert result["missing_transaction_rows"] == 1
    assert result["coverage_pct"] == 33.3

    # Evidence is a passive projection over governed accounting evidence.
    assert detail_csv.read_bytes() == detail_before
    index_after = pd.read_csv(paths["index"])
    pd.testing.assert_frame_equal(index_after, index_before)

    detail_html = (pack / row["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "Transaction evidence" in detail_html
    assert "1/3</strong> transaction rows linked (33.3%)" in detail_html
    assert "Review: 1. Missing: 1." in detail_html
    assert "evidence/synthetic-payment-proof.pdf" in detail_html
    assert ">PDF</a>" in detail_html
    assert ">Evidence<" in detail_html
    assert "Review" in detail_html
    assert "—" in detail_html

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    evidence_manifest = manifest["transaction_evidence"]
    assert evidence_manifest["approved_relations"] == 1
    assert evidence_manifest["coverage_pct"] == 33.3
    assert evidence_manifest["accounting_authority_changed"] is False
    assert evidence_manifest["private_evidence_publication_implied"] is False
    assert "not transaction validity" in evidence_manifest["coverage_semantics"]

    # Rerun is idempotent: the bounded evidence section is replaced, not duplicated.
    enrich_professional_drilldowns_with_evidence(pack_dir=pack, run_root=run)
    rerendered = (pack / row["detail_html_relpath"]).read_text(encoding="utf-8")
    assert rerendered.count("acct-transaction-evidence:start") == 1


def test_missing_evidence_sidecar_is_a_noop(tmp_path: Path) -> None:
    result = enrich_professional_drilldowns_with_evidence(
        pack_dir=tmp_path / "pack",
        run_root=tmp_path / "run",
    )
    assert result == {
        "evidence_loaded": False,
        "enriched_pages": 0,
        "transaction_rows": 0,
        "linked_transaction_rows": 0,
        "review_transaction_rows": 0,
        "missing_transaction_rows": 0,
        "coverage_pct": 0.0,
        "documents": 0,
        "relations": 0,
        "approved_relations": 0,
    }
