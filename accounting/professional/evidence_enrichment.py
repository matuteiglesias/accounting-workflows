from __future__ import annotations

"""Optional evidence projection over already-built professional drilldowns.

This module is intentionally downstream of accounting calculation. It reads the
existing drilldown index/detail CSVs and an optional transaction-evidence sidecar,
then adds a passive HTML evidence projection. It never changes ledger rows,
reconciliation values, drilldown membership, or accounting totals.
"""

import json
import re
from pathlib import Path

import pandas as pd

from accounting.evidence.relations import (
    EvidenceContractError,
    load_transaction_evidence,
    prepare_evidence_html_frame,
)
from accounting.professional.drilldown_legacy import INDEX_FILENAME, MANIFEST_FILENAME


_SECTION_START = "<!-- acct-transaction-evidence:start -->"
_SECTION_END = "<!-- acct-transaction-evidence:end -->"
_SECTION_RE = re.compile(
    re.escape(_SECTION_START) + r".*?" + re.escape(_SECTION_END),
    flags=re.DOTALL,
)


def _resolve_sidecar_paths(
    *,
    run_root: Path | None,
    evidence_documents_path: Path | None,
    transaction_evidence_path: Path | None,
) -> tuple[Path | None, Path | None]:
    documents = Path(evidence_documents_path) if evidence_documents_path else None
    relations = Path(transaction_evidence_path) if transaction_evidence_path else None
    if run_root is not None:
        root = Path(run_root)
        documents = documents or root / "evidence_documents.csv"
        relations = relations or root / "transaction_evidence.csv"

    documents_exists = documents is not None and documents.exists()
    relations_exists = relations is not None and relations.exists()
    if not documents_exists and not relations_exists:
        return None, None
    if documents_exists != relations_exists:
        raise EvidenceContractError(
            "optional transaction evidence is incomplete: both evidence_documents.csv "
            "and transaction_evidence.csv must be present"
        )
    return documents, relations


def _render_evidence_section(detail: pd.DataFrame, evidence_index) -> tuple[str, int]:
    display, replacements = prepare_evidence_html_frame(detail, evidence_index)
    if not replacements:
        return "", 0

    table_html = display.to_html(
        index=False,
        escape=True,
        classes="detail evidence-detail",
        border=0,
    )
    for token, replacement in replacements.items():
        table_html = table_html.replace(token, replacement)

    tx_ids = (
        detail["tx_id"].dropna().astype(str).str.strip()
        if "tx_id" in detail.columns
        else pd.Series(dtype=str)
    )
    linked = sum(bool(evidence_index.links_for_tx(tx_id)) for tx_id in tx_ids)
    section = (
        f"{_SECTION_START}\n"
        "<h2>Transaction evidence</h2>\n"
        "<p>Supporting documents enrich these already-governed transaction rows; "
        "they do not change accounting recognition or values.</p>\n"
        f"{table_html}\n"
        f"{_SECTION_END}"
    )
    return section, linked


def _inject_section(html_text: str, section: str) -> str:
    cleaned = _SECTION_RE.sub("", html_text)
    if not section:
        return cleaned
    if "</body>" in cleaned:
        return cleaned.replace("</body>", f"{section}\n</body>", 1)
    return cleaned + "\n" + section + "\n"


def enrich_professional_drilldowns_with_evidence(
    *,
    pack_dir: Path,
    run_root: Path | None = None,
    evidence_documents_path: Path | None = None,
    transaction_evidence_path: Path | None = None,
) -> dict[str, int | bool | str]:
    """Project an optional evidence sidecar into existing drilldown HTML pages.

    Missing sidecars are a supported no-op. A partially present or malformed
    sidecar fails closed. The function is idempotent: reruns replace the bounded
    evidence section rather than appending duplicate markup.
    """

    pack_dir = Path(pack_dir)
    documents_path, relations_path = _resolve_sidecar_paths(
        run_root=Path(run_root) if run_root is not None else None,
        evidence_documents_path=evidence_documents_path,
        transaction_evidence_path=transaction_evidence_path,
    )
    if documents_path is None or relations_path is None:
        return {
            "evidence_loaded": False,
            "enriched_pages": 0,
            "linked_transaction_rows": 0,
            "documents": 0,
            "relations": 0,
            "approved_relations": 0,
        }

    evidence_index = load_transaction_evidence(documents_path, relations_path)
    drill_dir = pack_dir / "drilldown"
    index_path = drill_dir / INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"professional drilldown index not found: {index_path}")

    index = pd.read_csv(index_path)
    enriched_pages = 0
    linked_rows = 0
    for _, row in index.iterrows():
        detail_csv_rel = str(row.get("detail_csv_relpath") or "").strip()
        detail_html_rel = str(row.get("detail_html_relpath") or "").strip()
        if not detail_csv_rel or not detail_html_rel:
            continue
        detail_csv = pack_dir / detail_csv_rel
        detail_html = pack_dir / detail_html_rel
        if not detail_csv.exists() or not detail_html.exists():
            continue

        detail = pd.read_csv(detail_csv)
        section, linked = _render_evidence_section(detail, evidence_index)
        existing = detail_html.read_text(encoding="utf-8")
        updated = _inject_section(existing, section)
        if updated != existing:
            detail_html.write_text(updated, encoding="utf-8")
        if section:
            enriched_pages += 1
            linked_rows += linked

    manifest_path = drill_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["transaction_evidence"] = {
            "evidence_documents_path": str(documents_path),
            "transaction_evidence_path": str(relations_path),
            "documents": evidence_index.document_count,
            "relations": evidence_index.relation_count,
            "approved_relations": evidence_index.approved_relation_count,
            "enriched_pages": enriched_pages,
            "linked_transaction_rows": linked_rows,
            "accounting_authority_changed": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "evidence_loaded": True,
        "enriched_pages": enriched_pages,
        "linked_transaction_rows": linked_rows,
        "documents": evidence_index.document_count,
        "relations": evidence_index.relation_count,
        "approved_relations": evidence_index.approved_relation_count,
        "evidence_documents_path": str(documents_path),
        "transaction_evidence_path": str(relations_path),
    }
