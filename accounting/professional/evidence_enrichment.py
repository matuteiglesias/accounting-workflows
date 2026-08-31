from __future__ import annotations

"""Optional evidence projection over already-built professional drilldowns.

This module is intentionally downstream of accounting calculation. It reads the
existing drilldown index/detail CSVs and an optional transaction-evidence sidecar,
then adds a passive HTML evidence projection. It never changes ledger rows,
reconciliation values, drilldown membership, or accounting totals.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class EvidenceCoverage:
    transaction_rows: int
    linked_rows: int
    review_rows: int
    missing_rows: int

    @property
    def coverage_pct(self) -> float:
        if self.transaction_rows == 0:
            return 0.0
        return round(100.0 * self.linked_rows / self.transaction_rows, 1)


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


def evidence_coverage(detail: pd.DataFrame, evidence_index) -> EvidenceCoverage:
    if detail.empty or "tx_id" not in detail.columns:
        return EvidenceCoverage(0, 0, 0, 0)

    transaction_rows = 0
    linked_rows = 0
    review_rows = 0
    missing_rows = 0
    for raw_tx_id in detail["tx_id"].tolist():
        if raw_tx_id is None or pd.isna(raw_tx_id):
            continue
        tx_id = str(raw_tx_id).strip()
        if not tx_id:
            continue
        transaction_rows += 1
        if evidence_index.links_for_tx(tx_id, status="approved"):
            linked_rows += 1
        elif evidence_index.has_candidate_for_tx(tx_id):
            review_rows += 1
        else:
            missing_rows += 1
    return EvidenceCoverage(
        transaction_rows=transaction_rows,
        linked_rows=linked_rows,
        review_rows=review_rows,
        missing_rows=missing_rows,
    )


def _render_evidence_section(
    detail: pd.DataFrame, evidence_index
) -> tuple[str, EvidenceCoverage]:
    display, replacements = prepare_evidence_html_frame(detail, evidence_index)
    coverage = evidence_coverage(detail, evidence_index)
    if not replacements:
        return "", coverage

    table_html = display.to_html(
        index=False,
        escape=True,
        classes="detail evidence-detail",
        border=0,
    )
    for token, replacement in replacements.items():
        table_html = table_html.replace(token, replacement)

    coverage_text = (
        f"<strong>{coverage.linked_rows}/{coverage.transaction_rows}</strong> "
        f"transaction rows linked ({coverage.coverage_pct:.1f}%). "
        f"Review: {coverage.review_rows}. Missing: {coverage.missing_rows}."
    )
    section = (
        f"{_SECTION_START}\n"
        "<h2>Transaction evidence</h2>\n"
        "<p>Supporting documents enrich these already-governed transaction rows; "
        "they do not change accounting recognition or values.</p>\n"
        f"<p class='evidence-coverage'>{coverage_text}</p>\n"
        f"{table_html}\n"
        f"{_SECTION_END}"
    )
    return section, coverage


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
) -> dict[str, int | float | bool | str]:
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
            "transaction_rows": 0,
            "linked_transaction_rows": 0,
            "review_transaction_rows": 0,
            "missing_transaction_rows": 0,
            "coverage_pct": 0.0,
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
    total_rows = 0
    linked_rows = 0
    review_rows = 0
    missing_rows = 0
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
        section, coverage = _render_evidence_section(detail, evidence_index)
        existing = detail_html.read_text(encoding="utf-8")
        updated = _inject_section(existing, section)
        if updated != existing:
            detail_html.write_text(updated, encoding="utf-8")
        if section:
            enriched_pages += 1
            total_rows += coverage.transaction_rows
            linked_rows += coverage.linked_rows
            review_rows += coverage.review_rows
            missing_rows += coverage.missing_rows

    overall_coverage_pct = (
        round(100.0 * linked_rows / total_rows, 1) if total_rows else 0.0
    )
    evidence_manifest = {
        "evidence_documents_path": str(documents_path),
        "transaction_evidence_path": str(relations_path),
        "documents": evidence_index.document_count,
        "relations": evidence_index.relation_count,
        "approved_relations": evidence_index.approved_relation_count,
        "enriched_pages": enriched_pages,
        "transaction_rows": total_rows,
        "linked_transaction_rows": linked_rows,
        "review_transaction_rows": review_rows,
        "missing_transaction_rows": missing_rows,
        "coverage_pct": overall_coverage_pct,
        "coverage_semantics": "supporting-evidence coverage only; not transaction validity",
        "accounting_authority_changed": False,
        "private_evidence_publication_implied": False,
    }

    manifest_path = drill_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["transaction_evidence"] = evidence_manifest
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {"evidence_loaded": True, **evidence_manifest}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Passively enrich professional drilldowns with transaction evidence"
    )
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--evidence-documents", type=Path)
    parser.add_argument("--transaction-evidence", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = enrich_professional_drilldowns_with_evidence(
        pack_dir=args.pack,
        run_root=args.run_root,
        evidence_documents_path=args.evidence_documents,
        transaction_evidence_path=args.transaction_evidence,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
