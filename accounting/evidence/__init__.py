"""Optional supporting-document relations for canonical accounting transactions.

Evidence enriches accounting outputs but never participates in transaction identity,
classification, or amount recognition.
"""

from accounting.evidence.relations import (
    EvidenceContractError,
    EvidenceLink,
    TransactionEvidenceIndex,
    load_transaction_evidence,
    prepare_evidence_html_frame,
)

__all__ = [
    "EvidenceContractError",
    "EvidenceLink",
    "TransactionEvidenceIndex",
    "load_transaction_evidence",
    "prepare_evidence_html_frame",
]
