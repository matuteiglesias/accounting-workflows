from __future__ import annotations

"""Validated optional transaction-to-document evidence relations.

The canonical ledger remains authoritative for transaction identity and accounting
semantics. This module only validates and projects supporting-document references
onto already-governed transaction rows.
"""

from dataclasses import dataclass
import html
import re
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlparse

import pandas as pd


EVIDENCE_DOCUMENT_COLUMNS = (
    "evidence_id",
    "content_sha256",
    "media_type",
    "display_name",
    "href",
)
TRANSACTION_EVIDENCE_COLUMNS = (
    "tx_id",
    "evidence_id",
    "relation",
    "status",
)

SUPPORTED_MEDIA_TYPES = frozenset(
    {"application/pdf", "image/png", "image/jpeg"}
)
SUPPORTED_RELATIONS = frozenset(
    {
        "payment_proof",
        "transfer_proof",
        "statement_context",
        "liability_source",
        "other_support",
    }
)
SUPPORTED_STATUSES = frozenset({"approved", "candidate", "rejected"})
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class EvidenceContractError(ValueError):
    """Raised when an optional evidence artifact violates its contract."""


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise EvidenceContractError(f"{name} missing required columns: {missing}")


def _clean(value: object, *, field: str, row: int) -> str:
    if value is None or pd.isna(value):
        raise EvidenceContractError(f"{field} must be non-empty at row {row}")
    text = str(value).strip()
    if not text:
        raise EvidenceContractError(f"{field} must be non-empty at row {row}")
    return text


def _validate_href(value: object, *, row: int) -> str:
    href = _clean(value, field="href", row=row)
    parsed = urlparse(href)
    if parsed.scheme not in {"", "file", "https"}:
        raise EvidenceContractError(
            f"href uses unsupported scheme at row {row}: {parsed.scheme!r}"
        )
    if parsed.scheme == "":
        path = PurePosixPath(href.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise EvidenceContractError(
                f"relative href must remain inside its evidence bundle at row {row}"
            )
    return href


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    evidence_id: str
    content_sha256: str
    media_type: str
    display_name: str
    href: str
    relation: str
    status: str

    @property
    def short_label(self) -> str:
        if self.media_type == "application/pdf":
            return "PDF"
        return "Image"


@dataclass(frozen=True, slots=True)
class TransactionEvidenceIndex:
    """Immutable lookup of validated evidence relations by canonical ``tx_id``."""

    _links_by_tx: Mapping[str, tuple[EvidenceLink, ...]]
    document_count: int
    relation_count: int

    @classmethod
    def from_frames(
        cls,
        documents: pd.DataFrame,
        relations: pd.DataFrame,
    ) -> "TransactionEvidenceIndex":
        _require_columns(
            documents, EVIDENCE_DOCUMENT_COLUMNS, name="evidence_documents"
        )
        _require_columns(
            relations, TRANSACTION_EVIDENCE_COLUMNS, name="transaction_evidence"
        )

        docs: dict[str, dict[str, str]] = {}
        for row_number, (_, row) in enumerate(documents.iterrows(), start=1):
            evidence_id = _clean(
                row["evidence_id"], field="evidence_id", row=row_number
            )
            if evidence_id in docs:
                raise EvidenceContractError(
                    f"duplicate evidence_id at row {row_number}: {evidence_id!r}"
                )
            sha256 = _clean(
                row["content_sha256"], field="content_sha256", row=row_number
            )
            if not _SHA256_RE.fullmatch(sha256):
                raise EvidenceContractError(
                    f"content_sha256 must be 64 hexadecimal characters at row {row_number}"
                )
            media_type = _clean(
                row["media_type"], field="media_type", row=row_number
            )
            if media_type not in SUPPORTED_MEDIA_TYPES:
                raise EvidenceContractError(
                    f"unsupported media_type at row {row_number}: {media_type!r}"
                )
            docs[evidence_id] = {
                "content_sha256": sha256.lower(),
                "media_type": media_type,
                "display_name": _clean(
                    row["display_name"], field="display_name", row=row_number
                ),
                "href": _validate_href(row["href"], row=row_number),
            }

        links_by_tx: dict[str, list[EvidenceLink]] = {}
        seen_relations: set[tuple[str, str, str, str]] = set()
        for row_number, (_, row) in enumerate(relations.iterrows(), start=1):
            tx_id = _clean(row["tx_id"], field="tx_id", row=row_number)
            evidence_id = _clean(
                row["evidence_id"], field="evidence_id", row=row_number
            )
            relation = _clean(row["relation"], field="relation", row=row_number)
            status = _clean(row["status"], field="status", row=row_number)
            if relation not in SUPPORTED_RELATIONS:
                raise EvidenceContractError(
                    f"unsupported relation at row {row_number}: {relation!r}"
                )
            if status not in SUPPORTED_STATUSES:
                raise EvidenceContractError(
                    f"unsupported status at row {row_number}: {status!r}"
                )
            if evidence_id not in docs:
                raise EvidenceContractError(
                    f"transaction_evidence references unknown evidence_id at row {row_number}: "
                    f"{evidence_id!r}"
                )
            relation_key = (tx_id, evidence_id, relation, status)
            if relation_key in seen_relations:
                raise EvidenceContractError(
                    f"duplicate transaction evidence relation at row {row_number}: "
                    f"{relation_key!r}"
                )
            seen_relations.add(relation_key)
            doc = docs[evidence_id]
            links_by_tx.setdefault(tx_id, []).append(
                EvidenceLink(
                    evidence_id=evidence_id,
                    content_sha256=doc["content_sha256"],
                    media_type=doc["media_type"],
                    display_name=doc["display_name"],
                    href=doc["href"],
                    relation=relation,
                    status=status,
                )
            )

        frozen = MappingProxyType(
            {
                tx_id: tuple(
                    sorted(
                        links,
                        key=lambda item: (
                            item.status != "approved",
                            item.relation,
                            item.evidence_id,
                        ),
                    )
                )
                for tx_id, links in links_by_tx.items()
            }
        )
        return cls(
            _links_by_tx=frozen,
            document_count=len(docs),
            relation_count=len(seen_relations),
        )

    def links_for_tx(
        self,
        tx_id: object,
        *,
        status: str | None = "approved",
    ) -> tuple[EvidenceLink, ...]:
        text = "" if tx_id is None or pd.isna(tx_id) else str(tx_id).strip()
        links = self._links_by_tx.get(text, ())
        if status is None:
            return links
        return tuple(link for link in links if link.status == status)

    def has_candidate_for_tx(self, tx_id: object) -> bool:
        return bool(self.links_for_tx(tx_id, status="candidate"))

    @property
    def approved_relation_count(self) -> int:
        return sum(
            1
            for links in self._links_by_tx.values()
            for link in links
            if link.status == "approved"
        )


def load_transaction_evidence(
    documents_path: Path,
    relations_path: Path,
) -> TransactionEvidenceIndex:
    return TransactionEvidenceIndex.from_frames(
        pd.read_csv(documents_path),
        pd.read_csv(relations_path),
    )


def _anchor(link: EvidenceLink, label: str) -> str:
    return (
        f"<a href='{html.escape(link.href, quote=True)}' "
        "target='_blank' rel='noopener noreferrer' "
        f"title='{html.escape(link.display_name, quote=True)}'>"
        f"{html.escape(label)}</a>"
    )


def _cell_html(index: TransactionEvidenceIndex, tx_id: object) -> str:
    approved = index.links_for_tx(tx_id, status="approved")
    if approved:
        if len(approved) == 1:
            return _anchor(approved[0], approved[0].short_label)
        return " · ".join(
            _anchor(link, f"{link.short_label} {position}")
            for position, link in enumerate(approved, start=1)
        )
    if index.has_candidate_for_tx(tx_id):
        return "Review"
    return "—"


def prepare_evidence_html_frame(
    df: pd.DataFrame,
    index: TransactionEvidenceIndex,
) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Add a final Evidence column using safe replacement tokens.

    Callers may render the returned frame with ordinary HTML escaping enabled,
    then replace only the exact generated tokens with the trusted anchor markup in
    the returned mapping. This keeps every original transaction field escaped.
    """

    if df.empty or "tx_id" not in df.columns:
        return df, MappingProxyType({})

    out = df.copy()
    replacements: dict[str, str] = {}
    tokens: list[str] = []
    for position, tx_id in enumerate(out["tx_id"].tolist()):
        token = f"__ACCT_EVIDENCE_LINK_{position:06d}__"
        tokens.append(token)
        replacements[token] = _cell_html(index, tx_id)
    out["Evidence"] = tokens
    return out, MappingProxyType(replacements)
