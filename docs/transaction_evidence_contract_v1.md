# Transaction evidence contract v1

## Purpose

`acct.transaction-evidence@1` is an **optional enrichment artifact** connecting an already-canonical accounting transaction (`tx_id`) to one or more supporting documents.

It does not create transactions, determine accounting classification, change amounts, or participate in ledger identity. Removing the artifact must leave accounting calculations unchanged.

## Producer / consumer boundary

- `accounting-doc-triage` is the intended producer of approved evidence relations.
- `accounting-workflows` consumes the relation passively for evidence-aware drilldowns and reports.
- raw document intake, OCR, parsing, classification candidates, and document custody remain outside this repository.

## Files

A complete sidecar contains both files:

### `evidence_documents.csv`

| column | meaning |
| --- | --- |
| `evidence_id` | stable producer-owned document identity |
| `content_sha256` | SHA-256 of the exact source bytes |
| `media_type` | `application/pdf`, `image/png`, or `image/jpeg` |
| `display_name` | human-facing non-authoritative label |
| `href` | controlled relative/private evidence link |

### `transaction_evidence.csv`

| column | meaning |
| --- | --- |
| `tx_id` | canonical transaction identity owned by Accounting Workflows |
| `evidence_id` | document identity from `evidence_documents.csv` |
| `relation` | semantic evidence relationship |
| `status` | `approved`, `candidate`, or `rejected` |

Initial relation vocabulary:

- `payment_proof`
- `transfer_proof`
- `statement_context`
- `liability_source`
- `other_support`

Only `approved` relations become clickable supporting evidence. `candidate` may be surfaced as review-required state but is not proof of a canonical accounting relationship.

## Safety and identity

- `tx_id` is never derived from a document path or filename.
- document identity is content-based upstream; friendly names and classifications are metadata.
- a malformed or partially present sidecar fails closed.
- unsafe href schemes are rejected.
- evidence rows may be many-to-many: one transaction can have multiple documents and a statement can support multiple transactions.
- no source PDF/image is copied into this repository or public report bundle by this contract.

## E0 implementation boundary

The first implementation is deliberately passive. Existing professional drilldowns are built exactly as before; a separate enrichment step reads their transaction-detail CSVs and inserts a bounded evidence section into detail HTML pages. The underlying detail CSV, drilldown index, accounting values, and reconciliation decisions remain unchanged.

Publication/bundling of private evidence is a separate future decision and is not implied by a clickable local/private href.
