"""Downstream factual projections for documentation consumers.

The docs-input layer is not an accounting authority. It packages already-governed
accounting/evidence populations and optional prospective commitments without
reclassifying ledger rows or inferring legal rights.
"""

DOCS_INPUT_SCHEMA = "acct.docs-input@1"
TREASURY_COMMITMENTS_SCHEMA = "treasury_commitments.v1"

__all__ = ["DOCS_INPUT_SCHEMA", "TREASURY_COMMITMENTS_SCHEMA"]
