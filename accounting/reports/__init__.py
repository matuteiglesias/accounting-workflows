"""Presentation-only human report products over governed accounting artifacts.

This package may select, format, validate, and render already-governed facts. It
must not classify ledger rows or create accounting authority.
"""

REPORT_CATALOG_SCHEMA = "accounting_report_catalog.v1"
REPORT_MANIFEST_SCHEMA = "accounting_report_manifest.v1"

__all__ = ["REPORT_CATALOG_SCHEMA", "REPORT_MANIFEST_SCHEMA"]
