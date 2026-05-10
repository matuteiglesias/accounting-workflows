"""Frontend snapshot manifest helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SCHEMA_NAME = "accounting_frontend_snapshot.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_frontend_snapshot_manifest(
    *,
    source_run_id: str | None,
    status: str,
    source_paths: dict[str, Any],
    files: list[str],
    metrics: dict[str, Any],
    debt: dict[str, Any],
    reports: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable frontend snapshot manifest shape."""
    manifest = {
        "schema_name": SCHEMA_NAME,
        "built_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "status": status,
        "source_paths": source_paths,
        "files": files,
        "metrics": metrics,
        "debt": debt,
        "reports": reports,
    }
    if extra:
        manifest.update(extra)
    return manifest
