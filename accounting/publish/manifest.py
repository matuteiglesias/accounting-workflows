"""Published accounting bundle manifest helpers.

The old ``accounting_frontend_snapshot.v1`` naming described a retired
viewer-oriented architecture.  Publication is now an artifact handoff:
governed metrics and debt contracts are packaged without owning a UI.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

SCHEMA_NAME = "accounting_public_bundle.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_public_bundle_manifest(
    *,
    source_run_id: str | None,
    status: str,
    source_paths: dict[str, Any],
    files: list[str],
    metrics: dict[str, Any],
    debt: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable published accounting bundle manifest."""
    manifest = {
        "schema_name": SCHEMA_NAME,
        "built_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "status": status,
        "source_paths": source_paths,
        "files": files,
        "metrics": metrics,
        "debt": debt,
    }
    if extra:
        manifest.update(extra)
    return manifest


def build_frontend_snapshot_manifest(**kwargs: Any) -> dict[str, Any]:
    """Deprecated import compatibility for external callers.

    Removal condition: delete once an external-import census confirms no
    consumer imports this historical function name.  It is not used by the
    repository and does not preserve the retired frontend/report runtime.
    """
    warnings.warn(
        "build_frontend_snapshot_manifest is deprecated; use build_public_bundle_manifest",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.pop("reports", None)
    return build_public_bundle_manifest(**kwargs)
