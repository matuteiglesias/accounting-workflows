from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from accounting.reports import REPORT_MANIFEST_SCHEMA
from accounting.reports.common import atomic_write_json, ensure_relative_bundle_path, sha256_file


@dataclass(frozen=True)
class ReportSource:
    name: str
    path: str
    sha256: str
    rows: int | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        rows: int | None = None,
        logical_path: str | None = None,
    ) -> "ReportSource":
        source = Path(path)
        return cls(
            name=source.name,
            path=logical_path or source.name,
            sha256=sha256_file(source),
            rows=rows,
        )


@dataclass(frozen=True)
class ReportOutput:
    path: str
    sha256: str

    @classmethod
    def from_file(cls, path: str | Path, *, bundle_root: str | Path) -> "ReportOutput":
        file_path = Path(path)
        rel = ensure_relative_bundle_path(
            str(file_path.resolve().relative_to(Path(bundle_root).resolve()))
        )
        return cls(path=rel, sha256=sha256_file(file_path))


def build_report_manifest(
    *,
    report_id: str,
    renderer_version: str,
    source_run_id: str,
    scope_tag: str,
    as_of_date: str,
    sources: Iterable[ReportSource],
    outputs: dict[str, ReportOutput],
    validation_status: str,
) -> dict[str, Any]:
    if not report_id.strip():
        raise ValueError("report_id is required")
    if validation_status not in {"pass", "warn", "fail"}:
        raise ValueError(f"unsupported validation_status: {validation_status}")
    if not outputs:
        raise ValueError("a report manifest requires at least one output")
    return {
        "schema": REPORT_MANIFEST_SCHEMA,
        "report_id": report_id,
        "renderer_version": renderer_version,
        "source_run_id": source_run_id,
        "scope_tag": scope_tag,
        "as_of_date": as_of_date,
        "sources": [asdict(item) for item in sources],
        "outputs": {name: asdict(item) for name, item in sorted(outputs.items())},
        "validation_status": validation_status,
    }


def write_report_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    if payload.get("schema") != REPORT_MANIFEST_SCHEMA:
        raise ValueError("invalid report manifest schema")
    atomic_write_json(path, payload)
