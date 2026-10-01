from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import re

from accounting.reports import REPORT_CATALOG_SCHEMA
from accounting.reports.common import atomic_write_json, ensure_relative_bundle_path


@dataclass(frozen=True)
class ReportCatalogItem:
    report_id: str
    title: str
    description: str
    period_label: str
    sort_order: int
    html: str
    pdf: str | None = None
    manifest: str | None = None
    attachments: tuple[dict[str, str], ...] = ()

    def normalized(self) -> "ReportCatalogItem":
        if not self.report_id.strip() or not self.title.strip():
            raise ValueError("report_id and title are required")
        attachments = []
        for attachment in self.attachments:
            if not isinstance(attachment, dict) or set(attachment) != {"path", "sha256"}:
                raise ValueError("report attachments must contain only path and sha256")
            path = ensure_relative_bundle_path(str(attachment["path"]))
            sha256 = str(attachment["sha256"]).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ValueError(f"invalid attachment sha256: {sha256!r}")
            attachments.append({"path": path, "sha256": sha256})
        if len({item["path"] for item in attachments}) != len(attachments):
            raise ValueError("duplicate report attachment paths")
        return ReportCatalogItem(
            report_id=self.report_id,
            title=self.title,
            description=self.description,
            period_label=self.period_label,
            sort_order=int(self.sort_order),
            html=ensure_relative_bundle_path(self.html),
            pdf=ensure_relative_bundle_path(self.pdf) if self.pdf else None,
            manifest=ensure_relative_bundle_path(self.manifest) if self.manifest else None,
            attachments=tuple(attachments),
        )


def build_report_catalog(
    *,
    source_run_id: str,
    scope_tag: str,
    as_of_date: str,
    generated_at_utc: str,
    reports: Iterable[ReportCatalogItem],
) -> dict[str, Any]:
    normalized = [item.normalized() for item in reports]
    ids = [item.report_id for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate report_id values: {ids}")
    normalized.sort(key=lambda item: (item.sort_order, item.report_id))
    return {
        "schema": REPORT_CATALOG_SCHEMA,
        "source_run_id": source_run_id,
        "scope_tag": scope_tag,
        "as_of_date": as_of_date,
        "generated_at_utc": generated_at_utc,
        "reports": [asdict(item) for item in normalized],
    }


def validate_catalog_files(catalog: dict[str, Any], *, bundle_root: str | Path) -> None:
    if catalog.get("schema") != REPORT_CATALOG_SCHEMA:
        raise ValueError("invalid report catalog schema")
    root = Path(bundle_root)
    for report in catalog.get("reports", []):
        for key in ("html", "pdf", "manifest"):
            rel = report.get(key)
            if rel and not (root / ensure_relative_bundle_path(rel)).is_file():
                raise FileNotFoundError(f"catalog {key} does not exist: {rel}")
        attachments = report.get("attachments", [])
        if not isinstance(attachments, (list, tuple)):
            raise ValueError(f"catalog attachments must be a list: {report.get('report_id')}")
        seen: set[str] = set()
        for attachment in attachments:
            if not isinstance(attachment, dict) or set(attachment) != {"path", "sha256"}:
                raise ValueError(f"invalid attachment allowlist entry: {attachment!r}")
            rel = ensure_relative_bundle_path(str(attachment["path"]))
            if rel in seen:
                raise ValueError(f"duplicate attachment path: {rel}")
            seen.add(rel)
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(attachment["sha256"])):
                raise ValueError(f"invalid attachment sha256: {attachment.get('sha256')!r}")
            if not (root / rel).is_file():
                raise FileNotFoundError(f"catalog attachment does not exist: {rel}")


def write_report_catalog(path: str | Path, payload: dict[str, Any]) -> None:
    if payload.get("schema") != REPORT_CATALOG_SCHEMA:
        raise ValueError("invalid report catalog schema")
    atomic_write_json(path, payload)
