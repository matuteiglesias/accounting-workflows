from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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

    def normalized(self) -> "ReportCatalogItem":
        if not self.report_id.strip() or not self.title.strip():
            raise ValueError("report_id and title are required")
        return ReportCatalogItem(
            report_id=self.report_id,
            title=self.title,
            description=self.description,
            period_label=self.period_label,
            sort_order=int(self.sort_order),
            html=ensure_relative_bundle_path(self.html),
            pdf=ensure_relative_bundle_path(self.pdf) if self.pdf else None,
            manifest=ensure_relative_bundle_path(self.manifest) if self.manifest else None,
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


def write_report_catalog(path: str | Path, payload: dict[str, Any]) -> None:
    if payload.get("schema") != REPORT_CATALOG_SCHEMA:
        raise ValueError("invalid report catalog schema")
    atomic_write_json(path, payload)
