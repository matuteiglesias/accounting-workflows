from __future__ import annotations

"""Compose the private viewer bundle from finished report bundles.

This module only joins document-discovery metadata and copies finished files. It
does not inspect or reinterpret accounting data.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from accounting.reports.catalog import (
    ReportCatalogItem,
    build_report_catalog,
    validate_catalog_files,
    write_report_catalog,
)
from accounting.reports.common import ensure_relative_bundle_path


def _load_catalog(root: Path) -> dict[str, Any]:
    path = root / "report_catalog.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing report catalog: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_catalog_files(catalog: dict[str, Any], source_root: Path, target_root: Path) -> None:
    for report in catalog["reports"]:
        paths = [report["html"], report["pdf"]]
        paths.extend(item["path"] for item in report.get("attachments", []))
        for value in paths:
            relative = ensure_relative_bundle_path(str(value))
            source = (source_root / relative).resolve(strict=True)
            if source_root.resolve() not in source.parents:
                raise ValueError(f"catalog path escapes source bundle: {relative}")
            destination = target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError(f"conflicting private bundle path: {relative}")
            if not destination.exists():
                shutil.copy2(source, destination)


def _items(catalog: dict[str, Any]) -> Iterable[ReportCatalogItem]:
    for report in catalog["reports"]:
        yield ReportCatalogItem(
            report_id=str(report["report_id"]),
            title=str(report["title"]),
            description=str(report["description"]),
            period_label=str(report["period_label"]),
            sort_order=int(report["sort_order"]),
            html=str(report["html"]),
            pdf=str(report["pdf"]) if report.get("pdf") else None,
            manifest=str(report["manifest"]) if report.get("manifest") else None,
            attachments=tuple(report.get("attachments", [])),
        )


def compose_private_report_bundle(
    *,
    normal_reports_root: Path,
    evidence_reports_root: Path,
    out_root: Path,
) -> Path:
    """Create one private catalog/bundle from ordinary and evidence reports."""

    normal_reports_root = Path(normal_reports_root).resolve(strict=True)
    evidence_reports_root = Path(evidence_reports_root).resolve(strict=True)
    out_root = Path(out_root).resolve()
    if out_root == normal_reports_root or out_root == evidence_reports_root:
        raise ValueError("private bundle output must differ from both source bundles")

    normal_catalog = _load_catalog(normal_reports_root)
    evidence_catalog = _load_catalog(evidence_reports_root)
    validate_catalog_files(normal_catalog, bundle_root=normal_reports_root)
    validate_catalog_files(evidence_catalog, bundle_root=evidence_reports_root)
    normal_items = list(_items(normal_catalog))
    evidence_items = list(_items(evidence_catalog))
    next_sort_order = max((item.sort_order for item in normal_items), default=0) + 1
    # Evidence reports are appended after the ordinary report surface while
    # retaining deterministic ordering if more private reports are added.
    evidence_items = [
        ReportCatalogItem(
            report_id=item.report_id,
            title=item.title,
            description=item.description,
            period_label=item.period_label,
            sort_order=next_sort_order + index,
            html=item.html,
            pdf=item.pdf,
            manifest=item.manifest,
            attachments=item.attachments,
        )
        for index, item in enumerate(evidence_items)
    ]
    reports = [*normal_items, *evidence_items]
    combined = build_report_catalog(
        source_run_id=str(normal_catalog["source_run_id"]),
        scope_tag=str(normal_catalog["scope_tag"]),
        as_of_date=str(normal_catalog["as_of_date"]),
        generated_at_utc=max(
            str(normal_catalog.get("generated_at_utc", "")),
            str(evidence_catalog.get("generated_at_utc", "")),
        ),
        reports=reports,
    )

    out_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_root.name}.", dir=out_root.parent) as staging:
        staged = Path(staging)
        _copy_catalog_files(normal_catalog, normal_reports_root, staged)
        _copy_catalog_files(evidence_catalog, evidence_reports_root, staged)
        write_report_catalog(staged / "report_catalog.json", combined)
        validate_catalog_files(combined, bundle_root=staged)
        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.copytree(staged, out_root)
    return out_root
