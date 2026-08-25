from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from accounting.reports.catalog import validate_catalog_files
from accounting.reports.common import atomic_write_json, ensure_relative_bundle_path


DEFAULT_PUBLIC_SUBDIR = Path("public") / "reports"


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Strip internal provenance references from the viewer-facing catalog."""

    public_catalog = deepcopy(catalog)
    for report in public_catalog.get("reports", []):
        report["manifest"] = None
    return public_catalog


def _public_files(catalog: dict[str, Any]) -> list[str]:
    """Return only finished human documents referenced by the catalog."""

    files: list[str] = []
    for report in catalog.get("reports", []):
        for key in ("html", "pdf"):
            value = report.get(key)
            if value:
                files.append(ensure_relative_bundle_path(str(value)))
    return sorted(set(files))


def publish_report_bundle(
    *,
    project_root: Path,
    scope_tag: str,
    reports_root: Path | None = None,
    public_subdir: Path = DEFAULT_PUBLIC_SUBDIR,
    clean: bool = True,
    dry_run: bool = False,
) -> Path:
    project_root = Path(project_root).resolve()
    source_root = (
        Path(reports_root).resolve(strict=True)
        if reports_root is not None
        else (project_root / "out" / "reports" / f"latest_{scope_tag}").resolve(strict=True)
    )
    catalog_path = source_root / "report_catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"missing report catalog: {catalog_path}")
    catalog = _catalog(catalog_path)
    if catalog.get("scope_tag") != scope_tag:
        raise ValueError(
            f"report catalog scope mismatch: expected={scope_tag} actual={catalog.get('scope_tag')}"
        )
    validate_catalog_files(catalog, bundle_root=source_root)
    missing_pdf = [
        report.get("report_id", "")
        for report in catalog.get("reports", [])
        if not report.get("pdf")
    ]
    if missing_pdf:
        raise ValueError(
            "public report publication requires PDF for every report: "
            f"missing={missing_pdf}"
        )

    target_root = project_root / public_subdir / f"latest_{scope_tag}"
    public_catalog = _public_catalog(catalog)
    files = _public_files(public_catalog)
    if dry_run:
        print(f"source={source_root}")
        print(f"target={target_root}")
        print("report_catalog.json")
        for rel in files:
            print(rel)
        return target_root

    if clean:
        _remove_path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target_root / "report_catalog.json", public_catalog)
    for rel in files:
        source = source_root / rel
        if not source.is_file():
            raise FileNotFoundError(f"report publication source missing: {source}")
        destination = target_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    validate_catalog_files(_catalog(target_root / "report_catalog.json"), bundle_root=target_root)
    return target_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish finished human report documents; no accounting computation."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scope-tag", required=True)
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument("--public-subdir", type=Path, default=DEFAULT_PUBLIC_SUBDIR)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = publish_report_bundle(
        project_root=args.project_root,
        scope_tag=args.scope_tag,
        reports_root=args.reports_root,
        public_subdir=args.public_subdir,
        clean=not args.no_clean,
        dry_run=args.dry_run,
    )
    print(target)


if __name__ == "__main__":
    main()
