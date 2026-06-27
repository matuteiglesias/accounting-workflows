from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from accounting.publish.manifest import build_frontend_snapshot_manifest


@dataclass(frozen=True)
class PublishPaths:
    project_root: Path
    out_root: Path
    public_root: Path
    human_latest: Path
    metrics_latest: Path
    debt_latest: Path


REPORT_DIRNAME = "balance_human_v2"
DEFAULT_PUBLIC_SUBDIR = Path("public") / "accounting" / "latest"


# Keep this intentionally small and dumb.
# The UI should depend on a stable, minimal bundle, not on the full producer tree.
METRIC_FILES = [
    "build_manifest.json",
    "balance_cash_y.csv",
    "income_statement_y.csv",
    "metric_views/income_statement_monthly_last6.csv",
    "metric_views/rent_rollup_by_place_m_last6.csv",
    "metric_views/flow_type_rollup_m_last6.csv",
    "metric_views/draws_discipline_monthly_last6.csv",
    "metric_contract_frontier.csv",
    "frontend_metric_series.csv",
    "metrics_frontier_qa.csv",
]

DEBT_FILES = [
    "debt_open_items.csv",
    "debt_repayment_events.csv",
    "debt_status_reconciliation.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the current producer latest artifacts into out/public for a thin UI. "
            "No computation, just packaging."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root. Defaults to the repository root inferred from this module.",
    )
    parser.add_argument(
        "--public-subdir",
        type=Path,
        default=DEFAULT_PUBLIC_SUBDIR,
        help="Path under project root where the public bundle will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Use copy for deployable bundles or symlink for fast local iteration.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the target public directory before publishing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned snapshot inputs/outputs without requiring sources or copying files.",
    )
    return parser.parse_args()


def resolve_paths(project_root: Path, public_subdir: Path, *, strict: bool = True) -> PublishPaths:
    out_root = project_root / "out"
    human_latest = (out_root / "human_reports" / "latest").resolve(strict=strict)
    metrics_latest = (out_root / "metrics" / "latest").resolve(strict=strict)
    debt_latest = (out_root / "debt_resolution" / "latest").resolve(strict=strict)
    public_root = project_root / public_subdir
    return PublishPaths(
        project_root=project_root,
        out_root=out_root,
        public_root=public_root,
        human_latest=human_latest,
        metrics_latest=metrics_latest,
        debt_latest=debt_latest,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_symlink(src: Path, dst: Path, mode: str) -> None:
    remove_path(dst)
    ensure_dir(dst.parent)
    if mode == "symlink":
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def publish_report(paths: PublishPaths, mode: str) -> dict[str, Any]:
    src = paths.human_latest / REPORT_DIRNAME
    if not src.exists():
        raise FileNotFoundError(f"Expected report directory not found: {src}")

    dst = paths.public_root / "report"
    copy_or_symlink(src, dst, mode)

    story_manifest = read_json(src / "story_manifest.json")
    return {
        "title": "Balance humano v2",
        "source_dir": relative_to_project(src, paths.project_root),
        "entry_html": "report/balance_humano_v2.html",
        "story_manifest": "report/story_manifest.json",
        "items": [
            {
                "item_id": item.get("item_id"),
                "slug": item.get("slug"),
                "title": item.get("title"),
                "kind": item.get("kind"),
            }
            for item in story_manifest.get("items", [])
        ],
    }


def publish_selected_files(src_root: Path, dst_root: Path, rel_paths: list[str], mode: str) -> list[str]:
    published: list[str] = []
    for rel in rel_paths:
        src = src_root / rel
        if not src.exists():
            continue
        dst = dst_root / rel
        copy_or_symlink(src, dst, mode)
        published.append(str(Path(dst_root.name) / rel))
    return published


def publish_metrics(paths: PublishPaths, mode: str) -> dict[str, Any]:
    dst_root = paths.public_root / "metrics"
    published = publish_selected_files(paths.metrics_latest, dst_root, METRIC_FILES, mode)

    build_manifest_path = paths.metrics_latest / "build_manifest.json"
    build_manifest = read_json(build_manifest_path) if build_manifest_path.exists() else {}

    return {
        "source_dir": relative_to_project(paths.metrics_latest, paths.project_root),
        "build_manifest": "metrics/build_manifest.json" if build_manifest_path.exists() else None,
        "published_files": published,
        "run_id": build_manifest.get("run_id"),
        "as_of_date": build_manifest.get("as_of_date"),
    }


def publish_debt(paths: PublishPaths, mode: str) -> dict[str, Any]:
    dst_root = paths.public_root / "debt"
    published = publish_selected_files(paths.debt_latest, dst_root, DEBT_FILES, mode)
    return {
        "source_dir": relative_to_project(paths.debt_latest, paths.project_root),
        "published_files": published,
        "primary_tables": {
            "open_items": "debt/debt_open_items.csv",
            "repayment_events": "debt/debt_repayment_events.csv",
            "reconciliation": "debt/debt_status_reconciliation.csv",
        },
    }


def _snapshot_file_list(report_info: dict[str, Any], metrics_info: dict[str, Any], debt_info: dict[str, Any]) -> list[str]:
    files = [
        "manifest.json",
        report_info.get("entry_html"),
        report_info.get("story_manifest"),
        metrics_info.get("build_manifest"),
        *metrics_info.get("published_files", []),
        *debt_info.get("published_files", []),
    ]
    return sorted({str(x) for x in files if x})


def build_surface_manifest(paths: PublishPaths, report_info: dict[str, Any], metrics_info: dict[str, Any], debt_info: dict[str, Any], mode: str) -> dict[str, Any]:
    story_manifest_path = paths.public_root / "report" / "story_manifest.json"
    story_manifest = read_json(story_manifest_path)

    build_manifest_path = paths.public_root / "metrics" / "build_manifest.json"
    build_manifest = read_json(build_manifest_path) if build_manifest_path.exists() else {}
    source_run_id = build_manifest.get("run_id") or story_manifest.get("run_root", "").split("/")[-1] or None

    return build_frontend_snapshot_manifest(
        source_run_id=source_run_id,
        status="ok",
        source_paths={
            "human_latest": relative_to_project(paths.human_latest, paths.project_root),
            "metrics_latest": relative_to_project(paths.metrics_latest, paths.project_root),
            "debt_latest": relative_to_project(paths.debt_latest, paths.project_root),
            "public_root": relative_to_project(paths.public_root, paths.project_root),
        },
        files=_snapshot_file_list(report_info, metrics_info, debt_info),
        metrics=metrics_info,
        debt=debt_info,
        reports={"balance_human_v2": report_info},
        extra={
            "surface_id": "accounting_surface",
            "published_at_utc": None,  # compatibility key; prefer built_at
            "publish_mode": mode,
            "run_id": source_run_id,
            "as_of_date": build_manifest.get("as_of_date"),
            "months": story_manifest.get("months"),
            "include_statuses": story_manifest.get("include_statuses", []),
            "report": report_info,  # compatibility key; prefer reports.balance_human_v2
            "navigation": [
                {"id": "home", "title": "Inicio", "path": "/"},
                {"id": "report", "title": "Reporte", "path": "/report"},
                {"id": "debt", "title": "Deudas", "path": "/debt"},
            ],
        },
    )


def build_dry_run_manifest(paths: PublishPaths, mode: str) -> dict[str, Any]:
    return build_frontend_snapshot_manifest(
        source_run_id=None,
        status="dry_run",
        source_paths={
            "human_latest": str(paths.human_latest),
            "metrics_latest": str(paths.metrics_latest),
            "debt_latest": str(paths.debt_latest),
            "public_root": str(paths.public_root),
        },
        files=[],
        metrics={},
        debt={},
        reports={},
        extra={"publish_mode": mode},
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths(
        project_root=args.project_root,
        public_subdir=args.public_subdir,
        strict=not args.dry_run,
    )

    if args.dry_run:
        print(json.dumps(build_dry_run_manifest(paths, args.mode), indent=2, ensure_ascii=False))
        return

    if args.clean and paths.public_root.exists():
        shutil.rmtree(paths.public_root)

    ensure_dir(paths.public_root)

    report_info = publish_report(paths, args.mode)
    metrics_info = publish_metrics(paths, args.mode)
    debt_info = publish_debt(paths, args.mode)

    manifest = build_surface_manifest(paths, report_info, metrics_info, debt_info, args.mode)
    manifest["published_at_utc"] = manifest["built_at"]
    write_json(paths.public_root / "manifest.json", manifest)

    print(f"Published accounting surface bundle to: {paths.public_root}")
    print(f"  report -> {paths.public_root / 'report'}")
    print(f"  metrics -> {paths.public_root / 'metrics'}")
    print(f"  debt -> {paths.public_root / 'debt'}")
    print(f"  manifest -> {paths.public_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
