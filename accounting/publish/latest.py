from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from accounting.artifacts.manifest import artifact_contract_for_name, write_artifact_contracts_csv
from accounting.publish.manifest import build_public_bundle_manifest
from accounting.support.latest import PRIMARY_SCOPE_TAG, update_primary_compatibility_latest


@dataclass(frozen=True)
class PublishPaths:
    project_root: Path
    out_root: Path
    public_root: Path
    metrics_latest: Path
    debt_latest: Path


DEFAULT_PUBLIC_SUBDIR = Path("public") / "accounting"

# Publication is a small artifact handoff, not a second reporting engine.
METRIC_FILES_BY_CLASS = {
    "public_contract": [
        "annual_balance_dashboard_contract.csv",
        "metric_contract_frontier.csv",
    ],
    "canonical_dashboard": [
        "annual_balance_dashboard_metrics.csv",
        "annual_balance_dashboard_qa.csv",
        "frontend_metric_series.csv",
        "monthly_operating_statement.csv",
        "monthly_flow_semantic_split.csv",
        "monthly_cash_close.csv",
        "monthly_debt_position.csv",
        "monthly_debt_activity.csv",
    ],
    "legacy_reconciliation": [
        "balance_cash_y.csv",
        "income_statement_y.csv",
    ],
    "internal_diagnostic": [
        "build_manifest.json",
        "metrics_frontier_qa.csv",
        "frontier_source_qa.csv",
    ],
}

DEBT_FILES_BY_CLASS = {
    "internal_diagnostic": [
        "debt_status_reconciliation.csv",
    ],
    "unsafe_for_frontend": [
        "debt_open_items.csv",
        "debt_repayment_events.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package current governed accounting artifacts for downstream consumers; no computation or UI runtime."
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
        help="Parent path under project root for scope-qualified public bundles.",
    )
    parser.add_argument("--scope-tag", default=PRIMARY_SCOPE_TAG)
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Use copy for deployable bundles or symlink for fast local iteration.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove the target public directory before publishing.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned bundle paths without requiring sources or copying files.",
    )
    return parser.parse_args()


def resolve_paths(
    project_root: Path,
    public_subdir: Path,
    *,
    scope_tag: str = PRIMARY_SCOPE_TAG,
    strict: bool = True,
) -> PublishPaths:
    out_root = project_root / "out"
    metrics_latest = (out_root / "metrics" / f"latest_{scope_tag}").resolve(strict=strict)
    debt_latest = (out_root / "debt_resolution" / f"latest_{scope_tag}").resolve(strict=strict)
    public_root = project_root / public_subdir / f"latest_{scope_tag}"
    identities = {path.name for path in [metrics_latest, debt_latest]}
    if strict and len(identities) != 1:
        raise ValueError(f"Publish inputs mix accounting runs for {scope_tag}: {sorted(identities)}")
    return PublishPaths(
        project_root=project_root,
        out_root=out_root,
        public_root=public_root,
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


def _published_contract_row(public_relpath: str) -> dict[str, Any]:
    public_rel = Path(public_relpath)
    contract = artifact_contract_for_name(public_rel.name, str(public_rel))
    if contract["artifact_role"] == "canonical_source" and contract["source_authority"] in {
        "frontend_contract",
        "source_of_truth",
        "source_of_truth_for_debt_stock",
        "source_of_truth_for_cash_only_when_is_frontend_safe_true",
    }:
        publish_class = "public_contract"
    elif contract["artifact_role"] == "presentation_only":
        publish_class = "presentation"
    elif contract["artifact_role"] == "legacy":
        publish_class = "legacy_reconciliation"
    elif contract["frontend_suitability"] in {"forbidden", "internal_only"}:
        publish_class = "unsafe_for_frontend" if contract["frontend_suitability"] == "forbidden" else "internal_diagnostic"
    else:
        publish_class = "internal_diagnostic"
    return {"name": public_rel.name, "relpath": str(public_rel), "publish_class": publish_class, **contract}


def write_publish_artifact_contracts(paths: PublishPaths, files: list[str]) -> str:
    rows = [_published_contract_row(rel) for rel in files if rel != "manifest.json"]
    out = paths.public_root / "artifact_contracts.csv"
    write_artifact_contracts_csv(out, rows)
    return relative_to_project(out, paths.project_root)


def _publish_classified_files(
    src_root: Path,
    public_root: Path,
    files_by_class: dict[str, list[str]],
    mode: str,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for publish_class, rels in files_by_class.items():
        out[publish_class] = publish_selected_files(src_root, public_root / publish_class, rels, mode)
    return out


def _flatten_published(published_by_class: dict[str, list[str]]) -> list[str]:
    return [rel for rels in published_by_class.values() for rel in rels]


def publish_metrics(paths: PublishPaths, mode: str) -> dict[str, Any]:
    published_by_class = _publish_classified_files(paths.metrics_latest, paths.public_root, METRIC_FILES_BY_CLASS, mode)
    build_manifest_path = paths.metrics_latest / "build_manifest.json"
    build_manifest = read_json(build_manifest_path) if build_manifest_path.exists() else {}
    return {
        "source_dir": relative_to_project(paths.metrics_latest, paths.project_root),
        "build_manifest": "internal_diagnostic/build_manifest.json" if build_manifest_path.exists() else None,
        "published_files": _flatten_published(published_by_class),
        "published_by_class": published_by_class,
        "run_id": build_manifest.get("run_id"),
        "as_of_date": build_manifest.get("as_of_date"),
    }


def publish_debt(paths: PublishPaths, mode: str) -> dict[str, Any]:
    published_by_class = _publish_classified_files(paths.debt_latest, paths.public_root, DEBT_FILES_BY_CLASS, mode)
    return {
        "source_dir": relative_to_project(paths.debt_latest, paths.project_root),
        "published_files": _flatten_published(published_by_class),
        "published_by_class": published_by_class,
        "primary_tables": {
            "stock_contract": "canonical_dashboard/monthly_debt_position.csv",
            "activity_contract": "canonical_dashboard/monthly_debt_activity.csv",
            "raw_open_items_diagnostic": "unsafe_for_frontend/debt_open_items.csv",
            "raw_repayment_events_diagnostic": "unsafe_for_frontend/debt_repayment_events.csv",
        },
    }


def _bundle_file_list(metrics_info: dict[str, Any], debt_info: dict[str, Any]) -> list[str]:
    files = [
        "manifest.json",
        metrics_info.get("build_manifest"),
        *metrics_info.get("published_files", []),
        *debt_info.get("published_files", []),
    ]
    return sorted({str(x) for x in files if x})


def build_publish_contract_qa(paths: PublishPaths, files: list[str]) -> str:
    rows: list[dict[str, str]] = []
    classes = {Path(f).parts[0] for f in files if Path(f).parts}

    def add(check: str, ok: bool, detail: str, severity: str = "error") -> None:
        rows.append({"check": check, "status": "pass" if ok else "fail", "detail": detail, "severity": severity})

    expected = {
        "public_contract",
        "canonical_dashboard",
        "legacy_reconciliation",
        "internal_diagnostic",
        "unsafe_for_frontend",
    }
    add(
        "publish_bundle_labels_all_artifacts",
        all(
            (Path(f).parts and Path(f).parts[0] in expected)
            or f in {"manifest.json", "artifact_contracts.csv", "publish_contract_qa.csv"}
            for f in files
        ),
        f"classes={sorted(classes)}",
    )
    add(
        "no_unsafe_artifacts_in_public_contract",
        not any(
            f.startswith("public_contract/")
            and Path(f).name in {"debt_open_items.csv", "debt_repayment_events.csv"}
            for f in files
        ),
        "raw debt files are diagnostic/internal only",
    )
    add(
        "legacy_artifacts_labeled_legacy",
        not any(
            Path(f).name in {"income_statement_y.csv", "balance_cash_y.csv"}
            and not f.startswith("legacy_reconciliation/")
            for f in files
        ),
        "legacy annual views under legacy_reconciliation",
    )
    stock_activity_contracts = {"debt_stock", "debt_activity"}
    stock_activity_present = stock_activity_contracts.intersection(
        {artifact_contract_for_name(Path(f).name, f).get("artifact_role") for f in files}
    )
    add(
        "debt_stock_activity_separated",
        stock_activity_present == stock_activity_contracts,
        "debt stock/activity contracts are properly separated in the contract/frontier",
    )

    out = paths.public_root / "publish_contract_qa.csv"
    ensure_dir(out.parent)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["check", "status", "detail", "severity"])
        writer.writeheader()
        writer.writerows(rows)
    return relative_to_project(out, paths.project_root)


def build_bundle_manifest(
    paths: PublishPaths,
    metrics_info: dict[str, Any],
    debt_info: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    build_manifest_path = paths.public_root / "internal_diagnostic" / "build_manifest.json"
    build_manifest = read_json(build_manifest_path) if build_manifest_path.exists() else {}
    source_run_id = build_manifest.get("run_id") or metrics_info.get("run_id")
    files = _bundle_file_list(metrics_info, debt_info)
    write_publish_artifact_contracts(paths, files)
    build_publish_contract_qa(paths, [*files, "artifact_contracts.csv"])
    return build_public_bundle_manifest(
        source_run_id=source_run_id,
        status="ok",
        source_paths={
            "metrics_latest": relative_to_project(paths.metrics_latest, paths.project_root),
            "debt_latest": relative_to_project(paths.debt_latest, paths.project_root),
            "public_root": relative_to_project(paths.public_root, paths.project_root),
        },
        files=sorted(set([*files, "artifact_contracts.csv", "publish_contract_qa.csv"])),
        metrics=metrics_info,
        debt=debt_info,
        extra={
            "surface_id": "accounting_public_bundle",
            "publish_mode": mode,
            "run_id": source_run_id,
            "as_of_date": build_manifest.get("as_of_date") or metrics_info.get("as_of_date"),
            "artifact_contracts": "artifact_contracts.csv",
            "publish_contract_qa": "publish_contract_qa.csv",
            "publish_contract_summary": {
                "public_contract": "metric frontier/series and explicitly safe contracts",
                "canonical_dashboard": "canonical/report-safe metrics and monthly contracts",
                "legacy_reconciliation": "kept for reconciliation; not source of truth",
                "internal_diagnostic": "internal evidence only",
                "unsafe_for_frontend": "must not be displayed as dashboard fact",
            },
        },
    )


def build_dry_run_manifest(paths: PublishPaths, mode: str) -> dict[str, Any]:
    return build_public_bundle_manifest(
        source_run_id=None,
        status="dry_run",
        source_paths={
            "metrics_latest": str(paths.metrics_latest),
            "debt_latest": str(paths.debt_latest),
            "public_root": str(paths.public_root),
        },
        files=[],
        metrics={},
        debt={},
        extra={"publish_mode": mode},
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths(
        project_root=args.project_root,
        public_subdir=args.public_subdir,
        scope_tag=args.scope_tag,
        strict=not args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(build_dry_run_manifest(paths, args.mode), indent=2, ensure_ascii=False))
        return

    if args.clean and paths.public_root.exists():
        shutil.rmtree(paths.public_root)
    ensure_dir(paths.public_root)

    metrics_info = publish_metrics(paths, args.mode)
    debt_info = publish_debt(paths, args.mode)
    manifest = build_bundle_manifest(paths, metrics_info, debt_info, args.mode)
    manifest["published_at_utc"] = manifest["built_at"]
    write_json(paths.public_root / "manifest.json", manifest)
    update_primary_compatibility_latest(paths.public_root.parent, paths.public_root.name, args.scope_tag)

    print(f"Published accounting artifact bundle to: {paths.public_root}")
    print(f"  public_contract -> {paths.public_root / 'public_contract'}")
    print(f"  canonical_dashboard -> {paths.public_root / 'canonical_dashboard'}")
    print(f"  manifest -> {paths.public_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
