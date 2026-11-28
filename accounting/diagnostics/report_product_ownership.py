from __future__ import annotations

"""Audit report-product ownership using existing run and report manifests.

This module does not define a new reporting catalog or execute any producer.
It reads one already-completed exact run and reports its observed dependencies.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_file(root: Path, logical_path: str) -> Path:
    rel = PurePosixPath(logical_path)
    if not logical_path or rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError(f"invalid artifact path: {logical_path!r}")
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / Path(*rel.parts)).resolve(strict=True)
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise ValueError(f"artifact escapes its run root: {logical_path!r}")
    return path


def _verify(path: Path, sha256: str) -> None:
    if not sha256:
        raise ValueError(f"missing checksum: {path}")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != sha256:
        raise ValueError(f"checksum mismatch for {path}")


def _produced_artifacts(run_root: Path, metrics_root: Path) -> dict[str, str]:
    """Derive artifact ownership from existing stage/metrics manifests."""
    producers: dict[str, str] = {}
    evidence = run_root / "meta" / "artifacts.jsonl"
    for line in evidence.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("role") not in ("output", "derived"):
            continue
        logical = f"run/{record['relpath']}"
        producer = str(record["stage"])
        previous = producers.setdefault(logical, producer)
        if previous != producer:
            raise ValueError(f"multiple producers for {logical}: {previous} and {producer}")

    metrics = _read_json(metrics_root / "build_manifest.json")
    if metrics.get("run_id") != run_root.name:
        raise ValueError("metrics build manifest belongs to a different run")
    for group in ("frontier_outputs", "annual_dashboard_outputs", "source_contract_outputs"):
        for artifact_path in metrics.get(group, {}).values():
            logical = f"metrics/{Path(artifact_path).name}"
            previous = producers.setdefault(logical, "accounting.metrics.build")
            if previous != "accounting.metrics.build":
                raise ValueError(f"multiple producers for {logical}")
    return producers


def audit_report_products(
    run_root: Path, metrics_root: Path, reports_root: Path, *, verify_hashes: bool = True,
) -> dict:
    run_root, metrics_root, reports_root = map(Path, (run_root, metrics_root, reports_root))
    run_id = run_root.name
    if run_id in ("latest", "latest_FBPM", "latest_HH") or run_id.startswith("latest_"):
        raise ValueError("audit requires an immutable exact run, not a latest symlink")
    if len({run_root.name, metrics_root.name, reports_root.name}) != 1:
        raise ValueError("run, metrics and reports must belong to one exact run")

    catalog = _read_json(reports_root / "report_catalog.json")
    if catalog["source_run_id"] != run_id:
        raise ValueError("report catalog identifies a different run")
    producers = _produced_artifacts(run_root, metrics_root)
    owners: dict[str, str] = {}
    input_consumers: dict[str, set[str]] = defaultdict(set)
    products = []
    for report in catalog["reports"]:
        report_id = report["report_id"]
        if not report_id or any(ch in report_id for ch in "/\\."):
            raise ValueError(f"invalid report_id {report_id!r}")
        if any(item["report_id"] == report_id for item in products):
            raise ValueError(f"duplicate report ID: {report_id}")
        html = _safe_file(reports_root, report["html"])
        manifest_path = html.parent / "report_manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("report_id") != report_id or manifest.get("source_run_id") != run_id:
            raise ValueError(f"manifest identity mismatch: {report_id}")
        if manifest.get("scope_tag") != catalog.get("scope_tag"):
            raise ValueError(f"manifest scope mismatch: {report_id}")
        if manifest.get("as_of_date") != catalog.get("as_of_date"):
            raise ValueError(f"manifest cutoff mismatch: {report_id}")
        if manifest.get("validation_status") == "fail":
            raise ValueError(f"failed product is cataloged: {report_id}")
        if not manifest.get("renderer_version"):
            raise ValueError(f"missing renderer owner: {report_id}")

        sources = []
        for source in manifest["sources"]:
            logical = source["path"]
            prefix, sep, remainder = logical.partition("/")
            if not sep or prefix not in ("run", "metrics", "reports"):
                raise ValueError(f"unknown input namespace: {logical}")
            if prefix == "reports" and not (report_id == "treasury_accountability" and remainder.startswith("treasury_accountability/projection/")):
                raise ValueError(f"unapproved report-local input: {logical}")
            filepath = _safe_file({"run": run_root, "metrics": metrics_root, "reports": reports_root}[prefix], remainder)
            if verify_hashes:
                _verify(filepath, source["sha256"])
            input_consumers[logical].add(report_id)
            sources.append(logical)

        outputs = []
        for item in manifest["outputs"].values():
            logical = item["path"]
            filepath = _safe_file(reports_root, logical)
            if verify_hashes:
                _verify(filepath, item["sha256"])
            if logical in owners:
                raise ValueError(f"report output has two owners: {logical}")
            owners[logical] = report_id
            outputs.append(logical)
        if report["html"] not in outputs:
            raise ValueError(f"HTML output missing from report manifest: {report_id}")
        if report.get("pdf") and report["pdf"] not in outputs:
            raise ValueError(f"PDF output missing from report manifest: {report_id}")
        products.append({
            "report_id": report_id,
            "owner": manifest["renderer_version"],
            "inputs": sorted(set(sources)),
            "outputs": sorted(outputs),
            "validation_status": manifest["validation_status"],
        })

    inputs = [{
        "artifact": logical,
        "producer": producers.get(logical),
        "consumers": sorted(consumers),
        "owner_declared": logical in producers,
    } for logical, consumers in sorted(input_consumers.items())]
    unattributed = [item["artifact"] for item in inputs if not item["owner_declared"]]
    return {
        "run_id": run_id,
        "source": "existing report catalog, report manifests, run stage artifact log, metrics build manifest",
        "report_count": len(products),
        "source_count": len(inputs),
        "output_count": len(owners),
        "source_owner_gaps": unattributed,
        "products": products,
        "inputs": inputs,
        "note": (
            "An unattributed source is not proven producerless; stage provenance may be "
            "missing. Unused-by-report artifacts are not necessarily unused by other consumers."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an existing exact-run report bundle; no builds or writes upstream.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--reports-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON inventory outside the audited run roots.")
    args = parser.parse_args()
    inventory = audit_report_products(args.run_root, args.metrics_root, args.reports_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{inventory['run_id']}: {inventory['report_count']} reports, "
          f"{inventory['source_count']} input artifacts, "
          f"{len(inventory['source_owner_gaps'])} source-owner gaps")
    for artifact in inventory["source_owner_gaps"]:
        print(f"  OWNER_UNDECLARED: {artifact}")


if __name__ == "__main__":
    main()
