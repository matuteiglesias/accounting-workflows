from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def delete(rel: str) -> None:
    path = ROOT / rel
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def must_replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Phase 1 expected Makefile block not found: {label}")
    return text.replace(old, new, 1)


def must_regex(text: str, pattern: str, repl: str, label: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"Phase 1 expected one Makefile regex match for {label}; got {n}")
    return out


# ---------------------------------------------------------------------------
# 1. Delete genuinely empty / alternate architecture.
# ---------------------------------------------------------------------------
for rel in [
    "accounting/publish/snapshot.py",
    "accounting/debt/models.py",
    "accounting/debt/rules.py",
    "accounting/config.py",
    "accounting/contracts/models.py",
    "accounting/human",
    "accounting/viz",
    "notes/frontend_snapshot_contract.md",
]:
    delete(rel)


# ---------------------------------------------------------------------------
# 2. Publish remains an artifact bundle, not a human-report/frontend app.
# ---------------------------------------------------------------------------
write(
    "accounting/publish/manifest.py",
    r'''
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
    '''
)

write(
    "accounting/publish/latest.py",
    r'''
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
    '''
)

# Contract smoke must follow the supported publication surface.
check_contracts = (ROOT / "scripts/check_contracts.py").read_text(encoding="utf-8")
check_contracts = must_replace(
    check_contracts,
    "from accounting.publish.latest import METRIC_FILES_BY_CLASS, DEBT_FILES_BY_CLASS, PRESENTATION_FILES, _published_contract_row",
    "from accounting.publish.latest import METRIC_FILES_BY_CLASS, DEBT_FILES_BY_CLASS, _published_contract_row",
    "check_contracts publish imports",
)
check_contracts = must_replace(
    check_contracts,
    'publish_files = [*METRIC_FILES_BY_CLASS.get("public_contract", []), *METRIC_FILES_BY_CLASS.get("canonical_dashboard", []), *PRESENTATION_FILES, *DEBT_FILES_BY_CLASS.get("unsafe_for_frontend", [])]',
    'publish_files = [*METRIC_FILES_BY_CLASS.get("public_contract", []), *METRIC_FILES_BY_CLASS.get("canonical_dashboard", []), *DEBT_FILES_BY_CLASS.get("unsafe_for_frontend", [])]',
    "check_contracts publish file list",
)
(ROOT / "scripts/check_contracts.py").write_text(check_contracts, encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. Remove the human/front live alternative from the Makefile.
# ---------------------------------------------------------------------------
make = (ROOT / "Makefile").read_text(encoding="utf-8")
make = must_replace(
    make,
    "# Official path: run-ingest -> run-materialize -> run-marts -> run-metrics -> run-human-report",
    "# Official path: run-ingest -> run-materialize -> run-marts -> run-debt-views -> run-metrics -> run-dashboard -> publish-latest",
    "official path",
)
make = make.replace("RUN_HUMAN_DIR   := $(OUT)/human_reports/$(RUN_RUN_ID)/balance_human_v2\n", "")
make = make.replace("HUMAN_LATEST   := $(OUT)/human_reports/latest_$(SCOPE_TAG)\n", "")
make = must_replace(
    make,
    '\t\t--base "$(RUN_BASE)" --base "$(OUT)/debt_resolution" \\\n\t\t--base "$(OUT)/metrics" --base "$(OUT)/human_reports"',
    '\t\t--base "$(RUN_BASE)" --base "$(OUT)/debt_resolution" \\\n\t\t--base "$(OUT)/metrics"',
    "latest bases",
)
make = must_replace(make, '"Live canonical / metrics / dashboard / human:"', '"Live canonical / metrics / dashboard:"', "help heading")
make = make.replace('\t@echo "  make run-human          # build human report from existing metrics/run artifacts"\n', "")
make = must_replace(
    make,
    '\t@echo "  make ledger | materialize | debt | debt-views | metrics | human-report | publish | build-all"',
    '\t@echo "  make ledger | materialize | debt | debt-views | metrics | publish | build-all"',
    "help aliases 1",
)
make = must_replace(
    make,
    '\t@echo "  make run-accounting | run-accounting-full | run-human-balance | run-debt-balance"',
    '\t@echo "  make run-accounting | run-accounting-full | run-debt-balance"',
    "help aliases 2",
)
make = make.replace('\t@echo "  make front-report       # presentation-only report factory stub"\n', "")
make = must_replace(
    make,
    ".PHONY: ledger materialize debt debt-views metrics human-report publish-latest publish",
    ".PHONY: ledger materialize debt debt-views metrics publish-latest publish",
    "canonical phony",
)
make = make.replace("\nhuman-report: run-human-report\n", "")
make = must_replace(
    make,
    "# Composite names: one clear path for full builds and frontend handoff.\n.PHONY: build-all build-report build-front\nbuild-all: run-full\n\nbuild-report: human-report\n\nbuild-front: publish-latest",
    "# Composite name: one clear path for the full accounting build and publication.\n.PHONY: build-all\nbuild-all: run-full",
    "composite aliases",
)
make = must_replace(
    make,
    ".PHONY: doctor validate clean-derived front-report",
    ".PHONY: doctor validate clean-derived",
    "support phony",
)
make = must_replace(
    make,
    '\trm -rf "$(OUT)/smoke/accounting" "$(OUT)/run/accounting" "$(OUT)/metrics" "$(OUT)/human_reports" "$(OUT)/debt_resolution" "$(ROOT)/public/accounting/latest" "$(ROOT)/public/accounting/latest_$(SCOPE_TAG)"',
    '\trm -rf "$(OUT)/smoke/accounting" "$(OUT)/run/accounting" "$(OUT)/metrics" "$(OUT)/debt_resolution" "$(ROOT)/public/accounting/latest" "$(ROOT)/public/accounting/latest_$(SCOPE_TAG)"',
    "clean derived",
)
make = must_regex(make, r"\nfront-report:\n.*?(?=\n\.PHONY: smoke-core)", "\n", "front-report block")
make = must_replace(
    make,
    ".PHONY: smoke-core smoke-full smoke-usd-ccl-valuation smoke-usd-ccl-management-flows run-usd-ccl-valuation run-usd-ccl-management-flows run-canonical run-full run-dashboard run-human metrics-from-run run-metrics-live smoke-accounting run-accounting run-accounting-full run-downstream-from-ledger run-metrics-and-human run-human-balance-only",
    ".PHONY: smoke-core smoke-full smoke-usd-ccl-valuation smoke-usd-ccl-management-flows run-usd-ccl-valuation run-usd-ccl-management-flows run-canonical run-full run-dashboard metrics-from-run run-metrics-live smoke-accounting run-accounting run-accounting-full run-downstream-from-ledger",
    "main phony",
)
make = must_replace(
    make,
    'smoke-full partial: fixture core + validation + publish dry-run passed; fixture debt/human publish remains documented follow-up',
    'smoke-full partial: fixture core + validation + publish dry-run passed; fixture debt and real professional-pack execution remain documented follow-up',
    "smoke full message",
)
make = must_replace(
    make,
    "run-full: run-canonical run-debt-views run-metrics run-dashboard run-human publish-latest release-check",
    "run-full: run-canonical run-debt-views run-metrics run-dashboard publish-latest release-check",
    "run-full",
)
# Both bounded downstream helpers used to invoke the alternate human producer.
make = make.replace(
    '\t@$(MAKE) _run_human_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"\n',
    "",
)
make = must_replace(
    make,
    '\t@$(PY) -m accounting.support.latest --scope-tag "$(SCOPE_TAG)" --target "$(RUN_REL)" \\\n\t\t--base "$(RUN_BASE)" --base "$(OUT)/metrics" --base "$(OUT)/human_reports"',
    '\t@$(PY) -m accounting.support.latest --scope-tag "$(SCOPE_TAG)" --target "$(RUN_REL)" \\\n\t\t--base "$(RUN_BASE)" --base "$(OUT)/metrics"',
    "latest light bases",
)
make = make.replace(
    '\t@$(MAKE) run-human RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"\n',
    "",
)
make = must_regex(
    make,
    r"\nrun-metrics-and-human:\n.*?(?=\n# ========================================\n# SMOKE MODE)",
    "\n",
    "run metrics/human helper blocks",
)
human_target_block = '''
.PHONY: run-human-report run-human-balance _run_human_balance_action
run-human-report: run-human

run-human: _run_human_balance_action

run-dashboard: run-metrics
\t@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv"
\t@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_contract.csv"
\t@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_qa.csv"

# Compatibility alias; prefer run-human-report.
run-human-balance: run-human-report

_run_human_balance_action:
\t@$(call _guard_out_dir,$(RUN_OUT))
\t@test -s "$(RUN_METRICS_DIR)/metric_values.csv" || (echo "ERROR: missing metric_values.csv at $(RUN_METRICS_DIR)"; exit 2)
\t@mkdir -p "$(RUN_HUMAN_DIR)"
\t@bash -eu -o pipefail -c '\\\
\t\t$(PY) -m accounting.human.document \\
\t\t\t--run-root "$(RUN_OUT)" \\
\t\t\t--metrics-dir "$(RUN_METRICS_DIR)" \\
\t\t\t--write-dir "$(RUN_HUMAN_DIR)" \\
\t\t\t--months "$(METRIC_MONTHS)" \\
\t\t\t--rent-place-col "$(RENT_PLACE_COL)" \\
\t\t\t--rent-detail-col "$(RENT_DETAIL_COL)" \\
\t\t\t--flow-rollup-groupby "$(FLOW_ROLLUP_GROUPBY)" \\
\t\t\t--include-statuses "$(INCLUDE_STATUSES)" \\
\t\t\t--noise-floor "$(NOISE_FLOOR)"; \\
\t\ttest -s "$(RUN_HUMAN_DIR)/balance_humano_v2.html"; \\
\t\ttest -s "$(RUN_HUMAN_DIR)/story_manifest.json"; \\
\t'
\t@$(MAKE) _update_latest \\
\t\tRUN_STAMP="$(RUN_STAMP)" \\
\t\tRUN_OUT="$(RUN_OUT)" \\
\t\tRUN_RUN_ID="$(RUN_RUN_ID)" \\
\t\tRUN_REL="$(RUN_REL)" \\
\t\tOUT="$(OUT)" \\
\t\tRUN_BASE="$(RUN_BASE)"

\t
'''
replacement_dashboard = '''
.PHONY: run-dashboard
run-dashboard: run-metrics
\t@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv"
\t@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_contract.csv"
\t@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_qa.csv"

'''
make = must_replace(make, human_target_block, replacement_dashboard, "human target block")

for forbidden in [
    "accounting.human",
    "human_reports",
    "run-human",
    "human-report",
    "front-report",
    "build-front",
    "build-report",
    "_run_human_balance_action",
    "run-metrics-and-human",
]:
    if forbidden in make:
        raise RuntimeError(f"retired Makefile surface still present: {forbidden}")
(ROOT / "Makefile").write_text(make, encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Current docs describe the consolidated reporting/publication architecture.
# Historical dated audits are intentionally left as history.
# ---------------------------------------------------------------------------
write(
    "README.md",
    r'''
    # Accounting workflows

    Python pipeline for ledger ingestion, canonicalization, materialization, semantic marts, debt resolution, governed metrics/dashboards, professional-pack drilldowns, and artifact publication.

    ## Official command surface

    Run commands from the repository root.

    ### Fixture and validation path

    ```bash
    make smoke-core
    make smoke-full
    make validate
    ```

    - `smoke-core` exercises fixture ingest and materialization with semantic and cash checks.
    - `smoke-full` adds repository validation and a publication dry-run.
    - `validate` runs compilation, contract checks, and the regression suite without private credentials.

    ### Live canonical core

    ```bash
    make run-canonical
    ```

    `run-canonical` resolves to `run-marts`, whose dependency chain performs live ingest, materialization, and semantic-mart generation for one timestamped run.

    ### Full live and publication path

    ```bash
    make run-full
    ```

    `run-full` runs the canonical core, debt views, governed metrics, annual-dashboard assertions, artifact publication, and the release-readiness check.  The retired `accounting.human` report stack is not a live pipeline stage.

    `make run-accounting` and `make run-accounting-full` are compatibility aliases for `run-full`.

    For bounded operation on an existing run, use the focused targets exposed by `make help`, including `metrics-from-run`, `run-dashboard`, and `publish-latest`.

    ### Human-facing / professional presentation

    The repository no longer owns a standalone Flask/front application or a parallel `human_reports` producer. Human-facing work is layered over governed artifacts:

    ```bash
    make professional-drilldowns
    make professional-linked-digest
    ```

    These operate on an existing professional pack. The linked digest is presentation-only and does not recalculate accounting semantics. Notebook/report consumers should likewise read governed metric/debt artifacts rather than introducing a second accounting engine.

    ## Runbook
    See `notes/accounting_spine_runbook.md` for the per-stage outputs and smoke checklist.

    ## Publication contract
    See `notes/public_bundle_contract.md` for the consumer-safe artifact handoff.

    ## Documentation compass
    Use `notes/documentation_compass.md` as the role-based guide to current docs.

    ## Repo hygiene
    - Generated outputs are not tracked (`out/`, `accounting/out/`, etc.).
    - Local secrets are kept in `private/` and never committed.
    - Historical audits may mention retired module paths; they are evidence, not live command authority.

    ## Logging convention
    Operational Python entrypoints use `YYYY-MM-DDTHH:MM:SSZ LEVEL [stage] message`. Keep `journalctl` as the operational log source of truth and retain per-run CSV/JSON/HTML artifacts under the governed run, metrics, professional-pack, drilldown, and publication roots rather than duplicating logs into report artifacts.
    '''
)

write(
    "notes/canonical_commands.md",
    r'''
    ---
    id: notes/canonical_commands
    title: "Accounting Canonical Commands"
    sidebar_label: "Accounting Canonical Commands"
    ---

    # Accounting Canonical Commands

    Status: current authority
    Last reviewed: 2026-08-24

    The Makefile is the command authority. `make help` is the live command list.

    ## Core pipeline

    ```text
    make ledger          # source inputs -> canonical ledger
    make materialize     # canonical ledger -> materialized analytical artifacts
    make debt            # canonical evidence -> resolved debt contracts
    make debt-views      # debt contracts -> stock/activity views
    make metrics         # semantic/debt artifacts -> governed metrics
    make run-dashboard   # assert governed annual dashboard outputs
    make publish-latest  # governed metrics/debt -> published artifact bundle
    ```

    ## Composite command

    ```text
    make build-all       # full canonical path through publication + release check
    ```

    ## Professional presentation

    ```text
    make professional-drilldowns
    make professional-linked-digest
    ```

    These are downstream presentation/reconciliation operations over an existing professional pack. They are not an alternate semantic pipeline.

    ## Removed Phase-1 surfaces

    The `human-report`, `run-human*`, `front-report`, `build-report`, and `build-front` command families were removed on 2026-08-24. Their former producer package (`accounting.human`) was an alternate presentation stack with no production Python caller outside its own package. The old front factory was static HTML scaffolding; the repository contains no Flask runtime.

    `accounting.publish.latest` remains supported, but it publishes a metrics/debt artifact bundle rather than requiring `human_reports` or a viewer application.
    '''
)

write(
    "notes/entrypoints.md",
    r'''
    ---
    id: notes/entrypoints
    title: "Accounting Backend Entrypoints"
    sidebar_label: "Accounting Backend Entrypoints"
    ---

    # Accounting Backend Entrypoints

    Status: current authority
    Last reviewed: 2026-08-24

    The Makefile is the command authority. Module CLIs are implementation entrypoints; start with `make help`.

    ## Canonical Make targets

    | Target | Responsibility |
    |---|---|
    | `make ledger` | Canonical ledger ingest. |
    | `make materialize` | Materialized Stage-D/semantic artifacts. |
    | `make debt` | Internal-debt resolution. |
    | `make debt-views` | Debt balance/activity views. |
    | `make metrics` | Metric values, registry, validation, views, drilldowns, annual dashboard artifacts. |
    | `make run-dashboard` | Assert governed annual dashboard contract outputs. |
    | `make publish-latest` | Package scope-qualified governed metrics/debt for downstream consumers. |
    | `make build-all` / `make run-full` | Full canonical path through publication and release check. |

    `make publish` is a compatibility alias for `publish-latest`. `run-accounting` and `run-accounting-full` remain compatibility aliases for `run-full`.

    ## Professional presentation targets

    | Target | Responsibility |
    |---|---|
    | `make professional-drilldowns` | Build/reconcile drilldowns for an existing professional pack. |
    | `make professional-linked-digest` | Render the professional pack plus drilldown links; presentation only. |

    These targets do not recompute accounting semantics and are not part of fixture CI because a real professional pack is local/external evidence.

    ## Canonical module CLIs

    - `python -m accounting.ledger.ingest`
    - `python -m accounting.stage_d.materialize`
    - `python -m accounting.marts.build`
    - `python -m accounting.debt.resolve`
    - `python -m accounting.debt.balance_views`
    - `python -m accounting.metrics.build`
    - `python -m accounting.publish.latest`
    - `python -m accounting.professional.drilldown`
    - `python -m accounting.professional.render_linked_digest`

    ## Retired Phase-1 entrypoints

    `accounting.human.*`, `accounting.viz.*`, `run-human*`, `human-report`, `front-report`, `build-report`, and `build-front` are removed. No supported code should recreate them as compatibility aliases. Historical documents may still mention them when describing prior architecture.
    '''
)

write(
    "notes/accounting_spine_runbook.md",
    r'''
    ---
    id: notes/accounting_spine_runbook
    title: "Accounting spine runbook"
    sidebar_label: "Accounting spine runbook"
    ---

    # Accounting spine runbook

    Status: current authority
    Last reviewed: 2026-08-24

    ## Official path

    The supported live order is:

    1. ingest / canonical ledger
    2. materialization + semantic marts
    3. debt resolution/views
    4. governed metrics + annual dashboard
    5. artifact publication

    `make run-full` is the canonical composite. The retired `human_reports` producer is not a stage.

    ## Key outputs

    ### Canonical run root — `out/run/accounting/<RUN_ID>/`
    - `ledger_canonical.csv`
    - `ledger_canonical_all_status.csv`
    - semantic/materialized monthly artifacts and QA

    ### Debt — `out/debt_resolution/<RUN_ID>/`
    - resolved debt evidence and status reconciliation
    - governed debt stock/activity inputs consumed downstream

    ### Metrics — `out/metrics/<RUN_ID>/`
    - `metric_registry.csv`
    - `metric_values.csv`
    - `validation_report.csv`
    - `build_manifest.json`
    - `metric_contract_frontier.csv`
    - `frontend_metric_series.csv`
    - `annual_balance_dashboard_metrics.csv`
    - `annual_balance_dashboard_contract.csv`
    - `annual_balance_dashboard_qa.csv`
    - governed metric views/drilldowns

    ### Publication — `public/accounting/latest_<SCOPE_TAG>/`
    - `manifest.json` (`accounting_public_bundle.v1`)
    - `artifact_contracts.csv`
    - `publish_contract_qa.csv`
    - classified governed metric/debt artifacts

    Publication is packaging only. It does not require `human_reports` and does not own a web application.

    ### Professional pack / drilldowns
    A real professional pack is generated/maintained outside fixture CI. `accounting.professional.drilldown` and `accounting.professional.render_linked_digest` are the supported richer human-facing surfaces. They must reconcile to displayed values and may not invent accounting semantics.

    ## Fixture validation

    ```bash
    make smoke-core
    make smoke-full
    make validate
    ```

    `smoke-full` is fixture-safe and includes publication dry-run. As frozen in the Phase-0 baseline, fixture debt and real professional-pack execution require separate evidence when affected.

    ## Logging
    Operational logs are evidence about execution, not substitutes for CSV/JSON contracts. A successful process exit is insufficient: validate totals, scope, currency grain, status, and affected drilldowns.
    '''
)

write(
    "notes/public_bundle_contract.md",
    r'''
    ---
    id: notes/public_bundle_contract
    title: "Accounting Public Bundle Contract"
    sidebar_label: "Accounting Public Bundle Contract"
    ---

    # Accounting Public Bundle Contract

    Status: current contract
    Last reviewed: 2026-08-24

    ## Purpose

    `accounting.publish.latest` packages governed accounting artifacts for downstream consumers. It performs no accounting computation and owns no Flask/frontend runtime.

    Canonical command:

    ```text
    make publish-latest
    ```

    Scope-qualified bundle:

    ```text
    public/accounting/latest_<SCOPE_TAG>/
    ```

    ## Sources

    Publication consumes only the matching latest metrics and debt roots:

    ```text
    out/metrics/latest_<SCOPE_TAG>
    out/debt_resolution/latest_<SCOPE_TAG>
    ```

    The resolved run identities must match. `out/human_reports` is not a source.

    ## Manifest

    `manifest.json` uses schema `accounting_public_bundle.v1` and contains source paths, source run identity, published files, metrics metadata, debt metadata, and publication mode. It intentionally has no report/navigation contract.

    ## Consumer rule

    Consumers may use the published bundle or the professional-pack/drilldown surfaces appropriate to their job. They must not treat presentation HTML, legacy reconciliation tables, or raw debt diagnostics as new accounting authority.

    The historical Python function name `build_frontend_snapshot_manifest` remains only as an explicitly deprecated external-import compatibility alias. Removal condition: an external-import census confirms no caller remains. The repository itself uses `build_public_bundle_manifest`.
    '''
)

write(
    "notes/documentation_compass.md",
    r'''
    ---
    id: notes/documentation_compass
    title: "Documentation Compass (Humans + Agents)"
    sidebar_label: "Documentation Compass (Humans + Agents)"
    ---

    # Documentation Compass (Humans + Agents)

    Status: current guidance
    Last reviewed: 2026-08-24

    ## Operator
    Read `notes/accounting_spine_runbook.md`, `notes/canonical_commands.md`, then `notes/public_bundle_contract.md`. Start with `make help`, `make doctor`, `make smoke-full`, and the smallest bounded live target needed.

    ## Developer
    Read `notes/current_state_map.md`, `notes/output_contracts.md`, `notes/entrypoints.md`, `tests/TESTING.md`, and the current Phase-0/Phase-1 simplification evidence notes. Preserve accounting authority and validate affected downstream layers.

    ## Analyst / stakeholder
    Use governed metrics/dashboard artifacts, or the professional pack plus linked drilldowns for human-facing review. The removed `out/human_reports` / `balance_humano_v2` path is not a supported surface.

    ## Coding agent
    Prefer canonical Make/module entrypoints. Do not recreate empty compatibility modules or alternate reporting engines. Historical dated audits are useful evidence but not current command authority.

    ## Current pipeline abstraction

    ```text
    ledger -> materialization/semantic marts -> debt -> metrics/dashboard -> publication
                                               \
                                                -> professional pack -> drilldowns/linked digest
    ```

    Publication is an artifact handoff; professional presentation is downstream. Neither is allowed to redefine accounting semantics.
    '''
)

write(
    "notes/current_state_map.md",
    r'''
    ---
    id: notes/current_state_map
    title: "Accounting Backend Current State Map"
    sidebar_label: "Accounting Backend Current State Map"
    ---

    # Accounting Backend Current State Map

    Status: current authority
    Last reviewed: 2026-08-24

    ## Artifact ladder

    ```text
    source inputs
      -> canonical ledger
      -> materialization + semantic marts
      -> debt stock/activity contracts
      -> governed metrics + annual dashboard
      -> published accounting bundle
      -> professional pack / drilldowns / linked digest (presentation)
    ```

    ## Ownership

    - `accounting.ledger` owns canonical ingest evidence.
    - `accounting.stage_d` / `accounting.marts` own materialized and semantic tables.
    - `accounting.debt.resolve` and `accounting.debt.balance_views` own debt resolution/balance evidence; empty re-export `models`/`rules` seams are gone.
    - `accounting.metrics` owns governed metric and annual-dashboard contracts.
    - `accounting.professional` owns professional table/drilldown/presentation machinery; it must consume governed values rather than become a parallel accounting engine.
    - `accounting.publish.latest` owns scope-safe packaging of metrics/debt into `public/accounting/latest_<SCOPE_TAG>`.

    ## Phase-1 removals

    `accounting.human`, `accounting.viz`, `accounting.config`, `accounting.contracts.models`, `accounting.debt.models`, `accounting.debt.rules`, and `accounting.publish.snapshot` were removed after exact reachability census showed no supported production caller. The old front factory was static HTML scaffolding; no Flask import/runtime was present.

    The former `human` capabilities were either pass-through projections of governed metric views or presentation duplication. Reusable current presentation belongs to professional table contracts, drilldowns, linked digest, and notebook/report consumers. No accounting formula was migrated into presentation code.
    '''
)

write(
    "notes/library/30-consumers/31-report-consumer-guide.md",
    r'''
    ---
    id: notes/library/30-consumers/31-report-consumer-guide
    title: "31 report consumer guide"
    sidebar_label: "31 report consumer guide"
    sidebar_position: 31
    ---

    # 31 report consumer guide

    Status: current (code-anchored)
    Last reviewed: 2026-08-24

    ## Choose the surface by job

    **Family/stakeholder review:** use the professional pack and `professional-linked-digest` when a human-facing pack is available. Its linked drilldowns must reconcile to the displayed cells.

    **Programmatic consumer:** use the scope-qualified `public/accounting/latest_<SCOPE_TAG>/manifest.json` and its listed governed artifacts.

    **Analyst/developer:** use canonical run, debt, and metrics roots directly for trace/reconciliation work; do not infer authority from presentation HTML.

    ## Retired surface

    `public/accounting/.../report/balance_humano_v2.html`, `out/human_reports/*`, and `accounting.human.*` are no longer produced or supported. No standalone Flask/frontend application is part of this repository.

    See `notes/public_bundle_contract.md` and `notes/accounting_spine_runbook.md`.
    '''
)

# Architecture regression: every surviving module must own something.
write(
    "tests/test_phase1_architecture_pruning.py",
    r'''
    from __future__ import annotations

    from pathlib import Path

    from accounting.publish.manifest import SCHEMA_NAME, build_public_bundle_manifest


    REPO_ROOT = Path(__file__).resolve().parents[1]


    def test_retired_empty_and_alternate_modules_are_absent() -> None:
        for rel in [
            "accounting/publish/snapshot.py",
            "accounting/debt/models.py",
            "accounting/debt/rules.py",
            "accounting/config.py",
            "accounting/contracts/models.py",
            "accounting/human",
            "accounting/viz",
        ]:
            assert not (REPO_ROOT / rel).exists(), rel


    def test_makefile_has_no_live_human_or_front_report_pipeline() -> None:
        make = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        for retired in [
            "accounting.human",
            "human_reports",
            "run-human",
            "human-report",
            "front-report",
            "build-report",
            "build-front",
            "_run_human_balance_action",
        ]:
            assert retired not in make
        assert "run-dashboard: run-metrics" in make
        assert "professional-drilldowns:" in make
        assert "professional-linked-digest:" in make


    def test_publication_is_metrics_debt_bundle_not_human_report_dependency() -> None:
        src = (REPO_ROOT / "accounting/publish/latest.py").read_text(encoding="utf-8")
        assert "human_latest" not in src
        assert "balance_human_v2" not in src
        assert "story_manifest" not in src
        assert "publish_report" not in src
        assert "publish_presentation" not in src
        assert SCHEMA_NAME == "accounting_public_bundle.v1"
        manifest = build_public_bundle_manifest(
            source_run_id="run-1",
            status="ok",
            source_paths={"metrics_latest": "m", "debt_latest": "d"},
            files=[],
            metrics={},
            debt={},
        )
        assert manifest["schema_name"] == "accounting_public_bundle.v1"
        assert "reports" not in manifest
        assert "navigation" not in manifest


    def test_no_flask_runtime_is_owned_by_accounting_source() -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (REPO_ROOT / "accounting").rglob("*.py")
        ).lower()
        assert "from flask" not in source
        assert "import flask" not in source
    '''
)

write(
    "notes/accounting_simplification_phase1_pruning_20260824.md",
    r'''
    # Accounting simplification Phase 1 — prune empty architecture

    Date: 2026-08-24  
    Base: `b9c391b159a62b6a08a81a6fb83fab86e1213eac`  
    Accounting-policy change: **none**  
    Intentional interface change: **retire the alternate human/front presentation stack**

    ## Invariant

    Semantic classification, monthly semantic totals, annual metric values/statuses, debt semantics, validated-cash rules, Box scope, native-currency separation, professional displayed values, and professional drilldown membership remain governed by the Phase-0 baseline. This PR removes alternate presentation/orchestration paths; it does not change an accounting rule.

    ## Reachability census and disposition

    | Candidate | Census | Disposition | Reason |
    |---|---|---|---|
    | `accounting/publish/snapshot.py` | no repository caller | DELETE | re-export-only “reserved future seam” |
    | `accounting/debt/models.py` | no repository caller | DELETE | re-export-only aliases from `debt.resolve` |
    | `accounting/debt/rules.py` | no repository caller | DELETE | re-export-only aliases from `debt.resolve` |
    | `accounting/config.py` | no repository caller | DELETE | stale parallel config loader; Make/env + stage CLIs are live control plane |
    | `accounting/contracts/models.py` | no repository caller | DELETE | unused parallel Pydantic ledger/money model; not canonical ledger authority |
    | `accounting/human/reports.py` | no external code caller | DELETE with package | deprecated wrapper over marts |
    | `accounting/viz/plots.py` | no live caller; CLI disabled | DELETE with package | obsolete plotting surface |

    ## `accounting.human` capability disposition

    The package had no Python caller outside itself. Make/docs exposed it as a second presentation pipeline.

    - `reports.py`: redundant wrapper over current marts — dismissed.
    - `tables.py`: adapters over already materialized metric/debt views — current reusable table governance lives in metric outputs and `accounting.professional.table_contracts` — dismissed.
    - `compact.py`: uncoupled compact-semester projection with no caller; equivalent analytical inputs remain available to notebooks/professional consumers — dismissed.
    - `document.py`: `balance_humano_v2` and a duplicate annual-dashboard projection — governed annual dashboard remains in `accounting.metrics`; richer human presentation belongs to the professional pack — dismissed.
    - `front.py`: static HTML factory containing explicit stub messages; no Flask import/runtime exists — dismissed. The professional linked digest is the maintained presentation path.

    No formula from these files is promoted to semantic authority. Removing a duplicate formatter is not permission to remove its upstream governed metric.

    ## Publication migration

    Before: `accounting.publish.latest` required matching `human_reports`, metrics, and debt latest roots; copied `balance_human_v2`; and published `accounting_frontend_snapshot.v1` with report/navigation fields.

    After: publication requires matching metrics + debt latest roots only and writes `accounting_public_bundle.v1`. It remains packaging-only. The old `build_frontend_snapshot_manifest` Python function name is retained only as an explicit deprecated external-import compatibility alias; repository code does not call it. Removal condition is a zero external-import census.

    ## Makefile migration

    Removed live `human-report`, `run-human*`, `front-report`, `build-report`, `build-front`, and `run-metrics-and-human` surfaces plus `out/human_reports` latest management. `run-full` now ends `... -> metrics -> dashboard -> publish -> release-check`. Professional drilldown/linked-digest targets remain separate because a real professional pack is not fixture-CI input.

    ## Evidence requirements

    Validation for this PR must include:

    1. `make validate`;
    2. `make smoke-full`;
    3. Phase-0 semantic/monthly/annual fixture parity;
    4. source reachability assertions that the retired modules/commands are absent;
    5. confirmation that the pre-existing `smoke-views` null-Box issue is not being reinterpreted or “fixed” here.

    A real professional-pack before/after is not fabricated: this PR does not modify professional calculation/drilldown code, and the repository fixture does not contain a real pack. Professional values remain governed by the unchanged professional regression suite.
    '''
)

# Remove the temporary applier from the final tree; the workflow removes itself too.
delete("scripts/_phase1_apply.py")
delete(".github/workflows/phase1-apply.yml")
