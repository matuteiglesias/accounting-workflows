#!/usr/bin/env python3
"""Static contract validation for the Makefile control-plane validate target."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from accounting.artifacts import manifest as artifact_manifest
from accounting.artifacts.manifest import artifact_contract_for_name, write_artifact_contract_qa, write_artifact_contracts_csv
try:
    from accounting.metrics.annual import ANNUAL_CONTRACT_COLUMNS, ANNUAL_METRICS_COLUMNS, QA_COLUMNS
    ANNUAL_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    ANNUAL_CONTRACT_COLUMNS = []
    ANNUAL_METRICS_COLUMNS = []
    QA_COLUMNS = []
    ANNUAL_IMPORT_ERROR = exc

from accounting.publish.latest import METRIC_FILES_BY_CLASS, DEBT_FILES_BY_CLASS, _published_contract_row

KNOWN_ARTIFACTS = [
    "ledger_canonical.csv",
    "semantic_rule_registry.csv",
    "classification_audit.csv",
    "semantic_dashboard_coverage.csv",
    "monthly_flow_semantic_split.csv",
    "monthly_operating_statement.csv",
    "monthly_cash_close.csv",
    "monthly_debt_position.csv",
    "monthly_debt_activity.csv",
    "per_flow_time_long.freq=M.csv",
    "daily_cash_position.csv",
    "metric_contract_frontier.csv",
    "frontend_metric_series.csv",
    "annual_balance_dashboard_metrics.csv",
    "annual_balance_dashboard_contract.csv",
    "income_statement_y.csv",
    "balance_cash_y.csv",
    "debt_open_items.csv",
    "build_manifest.json",
]

VOCABS = {
    "artifact_role": set(artifact_manifest.ARTIFACT_ROLES),
    "accounting_nature": set(artifact_manifest.ACCOUNTING_NATURES),
    "grain": set(artifact_manifest.GRAINS),
    "currency_policy": set(artifact_manifest.CURRENCY_POLICIES),
    "frontend_suitability": set(artifact_manifest.FRONTEND_SUITABILITIES),
    "source_authority": set(artifact_manifest.SOURCE_AUTHORITIES),
}


def fail(msg: str) -> None:
    print(f"artifact_contract_vocab_consistency: FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_contract_vocab() -> None:
    errors: list[str] = []
    for name in KNOWN_ARTIFACTS:
        contract = artifact_contract_for_name(name, name)
        for field, allowed in VOCABS.items():
            value = contract.get(field, "")
            if value not in allowed:
                errors.append(f"{name}: {field}={value!r} not in declared vocabulary")
    if errors:
        fail("; ".join(errors))
    print(f"artifact_contract_vocab_consistency: pass rows={len(KNOWN_ARTIFACTS)}")


def check_lookup_and_qa_smoke() -> None:
    rows = [{"name": name, "relpath": name} for name in KNOWN_ARTIFACTS]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        contracts_path = write_artifact_contracts_csv(root / "artifact_contracts.csv", rows)
        qa_path = write_artifact_contract_qa(root / "source_contract_qa.csv", rows)
        for path in [contracts_path, qa_path]:
            if not path.exists() or path.stat().st_size == 0:
                fail(f"expected non-empty smoke output: {path}")
        with contracts_path.open(encoding="utf-8", newline="") as fh:
            contract_rows = list(csv.DictReader(fh))
        if len(contract_rows) != len(rows):
            fail(f"artifact contract lookup smoke row mismatch: {len(contract_rows)} != {len(rows)}")
    print("artifact_contract_lookup_smoke: pass")
    print("source_contract_qa_smoke: pass")


def check_module_schema_smokes() -> None:
    if ANNUAL_IMPORT_ERROR is not None:
        print(f"annual_metrics_schema_import_smoke: warning skipped optional dependency import ({ANNUAL_IMPORT_ERROR})")
    else:
        for name, cols in {
            "annual_metrics_schema_import_smoke": ANNUAL_METRICS_COLUMNS,
            "annual_contract_schema_import_smoke": ANNUAL_CONTRACT_COLUMNS,
            "annual_qa_schema_import_smoke": QA_COLUMNS,
        }.items():
            if not cols:
                fail(f"{name} columns are empty")
            print(f"{name}: pass columns={len(cols)}")

    publish_files = [*METRIC_FILES_BY_CLASS.get("public_contract", []), *METRIC_FILES_BY_CLASS.get("canonical_dashboard", []), *DEBT_FILES_BY_CLASS.get("unsafe_for_frontend", [])]
    if not publish_files:
        fail("publish contract file lists are empty")
    for rel in ["public_contract/metric_contract_frontier.csv", "legacy_reconciliation/income_statement_y.csv", "unsafe_for_frontend/debt_open_items.csv"]:
        row = _published_contract_row(rel)
        if not row.get("publish_class"):
            fail(f"publish contract lookup missing class for {rel}")
    print(f"publish_contract_schema_import_smoke: pass files={len(publish_files)}")


def main() -> None:
    check_contract_vocab()
    check_lookup_and_qa_smoke()
    check_module_schema_smokes()


if __name__ == "__main__":
    main()
