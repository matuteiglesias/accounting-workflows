#!/usr/bin/env python3
"""Validate a dashboard-ready governed accounting publish bundle."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-root", type=Path, default=Path("public/accounting/latest")
    )
    args = parser.parse_args()
    root = args.public_root
    if not root.exists():
        print(
            f"release-check: missing public bundle at {root}. Run make publish-latest after a producer run, or make smoke-full for fixture-safe validation.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    manifest_path = root / "manifest.json"
    contracts_path = root / "artifact_contracts.csv"
    qa_path = root / "publish_contract_qa.csv"
    annual_path = root / "canonical_dashboard" / "annual_balance_dashboard_metrics.csv"
    annual_qa_path = root / "canonical_dashboard" / "annual_balance_dashboard_qa.csv"
    frontier_path = root / "public_contract" / "metric_contract_frontier.csv"
    series_path = root / "canonical_dashboard" / "frontend_metric_series.csv"

    add("public_manifest_exists", manifest_path.exists(), str(manifest_path))
    add("artifact_contracts_exists", contracts_path.exists(), str(contracts_path))
    add("annual_dashboard_metrics_exists", annual_path.exists(), str(annual_path))
    add("annual_dashboard_qa_exists", annual_qa_path.exists(), str(annual_qa_path))
    add("metric_contract_frontier_exists", frontier_path.exists(), str(frontier_path))
    add("frontend_metric_series_exists", series_path.exists(), str(series_path))

    files: list[str] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [str(x) for x in manifest.get("files", [])]
        add("public_manifest_has_files", bool(files), f"files={len(files)}")
        retired_names = {
            "metric_registry.csv",
            "metric_values.csv",
            "income_statement_y.csv",
            "balance_cash_y.csv",
            "balance_debt_y.csv",
            "income_statement_q.csv",
            "balance_cash_q.csv",
            "balance_debt_q.csv",
        }
        retired_published = [
            f for f in files if Path(f).name in retired_names
        ]
        add(
            "retired_metric_artifacts_absent",
            not retired_published,
            f"published={retired_published}",
        )

    contract_rows = read_csv(contracts_path) if contracts_path.exists() else []
    public_contract_rows = [
        r
        for r in contract_rows
        if str(r.get("relpath", "")).startswith("public_contract/")
    ]
    unsafe_public = [
        r
        for r in public_contract_rows
        if r.get("frontend_suitability") in {"forbidden", "internal_only"}
        or r.get("artifact_role")
        in {"internal_balance", "inferred_reconciliation", "unsafe_for_frontend", "legacy"}
    ]
    add(
        "no_unsafe_artifacts_in_public_contract",
        not unsafe_public,
        f"rows={len(unsafe_public)}",
    )

    if qa_path.exists():
        qa_rows = read_csv(qa_path)
        qa_failures = [
            r
            for r in qa_rows
            if r.get("status") == "fail" and r.get("severity", "error") == "error"
        ]
        add(
            "publish_contract_qa_passes",
            not qa_failures,
            f"failures={len(qa_failures)}",
        )

    if annual_path.exists():
        annual_rows = read_csv(annual_path)
        cash_rows = [
            r for r in annual_rows if r.get("metric_id", "").startswith("BS.CASH")
        ]
        bad_cash = [
            r
            for r in cash_rows
            if r.get("value_status") == "available"
            and r.get("value", "") in {"", "0", "0.0"}
            and "unavailable" in r.get("caveat", "").lower()
        ]
        add(
            "cash_status_explicit",
            not bad_cash,
            f"cash_rows={len(cash_rows)} bad={len(bad_cash)}",
        )
        debt_stock = any(
            r.get("metric_id", "").startswith("ID.DEBT.")
            and r.get("flow_or_stock") == "stock"
            for r in annual_rows
        )
        debt_flow = any(
            r.get("metric_id", "").startswith("ID.DEBT.ACTIVITY")
            and r.get("flow_or_stock") == "flow"
            for r in annual_rows
        )
        add(
            "debt_stock_activity_separated",
            debt_stock and debt_flow,
            f"stock={debt_stock} flow={debt_flow}",
        )
        public_totals = [
            r
            for r in annual_rows
            if r.get("public_flag") == "true" and r.get("Currency", "") == ""
        ]
        cross_currency_bad = [
            r
            for r in public_totals
            if r.get("flow_or_stock") in {"flow", "stock"}
        ]
        add(
            "no_cross_currency_public_totals",
            not cross_currency_bad,
            f"rows={len(cross_currency_bad)}",
        )

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"{name}: {'pass' if ok else 'fail'} ({detail})")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
