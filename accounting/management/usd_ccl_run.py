"""Explicit offline orchestration for one existing canonical run root."""

from __future__ import annotations

import argparse
from pathlib import Path

from accounting.management.usd_ccl_flows import (
    MANAGEMENT_IMPLEMENTATION_ID,
    build_usd_ccl_management_flows,
)
from accounting.valuation.usd_ccl import ValuationContractError, build_usd_ccl_valuation


def _run_root(value: str | Path) -> Path:
    raw = str(value)
    if "://" in raw:
        raise ValuationContractError("run root must be a local filesystem path")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ValuationContractError(f"run root does not exist: {root}")
    for name in ["ledger_canonical.csv", "classification_audit.csv"]:
        if not (root / name).is_file():
            raise ValuationContractError(f"existing run root missing required artifact: {root / name}")
    return root


def run_usd_ccl_management_flows(
    *, run_root: str | Path, rates_path: str | Path, policy_path: str | Path,
) -> dict[str, Path]:
    """Value and project an existing run without invoking ingest or publication."""
    root = _run_root(run_root)
    valuation = build_usd_ccl_valuation(
        ledger_path=root / "ledger_canonical.csv",
        rates_path=rates_path,
        policy_path=policy_path,
        output_dir=root / "valuations" / "usd_ccl",
        run_id=f"offline-usd-ccl-{root.name}",
        source_scope_tag=root.name,
        content_addressed=True,
        mode="offline",
    )
    management = build_usd_ccl_management_flows(
        ledger_path=root / "ledger_canonical.csv",
        semantic_audit_path=root / "classification_audit.csv",
        valuation_sidecar_path=valuation["sidecar"],
        valuation_manifest_path=valuation["manifest"],
        output_dir=(
            valuation["sidecar"].parent
            / "management"
            / MANAGEMENT_IMPLEMENTATION_ID
        ),
    )
    return {**valuation, **{f"management_{key}": value for key, value in management.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline USD/CCL valuation and management flows for an existing run"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--rates", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    outputs = run_usd_ccl_management_flows(
        run_root=args.run_root, rates_path=args.rates, policy_path=args.policy
    )
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
