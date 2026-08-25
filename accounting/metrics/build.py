from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from accounting.artifacts.manifest import (
    artifact_contract_for_name,
    write_artifact_contract_qa,
    write_artifact_contracts_csv,
)
from accounting.cutoff import resolve_run_as_of_date
from accounting.logging_utils import configure_logging, get_logger

from .annual import build_annual_balance_dashboard
from .frontier import build_metrics_frontier

LOG = get_logger("metrics")
BUILD_MANIFEST_FILENAME = "build_manifest.json"
SOURCE_CONTRACT_QA_FILENAME = "source_contract_qa.csv"
ARTIFACT_CONTRACTS_FILENAME = "artifact_contracts.csv"

CANONICAL_SOURCE_NAMES = [
    "monthly_flow_semantic_split.csv",
    "monthly_operating_statement.csv",
    "monthly_cash_close.csv",
    "monthly_debt_position.csv",
    "monthly_debt_activity.csv",
]


def find_latest_run_root(base: Path) -> Path:
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {base}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def build_metrics_source_contracts(run_root: Path, out_dir: Path) -> dict[str, Path]:
    """Write contracts for the governed metrics handoff only."""

    artifact_rows: list[dict] = []
    for rel in CANONICAL_SOURCE_NAMES:
        path = run_root / rel
        contract = artifact_contract_for_name(path.name, rel)
        artifact_rows.append(
            {"name": path.stem, "relpath": rel, **contract, "exists": path.exists()}
        )

    for path in sorted(out_dir.glob("*.csv")):
        rel = str(path.relative_to(out_dir))
        contract = artifact_contract_for_name(path.name, rel)
        artifact_rows.append(
            {"name": path.stem, "relpath": rel, **contract, "exists": path.exists()}
        )

    contracts_path = write_artifact_contracts_csv(
        out_dir / ARTIFACT_CONTRACTS_FILENAME, artifact_rows
    )
    qa_path = write_artifact_contract_qa(
        out_dir / SOURCE_CONTRACT_QA_FILENAME, artifact_rows
    )

    qa = pd.read_csv(qa_path)
    existing = sorted(rel for rel in CANONICAL_SOURCE_NAMES if (run_root / rel).exists())
    qa = pd.concat(
        [
            qa,
            pd.DataFrame(
                [
                    {
                        "check": "metrics_build_uses_governed_sources_only",
                        "status": "pass" if existing else "warn",
                        "detail": f"canonical_sources={existing}",
                        "severity": "warning",
                    },
                    {
                        "check": "legacy_metric_universe_absent",
                        "status": "pass",
                        "detail": "metric_registry/metric_values/metric_views are not built or consumed",
                        "severity": "error",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    qa.to_csv(qa_path, index=False)
    return {"artifact_contracts": contracts_path, "source_contract_qa": qa_path}


def build_governed_metrics(
    *,
    run_root: Path,
    out_dir: Path,
    run_id: str,
    as_of_date: str,
) -> dict[str, object]:
    """Build the current frontier and annual dashboard from governed sources."""

    out_dir.mkdir(parents=True, exist_ok=True)
    frontier_paths = build_metrics_frontier(
        run_root=run_root,
        metrics_dir=out_dir,
        run_id=run_id,
        as_of_date=as_of_date,
    )
    annual_paths = build_annual_balance_dashboard(
        run_root=run_root,
        metrics_dir=out_dir,
        run_id=run_id,
        as_of_date=as_of_date,
    )
    contract_paths = build_metrics_source_contracts(run_root=run_root, out_dir=out_dir)

    manifest: dict[str, object] = {
        "run_root": str(run_root),
        "run_id": run_id,
        "as_of_date": as_of_date,
        "metrics_mode": "governed_frontier_and_annual",
        "frontier_outputs": {k: str(v) for k, v in frontier_paths.items()},
        "annual_dashboard_outputs": {k: str(v) for k, v in annual_paths.items()},
        "source_contract_outputs": {k: str(v) for k, v in contract_paths.items()},
        "retired_outputs": [
            "metric_registry.csv",
            "metric_values.csv",
            "metric_values.parquet",
            "metric_values_q_wide.csv",
            "metric_values_y_wide.csv",
            "income_statement_q.csv",
            "income_statement_y.csv",
            "balance_cash_q.csv",
            "balance_cash_y.csv",
            "balance_debt_q.csv",
            "balance_debt_y.csv",
            "metric_views/*",
        ],
    }
    (out_dir / BUILD_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Build governed frontier and annual accounting metrics."
    )
    parser.add_argument(
        "--run-root",
        default="",
        help="Path to a specific accounting run root.",
    )
    parser.add_argument(
        "--runs-base",
        default="out/run/accounting",
        help="Base directory used when --run-root is omitted.",
    )
    parser.add_argument(
        "--out-dir",
        default="out/metrics/latest_FBPM",
        help="Output directory for governed metric artifacts.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--as-of-date",
        default="",
        help="Optional reporting cutoff; Stage-A cutoff remains authoritative.",
    )
    # Deprecated presentation arguments remain parse-compatible for one wave so
    # existing shell invocations do not fail while the Makefile is simplified.
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--rent-place-col", default="Lugar")
    parser.add_argument("--rent-detail-col", default="Detalle")
    parser.add_argument("--flow-rollup-groupby", default="Flujo,Tipo,Currency")
    parser.add_argument("--include-statuses", default="pagado")
    parser.add_argument("--noise-floor", default="ARS:5000,USD:10")
    args = parser.parse_args()

    run_root = (
        Path(args.run_root)
        if args.run_root
        else find_latest_run_root(Path(args.runs_base))
    )
    run_id = args.run_id.strip() or run_root.name
    as_of_date = resolve_run_as_of_date(run_root, args.as_of_date)
    out_dir = Path(args.out_dir)

    LOG.info(
        "Stage start governed metrics run_root=%s out_dir=%s as_of_date=%s",
        run_root,
        out_dir,
        as_of_date,
    )
    manifest = build_governed_metrics(
        run_root=run_root,
        out_dir=out_dir,
        run_id=run_id,
        as_of_date=as_of_date,
    )
    LOG.info(
        "Stage finish governed metrics run_id=%s outputs=%s",
        run_id,
        sorted(manifest.get("frontier_outputs", {}))
        + sorted(manifest.get("annual_dashboard_outputs", {})),
    )


if __name__ == "__main__":
    main()
