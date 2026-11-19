"""Generate Phase-A monthly policy inputs against an exact pinned source snapshot.

Uses the same materialization/debt/metrics stages as the scenario runner but
runs entirely inside an isolated workspace and exports only derived CSV/JSON.
No scenario policy is applied; no official output/latest is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from accounting.scenarios.policy_inputs import build_policy_inputs
from accounting.scenarios.runner import run_stages, seed_execution
from accounting.support.isolated_execution import isolated_accounting_execution


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *, repo_root: Path, base_run_id: str, out_dir: Path, as_of_date: str,
    through_date: str, start_month: str, agreement: Path,
) -> Path:
    repo_root = repo_root.resolve(strict=True)
    base = repo_root / "out" / "run" / "accounting" / base_run_id
    if not base.is_dir() or base.is_symlink():
        raise ValueError("base must be exact existing run directory")
    if not (base / "meta" / "stage_A_ingest.json").is_file():
        raise ValueError("missing pinned Stage A provenance")
    output = out_dir.resolve()
    if output.is_relative_to(repo_root / "out") or output.is_relative_to(repo_root / "public"):
        raise ValueError("derived outputs must be outside production run/output trees")
    if output.exists():
        raise FileExistsError("refuse overwrite of existing Phase A output")
    output.parent.mkdir(parents=True, exist_ok=True)
    base_hash = _sha(base / "ledger_canonical_all_status.csv")

    with tempfile.TemporaryDirectory(prefix=".policy-input-export-", dir=output.parent) as temp:
        export = Path(temp) / "result"
        export.mkdir()
        with isolated_accounting_execution(repo_root, temp_parent=output.parent) as execution:
            seed_execution(
                base, execution.canonical_root, execution.run_id,
                "monthly_policy_inputs", {"id": "pre_policy_baseline", "operations": []},
                projection={"as_of_date": as_of_date, "through_date": through_date},
            )
            run_stages(execution)
            derived = build_policy_inputs(
                run_root=execution.canonical_root, out_dir=export,
                as_of_date=as_of_date, through_date=through_date,
                start_month=start_month, agreement_path=agreement,
            )
            for value in derived.values():
                if not value.is_file():
                    raise ValueError(f"missing policy input output: {value}")
            manifest = derived["manifest"]
            meta = json.loads(manifest.read_text())
            meta.update({
                "base_run_id": base_run_id,
                "base_all_status_sha256": base_hash,
                "agreement_sha256": _sha(agreement),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_kind": "isolated_paid_plus_future_planned_pre_policy",
                "scenario_operations_applied": [],
            })
            manifest.write_text(json.dumps(meta, indent=2) + "\n")
        # Publish a complete folder only when the isolated stages succeed.
        export.replace(output)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--base-run-id", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--as-of-date", required=True)
    p.add_argument("--through-date", required=True)
    p.add_argument("--start-month", required=True)
    p.add_argument("--agreement", required=True, type=Path)
    args = p.parse_args()
    print(run(**vars(args)), flush=True)


if __name__ == "__main__":
    main()
