from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from accounting.artifacts.manifest import artifact_from_path, append_artifacts

REQUIRED_LEDGER_DIMENSIONS = ["Box", "Currency", "Flujo", "Tipo"]


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _relpath(root: Path, p: Path) -> str:
    return str(p.resolve().relative_to(root.resolve()))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv_if_exists(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False)


def _empty_required_dimension_mask(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Return rows where any required dimension is missing or blank."""
    if df.empty:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        values = df[col]
        mask = mask | values.isna() | values.astype("string").str.strip().isin(["", "nan", "None", "<NA>"])
    return mask


def _format_empty_dimension_error(df: pd.DataFrame, cols: List[str], sample_size: int = 10) -> str:
    mask = _empty_required_dimension_mask(df, cols)
    bad = df.loc[mask].copy()
    missing_counts = {}
    for col in cols:
        if col not in df.columns:
            missing_counts[col] = "missing_column"
            continue
        values = df[col]
        col_mask = values.isna() | values.astype("string").str.strip().isin(["", "nan", "None", "<NA>"])
        missing_counts[col] = int(col_mask.sum())

    amount_total = 0.0
    if "amount" in bad.columns:
        amount_total = float(pd.to_numeric(bad["amount"], errors="coerce").fillna(0).sum())

    sample_cols = [
        c
        for c in ["tx_id", "Date", "amount", "Box", "Currency", "Flujo", "Tipo", "source_file", "source_row", "notes"]
        if c in bad.columns
    ]
    sample = bad[sample_cols].head(sample_size).fillna("").to_dict(orient="records") if sample_cols else []
    return (
        "ledger_canonical has rows with empty required dimensions "
        f"{cols}: rows={int(mask.sum())}, amount_sum={amount_total}, missing_by_column={missing_counts}, "
        f"sample_first_{sample_size}={sample}. "
        "Fill Box/Currency/Flujo/Tipo in the source ledger for these rows; materialized flow tables keep these rows "
        "for reconciliation, but the pipeline treats blank dimensions as data-quality errors."
    )


def main() -> int:
    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    mode = (os.environ.get("MODE") or "").strip() or "unknown"
    freq = (os.environ.get("FREQ") or "W").strip() or "W"

    meta_dir = out_dir / "meta"
    stage_manifest_path = meta_dir / "stage_D_materialize.json"

    ledger_path = out_dir / "ledger_canonical.csv"
    per_flow_path = out_dir / f"per_flow_time_long.freq={freq}.csv"
    per_party_path = out_dir / f"per_party_time_long.freq={freq}.csv"
    daily_cash_path = out_dir / "daily_cash_position.csv"
    loans_path = out_dir / "loans_time.freq=M.csv"  # tu pipeline lo fija en M

    checks: List[Dict[str, Any]] = []
    errors: List[str] = []

    # required files
    for p, name in [
        (ledger_path, "ledger_canonical.csv"),
        (per_flow_path, f"per_flow_time_long.freq={freq}.csv"),
        (per_party_path, f"per_party_time_long.freq={freq}.csv"),
        (daily_cash_path, "daily_cash_position.csv"),
        (loans_path, "loans_time.freq=M.csv"),
    ]:
        if not p.exists():
            errors.append(f"missing materialized output: {name} at {p}")

    # basic content checks
    ledger = _read_csv_if_exists(ledger_path)
    per_flow = _read_csv_if_exists(per_flow_path)

    if ledger is not None:
        if ledger.shape[0] <= 0:
            errors.append("ledger_canonical has 0 rows")
        else:
            checks.append({"name": "ledger_rows", "ok": True, "details": {"rows": int(ledger.shape[0])}})

        missing_dimension_cols = [c for c in REQUIRED_LEDGER_DIMENSIONS if c not in ledger.columns]
        if missing_dimension_cols:
            errors.append(f"ledger_canonical missing required dimension columns: {missing_dimension_cols}")
        else:
            empty_dimension_mask = _empty_required_dimension_mask(ledger, REQUIRED_LEDGER_DIMENSIONS)
            empty_dimension_rows = int(empty_dimension_mask.sum())
            hard_fail_empty_dimensions = mode != "smoke"
            checks.append(
                {
                    "name": "ledger_required_dimensions_non_empty",
                    "ok": empty_dimension_rows == 0 or not hard_fail_empty_dimensions,
                    "details": {
                        "columns": REQUIRED_LEDGER_DIMENSIONS,
                        "empty_rows": empty_dimension_rows,
                        "severity": "error" if hard_fail_empty_dimensions else "warning",
                    },
                }
            )
            if empty_dimension_rows and hard_fail_empty_dimensions:
                errors.append(_format_empty_dimension_error(ledger, REQUIRED_LEDGER_DIMENSIONS))

    if per_flow is not None:
        if per_flow.shape[0] <= 0:
            errors.append("per_flow_time_long has 0 rows")
        else:
            checks.append({"name": "per_flow_rows", "ok": True, "details": {"rows": int(per_flow.shape[0])}})

        # sum consistency if amount column is present
        try:
            if ledger is not None and "amount" in ledger.columns and "amount" in per_flow.columns:
                ledger_sum = pd.to_numeric(ledger["amount"], errors="coerce").fillna(0).sum()
                flow_sum = pd.to_numeric(per_flow["amount"], errors="coerce").fillna(0).sum()
                diff = float(ledger_sum - flow_sum)
                checks.append({"name": "sum_match_ledger_vs_per_flow", "ok": abs(diff) < 1e-6, "details": {"diff": diff}})
                if abs(diff) >= 1e-6:
                    errors.append(f"ledger_sum != per_flow_sum (diff={diff})")
        except Exception as e:
            errors.append(f"sum check failed: {e}")

    # stage manifest presence
    if not stage_manifest_path.exists():
        errors.append(f"missing stage manifest: {stage_manifest_path}")
        stage_manifest = None
    else:
        try:
            stage_manifest = _read_json(stage_manifest_path)
            checks.append({"name": "stage_manifest_json", "ok": True, "details": {"path": _relpath(out_dir, stage_manifest_path)}})
        except Exception as e:
            errors.append(f"stage manifest invalid json: {e}")
            stage_manifest = None

    if stage_manifest is not None:
        if stage_manifest.get("stage") != "D.materialize":
            errors.append(f"stage manifest has wrong stage: {stage_manifest.get('stage')}")
        if not isinstance(stage_manifest.get("outputs"), list):
            errors.append("stage manifest outputs must be a list")
        if str(stage_manifest.get("mode", "")).strip() in ("", "$(MODE)"):
            errors.append("stage manifest mode must be real string, not empty or $(MODE)")

    ok = len(errors) == 0

    # write check manifest
    check_manifest_path = meta_dir / "check_D_materialize.json"
    check_manifest = {
        "stage": "D.materialize",
        "mode": mode,
        "ok": ok,
        "generated_at": _utc_now_iso(),
        "stage_manifest_relpath": _relpath(out_dir, stage_manifest_path) if stage_manifest_path.exists() else None,
        "checks": checks,
        "errors": errors,
    }
    _write_json(check_manifest_path, check_manifest)

    art_check = artifact_from_path(
        name="check_D_materialize",
        path=check_manifest_path,
        stage="D.materialize",
        mode=mode,
        run_id=(stage_manifest or {}).get("run_id", "") if stage_manifest_path.exists() else "",
        role="meta",
        root_dir=out_dir,
        content_type="application/json",
    )
    append_artifacts(meta_dir, [art_check])

    if not ok:
        print("[FAIL] check_materialize", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    print("[OK] check_materialize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
