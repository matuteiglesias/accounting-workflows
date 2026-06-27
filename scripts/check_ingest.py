from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from accounting.artifacts.manifest import artifact_from_path, append_artifacts
# from accounting.core.timeseries import period_bins_for_dates



REQUIRED_LEDGER_COLS = [
    "tx_id",
    "Date",
    "amount",
    "amount_cents",
    "Currency",
    "base_amount",
    "payer",
    "receiver",
    "Flujo",
    "Tipo",
    "source_file",
    "source_row",
    "ingest_ts",
    "notes",
]


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _relpath(root: Path, p: Path) -> str:
    return str(p.resolve().relative_to(root.resolve()))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    mode = (os.environ.get("MODE") or "").strip() or "unknown"

    meta_dir = out_dir / "meta"
    stage_manifest_path = meta_dir / "stage_A_ingest.json"
    ledger_path = out_dir / "ledger_canonical.csv"
    anomalies_path = out_dir / "anomalies.csv"

    checks: List[Dict[str, Any]] = []
    errors: List[str] = []

    # 1) ledger exists + readable + required cols
    if not ledger_path.exists():
        errors.append(f"missing ledger: {ledger_path}")
    else:
        try:
            df = pd.read_csv(ledger_path, low_memory=False)
            checks.append({"name": "ledger_readable", "ok": True, "details": {"rows": int(df.shape[0])}})
        except Exception as e:
            errors.append(f"failed reading ledger_canonical.csv: {e}")
            df = None

        if df is not None:
            missing = [c for c in REQUIRED_LEDGER_COLS if c not in df.columns]
            if missing:
                errors.append(f"ledger missing required cols: {missing}")
            else:
                checks.append({"name": "ledger_required_cols", "ok": True, "details": {"n_cols": int(df.shape[1])}})

            if df.shape[0] <= 0:
                errors.append("ledger has 0 rows")

    # 2) anomalies optional but if exists must be non-empty header parseable
    if anomalies_path.exists():
        try:
            an = pd.read_csv(anomalies_path, low_memory=False)
            checks.append({"name": "anomalies_readable", "ok": True, "details": {"rows": int(an.shape[0])}})
        except Exception as e:
            errors.append(f"anomalies.csv unreadable: {e}")
    else:
        checks.append({"name": "anomalies_optional", "ok": True, "details": {"present": False}})

    # 3) stage manifest exists + shape
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
        if stage_manifest.get("stage") != "A.ingest":
            errors.append(f"stage manifest has wrong stage: {stage_manifest.get('stage')}")
        if not isinstance(stage_manifest.get("outputs"), list):
            errors.append("stage manifest outputs must be a list")
        if str(stage_manifest.get("mode", "")).strip() in ("", "$(MODE)"):
            errors.append("stage manifest mode must be real string, not empty or $(MODE)")

    ok = len(errors) == 0

    # 4) write check manifest
    check_manifest_path = meta_dir / "check_A_ingest.json"
    check_manifest = {
        "stage": "A.ingest",
        "mode": mode,
        "ok": ok,
        "generated_at": _utc_now_iso(),
        "stage_manifest_relpath": _relpath(out_dir, stage_manifest_path) if stage_manifest_path.exists() else None,
        "checks": checks,
        "errors": errors,
    }
    _write_json(check_manifest_path, check_manifest)

    # 5) append artifacts row for the check manifest itself
    art_check = artifact_from_path(
        name="check_A_ingest",
        path=check_manifest_path,
        stage="A.ingest",
        mode=mode,
        run_id=(stage_manifest or {}).get("run_id", "") if stage_manifest_path.exists() else "",
        role="meta",
        root_dir=out_dir,
        content_type="application/json",
    )
    append_artifacts(meta_dir, [art_check])

    if not ok:
        print("[FAIL] check_ingest", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    print("[OK] check_ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
