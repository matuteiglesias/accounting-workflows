# src/accounting/materialize.py
"""
Materialization layer for accounting pipeline (CSV outputs).

Exports:
  - materialize_per_flow(ledger_df, out_dir, freq)
  - materialize_per_party(ledger_df, out_dir, freq)
  - materialize_daily_cash(ledger_df, out_dir, as_of=None)
  - materialize_loans(ledger_df, loan_register_df, out_dir, freq)
  - materialize_all(ledger_df, out_dir, freq="W", force=False)

Writes CSV files (atomic) and small JSON manifest / partitions files.
"""
from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Tuple

import pandas as pd

from accounting.core_timeseries import (
    aggregate_per_flow,
    expand_party_rows,
    aggregate_per_party,
    compute_daily_cash_position,
    compute_loans_time,
)



# CLI wrapper to run src.accounting.materialize.materialize_all with sensible defaults.

# Usage examples:
#   python scripts/materialize.py --out-dir ./out --freq W --force 1


LOG = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)



import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# LOG = logging.getLogger("materialize_script")




# -----------------------
# Small helpers (I/O & metadata)
# -----------------------
def _ensure_amount_float(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ledger has 'amount' float column. If only amount_cents exists, derive it.
    Returns a copy.
    """
    df = ledger.copy()
    if "amount" not in df.columns and "amount_cents" in df.columns:
        df["amount"] = pd.to_numeric(df["amount_cents"], errors="coerce").fillna(0).astype(float) / 100.0
    if "amount" not in df.columns:
        df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0).astype(float)
    return df


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Atomically write CSV to `path`: write to tmp then rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Use index=False by default
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        LOG.exception("Failed to hash file: %s", path)
        return None


def load_partitions_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        LOG.exception("Failed reading partitions.json - returning empty")
        return {}


def save_partitions_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    tmp.replace(path)


def _write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "meta/stage_D_materialize.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str, ensure_ascii=False), encoding="utf8")
    tmp.replace(manifest_path)
    return manifest_path


# -----------------------
# Per-artifact materializers (CSV outputs)
# -----------------------
def materialize_per_flow(
    ledger: pd.DataFrame, out_dir: Path, freq: str = "W", force: bool = False
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce per_flow_time_long.freq=<freq>.csv aggregated table.

    Returns (df, path)
    """
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"per_flow_time_long.freq={freq}.csv"
    target = out_dir / fname

    LOG.info("Materializing per-flow (freq=%s) -> %s", freq, target)
    df = aggregate_per_flow(ledger, freq=freq, amount_col="amount", date_col="Date")

    # normalize TimePeriod -> ISO-like string and TimePeriod_ts_end to date string
    if "TimePeriod" in df.columns:
        df["TimePeriod"] = df["TimePeriod"].astype(str)
    if "TimePeriod_ts_end" in df.columns:
        df["TimePeriod_ts_end"] = pd.to_datetime(df["TimePeriod_ts_end"], errors="coerce").dt.date.astype(str)

    # ensure columns order (best-effort)
    cols = ["TimePeriod", "TimePeriod_ts_end", "Flujo", "Tipo", "Currency", "amount", "n_tx"]
    out_df = df[[c for c in cols if c in df.columns]].copy()

    _atomic_write_csv(out_df, target)
    LOG.info("Wrote per_flow rows=%d -> %s", len(out_df), target)
    return out_df, target


def materialize_per_party(
    ledger: pd.DataFrame, out_dir: Path, freq: str = "W", force: bool = False
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce per_party_time_long.freq=<freq>.csv (weekly per-party aggregates).

    Returns (df, path)
    """
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"per_party_time_long.freq={freq}.csv"
    target = out_dir / fname

    LOG.info("Materializing per-party (freq=%s) -> %s", freq, target)
    # expand then aggregate
    print(f"check 1 shape {ledger.shape}, columns: {ledger.columns}, count: {ledger.count()}")
    expanded = expand_party_rows(ledger, amount_col="amount", date_col="Date")
    print(f"check 2 shape {expanded.shape}, columns: {expanded.columns}, count: {expanded.count()}")
    agg = aggregate_per_party(expanded, freq=freq, amount_col="signed_amount", date_col="Date")
    print(f"check 3 shape {agg.shape}, columns: {agg.columns}")

    # normalize TimePeriod -> string and TimePeriod_ts_end to date string
    if "TimePeriod" in agg.columns:
        agg["TimePeriod"] = agg["TimePeriod"].astype(str)
    if "TimePeriod_ts_end" in agg.columns:
        agg["TimePeriod_ts_end"] = pd.to_datetime(agg["TimePeriod_ts_end"], errors="coerce").dt.date.astype(str)

    cols = ["TimePeriod", "TimePeriod_ts_end", "party", "role", "Flujo", "Tipo", "Currency", "amount", "n_tx"]
    out_df = agg[[c for c in cols if c in agg.columns]].copy()

    _atomic_write_csv(out_df, target)
    LOG.info("Wrote per_party rows=%d -> %s", len(out_df), target)
    return out_df, target


def materialize_daily_cash(
    ledger: pd.DataFrame, out_dir: Path, as_of: Optional[pd.Timestamp] = None, force: bool = False
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce daily_cash_position.csv. If as_of is provided, produce snapshot up to that date;
    otherwise produce full daily series for the ledger range.
    """
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "daily_cash_position.csv"

    LOG.info("Materializing daily cash position -> %s", target)
    df = compute_daily_cash_position(ledger, opening_balances=None, freq="D", amount_col="amount", date_col="Date")

    # If as_of provided, filter up to that date
    if as_of is not None:
        asof_date = pd.to_datetime(as_of).date()
        df = df[pd.to_datetime(df["Date"]).dt.date <= asof_date].copy()

    # ensure Date as ISO date string for CSV
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype(str)

    out_df = df[["Date", "party", "balance", "Currency", "source_ledger_hash"]] if all(
        c in df.columns for c in ["Date", "party", "balance", "Currency", "source_ledger_hash"]
    ) else df.copy()

    _atomic_write_csv(out_df, target)
    LOG.info("Wrote daily_cash rows=%d -> %s", len(out_df), target)
    return out_df, target


def materialize_loans(
    ledger: pd.DataFrame,
    loan_register_df: Optional[pd.DataFrame],
    out_dir: Path,
    freq: str = "M",
    force: bool = False,
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce loans_time.freq=<freq>.csv (monthly loan schedules / reconciliation).
    """
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"loans_time.freq={freq}.csv"
    target = out_dir / fname

    LOG.info("Materializing loans (freq=%s) -> %s", freq, target)
    df = compute_loans_time(ledger, loan_register_df, freq=freq, amount_col="amount", date_col="Date")

    # normalize TimePeriod -> string and TimePeriod_ts_end to date string
    if "TimePeriod" in df.columns:
        df["TimePeriod"] = df["TimePeriod"].astype(str)
    if "TimePeriod_ts_end" in df.columns:
        df["TimePeriod_ts_end"] = pd.to_datetime(df["TimePeriod_ts_end"], errors="coerce").dt.date.astype(str)

    out_df = df.copy()
    _atomic_write_csv(out_df, target)
    LOG.info("Wrote loans rows=%d -> %s", len(out_df), target)
    return out_df, target


# -----------------------
# Orchestrator
# -----------------------
def materialize_all(
    ledger_df: pd.DataFrame,
    out_dir: Path,
    freq: str = "W",
    loan_register_df: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate materialization:
      - ledger_canonical.csv (copy of canonical ledger)
      - per_flow_time_long.freq=<freq>.csv
      - per_party_time_long.freq=<freq>.csv
      - loans_time.freq=<freq>.csv
      - daily_cash_position.csv
      - manifest.json and partitions.json

    Returns manifest dict with aggregate metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ensure ledger has amount float
    ledger_df = _ensure_amount_float(ledger_df)

    outputs = {}
    aggregates = {}

    # 0) write canonical ledger copy (csv)
    ledger_path = out_dir / "ledger_canonical.csv"
    LOG.info("Writing ledger_canonical -> %s", ledger_path)
    # enforce Date formatting and stable column order (best-effort)
    ldf = ledger_df.copy()
    if "Date" in ldf.columns:
        ldf["Date"] = pd.to_datetime(ldf["Date"], errors="coerce").dt.date.astype(str)
    _atomic_write_csv(ldf, ledger_path)
    aggregates["ledger_canonical.csv"] = {"path": str(ledger_path), "rows": len(ldf), "sha256": _sha256_file(ledger_path)}

    # 1) per_flow
    try:
        pf_df, pf_path = materialize_per_flow(ledger_df, out_dir, freq=freq, force=force)
        aggregates[pf_path.name] = {"path": str(pf_path), "rows": len(pf_df), "sha256": _sha256_file(pf_path)}
    except Exception:
        LOG.exception("Failed materialize_per_flow")
        aggregates["per_flow_failed"] = {"error": "failed"}

    # 2) per_party
    # ledger_df
    print(f"check 0 shape {ledger_df.shape}, columns: {ledger_df.columns}, count: {ledger_df.count()}")

    try:
        pp_df, pp_path = materialize_per_party(ledger_df, out_dir, freq=freq, force=force)
        aggregates[pp_path.name] = {"path": str(pp_path), "rows": len(pp_df), "sha256": _sha256_file(pp_path)}
    except Exception:
        LOG.exception("Failed materialize_per_party")
        aggregates["per_party_failed"] = {"error": "failed"}

    print(f"Check 4 shape {pp_df.shape}, columns {pp_df.columns}")

    # 3) loans
    try:
        loans_df, loans_path = materialize_loans(ledger_df, loan_register_df, out_dir, freq=("M" if freq.upper().startswith("M") else "M"), force=force)
        aggregates[loans_path.name] = {"path": str(loans_path), "rows": len(loans_df), "sha256": _sha256_file(loans_path)}
    except Exception:
        LOG.exception("Failed materialize_loans")
        aggregates["loans_failed"] = {"error": "failed"}

    # 4) daily cash
    try:
        dc_df, dc_path = materialize_daily_cash(ledger_df, out_dir, as_of=None, force=force)
        aggregates[dc_path.name] = {"path": str(dc_path), "rows": len(dc_df), "sha256": _sha256_file(dc_path)}
    except Exception:
        LOG.exception("Failed materialize_daily_cash")
        aggregates["daily_cash_failed"] = {"error": "failed"}

    # 5) partitions.json update (simple)
    parts_path = out_dir / "partitions.json"
    parts = load_partitions_json(parts_path)
    parts["last_materialized_at"] = pd.Timestamp.utcnow().isoformat()
    parts["freq"] = freq
    # best-effort last TimePeriod processed from per_flow or per_party
    try:
        if "TimePeriod_ts_end" in pf_df.columns:
            last_end = pd.to_datetime(pf_df["TimePeriod_ts_end"]).max().date().isoformat()
            parts["last_period_end"] = last_end
    except Exception:
        pass
    parts["outputs"] = {k: {"rows": v.get("rows")} for k, v in aggregates.items() if isinstance(v, dict) and "rows" in v}
    save_partitions_json(parts_path, parts)

    # 6) manifest
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "aggregates": aggregates,
        "partitions": parts,
        "anomalies": None,
    }
    # attach anomalies if present in ledger attrs
    anomalies = ledger_df.attrs.get("anomalies")
    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        anomalies_path = out_dir / "anomalies.csv"
        _atomic_write_csv(anomalies, anomalies_path)
        manifest["anomalies"] = {"path": str(anomalies_path), "rows": len(anomalies), "sha256": _sha256_file(anomalies_path)}

    manifest_path = _write_manifest(out_dir, manifest)

    LOG.info("Materialization complete. Manifest: %s", manifest_path)
    return manifest




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run materialize_all over ledger_canonical.csv")
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "./out"))
    p.add_argument("--freq", default=os.getenv("FREQ", "W"))
    p.add_argument("--force", default=os.getenv("FORCE", "0"))
    p.add_argument("--mode", default=os.getenv("MODE", "run"), choices=["smoke", "run"])
    p.add_argument("--run-id", default=os.getenv("RUN_ID", ""))
    return p.parse_args()



def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()
    out_dir = Path(args.out_dir)
    ledger_path = out_dir / "ledger_canonical.csv"
    if not ledger_path.exists():
        LOG.error("ledger_canonical.csv not found at %s", ledger_path)
        raise SystemExit(2)

    try:
        ledger = pd.read_csv(ledger_path, dtype=str)
    except Exception:
        LOG.exception("Failed reading ledger_canonical.csv")
        raise SystemExit(3)

    # coerce Date and numeric columns where possible
    if "Date" in ledger.columns:
        try:
            ledger["Date"] = pd.to_datetime(ledger["Date"], errors="coerce")
        except Exception:
            pass

    for col in ("amount", "amount_cents"):
        if col in ledger.columns:
            try:
                ledger[col] = pd.to_numeric(ledger[col].astype(str).str.replace(",", ""), errors="coerce")
            except Exception:
                pass

    freq = args.freq or os.getenv("FREQ", "W")
    force_flag = bool(int(str(args.force)))

    _ = materialize_all(ledger, out_dir=out_dir, freq=freq, force=force_flag)




    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)



    # inputs
    in_art = artifact_from_path(
        name="ledger_canonical",
        path=ledger_path,
        stage="D.materialize",
        mode=args.mode,
        run_id=(args.run_id or "smoke" if args.mode == "smoke" else ""),
        role="input",
        root_dir=out_dir,
        content_type="text/csv",
    )

    # outputs esperados
    out_arts = []

    per_flow = out_dir / f"per_flow_time_long.freq={freq}.csv"
    if per_flow.exists():
        out_arts.append(artifact_from_path(
            name="per_flow_time_long",
            path=per_flow,
            stage="D.materialize",
            mode=args.mode,
            run_id=(args.run_id or "smoke" if args.mode == "smoke" else ""),
            role="derived",
            root_dir=out_dir,
            content_type="text/csv",
        ))

    per_party = out_dir / f"per_party_time_long.freq={freq}.csv"
    if per_party.exists():
        out_arts.append(artifact_from_path(
            name="per_party_time_long",
            path=per_party,
            stage="D.materialize",
            mode=args.mode,
            run_id=(args.run_id or "smoke" if args.mode == "smoke" else ""),
            role="derived",
            root_dir=out_dir,
            content_type="text/csv",
        ))

    loans = out_dir / "loans_time.freq=M.csv"
    if loans.exists():
        out_arts.append(artifact_from_path(
            name="loans_time",
            path=loans,
            stage="D.materialize",
            mode=args.mode,
            run_id=(args.run_id or "smoke" if args.mode == "smoke" else ""),
            role="derived",
            root_dir=out_dir,
            content_type="text/csv",
        ))

    daily_cash = out_dir / "daily_cash_position.csv"
    if daily_cash.exists():
        out_arts.append(artifact_from_path(
            name="daily_cash_position",
            path=daily_cash,
            stage="D.materialize",
            mode=args.mode,
            run_id=(args.run_id or "smoke" if args.mode == "smoke" else ""),
            role="derived",
            root_dir=out_dir,
            content_type="text/csv",
        ))

    stage_manifest = {
        "stage": "D.materialize",
        "mode": args.mode,
        "run_id": (args.run_id or "smoke" if args.mode == "smoke" else ""),
        "inputs": [in_art],
        "params": {"freq": freq, "force": int(force_flag)},
        "outputs": out_arts,
        "warnings": [],
    }

    stage_manifest_rel = write_stage_manifest(meta_dir, stage_manifest)

    stage_meta_art = {
        "run_id": stage_manifest["run_id"],
        "stage": "D.materialize",
        "mode": args.mode,
        "name": "stage_D_materialize",
        "relpath": stage_manifest_rel,
        "sha256": artifact_from_path(  # reutilizamos para hash/bytes
            name="stage_D_materialize",
            path=(out_dir / stage_manifest_rel),
            stage="D.materialize",
            mode=args.mode,
            run_id=stage_manifest["run_id"],
            role="meta",
            root_dir=out_dir,
            content_type="application/json",
        )["sha256"],
        "bytes": (out_dir / stage_manifest_rel).stat().st_size,
        "rows": None,
        "content_type": "application/json",
        "created_at": stage_manifest.get("generated_at"),
        "role": "meta",
    }

    append_artifacts(meta_dir, [in_art, *out_arts, stage_meta_art])

    return 0





if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)