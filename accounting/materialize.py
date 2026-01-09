
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

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from accounting.core_timeseries import (
    aggregate_per_flow,
    aggregate_per_party,
    compute_daily_cash_position,
    compute_loans_time,
    expand_party_rows,
)

LOG = logging.getLogger(__name__)


def _materialize_debug_enabled() -> bool:
    return str(os.getenv("MATERIALIZE_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    _ = force
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"per_flow_time_long.freq={freq}.csv"
    target = out_dir / fname

    LOG.info("Materializing per-flow (freq=%s) -> %s", freq, target)
    df = aggregate_per_flow(ledger, freq=freq, amount_col="amount", date_col="Date")

    if "TimePeriod" in df.columns:
        df["TimePeriod"] = df["TimePeriod"].astype(str)
    if "TimePeriod_ts_end" in df.columns:
        df["TimePeriod_ts_end"] = pd.to_datetime(df["TimePeriod_ts_end"], errors="coerce").dt.date.astype(str)

    cols = ["TimePeriod", "TimePeriod_ts_end", "Box", "Flujo", "Tipo", "Currency", "amount", "n_tx"]
    out_df = df[[c for c in cols if c in df.columns]].copy()

    _atomic_write_csv(out_df, target)
    LOG.info("Wrote per_flow rows=%d -> %s", len(out_df), target)
    return out_df, target


def materialize_per_party(
    ledger: pd.DataFrame, out_dir: Path, freq: str = "W", force: bool = False
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce per_party_time_long.freq=<freq>.csv (per-party aggregates).

    Returns (df, path)
    """
    _ = force
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"per_party_time_long.freq={freq}.csv"
    target = out_dir / fname

    LOG.info("Materializing per-party (freq=%s) -> %s", freq, target)

    dbg = _materialize_debug_enabled()
    if dbg:
        LOG.debug("materialize_per_party: ledger shape=%s cols=%s", ledger.shape, list(ledger.columns))

    expanded = expand_party_rows(ledger, amount_col="amount", date_col="Date")
    if dbg:
        LOG.debug("materialize_per_party: expanded shape=%s cols=%s", expanded.shape, list(expanded.columns))

    agg = aggregate_per_party(expanded, freq=freq, amount_col="signed_amount", date_col="Date")
    if dbg:
        LOG.debug("materialize_per_party: agg shape=%s cols=%s", agg.shape, list(agg.columns))

    if "TimePeriod" in agg.columns:
        agg["TimePeriod"] = agg["TimePeriod"].astype(str)
    if "TimePeriod_ts_end" in agg.columns:
        agg["TimePeriod_ts_end"] = pd.to_datetime(agg["TimePeriod_ts_end"], errors="coerce").dt.date.astype(str)

    cols = ["TimePeriod", "TimePeriod_ts_end", "Box", "Flujo", "Tipo", "party", "role", "Currency", "amount", "n_tx"]
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
    _ = force
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "daily_cash_position.csv"

    LOG.info("Materializing daily cash position -> %s", target)
    df = compute_daily_cash_position(ledger, opening_balances=None, freq="D", amount_col="amount", date_col="Date")

    if as_of is not None:
        asof_date = pd.to_datetime(as_of).date()
        df = df[pd.to_datetime(df["Date"]).dt.date <= asof_date].copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype(str)

    out_df = (
        df[["Date", "Box", "party", "balance", "Currency", "source_ledger_hash"]]
        if all(c in df.columns for c in ["Date", "Box", "party", "balance", "Currency", "source_ledger_hash"])
        else df.copy()
    )

    _atomic_write_csv(out_df, target)
    LOG.info("Wrote daily_cash rows=%d -> %s", len(out_df), target)
    return out_df, target

import re

def _infer_box_party_from_box_name(box: str) -> str:
    """
    Heurística de fallback: sigla por iniciales del Box.
    Recomendación: evitar depender de esto y proveer BoxParty explícito.
    """
    if box is None:
        return ""
    b = str(box).strip()
    if not b:
        return ""
    # casos típicos tuyos
    if b.lower() == "household":
        return "HH"
    # iniciales (Family Business -> FB, Property Management -> PM)
    parts = [p for p in re.split(r"\s+", b) if p]
    initials = "".join(p[0].upper() for p in parts if p[0].isalpha())
    return initials



def materialize_box_balance_time_long(
    ledger: pd.DataFrame, out_dir: Path, freq: str = "W", force: bool = False
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce box_balance_time_long.freq=<freq>.csv

    Output columns:
      TimePeriod, Date_end, Box, currency, in_amt, out_amt, net, cum_net

    Semántica:
      - in_amt: suma de amount donde receiver == BoxParty
      - out_amt: suma de amount donde payer == BoxParty
      - net = in_amt - out_amt
      - cum_net: cumsum(net) por (Box, currency) ordenado por Date_end
    """
    _ = force
    ledger = _ensure_amount_float(ledger)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"box_balance_time_long.freq={freq}.csv"

    # Normalizar nombres de columnas
    ldf = ledger.copy()
    if "Currency" not in ldf.columns and "currency" in ldf.columns:
        ldf = ldf.rename(columns={"currency": "Currency"})
    if "amount" not in ldf.columns and "monto" in ldf.columns:
        ldf = ldf.rename(columns={"monto": "amount"})

    required = ["Date", "amount", "Currency", "Box", "payer", "receiver"]
    missing = [c for c in required if c not in ldf.columns]
    if missing:
        raise ValueError(f"materialize_box_balance_time_long: ledger missing required columns: {missing}")

    # BoxParty: prefer explícito
    if "BoxParty" not in ldf.columns:
        ldf["BoxParty"] = ldf["Box"].apply(_infer_box_party_from_box_name)

    # Parseos
    ldf["Date"] = pd.to_datetime(ldf["Date"], errors="coerce")
    ldf = ldf[~ldf["Date"].isna()].copy()
    ldf["amount"] = pd.to_numeric(ldf["amount"], errors="coerce")
    ldf = ldf[~ldf["amount"].isna()].copy()

    # Normalizar strings para matching
    payer = ldf["payer"].astype("string").str.strip().str.upper()
    receiver = ldf["receiver"].astype("string").str.strip().str.upper()
    box_party = ldf["BoxParty"].astype("string").str.strip().str.upper()

    in_mask = receiver == box_party
    out_mask = payer == box_party

    # Si ninguna coincide, esa fila no representa movimiento del Box
    unmatched = ~(in_mask | out_mask)
    if unmatched.any():
        # No rompemos el pipeline, pero esto es señal de gobernanza/datos
        LOG.warning(
            "box_balance: %d row(s) where BoxParty not in payer/receiver. Dropping them from box_balance.",
            int(unmatched.sum()),
        )
        ldf = ldf.loc[~unmatched].copy()
        payer = payer.loc[~unmatched]
        receiver = receiver.loc[~unmatched]
        box_party = box_party.loc[~unmatched]
        in_mask = receiver == box_party
        out_mask = payer == box_party

    # Periodización
    try:
        period = ldf["Date"].dt.to_period(freq)
    except Exception as e:
        raise ValueError(f"materialize_box_balance_time_long: invalid freq={freq!r}: {e}")

    ldf["TimePeriod"] = period.astype(str)
    ldf["Date_end"] = period.dt.end_time.dt.date.astype(str)

    # Flujos
    ldf["in_amt"] = ldf["amount"].where(in_mask, 0.0)
    ldf["out_amt"] = ldf["amount"].where(out_mask, 0.0)
    ldf["net"] = ldf["in_amt"] - ldf["out_amt"]

    agg = (
        ldf.groupby(["TimePeriod", "Date_end", "Box", "Currency"], as_index=False)[["in_amt", "out_amt", "net"]]
        .sum()
        .sort_values(["Box", "Currency", "Date_end", "TimePeriod"])
        .reset_index(drop=True)
    )
    agg["cum_net"] = agg.groupby(["Box", "Currency"])["net"].cumsum()

    out_df = agg.rename(columns={"Currency": "currency"})[
        ["TimePeriod", "Date_end", "Box", "currency", "in_amt", "out_amt", "net", "cum_net"]
    ].copy()

    _atomic_write_csv(out_df, target)
    LOG.info("Wrote box_balance rows=%d -> %s", len(out_df), target)
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
    _ = force
    ledger = _ensure_amount_float(ledger)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"loans_time.freq={freq}.csv"
    target = out_dir / fname

    LOG.info("Materializing loans (freq=%s) -> %s", freq, target)
    df = compute_loans_time(ledger, loan_register_df, freq=freq, amount_col="amount", date_col="Date")

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

      Inputs (conceptually):
        - ledger_canonical.csv (should already exist; treated as input by Stage D)

      Outputs:
        - per_flow_time_long.freq=<freq>.csv
        - per_party_time_long.freq=<freq>.csv
        - loans_time.freq=M.csv              (loans are always monthly)
        - daily_cash_position.csv
        - partitions.json                    (light metadata)

    Returns a dict with aggregate metadata (does NOT write a stage manifest file).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_df = _ensure_amount_float(ledger_df)

    aggregates: Dict[str, Any] = {}

    pf_df: Optional[pd.DataFrame] = None
    pp_df: Optional[pd.DataFrame] = None

    # 0) ledger_canonical.csv: only write if missing, or force=True
    ledger_path = out_dir / "ledger_canonical.csv"
    if force or not ledger_path.exists():
        LOG.info("Writing ledger_canonical -> %s (force=%s)", ledger_path, force)
        ldf = ledger_df.copy()
        if "Date" in ldf.columns:
            ldf["Date"] = pd.to_datetime(ldf["Date"], errors="coerce").dt.date.astype(str)
        _atomic_write_csv(ldf, ledger_path)
    else:
        LOG.info("Keeping existing ledger_canonical -> %s", ledger_path)

    if ledger_path.exists():
        aggregates[ledger_path.name] = {
            "path": str(ledger_path),
            "rows": None,
            "sha256": _sha256_file(ledger_path),
        }

    # 1) per_flow
    try:
        pf_df, pf_path = materialize_per_flow(ledger_df, out_dir, freq=freq, force=force)
        aggregates[pf_path.name] = {"path": str(pf_path), "rows": len(pf_df), "sha256": _sha256_file(pf_path)}
    except Exception:
        LOG.exception("Failed materialize_per_flow")
        aggregates["per_flow_failed"] = {"error": "failed"}

    # 2) per_party
    try:
        pp_df, pp_path = materialize_per_party(ledger_df, out_dir, freq=freq, force=force)
        aggregates[pp_path.name] = {"path": str(pp_path), "rows": len(pp_df), "sha256": _sha256_file(pp_path)}
    except Exception:
        LOG.exception("Failed materialize_per_party")
        aggregates["per_party_failed"] = {"error": "failed"}

    if _materialize_debug_enabled() and pp_df is not None:
        LOG.debug("materialize_all: per_party shape=%s cols=%s", pp_df.shape, list(pp_df.columns))



    # 2.5) box balance
    try:
        bb_df, bb_path = materialize_box_balance_time_long(ledger_df, out_dir, freq=freq, force=force)
        aggregates[bb_path.name] = {"path": str(bb_path), "rows": len(bb_df), "sha256": _sha256_file(bb_path)}
    except Exception:
        LOG.exception("Failed materialize_box_balance_time_long")
        aggregates["box_balance_failed"] = {"error": "failed"}



    # 3) loans (always monthly, independent of pipeline freq)
    loans_freq = "M"
    try:
        loans_df, loans_path = materialize_loans(ledger_df, loan_register_df, out_dir, freq=loans_freq, force=force)
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

    try:
        if pf_df is not None and "TimePeriod_ts_end" in pf_df.columns:
            last_end = pd.to_datetime(pf_df["TimePeriod_ts_end"]).max().date().isoformat()
            parts["last_period_end"] = last_end
    except Exception:
        pass

    parts["outputs"] = {
        k: {"rows": v.get("rows")}
        for k, v in aggregates.items()
        if isinstance(v, dict) and "rows" in v
    }
    save_partitions_json(parts_path, parts)

    # anomalies are an output artifact, not a second manifest system
    anomalies_meta = None
    anomalies = ledger_df.attrs.get("anomalies")
    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        anomalies_path = out_dir / "anomalies.csv"
        _atomic_write_csv(anomalies, anomalies_path)
        anomalies_meta = {
            "path": str(anomalies_path),
            "rows": len(anomalies),
            "sha256": _sha256_file(anomalies_path),
        }

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "aggregates": aggregates,
        "partitions_path": str(parts_path),
        "anomalies": anomalies_meta,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run materialize_all over ledger_canonical.csv")
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "./out"))
    p.add_argument("--freq", default=os.getenv("FREQ", "W"))
    p.add_argument("--force", default=os.getenv("FORCE", "0"))
    p.add_argument("--mode", default=os.getenv("MODE", "run"), choices=["smoke", "run"])
    p.add_argument("--run-id", default=os.getenv("RUN_ID", ""))
    return p.parse_args()


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    return "smoke" if args.mode == "smoke" else ""


def main() -> int:
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
        return 2

    try:
        ledger = pd.read_csv(ledger_path, dtype=str)
    except Exception:
        LOG.exception("Failed reading ledger_canonical.csv")
        return 3

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
    run_id = _resolve_run_id(args)

    # run materialization (no stage-manifest written here)
    result = materialize_all(ledger, out_dir=out_dir, freq=freq, force=force_flag)

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    from accounting.manifest import artifact_from_path, write_stage_manifest, append_artifacts

    stage_generated_at = pd.Timestamp.utcnow().isoformat()

    # inputs
    in_art = artifact_from_path(
        name="ledger_canonical",
        path=ledger_path,
        stage="D.materialize",
        mode=args.mode,
        run_id=run_id,
        role="input",
        root_dir=out_dir,
        content_type="text/csv",
    )

    # outputs esperados
    out_arts = []

    per_flow = out_dir / f"per_flow_time_long.freq={freq}.csv"
    if per_flow.exists():
        out_arts.append(
            artifact_from_path(
                name="per_flow_time_long",
                path=per_flow,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="derived",
                root_dir=out_dir,
                content_type="text/csv",
            )
        )

    per_party = out_dir / f"per_party_time_long.freq={freq}.csv"
    if per_party.exists():
        out_arts.append(
            artifact_from_path(
                name="per_party_time_long",
                path=per_party,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="derived",
                root_dir=out_dir,
                content_type="text/csv",
            )
        )

    # Box balance (per Box, per Currency, per period)
    box_balance = out_dir / f"box_balance_time_long.freq={freq}.csv"
    if box_balance.exists():
        out_arts.append(
            artifact_from_path(
                name="box_balance_time_long",
                path=box_balance,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="derived",
                root_dir=out_dir,
                content_type="text/csv",
            )
        )



    loans = out_dir / "loans_time.freq=M.csv"
    if loans.exists():
        out_arts.append(
            artifact_from_path(
                name="loans_time",
                path=loans,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="derived",
                root_dir=out_dir,
                content_type="text/csv",
            )
        )

    daily_cash = out_dir / "daily_cash_position.csv"
    if daily_cash.exists():
        out_arts.append(
            artifact_from_path(
                name="daily_cash_position",
                path=daily_cash,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="derived",
                root_dir=out_dir,
                content_type="text/csv",
            )
        )

    parts = out_dir / "partitions.json"
    if parts.exists():
        out_arts.append(
            artifact_from_path(
                name="partitions",
                path=parts,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="meta",
                root_dir=out_dir,
                content_type="application/json",
            )
        )

    anomalies = out_dir / "anomalies.csv"
    if anomalies.exists():
        out_arts.append(
            artifact_from_path(
                name="anomalies",
                path=anomalies,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="derived",
                root_dir=out_dir,
                content_type="text/csv",
            )
        )

    stage_manifest = {
        "generated_at": stage_generated_at,  # fixes created_at=None downstream
        "stage": "D.materialize",
        "mode": args.mode,
        "run_id": run_id,
        "inputs": [in_art],
        "params": {"freq": freq, "force": int(force_flag)},
        "outputs": out_arts,
        "warnings": [],
        # optional: keep the returned summary (small + useful)
        "result": {
            "generated_at": result.get("generated_at"),
            "aggregates": result.get("aggregates", {}),
            "partitions_path": result.get("partitions_path"),
            "anomalies": result.get("anomalies"),
        },
    }

    stage_manifest_rel = write_stage_manifest(meta_dir, stage_manifest)

    stage_meta_path = out_dir / stage_manifest_rel
    stage_meta_sha = artifact_from_path(
        name="stage_D_materialize",
        path=stage_meta_path,
        stage="D.materialize",
        mode=args.mode,
        run_id=run_id,
        role="meta",
        root_dir=out_dir,
        content_type="application/json",
    )["sha256"]

    stage_meta_art = {
        "run_id": run_id,
        "stage": "D.materialize",
        "mode": args.mode,
        "name": "stage_D_materialize",
        "relpath": stage_manifest_rel,
        "sha256": stage_meta_sha,
        "bytes": stage_meta_path.stat().st_size,
        "rows": None,
        "content_type": "application/json",
        "created_at": stage_manifest.get("generated_at"),
        "role": "meta",
    }

    append_artifacts(meta_dir, [in_art, *out_arts, stage_meta_art])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
