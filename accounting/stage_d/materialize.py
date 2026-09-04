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
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from accounting.box_cash import box_party_match_masks, infer_box_party
from accounting.marts.cash import build_monthly_cash_close
from accounting.marts.semantic import build_monthly_operating_statement, build_semantic_outputs
from accounting.scope import assert_frame_within_scope, load_run_scope, load_run_scope_if_present
from accounting.core.timeseries import (
    aggregate_per_flow,
    aggregate_per_party,
    compute_daily_cash_position,
    compute_loans_time,
    expand_party_rows,
)

from accounting.logging_utils import configure_logging, get_logger
from accounting.support.hashing import sha256_file
from accounting.support.io import atomic_write_df
from accounting.support.partitions import load_partitions_json, save_partitions_json
LOG = get_logger("materialize")


def _analytical_ledger(ledger: pd.DataFrame, *, include_household: bool = True, exclude_legacy: bool = True) -> pd.DataFrame:
    """Exclude audit-only inferred residuals from every analytical motor."""
    legacy = (
        ledger["tag"].astype(str).str.strip().str.casefold().eq("legacy_inferred_net")
        if "tag" in ledger.columns
        else pd.Series(False, index=ledger.index)
    )
    eligible = ~legacy if exclude_legacy else pd.Series(True, index=ledger.index)
    # The owning Box governs scope. Household may remain a counterparty or
    # stakeholder actor on an in-scope PM/FB transaction.
    out = ledger.loc[eligible].copy()
    if "anomalies" in ledger.attrs:
        out.attrs["anomalies"] = ledger.attrs["anomalies"]
    return out


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



def _safe_sha256(path: Path) -> Optional[str]:
    """Keep Stage D's historical fail-soft metadata behavior over shared hashing."""
    try:
        return sha256_file(path)
    except Exception:
        LOG.exception("Failed to hash file: %s", path)
        return None


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

    cols = ["TimePeriod", "TimePeriod_ts_end", "Box", "Currency", "Flujo", "Tipo", "amount", "n_tx"]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"per_flow_time_long missing columns from aggregate_per_flow: {missing_cols}. "
                       f"Available={list(df.columns)}")
    out_df = df[cols].copy()

    atomic_write_df(out_df, target, index=False)
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
        agg["TimePeriod_end"] = pd.to_datetime(agg["TimePeriod_ts_end"], errors="coerce").dt.date.astype(str)

    cols = ["TimePeriod", "TimePeriod_end", "Box", "Currency", "party", "role", "Flujo", "Tipo", "amount", "n_tx"]
    missing_cols = [c for c in cols if c not in agg.columns]
    if missing_cols:
        raise KeyError(f"per_party_time_long missing columns from aggregate_per_party: {missing_cols}. "
                       f"Available={list(agg.columns)}")
    out_df = agg[cols].copy()

    atomic_write_df(out_df, target, index=False)
    LOG.info("Wrote per_party rows=%d -> %s", len(out_df), target)
    return out_df, target


def materialize_daily_cash(
    ledger: pd.DataFrame,
    out_dir: Path,
    as_of: Optional[pd.Timestamp] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, Path]:
    _ = force
    ledger = _ensure_amount_float(ledger)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "daily_cash_position.csv"

    LOG.info("Materializing daily cash position -> %s", target)

    df = compute_daily_cash_position(ledger, freq="D", amount_col="amount", date_col="Date")

    if as_of is not None:
        asof_date = pd.to_datetime(as_of).date()
        df = df[pd.to_datetime(df["Date"]).dt.date <= asof_date].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype(str)

    required = ["Date", "Box", "party", "Currency", "balance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"daily_cash_position missing columns: {missing}")

    out_df = df[required].copy()
    # Keep provenance hash if available (optional)
    if "source_ledger_hash" in df.columns:
        out_df["source_ledger_hash"] = df["source_ledger_hash"]
    atomic_write_df(out_df, target, index=False)
    LOG.info("Wrote daily_cash rows=%d -> %s", len(out_df), target)
    return out_df, target



def materialize_box_balance_time_long(
    ledger: pd.DataFrame, out_dir: Path, freq: str = "W", force: bool = False
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce box_balance_time_long.freq=<freq>.csv

    Output columns:
      TimePeriod, TimePeriod_end, Box, Currency, in_amt, out_amt, net, cum_net

    Semántica:
      - in_amt: suma de amount donde receiver == BoxParty
      - out_amt: suma de amount donde payer == BoxParty
      - net = in_amt - out_amt
      - cum_net: cumsum(net) por (Box, Currency) ordenado por TimePeriod_end
    """
    _ = force
    ledger = _ensure_amount_float(ledger)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"box_balance_time_long.freq={freq}.csv"

    # Normalizar nombres de columnas
    ldf = ledger.copy()
    # if "Currency" not in ldf.columns and "currency" in ldf.columns:
    #     ldf = ldf.rename(columns={"currency": "Currency"})
    if "amount" not in ldf.columns and "monto" in ldf.columns:
        ldf = ldf.rename(columns={"monto": "amount"})

    required = ["Date", "amount", "Currency", "Box", "payer", "receiver"]
    missing = [c for c in required if c not in ldf.columns]
    if missing:
        raise ValueError(f"materialize_box_balance_time_long: ledger missing required columns: {missing}")

    ldf["BoxParty"] = ldf["Box"].map(infer_box_party)

    # # BoxParty is required to compute box-level motor flows correctly.
    # # We allow an explicit, opt-in fallback heuristic only for transition periods.
    # if "BoxParty" not in ldf.columns:
    #     if str(os.getenv("BOXPARTY_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}:
    #         LOG.warning("BOXPARTY_FALLBACK enabled: inferring BoxParty from Box names. This is not decision-grade.")
    #         ldf["BoxParty"] = ldf["Box"].map(infer_box_party)
    #     else:
    #         raise ValueError(
    #             "materialize_box_balance_time_long requires column 'BoxParty' in ledger. "
    #             "Add it upstream (ingest / canonical mapping). "
    #             "If you must use a heuristic temporarily, set BOXPARTY_FALLBACK=1."
    #         )

    # Parseos
    ldf["Date"] = pd.to_datetime(ldf["Date"], errors="coerce")
    ldf = ldf[~ldf["Date"].isna()].copy()
    ldf["amount"] = pd.to_numeric(ldf["amount"], errors="coerce")
    ldf = ldf[~ldf["amount"].isna()].copy()

    # Normalizar strings para matching
    in_mask, out_mask = box_party_match_masks(ldf)

    # Si ninguna coincide, esa fila no representa movimiento del Box
    unmatched = ~(in_mask | out_mask)
    if unmatched.any():
        # No rompemos el pipeline, pero esto es señal de gobernanza/datos
        LOG.warning(
            "box_balance: %d row(s) where BoxParty not in payer/receiver. Dropping them from box_balance.",
            int(unmatched.sum()),
        )
        ldf = ldf.loc[~unmatched].copy()
        in_mask, out_mask = box_party_match_masks(ldf)

    # Periodización
    try:
        period = ldf["Date"].dt.to_period(freq)
    except Exception as e:
        raise ValueError(f"materialize_box_balance_time_long: invalid freq={freq!r}: {e}")

    ldf["TimePeriod"] = period.astype(str)
    ldf["TimePeriod_end"] = period.dt.end_time.dt.date.astype(str)

    # Flujos
    ldf["in_amt"] = ldf["amount"].where(in_mask, 0.0)
    ldf["out_amt"] = ldf["amount"].where(out_mask, 0.0)
    ldf["net"] = ldf["in_amt"] - ldf["out_amt"]

    agg = (
        ldf.groupby(["TimePeriod", "TimePeriod_end", "Box", "Currency"], as_index=False)[["in_amt", "out_amt", "net"]]
        .sum()
        .sort_values(["Box", "Currency", "TimePeriod_end", "TimePeriod"])
        .reset_index(drop=True)
    )
    agg["cum_net"] = agg.groupby(["Box", "Currency"])["net"].cumsum()

    out_df = agg[["TimePeriod", "TimePeriod_end", "Box", "Currency", "in_amt", "out_amt", "net", "cum_net"]
    ].copy()

    atomic_write_df(out_df, target, index=False)
    LOG.info("Wrote box_balance rows=%d -> %s", len(out_df), target)
    return out_df, target


BOXPARTY_FALLBACK=1

def materialize_box_flow_balance_time_long(
    ledger: pd.DataFrame, out_dir: Path, freq: str = "W", force: bool = False
) -> Tuple[pd.DataFrame, Path]:
    """
    Produce box_flow_balance_time_long.freq=<freq>.csv

    This is a generic, narrative-agnostic decomposition of the Box motor by (Flujo, Tipo).

    Grain:
      TimePeriod, TimePeriod_end, Box, Currency, Flujo, Tipo

    Measures:
      in_amt, out_amt, net, n_tx

    Semantics:
      - in_amt: sum(amount) where receiver == BoxParty
      - out_amt: sum(amount) where payer == BoxParty
      - net = in_amt - out_amt
      - n_tx: count of ledger rows contributing to the group (after BoxParty match filter)
    """
    _ = force
    ledger = _ensure_amount_float(ledger)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"box_flow_balance_time_long.freq={freq}.csv"

    ldf = ledger.copy()
    if "amount" not in ldf.columns and "monto" in ldf.columns:
        ldf = ldf.rename(columns={"monto": "amount"})

    required = ["Date", "amount", "Currency", "Box", "payer", "receiver", "Flujo", "Tipo"]
    missing = [c for c in required if c not in ldf.columns]
    if missing:
        raise ValueError(f"materialize_box_flow_balance_time_long: ledger missing required columns: {missing}")

    ldf["BoxParty"] = ldf["Box"].map(infer_box_party)

    # # BoxParty required (opt-in heuristic fallback only)
    # if "BoxParty" not in ldf.columns:
    #     if str(os.getenv("BOXPARTY_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}:
    #         LOG.warning("BOXPARTY_FALLBACK enabled: inferring BoxParty from Box names. This is not decision-grade.")
    #         ldf["BoxParty"] = ldf["Box"].map(infer_box_party)
    #     else:
    #         raise ValueError(
    #             "materialize_box_flow_balance_time_long requires column 'BoxParty' in ledger. "
    #             "Add it upstream (ingest / canonical mapping). "
    #             "If you must use a heuristic temporarily, set BOXPARTY_FALLBACK=1."
    #         )

    # Parse/coerce
    ldf["Date"] = pd.to_datetime(ldf["Date"], errors="coerce")
    ldf = ldf[~ldf["Date"].isna()].copy()
    ldf["amount"] = pd.to_numeric(ldf["amount"], errors="coerce")
    ldf = ldf[~ldf["amount"].isna()].copy()

    # Normalize strings for matching
    in_mask, out_mask = box_party_match_masks(ldf)

    unmatched = ~(in_mask | out_mask)
    if unmatched.any():
        LOG.warning(
            "box_flow_balance: %d row(s) where BoxParty not in payer/receiver. Dropping them from box_flow_balance.",
            int(unmatched.sum()),
        )
        ldf = ldf.loc[~unmatched].copy()
        in_mask, out_mask = box_party_match_masks(ldf)

    # Periodization
    try:
        period = ldf["Date"].dt.to_period(freq)
    except Exception as e:
        raise ValueError(f"materialize_box_flow_balance_time_long: invalid freq={freq!r}: {e}")

    ldf["TimePeriod"] = period.astype(str)
    ldf["TimePeriod_end"] = period.dt.end_time.dt.date.astype(str)

    # Motor flows per row
    ldf["in_amt"] = ldf["amount"].where(in_mask, 0.0)
    ldf["out_amt"] = ldf["amount"].where(out_mask, 0.0)
    ldf["net"] = ldf["in_amt"] - ldf["out_amt"]
    ldf["n_tx"] = 1

    agg = (
        ldf.groupby(["TimePeriod", "TimePeriod_end", "Box", "Currency", "Flujo", "Tipo"], as_index=False)[
            ["in_amt", "out_amt", "net", "n_tx"]
        ]
        .sum()
        .sort_values(["Box", "Currency", "TimePeriod_end", "TimePeriod", "Flujo", "Tipo"])
        .reset_index(drop=True)
    )

    out_df = agg[["TimePeriod", "TimePeriod_end", "Box", "Currency", "Flujo", "Tipo", "in_amt", "out_amt", "net", "n_tx"]].copy()

    atomic_write_df(out_df, target, index=False)
    LOG.info("Wrote box_flow_balance rows=%d -> %s", len(out_df), target)
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
    atomic_write_df(out_df, target, index=False)
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
        - box_balance_time_long.freq=<freq>.csv
        - box_flow_balance_time_long.freq=<freq>.csv
        - loans_time.freq=M.csv              (loans are always monthly)
        - daily_cash_position.csv

        - partitions.json                    (light metadata)

    Returns a dict with aggregate metadata (does NOT write a stage manifest file).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_ledger_df = _ensure_amount_float(ledger_df)
    run_scope = load_run_scope_if_present(out_dir)
    ledger_df = _analytical_ledger(
        source_ledger_df,
        include_household=run_scope is None or "Household" in run_scope.boxes,
    )
    if run_scope is not None:
        assert_frame_within_scope(
            ledger_df, run_scope, source="ledger_canonical.csv", require_box=True
        )

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
        atomic_write_df(ldf, ledger_path, index=False)
    else:
        LOG.info("Keeping existing ledger_canonical -> %s", ledger_path)

    if ledger_path.exists():
        aggregates[ledger_path.name] = {
            "path": str(ledger_path),
            "rows": None,
            "sha256": _safe_sha256(ledger_path),
        }

    # 1) per_flow
    try:
        pf_df, pf_path = materialize_per_flow(ledger_df, out_dir, freq=freq, force=force)
        aggregates[pf_path.name] = {"path": str(pf_path), "rows": len(pf_df), "sha256": _safe_sha256(pf_path)}
    except Exception:
        LOG.exception("Failed materialize_per_flow")
        aggregates["per_flow_failed"] = {"error": "failed"}

    # 2) per_party
    try:
        pp_df, pp_path = materialize_per_party(ledger_df, out_dir, freq=freq, force=force)
        aggregates[pp_path.name] = {"path": str(pp_path), "rows": len(pp_df), "sha256": _safe_sha256(pp_path)}
    except Exception:
        LOG.exception("Failed materialize_per_party")
        aggregates["per_party_failed"] = {"error": "failed"}

    if _materialize_debug_enabled() and pp_df is not None:
        LOG.debug("materialize_all: per_party shape=%s cols=%s", pp_df.shape, list(pp_df.columns))



    # 2.5) box balance (Box motor)
    try:
        bb_df, bb_path = materialize_box_balance_time_long(ledger_df, out_dir, freq=freq, force=force)
        aggregates[bb_path.name] = {"path": str(bb_path), "rows": len(bb_df), "sha256": _safe_sha256(bb_path)}
    except Exception:
        LOG.exception("Failed materialize_box_balance_time_long")
        aggregates["box_balance_failed"] = {"error": "failed"}

    # 2.6) box flow balance (motor decomposition by Flujo/Tipo)
    try:
        bfb_df, bfb_path = materialize_box_flow_balance_time_long(ledger_df, out_dir, freq=freq, force=force)
        aggregates[bfb_path.name] = {"path": str(bfb_path), "rows": len(bfb_df), "sha256": _safe_sha256(bfb_path)}
    except Exception:
        LOG.exception("Failed materialize_box_flow_balance_time_long")
        aggregates["box_flow_balance_failed"] = {"error": "failed"}



    # 3) loans (always monthly, independent of pipeline freq)
    loans_freq = "M"
    try:
        loans_df, loans_path = materialize_loans(ledger_df, loan_register_df, out_dir, freq=loans_freq, force=force)
        aggregates[loans_path.name] = {"path": str(loans_path), "rows": len(loans_df), "sha256": _safe_sha256(loans_path)}
    except Exception:
        LOG.exception("Failed materialize_loans")
        aggregates["loans_failed"] = {"error": "failed"}

    # 4) daily cash
    try:
        dc_df, dc_path = materialize_daily_cash(ledger_df, out_dir, as_of=None, force=force)
        aggregates[dc_path.name] = {"path": str(dc_path), "rows": len(dc_df), "sha256": _safe_sha256(dc_path)}
    except Exception:
        LOG.exception("Failed materialize_daily_cash")
        aggregates["daily_cash_failed"] = {"error": "failed"}

    # 4.1) monthly cash consumption wrapper with explicit suitability flags
    try:
        cash_paths = build_monthly_cash_close(out_dir=out_dir, freq=freq)
        for name, path in cash_paths.items():
            aggregates[path.name] = {"path": str(path), "rows": None, "sha256": _safe_sha256(path)}
    except Exception:
        LOG.exception("Failed monthly cash close build")
        raise

    # 4.5) conservative semantic classification mart and monthly operating statement
    try:
        semantic_source = _analytical_ledger(
            source_ledger_df,
            include_household=run_scope is None or "Household" in run_scope.boxes,
            exclude_legacy=False,
        )
        semantic_paths = build_semantic_outputs(semantic_source, out_dir=out_dir, freq="M")
        operating_statement_paths = build_monthly_operating_statement(out_dir=out_dir)
        for name, path in {**semantic_paths, **operating_statement_paths}.items():
            aggregates[path.name] = {"path": str(path), "rows": None, "sha256": _safe_sha256(path)}
    except Exception:
        LOG.exception("Failed semantic mart / monthly operating statement build")
        raise

    # 5) partitions.json update (simple)
    parts_path = out_dir / "partitions.json"
    parts = load_partitions_json(parts_path)
    parts["last_materialized_at"] = pd.Timestamp.now("UTC").isoformat()
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
        atomic_write_df(anomalies, anomalies_path, index=False)
        anomalies_meta = {
            "path": str(anomalies_path),
            "rows": len(anomalies),
            "sha256": _safe_sha256(anomalies_path),
        }

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
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


def main() -> int:
    configure_logging()

    args = parse_args()
    out_dir = Path(args.out_dir)
    ledger_path = out_dir / "ledger_canonical.csv"
    LOG.info("Stage start mode=%s out_dir=%s freq=%s", args.mode, out_dir, args.freq)
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

    # Consequential runs must carry the immutable Stage A contract. Legacy
    # smoke fixtures without Box metadata remain supported as unscoped checks.
    if args.mode == "run":
        load_run_scope(out_dir)

    from accounting.support.run_id import resolve_run_id

    # run_id = _resolve_run_id(args)
    run_id = resolve_run_id(mode=args.mode, run_id=getattr(args, "run_id", None), root_dir=out_dir, strict=True)

    # run materialization (no stage-manifest written here)
    result = materialize_all(
        ledger,
        out_dir=out_dir,
        freq=freq,
        force=force_flag,
    )
    all_status_path = out_dir / "ledger_canonical_all_status.csv"
    if all_status_path.exists():
        from accounting.marts.semantic import build_cost_allocation_gap_outputs

        all_status = pd.read_csv(all_status_path, dtype=str)
        gap_paths = build_cost_allocation_gap_outputs(all_status, out_dir)
        for name, path in gap_paths.items():
            result.setdefault("aggregates", {})[path.name] = {
                "path": str(path),
                "rows": len(pd.read_csv(path)),
                "sha256": _safe_sha256(path),
            }
    aggregate_rows = {k: v.get("rows") for k, v in result.get("aggregates", {}).items() if isinstance(v, dict) and "rows" in v}

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    from accounting.artifacts.manifest import (
        append_artifacts,
        artifact_from_path,
        write_artifact_contract_qa,
        write_artifact_contracts_csv,
        write_stage_manifest,
    )

    stage_generated_at = pd.Timestamp.now("UTC").isoformat()

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

    # Stable Stage D artifact inventory. The manifest authority remains
    # accounting.artifacts.manifest; Stage D only declares expected paths.
    out_arts = []
    output_specs = [
        ("per_flow_time_long", out_dir / f"per_flow_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("per_party_time_long", out_dir / f"per_party_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("box_balance_time_long", out_dir / f"box_balance_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("box_flow_balance_time_long", out_dir / f"box_flow_balance_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("loans_time", out_dir / "loans_time.freq=M.csv", "derived", "text/csv"),
        ("daily_cash_position", out_dir / "daily_cash_position.csv", "derived", "text/csv"),
        ("partitions", out_dir / "partitions.json", "meta", "application/json"),
        ("anomalies", out_dir / "anomalies.csv", "derived", "text/csv"),
    ]
    for name, path, role, content_type in output_specs:
        if path.exists():
            out_arts.append(
                artifact_from_path(
                    name=name,
                    path=path,
                    stage="D.materialize",
                    mode=args.mode,
                    run_id=run_id,
                    role=role,
                    root_dir=out_dir,
                    content_type=content_type,
                )
            )

    known_relpaths = {art["relpath"] for art in [in_art, *out_arts]}
    for filename, meta in sorted(result.get("aggregates", {}).items()):
        if not isinstance(meta, dict) or not meta.get("path"):
            continue
        path = Path(meta["path"])
        if not path.exists():
            continue
        relpath = str(path.resolve().relative_to(out_dir.resolve()))
        if relpath in known_relpaths:
            continue
        out_arts.append(
            artifact_from_path(
                name=path.stem,
                path=path,
                stage="D.materialize",
                mode=args.mode,
                run_id=run_id,
                role="derived",
                root_dir=out_dir,
                content_type="text/csv" if path.suffix.lower() == ".csv" else None,
            )
        )
        known_relpaths.add(relpath)

    contract_path = write_artifact_contracts_csv(out_dir / "artifact_contracts.csv", [in_art, *out_arts])
    qa_path = write_artifact_contract_qa(out_dir / "artifact_contract_qa.csv", [in_art, *out_arts])
    for contract_artifact, contract_name in [(contract_path, "artifact_contracts"), (qa_path, "artifact_contract_qa")]:
        out_arts.append(
            artifact_from_path(
                name=contract_name,
                path=contract_artifact,
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
    stage_meta_art = artifact_from_path(
        name="stage_D_materialize",
        path=stage_meta_path,
        stage="D.materialize",
        mode=args.mode,
        run_id=run_id,
        role="meta",
        root_dir=out_dir,
        content_type="application/json",
    )

    append_artifacts(meta_dir, [in_art, *out_arts, stage_meta_art])
    LOG.info("Stage finish outputs=%s partitions=%s", aggregate_rows, result.get("partitions_path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
