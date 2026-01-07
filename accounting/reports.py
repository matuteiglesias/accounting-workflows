# src/accounting/reports.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence

import re
import pandas as pd
import numpy as np

from accounting.utils import atomic_write_df, sha256_file

import logging, sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ---------------------------------------------------------------------
# IO helpers (CSV-only, amounts in currency units)
# ---------------------------------------------------------------------
def _safe_read_csv(p: Path) -> pd.DataFrame:
    p = Path(p)
    if not p.exists():
        return pd.DataFrame()
    # read with low_memory False to avoid dtype surprises
    return pd.read_csv(p, dtype=object, low_memory=False)

def load_materialized_folder(out_dir: Path, freq: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Read canonical materialized CSV files from out_dir and return a dict:
      {
        "per_party_time_long": DataFrame,
        "per_flow_time_long": DataFrame,
        "loans_time": DataFrame,
        "daily_cash_position": DataFrame,
        "ledger_canonical": DataFrame
      }

    Filenames expected (freq-aware):
      per_party_time_long.freq={freq}.csv
      per_flow_time_long.freq={freq}.csv
      loans_time.freq={freq}.csv
      daily_cash_position.csv
      ledger_canonical.csv
    """
    out = Path(out_dir)
    suffix = f".freq={freq}.csv" if freq else ".csv"
    data = {}
    data["per_party_time_long"] = _safe_read_csv(out / (f"per_party_time_long{suffix}" if freq else "per_party_time_long.csv"))
    data["per_flow_time_long"] = _safe_read_csv(out / (f"per_flow_time_long{suffix}" if freq else "per_flow_time_long.csv"))
    data["loans_time"] = _safe_read_csv(out / (f"loans_time{suffix}" if freq else "loans_time.csv"))
    data["daily_cash_position"] = _safe_read_csv(out / "daily_cash_position.csv")
    data["ledger_canonical"] = _safe_read_csv(out / "ledger_canonical.csv")
    return data

# ---------------------------------------------------------------------
# TimePeriod helpers
# ---------------------------------------------------------------------
def _period_to_timestamp_end(df: pd.DataFrame, period_col: str = "TimePeriod", tscol: str = "TimePeriod_ts_end") -> pd.DatetimeIndex:
    """
    Robust conversion of period-like columns to timestamp (period end).
    Accepts:
      - explicit TimePeriod_ts_end column (ISO datelike strings)
      - TimePeriod that may be Periods, strings, or other (best-effort)
    Returns a DatetimeIndex aligned to df (same length).
    """
    if tscol in df.columns:
        return pd.to_datetime(df[tscol], errors="coerce")
    if period_col not in df.columns:
        return pd.DatetimeIndex([pd.NaT] * len(df))
    s = df[period_col]

    # If already datetime dtype
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    # If Series of Period-like objects: use .dt when available
    try:
        if hasattr(s.dt, "to_timestamp"):
            return s.dt.to_timestamp(how="end")
    except Exception:
        pass

    # Try coercion to PeriodIndex (best-effort)
    try:
        pidx = pd.PeriodIndex(s.astype(str), freq=None)
        return pidx.to_timestamp(how="end")
    except Exception:
        # fallback: try parsing as datetimes directly
        return pd.to_datetime(s.astype(str), errors="coerce")

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _ensure_amount_series(df: pd.DataFrame, col: str = "amount") -> pd.Series:
    """
    Return a numeric Series in currency units (float) for column `col`.
    If column missing, returns zeros aligned with df.index.
    """
    if df is None or df.empty:
        return pd.Series([], dtype=float)
    if col in df.columns:
        ser = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    else:
        ser = pd.Series(0.0, index=df.index, dtype=float)
    return ser

def _sanitize_name(s: str) -> str:
    """Make a filesystem/key-safe single-token name from arbitrary string."""
    if pd.isna(s):
        return "NA"
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_\-]", "_", s)
    return s

# ---------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------
def validate_materialization_totals(ledger_df: pd.DataFrame, materialized: Dict[str, pd.DataFrame]) -> Dict:
    """
    Check invariants. Compare sum(amount) in ledger vs per_flow_time_long.
    Returns dict with keys: ok, ledger_sum, materialized_sum, diff, anomalies.
    Amounts are expected in currency units (floats).
    """
    ledger_sum = 0.0
    if ledger_df is not None and not ledger_df.empty:
        ledger_amt = _ensure_amount_series(ledger_df, "amount")
        ledger_sum = float(ledger_amt.sum())

    pf = materialized.get("per_flow_time_long")
    mat_sum = 0.0
    if pf is not None and not pf.empty:
        pf_amt = _ensure_amount_series(pf, "amount")
        mat_sum = float(pf_amt.sum())

    diff = ledger_sum - mat_sum
    ok = abs(diff) < 1e-6  # tolerate tiny float rounding
    anomalies = []
    if not ok:
        anomalies.append({"reason": "sum_mismatch", "ledger_sum": ledger_sum, "materialized_sum": mat_sum, "diff": diff})
    return {"ok": ok, "ledger_sum": ledger_sum, "materialized_sum": mat_sum, "diff": diff, "anomalies": anomalies}

# ---------------------------------------------------------------------
# Reporting primitives (thin, read-only) - amounts in currency units
# ---------------------------------------------------------------------
def build_fondos_report_from_materialized(materialized: Dict[str, pd.DataFrame], parties: Sequence[str], freq: str) -> pd.DataFrame:
    """
    Return pivot table indexed by Date (period end timestamp) with columns like:
      'Credit PM_Cobros', 'Debit PM_Repago', etc.
    Values are in major currency units (float).
    """
    pp = materialized.get("per_party_time_long")
    if pp is None or pp.empty:
        return pd.DataFrame()
    
    print(pp.columns)

    df = pp.copy()

    # index date
    df["Date"] = _period_to_timestamp_end(df)

    # normalize amount (units, float)
    df["amount"] = _ensure_amount_series(df, "amount")

    print(df.columns)

    # ensure currency exists and normalize
    if "Currency" not in df.columns:   ## Here 1st
        df["Currency"] = "NA"
        print("NaN at position 4")
    else:
        print("Ok at position 5")
        df["Currency"] = df["Currency"].astype(str).str.upper().fillna("NA")

    # side and colname (sanitize party and Flujo)
    df["side"] = np.where(df.get("role", "") == "receiver", "Credit", "Debit")
    df = df[df["party"].isin(parties)].copy()
    df["party_s"] = df["party"].apply(_sanitize_name)
    df["flujo_s"] = df["Flujo"].astype(str).apply(_sanitize_name)
    df["colname"] = df["side"] + " " + df["party_s"] + "_" + df["flujo_s"]

    # group by Date x colname x currency, then unstack both colname and currency to form columns
    agg = (
        df.groupby(["Date", "colname", "Currency"])["amount"]
        .sum()
        .unstack(["colname", "Currency"])
        .fillna(0)
        .sort_index()
    )

    # flatten multiindex columns to "colname|CURRENCY" for CSV friendliness
    if isinstance(agg.columns, pd.MultiIndex):
        agg.columns = [f"{col}|{cur}" for col, cur in agg.columns]
    else:
        # if only one level (unlikely), keep as-is
        agg.columns = [str(c) for c in agg.columns]

    print(f"{agg.columns}")

    return agg.astype(float)



def build_renta_series_from_materialized(materialized: Dict[str, pd.DataFrame], target: str, freq: str) -> pd.DataFrame:
    """
    Return renta series for a given actor as a DataFrame with columns per currency.
    Index is timestamp end of period. Columns are currencies (e.g., 'ARS','USD').
    If only one currency present, the DataFrame will have a single column.
    """
    pp = materialized.get("per_party_time_long")
    if pp is None or pp.empty:
        return pd.DataFrame(dtype=float)

    df = pp.copy()
    df = df[df["party"] == target]
    if df.empty:
        return pd.DataFrame(dtype=float)

    # ensure currency present
    if "Currency" not in df.columns:
        print("NaN at position 1")   ### Here 2nd
        df["Currency"] = "NA"
    else:
        print("Ok at position 2")

        df["Currency"] = df["Currency"].astype(str).str.upper().fillna("NA")

    df["amount"] = _ensure_amount_series(df, "amount")
    idx = _period_to_timestamp_end(df)

    # group by period end and currency, return unstacked DataFrame (columns are currencies)
    s = df.groupby([idx, "Currency"])["amount"].sum().unstack("Currency").fillna(0).sort_index()
    # ensure floats
    s = s.astype(float)
    s.index.name = "Date"
    s.name = f"Renta_{_sanitize_name(target)}"
    return s

def compute_transaction_series_from_aggregates(filter_mask_df: pd.DataFrame, cols: Sequence[str], cumulative: bool = False) -> pd.DataFrame:
    """
    Produce a time-series from a filtered view of a materialized table (e.g., per_party or per_flow).
    filter_mask_df must include a TimePeriod or Date-like column.
    cols are column names available in filter_mask_df to sum.
    Returns aggregated DF indexed by Date (timestamp end if TimePeriod present).
    """
    if filter_mask_df is None or filter_mask_df.empty:
        return pd.DataFrame()
    df = filter_mask_df.copy()

    # get index
    if "TimePeriod_ts_end" in df.columns:
        df["Date"] = pd.to_datetime(df["TimePeriod_ts_end"], errors="coerce")
    elif "TimePeriod" in df.columns:
        df["Date"] = _period_to_timestamp_end(df)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise KeyError("filter_mask_df must contain TimePeriod, TimePeriod_ts_end or Date")

    # ensure numeric for requested columns (fall back to 0)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    out = df.groupby("Date")[list(cols)].sum().sort_index()
    if cumulative:
        out = out.cumsum()
    return out

def materialized_daily_cash_position_as_of(materialized: Dict[str, pd.DataFrame], as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Return daily cash snapshot per party as of `as_of`.
    If materialized daily_cash_position exists, use it. Otherwise derive from per_party_time_long.
    Output columns: Date, party, balance (float)
    """
    daily = materialized.get("daily_cash_position")
    if daily is not None and not daily.empty:
        d = daily.copy()
        # support either as_of or Date column naming
        if "as_of" in d.columns:
            d["Date"] = pd.to_datetime(d["as_of"], errors="coerce")
        elif "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        else:
            d["Date"] = pd.NaT
        # normalize balance column name if present (balance or balance_cents previously)
        if "balance" not in d.columns and "balance_cents" in d.columns:
            # assume balance_cents not used in new format, but keep fallback: divide by 100 if present
            d["balance"] = pd.to_numeric(d["balance_cents"], errors="coerce").fillna(0.0).astype(float) / 100.0
        elif "balance" in d.columns:
            d["balance"] = pd.to_numeric(d["balance"], errors="coerce").fillna(0.0).astype(float)
        else:
            # no balance column - keep as is and user must derive
            pass

        if as_of is not None:
            return d.loc[d["Date"] <= as_of].sort_values(["party", "Date"])
        return d.sort_values(["party", "Date"])

    # fallback derive from per_party_time_long
    pp = materialized.get("per_party_time_long")
    if pp is None or pp.empty:
        return pd.DataFrame()
    df = pp.copy()
    df["amount"] = _ensure_amount_series(df, "amount")
    # signed amount: receiver => +, else - (business rule)
    df["signed"] = np.where(df.get("role", "") == "receiver", df["amount"], -df["amount"])

    if "TimePeriod_ts_end" in df.columns:
        df["Date"] = pd.to_datetime(df["TimePeriod_ts_end"], errors="coerce")
    else:
        df["Date"] = _period_to_timestamp_end(df)

    # group by Date x party, then cumulative sum across time (sorted by date)
    agg = df.groupby(["Date", "party"])["signed"].sum().unstack(fill_value=0).sort_index().cumsum()
    melted = agg.reset_index().melt(id_vars=["Date"], var_name="party", value_name="balance")
    return melted

def rolling_forecast_90d_from_daily(daily_df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Baseline forecast. Returns DataFrame with Date, party, forecast_balance.
    Uses simple mean daily change over a 30-day lookback (or less if not available).
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    out_frames = []
    parties = df["party"].unique()
    today = df["Date"].max()

    for p in parties:
        sub = df.loc[df["party"] == p].sort_values("Date")
        sub = sub.set_index("Date")["balance"].astype(float)
        if len(sub) == 0:
            continue
        diffs = sub.diff().dropna()
        lookback = min(len(diffs), 30)
        mean_daily = float(diffs.iloc[-lookback:].mean()) if lookback > 0 else 0.0
        idx = pd.date_range(start=today + pd.Timedelta(days=1), periods=days, freq="D")
        last = float(sub.iloc[-1])
        forecast = last + mean_daily * (np.arange(1, days + 1))
        out = pd.DataFrame({"Date": idx, "party": p, "forecast_balance": forecast})
        out_frames.append(out)
    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)

def upcoming_payables_receivables(ledger_df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Filter ledger_canonical for transactions between today and today+days and not marked 'pagado'.
    Returns rows sorted by Date.
    """
    if ledger_df is None or ledger_df.empty:
        return pd.DataFrame()
    now = pd.Timestamp.now().normalize()
    end = now + pd.Timedelta(days=days)
    df = ledger_df.copy()
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    mask_date = df["Date"].notna() & (df["Date"] >= now) & (df["Date"] <= end)
    if "status" in df.columns:
        mask_status = ~df["status"].astype(str).str.lower().eq("pagado")
    else:
        mask_status = pd.Series(True, index=df.index)
    return df.loc[mask_date & mask_status].sort_values("Date")

# ---------------------------------------------------------------------
# IO / CLI helper
# ---------------------------------------------------------------------
def write_report_csv(df: pd.DataFrame, target: Path) -> Path:
    p = Path(target)
    atomic_write_df(df, p)
    return p

def run_write_all(out_dir: Path, freq: str, parties: Sequence[str], write_dir: Optional[Path] = None) -> Dict:
    """
    Convenience runner that:
      - loads materialized folder (CSV-based)
      - runs validation
      - writes fondos_report.csv and renta_{party}.csv to write_dir (defaults to out_dir)
    Returns summary dict with validation and outputs metadata.
    """
    write_dir = Path(write_dir or out_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    materialized = load_materialized_folder(Path(out_dir), freq=freq)
    ledger = materialized.get("ledger_canonical", pd.DataFrame())
    validation = validate_materialization_totals(ledger, materialized)
    outputs = {}

    # fondos
    fondos = build_fondos_report_from_materialized(materialized, parties, freq)
    p = write_report_csv(fondos, write_dir / "fondos_report.csv")
    outputs["fondos_report.csv"] = {"rows": len(fondos.index) if hasattr(fondos, "index") else 0, "sha256": sha256_file(p)}

    # renta per party
    for t in parties:
        df_s = build_renta_series_from_materialized(materialized, t, freq)
        if df_s.empty:
            outputs[f"renta_{_sanitize_name(t)}.csv"] = {"rows": 0, "sha256": ""}
            continue
        # write one file per currency column
        for cur in df_s.columns:
            df_cur = df_s[[cur]].reset_index().rename(columns={cur: "amount"})
            s_path = write_dir / f"renta_{_sanitize_name(t)}_{cur}.csv"
            atomic_write_df(df_cur, s_path)
            outputs[s_path.name] = {"rows": len(df_cur), "sha256": sha256_file(s_path)}


    return {"validation": validation, "outputs": outputs}


# CLI glue for src/accounting/reports.py

import argparse
import json
import sys
# from pathlib import Path
from typing import List, Optional

def _parse_parties_arg(raw: Optional[List[str]]) -> Optional[List[str]]:
    """
    Accept either:
      - multiple --parties A B C
      - a single comma-separated string in the first position
    Returns None if raw is None or empty.
    """
    if not raw:
        return None
    if len(raw) == 1 and "," in raw[0]:
        return [p.strip() for p in raw[0].split(",") if p.strip()]
    return [p for p in raw if p is not None and p != ""]

def _derive_top_parties(materialized: dict, top_n: int = 6) -> List[str]:
    """
    Derive top parties by absolute total movement (uses 'amount' column).
    Returns an empty list if per_party_time_long missing or empty.
    """
    pp = materialized.get("per_party_time_long")
    if pp is None or pp.empty:
        return []
    # try to get numeric amount; fallback to zeros
    amt = _ensure_amount_series(pp, "amount")
    tmp = pp.copy()
    tmp["_amount_for_rank"] = amt.abs()
    totals = tmp.groupby("party")["_amount_for_rank"].sum().sort_values(ascending=False)
    return [p for p in totals.index.tolist()[:top_n]]

import os

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="reports", description="Generate accounting reports from materialized CSVs.")
    p.add_argument("-o", "--out-dir", default="out", help="Path to materialized outputs (CSV folder).")
    p.add_argument("-f", "--freq", default="W", help="Frequency label used in materialized filenames (default: W).")
    p.add_argument("-w", "--write-dir", default=None, help="Directory to write reports to (defaults to out-dir).")
    p.add_argument("--parties", nargs="+", help="List of parties (space separated) or a single comma-separated string. If omitted, top parties will be auto-derived.")
    p.add_argument("--top", type=int, default=6, help="When parties omitted, pick top N parties by volume (default: 6).")
    p.add_argument("--no-validate", action="store_true", help="Skip validation step (not recommended).")
    p.add_argument("--pretty-json", action="store_true", help="Print summary JSON prettily.")
    p.add_argument("--summary-path", default=None, help="Write summary JSON to this path.")
    p.add_argument("--mode", default=os.getenv("MODE", "run"), choices=["smoke", "run"])
    p.add_argument("--run-id", default=os.getenv("RUN_ID", ""))


    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    write_dir = Path(args.write_dir) if args.write_dir else out_dir
    write_dir.mkdir(parents=True, exist_ok=True)

    # load materialized CSVs
    materialized = load_materialized_folder(out_dir, freq=args.freq)

    # determine parties
    parties_arg = _parse_parties_arg(args.parties)
    if parties_arg:
        parties = parties_arg
    else:
        parties = _derive_top_parties(materialized, top_n=args.top)

    # run validation unless skipped
    ledger = materialized.get("ledger_canonical", None)
    validation = None
    if not args.no_validate:
        validation = validate_materialization_totals(ledger, materialized)

    # produce reports (fondos + renta per party) using the existing runner
    result = run_write_all(out_dir=out_dir, freq=args.freq, parties=parties, write_dir=write_dir)

    # attach validation summary if present
    if validation is not None:
        result.setdefault("validation", validation)




    from accounting.manifest import artifact_from_path, write_stage_manifest, append_artifacts

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    rid = args.run_id or ("smoke" if args.mode == "smoke" else "")

    # outputs reportados por tu runner
    out_arts = []
    outputs = (result.get("outputs") or {})

    for fname, meta in outputs.items():
        fpath = (write_dir / fname)
        if fpath.exists():
            out_arts.append(artifact_from_path(
                name=fname.replace(".csv", ""),
                path=fpath,
                stage="E.reports",
                mode=args.mode,
                run_id=rid,
                role="primary",
                root_dir=out_dir,
                rows=meta.get("rows"),
                content_type="text/csv",
            ))

    stage_manifest = {
        "stage": "E.reports",
        "mode": args.mode,
        "run_id": rid,
        "inputs": [
            # input principal: la materialización ya hecha
            # opcional: si querés, podés agregar artifacts de per_party/per_flow aquí
            {"kind": "materialized_folder", "relpath": str(Path(args.out_dir).resolve().relative_to(out_dir.resolve())) if False else "."}
        ],
        "params": {"freq": args.freq, "top": args.top, "parties": parties, "write_dir": str(write_dir.resolve().relative_to(out_dir.resolve()))},
        "outputs": out_arts,
        "validation": result.get("validation"),
        "warnings": [],
    }

    stage_manifest_rel = write_stage_manifest(meta_dir, stage_manifest)

    stage_meta_art = artifact_from_path(
        name="stage_E_reports",
        path=(out_dir / stage_manifest_rel),
        stage="E.reports",
        mode=args.mode,
        run_id=rid,
        role="meta",
        root_dir=out_dir,
        content_type="application/json",
    )

    append_artifacts(meta_dir, [*out_arts, stage_meta_art])

    # stdout sigue existiendo
    if args.pretty_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if args.summary_path:
            with open(args.summary_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        print(json.dumps(result, ensure_ascii=False))

    return 0




if __name__ == "__main__":
    sys.exit(main())
