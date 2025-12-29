# src/accounting/ingest.py
"""
Ingest / canonicalization for accounting pipeline.

Primary entrypoint:
    build_ledger_base(fixture_path=None, sheet_url=None, service_account_file=None, ...)
returns:
    pandas.DataFrame with canonical columns:
      tx_id, Date (datetime), amount (float), amount_cents (int), currency,
      base_amount (float, optional), payer, receiver, flujo, tipo,
      source_file, source_row, ingest_ts, notes

The returned DataFrame will have an attribute .attrs["anomalies"] which is a
pandas.DataFrame with rows that were flagged during ingest.
"""
from __future__ import annotations

import sys
sys.path.append('./../../')
sys.path.append('./../')
sys.path.append('./')


import os
import logging
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import hashlib
import json
import datetime

import pandas as pd

from src.accounting.core_timeseries import process_time_aggregation


# CLI wrapper to run src.accounting.ingest.build_ledger_base and write out ledger_canonical.csv
# and anomalies.csv (if any).

# Usage examples:
#   python scripts/run_ingest.py --fixture ./fixtures/ledger_fixture.csv --out-dir ./out
#   python scripts/run_ingest.py --service-account /path/sa.json --sheet-url "https://..." --out-dir ./out

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# LOG = logging.getLogger("run_ingest")


LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_sheet_to_df(sheet_url: str, service_account_file: str, sheet_name: str = "LEDGERS") -> pd.DataFrame:
    """
    Helper to load a google sheet tab into a DataFrame using functions.get_google_sheets_client
    and functions.load_google_sheet that exist in src.accounting.functions.
    """
    try:
        from src.accounting.utils import get_google_sheets_client, load_google_sheet
    except Exception as e:
        LOG.exception("Google Sheets helpers not available in src.accounting.functions: %s", e)
        raise

    client = get_google_sheets_client(service_account_file)
    df = load_google_sheet(client, sheet_url, sheet_name)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("load_google_sheet did not return a DataFrame")
    return df


def _deterministic_tx_id(row_values: Tuple[str, ...]) -> str:
    """
    Compute a deterministic tx id from a tuple of stable strings.
    """
    h = hashlib.sha256()
    concat = "|".join("" if v is None else str(v) for v in row_values)
    h.update(concat.encode("utf8"))
    return h.hexdigest()


def _load_fx_rates(fx_rates_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not fx_rates_path:
        return None
    p = Path(fx_rates_path).expanduser()
    if not p.exists():
        LOG.warning("FX rates file not found: %s", p)
        return None
    try:
        fx = pd.read_csv(p, parse_dates=["date"], dtype={"Currency": str})
        # expect columns: date, currency, rate_to_base (float)
        expected = {"date", "Currency", "rate_to_base"}
        if not expected.issubset(set(fx.columns)):
            LOG.warning("FX table missing expected columns (date,currency,rate_to_base). Found: %s", fx.columns.tolist())
            return None
        fx["date"] = pd.to_datetime(fx["date"]).dt.date
        return fx
    except Exception:
        LOG.exception("Failed to load FX table: %s", fx_rates_path)
        return None


def _apply_fx_for_row(amount: float, currency: str, date: pd.Timestamp, fx_df: pd.DataFrame, base_currency: str) -> Optional[float]:
    """
    Given amount, currency, date, and fx dataframe (date,currency,rate_to_base),
    return base amount (float) or None if not found.
    Assumes rate_to_base means: base_amount = amount * rate_to_base
    """
    if fx_df is None or pd.isna(amount) or not currency:
        return None
    try:
        d = pd.to_datetime(date).date()
    except Exception:
        return None
    # try exact date match first
    match = fx_df[(fx_df["date"] == d) & (fx_df["Currency"].astype(str) == str(currency))]
    if match.shape[0] == 0:
        # try nearest previous date for that currency
        sub = fx_df[fx_df["Currency"].astype(str) == str(currency)].sort_values("date")
        sub = sub[sub["date"] <= d]
        if sub.shape[0] == 0:
            return None
        rate = sub.iloc[-1]["rate_to_base"]
    else:
        rate = match.iloc[0]["rate_to_base"]
    try:
        return float(amount) * float(rate)
    except Exception:
        return None


def _collect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect canonical-like df for rows with obvious problems and return
    a small anomalies dataframe listing row index and issue.
    """
    rows = []
    for i, row in df.iterrows():
        issues = []
        if pd.isna(row.get("Date")):
            issues.append("invalid_date")
        if pd.isna(row.get("amount")):
            issues.append("missing_amount")
        # if not row.get("Currency"):
        #     issues.append("missing_currency")
        if issues:
            rows.append({
                "row_index": i,
                "tx_id": row.get("tx_id"),
                "issues": ";".join(issues),
                "raw_payer": row.get("payer"),
                "raw_receiver": row.get("receiver"),
            })
    if not rows:
        return pd.DataFrame(columns=["row_index", "tx_id", "issues", "raw_payer", "raw_receiver"])
    return pd.DataFrame(rows)


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def build_ledger_base(
    fixture_path: Optional[str] = None,
    sheet_url: Optional[str] = None,
    service_account_file: Optional[str] = None,
    sheet_name: str = "LEDGERS",
    party_map_path: Optional[str] = None,
    fx_rates_path: Optional[str] = '',
    base_currency: str = "ARS",
    require_tx_id: bool = False,
) -> pd.DataFrame:
    """
    Build canonical ledger DataFrame from fixture CSV or Google Sheet.

    NOTE: this function returns the canonical DataFrame and DOES NOT write final artifact CSVs.
    The returned DataFrame has .attrs["anomalies"] with a small DataFrame describing issues found.

    Canonical columns produced:
      tx_id (str), Date (datetime64[ns]), amount (float), amount_cents (optional int),
      currency (str), base_amount (float or NaN), payer, receiver, flujo, tipo,
      source_file (str), source_row (int), ingest_ts (iso str), notes (str)

    Defaults: if fixture_path is None, this function attempts to read from
    environment variables SHEET_URL and SERVICE_ACCOUNT_FILE or the arguments provided.
    """
    # 1) Load raw data
    if fixture_path:
        p = Path(fixture_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Fixture not found: {p}")
        LOG.info("Loading ledger fixture: %s", p)
        raw = pd.read_csv(p, dtype=str, low_memory=False)
        source_file = str(p)
    else:
        # prefer explicit args, fall back to env
        sheet_url = sheet_url or os.getenv("SHEET_URL")
        service_account_file = service_account_file or os.getenv("SERVICE_ACCOUNT_FILE")
        if not sheet_url or not service_account_file:
            raise ValueError("No fixture_path provided and SHEET_URL or SERVICE_ACCOUNT_FILE not set.")
        LOG.info("Loading ledger from Google Sheet: %s (sheet=%s)", sheet_url, sheet_name)
        raw = read_sheet_to_df(sheet_url, service_account_file, sheet_name)
        source_file = sheet_url

    if raw is None or raw.shape[0] == 0:
        LOG.warning("No data loaded from source.")
        return pd.DataFrame()

    # normalize column names (strip)
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})

    # prefer the known columns mapping (try to be forgiving)
    # expected sheet columns (examples): transaction_id, Date, Box, payer, receiver, amount, Currency, Flujo, Tipo, Lugar, Detalle, issuer, account_id, status, medio
    colmap = {
        "transaction_id": "transaction_id",
        "transaction id": "transaction_id",
        "tx_id": "transaction_id",
        "date": "Date",
        "fecha": "Date",
        "box": "Box",
        "payer": "payer",
        "payee": "receiver",
        "receiver": "receiver",
        "amount": "amount",
        "monto": "amount",
        "Currency": "Currency",
        "moneda": "Currency",
        "flujo": "Flujo",
        "tipo": "Tipo",
        "lugar": "Lugar",
        "detalle": "Detalle",
        "issuer": "issuer",
        "account_id": "account_id",
        "status": "status",
        "medio": "medio",
    }
    # perform case-insensitive mapping
    lower_to_orig = {c.lower(): c for c in raw.columns}
    normalized = {}
    for k, v in colmap.items():
        if k in lower_to_orig:
            normalized[v] = lower_to_orig[k]
    # apply renaming to canonical intermediate names
    df = raw.rename(columns={orig: new for new, orig in normalized.items()})

    # df = df.loc[df.Box != 'Household'] ### Hardcoded FB BOX only
    df = df.loc[df.status.isin(['pagado'])] ### Hardcoded pagado only

    # If some core columns are missing, try to fallback using presence of approximate names
    # Ensure we have at least Date, amount, Currency, payer/receiver if possible
    # cast amount to float where possible
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(",", ""), errors="coerce")
    else:
        # try other heuristics
        amount_cands = [c for c in df.columns if "amount" in c.lower() or "monto" in c.lower() or "total" in c.lower()]
        if amount_cands:
            df["amount"] = pd.to_numeric(df[amount_cands[0]].astype(str).str.replace(",", ""), errors="coerce")
        else:
            df["amount"] = pd.NA

    # parse Date conservatively: do not coerce to timezone
    date_col_candidates = [c for c in ("Date", "date", "Fecha", "fecha") if c in df.columns]
    if date_col_candidates:
        df["Date"] = pd.to_datetime(df[date_col_candidates[0]], errors="coerce")
    else:
        df["Date"] = pd.NaT

    # ensure currency column exists
    if "Currency" not in df.columns:
        # try to detect common currency column
        cands = [c for c in df.columns if c.lower() in ("Currency", "moneda", "curr")]
        if cands:
            df["Currency"] = df[cands[0]].astype(str).str.upper()
        else:
            df["Currency"] = ""

    # canonical textual fields
    for col in ("payer", "receiver", "Flujo", "Tipo", "Lugar", "Detalle", "issuer", "account_id", "status", "medio"):
        if col not in df.columns:
            df[col] = ""

    # source provenance: source_file and source_row (1-based)
    df = df.reset_index(drop=True)
    df["source_row"] = df.index + 1
    df["source_file"] = source_file
    ingest_ts = datetime.datetime.utcnow().isoformat() + "Z"
    df["ingest_ts"] = ingest_ts

    # notes: concat Lugar + Detalle + issuer + medio (if present)
    df["notes"] = (
        df.get("Lugar", "").astype(str).fillna("")
        + " | "
        + df.get("Detalle", "").astype(str).fillna("")
        + " | "
        + df.get("issuer", "").astype(str).fillna("")
        + " | "
        + df.get("medio", "").astype(str).fillna("")
    )

    # deterministic tx_id generation (always compute to guarantee stable ids)
    tx_ids = []
    for _, r in df.iterrows():
        row_vals = (
            pd.to_datetime(r["Date"]).strftime("%Y-%m-%d") if not pd.isna(r["Date"]) else "",
            _safe_str(r.get("payer")),
            _safe_str(r.get("receiver")),
            _safe_str(r.get("amount")),
            _safe_str(r.get("Currency")),
            _safe_str(r.get("Flujo")),
            _safe_str(r.get("Tipo")),
            str(r.get("source_row")),
        )
        tx_ids.append(_deterministic_tx_id(row_vals))
    df["tx_id"] = tx_ids

    # compute amount_cents as optional compatibility column (rounded int)
    df["amount_cents"] = pd.to_numeric((df["amount"].astype(float) * 100).round(0), errors="coerce").astype("Int64")

    # load FX if requested
    fx_df = _load_fx_rates(fx_rates_path)
    # compute base_amount using fx_df if available and Currency != base_currency
    base_amounts = []
    for _, r in df.iterrows():
        amt = r.get("amount")
        cur = r.get("Currency")
        if amt is None or pd.isna(amt):
            base_amounts.append(pd.NA)
            continue
        if not fx_df or (not cur) or str(cur).upper() == str(base_currency).upper():
            base_amounts.append(pd.NA)
            continue
        base = _apply_fx_for_row(amt, cur, r.get("Date"), fx_df, base_currency)
        base_amounts.append(base)
    df["base_amount"] = base_amounts

    # collect anomalies: bad dates, missing amounts, missing currency
    anomalies_df = _collect_anomalies(df)

    # optionally add party normalization if party_map provided
    if party_map_path:
        try:
            pmap = json.loads(Path(party_map_path).read_text(encoding="utf8"))
            # apply simple mapping for payer and receiver
            def normalize_party(x):
                if not x:
                    return x
                s = str(x).strip()
                return pmap.get(s, s)
            df["payer"] = df["payer"].apply(normalize_party)
            df["receiver"] = df["receiver"].apply(normalize_party)
        except Exception:
            LOG.exception("Failed to load/apply party_map: %s", party_map_path)

    # run lightweight time aggregation helper (adds TimePeriod)
    try:
        df = process_time_aggregation(df, time_freq="W", date_col="Date")
    except Exception:
        LOG.exception("process_time_aggregation failed; proceeding without TimePeriod.")

    # final column ordering for canonical ledger
    canonical_cols = [
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
    canonical_df = df[[c for c in canonical_cols if c in df.columns]].copy()

    # attach anomalies
    canonical_df.attrs["anomalies"] = anomalies_df

    LOG.info("Built ledger_base rows=%d anomalies=%d", len(canonical_df), len(anomalies_df))
    return canonical_df




def parse_args():
    p = argparse.ArgumentParser(description="Run ingest and write ledger_canonical.csv")
    p.add_argument("--fixture", help="Local fixture CSV/Parquet path (prefer)", default=os.getenv("FIXTURE"))
    p.add_argument("--service-account", help="Google service account JSON path", default=os.getenv("SERVICE_ACCOUNT") or os.getenv("ACCOUNT_SERVICE_ACCOUNT"))
    p.add_argument("--sheet-url", help="Google Sheet URL", default=os.getenv("SHEET_URL") or os.getenv("ACCOUNT_SHEET_URL"))
    p.add_argument("--sheet-name", help="Sheet/tab name (when using Google Sheets)", default="C. Long Ledger")
    p.add_argument("--out-dir", help="Output directory", default=os.getenv("OUT_DIR", "./out"))
    p.add_argument("--exclude-household", action="store_true", help="Exclude household Box rows", default=False)
    p.add_argument("--require-tx-id", action="store_true", help="Require tx_id generation/enforcement", default=False)
    return p.parse_args()

    


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from src.accounting.ingest import build_ledger_base
    except Exception as e:
        LOG.exception("Failed importing ingest module: %s", e)
        sys.exit(10)

    fixture = args.fixture
    service_account = args.service_account
    sheet_url = args.sheet_url

    use_fixture = False
    if fixture:
        f = Path(fixture)
        if f.exists():
            use_fixture = True
            LOG.info("Using fixture: %s", f)
        else:
            LOG.warning("Fixture provided but not found: %s", f)

    try:
        if use_fixture:
            # ledger = build_ledger_base(fixture_path=str(f), sheet_name=args.sheet_name, exclude_household=args.exclude_household, require_tx_id=args.require_tx_id)
            ledger = build_ledger_base(fixture_path=str(f), sheet_name=args.sheet_name, require_tx_id=args.require_tx_id)
        else:
            if not service_account or not sheet_url:
                LOG.error("Live mode requires --service-account and --sheet-url when fixture not provided.")
                sys.exit(2)
            # ledger = build_ledger_base(service_account_file=service_account, sheet_url=sheet_url, sheet_name=args.sheet_name, exclude_household=args.exclude_household, require_tx_id=args.require_tx_id)
            ledger = build_ledger_base(service_account_file=service_account, sheet_url=sheet_url, sheet_name=args.sheet_name, require_tx_id=args.require_tx_id)

            # Make sure to cover for currency... build_ledger_base(..., fx_rates_path = 
    except Exception as e:
        LOG.exception("Ingest failed: %s", e)
        sys.exit(3)

    if ledger is None or getattr(ledger, "shape", (0,))[0] == 0:
        LOG.error("Ingest produced no rows - aborting")
        sys.exit(4)

    # write canonical ledger
    ledger_path = out_dir / "ledger_canonical.csv"
    # ensure Date formatting for readability
    if "Date" in ledger.columns:
        try:
            ledger["Date"] = pd.to_datetime(ledger["Date"], errors="coerce").dt.date.astype(str)
        except Exception:
            pass
    ledger.to_csv(ledger_path, index=False)
    LOG.info("Wrote ledger_canonical rows=%d -> %s", len(ledger), ledger_path)

    # anomalies
    anoms = ledger.attrs.get("anomalies")
    if isinstance(anoms, pd.DataFrame) and not anoms.empty:
        anoms_path = out_dir / "anomalies.csv"
        anoms.to_csv(anoms_path, index=False)
        LOG.info("Wrote anomalies rows=%d -> %s", len(anoms), anoms_path)
    else:
        LOG.info("No anomalies to write")

    print(ledger_path)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)