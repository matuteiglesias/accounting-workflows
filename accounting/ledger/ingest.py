# src/accounting/ingest.py
"""
Ingest and canonicalization for the accounting pipeline.

Primary entrypoint:
    build_ledger_base(...)

Returns:
    pandas.DataFrame with canonical columns (pipeline internal contract):
      - tx_id (str)
      - Date (datetime64[ns])
      - amount (float)                 native currency signed amount (do not mix currencies)
      - amount_cents (Int64)           derived from amount when possible
      - Currency (str)                native currency code, eg "ARS", "USD"
      - base_amount (float, optional) amount converted to base_currency when fx_rates_path is provided
      - payer (str), receiver (str)   parties (external can be NaN)
      - Flujo (str), Tipo (str)       flow and type classifiers (stable labels)
      - status (str, optional)
      - Box (str, optional)
      - source_file (str), source_row (Int64), ingest_ts (str), notes (str)

The returned DataFrame will have an attribute .attrs["anomalies"] which is a
pandas.DataFrame with issues flagged during ingest (for observability, not policy).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from accounting.core.timeseries import period_bins_for_dates
from accounting.logging_utils import configure_logging, get_logger
from accounting.scope import box_scope_mask, canonical_box_name, parse_box_scope, scope_metadata


LOG = get_logger("ingest")


# -----------------------
# Source loaders
# -----------------------
def read_sheet_to_df(sheet_url: str, service_account_file: str, sheet_name: str = "LEDGERS") -> pd.DataFrame:
    """
    Load a Google Sheet tab into a pandas DataFrame.

    Notes:
      - Imports gspread lazily so the module remains import-safe without gspread installed.
      - This function is intentionally small and I/O only.
    """
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception as e:
        raise RuntimeError("gspread/google-auth not available; install deps to use Google Sheets ingest") from e

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(service_account_file, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_url(sheet_url)
    ws = sh.worksheet(sheet_name)
    records = ws.get_all_records()
    return pd.DataFrame(records)


def _read_fixture(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


# -----------------------
# Canonicalization helpers
# -----------------------
_CANON_COLS = {
    # time and money
    "date": "Date",
    "fecha": "Date",
    "day": "Date",
    "amount": "amount",
    "importe": "amount",
    "monto": "amount",
    "amount_cents": "amount_cents",
    "currency": "Currency",
    "moneda": "Currency",
    "curr": "Currency",
    # parties
    "payer": "payer",
    "pagador": "payer",
    "from": "payer",
    "receiver": "receiver",
    "to": "receiver",
    "receptor": "receiver",
    # classifiers and ops
    "flujo": "Flujo",
    "flow": "Flujo",
    "tipo": "Tipo",
    "type": "Tipo",
    "status": "status",
    "estado": "status",
    "box": "Box",
    "caja": "Box",
    # misc
    "detalle": "Detalle",
    "concepto": "Detalle",
    "notes": "notes",
    "nota": "notes",
}


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map input columns (case-insensitive) into canonical names used internally.

    This does not invent business semantics. It only standardizes naming.
    """
    df = raw.copy()

    rename: Dict[str, str] = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in _CANON_COLS:
            rename[c] = _CANON_COLS[key]

    df = df.rename(columns=rename)

    # Normalize a couple of frequent variants without guessing.
    if "Currency" not in df.columns:
        for c in list(df.columns):
            if str(c).strip().lower() in ("currency", "moneda", "curr"):
                df = df.rename(columns={c: "Currency"})
                break

    if "Date" not in df.columns:
        for c in list(df.columns):
            if str(c).strip().lower() in ("date", "fecha", "day"):
                df = df.rename(columns={c: "Date"})
                break

    return df


def _coerce_money_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    # Amount
    if "amount" in out.columns:
        # Tolerate:
        # - "1234.56" (dot decimal)
        # - "1.234,56" (dot thousands, comma decimal)
        # - "1234,56" (comma decimal)
        s = out["amount"].astype("string").fillna("").str.strip()
        s = s.str.replace(" ", "", regex=False)

        has_comma = s.str.contains(",", na=False)
        has_dot = s.str.contains(r"\.", na=False)

        # both comma and dot: assume dot thousands, comma decimal
        both = has_comma & has_dot
        s_both = s.where(~both, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))

        # comma only: assume comma decimal
        comma_only = has_comma & ~has_dot
        s_norm = s_both.where(~comma_only, s_both.str.replace(",", ".", regex=False))

        out["amount"] = pd.to_numeric(s_norm, errors="coerce")
    elif "amount_cents" in out.columns:
        out["amount_cents"] = pd.to_numeric(out["amount_cents"], errors="coerce")
        out["amount"] = out["amount_cents"] / 100.0
    else:
        out["amount"] = pd.NA

    # Amount cents (prefer stable integer)
    if "amount_cents" not in out.columns:
        out["amount_cents"] = (out["amount"] * 100.0).round().astype("Int64")
    else:
        out["amount_cents"] = pd.to_numeric(out["amount_cents"], errors="coerce").round().astype("Int64")

    # Currency cleanup: keep as uppercase strings, but do not invent values
    if "Currency" in out.columns:
        out["Currency"] = out["Currency"].astype(str).str.strip().str.upper()
        out.loc[out["Currency"].isin(["", "NAN", "NONE"]), "Currency"] = pd.NA
    else:
        out["Currency"] = pd.NA

    return out


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["payer", "receiver", "Flujo", "Tipo", "status", "Box", "Detalle", "notes"]:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def filter_ledger_statuses(df: pd.DataFrame, statuses: Optional[Sequence[str]]) -> pd.DataFrame:
    """Return the accounting-recognition subset without altering scoped evidence."""
    if statuses is None or "status" not in df.columns:
        return df.copy()
    allowed = {str(value).strip().lower() for value in statuses}
    mask = df["status"].astype(str).str.strip().str.lower().isin(allowed)
    out = df.loc[mask].copy()
    if "anomalies" in df.attrs:
        out.attrs["anomalies"] = df.attrs["anomalies"]
    return out


def _apply_party_map(df: pd.DataFrame, party_map: Optional[Dict[str, str]]) -> pd.DataFrame:
    if not party_map:
        return df
    out = df.copy()
    out["payer"] = out["payer"].replace(party_map)
    out["receiver"] = out["receiver"].replace(party_map)
    return out


def _build_tx_id(df: pd.DataFrame) -> pd.Series:
    date_str = df["Date"].dt.date.astype("string").fillna("")
    payer = df["payer"].astype("string").fillna("").str.strip()
    receiver = df["receiver"].astype("string").fillna("").str.strip()
    flujo = df["Flujo"].astype("string").fillna("").str.strip()
    tipo = df["Tipo"].astype("string").fillna("").str.strip()
    cur = df["Currency"].astype("string").fillna("").str.strip().str.upper()
    cents = df["amount_cents"].astype("Int64").astype("string").fillna("")
    src_row = df["source_row"].astype("Int64").astype("string").fillna("")
    src_file = df["source_file"].astype("string").fillna("")

    sig = (
        date_str
        + "|"
        + payer
        + "|"
        + receiver
        + "|"
        + cur
        + "|"
        + flujo
        + "|"
        + tipo
        + "|"
        + cents
        + "|"
        + src_row
        + "|"
        + src_file
    )

    def _h(x: str) -> str:
        return hashlib.sha1(x.encode("utf-8")).hexdigest()[:16]

    return sig.map(_h)


def _load_party_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".json",):
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix.lower() in (".csv",):
        df = pd.read_csv(p)
        cols = [c.lower() for c in df.columns]
        if "from" in cols and "to" in cols:
            c_from = df.columns[cols.index("from")]
            c_to = df.columns[cols.index("to")]
            return dict(zip(df[c_from].astype(str), df[c_to].astype(str)))
        raise ValueError("party_map csv must have columns 'from' and 'to'")
    raise ValueError(f"Unsupported party_map_path: {p}")


def _load_fx_rates(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    fx = pd.read_csv(p) if p.suffix.lower() not in (".parquet", ".pq") else pd.read_parquet(p)
    fx = _normalize_columns(fx)

    if "Date" not in fx.columns:
        raise ValueError("fx_rates missing Date column")
    if "Currency" not in fx.columns:
        raise ValueError("fx_rates missing Currency column")

    fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
    fx["Currency"] = fx["Currency"].astype(str).str.strip().str.upper()
    fx.loc[fx["Currency"].isin(["", "NAN", "NONE"]), "Currency"] = pd.NA

    rate_col = None
    for c in fx.columns:
        if str(c).strip().lower() in ("rate_to_base", "rate", "fx", "tc"):
            rate_col = c
            break
    if rate_col is None:
        raise ValueError("fx_rates missing rate_to_base/rate column")

    fx["rate_to_base"] = pd.to_numeric(fx[rate_col], errors="coerce")
    fx = fx[["Date", "Currency", "rate_to_base"]].dropna(subset=["Date", "Currency"])
    return fx


def _attach_base_amount(
    df: pd.DataFrame,
    fx: pd.DataFrame,
    base_currency: str = "ARS",
) -> Tuple[pd.Series, pd.Series]:
    base_currency = str(base_currency).strip().upper()
    cur = df["Currency"].astype("string").str.upper()
    needs_fx = cur.notna() & (cur != base_currency)

    base_amount = pd.Series(pd.NA, index=df.index, dtype="Float64")
    base_amount.loc[~needs_fx] = df.loc[~needs_fx, "amount"].astype("Float64")

    if needs_fx.any():
        left = df.loc[needs_fx, ["Date", "Currency", "amount"]].copy()
        right = fx.copy()

        merged = left.merge(
            right,
            how="left",
            on=["Date", "Currency"],
            validate="m:1",
        )
        fx_missing = merged["rate_to_base"].isna()

        computed = merged["amount"].astype("Float64") * merged["rate_to_base"].astype("Float64")
        base_amount.loc[left.index] = computed.values

        fx_missing_mask = pd.Series(False, index=df.index)
        fx_missing_mask.loc[left.index] = fx_missing.values
    else:
        fx_missing_mask = pd.Series(False, index=df.index)

    return base_amount, fx_missing_mask


def _collect_anomalies(df: pd.DataFrame, fx_missing_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    issues: List[pd.DataFrame] = []

    def _pack(mask: pd.Series, issue: str, detail: str) -> None:
        if mask is None or not mask.any():
            return
        sub = df.loc[mask, ["tx_id", "source_file", "source_row"]].copy()
        sub["issue"] = issue
        sub["detail"] = detail
        issues.append(sub)

    _pack(df["Date"].isna(), "missing_date", "Date could not be parsed")
    _pack(df["Currency"].isna() | (df["Currency"].astype("string").str.strip() == ""), "missing_currency", "Currency missing/blank")
    _pack(df["amount"].isna(), "missing_amount", "amount could not be parsed")
    _pack(df["payer"].isna() & df["receiver"].isna(), "missing_parties", "payer and receiver are both null")
    _pack(df["Flujo"].isna() | (df["Flujo"].astype("string").str.strip() == ""), "missing_flujo", "Flujo missing/blank")
    _pack(df["Tipo"].isna() | (df["Tipo"].astype("string").str.strip() == ""), "missing_tipo", "Tipo missing/blank")

    if fx_missing_mask is not None:
        _pack(fx_missing_mask, "missing_fx_rate", "fx rate missing for (Date, Currency)")

    if not issues:
        return pd.DataFrame(columns=["tx_id", "source_file", "source_row", "issue", "detail"])

    out = pd.concat(issues, ignore_index=True)
    if "source_row" in out.columns:
        out["source_row"] = pd.to_numeric(out["source_row"], errors="coerce").astype("Int64")
    return out


# -----------------------
# Public API
# -----------------------

def build_ledger_base(
    fixture_path: Optional[str] = None,
    sheet_url: Optional[str] = None,
    service_account_file: Optional[str] = None,
    sheet_name: str = "LEDGERS",
    party_map_path: Optional[str] = None,
    fx_rates_path: Optional[str] = "",
    base_currency: str = "ARS",
    require_tx_id: bool = False,
    exclude_household: bool = False,
    boxes: Optional[set[str]] = None,
    only_status: Optional[Sequence[str]] = ("pagado",),
    add_time_period: bool = False,
    time_freq: str = "W",
) -> pd.DataFrame:
    # sheet_url="https://docs.google.com/spreadsheets/d/1G92WIwtaayQ0MOSLH3vWph-7PKVseXF4vhiTOi2RVls/edit?gid=1682108108#gid=1682108108"


    if not fixture_path and not sheet_url:
        fixture_path = os.getenv("FIXTURE") or None
        sheet_url = os.getenv("SHEET_URL") or os.getenv("ACCOUNT_SHEET_URL") or None
    if not service_account_file:
        service_account_file = os.getenv("SERVICE_ACCOUNT") or os.getenv("ACCOUNT_SERVICE_ACCOUNT") or None

    if fixture_path:
        raw = _read_fixture(fixture_path)
        source_file = str(Path(fixture_path).resolve())
    elif sheet_url and service_account_file:
        raw = read_sheet_to_df(sheet_url=sheet_url, service_account_file=service_account_file, sheet_name=sheet_name)
        source_file = sheet_url
    else:
        raise ValueError("Must provide fixture_path or (sheet_url and service_account_file)")

    df = _normalize_columns(raw)
    source_has_box = "Box" in df.columns
    df = _ensure_columns(df)

    df["source_file"] = source_file
    if "source_row" not in df.columns:
        df["source_row"] = pd.RangeIndex(start=1, stop=len(df) + 1, step=1).astype("Int64")
    else:
        df["source_row"] = pd.to_numeric(df["source_row"], errors="coerce").astype("Int64")

    df["ingest_ts"] = _dt.datetime.utcnow().replace(microsecond=0).isoformat()

    df = _coerce_money_and_dates(df)

    # Box is the sole ledger-partition key. Normalize canonical spelling before
    # selecting the declared universe; counterparty/semantic fields play no role.
    if boxes is not None:
        if exclude_household:
            raise ValueError("Use boxes or legacy exclude_household, not both")
        canonical_boxes = {canonical_box_name(box) for box in boxes}
        if not source_has_box:
            raise ValueError("Scoped canonical ingest requires a Box column")
        nonempty = df["Box"].notna() & df["Box"].astype(str).str.strip().ne("")
        normalized = df.loc[nonempty, "Box"].map(canonical_box_name)
        df.loc[nonempty, "Box"] = normalized
        df = df.loc[box_scope_mask(df, canonical_boxes)].copy()

    if only_status is not None and "status" in df.columns:
        allowed = {str(s).strip().lower() for s in only_status}
        df = df[df["status"].astype(str).str.strip().str.lower().isin(allowed)].copy()

    if exclude_household and "Box" in df.columns:
        household_markers = {"household", "hogar", "h", "casa"}
        mask_house = df["Box"].astype(str).str.strip().str.lower().isin(household_markers)
        df = df[~mask_house].copy()

    party_map = _load_party_map(party_map_path)
    df = _apply_party_map(df, party_map)

    notes_parts = []
    if "Detalle" in df.columns:
        notes_parts.append(df["Detalle"].astype("string").fillna(""))
    if "notes" in df.columns:
        notes_parts.append(df["notes"].astype("string").fillna(""))
    if notes_parts:
        df["notes"] = notes_parts[0]
        for part in notes_parts[1:]:
            df["notes"] = (df["notes"].astype("string") + " | " + part.astype("string")).str.strip(" |")
    else:
        df["notes"] = pd.NA

    if "tx_id" in df.columns:
        df["tx_id"] = df["tx_id"].astype("string").str.strip()
        missing_tx = df["tx_id"].isna() | (df["tx_id"].str.strip() == "")
    else:
        df["tx_id"] = pd.NA
        missing_tx = pd.Series(True, index=df.index)

    if require_tx_id or missing_tx.any():
        df.loc[missing_tx, "tx_id"] = _build_tx_id(df.loc[missing_tx])

    fx_missing_mask = None
    if fx_rates_path:
        fx = _load_fx_rates(fx_rates_path)
        if fx is None:
            df["base_amount"] = pd.NA
        else:
            base_amount, fx_missing_mask = _attach_base_amount(df, fx, base_currency=base_currency)
            df["base_amount"] = base_amount
    else:
        df["base_amount"] = pd.NA

    if add_time_period:
        p = period_bins_for_dates(df["Date"], freq=time_freq)
        if isinstance(p, pd.DataFrame) and "TimePeriod" in p.columns:
            df["TimePeriod"] = p["TimePeriod"].astype(str)
        else:
            df["TimePeriod"] = df["Date"].dt.to_period(time_freq).astype(str)

    anomalies = _collect_anomalies(df, fx_missing_mask=fx_missing_mask)
    df.attrs["anomalies"] = anomalies

    preferred = [
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
        "status",
        "Box",
        "source_file",
        "source_row",
        "ingest_ts",
        "notes",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].copy()


def build_stable_ledger_snapshot(
    fixture_path: Optional[str] = None,
    sheet_url: Optional[str] = None,
    service_account_file: Optional[str] = None,
    sheet_name: str = "LEDGERS",
    party_map_path: Optional[str] = None,
    fx_rates_path: Optional[str] = "",
    base_currency: str = "ARS",
    require_tx_id: bool = False,
    exclude_household: bool = False,
    boxes: Optional[set[str]] = None,
    only_status: Optional[Sequence[str]] = ("pagado",),
    add_time_period: bool = False,
    time_freq: str = "W",
    drop_volatile: bool = True,
) -> pd.DataFrame:
    """
    Build a deterministic ledger snapshot suitable for change detection.

    This reuses build_ledger_base() and then removes volatile/non-business fields,
    normalizes types, sorts rows deterministically, and sorts columns
    alphabetically so equivalent ledger content yields the same serialized bytes.
    """
    df = build_ledger_base(
        fixture_path=fixture_path,
        sheet_url=sheet_url,
        service_account_file=service_account_file,
        sheet_name=sheet_name,
        party_map_path=party_map_path,
        fx_rates_path=fx_rates_path,
        base_currency=base_currency,
        require_tx_id=require_tx_id,
        exclude_household=exclude_household,
        boxes=boxes,
        only_status=only_status,
        add_time_period=add_time_period,
        time_freq=time_freq,
    ).copy()

    if drop_volatile:
        df = df.drop(columns=[c for c in ["ingest_ts"] if c in df.columns], errors="ignore")

    for c in df.columns:
        if c == "Date":
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date.astype("string")
        else:
            df[c] = df[c].astype("string")

    df = df.fillna("")

    sort_cols = [c for c in ["tx_id", "Date", "source_row"] if c in df.columns]
    if not sort_cols:
        sort_cols = list(df.columns)
    df = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    return df.reindex(sorted(df.columns), axis=1)


def compute_ledger_fingerprint(
    fixture_path: Optional[str] = None,
    sheet_url: Optional[str] = None,
    service_account_file: Optional[str] = None,
    sheet_name: str = "LEDGERS",
    party_map_path: Optional[str] = None,
    fx_rates_path: Optional[str] = "",
    base_currency: str = "ARS",
    require_tx_id: bool = False,
    exclude_household: bool = False,
    boxes: Optional[set[str]] = None,
    only_status: Optional[Sequence[str]] = ("pagado",),
    add_time_period: bool = False,
    time_freq: str = "W",
) -> str:
    """
    Return a stable SHA-256 fingerprint for the business-relevant ledger content.
    """
    df = build_stable_ledger_snapshot(
        fixture_path=fixture_path,
        sheet_url=sheet_url,
        service_account_file=service_account_file,
        sheet_name=sheet_name,
        party_map_path=party_map_path,
        fx_rates_path=fx_rates_path,
        base_currency=base_currency,
        require_tx_id=require_tx_id,
        exclude_household=exclude_household,
        boxes=boxes,
        only_status=only_status,
        add_time_period=add_time_period,
        time_freq=time_freq,
    )
    payload = df.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -----------------------
# CLI (thin wrapper)
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ingest and write ledger_canonical.csv")
    p.add_argument("--fixture", help="Local fixture CSV/Parquet path", default=os.getenv("FIXTURE"))
    p.add_argument(
        "--service-account",
        help="Google service account JSON path",
        default=os.getenv("SERVICE_ACCOUNT") or os.getenv("ACCOUNT_SERVICE_ACCOUNT"),
    )
    p.add_argument("--sheet-url", help="Google Sheet URL", default=os.getenv("SHEET_URL") or os.getenv("ACCOUNT_SHEET_URL"))
    p.add_argument("--sheet-name", help="Sheet/tab name (when using Google Sheets)", default=os.getenv("SHEET_NAME", "C. Long Ledger"))
    p.add_argument("--out-dir", help="Output directory", default=os.getenv("OUT_DIR", "./out"))
    p.add_argument("--probe-fingerprint", action="store_true", default=False, help="Print stable ledger fingerprint and exit")

    p.add_argument(
        "--exclude-household",
        action="store_true",
        default=False,
        help="Legacy unscoped-ingest compatibility flag; do not combine with --boxes.",
    )
    p.add_argument(
        "--boxes",
        default=None,
        help="Comma-separated canonical Box universe. Supersedes legacy --exclude-household.",
    )
    p.add_argument("--require-tx-id", action="store_true", default=False)

    p.add_argument(
        "--only-status",
        default=os.getenv("ONLY_STATUS", "pagado"),
        help="Comma or space separated status filter. Use empty string to disable filtering.",
    )

    p.add_argument("--fx-rates", default=os.getenv("FX_RATES", ""))
    p.add_argument("--base-currency", default=os.getenv("BASE_CURRENCY", "ARS"))
    p.add_argument("--add-time-period", action="store_true", default=bool(int(os.getenv("ADD_TIME_PERIOD", "0"))))
    p.add_argument("--time-freq", default=os.getenv("TIME_FREQ", "W"))

    p.add_argument("--mode", choices=["smoke", "run"], default=os.getenv("MODE", "run"))
    p.add_argument("--run-id", default=os.getenv("RUN_ID", ""))

    return p.parse_args()


def _parse_list_arg(s: str) -> Optional[List[str]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return parts or None




def main() -> int:
    configure_logging()

    args = parse_args()
    only_status = _parse_list_arg(args.only_status)
    boxes = parse_box_scope(args.boxes) if args.boxes is not None else None

    from accounting.artifacts.manifest import artifact_from_path, append_artifacts, write_stage_manifest
    from accounting.support.run_id import resolve_run_id

    if args.probe_fingerprint:
        fp = compute_ledger_fingerprint(
            fixture_path=args.fixture or None,
            sheet_url=args.sheet_url or None,
            service_account_file=args.service_account or None,
            sheet_name=args.sheet_name,
            party_map_path=os.getenv("PARTY_MAP") or None,
            fx_rates_path=(args.fx_rates or ""),
            base_currency=args.base_currency,
            require_tx_id=bool(args.require_tx_id),
            exclude_household=bool(args.exclude_household),
            boxes=boxes,
            only_status=only_status if only_status is not None else None,
            add_time_period=bool(args.add_time_period),
            time_freq=str(args.time_freq),
        )
        print(fp)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Stage start mode=%s out_dir=%s", args.mode, out_dir)

    ledger_all_status = build_ledger_base(
        fixture_path=args.fixture or None,
        sheet_url=args.sheet_url or None,
        service_account_file=args.service_account or None,
        sheet_name=args.sheet_name,
        party_map_path=os.getenv("PARTY_MAP") or None,
        fx_rates_path=(args.fx_rates or ""),
        base_currency=args.base_currency,
        require_tx_id=bool(args.require_tx_id),
        exclude_household=bool(args.exclude_household),
        boxes=boxes,
        only_status=None,
        add_time_period=bool(args.add_time_period),
        time_freq=str(args.time_freq),
    )
    ledger = filter_ledger_statuses(ledger_all_status, only_status)

    ledger_path = out_dir / "ledger_canonical.csv"
    all_status_path = out_dir / "ledger_canonical_all_status.csv"
    all_status_df = ledger_all_status.copy()
    if "Date" in all_status_df.columns:
        all_status_df["Date"] = pd.to_datetime(all_status_df["Date"], errors="coerce").dt.date.astype(str)
    all_status_df.to_csv(all_status_path, index=False)
    LOG.info("Wrote scoped all-status evidence rows=%d -> %s", len(all_status_df), all_status_path)
    ldf = ledger.copy()
    if "Date" in ldf.columns:
        ldf["Date"] = pd.to_datetime(ldf["Date"], errors="coerce").dt.date.astype(str)
    ldf.to_csv(ledger_path, index=False)
    LOG.info("Wrote ledger_canonical rows=%d -> %s", len(ldf), ledger_path)

    anoms = ledger.attrs.get("anomalies")
    if isinstance(anoms, pd.DataFrame) and not anoms.empty:
        anoms_path = out_dir / "anomalies.csv"
        anoms.to_csv(anoms_path, index=False)
        LOG.info("Wrote anomalies rows=%d -> %s", len(anoms), anoms_path)

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved_run_id = resolve_run_id(
            mode=args.mode,
            run_id=getattr(args, "run_id", None),
            root_dir=out_dir,
            strict=True,
        )

        out_art = artifact_from_path(
            name="ledger_canonical",
            path=ledger_path,
            stage="A.ingest",
            mode=args.mode,
            run_id=resolved_run_id,
            role="output",
            root_dir=out_dir,
            content_type="text/csv",
        )

        all_status_art = artifact_from_path(
            name="ledger_canonical_all_status",
            path=all_status_path,
            stage="A.ingest",
            mode=args.mode,
            run_id=resolved_run_id,
            role="output",
            root_dir=out_dir,
            content_type="text/csv",
        )

        out_arts = [all_status_art, out_art]
        if isinstance(anoms, pd.DataFrame) and not anoms.empty:
            out_arts.append(
                artifact_from_path(
                    name="anomalies",
                    path=(out_dir / "anomalies.csv"),
                    stage="A.ingest",
                    mode=args.mode,
                    run_id=resolved_run_id,
                    role="derived",
                    root_dir=out_dir,
                    content_type="text/csv",
                )
            )

        stage_manifest = {
            "stage": "A.ingest",
            "mode": args.mode,
            "run_id": resolved_run_id,
            "generated_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat(),
            "inputs": [],
            "params": {
                "only_status": only_status,
                "exclude_household": int(bool(args.exclude_household)),
                "require_tx_id": int(bool(args.require_tx_id)),
                "fx_rates": bool(args.fx_rates),
                "base_currency": args.base_currency,
                **(scope_metadata(boxes) if boxes is not None else {}),
            },
            "outputs": out_arts,
            "warnings": [],
        }

        rel = write_stage_manifest(meta_dir, stage_manifest)

        stage_meta_art = artifact_from_path(
            name="stage_A_ingest",
            path=(out_dir / rel),
            stage="A.ingest",
            mode=args.mode,
            run_id=resolved_run_id,
            role="meta",
            root_dir=out_dir,
            content_type="application/json",
        )

        append_artifacts(meta_dir, [*out_arts, stage_meta_art])

    except Exception:
        if str(args.mode).strip().lower() == "run":
            LOG.exception("Manifest write failed (fatal in run mode)")
            return 4


if __name__ == "__main__":
    raise SystemExit(main())
