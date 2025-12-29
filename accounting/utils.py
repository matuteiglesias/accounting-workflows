from pathlib import Path
import hashlib
import json
import pandas as pd
import tempfile
import os
from typing import Dict
import logging
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Reuires gspread

def sha256_file(path: Path) -> str:
    """Return sha256 hex digest for given file path."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()



import os

def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var {name}. Set it in private/.env (not committed) or export it.")
    return v

ACCOUNT_SHEET_URL = os.getenv("ACCOUNT_SHEET_URL", "").strip()
RENTALS_SHEET_URL = os.getenv("RENTALS_SHEET_URL", "").strip()
SERVICE_ACCOUNT_FILE = os.getenv("ACCOUNT_SA", "").strip()




def _read_csv_if_exists(p: Path, **kwargs) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False, **kwargs)

def _normalize_currency_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a canonical 'currency' column (uppercase, 'NA' for missing).
    Avoid creating multiple columns; always set 'currency' lowercase.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # accept 'Currency' or 'currency' or none
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.upper().fillna("NA").replace({"NAN": "NA"})
    elif "Currency" in df.columns:
        df["currency"] = df["Currency"].astype(str).str.upper().fillna("NA").replace({"NAN": "NA"})
    else:
        df["currency"] = "NA"
    return df


def _ensure_amount(df: pd.DataFrame, amount_cols=("amount","signed_amount","_amt","Monto")) -> pd.DataFrame:
    """
    Ensure a numeric 'amount' column exists and is float.
    Prefer existing 'amount', else try fallbacks.
    """
    df = df.copy()
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        return df
    for c in amount_cols:
        if c in df.columns:
            df["amount"] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return df
    # fallback: create zero amount to avoid crashes downstream
    df["amount"] = 0.0
    return df

def _find_first_existing(base: Path, patterns, freq: str) -> Path | None:
    for pat in patterns:
        candidate = base / pat.format(freq=freq)
        if candidate.exists():
            return candidate
    return None



def atomic_write_df(obj: pd.DataFrame, path: Path, index: bool = True, date_format: str = None) -> None:
    """
    Atomically write a DataFrame to CSV at `path`.
    Writes to a temporary file in the same directory and then renames.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use tempfile in target dir to ensure atomic move is on same FS
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path)
    try:
        # pandas will write to the tmp location
        obj.to_csv(tmp_path, index=index, date_format=date_format)
        tmp_path.replace(path)  # atomic move on POSIX
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

def save_manifest(manifest_path: Path, records: Dict) -> None:
    """Write manifest JSON (pretty) atomically."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(manifest_path)




### canonicalization/time helpers
def convert_currency(df, numeric_columns, target_currency="USD"):
    if "Currency" in df.columns and "Rate" in df.columns:
        if target_currency == "USD":
            mask = df["Currency"] == "ARS"
            df.loc[mask, numeric_columns] = (
                df.loc[mask, numeric_columns].div(df.loc[mask, "Rate"], axis=0)
            )
        elif target_currency == "ARS":
            mask = df["Currency"] == "USD"
            df.loc[mask, numeric_columns] = (
                df.loc[mask, numeric_columns].multiply(df.loc[mask, "Rate"], axis=0)
            )
        df["Currency"] = target_currency
    df = df.drop(columns=["Rate"], errors="ignore")
    return df

from typing import Sequence, Optional, Tuple, Dict, Any

# -----------------------
# Helpers: hashing / partitions
# -----------------------
def compute_source_hash(ledger: pd.DataFrame, keys: Optional[Sequence[str]] = None) -> str:
    """
    Computes a reproducible sha256 hash for the ledger's essential identity.
    Default uses tx_id, Date, amount_cents. If `keys` passed, use those columns (in order).
    """
    if keys is None:
        keys = ["tx_id", "Date", "amount_cents"]
    missing = [k for k in keys if k not in ledger.columns]
    if missing:
        # fallback to rowcount + max date + sum amounts (less collision-proof but safe)
        payload = f"{len(ledger)}|{pd.to_datetime(ledger.get('Date')).max()}|{int(ledger.get('amount_cents', 0).sum())}"
        return hashlib.sha256(payload.encode("utf8")).hexdigest()
    subset = ledger.loc[:, keys].copy()
    # Normalise Date into ISO and ensure deterministic ordering
    subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    csv = subset.sort_values(list(keys), na_position="first").to_csv(index=False).encode("utf8")
    return hashlib.sha256(csv).hexdigest()


def _atomic_write_parquet(df: pd.DataFrame, dest: Path, partition_cols: Optional[Sequence[str]] = None, **kwargs) -> Path:
    """
    Write parquet atomically (temp -> rename). `partition_cols` forwarded to pandas.to_parquet if provided.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    # pandas will create directory if needed for partitioning
    if partition_cols:
        df.to_parquet(tmp, partition_cols=list(partition_cols), index=False, engine="pyarrow", **kwargs)
        # pandas writes a directory for partitioned output; move tmp -> dest (dest should be a dir path)
        if dest.exists():
            # remove previous
            if dest.is_file():
                dest.unlink()
        # rename/move tmp -> dest (tmp here is a directory created by pandas)
        tmp_path = Path(tmp)
        # if user passed dest as filename, we keep behaviour: treat dest as directory for partitioned output
        if dest.exists():
            # overwrite behavior: remove then rename
            if dest.is_dir():
                for p in dest.iterdir():
                    if p.is_file():
                        p.unlink()
        tmp_path.rename(dest)
    else:
        df.to_parquet(tmp, index=False, engine="pyarrow", **kwargs)
        tmp.replace(dest)
    return dest


def load_partitions_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        LOG.exception("Failed loading partitions json: %s", path)
        return {}


def save_partitions_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)





# # ---- Small utilities -----------------------------------------------------
# def _safe_read_parquet(p: Path) -> pd.DataFrame:
#     p = Path(p)
#     if not p.exists():
#         return pd.DataFrame()
#     return pd.read_parquet(p)

# def _to_major_units(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
#     """Convert amount_cents int columns -> float in major units (divide by 100)."""
#     out = df.copy()
#     for c in cols:
#         if c in out.columns:
#             out[c] = out[c].astype("Int64") / 100.0
#     return out





import pandas as pd
import gspread


def get_google_sheets_client(service_account_file):
    from google.oauth2 import service_account
    import gspread

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(credentials)
    return client


def load_google_sheet(client, sheet_url, sheet_name):
    spreadsheet = client.open_by_url(sheet_url)
    worksheet = spreadsheet.worksheet(sheet_name)
    data = worksheet.get_all_values()
    if not data or len(data) < 2:
        raise ValueError(f"Sheet '{sheet_name}' is empty or missing headers.")
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


