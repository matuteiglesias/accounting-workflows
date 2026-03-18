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



# def _resolve_run_id(args: argparse.Namespace) -> str:
#     if args.run_id:
#         return str(args.run_id)
#     return "smoke" if args.mode == "smoke" else ""



# def _resolve_run_id(mode: str, run_id: str) -> str:
#     rid = str(run_id or "").strip()
#     if rid:
#         return rid
#     return "smoke" if str(mode).strip().lower() == "smoke" else ""

 

# import os
import re
# from pathlib import Path
from typing import Optional, Union

_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")  # e.g. 20260109T142110Z

def _infer_run_id_from_path(root_dir: Union[str, Path]) -> Optional[str]:
    """
    Scan the directory and its parents for a folder name that looks like a RUN_ID.
    This lets stages infer run_id when orchestrator didn't pass --run-id.
    """
    p = Path(root_dir).resolve()
    for cand in [p, *p.parents]:
        name = cand.name.strip()
        if _RUN_ID_RE.match(name):
            return name
    return None

def resolve_run_id(
    *,
    mode: str,
    run_id: Optional[str] = None,
    root_dir: Optional[Union[str, Path]] = None,
    env_var: str = "RUN_ID",
    strict: bool = True,
) -> str:
    """
    Canonical RUN_ID resolution.

    Precedence:
      1) explicit run_id argument
      2) environment variable RUN_ID (configurable)
      3) infer from root_dir (or its parents)
      4) if mode == smoke -> "smoke"
      5) else: error (strict) or fallback "untracked"

    In run mode, returning "" is forbidden.
    """
    m = (mode or "").strip().lower()

    rid = (run_id or "").strip()
    if rid and rid.lower() != "none":
        return rid

    env_rid = (os.getenv(env_var) or "").strip()
    if env_rid:
        return env_rid

    if root_dir is not None:
        inferred = _infer_run_id_from_path(root_dir)
        if inferred:
            return inferred

    if m == "smoke":
        return "smoke"

    if strict:
        raise ValueError(
            "run_id missing for non-smoke run. "
            "Pass --run-id, set RUN_ID, or run inside out/run/accounting/<RUN_ID>/"
        )
    return "untracked"




# Requires gspread

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


# accounting/utils.py
import pandas as pd

def _normalize_currency_col(
    df: pd.DataFrame,
    *,
    allow_missing: bool = False,
    out_col: str = "Currency",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if out_col in out.columns:
        col = out_col
    elif "currency" in out.columns:
        out = out.rename(columns={"currency": out_col})
        col = out_col
    else:
        # if allow_missing:
        out[out_col] = pd.NA
        return out
        # raise KeyError(f"Missing required column '{out_col}' (or alias 'currency')")

    s = out[col].astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, "NAN": pd.NA, "NA": pd.NA})
    out[col] = s
    return out

def require_currency(df: pd.DataFrame, *, name: str, col: str = "Currency") -> pd.DataFrame:
    out = _normalize_currency_col(df, allow_missing=False, out_col=col)
    if out[col].isna().any():
        # raise ValueError(f"{name} has null/empty values in '{col}'")
        print(f"{name} has {str(out[col].isna().sum())} null/empty values in '{col}'")
    return out





# def _require_currency(df: pd.DataFrame, name: str) -> pd.DataFrame:
#     """
#     Enforce that a canonical `Currency` column exists, is non-null, and non-empty after normalization.
#     This intentionally fails fast instead of patching.
#     """
#     out = _normalize_currency_col(df.copy())
#     if "Currency" not in out.columns:
#         raise KeyError(f"{name} missing required column 'Currency' after normalization")

#     cur = out["Currency"]
#     if cur.isna().any():
#         raise ValueError(f"{name} has null values in 'Currency'")

#     # Reject empty/blank strings (common leak path)
#     cur_str = cur.astype(str).str.strip()
#     if (cur_str == "").any():
#         raise ValueError(f"{name} has blank/empty values in 'Currency'")

#     out["Currency"] = cur_str
#     return out



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


