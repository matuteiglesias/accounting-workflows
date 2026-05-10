"""Hashing helpers for files and source dataframes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def sha256_file(path: Path) -> str:
    """Return sha256 hex digest for given file path."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_source_hash(ledger: pd.DataFrame, keys: Optional[Sequence[str]] = None) -> str:
    """
    Computes a reproducible sha256 hash for the ledger's essential identity.
    Default uses tx_id, Date, amount_cents. If `keys` passed, use those columns (in order).
    """
    if keys is None:
        keys = ["tx_id", "Date", "amount_cents"]
    missing = [k for k in keys if k not in ledger.columns]
    if missing:
        payload = f"{len(ledger)}|{pd.to_datetime(ledger.get('Date')).max()}|{int(ledger.get('amount_cents', 0).sum())}"
        return hashlib.sha256(payload.encode("utf8")).hexdigest()
    subset = ledger.loc[:, keys].copy()
    subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    csv = subset.sort_values(list(keys), na_position="first").to_csv(index=False).encode("utf8")
    return hashlib.sha256(csv).hexdigest()
