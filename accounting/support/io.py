"""CSV and manifest I/O helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd


def _read_csv_if_exists(p: Path, **kwargs) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False, **kwargs)


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
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path)
    try:
        obj.to_csv(tmp_path, index=index, date_format=date_format)
        tmp_path.replace(path)
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
