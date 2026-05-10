"""Partition metadata and parquet write helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

LOG = logging.getLogger(__name__)


def _atomic_write_parquet(df: pd.DataFrame, dest: Path, partition_cols: Optional[Sequence[str]] = None, **kwargs) -> Path:
    """
    Write parquet atomically (temp -> rename). `partition_cols` forwarded to pandas.to_parquet if provided.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if partition_cols:
        df.to_parquet(tmp, partition_cols=list(partition_cols), index=False, engine="pyarrow", **kwargs)
        if dest.exists():
            if dest.is_file():
                dest.unlink()
        tmp_path = Path(tmp)
        if dest.exists():
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
