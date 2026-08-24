"""Attach governed FX grain identity from stable producer metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.contracts.fx_reporting_grain import (
    FX_REPORTING_GRAIN_COLUMN,
    producer_fx_reporting_grain,
)


def enrich_fx_reporting_grain_tables(tables_dir: Path) -> list[Path]:
    tables_dir = Path(tables_dir)
    written: list[Path] = []
    if not tables_dir.exists():
        return written
    for path in sorted(tables_dir.glob("*.csv")):
        table_id = path.stem
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        grains = [producer_fx_reporting_grain(table_id, row) or "" for _, row in frame.iterrows()]
        if not any(grains):
            continue
        out = frame.copy()
        out[FX_REPORTING_GRAIN_COLUMN] = grains
        out.to_csv(path, index=False)
        written.append(path)
    return written
