"""Shared physical Box cash-counterparty matching primitives."""

from __future__ import annotations

from typing import Any

import pandas as pd


def infer_box_party(box: Any) -> str:
    """Return the canonical party token implied by a Box name."""
    if box is None or pd.isna(box):
        return ""
    text = str(box).strip()
    if not text:
        return ""
    if text.casefold() == "household":
        return "HH"
    return "".join(
        part[0].upper()
        for part in text.split()
        if part and part[0].isalpha()
    )


def box_party_match_masks(
    df: pd.DataFrame,
    *,
    require_nonempty_box_party: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Return physical cash-in / cash-out masks for the row's Box."""
    required = {"Box", "payer", "receiver"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Box cash match requires columns: {missing}")

    box_party = (
        df["Box"]
        .map(infer_box_party)
        .astype("string")
        .str.strip()
        .str.upper()
    )
    payer = df["payer"].astype("string").str.strip().str.upper()
    receiver = df["receiver"].astype("string").str.strip().str.upper()

    matched_in = receiver.eq(box_party)
    matched_out = payer.eq(box_party)
    if require_nonempty_box_party:
        nonempty = box_party.ne("")
        matched_in &= nonempty
        matched_out &= nonempty
    return matched_in.fillna(False), matched_out.fillna(False)
