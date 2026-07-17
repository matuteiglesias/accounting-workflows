"""Canonical accounting-universe governance helpers."""

from __future__ import annotations

import pandas as pd


PROPERTY_BUSINESS_BOXES = {"Family Business", "Property Management"}
HOUSEHOLD_BOXES = {"Household"}


def property_business_scope_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows attributable to the property/business accounting universe.

    A Household cash row is included only when an explicit semantic dimension
    attributes its target, beneficiary, or obligation to FB/PM.  Its semantic
    bucket alone never grants entry to the professional reporting universe.
    """
    mask = pd.Series(False, index=df.index)

    if "Box" in df.columns:
        mask |= df["Box"].astype(str).isin(PROPERTY_BUSINESS_BOXES)

    for col in ["target_box", "beneficiary_box", "obligation_box"]:
        if col in df.columns:
            mask |= df[col].astype(str).isin(PROPERTY_BUSINESS_BOXES)

    return mask
