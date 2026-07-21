"""Canonical accounting-universe governance helpers."""

from __future__ import annotations

import os
import pandas as pd


PROPERTY_BUSINESS_BOXES = {"Family Business", "Property Management"}
HOUSEHOLD_BOXES = {"Household"}


def parse_box_scope(value: str | None) -> set[str]:
    """Parse a comma-separated Box selection and reject an empty universe."""
    boxes = {part.strip() for part in (value or "").split(",") if part.strip()}
    if not boxes:
        raise ValueError("Box scope must name at least one Box")
    return boxes


def box_scope_mask(df: pd.DataFrame, boxes: set[str]) -> pd.Series:
    """Return rows whose owning Box belongs to a materialized run's universe."""
    if "Box" not in df.columns:
        # Legacy drilldown fixtures and historical extracts can predate the
        # Box dimension. They are already run-scoped and cannot be narrowed
        # further without silently dropping their entire evidence set.
        return pd.Series(True, index=df.index)
    return df["Box"].astype(str).str.strip().isin(boxes)


def configured_box_scope() -> set[str]:
    """Return the run universe selected by BOXES, defaulting to PM/FB."""
    return parse_box_scope(os.getenv("BOXES") or "Family Business,Property Management")


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
