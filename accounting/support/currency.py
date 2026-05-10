"""Currency and amount normalization helpers."""

from __future__ import annotations

import logging

import pandas as pd

LOG = logging.getLogger(__name__)


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
        out[out_col] = pd.NA
        return out

    s = out[col].astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, "NAN": pd.NA, "NA": pd.NA})
    out[col] = s
    return out


def require_currency(df: pd.DataFrame, *, name: str, col: str = "Currency") -> pd.DataFrame:
    out = _normalize_currency_col(df, allow_missing=False, out_col=col)
    if out[col].isna().any():
        LOG.warning("%s has %s null/empty values in %r", name, int(out[col].isna().sum()), col)
    return out


def _ensure_amount(df: pd.DataFrame, amount_cols=("amount", "signed_amount", "_amt", "Monto")) -> pd.DataFrame:
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
    df["amount"] = 0.0
    return df


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
