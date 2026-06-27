# src/accounting/views.py
from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import pandas as pd

from accounting.logging_utils import configure_logging, get_logger
from accounting.support.currency import _ensure_amount, require_currency
from accounting.support.io import _find_first_existing, _read_csv_if_exists, atomic_write_df
from accounting.support.run_id import resolve_run_id

from accounting.core.timeseries import period_bins_for_dates


FONDOS_FN = "fondos_report.csv"
RENTA_GLOB = "renta_*.csv"
LEDGER_FN = "ledger_canonical.csv"

_PER_PARTY_PATTERNS = [
    "per_party_time_long.freq={freq}.csv",
    "per_party_time_long.freq=M.csv",
    "per_party_time_long.csv",
]
_PER_FLOW_PATTERNS = [
    "per_flow_time_long.freq={freq}.csv",
    "per_flow_time_long.freq=M.csv",
    "per_flow_time_long.csv",
]


_BOX_BALANCE_PATTERNS = [
    "box_balance_time_long.freq={freq}.csv",
    "box_balance_time_long.freq=M.csv",
    "box_balance_time_long.csv",
]
_BOX_FLOW_BALANCE_PATTERNS = [
    "box_flow_balance_time_long.freq={freq}.csv",
    "box_flow_balance_time_long.freq=M.csv",
    "box_flow_balance_time_long.csv",
]
_DAILY_CASH_FN = "daily_cash_position.csv"
_MANIFEST_FN = "manifest.json"

LOG = get_logger("views")

def _legacy_zero_sum_outputs_enabled() -> bool:
    return str(os.getenv("KEEP_LEGACY_ZERO_SUM_OUTPUTS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


# -----------------------
# Small contract helpers
# -----------------------
def _assert_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def _require_nonempty_col(df: pd.DataFrame, cols: List[str], name: str) -> None:
    """Assert that required columns exist and have no empty/blank values."""
    _assert_cols(df, cols, name)
    for c in cols:
        s = df[c]
        if s.isna().any():
            raise ValueError(f"{name} has NA values in required column '{c}'")
        # treat empty strings as invalid for key columns
        if (s.astype(str).str.strip() == "").any():
            raise ValueError(f"{name} has blank values in required column '{c}'")

def _parse_date_col(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    elif "TimePeriod_ts_end" in out.columns:
        out["Date"] = pd.to_datetime(out["TimePeriod_ts_end"], errors="coerce")
    elif "TimePeriod_end" in out.columns:
        out["Date"] = pd.to_datetime(out["TimePeriod_end"], errors="coerce")
    elif "Date_end" in out.columns:
        out["Date"] = pd.to_datetime(out["Date_end"], errors="coerce")
    elif "TimePeriod" in out.columns:
        out["Date"] = pd.to_datetime(out["TimePeriod"].astype(str), errors="coerce")
    else:
        raise KeyError(
            f"{name} missing date column (expected Date/TimePeriod_ts_end/TimePeriod_end/Date_end/TimePeriod)"
        )
    return out

def _df_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows": 0, "date_min": None, "date_max": None, "currencies": []}
    date_min = date_max = None
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce")
        if d.notna().any():
            date_min = d.min().date().isoformat()
            date_max = d.max().date().isoformat()
    currencies: List[str] = []
    if "Currency" in df.columns:
        currencies = sorted([c for c in df["Currency"].dropna().astype(str).unique().tolist() if c.strip() != ""])
    return {"rows": int(len(df)), "date_min": date_min, "date_max": date_max, "currencies": currencies}


def _is_period_aggregated(df: pd.DataFrame) -> bool:
    """
    Heuristic: materialized time-aggregation artifacts from core_timeseries carry TimePeriod[_ts_end].
    If present, treat the data as already binned and do not re-bin in views.
    """
    return ("TimePeriod_ts_end" in df.columns) or ("TimePeriod" in df.columns)


def _signed_from_materialized_per_party(perp: pd.DataFrame) -> pd.Series:
    """
    Contract:
      - per_party_time_long.amount is already signed (it is the sum of signed_amount upstream).
      - views must not re-sign via role/Flujo/Tipo heuristics.
    We still require `role` to be present so we can sanity-check sign polarity.
    """
    _assert_cols(perp, ["amount", "role"], "per_party_time_long")
    return pd.to_numeric(perp["amount"], errors="coerce").fillna(0.0).astype(float)


def _compute_in_out_net(df: pd.DataFrame, signed_col: str = "signed") -> pd.DataFrame:
    out = df.copy()
    signed = pd.to_numeric(out[signed_col], errors="coerce").fillna(0.0).astype(float)
    out["in_amt"] = signed.clip(lower=0.0)
    out["out_amt"] = (-signed.clip(upper=0.0))
    out["net"] = out["in_amt"] - out["out_amt"]
    return out


# -----------------------
# Loading
# -----------------------
def load_reports_folder(reports_dir: Path, freq: str = "M") -> Dict[str, Any]:
    """
    Load Stage D artifacts (materialize outputs) plus optional legacy report artifacts.

    Contract:
      - Stage D artifacts are the source of truth for Views.
      - Legacy report artifacts (fondos_report.csv, renta_*.csv) are best-effort only and must never be required.
    """
    reports_dir = Path(reports_dir)
    out: Dict[str, Any] = {}
    paths: Dict[str, Optional[str]] = {}
    load_warnings: List[str] = []

    parent_out = reports_dir.parent

    # Stage D aggregates
    per_party_path = _find_first_existing(parent_out, _PER_PARTY_PATTERNS, freq=freq) or _find_first_existing(
        reports_dir, _PER_PARTY_PATTERNS, freq=freq
    )
    per_flow_path = _find_first_existing(parent_out, _PER_FLOW_PATTERNS, freq=freq) or _find_first_existing(
        reports_dir, _PER_FLOW_PATTERNS, freq=freq
    )
    paths["per_party_time_long"] = str(per_party_path) if per_party_path else None
    paths["per_flow_time_long"] = str(per_flow_path) if per_flow_path else None
    out["per_party_time_long"] = _read_csv_if_exists(per_party_path) if per_party_path else pd.DataFrame()
    out["per_flow_time_long"] = _read_csv_if_exists(per_flow_path) if per_flow_path else pd.DataFrame()

    # Box motor tables (recommended inputs for narrative views)
    box_bal_path = _find_first_existing(parent_out, _BOX_BALANCE_PATTERNS, freq=freq) or _find_first_existing(
        reports_dir, _BOX_BALANCE_PATTERNS, freq=freq
    )
    box_flow_bal_path = _find_first_existing(parent_out, _BOX_FLOW_BALANCE_PATTERNS, freq=freq) or _find_first_existing(
        reports_dir, _BOX_FLOW_BALANCE_PATTERNS, freq=freq
    )
    paths["box_balance_time_long"] = str(box_bal_path) if box_bal_path else None
    paths["box_flow_balance_time_long"] = str(box_flow_bal_path) if box_flow_bal_path else None
    out["box_balance_time_long"] = _read_csv_if_exists(box_bal_path) if box_bal_path else pd.DataFrame()
    out["box_flow_balance_time_long"] = _read_csv_if_exists(box_flow_bal_path) if box_flow_bal_path else pd.DataFrame()

    # Canonical ledger (optional input for convenience extracts only)
    p_ledger = parent_out / LEDGER_FN
    if not p_ledger.exists():
        p_ledger = reports_dir / LEDGER_FN
    paths["ledger"] = str(p_ledger) if p_ledger.exists() else None
    out["ledger"] = _read_csv_if_exists(p_ledger, parse_dates=["Date"]) if p_ledger.exists() else pd.DataFrame()

    # Daily cash (optional)
    p_daily = parent_out / _DAILY_CASH_FN
    if not p_daily.exists():
        p_daily = reports_dir / _DAILY_CASH_FN
    paths["daily_cash_position"] = str(p_daily) if p_daily.exists() else None
    out["daily_cash_position"] = _read_csv_if_exists(p_daily) if p_daily.exists() else pd.DataFrame()

    # Legacy artifacts (best-effort only). Keep for backwards compatibility but do not depend on them.
    p_fondos = reports_dir / FONDOS_FN
    if p_fondos.exists():
        out["fondos_legacy"] = _read_csv_if_exists(p_fondos, index_col=0, dtype=object)
        paths["fondos_legacy"] = str(p_fondos)
        load_warnings.append("Loaded legacy fondos_report.csv. Prefer fondos_wide built from per_party_time_long.")
    else:
        out["fondos_legacy"] = pd.DataFrame()
        paths["fondos_legacy"] = None

    renta_dfs = []
    for p in sorted(reports_dir.glob(RENTA_GLOB)):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            load_warnings.append(f"Failed reading legacy renta file {p.name}: {e}")
            continue

        # best-effort normalization of common renta layouts
        if "TimePeriod_ts_end" in df.columns and "amount" in df.columns:
            df = df.rename(columns={"TimePeriod_ts_end": "Date"})
        elif "Date" not in df.columns and df.shape[1] >= 2:
            df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "amount"})

        df = _parse_date_col(df, name=f"renta:{p.name}")
        df = _ensure_amount(df)
        df = require_currency(df, name=f"renta:{p.name}")

        df["party"] = p.stem.replace("renta_", "")
        # Box is required for invariant-safe pivots. If missing, keep as legacy-only and do not crash Views.
        if "Box" not in df.columns:
            load_warnings.append(f"Legacy renta file {p.name} missing Box column; it will be loaded without Box.")
            df["Box"] = "(missing)"
        renta_dfs.append(df[["Date", "amount", "Box", "party", "Currency"]])

    out["renta_all_legacy"] = (
        pd.concat(renta_dfs, ignore_index=True)
        if renta_dfs
        else pd.DataFrame(columns=["Date", "amount", "Box", "party", "Currency"])
    )
    paths["renta_all_legacy_glob"] = str(reports_dir / RENTA_GLOB)

    # Manifest (optional)
    p_manifest = parent_out / _MANIFEST_FN
    if not p_manifest.exists():
        p_manifest = reports_dir / _MANIFEST_FN
    paths["_manifest_path"] = str(p_manifest) if p_manifest.exists() else None
    if p_manifest.exists():
        try:
            out["_manifest"] = json.loads(p_manifest.read_text(encoding="utf-8"))
        except Exception:
            out["_manifest"] = {}
    else:
        out["_manifest"] = {}

    # Normalize + enforce currency for accountant-facing inputs
    for key in ("per_party_time_long", "per_flow_time_long", "box_balance_time_long", "box_flow_balance_time_long", "daily_cash_position", "ledger"):
        df = out.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _ensure_amount(df) if ("amount" in df.columns or "amount_cents" in df.columns) else df
            df = require_currency(df, name=key)

            # Box invariant for Stage D and decision-grade inputs.
            # Ledger is allowed to be raw-ish, but still should contain Box for downstream story.
            if key in ("per_party_time_long", "per_flow_time_long", "box_balance_time_long", "box_flow_balance_time_long", "daily_cash_position"):
                _require_nonempty_col(df, ["Box"], name=key)

            out[key] = df

    out["_paths"] = paths
    out["_load_warnings"] = load_warnings
    return out


# -----------------------
# Views
# -----------------------
def build_renta_pivot_view(materialized: Dict[str, Any], freq: str = "M") -> pd.DataFrame:
    """
    Build a renta pivot (wide) for notebook consumption.

    Preferred source: per_party_time_long (Stage D), filtered to renta-like rows by (Flujo/Tipo).
    Legacy fallback: renta_*.csv files loaded into renta_all_legacy.

    Output:
      index: Date (period end)
      columns: (Box, party, Currency) when Box exists, otherwise (party, Currency)
      values: amount
    """
    # 1) Prefer Stage D per_party
    pp = materialized.get("per_party_time_long")
    if isinstance(pp, pd.DataFrame) and not pp.empty:
        df = pp.copy()
        df = _parse_date_col(df, name="per_party_time_long")
        df = _ensure_amount(df)
        df = require_currency(df, name="per_party_time_long")
        # Ensure required dims
        _assert_cols(df, ["party", "role"], "per_party_time_long")
        if "Box" in df.columns:
            df = df.dropna(subset=["Date", "Box", "party", "Currency"]).copy()
        else:
            df = df.dropna(subset=["Date", "party", "Currency"]).copy()

        # renta-like selector: deterministic string match on Flujo/Tipo
        renta_mask = pd.Series(False, index=df.index)
        for col in ("Flujo", "Tipo"):
            if col in df.columns:
                s = df[col].astype(str).str.lower()
                renta_mask = renta_mask | s.eq("renta") | s.str.contains(r"\brenta\b", regex=True)
        df = df.loc[renta_mask].copy()

        if not df.empty:
            group_cols = ["Date", "party", "Currency"]
            if "Box" in df.columns:
                group_cols.insert(1, "Box")
            df = df.groupby(group_cols, as_index=False)["amount"].sum()
            cols = ["party", "Currency"] if "Box" not in df.columns else ["Box", "party", "Currency"]
            pivot = (
                df.pivot_table(index="Date", columns=cols, values="amount", aggfunc="sum")
                .fillna(0.0)
                .sort_index()
            )
            return pivot

    # 2) Legacy fallback
    renta = materialized.get("renta_all_legacy")
    if not isinstance(renta, pd.DataFrame) or renta.empty:
        return pd.DataFrame()

    renta = _parse_date_col(renta, name="renta_all_legacy")
    renta = _ensure_amount(renta)
    renta = require_currency(renta, name="renta_all_legacy")
    _assert_cols(renta, ["party"], "renta_all_legacy")
    renta = renta.dropna(subset=["Date", "party", "Currency"]).copy()

    cols = ["party", "Currency"]
    if "Box" in renta.columns:
        cols = ["Box", "party", "Currency"]

    pivot = (
        renta.pivot_table(index="Date", columns=cols, values="amount", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    return pivot


def build_fondos_wide_view(materialized: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a fondos wide table for notebook consumption.

    Preferred source: per_party_time_long (Stage D). This avoids any dependency on reports-stage files.

    The table is intentionally "wide":
      index: Date (period end)
      columns: (Box, label, Currency) where label is deterministic: "<role>|<party>|<Flujo>"
      values: amount (signed, as provided by per_party_time_long)

    Parties selection:
      - If VIEWS_FONDOS_PARTIES env var is set (comma-separated), use those parties.
      - Else choose top parties by absolute movement within each Box/Currency.
    """
    pp = materialized.get("per_party_time_long")
    if not isinstance(pp, pd.DataFrame) or pp.empty:
        return pd.DataFrame()

    df = pp.copy()
    df = _parse_date_col(df, name="per_party_time_long")
    df = _ensure_amount(df)
    df = require_currency(df, name="per_party_time_long")
    _assert_cols(df, ["party", "role"], "per_party_time_long")

    # Box is required for invariant-safe pivots; if missing, bail out quietly.
    if "Box" not in df.columns:
        return pd.DataFrame()

    # Select parties
    parties_env = os.getenv("VIEWS_FONDOS_PARTIES", "").strip()
    if parties_env:
        selected = [p.strip() for p in parties_env.split(",") if p.strip()]
    else:
        # top parties by absolute movement across all flows
        tmp = df.dropna(subset=["party", "Box", "Currency"]).copy()
        scores = (
            tmp.groupby(["Box", "Currency", "party"], as_index=False)["amount"]
            .apply(lambda s: s.abs().sum())
            .rename(columns={"amount": "abs_sum"})
        )
        selected = (
            scores.sort_values(["Box", "Currency", "abs_sum"], ascending=[True, True, False])
            .groupby(["Box", "Currency"])["party"]
            .head(6)
            .unique()
            .tolist()
        )

    if not selected:
        return pd.DataFrame()

    df = df[df["party"].isin(selected)].copy()
    df = df.dropna(subset=["Date", "Box", "Currency", "party"]).copy()

    if "Flujo" not in df.columns:
        df["Flujo"] = "(missing)"
    # Build deterministic label
    df["label"] = df["role"].astype(str) + "|" + df["party"].astype(str) + "|" + df["Flujo"].astype(str)

    g = df.groupby(["Date", "Box", "Currency", "label"], as_index=False)["amount"].sum()

    pivot = (
        g.pivot_table(index="Date", columns=["Box", "label", "Currency"], values="amount", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )

    # Return as a normal DataFrame (export will reset_index)
    return pivot



def _standardize_period_end(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ensure we have `TimePeriod_end` (ISO date string) alongside `TimePeriod`."""
    out = df.copy()
    if "TimePeriod_end" in out.columns:
        out["TimePeriod_end"] = pd.to_datetime(out["TimePeriod_end"], errors="coerce").dt.date.astype(str)
        return out
    if "Date_end" in out.columns:
        out["TimePeriod_end"] = pd.to_datetime(out["Date_end"], errors="coerce").dt.date.astype(str)
        return out
    if "TimePeriod_ts_end" in out.columns:
        out["TimePeriod_end"] = pd.to_datetime(out["TimePeriod_ts_end"], errors="coerce").dt.date.astype(str)
        return out
    # last resort: derive from TimePeriod (not ideal, but avoids breaking on legacy files)
    if "TimePeriod" in out.columns:
        try:
            out["TimePeriod_end"] = pd.to_datetime(out["TimePeriod"].astype(str), errors="coerce").dt.date.astype(str)
        except Exception:
            pass
        return out
    raise KeyError(f"{name} missing period-end column (expected TimePeriod_end/Date_end/TimePeriod_ts_end)")


def build_v_cashflow_monthly(materialized: Dict[str, Any]) -> pd.DataFrame:
    """
    V1 motor mart: strictly based on box_balance_time_long.
    Output columns:
      TimePeriod, TimePeriod_end, Box, Currency, in_amt, out_amt, net, cum_net
    """
    bb = materialized.get("box_balance_time_long")
    if not isinstance(bb, pd.DataFrame) or bb.empty:
        return pd.DataFrame()

    df = bb.copy()
    df = require_currency(df, name="box_balance_time_long")
    _require_nonempty_col(df, ["Box"], name="box_balance_time_long")

    df = _standardize_period_end(df, name="box_balance_time_long")
    _assert_cols(df, ["TimePeriod", "Box", "Currency", "in_amt", "out_amt", "net", "cum_net"], "box_balance_time_long")

    # typing safety
    for c in ("in_amt", "out_amt", "net", "cum_net"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    out = df[["TimePeriod", "TimePeriod_end", "Box", "Currency", "in_amt", "out_amt", "net", "cum_net"]].copy()
    out = out.sort_values(["Box", "Currency", "TimePeriod_end", "TimePeriod"]).reset_index(drop=True)
    return out



def build_v_cash_position_period_end(materialized: Dict[str, Any], freq: str = "M") -> pd.DataFrame:
    """
    Cash position (end-of-period) by Box/Currency.
    Source: box_balance_time_long (must contain cum_net).
    Output columns:
      Period, Period_end, Box, Currency, cash_balance_end
    """
    bb = materialized.get("box_balance_time_long")
    if not isinstance(bb, pd.DataFrame) or bb.empty:
        return pd.DataFrame()

    df = bb.copy()
    df = require_currency(df, name="box_balance_time_long")
    _require_nonempty_col(df, ["Box"], name="box_balance_time_long")
    df = _standardize_period_end(df, name="box_balance_time_long")
    _assert_cols(df, ["TimePeriod", "TimePeriod_end", "Box", "Currency", "cum_net"], "box_balance_time_long")

    df["TimePeriod_end"] = pd.to_datetime(df["TimePeriod_end"], errors="coerce")
    df["cum_net"] = pd.to_numeric(df["cum_net"], errors="coerce").fillna(0.0).astype(float)

    if freq == "M":
        key = df["TimePeriod_end"].dt.to_period("M").astype(str)
    elif freq == "Q":
        key = df["TimePeriod_end"].dt.to_period("Q").astype(str)
    elif freq == "Y":
        key = df["TimePeriod_end"].dt.to_period("Y").astype(str)
    else:
        raise ValueError(f"Unsupported freq={freq}")

    df["Period"] = key

    # choose last month-end inside each target Period as the period-end
    df = df.sort_values(["Box", "Currency", "TimePeriod_end", "TimePeriod"])
    last = df.groupby(["Period", "Box", "Currency"], as_index=False).tail(1)

    out = last.rename(columns={"TimePeriod_end": "Period_end", "cum_net": "cash_balance_end"})[
        ["Period", "Period_end", "Box", "Currency", "cash_balance_end"]
    ].reset_index(drop=True)

    out["Period_end"] = out["Period_end"].dt.date.astype(str)
    return out.sort_values(["Box", "Currency", "Period_end", "Period"]).reset_index(drop=True)



def build_v_cashflow_period(materialized: Dict[str, Any], freq: str = "M") -> pd.DataFrame:
    """
    Cashflow by period. Sums in/out/net within period, and keeps ending cum_net as position check.
    Output:
      Period, Period_end, Box, Currency, in_amt, out_amt, net, cash_balance_end
    """
    bb = materialized.get("box_balance_time_long")
    if not isinstance(bb, pd.DataFrame) or bb.empty:
        return pd.DataFrame()

    df = bb.copy()
    df = require_currency(df, name="box_balance_time_long")
    _require_nonempty_col(df, ["Box"], name="box_balance_time_long")
    df = _standardize_period_end(df, name="box_balance_time_long")
    _assert_cols(df, ["TimePeriod_end", "Box", "Currency", "in_amt", "out_amt", "net", "cum_net"], "box_balance_time_long")

    df["TimePeriod_end"] = pd.to_datetime(df["TimePeriod_end"], errors="coerce")

    for c in ("in_amt", "out_amt", "net", "cum_net"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    if freq == "M":
        df["Period"] = df["TimePeriod_end"].dt.to_period("M").astype(str)
    elif freq == "Q":
        df["Period"] = df["TimePeriod_end"].dt.to_period("Q").astype(str)
    elif freq == "Y":
        df["Period"] = df["TimePeriod_end"].dt.to_period("Y").astype(str)
    else:
        raise ValueError(f"Unsupported freq={freq}")

    df = df.sort_values(["Box", "Currency", "TimePeriod_end"])
    # aggregate flows, then take last cum_net as end balance
    flows = df.groupby(["Period", "Box", "Currency"], as_index=False)[["in_amt", "out_amt", "net"]].sum()
    last = df.groupby(["Period", "Box", "Currency"], as_index=False).tail(1)[["Period", "Box", "Currency", "TimePeriod_end", "cum_net"]]
    last = last.rename(columns={"TimePeriod_end": "Period_end", "cum_net": "cash_balance_end"})

    out = flows.merge(last, on=["Period", "Box", "Currency"], how="left")
    out["Period_end"] = pd.to_datetime(out["Period_end"], errors="coerce").dt.date.astype(str)
    return out.sort_values(["Box", "Currency", "Period_end", "Period"]).reset_index(drop=True)


def build_v_contributions_monthly(
    materialized: Dict[str, Any],
    flujo_values: Optional[List[str]] = None,
    payer_roles: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    V2 mart: who finances deficits.
    Source: per_party_time_long filtered to Flujo == Contribucion (or equivalents).
    contributor_party is defined as the party on payer-side (role in payer_roles).
    Amount is expressed as positive "funding provided".
    """
    perp = materialized.get("per_party_time_long")
    if not isinstance(perp, pd.DataFrame) or perp.empty:
        return pd.DataFrame()

    df = perp.copy()
    df = require_currency(df, name="per_party_time_long")
    _require_nonempty_col(df, ["Box"], name="per_party_time_long")
    df = _standardize_period_end(df, name="per_party_time_long")

    _assert_cols(df, ["TimePeriod", "TimePeriod_end", "Box", "Currency", "party", "role", "Flujo", "amount", "n_tx"], "per_party_time_long")

    if flujo_values is None:
        flujo_values = ["contribucion", "contribución", "contribution"]

    if payer_roles is None:
        payer_roles = ["payer", "pagador", "paga", "debit", "deudor"]

    flujo_norm = df["Flujo"].astype(str).str.strip().str.lower()
    allowed = set([v.strip().lower() for v in flujo_values])
    df = df[flujo_norm.isin(allowed)].copy()
    if df.empty:
        return pd.DataFrame()

    role_norm = df["role"].astype(str).str.strip().str.lower()
    payer_set = set([r.strip().lower() for r in payer_roles])
    df = df[role_norm.isin(payer_set)].copy()
    if df.empty:
        return pd.DataFrame()

    df["contributor_party"] = df["party"].astype(str)

    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)
    # robust across sign conventions: funding provided is magnitude on payer side
    df["amount"] = amt.abs()

    out = (
        df.groupby(["TimePeriod", "TimePeriod_end", "Box", "Currency", "contributor_party"], as_index=False)[["amount", "n_tx"]]
        .sum()
        .sort_values(["Box", "Currency", "TimePeriod_end", "contributor_party"])
        .reset_index(drop=True)
    )
    return out


def build_v_opex_category_monthly(materialized: Dict[str, Any]) -> pd.DataFrame:
    """
    V3-lite mart: opex by Tipo using the Box motor decomposition.
    Source: box_flow_balance_time_long filtered to outflows and grouped by Tipo.
    Output columns:
      TimePeriod, TimePeriod_end, Box, Currency, Tipo, amount_out, n_tx
    """
    bfb = materialized.get("box_flow_balance_time_long")
    if not isinstance(bfb, pd.DataFrame) or bfb.empty:
        return pd.DataFrame()

    df = bfb.copy()
    df = require_currency(df, name="box_flow_balance_time_long")
    _require_nonempty_col(df, ["Box"], name="box_flow_balance_time_long")

    df = _standardize_period_end(df, name="box_flow_balance_time_long")
    _assert_cols(df, ["TimePeriod", "TimePeriod_end", "Box", "Currency", "Tipo", "out_amt", "n_tx"], "box_flow_balance_time_long")

    df["out_amt"] = pd.to_numeric(df["out_amt"], errors="coerce").fillna(0.0).astype(float)
    df["n_tx"] = pd.to_numeric(df["n_tx"], errors="coerce").fillna(0.0).astype(float)

    df = df[df["out_amt"] > 0].copy()
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(["TimePeriod", "TimePeriod_end", "Box", "Currency", "Tipo"], as_index=False)[["out_amt", "n_tx"]]
        .sum()
        .rename(columns={"out_amt": "amount_out"})
        .sort_values(["Box", "Currency", "TimePeriod_end", "Tipo"])
        .reset_index(drop=True)
    )
    return out

def build_party_timeseries_view(
    materialized: Dict[str, Any],
    freq: str = "M",
    classifier_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Produce a party x currency (optionally x Flujo/Tipo) time series view.

    Contract rules:
      - If per_party_time_long exists (materialized), it is already period-binned by core_timeseries.
        Views will NOT resample or re-bin; it only collapses duplicates and derives net/in/out.
      - If per_party_time_long is missing, renta_all may be used as a fallback.
        In that path, binning uses core_timeseries.period_bins_for_dates (not pandas resample).

    Output grain:
      Date (period end), party, currency, [Flujo,Tipo], in_amt, out_amt, net
    """
    if classifier_cols is None:
        classifier_cols = ["Flujo", "Tipo"]

    perp = materialized.get("per_party_time_long")
    renta = materialized.get("renta_all")

    if isinstance(perp, pd.DataFrame) and not perp.empty:
        base = perp.copy()
        base = require_currency(base, name="per_party_time_long")
        base = _parse_date_col(base, name="per_party_time_long")
        _assert_cols(base, ["party", "Box"], "per_party_time_long")

        # Require a "proof" column that this came from party expansion
        if "role" not in base.columns:
            raise KeyError("per_party_time_long missing 'role' (required to trust signed amount contract)")

        signed = _signed_from_materialized_per_party(base)
        base["signed"] = signed

        # Collapse role away: view is party-level (not party-role)
        keep_classifiers = [c for c in classifier_cols if c in base.columns]
        for c in keep_classifiers:
            base[c] = base[c].astype(str)

        grp = ["Date", "Box", "party", "Currency"] + keep_classifiers
        collapsed = base.groupby(grp, as_index=False)["signed"].sum()

        collapsed = _compute_in_out_net(collapsed, signed_col="signed")
        out_cols = ["Date", "Box", "party", "Currency"] + keep_classifiers + ["in_amt", "out_amt", "net"]
        return collapsed[out_cols].sort_values(["Box", "Currency", "party", "Date"]).reset_index(drop=True)

    # No fallback: party views must be driven by materialized per_party_time_long.
    return pd.DataFrame()


def _build_party_level_net_and_cum(party_detailed: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute party-level net and cumulative net safely (per party, per currency) from detailed net rows.
    This avoids summing classifier-level cumulative values which is operationally fragile.
    """
    _assert_cols(party_detailed, ["Date", "Box", "party", "Currency", "net"], "party_detailed")

    party_net = (
        party_detailed.groupby(["Date", "Box", "party", "Currency"], as_index=False)["net"]
        .sum()
        .sort_values(["Box", "Currency", "party", "Date"])
    )
    party_net["cum_net"] = party_net.groupby(["Box", "party", "Currency"])["net"].cumsum()
    return party_net, party_net.copy()


def export_views(
    reports_dir: Path,
    write_dir: Path,
    freq: str = "M",
    allow_cross_currency_sum: bool = False,
) -> Dict[str, str]:
    reports = load_reports_folder(Path(reports_dir), freq=freq)
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    outputs_meta: Dict[str, Dict[str, Any]] = {}
    inv_errors: List[str] = []
    inv_warnings: List[str] = []

    # Carry forward any loader warnings (legacy artifacts, missing columns, etc.)
    inv_warnings.extend(reports.get('_load_warnings', []) if isinstance(reports.get('_load_warnings', []), list) else [])

    # Narrative marts (V1-V3): these are the decision-grade story inputs.
    v_cashflow = build_v_cashflow_monthly(reports)
    if v_cashflow.empty:
        inv_errors.append("Missing or empty input for V1: box_balance_time_long")
    else:
        p = write_dir / "v_cashflow_monthly.csv"
        atomic_write_df(v_cashflow, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(v_cashflow)

    v_contrib = build_v_contributions_monthly(reports)
    if v_contrib.empty:
        inv_warnings.append("V2 empty: no contributions found or per_party_time_long missing")
    else:
        p = write_dir / "v_contributions_monthly.csv"
        atomic_write_df(v_contrib, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(v_contrib)

    v_opex = build_v_opex_category_monthly(reports)
    if v_opex.empty:
        inv_warnings.append("V3-lite empty: box_flow_balance_time_long missing or no outflows")
    else:
        p = write_dir / "v_opex_category_monthly.csv"
        atomic_write_df(v_opex, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(v_opex)


    renta_pivot = build_renta_pivot_view(reports, freq=freq)
    if not renta_pivot.empty:
        p = write_dir / "renta_pivot.party_currency.csv"
        atomic_write_df(renta_pivot.reset_index(), p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(renta_pivot.reset_index())

    fondos_w = build_fondos_wide_view(reports)
    if not fondos_w.empty:
        p = write_dir / "fondos_wide.csv"
        atomic_write_df(fondos_w, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(fondos_w)

    party_detailed = build_party_timeseries_view(reports, freq=freq)
    if not party_detailed.empty:
        party_detailed = require_currency(party_detailed, name="party_balance_detailed")
        p = write_dir / "party_balance_detailed.csv"
        atomic_write_df(party_detailed, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(party_detailed)

    # Party-level wide outputs (currency-safe)
    if not party_detailed.empty:
        party_net, party_net_with_cum = _build_party_level_net_and_cum(party_detailed)

        net_wide = (
            party_net.pivot_table(index="Date", columns=["Box", "party", "Currency"], values="net", aggfunc="sum")
            .fillna(0.0)
            .sort_index()
        )
        cum_wide = (
            party_net_with_cum.pivot_table(index="Date", columns=["Box", "party", "Currency"], values="cum_net", aggfunc="sum")
            .fillna(0.0)
            .sort_index()
        )

        p1 = write_dir / "party_balance_net_wide.party_currency.csv"
        p2 = write_dir / "party_balance_cum_wide.party_currency.csv"
        atomic_write_df(net_wide.reset_index(), p1)
        atomic_write_df(cum_wide.reset_index(), p2)
        outputs[p1.name] = str(p1)
        outputs[p2.name] = str(p2)
        outputs_meta[p1.name] = _df_summary(net_wide.reset_index())
        outputs_meta[p2.name] = _df_summary(cum_wide.reset_index())

        if allow_cross_currency_sum:
            if isinstance(net_wide.columns, pd.MultiIndex):
                unsafe_net = net_wide.groupby(level=0, axis=1).sum()
                unsafe_cum = cum_wide.groupby(level=0, axis=1).sum()
            else:
                unsafe_net = net_wide
                unsafe_cum = cum_wide

            p3 = write_dir / "UNSAFE_sum_across_currency.party_balance_net_wide_party_only.csv"
            p4 = write_dir / "UNSAFE_sum_across_currency.party_balance_cum_wide_party_only.csv"
            atomic_write_df(unsafe_net.reset_index(), p3)
            atomic_write_df(unsafe_cum.reset_index(), p4)
            outputs[p3.name] = str(p3)
            outputs[p4.name] = str(p4)
            outputs_meta[p3.name] = _df_summary(unsafe_net.reset_index())
            outputs_meta[p4.name] = _df_summary(unsafe_cum.reset_index())
            inv_warnings.append("UNSAFE outputs written: sums across currencies are not accountant-safe")

        
    # Legacy outputs that can be misleading (zero-sum risk). Disabled by default.
    # if _legacy_zero_sum_outputs_enabled():
    # Flujo/Tipo aggregate view (currency-safe but sums across parties, so interpret as attribution only)
    if not party_detailed.empty:
        if not all(c in party_detailed.columns for c in ("Flujo", "Tipo")):
            inv_warnings.append("Skipped balance_by_flujo_tipo: party_balance_detailed missing Flujo/Tipo")
        else:
            by_ft = (
                party_detailed.groupby(["Date", "Box", "Currency", "Flujo", "Tipo"], as_index=False)[["in_amt", "out_amt", "net"]]
                .sum()
                .sort_values(["Box", "Currency", "Flujo", "Tipo", "Date"])
            )
            by_ft["cum_net"] = by_ft.groupby(["Box", "Currency", "Flujo", "Tipo"])["net"].cumsum()
            by_ft = require_currency(by_ft, name="balance_by_flujo_tipo")

            p = write_dir / "balance_by_flujo_tipo.currency_safe.csv"
            atomic_write_df(by_ft, p)
            outputs[p.name] = str(p)
            outputs_meta[p.name] = _df_summary(by_ft)

    # Consolidated balance (sums across parties; still Box-keyed but can hide zero-sum effects)
    if not party_detailed.empty:
        consol = (
            party_detailed.groupby(["Date", "Box", "Currency"], as_index=False)[["in_amt", "out_amt", "net"]]
            .sum()
            .sort_values(["Box", "Currency", "Date"])
        )
        consol["cum_net"] = consol.groupby(["Box", "Currency"])["net"].cumsum()
        consol = require_currency(consol, name="consolidated_balance")

        p = write_dir / "consolidated_balance.currency_safe.csv"
        atomic_write_df(consol, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(consol)
    # else:
    #     inv_warnings.append("Skipped legacy outputs: balance_by_flujo_tipo and consolidated_balance (zero-sum risk)")



# Upcoming 90 days: label as raw convenience extract by default.
    # If currency exists, we enforce it; otherwise we keep raw and emit a warning in sanity.
    ledger = reports.get("ledger")
    if isinstance(ledger, pd.DataFrame) and not ledger.empty and "Date" in ledger.columns:
        led = _parse_date_col(ledger, name="ledger")
        now = pd.Timestamp.now().normalize()
        d = pd.to_datetime(led["Date"], errors="coerce")
        upcoming = led.loc[(d.notna()) & (d >= now) & (d <= now + pd.Timedelta(days=90))].copy()
        upcoming = upcoming.sort_values("Date")

        if "Currency" in upcoming.columns:
            try:
                upcoming = require_currency(upcoming, name="upcoming_90")
            except Exception as e:
                inv_warnings.append(f"upcoming_90 has currency column but failed currency invariant: {e}")

        p = write_dir / "upcoming_90.raw.csv"
        atomic_write_df(upcoming, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(upcoming)

        if not any(c in upcoming.columns for c in ("Currency",)):
            inv_warnings.append("upcoming_90.raw.csv written without currency enforcement (no currency column found)")

    # Observability / sanity
    sanity = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "reports_dir": str(Path(reports_dir)),
        "write_dir": str(write_dir),
        "freq": str(freq),
        "paths": reports.get("_paths", {}),
        "inputs": {},
        "outputs": outputs_meta,
        "invariants": {"errors": inv_errors, "warnings": inv_warnings},
    }

    for k in ("per_party_time_long", "per_flow_time_long", "daily_cash_position", "ledger", "renta_all"):
        df = reports.get(k)
        if isinstance(df, pd.DataFrame):
            df_s = df.copy()
            if "Date" not in df_s.columns and any(c in df_s.columns for c in ("TimePeriod_ts_end", "TimePeriod")):
                try:
                    df_s = _parse_date_col(df_s, name=k)
                except Exception:
                    pass
            sanity["inputs"][k] = _df_summary(df_s)
        else:
            sanity["inputs"][k] = {"rows": 0, "date_min": None, "date_max": None, "currencies": []}

    sanity_path = write_dir / "views_sanity.json"
    sanity_path.write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs[sanity_path.name] = str(sanity_path)

    return outputs


# -----------------------
# CLI
# -----------------------
def _artifact_name_for_file(filename: str) -> str:
    """Normalize a filename into a stable artifact `name` (no extensions, no dots)."""
    base = str(filename).strip()
    for ext in (".csv", ".json", ".parquet", ".pq", ".txt"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    # keep it filename-derived but schema-friendly
    base = base.replace(".", "_")
    base = base.replace("-", "_")
    return base


def _content_type_for_path(p: Path) -> str:
    suf = p.suffix.lower()
    if suf == ".csv":
        return "text/csv"
    if suf in {".json", ".jsonl"}:
        return "application/json"
    if suf in {".parquet", ".pq"}:
        return "application/octet-stream"
    return "application/octet-stream"


def _parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="out/reports")
    p.add_argument("--write-dir", default="out/views")
    p.add_argument("--freq", default="M")
    p.add_argument("--allow-cross-currency-sum", default=os.getenv("ALLOW_CROSS_CURRENCY_SUM", "0"))
    p.add_argument("--mode", choices=["smoke", "run"], default=os.getenv("MODE", "run"))
    p.add_argument("--run-id", default=os.getenv("RUN_ID", ""))
    return p.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = _parse_args(argv)

    reports_dir = Path(args.reports_dir)
    write_dir = Path(args.write_dir)

    LOG.info("Stage start mode=%s reports_dir=%s write_dir=%s freq=%s", args.mode, reports_dir, write_dir, args.freq)

    out = export_views(
        reports_dir,
        write_dir,
        freq=str(args.freq),
        allow_cross_currency_sum=bool(int(str(args.allow_cross_currency_sum))),
    )
    # out_dir = Path(args.out_dir)
    # out_dir = Path(os.getenv("OUT_DIR", "./out"))

    # 

    
    # Align artifact recording with A.ingest / D.materialize / E.reports.
    # Non-fatal: views are useful even if manifest writing fails.
    try:
        root_dir = write_dir.resolve().parent  # expected: out/
        meta_dir = root_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        from accounting.artifacts.manifest import artifact_from_path, write_stage_manifest, append_artifacts

        stage = "F.views"
        mode = str(args.mode)
        # run_id = _resolve_run_id(mode=mode, run_id=str(args.run_id))
        # run_id = resolve_run_id(mode=args.mode, run_id=getattr(args, "run_id", None), root_dir=out_dir, strict=True)
        run_id = resolve_run_id(mode=mode, run_id=getattr(args, "run_id", None), root_dir=root_dir, strict=True)
        # run_id = _resolve_run_id(args)

        stage_generated_at = pd.Timestamp.now("UTC").isoformat()

        # Inputs: best-effort from the written sanity file (it captures resolved paths).
        inputs = []
        sanity_path = write_dir / "views_sanity.json"
        if sanity_path.exists():
            try:
                sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
                paths = sanity.get("paths", {}) or {}
            except Exception:
                paths = {}
        else:
            paths = {}

        key_to_name = {
            "fondos": "fondos_report",
            "per_party_time_long": "per_party_time_long",
            "per_flow_time_long": "per_flow_time_long",
            "ledger": "ledger_canonical",
            "daily_cash_position": "daily_cash_position",
            "_manifest_path": "manifest",
        }

        for k, v in (paths or {}).items():
            if not v or "*" in str(v):
                continue
            p = Path(v)
            if not (p.exists() and p.is_file()):
                continue
            nm = key_to_name.get(k, _artifact_name_for_file(p.name))
            inputs.append(
                artifact_from_path(
                    name=nm,
                    path=p,
                    stage=stage,
                    mode=mode,
                    run_id=run_id,
                    role="input",
                    root_dir=root_dir,
                    content_type=_content_type_for_path(p),
                )
            )

        # Outputs: all view files created by this stage.
        out_arts = []
        for fn, pth in out.items():
            p = Path(pth)
            if not (p.exists() and p.is_file()):
                continue
            out_arts.append(
                artifact_from_path(
                    name=_artifact_name_for_file(fn),
                    path=p,
                    stage=stage,
                    mode=mode,
                    run_id=run_id,
                    role="derived",
                    root_dir=root_dir,
                    content_type=_content_type_for_path(p),
                )
            )

        stage_manifest = {
            "generated_at": stage_generated_at,
            "stage": stage,
            "mode": mode,
            "run_id": run_id,
            "inputs": inputs,
            "params": {
                "reports_dir": str(reports_dir),
                "write_dir": str(write_dir),
                "freq": str(args.freq),
                "allow_cross_currency_sum": int(bool(int(str(args.allow_cross_currency_sum)))),
            },
            "outputs": out_arts,
            "warnings": [],
        }

        stage_manifest_rel = write_stage_manifest(meta_dir, stage_manifest)

        stage_meta_path = root_dir / stage_manifest_rel
        stage_meta_sha = artifact_from_path(
            name="stage_F_views",
            path=stage_meta_path,
            stage=stage,
            mode=mode,
            run_id=run_id,
            role="meta",
            root_dir=root_dir,
            content_type="application/json",
        )["sha256"]

        stage_meta_art = {
            "run_id": run_id,
            "stage": stage,
            "mode": mode,
            "name": "stage_F_views",
            "relpath": stage_manifest_rel,
            "sha256": stage_meta_sha,
            "bytes": stage_meta_path.stat().st_size,
            "rows": None,
            "content_type": "application/json",
            "created_at": stage_generated_at,
            "role": "meta",
        }

        append_artifacts(meta_dir, [*inputs, *out_arts, stage_meta_art])
    except Exception:
        LOG.exception("Views manifest write failed (non-fatal)")

    LOG.info("Stage finish outputs=%s", json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())