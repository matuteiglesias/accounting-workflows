# src/accounting/views.py
from __future__ import annotations

import json
import os
import sys
import logging
import datetime as _dt
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import pandas as pd

from accounting.utils import atomic_write_df
from accounting.utils import (
    _read_csv_if_exists,
    _normalize_currency_col,
    _find_first_existing,
    _ensure_amount,
)

from accounting.core_timeseries import period_bins_for_dates

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

_DAILY_CASH_FN = "daily_cash_position.csv"
_MANIFEST_FN = "manifest.json"

LOG = logging.getLogger(__name__)


# -----------------------
# Small contract helpers
# -----------------------
def _assert_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def _parse_date_col(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    elif "TimePeriod_ts_end" in out.columns:
        out["Date"] = pd.to_datetime(out["TimePeriod_ts_end"], errors="coerce")
    elif "TimePeriod" in out.columns:
        out["Date"] = pd.to_datetime(out["TimePeriod"].astype(str), errors="coerce")
    else:
        raise KeyError(f"{name} missing date column (expected Date/TimePeriod_ts_end/TimePeriod)")
    return out


def _require_currency(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Enforce that a canonical `currency` column exists, is non-null, and non-empty after normalization.
    This intentionally fails fast instead of patching.
    """
    out = _normalize_currency_col(df.copy())
    if "currency" not in out.columns:
        raise KeyError(f"{name} missing required column 'currency' after normalization")

    cur = out["currency"]
    if cur.isna().any():
        raise ValueError(f"{name} has null values in 'currency'")

    # Reject empty/blank strings (common leak path)
    cur_str = cur.astype(str).str.strip()
    if (cur_str == "").any():
        raise ValueError(f"{name} has blank/empty values in 'currency'")

    out["currency"] = cur_str
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
    if "currency" in df.columns:
        currencies = sorted([c for c in df["currency"].dropna().astype(str).unique().tolist() if c.strip() != ""])
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
    Load materialized artifacts (stage D) plus optional reports artifacts.
    This function is intentionally "dumb I/O" (no business semantics beyond column normalization/validation).
    """
    reports_dir = Path(reports_dir)
    out: Dict[str, Any] = {}
    paths: Dict[str, Optional[str]] = {}

    p_fondos = reports_dir / FONDOS_FN
    out["fondos"] = _read_csv_if_exists(p_fondos, index_col=0, dtype=object)
    paths["fondos"] = str(p_fondos) if p_fondos.exists() else None

    parent_out = reports_dir.parent
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

    p_ledger = parent_out / LEDGER_FN
    if not p_ledger.exists():
        p_ledger = reports_dir / LEDGER_FN
    paths["ledger"] = str(p_ledger) if p_ledger.exists() else None
    out["ledger"] = _read_csv_if_exists(p_ledger, parse_dates=["Date"]) if p_ledger.exists() else pd.DataFrame()

    p_daily = parent_out / _DAILY_CASH_FN
    if not p_daily.exists():
        p_daily = reports_dir / _DAILY_CASH_FN
    paths["daily_cash_position"] = str(p_daily) if p_daily.exists() else None
    out["daily_cash_position"] = _read_csv_if_exists(p_daily) if p_daily.exists() else pd.DataFrame()

    # Optional renta inputs (kept for now; candidate to move into ingest/reports layer later)
    renta_dfs = []
    for p in sorted(reports_dir.glob(RENTA_GLOB)):
        df = pd.read_csv(p, low_memory=False)

        # best-effort normalization of common renta layouts
        if "TimePeriod_ts_end" in df.columns and "amount" in df.columns:
            df = df.rename(columns={"TimePeriod_ts_end": "Date"})
        elif "Date" not in df.columns and df.shape[1] >= 2:
            df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "amount"})

        df = _parse_date_col(df, name=f"renta:{p.name}")
        df = _ensure_amount(df)
        df = _require_currency(df, name=f"renta:{p.name}")

        df["party"] = p.stem.replace("renta_", "")
        renta_dfs.append(df[["Date", "amount", "party", "currency"]])

    out["renta_all"] = (
        pd.concat(renta_dfs, ignore_index=True)
        if renta_dfs
        else pd.DataFrame(columns=["Date", "amount", "party", "currency"])
    )
    paths["renta_all_glob"] = str(reports_dir / RENTA_GLOB)

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

    # Normalize + enforce currency for money-bearing, accountant-facing inputs
    for key in ("per_party_time_long", "per_flow_time_long", "daily_cash_position", "ledger"):
        df = out.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _ensure_amount(df) if "amount" in df.columns or "amount_cents" in df.columns else df
            # ledger might have Currency but always normalize to currency for any money-facing downstream
            df = _require_currency(df, name=key) if any(c in df.columns for c in ("Currency", "currency")) else df
            out[key] = df

    out["_paths"] = paths
    return out


# -----------------------
# Views
# -----------------------
def build_renta_pivot_view(materialized: Dict[str, Any], freq: str = "M") -> pd.DataFrame:
    """
    Pivot renta series into wide table indexed by period-end Date, columns (party,currency).

    If renta rows are not already period-aggregated, bins are produced via core_timeseries.period_bins_for_dates
    to avoid pandas resample semantics.
    """
    renta = materialized.get("renta_all")
    if not isinstance(renta, pd.DataFrame) or renta.empty:
        return pd.DataFrame()

    renta = _parse_date_col(renta, name="renta_all")
    renta = _ensure_amount(renta)
    renta = _require_currency(renta, name="renta_all")
    _assert_cols(renta, ["party"], "renta_all")

    renta = renta.dropna(subset=["Date"]).copy()

    if not _is_period_aggregated(renta):
        bins = period_bins_for_dates(renta["Date"], freq=freq)
        renta["Date"] = pd.to_datetime(bins["TimePeriod_ts_end"], errors="coerce")

    pivot = (
        renta.pivot_table(index="Date", columns=["party", "currency"], values="amount", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    return pivot


def build_fondos_wide_view(materialized: Dict[str, Any]) -> pd.DataFrame:
    fondos = materialized.get("fondos")
    if not isinstance(fondos, pd.DataFrame) or fondos.empty:
        return pd.DataFrame()

    out = fondos.copy()
    try:
        parsed = pd.to_datetime(out.index.astype(str), errors="coerce")
        out = out.reset_index().rename(columns={"index": "period_label"})
        if parsed.notna().any():
            out["__Date_parsed"] = parsed
    except Exception:
        out = out.reset_index().rename(columns={"index": "period_label"})

    for c in out.columns:
        if c in ("period_label", "__Date_parsed"):
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def build_party_timeseries_view(
    materialized: Dict[str, Any],
    freq: str = "W",
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
        base = _require_currency(base, name="per_party_time_long")
        base = _parse_date_col(base, name="per_party_time_long")
        _assert_cols(base, ["party"], "per_party_time_long")

        # Require a "proof" column that this came from party expansion
        if "role" not in base.columns:
            raise KeyError("per_party_time_long missing 'role' (required to trust signed amount contract)")

        signed = _signed_from_materialized_per_party(base)
        base["signed"] = signed

        # Collapse role away: view is party-level (not party-role)
        keep_classifiers = [c for c in classifier_cols if c in base.columns]
        for c in keep_classifiers:
            base[c] = base[c].astype(str)

        grp = ["Date", "party", "currency"] + keep_classifiers
        collapsed = base.groupby(grp, as_index=False)["signed"].sum()

        collapsed = _compute_in_out_net(collapsed, signed_col="signed")
        out_cols = ["Date", "party", "currency"] + keep_classifiers + ["in_amt", "out_amt", "net"]
        return collapsed[out_cols].sort_values(["currency", "party", "Date"]).reset_index(drop=True)

    # Fallback: renta_all only (no Flujo/Tipo classifiers available)
    if not isinstance(renta, pd.DataFrame) or renta.empty:
        return pd.DataFrame()

    if classifier_cols:
        # classifier_cols are incompatible with renta fallback
        missing = [c for c in classifier_cols if c not in ("", None)]
        if missing:
            raise KeyError("classifier_cols requested but per_party_time_long missing; renta fallback has no classifiers")

    renta = _parse_date_col(renta, name="renta_all")
    renta = _ensure_amount(renta)
    renta = _require_currency(renta, name="renta_all")
    _assert_cols(renta, ["party"], "renta_all")
    renta = renta.dropna(subset=["Date"]).copy()

    bins = period_bins_for_dates(renta["Date"], freq=freq)
    renta["Date"] = pd.to_datetime(bins["TimePeriod_ts_end"], errors="coerce")

    # In renta, amount is treated as signed inflow by definition of this dataset (contract for now)
    renta["signed"] = pd.to_numeric(renta["amount"], errors="coerce").fillna(0.0).astype(float)
    collapsed = renta.groupby(["Date", "party", "currency"], as_index=False)["signed"].sum()
    collapsed = _compute_in_out_net(collapsed, signed_col="signed")
    return collapsed[["Date", "party", "currency", "in_amt", "out_amt", "net"]].sort_values(
        ["currency", "party", "Date"]
    ).reset_index(drop=True)


def _build_party_level_net_and_cum(party_detailed: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute party-level net and cumulative net safely (per party, per currency) from detailed net rows.
    This avoids summing classifier-level cumulative values which is operationally fragile.
    """
    _assert_cols(party_detailed, ["Date", "party", "currency", "net"], "party_detailed")

    party_net = (
        party_detailed.groupby(["Date", "party", "currency"], as_index=False)["net"]
        .sum()
        .sort_values(["currency", "party", "Date"])
    )
    party_net["cum_net"] = party_net.groupby(["party", "currency"])["net"].cumsum()
    return party_net, party_net.copy()


def export_views(
    reports_dir: Path,
    write_dir: Path,
    freq: str = "W",
    allow_cross_currency_sum: bool = False,
) -> Dict[str, str]:
    reports = load_reports_folder(Path(reports_dir), freq=freq)
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    outputs_meta: Dict[str, Dict[str, Any]] = {}
    inv_errors: List[str] = []
    inv_warnings: List[str] = []

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
        party_detailed = _require_currency(party_detailed, name="party_balance_detailed")
        p = write_dir / "party_balance_detailed.csv"
        atomic_write_df(party_detailed, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(party_detailed)

    # Party-level wide outputs (currency-safe)
    if not party_detailed.empty:
        party_net, party_net_with_cum = _build_party_level_net_and_cum(party_detailed)

        net_wide = (
            party_net.pivot_table(index="Date", columns=["party", "currency"], values="net", aggfunc="sum")
            .fillna(0.0)
            .sort_index()
        )
        cum_wide = (
            party_net_with_cum.pivot_table(index="Date", columns=["party", "currency"], values="cum_net", aggfunc="sum")
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

    # Flujo/Tipo aggregate view (currency-safe)
    if not party_detailed.empty:
        if not all(c in party_detailed.columns for c in ("Flujo", "Tipo")):
            inv_warnings.append("Skipped balance_by_flujo_tipo: party_balance_detailed missing Flujo/Tipo")
        else:
            by_ft = (
                party_detailed.groupby(["Date", "currency", "Flujo", "Tipo"], as_index=False)[["in_amt", "out_amt", "net"]]
                .sum()
                .sort_values(["currency", "Flujo", "Tipo", "Date"])
            )
            by_ft["cum_net"] = by_ft.groupby(["currency", "Flujo", "Tipo"])["net"].cumsum()
            by_ft = _require_currency(by_ft, name="balance_by_flujo_tipo")

            p = write_dir / "balance_by_flujo_tipo.currency_safe.csv"
            atomic_write_df(by_ft, p)
            outputs[p.name] = str(p)
            outputs_meta[p.name] = _df_summary(by_ft)

    # Consolidated per-currency net + cum
    if not party_detailed.empty:
        consol = (
            party_detailed.groupby(["Date", "currency"], as_index=False)["net"]
            .sum()
            .sort_values(["currency", "Date"])
        )
        consol["cum_net"] = consol.groupby("currency")["net"].cumsum()
        consol = _require_currency(consol, name="consolidated_balance")

        p = write_dir / "consolidated_balance.currency_safe.csv"
        atomic_write_df(consol, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(consol)

    # Upcoming 90 days: label as raw convenience extract by default.
    # If currency exists, we enforce it; otherwise we keep raw and emit a warning in sanity.
    ledger = reports.get("ledger")
    if isinstance(ledger, pd.DataFrame) and not ledger.empty and "Date" in ledger.columns:
        led = _parse_date_col(ledger, name="ledger")
        now = pd.Timestamp.now().normalize()
        d = pd.to_datetime(led["Date"], errors="coerce")
        upcoming = led.loc[(d.notna()) & (d >= now) & (d <= now + pd.Timedelta(days=90))].copy()
        upcoming = upcoming.sort_values("Date")

        if any(c in upcoming.columns for c in ("Currency", "currency")):
            try:
                upcoming = _require_currency(upcoming, name="upcoming_90")
            except Exception as e:
                inv_warnings.append(f"upcoming_90 has currency column but failed currency invariant: {e}")

        p = write_dir / "upcoming_90.raw.csv"
        atomic_write_df(upcoming, p)
        outputs[p.name] = str(p)
        outputs_meta[p.name] = _df_summary(upcoming)

        if not any(c in upcoming.columns for c in ("currency",)):
            inv_warnings.append("upcoming_90.raw.csv written without currency enforcement (no currency column found)")

    # Observability / sanity
    sanity = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
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


def _resolve_run_id(mode: str, run_id: str) -> str:
    rid = str(run_id or "").strip()
    if rid:
        return rid
    return "smoke" if str(mode).strip().lower() == "smoke" else ""


def _parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="out/reports")
    p.add_argument("--write-dir", default="out/views")
    p.add_argument("--freq", default="W")
    p.add_argument("--allow-cross-currency-sum", default=os.getenv("ALLOW_CROSS_CURRENCY_SUM", "0"))
    p.add_argument("--mode", choices=["smoke", "run"], default=os.getenv("MODE", "run"))
    p.add_argument("--run-id", default=os.getenv("RUN_ID", ""))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    reports_dir = Path(args.reports_dir)
    write_dir = Path(args.write_dir)

    out = export_views(
        reports_dir,
        write_dir,
        freq=str(args.freq),
        allow_cross_currency_sum=bool(int(str(args.allow_cross_currency_sum))),
    )

    # Align artifact recording with A.ingest / D.materialize / E.reports.
    # Non-fatal: views are useful even if manifest writing fails.
    try:
        root_dir = write_dir.resolve().parent  # expected: out/
        meta_dir = root_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        from accounting.manifest import artifact_from_path, write_stage_manifest, append_artifacts

        stage = "F.views"
        mode = str(args.mode)
        run_id = _resolve_run_id(mode=mode, run_id=str(args.run_id))

        stage_generated_at = pd.Timestamp.utcnow().isoformat()

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

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())