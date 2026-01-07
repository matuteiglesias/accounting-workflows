# src/accounting/views.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

# helper from your utils (atomic_write_df) or fallback to pd.to_csv
from accounting.utils import atomic_write_df

# --- Config: filenames we expect in reports folder ---
FONDOS_FN = "fondos_report.csv"
RENTA_GLOB = "renta_*.csv"
LEDGER_FN = "ledger_canonical.csv"
# PER_PARTY_FN = "per_party_time_long.freq={freq}.csv"


# canonical filenames we expect materialize to produce (freq suffix may exist)
_PER_PARTY_PATTERNS = [
    "per_party_time_long.freq={freq}.csv",  # preferred
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


# --- Small validators ---
def _assert_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")

# # # --- Loading helpers ---
# def load_reports_folder(reports_dir: Path) -> Dict[str, pd.DataFrame]:
#     reports_dir = Path(reports_dir)
#     out = {}
#     f_fondos = reports_dir / FONDOS_FN
#     out['fondos'] = pd.read_csv(f_fondos, index_col=0, dtype=object) if f_fondos.exists() else pd.DataFrame()

#     # --- load renta files (robust to missing currency) ---
#     renta_dfs = []
#     for p in sorted(reports_dir.glob("renta_*.csv")):
#         df = pd.read_csv(p, low_memory=False)
#         # normalize names: Date / amount
#         if "TimePeriod_ts_end" in df.columns and "amount" in df.columns:
#             df = df.rename(columns={"TimePeriod_ts_end": "Date"})[["Date", "amount"] + [c for c in df.columns if c not in ["TimePeriod_ts_end","amount"]]]
#         elif "Date" in df.columns and "amount" in df.columns:
#             df = df[["Date", "amount"] + [c for c in df.columns if c not in ["Date","amount"]]]
#         else:
#             # fallback: try first two columns as Date, amount
#             if df.shape[1] >= 2:
#                 df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "amount"})
#             else:
#                 # single column: treat as amount with synthetic index -> skip
#                 continue

#         party = p.stem.replace("renta_", "")
#         df["party"] = party
#         df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#         df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

#         # --- ensure a currency column exists (default NA) ---
#         # If the file contains a 'Currency' column already, keep it (normalize to upper-case).


#         if "Currency" in df.columns:
#             df["Currency"] = df["Currency"].astype(str).str.upper().fillna("NA")
#             print("NaN at position 7")
#         else:
#             df["Currency"] = "NA"
#             print("Ok at position 7.5")  ## Here 3rd

#         renta_dfs.append(df[["Date", "amount", "party", "Currency"]])
#     out['renta_all'] = pd.concat(renta_dfs, ignore_index=True) if renta_dfs else pd.DataFrame(columns=["Date","amount","party","Currency"])

#     return out



from accounting.utils import _read_csv_if_exists, _normalize_currency_col, _find_first_existing, _ensure_amount
import json


def load_reports_folder(reports_dir: Path, freq: str = "M") -> Dict[str, pd.DataFrame]:
    """
    Load canonical materialized report artifacts into a dict with predictable keys.

    - reports_dir: the directory where 'reports' outputs live (./out/reports by default).
    - freq: used to match freq-suffixed filenames produced by materialize.
    """
    reports_dir = Path(reports_dir)
    out: Dict[str, pd.DataFrame] = {}

    # 1) fondos (report writers live in reports_dir)
    p_fondos = reports_dir / FONDOS_FN
    out["fondos"] = _read_csv_if_exists(p_fondos, index_col=0, dtype=object)

    # 2) try to load per_party_time_long (materialize typically writes to out/)
    # check reports_dir parent (usually OUT_DIR)
    parent_out = reports_dir.parent
    per_party_path = _find_first_existing(parent_out, _PER_PARTY_PATTERNS, freq=freq)
    per_flow_path = _find_first_existing(parent_out, _PER_FLOW_PATTERNS, freq=freq)

    # if not in parent_out, also check reports_dir itself (defensive but not legacy-heavy)
    if per_party_path is None:
        per_party_path = _find_first_existing(reports_dir, _PER_PARTY_PATTERNS, freq=freq)
    if per_flow_path is None:
        per_flow_path = _find_first_existing(reports_dir, _PER_FLOW_PATTERNS, freq=freq)

    out["per_party_time_long"] = _read_csv_if_exists(per_party_path) if per_party_path else pd.DataFrame()
    out["per_flow_time_long"] = _read_csv_if_exists(per_flow_path) if per_flow_path else pd.DataFrame()

    # 3) ledger canonical (in out/)
    p_ledger = parent_out / LEDGER_FN
    if not p_ledger.exists():
        p_ledger = reports_dir / LEDGER_FN
    out["ledger"] = _read_csv_if_exists(p_ledger, parse_dates=["Date"]) if p_ledger.exists() else pd.DataFrame()

    # 4) daily cash position (materialize output)
    p_daily = parent_out / _DAILY_CASH_FN
    if not p_daily.exists():
        p_daily = reports_dir / _DAILY_CASH_FN
    out["daily_cash_position"] = _read_csv_if_exists(p_daily) if p_daily.exists() else pd.DataFrame()

    # 5) renta_*.csv series in reports_dir (normalize Date, amount, currency)
    renta_dfs = []
    for p in sorted(reports_dir.glob("renta_*.csv")):
        df = pd.read_csv(p, low_memory=False)
        # normalize Date / amount columns
        if "TimePeriod_ts_end" in df.columns and "amount" in df.columns:
            df = df.rename(columns={"TimePeriod_ts_end": "Date"})
        elif "Date" not in df.columns and df.shape[1] >= 2:
            df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "amount"})
        # keep only essential columns then normalize
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = _ensure_amount(df)
        # canonical currency
        df = _normalize_currency_col(df)
        party = p.stem.replace("renta_", "")
        df["party"] = party
        renta_dfs.append(df[["Date", "amount", "party", "currency"]])
    out["renta_all"] = pd.concat(renta_dfs, ignore_index=True) if renta_dfs else pd.DataFrame(columns=["Date","amount","party","currency"])

    # 6) manifest (optional)
    p_manifest = parent_out / _MANIFEST_FN
    if not p_manifest.exists():
        p_manifest = reports_dir / _MANIFEST_FN
    if p_manifest.exists():
        try:
            with open(p_manifest, "r") as fh:
                out["_manifest"] = json.load(fh)
        except Exception:
            out["_manifest"] = {}
    else:
        out["_manifest"] = {}

    # 7) canonicalize column names and types for loaded DataFrames
    for key in ("per_party_time_long", "per_flow_time_long", "ledger", "daily_cash_position"):
        df = out.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # ensure currency column and amount numeric
            df = _normalize_currency_col(df)
            df = _ensure_amount(df)
            out[key] = df

    return out




# --- View builders ---
def build_renta_pivot_view(materialized_reports: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    "Return pivot table Date x party with amounts (float)."
    renta = materialized_reports.get("renta_all", pd.DataFrame()).copy()
    if renta.empty:
        return pd.DataFrame()
    renta = renta.dropna(subset=["Date"])
    pivot = renta.pivot_table(index="Date", columns="party", values="amount", aggfunc="sum").fillna(0.0).sort_index()
    return pivot

def build_fondos_wide_view(materialized_reports: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    "Return fondos wide table with parsed index Date if possible and numeric columns."
    fondos = materialized_reports.get("fondos", pd.DataFrame()).copy()
    if fondos.empty:
        return pd.DataFrame()
    # try parse index to Date column if index looks like date
    try:
        fondos_index_parsed = pd.to_datetime(fondos.index.astype(str), errors="coerce")
        if fondos_index_parsed.notna().sum() > 0:
            fondos['__Date_parsed'] = fondos_index_parsed
            fondos = fondos.reset_index().rename(columns={"index": "period_label"})
        else:
            fondos = fondos.reset_index().rename(columns={"index": "period_label"})
    except Exception:
        fondos = fondos.reset_index().rename(columns={"index": "period_label"})
    # coerce numeric columns (except meta)
    for c in fondos.columns:
        if c in ("period_label", "__Date_parsed"):
            continue
        fondos[c] = pd.to_numeric(fondos[c], errors="coerce").fillna(0.0)
    return fondos

# def build_party_balance_view(materialized_reports: Dict[str, pd.DataFrame], freq: str = "W") -> pd.DataFrame:
#     """
#     Build Date x party cumulative balance derived from per_party_time_long if present;
#     fallback: derive from renta pivot (treat renta as incoming receipts).
#     Returns melted DataFrame with columns: Date, party, balance
#     """
#     # try reading explicit per_party materialization CSV if available
#     reports_dir = None
#     # caller should pass materialized_reports loaded already; try to access per_party in ledger fallback
#     if 'per_party' in materialized_reports:
#         perp = materialized_reports['per_party']
#     else:
#         # try loading from CSV path convention (not present here) -> fallback
#         perp = pd.DataFrame()
#     if not perp.empty:
#         perp = perp.copy()
#         if "TimePeriod_ts_end" in perp.columns:
#             perp["Date"] = pd.to_datetime(perp["TimePeriod_ts_end"], errors="coerce")
#         elif "TimePeriod" in perp.columns:
#             perp["Date"] = pd.to_datetime(perp["TimePeriod"].astype(str), errors="coerce")
#         else:
#             perp["Date"] = pd.NaT
#         perp["amount"] = pd.to_numeric(perp.get("amount", 0), errors="coerce").fillna(0.0)
#         perp["signed"] = perp.apply(lambda r: r["amount"] if r.get("role","")=="receiver" else -r["amount"], axis=1)
#         # pivot and cumsum
#         wide = perp.pivot_table(index="Date", columns="party", values="signed", aggfunc="sum").fillna(0.0).sort_index().cumsum()
#         melted = wide.reset_index().melt(id_vars=["Date"], var_name="party", value_name="balance")
#         return melted
#     # fallback: derive from renta pivot as cumulative receipts (no outflows considered)
#     renta_pivot = build_renta_pivot_view(materialized_reports)
#     if renta_pivot.empty:
#         return pd.DataFrame()
#     wide = renta_pivot.sort_index().cumsum()
#     melted = wide.reset_index().melt(id_vars=["Date"], var_name="party", value_name="balance")
#     return melted



import pandas as pd
from typing import Dict, List, Optional

def build_party_timeseries_view(
    materialized_reports: Dict[str, pd.DataFrame],
    freq: str = "W",
    classifier_cols: Optional[List[str]] = None,
    remove_internal_transfers: bool = False,   # default False: produce full per-party view
) -> pd.DataFrame:
    """
    Build a long Date x (party + classifiers) time series with columns:
      Date, party, <classifier_cols...>, in_amt, out_amt, net, cum_net

    Notes:
    - Uses 'per_party' if present; falls back to a renta pivot if not.
    - Resamples by `freq` (e.g. 'W','M'), sums flows, computes cumulative net per group.
    - If remove_internal_transfers True, rows where both party and other_party are in party set are excluded.
    """

    # --- load base per-party table (fallback to renta pivot) ---
    perp = materialized_reports.get("per_party_time_long", pd.DataFrame()).copy()

    print(f"perp.columns: {perp.columns} should have Currency")

    if perp.empty:
        try:
            renta = build_renta_pivot_view(materialized_reports)
            if renta.empty:
                return pd.DataFrame()
            perp = renta.reset_index().melt(id_vars=["Date"], var_name="party", value_name="amount")
            perp["role"] = "receiver"
        except Exception:
            return pd.DataFrame()

    # --- ensure Date and numeric amount ---
    if "Date" not in perp.columns:
        if "TimePeriod_ts_end" in perp.columns:
            perp["Date"] = pd.to_datetime(perp["TimePeriod_ts_end"], errors="coerce")
        elif "TimePeriod" in perp.columns:
            perp["Date"] = pd.to_datetime(perp["TimePeriod"].astype(str), errors="coerce")
        elif "date" in perp.columns:
            perp["Date"] = pd.to_datetime(perp["date"], errors="coerce")
        else:
            perp["Date"] = pd.NaT

    perp["amount"] = pd.to_numeric(perp.get("amount", perp.get("Monto", 0)), errors="coerce").fillna(0.0)

    # --- sign logic: prefer role, fallback heuristics on Flujo/Tipo ---
    def _signed(r):
        role = str(r.get("role", "")).lower()
        if role in ("receiver", "creditor", "in"):
            return r["amount"]
        if role in ("payer", "debtor", "out"):
            return -r["amount"]
        # heuristics
        flujo = str(r.get("Flujo", "")).lower()
        tipo = str(r.get("Tipo", "")).lower()
        if "cobro" in flujo or "renta" in tipo or "renta" in flujo:
            return r["amount"]
        if "pago" in flujo or "mantenimiento" in tipo or "prestamo" in tipo:
            return -r["amount"]
        # default: positive
        return r["amount"]

    perp["signed"] = perp.apply(_signed, axis=1)

    # in/out as non-negative flows
    perp["in_amt"] = perp["signed"].clip(lower=0.0)
    perp["out_amt"] = (-perp["signed"].clip(upper=0.0))

    # --- classifier columns defaults ---
    if classifier_cols is None:
        classifier_cols = [c for c in ["Flujo", "Tipo", "Lugar", "lugar"] if c in perp.columns]
    # ensure all classifier cols exist and are strings
    for c in classifier_cols:
        if c not in perp.columns:
            perp[c] = "NA"
        else:
            perp[c] = perp[c].fillna("NA").astype(str)

    print(f"once again: perp.columns: {perp.columns} should have Currency")

    if "Currency" not in perp.columns:
        # keep consistent default label (use config.base_currency if you prefer)
        print("NaN at position 10 - Currency not at per_person columns")    ## Here 4th and 7th
        perp["Currency"] = "NA"
    else:
        print("Ok at position 11")
        perp["Currency"] = perp["Currency"].fillna("NA").astype(str).str.upper()


    # # --- internal-transfer detection & removal (optional) ---
    # if remove_internal_transfers and ("other_party" in perp.columns or "party_to" in perp.columns):
    #     other_col = "other_party" if "other_party" in perp.columns else "party_to"
    #     parties_set = set(perp["party"].dropna().unique())
    #     perp["_is_internal"] = perp.apply(
    #         lambda r: (str(r.get(other_col, "")).strip() in parties_set) and (str(r.get("party", "")).strip() in parties_set),
    #         axis=1
    #     )
    #     perp = perp.loc[~perp["_is_internal"]].copy()
    #     perp.drop(columns=["_is_internal"], inplace=True)

    # drop rows without Date
    perp = perp.dropna(subset=["Date"])
    perp = perp.set_index("Date")

    # --- group + resample ---
    # include currency in grouping only if present and meaningful
    group_cols = classifier_cols + ["party"]
    if "Currency" in perp.columns and perp["Currency"].nunique() > 1:
        # only add currency as grouping level if it actually varies; avoids useless duplicates
        group_cols.append("Currency")
    # fallback: if currency exists but always "NA", grouping by party alone is cleaner

    agg_cols = ["in_amt", "out_amt", "signed"]

    # groupby + resample uses DateIndex
    grouped = perp.groupby(group_cols).resample(freq)[agg_cols].sum().reset_index()

    # compute net and cumulative net
    grouped["net"] = grouped["in_amt"] - grouped["out_amt"]
    grouped = grouped.sort_values(group_cols + ["Date"])
    grouped["cum_net"] = grouped.groupby(group_cols)["net"].cumsum()

    # final columns
    out_cols = ["Date"] + group_cols + ["in_amt", "out_amt", "net", "cum_net"]
    result = grouped[out_cols].copy()
    return result




# --- Export runner ---
def export_views(reports_dir: Path, write_dir: Path, freq: str = "W") -> Dict[str, str]:
    reports = load_reports_folder(Path(reports_dir))
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    renta_pivot = build_renta_pivot_view(reports)
    if not renta_pivot.empty:
        atomic_write_df(renta_pivot.reset_index().rename(columns={"index":"Date"}), write_dir / "renta_pivot.csv")
        outputs["renta_pivot.csv"] = str(write_dir / "renta_pivot.csv")
    fondos_w = build_fondos_wide_view(reports)
    if not fondos_w.empty:
        atomic_write_df(fondos_w, write_dir / "fondos_wide.csv")
        outputs["fondos_wide.csv"] = str(write_dir / "fondos_wide.csv")


    # 1) Full per-party detailed long series (keeps internal transfers)
    party_detailed = build_party_timeseries_view(reports, freq=freq, remove_internal_transfers=False)
    if not party_detailed.empty:
        atomic_write_df(party_detailed, write_dir / "party_balance_detailed.csv")
        outputs["party_balance_detailed.csv"] = str(write_dir / "party_balance_detailed.csv")

    # # 2) Wide pivots (net & cum) per party (from full detailed)
    # if not party_detailed.empty:
    #     net_wide = party_detailed.pivot_table(index="Date", columns=["party", "Currency"], values="net", aggfunc="sum").fillna(0)
    #     cum_wide = party_detailed.pivot_table(index="Date", columns=["party", "Currency"], values="cum_net", aggfunc="sum").fillna(0)
    #     # write as CSVs with Date column
    #     atomic_write_df(net_wide.reset_index().rename(columns={"index": "Date"}), write_dir / "party_balance_net_wide.csv")
    #     atomic_write_df(cum_wide.reset_index().rename(columns={"index": "Date"}), write_dir / "party_balance_cum_wide.csv")
    #     outputs["party_balance_net_wide.csv"] = str(write_dir / "party_balance_net_wide.csv")
    #     outputs["party_balance_cum_wide.csv"] = str(write_dir / "party_balance_cum_wide.csv")


    # 2) Wide pivots (net & cum) per party (from full detailed)
    if not party_detailed.empty:
        # ensure currency exists (defensive)

        

        if "Currency" not in party_detailed.columns:
            print("NaN at position 3")
            party_detailed["Currency"] = "NA"
        # pivot by party+currency (this will create columns like ('PM','ARS'), etc.)
        net_wide_pc = party_detailed.pivot_table(
            index="Date", columns=["party", "Currency"], values="net", aggfunc="sum"
        ).fillna(0)
        cum_wide_pc = party_detailed.pivot_table(
            index="Date", columns=["party", "Currency"], values="cum_net", aggfunc="sum"
        ).fillna(0)

        # write multi-index CSV (pandas flattens multiindex when saving -> keep as-is)
        atomic_write_df(net_wide_pc.reset_index().rename(columns={"index": "Date"}), write_dir / "party_balance_net_wide.csv")
        atomic_write_df(cum_wide_pc.reset_index().rename(columns={"index": "Date"}), write_dir / "party_balance_cum_wide.csv")
        outputs["party_balance_net_wide.csv"] = str(write_dir / "party_balance_net_wide.csv")
        outputs["party_balance_cum_wide.csv"] = str(write_dir / "party_balance_cum_wide.csv")

        # ALSO provide party-only wide summary (sum across currencies)
        net_wide_party = net_wide_pc.groupby(level=0, axis=1).sum() if isinstance(net_wide_pc.columns, pd.MultiIndex) else net_wide_pc
        cum_wide_party = cum_wide_pc.groupby(level=0, axis=1).sum() if isinstance(cum_wide_pc.columns, pd.MultiIndex) else cum_wide_pc

        atomic_write_df(net_wide_party.reset_index().rename(columns={"index": "Date"}), write_dir / "party_balance_net_wide_party_only.csv")
        atomic_write_df(cum_wide_party.reset_index().rename(columns={"index": "Date"}), write_dir / "party_balance_cum_wide_party_only.csv")
        outputs["party_balance_net_wide_party_only.csv"] = str(write_dir / "party_balance_net_wide_party_only.csv")
        outputs["party_balance_cum_wide_party_only.csv"] = str(write_dir / "party_balance_cum_wide_party_only.csv")



    # 3) Aggregation by Flujo & Tipo (drop party to get grouped totals)
    # prefer classifier columns in priority: Flujo, Tipo
    if not party_detailed.empty:
        # Ensure Flujo and Tipo exist
        tmp = party_detailed.copy()
        if "Flujo" not in tmp.columns:
            print(f"NaN at position 8 - Flujo not in party_detailed.columns: {party_detailed.columns}")   ## Here 5th
            tmp["Flujo"] = "NA"
        if "Tipo" not in tmp.columns:
            print(f"NaN at position 9 - Flujo not in party_detailed.columns: {party_detailed.columns}")   ## Here 6th
            tmp["Tipo"] = "NA"
        by_f = tmp.groupby(["Date", "Flujo", "Tipo"])[["in_amt", "out_amt", "net"]].sum().reset_index()
        # compute cumulative per Flujo/Tipo grouping
        by_f = by_f.sort_values(["Flujo", "Tipo", "Date"])
        by_f["cum_net"] = by_f.groupby(["Flujo", "Tipo"])["net"].cumsum()
        atomic_write_df(by_f, write_dir / "party_balance_by_flujo_tipo.csv")
        outputs["party_balance_by_flujo_tipo.csv"] = str(write_dir / "party_balance_by_flujo_tipo.csv")

    # 4) Consolidated family view: remove internal transfers then aggregate across parties
    # party_consol = build_party_timeseries_view(reports, freq=freq, remove_internal_transfers=True)
    party_consol = build_party_timeseries_view(reports, freq=freq, remove_internal_transfers=False)
    if not party_consol.empty:
        # ensure Currency exists
        if "Currency" not in party_consol.columns:
            party_consol["Currency"] = "NA"

        # net per date + currency
        consol = (
            party_consol
            .groupby(["Date", "Currency"], as_index=False)["net"]
            .sum()
            .sort_values(["Currency", "Date"])
        )
        # cum_net per currency
        consol["cum_net"] = consol.groupby("Currency")["net"].cumsum()

        atomic_write_df(consol, write_dir / "consolidated_balance.csv")
        outputs["consolidated_balance.csv"] = str(write_dir / "consolidated_balance.csv")

        # optional alias, keep if you want
        atomic_write_df(consol, write_dir / "consolidated_balance_wide.csv")
        outputs["consolidated_balance_wide.csv"] = str(write_dir / "consolidated_balance_wide.csv")


    # 5) upcoming payables from ledger (keeps your original logic)
    ledger = reports.get("ledger", pd.DataFrame())
    if not ledger.empty:
        now = pd.Timestamp.now().normalize()
        upcoming = ledger.loc[
            (ledger["Date"].notna()) & (ledger["Date"] >= now) & (ledger["Date"] <= now + pd.Timedelta(days=90))
        ]
        atomic_write_df(upcoming.sort_values("Date"), write_dir / "upcoming_90.csv")
        outputs["upcoming_90.csv"] = str(write_dir / "upcoming_90.csv")

    return outputs


# src/accounting/views_cli.py
from pathlib import Path
import argparse
import json
# from accounting.views import export_views # its here

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="out/reports")
    p.add_argument("--write-dir", default="out/views")
    p.add_argument("--freq", default="W")
    args = p.parse_args(argv)
    out = export_views(Path(args.reports_dir), Path(args.write_dir), freq=args.freq)
    print(json.dumps(out, indent=2))
if __name__ == "__main__":
    main()


