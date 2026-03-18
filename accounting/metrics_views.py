from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


DEFAULT_NOISE_FLOOR = {"ARS": 5000.0, "USD": 10.0}
DEFAULT_INCLUDE_STATUSES = ("pagado",)


def resolve_amount_col(df: pd.DataFrame) -> str:
    candidates = ["amount", "monto", "Amount", "Debit", "Credit"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No amount-like column found. Available columns: {list(df.columns)}")


def resolve_col(df: pd.DataFrame, preferred: str, aliases: list[str]) -> str:
    if preferred in df.columns:
        return preferred
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(f"Missing required column '{preferred}'. Available columns: {list(df.columns)}")


def parse_noise_floor(text: str) -> Dict[str, float]:
    if not text.strip():
        return dict(DEFAULT_NOISE_FLOOR)
    out: Dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        curr, val = part.split(":", 1)
        out[curr.strip().upper()] = float(val.strip())
    return out


def load_ledger(run_root: Path) -> pd.DataFrame:
    path = run_root / "ledger_canonical.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ledger file: {path}")

    df = pd.read_csv(path)

    date_col = resolve_col(df, "Date", ["date", "posted_date"])
    amount_col = resolve_amount_col(df)
    currency_col = resolve_col(df, "Currency", ["currency"])
    status_col = resolve_col(df, "status", ["Status"])
    flujo_col = resolve_col(df, "Flujo", ["flujo"])
    tipo_col = resolve_col(df, "Tipo", ["tipo"])

    rename_map = {
        date_col: "Date",
        amount_col: "amount",
        currency_col: "Currency",
        status_col: "status",
        flujo_col: "Flujo",
        tipo_col: "Tipo",
    }

    for optional in ["Box", "Lugar", "Detalle", "payer", "receiver", "medio", "tag"]:
        if optional in df.columns:
            rename_map[optional] = optional

    df = df.rename(columns=rename_map).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["status"] = df["status"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=["Date", "amount", "Currency"]).copy()
    df["period_m"] = df["Date"].dt.to_period("M").astype(str)
    return df


def last_n_months(df: pd.DataFrame, months: int) -> List[str]:
    periods = sorted(df["period_m"].dropna().astype(str).unique().tolist())
    return periods[-months:]


def apply_noise_floor_rows(
    df: pd.DataFrame,
    total_col: str,
    currency_col: str,
    noise_floor_by_currency: Dict[str, float],
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()

    def keep_row(row: pd.Series) -> bool:
        curr = str(row.get(currency_col, "")).upper()
        thr = noise_floor_by_currency.get(curr, None)
        if thr is None:
            return True
        val = row.get(total_col, pd.NA)
        if pd.isna(val):
            return True
        return abs(float(val)) >= thr

    mask = work.apply(keep_row, axis=1)
    return work.loc[mask].reset_index(drop=True)


def build_flow_rollup_last_n_months(
    ledger: pd.DataFrame,
    *,
    flow_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    groupby_cols: Sequence[str],
    months: int = 6,
    include_statuses: Sequence[str] = DEFAULT_INCLUDE_STATUSES,
    amount_col: str = "amount",
    currency_col: str = "Currency",
    status_col: str = "status",
    noise_floor_by_currency: Optional[Dict[str, float]] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    work = ledger.copy()

    if flow_filter is not None:
        work = work.loc[work["Flujo"].astype(str) == flow_filter].copy()
    if type_filter is not None:
        work = work.loc[work["Tipo"].astype(str) == type_filter].copy()
    if include_statuses:
        allowed = {str(x).strip().lower() for x in include_statuses}
        work = work.loc[work[status_col].astype(str).str.strip().str.lower().isin(allowed)].copy()

    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
    work = work.dropna(subset=[amount_col, currency_col]).copy()

    months_list = last_n_months(work, months)
    if months_list:
        work = work.loc[work["period_m"].isin(months_list)].copy()

    missing_cols = [c for c in groupby_cols if c not in work.columns]
    if missing_cols:
        raise ValueError(f"Missing groupby columns in ledger: {missing_cols}")

    work[groupby_cols] = work[groupby_cols].fillna("").astype(str)

    base_group_cols = []
    for c in list(groupby_cols):
        if c not in base_group_cols:
            base_group_cols.append(c)

    if currency_col not in base_group_cols:
        base_group_cols.append(currency_col)

    group_cols = base_group_cols + ["period_m"]

    grouped = work.groupby(group_cols, dropna=False)[amount_col].sum().reset_index()

    if grouped.empty:
        cols = base_group_cols + months_list + ["total_6m", "avg_m", "last_m", "delta_last_vs_prev"]
        return pd.DataFrame(columns=cols)

    wide = (
        grouped.pivot_table(
            index=base_group_cols,
            columns="period_m",
            values=amount_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )

    for m in months_list:
        if m not in wide.columns:
            wide[m] = 0.0

    wide["total_6m"] = wide[months_list].sum(axis=1) if months_list else 0.0
    wide["avg_m"] = wide["total_6m"] / max(len(months_list), 1)
    wide["last_m"] = wide[months_list[-1]] if months_list else 0.0
    wide["delta_last_vs_prev"] = (
        wide[months_list[-1]] - wide[months_list[-2]] if len(months_list) >= 2 else pd.NA
    )

    if noise_floor_by_currency:
        wide = apply_noise_floor_rows(wide, "total_6m", currency_col, noise_floor_by_currency)

    wide = wide.sort_values(["total_6m"], ascending=False).reset_index(drop=True)

    if top_n is not None and len(wide) > top_n:
        wide = wide.head(top_n).reset_index(drop=True)

    ordered_cols = base_group_cols + months_list + ["total_6m", "avg_m", "last_m", "delta_last_vs_prev"]
    return wide[ordered_cols]


def build_income_statement_monthly_last6(
    ledger: pd.DataFrame,
    *,
    months: int,
    include_statuses: Sequence[str],
) -> pd.DataFrame:
    work = ledger.copy()
    allowed = {str(x).strip().lower() for x in include_statuses}
    work = work.loc[work["status"].astype(str).str.strip().str.lower().isin(allowed)].copy()
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work = work.dropna(subset=["amount", "Currency"]).copy()

    months_list = last_n_months(work, months)
    work = work.loc[work["period_m"].isin(months_list)].copy()

    specs = [
        ("IS.RENT.TOTAL", "Renta total", (work["Flujo"].astype(str) == "Cobros") & (work["Tipo"].astype(str) == "Renta")),
        ("IS.CONTRIB.TOTAL", "Contribuciones totales", work["Flujo"].astype(str) == "Contribucion"),
        ("IS.OPEX.TOTAL", "Costos operativos totales", work["Flujo"].astype(str) == "Pagos"),
    ]

    rows: list[dict[str, Any]] = []
    for metric_id, label, mask in specs:
        sub = work.loc[mask].copy()
        if sub.empty:
            continue
        agg = sub.groupby(["Currency", "period_m"], dropna=False)["amount"].sum().reset_index()
        wide = (
            agg.pivot_table(index=["Currency"], columns="period_m", values="amount", aggfunc="sum", fill_value=0.0)
            .reset_index()
        )
        for m in months_list:
            if m not in wide.columns:
                wide[m] = 0.0
        for _, row in wide.iterrows():
            out = {"metric_id": metric_id, "label": label, "currency": row["Currency"]}
            vals = [row[m] for m in months_list]
            for m in months_list:
                out[m] = row[m]
            out["total_6m"] = sum(vals)
            out["avg_m"] = out["total_6m"] / max(len(months_list), 1)
            rows.append(out)

    df = pd.DataFrame(rows)
    if not df.empty:
        derived_rows = []
        for currency in sorted(df["currency"].astype(str).unique().tolist()):
            sub = df.loc[df["currency"].astype(str) == currency].set_index("metric_id")
            rent = sub.loc["IS.RENT.TOTAL"] if "IS.RENT.TOTAL" in sub.index else None
            contrib = sub.loc["IS.CONTRIB.TOTAL"] if "IS.CONTRIB.TOTAL" in sub.index else None
            opex = sub.loc["IS.OPEX.TOTAL"] if "IS.OPEX.TOTAL" in sub.index else None

            if rent is not None or contrib is not None:
                out = {"metric_id": "IS.INCOME.TOTAL", "label": "Ingresos totales", "currency": currency}
                for m in months_list:
                    out[m] = (rent[m] if rent is not None else 0.0) + (contrib[m] if contrib is not None else 0.0)
                out["total_6m"] = sum(out[m] for m in months_list)
                out["avg_m"] = out["total_6m"] / max(len(months_list), 1)
                derived_rows.append(out)

            if (rent is not None or contrib is not None) and opex is not None:
                base = derived_rows[-1] if derived_rows else None
                if base and base["metric_id"] == "IS.INCOME.TOTAL":
                    out = {"metric_id": "IS.NET.AFTER_COSTS", "label": "Neto después de costos", "currency": currency}
                    for m in months_list:
                        out[m] = base[m] - opex[m]
                    out["total_6m"] = sum(out[m] for m in months_list)
                    out["avg_m"] = out["total_6m"] / max(len(months_list), 1)
                    derived_rows.append(out)

        if derived_rows:
            df = pd.concat([df, pd.DataFrame(derived_rows)], ignore_index=True)

    ordered_metric_ids = [
        "IS.RENT.TOTAL",
        "IS.CONTRIB.TOTAL",
        "IS.INCOME.TOTAL",
        "IS.OPEX.TOTAL",
        "IS.NET.AFTER_COSTS",
    ]
    df["__sort"] = df["metric_id"].map({k: i for i, k in enumerate(ordered_metric_ids)})
    cols = ["metric_id", "label", "currency"] + months_list + ["total_6m", "avg_m"]
    return df.sort_values(["currency", "__sort"]).reset_index(drop=True)[cols]


def build_draws_discipline_monthly_last6(
    ledger: pd.DataFrame,
    *,
    months: int,
    include_statuses: Sequence[str],
    noise_floor_by_currency: Dict[str, float],
) -> pd.DataFrame:
    work = ledger.copy()
    allowed = {str(x).strip().lower() for x in include_statuses}
    work = work.loc[work["status"].astype(str).str.strip().str.lower().isin(allowed)].copy()
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work = work.dropna(subset=["amount", "Currency"]).copy()
    months_list = last_n_months(work, months)
    work = work.loc[work["period_m"].isin(months_list)].copy()

    text_cols = [c for c in ["Tipo", "Detalle", "tag", "Lugar"] if c in work.columns]
    mask = pd.Series(False, index=work.index)
    for c in text_cols:
        mask = mask | work[c].astype(str).str.contains(r"personal|retiro|draw|owner|dividend", case=False, na=False)

    draws = work.loc[mask].groupby(["Currency", "period_m"], dropna=False)["amount"].sum().reset_index()
    net = build_income_statement_monthly_last6(work, months=months, include_statuses=include_statuses)
    net = net.loc[net["metric_id"] == "IS.NET.AFTER_COSTS"].copy()

    rows = []
    for currency in sorted(set(draws["Currency"].astype(str)) | set(net["currency"].astype(str))):
        dsub = draws.loc[draws["Currency"].astype(str) == currency]
        nsub = net.loc[net["currency"].astype(str) == currency]
        row = {"currency": currency}
        total_draws = 0.0
        distress = 0
        for m in months_list:
            d = dsub.loc[dsub["period_m"].astype(str) == m, "amount"]
            n = nsub[m].iloc[0] if (not nsub.empty and m in nsub.columns) else pd.NA
            draw_val = float(d.iloc[0]) if not d.empty else 0.0
            row[f"draws_{m}"] = draw_val
            row[f"net_{m}"] = n
            if pd.notna(n) and float(n) <= 0 and draw_val > 0:
                distress += 1
            total_draws += draw_val
        row["draws_total_6m"] = total_draws
        row["distress_months"] = distress
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = apply_noise_floor_rows(df, "draws_total_6m", "currency", noise_floor_by_currency)
    return df
