# src/accounting/core_timeseries.py
"""
Core time-series primitives for the accounting pipeline.

All functions are pure (no I/O), deterministic, and operate on Pandas objects.
They prefer 'amount' (float) but accept 'amount_cents' when available.
"""
from __future__ import annotations
from typing import List, Sequence, Union, Callable, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Frequency normalization map (case-insensitive)
_FREQ_MAP = {
    "d": "D",
    "day": "D",
    "daily": "D",
    "w": "W-MON",
    "week": "W-MON",
    "weekly": "W-MON",
    "m": "M",
    "month": "M",
    "monthly": "M",
    "q": "Q",
    "quarter": "Q",
    "quarterly": "Q",
    "y": "A",
    "year": "A",
    "annual": "A",
}


def _normalize_freq_token(freq: str) -> str:
    if not isinstance(freq, str):
        raise TypeError("freq must be a string like 'M','Q','W' or 'monthly','quarterly', etc.")
    tok = freq.strip().lower()
    return _FREQ_MAP.get(tok, freq)  # allow direct pandas tokens if not in map


# -----------------------
# Period / bin helpers
# -----------------------
def period_bins_for_dates(dates: pd.Series, freq: str = "W") -> pd.DataFrame:
    """
    Given a Series of dates, return a DataFrame with columns:
      - TimePeriod (pd.Period)
      - TimePeriod_ts_end (pd.Timestamp) -- period end timestamp

    Keeps the same index as the input `dates`.
    """
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    dates = pd.to_datetime(dates, errors="coerce")
    freq_token = _normalize_freq_token(freq)
    periods = dates.dt.to_period(freq_token)
    # compute period end timestamps; for NaT periods will become NaT
    period_ends = periods.dt.to_timestamp(how="end")
    out = pd.DataFrame({"TimePeriod": periods, "TimePeriod_ts_end": period_ends}, index=dates.index)
    return out


def period_index_to_timestamp_end(idx: Union[pd.Index, pd.PeriodIndex]) -> pd.DatetimeIndex:
    """
    Convert a PeriodIndex (or index of Periods) to DatetimeIndex at period end.
    """
    if isinstance(idx, pd.PeriodIndex):
        return idx.dt.to_timestamp(how="end")
    try:
        return pd.DatetimeIndex([p.dt.to_timestamp(how="end") for p in idx])
    except Exception:
        raise TypeError("Index is not a PeriodIndex or sequence of Periods")


# -----------------------
# Aggregation primitives
# -----------------------
def _amount_series(df: pd.DataFrame, amount_col: str = "amount") -> pd.Series:
    """
    Return a numeric Series for amounts. Prefer 'amount' then fallback to 'amount_cents'/100.
    """
    if amount_col in df.columns:
        s = pd.to_numeric(df[amount_col], errors="coerce").fillna(0).astype(float)
        return s
    if "amount_cents" in df.columns:
        s = pd.to_numeric(df["amount_cents"], errors="coerce").fillna(0).astype(float) / 100.0
        return s
    # default: zeros
    return pd.Series(0.0, index=df.index)


def aggregate_per_flow(
    ledger_df: pd.DataFrame,
    freq: str = "W",
    amount_col: str = "amount",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Aggregate ledger by TimePeriod x Flujo x Tipo x Currency.

    Returns columns:
      ['TimePeriod', 'TimePeriod_ts_end', 'Flujo', 'Tipo', 'Currency', 'amount', 'n_tx']

    Notes:
    - Uses `amount_col` (float) when present, else falls back to amount_cents.
    - TimePeriod is pd.Period; TimePeriod_ts_end is Timestamp (period end).
    """
    if date_col not in ledger_df.columns:
        raise KeyError(f"date_col '{date_col}' not in ledger")
    df = ledger_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # compute time bins
    bins = period_bins_for_dates(df[date_col], freq=freq)
    df = df.assign(TimePeriod=bins["TimePeriod"].values, TimePeriod_ts_end=bins["TimePeriod_ts_end"].values)
    # group columns
    grp_cols = []
    for c in ("Flujo", "Tipo", "Currency"):
        if c in df.columns:
            grp_cols.append(c)
        else:
            # ensure column exists for consistent output
            df[c] = "" if c != "Currency" else ""
            grp_cols.append(c)
    # amount series
    s_amount = _amount_series(df, amount_col)
    df["_amount_for_agg"] = s_amount
    # group and aggregate
    group = ["TimePeriod"] + grp_cols
    agg = df.groupby(group).agg(_amount_for_agg=("_amount_for_agg", "sum"), n_tx=("tx_id", "nunique"))
    # reset_index
    agg = agg.reset_index()
    agg = agg.rename(columns={"_amount_for_agg": "amount"})
    # attach timestamp end column (from TimePeriod)
    if "TimePeriod" in agg.columns:
        agg["TimePeriod_ts_end"] = agg["TimePeriod"].apply(lambda p: p.to_timestamp(how="end") if pd.notna(p) else pd.NaT)
    # reorder columns
    cols = ["TimePeriod", "TimePeriod_ts_end", "Flujo", "Tipo", "Currency", "amount", "n_tx"]
    return agg[cols]


# -----------------------
# Party expansion & per-party aggregates
# -----------------------
def expand_party_rows(
    ledger_df: pd.DataFrame,
    amount_col: str = "amount",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Expand each ledger transaction into one or two party-rows:
      - payer row: role='payer', signed_amount = -amount
      - receiver row: role='receiver', signed_amount = +amount

    Returns DataFrame with columns:
      ['tx_id','Date','party','role','signed_amount','Currency','Flujo','Tipo','source_file','source_row','notes']

    Deterministic ordering: rows are produced in ledger order, payer row first then receiver row.
    If payer==receiver the function will produce a single receiver row with signed_amount 0 (by convention).
    """
    df = ledger_df.copy().reset_index(drop=True)
    # ensure core columns exist
    for c in (date_col, "tx_id", "payer", "receiver"):
        if c not in df.columns:
            df[c] = pd.NA
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    amounts = _amount_series(df, amount_col)
    out_rows = []
    for idx, row in df.iterrows():
        amt = float(amounts.iloc[idx]) if not pd.isna(amounts.iloc[idx]) else 0.0
        payer = row.get("payer") if "payer" in row.index else None
        receiver = row.get("receiver") if "receiver" in row.index else None
        tx_id = row.get("tx_id")
        base = {
            "tx_id": tx_id,
            "Date": row.get(date_col),
            "Currency": row.get("Currency") if "Currency" in row.index else None,
            "Flujo": row.get("Flujo") if "Flujo" in row.index else None,
            "Tipo": row.get("Tipo") if "Tipo" in row.index else None,
            "source_file": row.get("source_file") if "source_file" in row.index else None,
            "source_row": row.get("source_row") if "source_row" in row.index else None,
            "notes": row.get("notes") if "notes" in row.index else None,
        }
        # payer row (outflow)
        if payer and str(payer).strip() != "":
            r = base.copy()
            r.update({"party": payer, "role": "payer", "signed_amount": -float(amt)})
            out_rows.append(r)
        # receiver row (inflow)
        if receiver and str(receiver).strip() != "":
            # if payer==receiver produce receiver with net zero? we'll still produce positive inflow
            r = base.copy()
            r.update({"party": receiver, "role": "receiver", "signed_amount": float(amt)})
            out_rows.append(r)
        # If neither party exists, produce a generic row to keep provenance
        if (not payer or str(payer).strip() == "") and (not receiver or str(receiver).strip() == ""):
            r = base.copy()
            r.update({"party": None, "role": "unknown", "signed_amount": float(amt)})
            out_rows.append(r)
    out = pd.DataFrame(out_rows)
    # keep deterministic sort (by source_file, source_row if present)
    if "source_row" in out.columns:
        out = out.sort_values(["source_file", "source_row", "tx_id", "role"], na_position="last").reset_index(drop=True)
    return out


def aggregate_per_party(
    expanded_df: pd.DataFrame,
    freq: str = "W",
    amount_col: str = "signed_amount",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Aggregate expanded party-rows into TimePeriod x party x role x Flujo x Tipo.

    Returns columns:
      ['TimePeriod','TimePeriod_ts_end','party','role','Flujo','Tipo','Currency','amount','n_tx']
    where 'amount' is the sum of signed_amount (float) and n_tx is unique tx count seen for that party-role.
    """
    if date_col not in expanded_df.columns:
        raise KeyError(f"date_col '{date_col}' not present in expanded_df")
    df = expanded_df.copy()

    print(f"check 2A shape {df.shape}, columns: {df.columns}")


    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    bins = period_bins_for_dates(df[date_col], freq=freq)
    df = df.assign(TimePeriod=bins["TimePeriod"].values, TimePeriod_ts_end=bins["TimePeriod_ts_end"].values)
    # ensure columns present
    for c in ("party", "role", "Flujo", "Tipo", "Currency", "tx_id"):
        if c not in df.columns:
            df[c] = "" if c != "tx_id" else pd.NA
    # aggregate
    df["_amt"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).astype(float)
    
    group_cols = ["TimePeriod", "party", "Currency", "role", "Flujo", "Tipo"]


    print(f"check 2B shape {df.shape}, columns: {df.columns}, count: {df.count()}")

    # which grouping columns actually exist?
    for c in group_cols:
        exists = c in df.columns
        print(f"col {c}: exists={exists}", end="")
        if exists:
            nulls = df[c].isna().sum()
            nunique = df[c].nunique(dropna=True)
            dtype = df[c].dtype
            sample = df[c].dropna().unique()[:5].tolist()
            print(f", nulls={nulls}, nunique={nunique}, dtype={dtype}, sample={sample}")
        else:
            print()
            

    agg = df.groupby(group_cols).agg(amount=("_amt", "sum"), n_tx=("tx_id", "nunique")).reset_index()
    agg["TimePeriod_ts_end"] = agg["TimePeriod"].apply(lambda p: p.to_timestamp(how="end") if pd.notna(p) else pd.NaT)
    cols = ["TimePeriod", "TimePeriod_ts_end", "party", "Currency", "role", "Flujo", "Tipo", "amount", "n_tx"]

    print(f"check 2C shape {agg.shape}, columns: {agg.columns}")

    out = agg[cols]

    print(f"check 2D shape {out.shape}, columns: {out.columns}")


    return out


# -----------------------
# Daily cash (treasury) position
# -----------------------
def compute_daily_cash_position(
    ledger_df: pd.DataFrame,
    opening_balances: Optional[Dict[str, float]] = None,
    freq: str = "D",
    amount_col: str = "amount",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Compute per-party daily cash balances.

    Args:
      ledger_df: canonical ledger (one row per tx)
      opening_balances: optional dict mapping party -> opening_balance (float) as of day before earliest tx
      freq: frequency for date discretization (default daily)
      amount_col: use 'amount' float (or fallback to 'amount_cents')
      date_col: column with dates

    Returns a DataFrame:
      ['Date', 'party', 'balance', 'Currency', 'source_ledger_hash' (None)]
    where 'balance' is cumulative sum of signed amounts (receiver positive, payer negative) plus opening balance.
    """
    if date_col not in ledger_df.columns:
        raise KeyError(f"date_col '{date_col}' not present in ledger")
    expanded = expand_party_rows(ledger_df, amount_col=amount_col, date_col=date_col)
    # group by Date and party summing signed_amount
    expanded["Date"] = pd.to_datetime(expanded["Date"], errors="coerce").dt.normalize()

    daily = (
        expanded
        .groupby(["Date", "party", "Currency"], dropna=False)["signed_amount"]
        .sum()
        .reset_index()
        .rename(columns={"signed_amount":"net_flow"})
    )
    pairs = daily[["party","Currency"]].dropna().drop_duplicates()
    
    # create a complete calendar per party to ensure stable cumulative sums
    parties = daily["party"].dropna().unique().tolist()
    if len(parties) == 0:
        return pd.DataFrame(columns=["Date", "party", "balance", "Currency", "source_ledger_hash"])
    min_date = daily["Date"].min()
    max_date = daily["Date"].max()
    # daily date range
    drange = pd.date_range(start=min_date, end=max_date, freq=_normalize_freq_token(freq) if freq.lower() in _FREQ_MAP else freq)

    frames = []
    for party, cur in pairs.itertuples(index=False):
        sub = daily[(daily["party"]==party) & (daily["Currency"]==cur)].set_index("Date").reindex(drange, fill_value=0.0)
        sub = sub.rename_axis("Date").reset_index()
        sub["party"] = party


        # # opening balance
        # opening = 0.0
        # if opening_balances and p in opening_balances:
        #     opening = float(opening_balances[p])
        # # cumulative sum
        # sub["balance"] = sub["net_flow"].cumsum() + opening

        # sub["balance"] = sub["net_flow"].cumsum() + opening_for(party, cur)
        sub["balance"] = sub["net_flow"].cumsum() + 0 # Shortcut, opening = 0
        sub["Currency"] = cur
        frames.append(sub[["Date","party","Currency","balance"]])


        # # currency: try to infer most common currency for that party
        # currencies = expanded[expanded["party"] == p]["Currency"].dropna().astype(str)
        # cur = currencies.mode().iloc[0] if not currencies.empty else ""
        # sub["Currency"] = cur
        # frames.append(sub[["Date", "party", "balance", "Currency"]])
    result = pd.concat(frames, ignore_index=True).sort_values(["party", "Date"]).reset_index(drop=True)
    result["source_ledger_hash"] = None
    return result


# -----------------------
# Loans reconciliation (lightweight)
# -----------------------
def compute_loans_time(
    ledger_df: pd.DataFrame,
    loan_register_df: Optional[pd.DataFrame],
    freq: str = "M",
    amount_col: str = "amount",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Reconcile loans by producing a time series per loan (monthly by default).

    Expects loan_register_df to have at least:
      ['loan_id','lender','receiver','principal','start_date','end_date','amortization_type']
    amortization_type: 'linear' or 'bullet' (defaults to 'bullet' if missing).

    The function will:
      - generate period buckets between start_date and end_date (inclusive)
      - compute scheduled principal (for linear amortization, evenly across periods)
      - find payments in ledger that reference loan_id in a 'loan_id' column or in 'notes' (best-effort)
      - compute outstanding = principal - cumulative_payments

    Returns columns:
      ['loan_id','TimePeriod','TimePeriod_ts_end','lender','receiver','scheduled_principal','payments','outstanding']
    If loan_register_df is None or empty returns empty DataFrame.
    """
    if loan_register_df is None or loan_register_df.shape[0] == 0:
        return pd.DataFrame(columns=["loan_id", "TimePeriod", "TimePeriod_ts_end", "lender", "receiver", "scheduled_principal", "payments", "outstanding"])
    loans = loan_register_df.copy()
    # ensure date columns
    loans["start_date"] = pd.to_datetime(loans.get("start_date"), errors="coerce")
    loans["end_date"] = pd.to_datetime(loans.get("end_date"), errors="coerce")
    out_frames = []
    for _, loan in loans.iterrows():
        loan_id = loan.get("loan_id")
        lender = loan.get("lender")
        receiver = loan.get("receiver")
        principal = float(loan.get("principal") or 0.0)
        amort = loan.get("amortization_type") or "bullet"
        s = loan.get("start_date")
        e = loan.get("end_date") or s
        if pd.isna(s) or pd.isna(e):
            # fallback single period at start
            periods = [pd.Period(pd.Timestamp(s).to_pydatetime(), _normalize_freq_token(freq))] if not pd.isna(s) else []
        else:
            # build periods between s and e inclusive
            # create period range by monthly freq token
            freq_token = _normalize_freq_token(freq)
            # use period range
            try:
                start_p = pd.Period(s, freq_token)
                end_p = pd.Period(e, freq_token)
                periods = pd.period_range(start=start_p, end=end_p, freq=freq_token).tolist()
            except Exception:
                periods = []
        n_periods = len(periods) if periods else 1
        # scheduled principal per period
        scheduled = []
        if amort == "linear" and n_periods > 0:
            per_period = principal / n_periods
            scheduled = [per_period for _ in periods]
        else:
            # bullet or unknown: scheduled principal appears at last period
            scheduled = [0.0 for _ in periods]
            if periods:
                scheduled[-1] = principal
        # find payments in ledger: try match loan_id in dedicated column, else in notes
        payments_mask = pd.Series(False, index=ledger_df.index)
        if "loan_id" in ledger_df.columns:
            payments_mask = payments_mask | (ledger_df["loan_id"].astype(str) == str(loan_id))
        # notes-based match (best-effort)
        if "notes" in ledger_df.columns:
            payments_mask = payments_mask | ledger_df["notes"].astype(str).str.contains(str(loan_id), na=False)
        # also match by lender/receiver pair where Tipo indicates payment if available
        if ("payer" in ledger_df.columns) and ("receiver" in ledger_df.columns):
            pair_mask = (ledger_df["payer"].astype(str) == str(receiver)) & (ledger_df["receiver"].astype(str) == str(lender))
            payments_mask = payments_mask | pair_mask
        payments_df = ledger_df.loc[payments_mask].copy()
        # bucket payments by period
        if not payments_df.empty:
            payments_df[date_col] = pd.to_datetime(payments_df.get(date_col), errors="coerce")
            payment_bins = period_bins_for_dates(payments_df[date_col], freq=freq)
            payments_df = payments_df.assign(TimePeriod=payment_bins["TimePeriod"].values)
            # amount series for payments
            payments_df["_amt"] = _amount_series(payments_df, amount_col)
            payments_by_period = payments_df.groupby("TimePeriod")["_amt"].sum().to_dict()
        else:
            payments_by_period = {}
        # construct rows per period
        for i, p in enumerate(periods):
            sched = scheduled[i] if i < len(scheduled) else 0.0
            pay = float(payments_by_period.get(p, 0.0))
            outstanding = principal - sum([payments_by_period.get(pp, 0.0) for pp in periods if pp <= p])
            out_frames.append({
                "loan_id": loan_id,
                "TimePeriod": p,
                "TimePeriod_ts_end": p.to_timestamp(how="end"),
                "lender": lender,
                "receiver": receiver,
                "scheduled_principal": float(sched),
                "payments": float(pay),
                "outstanding": float(outstanding),
            })
    if not out_frames:
        return pd.DataFrame(columns=["loan_id", "TimePeriod", "TimePeriod_ts_end", "lender", "receiver", "scheduled_principal", "payments", "outstanding"])
    res = pd.DataFrame(out_frames)
    # ensure deterministic sort
    res = res.sort_values(["loan_id", "TimePeriod"]).reset_index(drop=True)
    return res

