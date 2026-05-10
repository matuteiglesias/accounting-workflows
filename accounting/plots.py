# src/accounting/plots.py
from __future__ import annotations
from typing import Optional, Iterable
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# inside src/accounting/plots.py (replace old plot_renta_series)
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Iterable

def plot_renta_series(pivot_df: pd.DataFrame, parties: Optional[Iterable[str]] = None,
                      show_rolling: bool = True, rolling_window: int = 4) -> plt.Figure:
    """
    Robust renta plot:
     - stacked area for columns that are purely non-negative (or purely non-positive),
     - line plot for columns that contain both positive and negative values (mixed-sign).
    pivot_df: Date-indexed wide DataFrame (columns = parties)
    """
    if pivot_df is None or pivot_df.empty:
        raise ValueError("pivot_df is empty")

    # ensure Date index is datetime-like
    df = pivot_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            pass

    # select parties subset if requested
    if parties is not None:
        cols = [c for c in parties if c in df.columns]
        if not cols:
            raise ValueError("no matching parties found in pivot_df")
        data = df[cols].fillna(0.0)
    else:
        data = df.fillna(0.0)

    # classify columns by sign behavior
    purely_pos = [c for c in data.columns if (data[c] >= 0).all()]
    purely_neg = [c for c in data.columns if (data[c] <= 0).all()]
    mixed = [c for c in data.columns if c not in purely_pos + purely_neg]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # stacked positives (above axis)
    if purely_pos:
        data[purely_pos].plot.area(ax=ax, stacked=True)

    # stacked negatives (below axis) — pandas can stack columns that are all non-positive
    if purely_neg:
        # plot negatives so they appear below zero (they are <= 0 already)
        data[purely_neg].plot.area(ax=ax, stacked=True)

    # mixed-sign columns -> plot as lines (clearest)
    if mixed:
        data[mixed].plot(ax=ax, linewidth=2)

    # rolling overlay (lines)
    if show_rolling:
        try:
            data.rolling(window=rolling_window, min_periods=1).mean().plot(ax=ax, linewidth=1.25, linestyle="--")
        except Exception:
            # ignore rolling errors for weird indexes
            pass

    ax.set_title("Renta time-series")
    ax.set_ylabel("Amount (currency units)")
    ax.set_xlabel("")  # keep x-label minimal
    fig.tight_layout()
    return fig

def plot_fondos_heatmap(fondos_wide_df: pd.DataFrame) -> plt.Figure:
    """
    fondos_wide_df: view with 'period_label', optional '__Date_parsed', and numeric columns
    We'll create a heatmap of top N columns across periods using matplotlib.imshow (no seaborn).
    """
    if fondos_wide_df is None or fondos_wide_df.empty:
        raise ValueError("fondos empty")
    # drop meta columns
    df = fondos_wide_df.copy()
    meta_cols = [c for c in ("period_label", "__Date_parsed") if c in df.columns]
    numeric = df.drop(columns=meta_cols).select_dtypes(include="number")
    # reduce to top 40 columns by total
    totals = numeric.sum().sort_values(ascending=False)
    top_cols = totals.index[:40]
    mat = numeric[top_cols].fillna(0.0).values
    fig, ax = plt.subplots(figsize=(12, max(3, mat.shape[0]*0.2)))
    im = ax.imshow(mat, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(mat.shape[0]))
    labels = (df.get("__Date_parsed") if "__Date_parsed" in df.columns else df["period_label"]).astype(str).tolist()
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=90)
    ax.set_title("Fondos heatmap (top flows)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def plot_party_balance(balance_df: pd.DataFrame, party: str) -> plt.Figure:
    """
    balance_df: columns: Date, party, balance (long format). Returns line plot for the chosen party.
    """
    df = balance_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    sub = df[df["party"] == party].sort_values("Date")
    if sub.empty:
        raise ValueError(f"no data for party {party}")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(sub["Date"], sub["balance"])
    ax.set_title(f"Cumulative balance - {party}")
    ax.axhline(0, color='k', linewidth=0.5)
    fig.tight_layout()
    return fig

# convenience save helper
def save_fig(fig: plt.Figure, path: Path, fmt: str = "png"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


# src/accounting/plots_cli.py
from pathlib import Path
import argparse
import pandas as pd
from accounting.plots import plot_renta_series, save_fig

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--views-dir", default="out/views")
    p.add_argument("--out-dir", default="out/figs")
    p.add_argument("--party", default=None)
    args = p.parse_args(argv)
    vd = Path(args.views_dir)
    pivot = pd.read_csv(vd / "renta_pivot.csv", parse_dates=["Date"], index_col="Date")
    fig = plot_renta_series(pivot, parties=[args.party] if args.party else None)
    save_fig(fig, Path(args.out_dir) / "renta_series.png")
    
if __name__ == "__main__":
    pass
    # main()
