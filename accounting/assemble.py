# src/accounting/assemble.py
from pathlib import Path
import pandas as pd
from typing import List, Dict

def load_series_map(out_dir: Path) -> Dict[str, pd.DataFrame]:
    """Lee todos los CSV de series en out_dir en un dict name->df"""
    out_dir = Path(out_dir)
    series_files = list(out_dir.glob("*.csv"))
    series_map = {}
    for f in series_files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        series_map[f.name] = df
    return series_map

def extract_columns_by_prefix(df: pd.DataFrame, prefix_list: List[str]) -> List[str]:
    cols = []
    for p in prefix_list:
        cols += [c for c in df.columns if c.startswith(p)]
    # unique preserve order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def assemble_all_tables(out_dir: str, adjusted_debt_path: str, weekly_index_start: str = "2023-01-01", weekly_index_end: str = None):
    out_dir = Path(out_dir)
    series_map = load_series_map(out_dir)
    # If you have some specific mapping of names -> keys (like 'fondos_report.csv' vs 'fondos_report')
    # normalize keys by removing .csv:
    series_map = {k.replace(".csv",""): v for k,v in series_map.items()}

    # ---- reconstruct series_results as in legacy ----
    # If some series were exported as single-column DataFrames, keep them as-is.
    # table_specs: if you keep this mapping in code, replicate it here (copy from legacy)
    table_specs = {
        "Main Financial Table": [
            ('fondos_report', [
                'Credit FB_Cobros', 'Debit FB_Transfer', 'Credit PM_Cobros',
                'Debit PM_Dividendos', 'Debit PM_Pagos', 'Debit PM_Repago',
                'Neto_Fondos_FB', 'Neto_Fondos_PM', 'Fondos_FB', 'Fondos_PM'
            ]),
            ('renta_PM', None),
            ('renta_FB', None),
            ('neto_de_costos', None),
            ('gastos_FB', None),
            ('net_profit', None)
        ],
        # add other tables as in legacy...
    }

    tables = {}
    for table_name, components in table_specs.items():
        frames = []
        for series_key, subcols in components:
            df = series_map.get(series_key)
            if df is None:
                # missing series: insert empty DF with index from one of the others (best effort)
                continue
            if subcols is not None and isinstance(df, pd.DataFrame):
                # keep only requested cols that exist
                cols = [c for c in subcols if c in df.columns]
                frames.append(df[cols])
            else:
                frames.append(df)
        if frames:
            tables[table_name] = pd.concat(frames, axis=1).fillna(0)
        else:
            tables[table_name] = pd.DataFrame()

    # all_tables combined
    all_tables = pd.concat([t for t in tables.values() if not t.empty], axis=1).fillna(0)

    # weekly index
    if weekly_index_end is None:
        weekly_index_end = pd.Timestamp.today().strftime("%Y-%m-%d")
    weekly_index = pd.date_range(start=weekly_index_start, end=weekly_index_end, freq="W-MON")

    # categories
    series_categories = [
        {"name": "cumulative", "resample": "last", "fill": "ffill"},
        {"name": "net", "resample": "sum", "fill": "zero"}
    ]

    # prefix list — copy from legacy or configure
    prefix_list = [
        'Fondos', 'Outstanding', 'Saldo', 'Credit', 'Debit',
        'Neto', 'Renta', 'Impuestos', 'Servicio', 'Legal',
        'Pagos', 'Contribuciones', 'Repago', 'Net', 'Gastos'
    ]

    # build column metadata
    column_metadata = {}
    for cat in series_categories:
        # for simplicity, use same prefix_list for both categories, but you may refine this mapping
        column_metadata[cat["name"]] = extract_columns_by_prefix(all_tables, prefix_list)

    all_resampled = []
    for cat in series_categories:
        cols = column_metadata[cat["name"]]
        if not cols:
            continue
        if cat["resample"] == "last":
            df_resampled = all_tables[cols].resample("W-MON").last().reindex(weekly_index)
            if cat["fill"] == "ffill":
                df_resampled = df_resampled.fillna(method='ffill', limit=2)
        elif cat["resample"] == "sum":
            df_resampled = all_tables[cols].resample("W-MON").sum().reindex(weekly_index).fillna(0)
        all_resampled.append(df_resampled)

    # load adjusted debt and reindex
    adjusted_debt_df = pd.read_csv(adjusted_debt_path, index_col=0, parse_dates=True)
    adjusted_debt_df = adjusted_debt_df.resample("W-MON").last().reindex(weekly_index).fillna(method='ffill', limit=2)

    all_tables_final = pd.concat(all_resampled + [adjusted_debt_df], axis=1).fillna(0)

    # write the final table
    final_path = out_dir / "all_tables_final.csv"
    all_tables_final.to_csv(final_path)
    return final_path, all_tables_final
