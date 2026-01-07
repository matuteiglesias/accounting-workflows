


# # 🔎 Automated Column Metadata Extraction
# prefix_list = [
#     'Fondos', 'Outstanding', 'Saldo', 'Credit', 'Debit',
#     'Neto', 'Renta', 'Impuestos', 'Servicio', 'Legal',
#     'Pagos', 'Contribuciones', 'Repago', 'Net', 'Gastos'
# ]

# # 🔎 Function to Extract Columns by Prefix
# def extract_columns_by_prefix(all_tables, prefix_list):
#     """
#     Extracts all columns that start with any prefix in prefix_list.
#     """
#     return [
#         col for col in all_tables.columns
#         if any(col.startswith(prefix) for prefix in prefix_list)
#     ]



# # 🔎 Conditional Export to Google Sheets
# def export_to_gsheets_if_needed(all_tables, freq, spreadsheet, sheet_name="Balance Input"):
#     if freq.upper() == "YEAR":
#         data_to_upload = (
#             all_tables
#             .round()
#             .fillna(0)
#             .reset_index()
#             .values
#             .tolist()
#         )
#         sheet = spreadsheet.worksheet(sheet_name)
#         sheet.clear()
#         sheet.update("A1", data_to_upload)
#         print(f"✅ Yearly Financial Data successfully uploaded to '{sheet_name}'.")

# # Example Usage:
# # export_to_gsheets_if_needed(all_tables_final, FREQ, spreadsheet)




# src/accounting/export.py
from __future__ import annotations
from typing import Tuple, Optional
from pathlib import Path
import math
import pandas as pd
import gspread
from gspread.utils import rowcol_to_a1, a1_to_rowcol
from gspread.exceptions import WorksheetNotFound
from datetime import datetime

# re-use your helper that builds an authorized gspread client
# # (you already defined get_google_sheets_client in src.accounting.utils)
# try:
#     from accounting.utils import get_google_sheets_client
# except Exception:
#     # fallback: if utils not available, expect caller to pass an authorized client
#     get_google_sheets_client = None

import sys

sys.path.append('./../')
sys.path.append('./../../')
sys.path.append('./../../../')

from accounting.utils import get_google_sheets_client

def _val_to_cell(v):
    """Coerce dataframe value to a sheet-friendly scalar (empty string for NaN)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        # preserve isoformat so sheet can parse as date if USER_ENTERED
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    # booleans/ints/floats are fine
    return v


import ast

def _try_parse_tuple_string(v):
    # If v is a string that looks like a tuple "('a','b')" try to parse it.
    if not isinstance(v, str):
        return v
    v = v.strip()
    if not v.startswith("(") or not v.endswith(")"):
        return v
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, tuple):
            return parsed
    except Exception:
        pass
    return v


def df_to_sheet(
    df: pd.DataFrame,
    sheet_url: str,
    service_account_file: Optional[str] = None,
    client: Optional[gspread.Client] = None,
    sheet_name: str = "RENTALS",
    start_cell: str = "A1",
    include_index: bool = False,
    value_input_option: str = "USER_ENTERED",
    resize_sheet: bool = True,
    merge_header: Optional[bool] = True,
) -> Tuple[gspread.models.Worksheet, str]:
    """
    Write a DataFrame to a Google Sheets worksheet WITHOUT wiping formatting.
    Key points:
      - Uses worksheet.update(range, values, value_input_option) which replaces values
        but does not clear cell formatting.
      - Does NOT call worksheet.clear() (that would wipe formatting).
      - Resizes worksheet if needed (resize shouldn't wipe formatting).
      - Uses USER_ENTERED so Google Sheets will parse numbers/dates like manual entry.

    Args:
      df: pandas DataFrame to write
      sheet_url: full URL of target spreadsheet
      service_account_file: path to service account JSON file (optional if client given)
      client: authorized gspread.Client (optional). If not provided, service_account_file is required.
      sheet_name: name of the tab to write into (will be created if missing)
      start_cell: top-left A1 cell to start writing (default "A1")
      include_index: whether to include df.index as the first column
      value_input_option: "USER_ENTERED" or "RAW"
      resize_sheet: whether to resize the sheet to fit the data (default True)

    Returns:
      (worksheet, updated_range_str) where updated_range_str is e.g. "A1:E73"

    Caveats:
      - If a header row has custom formatting, updating cell values will keep that formatting.
      - If you create a new sheet (tab), you'll need to set header formatting manually once.
      - Merged cells and complex formatting spanning the updated range might behave unexpectedly.
    """
    if client is None:
        if get_google_sheets_client is None and not service_account_file:
            raise RuntimeError("No gspread client provided and no service_account_file available")
        client = get_google_sheets_client(service_account_file)

    # open spreadsheet
    sh = client.open_by_url(sheet_url)

    # get or create worksheet
    try:
        ws = sh.worksheet(sheet_name)
    except WorksheetNotFound:
        # create with a few rows/cols; resizing will happen below.
        nrows = max(100, len(df) + 5)
        ncols = max(10, (len(df.columns) + (1 if include_index else 0)))
        ws = sh.add_worksheet(title=sheet_name, rows=str(nrows), cols=str(ncols))

    # prepare values matrix (list of lists) starting with header
    values = []

    # optionally try to convert stringified tuple headers to real tuples (if all parse ok)
    cols = list(df.columns)
    maybe_parsed = [_try_parse_tuple_string(c) for c in cols]
    if all(isinstance(x, tuple) for x in maybe_parsed):
        cols = maybe_parsed
    # if DataFrame has MultiIndex columns use that; otherwise if columns are tuple objects, treat as multi-level
    if isinstance(df.columns, pd.MultiIndex):
        nlevels = df.columns.nlevels
        # for each level produce a header row
        for lvl in range(nlevels):
            row = []
            if include_index:
                # index placeholder for this header row (if index is not multi, put name in first header row)
                if lvl == 0:
                    idx_name = df.index.name if df.index.name else ""
                    row.append(str(idx_name))
                else:
                    row.append("")
            level_labels = [str(x) for x in df.columns.get_level_values(lvl)]
            row.extend(level_labels)
            values.append(row)
    elif all(isinstance(c, tuple) for c in cols):
        # columns are tuples but not a pandas MultiIndex; handle variable tuple lengths
        max_lv = max(len(c) for c in cols)
        for lvl in range(max_lv):
            row = []
            if include_index:
                if lvl == 0:
                    row.append(df.index.name if df.index.name else "")
                else:
                    row.append("")
            for c in cols:
                # if tuple shorter than max_lv pad with empty string
                row.append(str(c[lvl]) if lvl < len(c) else "")
            values.append(row)
        # replace df.columns with plain strings for body iteration (we will still access by original names)
    else:
        # single header row (default)
        header = []
        if include_index:
            header.append(df.index.name if df.index.name else "index")
        header.extend([str(h) for h in df.columns])
        values.append(header)

    # body rows
    for idx, row in df.iterrows():
        row_vals = []
        if include_index:
            row_vals.append(_val_to_cell(idx))
        for c in df.columns:
            row_vals.append(_val_to_cell(row[c]))
        values.append(row_vals)

    # compute A1 range
    start_row, start_col = a1_to_rowcol(start_cell)
    n_rows = len(values)
    n_cols = max(len(r) for r in values) if values else 0
    end_row = start_row + n_rows - 1
    end_col = start_col + n_cols - 1
    end_cell = rowcol_to_a1(end_row, end_col)
    range_a1 = f"{start_cell}:{end_cell}"

    # resize sheet if requested (this does not clear formatting)
    try:
        if resize_sheet:
            # only resize if needed (avoid unnecessary API calls)
            cur_rows = int(ws.row_count)
            cur_cols = int(ws.col_count)
            need_rows = end_row > cur_rows
            need_cols = end_col > cur_cols
            if need_rows or need_cols:
                new_rows = max(cur_rows, end_row)
                new_cols = max(cur_cols, end_col)
                ws.resize(rows=new_rows, cols=new_cols)
    except Exception:
        # non-fatal: continue to update values even if resize fails
        pass





    # OPTIONAL: merge contiguous equal header cells (opt-in)
    if merge_header:
        try:
            # do merges per header row for contiguous equal cells (skip index column cell)
            header_row_count = len(values) - len(df) if len(values) > len(df) else 1
            # header rows occupy start_row .. start_row+header_row_count-1
            for lvl in range(header_row_count):
                rownum = start_row + lvl
                col_idx = start_col + (1 if include_index else 0)
                run_val = None
                run_start = col_idx
                for j in range(col_idx, end_col + 1):
                    # A1 address for this cell
                    a1 = rowcol_to_a1(rownum, j)
                    # index into values matrix: lvl row, j-start_col col
                    v = values[lvl][j - start_col] if (j - start_col) < len(values[lvl]) else ""
                    if v == run_val:
                        # continue run
                        pass
                    else:
                        # close previous run
                        if run_val not in (None, "") and j - 1 > run_start:
                            try:
                                ws.merge_cells(start_row=rownum, start_col=run_start, end_row=rownum, end_col=j - 1)
                            except Exception:
                                pass
                        run_val = v
                        run_start = j
                # close final run
                if run_val not in (None, "") and end_col > run_start:
                    try:
                        ws.merge_cells(start_row=rownum, start_col=run_start, end_row=rownum, end_col=end_col)
                    except Exception:
                        pass
        except Exception:
            # merging is best-effort and non-fatal
            pass



    # update values (this replaces values but preserves formatting)
    ws.update(range_a1, values, value_input_option=value_input_option)

    return ws, range_a1
