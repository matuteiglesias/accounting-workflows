#!/usr/bin/env python3
"""Static audit of accounting report notebooks and professional-pack outputs.

This script intentionally does not execute notebooks. It parses notebook JSON,
cell source text, and already-materialized CSV artifacts to build an inventory,
lineage map, dashboard gap analysis, and a markdown implementation plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

INVENTORY_COLUMNS = [
    "notebook_path","cell_index","cell_type","output_name","output_path","output_kind","output_exists",
    "output_shape_rows","output_shape_cols","output_columns","producer_function","source_dataframe_guess",
    "source_artifact_guess","uses_display_table","uses_export_table","uses_to_csv","period_grain","is_monthly",
    "is_annual","is_wide","is_long","has_currency","has_box","has_metric_id","has_line_id",
    "has_dimension_columns","drilldown_supported_now","linked_digest_included_now","notes",
]
LINEAGE_COLUMNS = [
    "output_name","notebook_path","cell_index","producer_function","source_dataframe","upstream_csv_or_artifact",
    "upstream_columns_used","groupby_keys","pivot_index","pivot_columns","value_columns","aggregation_function",
    "filter_logic_summary","calculation_rule_summary","stock_or_flow","currency_policy","box_policy","known_caveats",
]
GAP_COLUMNS = [
    "dashboard_need","recommended_output_csv","recommended_notebook","recommended_cell_anchor","source_dataframe_or_artifact",
    "source_columns_needed","available_now","missing_columns","computation_type","grain","dimensions","value_columns",
    "suggested_metric_id","suggested_line_id","drilldown_source","drilldown_feasibility","implementation_complexity",
    "recommended_next_action",
]

DISPLAY_RE = re.compile(r"display_table\s*\((?P<df>[^,\n\)]+).*?[\"'](?P<name>[^\"']+\.(?:csv|html|md))[\"']", re.S)
EXPORT_RE = re.compile(r"export_table\s*\((?P<df>[^,\n\)]+).*?[\"'](?P<name>[^\"']+\.csv)[\"']", re.S)
TOCSV_RE = re.compile(r"(?P<df>[A-Za-z_][A-Za-z0-9_]*)\.to_csv\s*\(\s*(?P<path>[^\)]+)", re.S)
READ_RE = re.compile(r"pd\.read_csv\s*\((?P<path>[^\)]+)\)")
MONTH_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")
YEAR_RE = re.compile(r"^20\d{2}$")

KNOWN_OUTPUTS: dict[str, dict[str, str]] = {
    "cash_annual_box_flow_bridge_wide.csv": dict(source_dataframe="annual_bridge_wide", upstream="monthly_flow_semantic_split.csv", groupby="year, Currency_norm, Box_norm", pivot="metric rows x annual year columns", values="amount_in, amount_out, net_amount", agg="sum", rule="Annual flow bridge by Box/Currency; sums semantic flow rows into operating/funding/debt/unknown/treasury components.", stock="flow"),
    "cash_annual_box_flow_bridge_long.csv": dict(source_dataframe="annual_bridge", upstream="monthly_flow_semantic_split.csv", groupby="year, Currency_norm, Box_norm, metric", pivot="long rows", values="value", agg="sum", rule="Long companion to annual box flow bridge.", stock="flow"),
    "income_operating_statement.csv": dict(source_dataframe="income_statement", upstream="annual_balance_dashboard_metrics.csv", groupby="period, Currency, statement_line", pivot="statement lines x annual columns", values="value", agg="formula/metric selection", rule="Annual operating statement from governed annual metric IDs; excludes funding, debt and distributions from operating result.", stock="formula"),
    "overview_balance_dashboard.csv": dict(source_dataframe="overview_table", upstream="annual_balance_dashboard_metrics.csv", groupby="period, Currency, metric_id", pivot="dashboard rows x annual columns", values="value", agg="formula/metric selection", rule="Executive annual dashboard assembled from metric specs against annual dashboard metrics.", stock="formula"),
    "monthly_tables_debt_position_matrix.csv": dict(source_dataframe="debt_pos", upstream="monthly_debt_position.csv", groupby="Currency, pair, measure", pivot="period months as columns", values="open_principal, open_interest, open_total", agg="latest/snapshot matrix", rule="Monthly debt stock by debtor/creditor pair; do not sum monthly positions.", stock="stock"),
    "monthly_tables_cash_close_matrix.csv": dict(source_dataframe="cash", upstream="monthly_cash_close.csv", groupby="Currency, Box, metric", pivot="period months as columns", values="value", agg="snapshot matrix", rule="Monthly cash close stock by Box/Currency/metric; frontend-safe status matters.", stock="stock"),
    "monthly_tables_diagnostic_box_level_matrix.csv": dict(source_dataframe="box_level", upstream="box_balance_time_long.freq=M.csv", groupby="Currency, Box, metric", pivot="period months as columns", values="value", agg="snapshot/delta matrix", rule="Diagnostic box-level monthly net/delta from box balance motor; stock-derived diagnostic, not validated cash.", stock="diagnostic"),
    "monthly_tables_debt_activity_matrix.csv": dict(source_dataframe="debt_act", upstream="monthly_debt_activity.csv", groupby="Currency, pair, measure", pivot="period months as columns", values="new_principal, interest_accrued, repayments, adjustments, net_change", agg="sum by month", rule="Monthly debt activity flows by pair.", stock="flow"),
}

def rel(path: Path) -> str:
    try: return str(path.relative_to(Path.cwd()))
    except Exception: return str(path)

def read_nb(path: Path) -> list[dict[str, Any]]:
    try: return json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    except Exception: return []

def csv_info(path: Path) -> tuple[bool, int|str, int|str, str, set[str]]:
    if not path.exists(): return False, "", "", "", set()
    try:
        df = pd.read_csv(path, nrows=2000)
        return True, len(df), len(df.columns), ";".join(map(str, df.columns)), set(map(str, df.columns))
    except Exception as exc:
        return True, "", "", f"READ_ERROR:{type(exc).__name__}:{exc}", set()

def kind(name: str, pack: Path, path: Path) -> str:
    s = str(path)
    if "/tables/" in s or path.parent == pack / "tables": return "professional_table_csv"
    if "/figures/" in s or path.parent == pack / "figures": return "figure"
    if "/digest/" in s or path.parent == pack / "digest": return "digest"
    if "cash_position_eda" in s: return "eda_csv"
    return "unknown_csv"

def resolve_output(name: str, pack: Path) -> Path:
    n = Path(name).name
    for sub in ["tables", "figures", "digest", "drilldown", "qa", "html", "markdown"]:
        p = pack / sub / n
        if p.exists() or sub == "tables":
            return p
    return pack / "tables" / n

def source_guess(cell_src: str, df: str, output: str) -> tuple[str, str]:
    known = KNOWN_OUTPUTS.get(Path(output).name, {})
    srcdf = known.get("source_dataframe") or df.strip()
    upstream = known.get("upstream", "")
    if not upstream:
        reads = [m.group("path").replace("\n", " ")[:120] for m in READ_RE.finditer(cell_src)]
        upstream = "; ".join(reads)
    return srcdf, upstream

def infer_lineage(output: str, nb: str, idx: int, prod: str, df: str, cell_src: str) -> dict[str, Any]:
    name = Path(output).name
    k = KNOWN_OUTPUTS.get(name, {})
    lower = (name + " " + cell_src).lower()
    if k: stock = k["stock"]
    elif "debt_position" in lower or "cash_close" in lower: stock = "stock"
    elif "diagnostic" in lower: stock = "diagnostic"
    elif "income" in lower or "overview" in lower: stock = "formula"
    elif "flow" in lower or "funding" in lower or "activity" in lower: stock = "flow"
    else: stock = "unknown"
    group = k.get("groupby", "")
    gm = re.search(r"groupby\s*\(\s*\[([^\]]+)\]", cell_src)
    if gm and not group: group = gm.group(1).replace("\n", " ")
    return dict(zip(LINEAGE_COLUMNS, [
        name, nb, idx, prod, k.get("source_dataframe", df), k.get("upstream", ""), "inferred from source cell and current CSV columns",
        group, k.get("pivot", "period columns if matrix/wide; long rows otherwise"), "period/month/year columns", k.get("values", "value/net_amount/amount columns inferred"),
        k.get("agg", "inferred from notebook cell"), "see notebook cell filters; static audit only", k.get("rule", "Static parser found output; inspect cell for exact filters before implementation."),
        stock, "Currency kept as dimension; do not sum ARS and USD", "Box kept as dimension where present; do not infer physical cash from governance box", "Static audit; notebooks were not executed.",
    ]))

def discover(notebooks_dir: Path, pack: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv, lin = [], []
    for nb in sorted(notebooks_dir.glob("*.ipynb")):
        nb_rel = rel(nb)
        for i, cell in enumerate(read_nb(nb)):
            src = "".join(cell.get("source", []))
            matches = []
            for m in DISPLAY_RE.finditer(src): matches.append((m.group("df").strip(), m.group("name"), "display_table"))
            for m in EXPORT_RE.finditer(src): matches.append((m.group("df").strip(), m.group("name"), "export_table"))
            for m in TOCSV_RE.finditer(src):
                raw = m.group("path")
                names = re.findall(r"[\"']([^\"']+\.csv)[\"']", raw)
                matches.append((m.group("df"), names[0] if names else raw.strip()[:80], "to_csv"))
            for dfname, outname, prod in matches:
                outpath = resolve_output(outname, pack)
                exists, rows, cols, columns, colset = csv_info(outpath)
                source_df, upstream = source_guess(src, dfname, outname)
                name = Path(outname).name
                period_cols = [c for c in colset if MONTH_RE.match(c) or YEAR_RE.match(c)]
                inv.append(dict(zip(INVENTORY_COLUMNS, [
                    nb_rel, i, cell.get("cell_type", ""), name, rel(outpath), kind(name, pack, outpath), exists, rows, cols, columns,
                    prod, source_df, upstream, "display_table" in src, "export_table" in src, ".to_csv" in src,
                    "monthly" if any(MONTH_RE.match(c) for c in colset) or "monthly" in name else ("annual" if any(YEAR_RE.match(c) for c in colset) or "annual" in name or "income" in name or "overview" in name else "unknown"),
                    bool(any(MONTH_RE.match(c) for c in colset) or "monthly" in name), bool(any(YEAR_RE.match(c) for c in colset) or "annual" in name),
                    bool(len(period_cols) >= 2), bool({"period","value"} <= colset), "Currency" in colset or "currency" in colset,
                    "Box" in colset or "box" in colset, "metric_id" in colset, "line_id" in colset or "line" in colset or "statement_line" in colset,
                    bool(colset & {"dimension_name","dimension_value","funding_actor","funding_channel","pair","debtor","creditor"}),
                    name in {"overview_balance_dashboard.csv","income_operating_statement.csv","cash_annual_box_flow_bridge_wide.csv","monthly_tables_debt_position_matrix.csv","monthly_tables_cash_close_matrix.csv","monthly_tables_diagnostic_box_level_matrix.csv"},
                    name in {"overview_balance_dashboard.csv","income_operating_statement.csv","cash_annual_box_flow_bridge_wide.csv"} or name.startswith("monthly_tables_"),
                    "Static parse; verify exact accounting treatment before metric implementation.",
                ])))
                lin.append(infer_lineage(outname, nb_rel, i, prod, source_df, src))
    return pd.DataFrame(inv, columns=INVENTORY_COLUMNS), pd.DataFrame(lin, columns=LINEAGE_COLUMNS)

def gap_rows(inv: pd.DataFrame) -> pd.DataFrame:
    def has_output(n): return bool((inv["output_name"] == n).any())
    base = [
        ("Fondos Familiares / FB cash close","annual_cash_close_by_box_wide.csv; annual_cash_close_by_box_long.csv","07_monthly_dynamics_tables.ipynb","after cell producing monthly_tables_cash_close_matrix.csv","monthly_cash_close.csv or box_balance_time_long.freq=M.csv","period/year, Box, Currency, metric/value or cum_net","yes" if has_output("monthly_tables_cash_close_matrix.csv") else "partial","annual output missing","stock/latest annual close","year","Box;Currency","value","CASH.CLOSE.BY_BOX","CASH.CLOSE.BY_BOX.{Box}","monthly_cash_close.csv + cash_position_eda contract","high; stock lineage exists but use latest month, not sum","low","Add annual long+wide cells after monthly cash close matrix; wire drilldown to monthly cash close/box motor."),
        ("Fondos Operativos / PM cash close","annual_cash_close_by_box_wide.csv; annual_cash_close_by_box_long.csv","07_monthly_dynamics_tables.ipynb","after cell producing monthly_tables_cash_close_matrix.csv","monthly_cash_close.csv or box_balance_time_long.freq=M.csv","period/year, Box=Property Management, Currency, value","yes" if has_output("monthly_tables_cash_close_matrix.csv") else "partial","annual output missing","stock/latest annual close","year","Box;Currency","value","CASH.CLOSE.BY_BOX","CASH.CLOSE.BY_BOX.Property Management","monthly_cash_close.csv + cash_position_eda contract","high","low","Same table as FB cash close; include PM row and metadata."),
        ("Cash close by Box/Currency/year","annual_cash_close_by_box_wide.csv; annual_cash_close_by_box_long.csv","07_monthly_dynamics_tables.ipynb","after monthly_tables_cash_close_matrix.csv","monthly_cash_close.csv","period, Box, Currency, metric, value","yes","annual output missing","stock/latest annual close","year","Box;Currency","value","CASH.CLOSE.BY_BOX","CASH.CLOSE.BY_BOX","monthly_cash_close.csv","high","low","Do not sum monthly closes; pick last available period per year/Box/Currency."),
        ("Contributions by actor","annual_funding_by_actor_channel_wide.csv; annual_funding_by_actor_channel_long.csv","07_monthly_dynamics_tables.ipynb","after flow subbucket/funding-related cells","monthly_flow_semantic_split.csv + classification_audit.csv + funding_lineage_audit.csv","year, Currency, funding_actor, funding_channel, cash_effect, target_box, obligation_box, amount_in/net_amount","partial","annual actor/channel output missing","flow annual sum","year","Currency;funding_actor;funding_channel;cash_effect;target_box;obligation_box","amount_in;net_amount","FUND.CONTRIB.BY_FUNDING_ACTOR","FUND.CONTRIB.BY_FUNDING_ACTOR","monthly_flow_semantic_split.csv + classification_audit.csv","high if dimensions populated","medium","Add dedicated annual funding tables; keep direct obligation vs cash-to-box channels explicit."),
        ("Contributions tenant direct vs tenant to box","annual_funding_by_actor_channel_wide.csv; annual_funding_by_actor_channel_long.csv","07_monthly_dynamics_tables.ipynb","after flow subbucket/funding-related cells","monthly_flow_semantic_split.csv","funding_channel,cash_effect,obligation_box,target_box,amount_in/net_amount","partial","annual output missing","flow annual sum","year","funding_channel;cash_effect;target_box;obligation_box","amount_in;net_amount","FUND.CONTRIB.BY_CHANNEL","FUND.CONTRIB.BY_CHANNEL","classification_audit.csv rows","high","medium","Separate tenant_direct_tax_payment/service_payment from tenant_to_box/cash_to_box."),
        ("Funding by actor","annual_funding_by_actor_channel_wide.csv; annual_funding_by_actor_channel_long.csv","07_monthly_dynamics_tables.ipynb","after monthly flow bucket/subbucket matrices","monthly_flow_semantic_split.csv","funding_actor, Currency, amount_in","partial","annual output missing","flow annual sum","year","Currency;funding_actor","amount_in","FUND.CONTRIB.BY_FUNDING_ACTOR","FUND.CONTRIB.BY_FUNDING_ACTOR","monthly_flow_semantic_split.csv","high","medium","Implement with channel/cash_effect in same contract to avoid renderer-only semantics."),
        ("Funding by channel","annual_funding_by_actor_channel_wide.csv; annual_funding_by_actor_channel_long.csv","07_monthly_dynamics_tables.ipynb","after monthly flow bucket/subbucket matrices","monthly_flow_semantic_split.csv","funding_channel, Currency, amount_in","partial","annual output missing","flow annual sum","year","Currency;funding_channel","amount_in","FUND.CONTRIB.BY_CHANNEL","FUND.CONTRIB.BY_CHANNEL","monthly_flow_semantic_split.csv","high","medium","Use stable funding_channel dimension."),
        ("Funding by cash_effect","annual_funding_by_actor_channel_wide.csv; annual_funding_by_actor_channel_long.csv","07_monthly_dynamics_tables.ipynb","after monthly flow bucket/subbucket matrices","monthly_flow_semantic_split.csv","cash_effect, Currency, amount_in/net_amount","partial","annual output missing","flow annual sum","year","Currency;cash_effect","amount_in;net_amount","FUND.CONTRIB.BY_CASH_EFFECT","FUND.CONTRIB.BY_CASH_EFFECT","monthly_flow_semantic_split.csv","high","medium","Ensure direct obligation support is not presented as cash inflow."),
        ("Funding tenant direct obligation","annual_funding_by_actor_channel_wide.csv; annual_funding_by_actor_channel_long.csv","07_monthly_dynamics_tables.ipynb","after funding annual table cell","monthly_flow_semantic_split.csv","funding_channel in tenant_direct_*; obligation_box; value","partial","annual output missing","flow/support annual sum","year","Currency;funding_channel;obligation_box","amount_in;net_amount","FUND.CONTRIB.DIRECT_OBLIGATION","FUND.CONTRIB.DIRECT_OBLIGATION","classification_audit.csv","medium-high","medium","Use cash_effect metadata to flag non-cash direct obligation support."),
        ("Funding tenant-to-box","annual_funding_by_actor_channel_wide.csv; annual_funding_by_actor_channel_long.csv","07_monthly_dynamics_tables.ipynb","after funding annual table cell","monthly_flow_semantic_split.csv","funding_channel=tenant_to_box/cash_to_box; target_box; value","partial","annual output missing","flow annual sum","year","Currency;funding_channel;target_box","amount_in","FUND.CONTRIB.CASH_TO_BOX","FUND.CONTRIB.CASH_TO_BOX","monthly_flow_semantic_split.csv","high","medium","Keep separate from tenant direct obligation."),
        ("Debt stock by pair","annual_debt_stock_by_pair_wide.csv; annual_debt_stock_by_pair_long.csv","07_monthly_dynamics_tables.ipynb","after monthly_tables_debt_position_matrix.csv","monthly_debt_position.csv","period/year, Currency, debtor, creditor, pair, open_principal/open_interest/open_total","yes","annual output missing","stock/latest annual close","year","Currency;debtor;creditor;pair;component","open_principal;open_interest;open_total","DEBT.STOCK.BY_PAIR.OPEN_TOTAL","DEBT.STOCK.BY_PAIR.OPEN_TOTAL","monthly_debt_position.csv","high","low-medium","Pick latest monthly selected as_of_date per year/pair; do not sum positions."),
        ("Saldo Matías","annual_debt_stock_by_pair_wide.csv; annual_debt_stock_by_pair_long.csv","07_monthly_dynamics_tables.ipynb","after debt position matrix","monthly_debt_position.csv","creditor/debtor contains Matías/Matias; open_total","yes","annual Matías-specific output missing","stock/latest annual close","year","Currency;debtor;creditor;pair","open_total","DEBT.STOCK.BY_PAIR.OPEN_TOTAL","DEBT.STOCK.MATIAS","monthly_debt_position.csv","high","low-medium","Filter pair/creditor dimension in consuming dashboard, not renderer-only label inference."),
        ("Repago a Matías","annual_debt_activity_by_pair_wide.csv; annual_debt_activity_by_pair_long.csv","07_monthly_dynamics_tables.ipynb","after monthly_tables_debt_activity_matrix.csv","monthly_debt_activity.csv + classification_audit.csv","year, Currency, debtor, creditor, pair, repayments/activity_type","yes" if has_output("monthly_tables_debt_activity_matrix.csv") else "partial","annual output missing","flow annual sum","year","Currency;debtor;creditor;pair;activity_type","repayments","DEBT.ACTIVITY.REPAYMENT.BY_PAIR","DEBT.ACTIVITY.REPAYMENT.MATIAS","monthly_debt_activity.csv","high","medium","Add annual debt activity table and map Matías via explicit pair/creditor dimensions."),
        ("Debt settlements / repayments by pair","annual_debt_activity_by_pair_wide.csv; annual_debt_activity_by_pair_long.csv","07_monthly_dynamics_tables.ipynb","after debt activity matrix","monthly_debt_activity.csv","repayments, adjustments, net_change by pair","yes","annual output missing","flow annual sum","year","Currency;debtor;creditor;pair;activity_type","repayments;net_change","DEBT.ACTIVITY.REPAYMENT.BY_PAIR","DEBT.ACTIVITY.REPAYMENT.BY_PAIR","monthly_debt_activity.csv + debt repayment events if available","high","medium","Keep repayments/settlements as flows separate from stock."),
        ("Depósitos de garantía","annual_deposits_guarantees_wide.csv; annual_deposits_guarantees_long.csv","01_balance_dashboard_overview.ipynb or new focused notebook","near overview cash/debt rows after source evidence exists","search evidence: annual metric BS.SECURITY_DEPOSITS.HELD; source ledger/artifact pending","year, Currency, tenant/property/contract if source exists, value","partial","source evidence not yet sufficient for implementation","audit-only stock/liability pending contract","year","Currency;property/tenant if available","value","BS.SECURITY_DEPOSITS.HELD","BS.SECURITY_DEPOSITS.HELD","manual/external source pending unless source artifact added","low until source exists","high","Do not implement accounting treatment without source evidence; add source artifact/contract first."),
    ]
    return pd.DataFrame([dict(zip(GAP_COLUMNS, r)) for r in base], columns=GAP_COLUMNS)

def _md_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(x).replace('\n', ' ') for x in row) + ' |')
    return '\n'.join(out)

def markdown_counts(inv: pd.DataFrame) -> str:
    if inv.empty:
        return 'No outputs discovered.'
    counts = inv.groupby('notebook_path')['output_name'].count().sort_values(ascending=False).head(20)
    return _md_table([[idx, val] for idx, val in counts.items()], ['notebook_path', 'output_count'])

def markdown_gap(gap: pd.DataFrame) -> str:
    cols = ['dashboard_need','recommended_output_csv','recommended_notebook','implementation_complexity']
    return _md_table(gap[cols].astype(str).values.tolist(), cols)

def write_report(path: Path, inv: pd.DataFrame, lin: pd.DataFrame, gap: pd.DataFrame) -> None:
    def bullet_outputs(names):
        rows=[]
        for n in names:
            m=inv[inv.output_name.eq(n)]
            if m.empty: rows.append(f"- `{n}`: not found in parsed notebook outputs.")
            else:
                r=m.iloc[0]; l=lin[lin.output_name.eq(n)].iloc[0]
                rows.append(f"- `{n}`: `{r.notebook_path}` cell {r.cell_index}; source `{l.source_dataframe}` from `{l.upstream_csv_or_artifact}`; grouping `{l.groupby_keys}`; pivot `{l.pivot_index}`; aggregation `{l.aggregation_function}`; classification `{l.stock_or_flow}`.")
        return "\n".join(rows)
    key = ["cash_annual_box_flow_bridge_wide.csv","income_operating_statement.csv","overview_balance_dashboard.csv","monthly_tables_debt_position_matrix.csv","monthly_tables_cash_close_matrix.csv","monthly_tables_diagnostic_box_level_matrix.csv","monthly_tables_debt_activity_matrix.csv"]
    md = f"""# Accounting report notebook audit

Generated by `scripts/audit_accounting_report_notebooks.py` without executing notebooks.

## 1. Executive summary

- Parsed {inv['notebook_path'].nunique()} notebooks and discovered {len(inv)} output references.
- The safest insertion point for most requested annual professional tables is `07_monthly_dynamics_tables.ipynb`, immediately after the existing monthly source-specific matrix cells.
- Cash close and debt position are stock metrics and must use latest annual monthly close/snapshot, not sums. Funding and debt repayments are flows and should sum annually.
- Deposits/guarantees remain audit-only until a source artifact or contract evidence is explicit.

## 2. Notebook inventory

Top parsed output notebooks:

{markdown_counts(inv)}

## 3. Existing CSV outputs and their source logic

{bullet_outputs(key)}

Funding/contribution outputs currently appear mostly as monthly flow and bridge matrices, not as dedicated annual actor/channel tables. Debt position and activity outputs are produced by `07_monthly_dynamics_tables.ipynb` monthly matrices.

## 4. Current professional tables used by linked digest

The linked digest currently includes `overview_balance_dashboard.csv`, `income_operating_statement.csv`, `cash_annual_box_flow_bridge_wide.csv`, and `monthly_tables_*.csv` tables. Dedicated annual cash-close-by-box, funding-by-actor/channel, debt-stock-by-pair, and debt-activity-by-pair tables are not yet present as stable annual professional contracts.

## 5. Current drilldown-supported vs unsupported outputs

Drilldown-supported now: overview dashboard, income statement, annual cash flow bridge, monthly debt position, monthly cash close, and diagnostic box-level matrices. Unsupported or weakly supported: dedicated annual funding actor/channel, annual debt pair stock/activity, and deposits/guarantees because those output CSVs do not yet exist.

## 6. Missing dashboard needs

{markdown_gap(gap)}

## 7. Best insertion points for new annual tables

- `annual_cash_close_by_box_*`: `07_monthly_dynamics_tables.ipynb` after `monthly_tables_cash_close_matrix.csv`; source DataFrame is already loaded as `cash`; no new helper required beyond latest-period selector; output both wide and long; add to drilldown immediately; include in linked digest after QA.
- `annual_funding_by_actor_channel_*`: `07_monthly_dynamics_tables.ipynb` after monthly flow bucket/subbucket/funding cells; source DataFrame is already loaded as `flow`; may require small helper for dimensions; output both wide and long; add to drilldown immediately from semantic/classification rows; include in linked digest.
- `annual_debt_stock_by_pair_*`: `07_monthly_dynamics_tables.ipynb` after `monthly_tables_debt_position_matrix.csv`; source DataFrame `debt_pos` is already loaded; latest annual snapshot helper recommended; output both wide and long; add to drilldown immediately.
- `annual_debt_activity_by_pair_*`: `07_monthly_dynamics_tables.ipynb` after `monthly_tables_debt_activity_matrix.csv`; source DataFrame `debt_act` is already loaded; output both wide and long; add drilldowns to monthly debt activity/classification evidence.
- `annual_deposits_guarantees_*`: defer or add in `01_balance_dashboard_overview.ipynb` only after source evidence exists; should not be renderer-only.

## 8. Proposed new CSV contracts

All new long CSVs should include `metric_id`, `line_id`, `period`, `Currency`, `value`, `source_table`, `source_filter`, and `calculation_rule`, plus explicit dimensions (`Box`, `funding_actor`, `funding_channel`, `cash_effect`, `target_box`, `obligation_box`, `debtor`, `creditor`, `pair`, `component`, `activity_type`) as applicable. Wide CSVs should be presentation companions with year columns and the same stable row identifiers.

## 9. Risks and accounting caveats

- Do not sum stocks: cash close and debt position use latest annual monthly close/snapshot.
- Do not present tenant direct tax/service payments as PM/FB cash inflow; mark cash effect explicitly.
- Do not infer deposits/guarantees treatment without source evidence and contract semantics.
- Do not sum ARS and USD.
- Keep renderer mappings downstream from stable CSV metadata; avoid renderer-only accounting semantics.

## 10. Recommended implementation sequence

1. Patch 1 — annual cash close by box.
2. Patch 2 — annual funding by actor/channel/cash_effect.
3. Patch 3 — annual debt stock by pair.
4. Patch 4 — annual debt activity / repayment by pair.
5. Patch 5 — deposits/guarantees if source exists.
6. Patch 6 — wire new tables into drilldown and linked digest.
7. Patch 7 — tests and QA checks.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebooks-dir", type=Path, default=Path("accounting/notebooks/accounting_reports"))
    ap.add_argument("--pack", type=Path, default=Path("out/professional_pack/latest"))
    ap.add_argument("--run-root", type=Path, default=Path("out/run/accounting/latest"))
    ap.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = ap.parse_args()
    (args.pack / "drilldown").mkdir(parents=True, exist_ok=True)
    inv, lin = discover(args.notebooks_dir, args.pack)
    gap = gap_rows(inv)
    inv.to_csv(args.pack / "drilldown/notebook_report_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    lin.to_csv(args.pack / "drilldown/notebook_output_lineage.csv", index=False)
    gap.to_csv(args.pack / "drilldown/notebook_metric_gap_analysis.csv", index=False)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    report = args.docs_dir / f"accounting_report_notebook_audit_{date}.md"
    write_report(report, inv, lin, gap)
    print(f"inventory: {inv.shape} -> {args.pack / 'drilldown/notebook_report_inventory.csv'}")
    print(f"lineage: {lin.shape} -> {args.pack / 'drilldown/notebook_output_lineage.csv'}")
    print(f"gap: {gap.shape} -> {args.pack / 'drilldown/notebook_metric_gap_analysis.csv'}")
    print(f"report: {report}")

if __name__ == "__main__":
    main()
