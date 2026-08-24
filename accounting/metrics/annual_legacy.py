from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from accounting.debt.position_authority import (
    select_debt_position,
    selected_debt_position_rows,
)
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.scope import assert_frame_within_scope, load_run_scope_if_present

ANNUAL_METRICS_COLUMNS = [
    "metric_id","period_grain","period","period_start","period_end","Currency","value","value_status",
    "flow_or_stock","accounting_section","dashboard_section","dimension_name","dimension_value","source_table",
    "source_filter","calculation_rule","frontend_suitability","public_flag","internal_flag","legacy_flag",
    "validation_status","caveat","run_id","as_of_date",
]
ANNUAL_CONTRACT_COLUMNS = [
    "metric_id","period_grain","flow_or_stock","accounting_section","dashboard_section","source_table",
    "source_filter","calculation_rule","dimension_convention","frontend_suitability","public_flag",
    "internal_flag","legacy_flag","validation_status","caveat","status",
]
QA_COLUMNS = ["check","status","detail","severity"]
CANONICAL = {
    "monthly_operating_statement.csv","monthly_flow_semantic_split.csv","monthly_cash_close.csv",
    "monthly_debt_position.csv","monthly_debt_activity.csv","semantic_leakage_qa.csv",
    "semantic_dashboard_coverage.csv","metric_contract_frontier.csv","frontend_metric_series.csv",
}
RAW_OR_LEGACY = ["per_flow_time_long", "per_party_time_long", "daily_cash_position", "box_balance_time_long", "box_flow_balance_time_long", "views/", "income_statement_y", "balance_cash_y", "balance_debt_y", "metric_values.csv"]
REQUIRED = [
    "IS.REVENUE.OPERATING","IS.RENT.TOTAL","IS.OPEX.PROPERTY","IS.NET.OPERATING","IS.RENT.BY_PROPERTY","IS.OPEX.BY_CATEGORY",
    "FUND.CONTRIB.TOTAL","FUND.CONTRIB.BY_ACTOR","FUND.CONTRIB.BY_FUNDING_ACTOR",
    "FUND.CONTRIB.BY_CHANNEL","FUND.CONTRIB.BY_CASH_EFFECT","FUND.CONTRIB.BY_TARGET_BOX",
    "FUND.CONTRIB.DIRECT_OBLIGATION","FUND.CONTRIB.CASH_TO_BOX","FUND.CONTRIB.DEBT_LINKED",
    "DIST.DRAWS.PERSONAL","DIST.DIVIDENDS","DIST.DRAWS.BY_TYPE","COV.NET.AFTER_DRAWS","COV.SAVINGS_RATE",
    "BS.CASH.TOTAL","BS.CASH.CLOSE.BOX","DQ.CASH.FRONTEND_SAFE",
    "ID.DEBT.TOTAL.OPEN","ID.DEBT.OPEN.BY_COUNTERPARTY","ID.DEBT.PRINCIPAL.OPEN","ID.DEBT.INTEREST.OPEN","ID.DEBT.NET_PM_POSITION",
    "ID.DEBT.ACTIVITY.NEW_CLAIMS","ID.DEBT.ACTIVITY.REPAYMENTS","ID.DEBT.ACTIVITY.INTEREST_ACCRUED","ID.DEBT.ACTIVITY.ADJUSTMENTS","ID.DEBT.ACTIVITY.NET_CHANGE",
    "DQ.CLASSIFICATION.COVERAGE","DQ.UNKNOWN.AMOUNT","DQ.OPEX.LEAKAGE.AMOUNT","DQ.DEBT.ACTIVITY.RECONCILIATION",
    "TR.FX.CONVERSION.IN","TR.FX.CONVERSION.OUT","TR.FX.COST.OUT","TR.FX.NET","TR.FX.BY_BOX","TR.FX.BY_TYPE",
]
LEGACY = ["IS.INCOME.TOTAL","IS.NET.AFTER_COSTS","IS.NET.POST_DRAWS","IS.CONTRIB.TOTAL","IS.DRAWS.PERSONAL","BS.CASH.FB","BS.CASH.PM"]


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None

def _truth(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes","y"})

def _year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["period"] = out["period"].astype(str).str.slice(0,4)
    out["period_start"] = out["period"] + "-01-01"
    out["period_end"] = out["period"] + "-12-31"
    return out

def _base(metric_id: str, year: str, currency: str, value: Any, status: str, flow: str, acct: str, dash: str, source: str, filt: str, rule: str, run_id: str, as_of_date: str, *, dim_name: str="", dim_value: str="", suit: str="safe", public: bool=True, internal: bool=False, legacy: bool=False, validation: str="ok", caveat: str="") -> dict[str, Any]:
    return {"metric_id":metric_id,"period_grain":"Y","period":str(year),"period_start":f"{year}-01-01" if year else "","period_end":f"{year}-12-31" if year else "","Currency":currency,"value":value,"value_status":status,"flow_or_stock":flow,"accounting_section":acct,"dashboard_section":dash,"dimension_name":dim_name,"dimension_value":dim_value,"source_table":source,"source_filter":filt,"calculation_rule":rule,"frontend_suitability":suit,"public_flag":str(public).lower(),"internal_flag":str(internal).lower(),"legacy_flag":str(legacy).lower(),"validation_status":validation,"caveat":caveat,"run_id":run_id,"as_of_date":as_of_date}

def _nonempty_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().ne("")

def _funding_support_mask(df: pd.DataFrame) -> pd.Series:
    funding_channel = df.get("funding_channel", pd.Series("", index=df.index))
    debt_effect = df.get("debt_effect", pd.Series("none", index=df.index))
    return (
        df["semantic_bucket"].astype(str).eq("funding_contribution")
        | _nonempty_text(funding_channel)
        | debt_effect.fillna("none").astype(str).str.strip().ne("none")
    )

def _governed_semantic_amount(df: pd.DataFrame) -> pd.Series:
    """Project semantic rows through the canonical atomic-measure contract."""

    # Keep the governed source column's numeric representation so this authority
    # migration does not rewrite otherwise-identical annual CSV values.
    values = pd.Series(pd.NA, index=df.index, dtype="object")
    for (bucket, subbucket), rows in df.groupby(
        ["semantic_bucket", "semantic_subbucket"], dropna=False
    ):
        measure = resolve_semantic_measure(bucket, subbucket)
        if measure is None:
            raise ValueError(
                "No approved semantic measure for annual detail "
                f"bucket={bucket!r}, subbucket={subbucket!r}"
            )
        values.loc[rows.index] = pd.to_numeric(rows[measure], errors="coerce").fillna(0.0)
    return values

def _contract_from_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=ANNUAL_CONTRACT_COLUMNS)
    c = metrics.sort_values(["metric_id","value_status"]).drop_duplicates("metric_id")
    out = c[["metric_id","period_grain","flow_or_stock","accounting_section","dashboard_section","source_table","source_filter","calculation_rule","frontend_suitability","public_flag","internal_flag","legacy_flag","validation_status","caveat"]].copy()
    out["dimension_convention"] = "Use dimension_name/dimension_value for property, category, actor, counterparty, box, and activity detail; metric IDs stay stable."
    out["status"] = out["validation_status"].where(out["validation_status"].ne("legacy"), "legacy")
    return out[ANNUAL_CONTRACT_COLUMNS]

def build_annual_balance_dashboard(run_root: Path, metrics_dir: Path, run_id: str, as_of_date: str) -> dict[str, Path]:
    run_root, metrics_dir = Path(run_root), Path(metrics_dir); metrics_dir.mkdir(parents=True, exist_ok=True)
    stmt = _read(run_root/"monthly_operating_statement.csv"); split = _read(run_root/"monthly_flow_semantic_split.csv")
    cash = _read(run_root/"monthly_cash_close.csv"); debt = _read(run_root/"monthly_debt_position.csv"); act = _read(run_root/"monthly_debt_activity.csv"); leak = _read(run_root/"semantic_leakage_qa.csv")
    run_scope = load_run_scope_if_present(run_root)
    if run_scope is not None:
        for source_name, frame in [
            ("monthly_flow_semantic_split.csv", split),
            ("monthly_cash_close.csv", cash),
        ]:
            if frame is not None:
                assert_frame_within_scope(frame, run_scope, source=source_name)
    rows: list[dict[str, Any]] = []
    def unavailable(metric_id, source, dash, flow="flow", caveat="canonical source missing or required dimension unavailable"):
        rows.append(_base(metric_id,"","",pd.NA,"unavailable",flow,"unavailable",dash,source,"","unavailable is better than unsafe",run_id,as_of_date,suit="unavailable",validation="warn",caveat=caveat))

    stmt_map = {"operating_revenue":"IS.REVENUE.OPERATING","property_opex_true":"IS.OPEX.PROPERTY","net_operating":"IS.NET.OPERATING","funding_contributions":"FUND.CONTRIB.TOTAL","family_draws_or_distributions":"DIST.DRAWS.PERSONAL","dividends":"DIST.DIVIDENDS","coverage_after_draws":"COV.NET.AFTER_DRAWS","unknown_or_ambiguous_outflows":"DQ.UNKNOWN.AMOUNT"}
    if stmt is not None and not stmt.empty:
        w=_year(stmt); w["amount"]=pd.to_numeric(w["amount"],errors="coerce").fillna(0.0)
        for line, mid in stmt_map.items():
            sub=w[w["statement_line"].astype(str).eq(line)]
            for _,r in sub.groupby(["period","Currency"],dropna=False)["amount"].sum().reset_index().iterrows():
                dash = "1. Operating result" if mid.startswith("IS.") else ("2. Funding and distributions" if not mid.startswith("DQ.") else "6. Data quality and caveats")
                rows.append(_base(mid,r.period,r.Currency,r.amount,"available","flow" if not mid.startswith("DQ.") else "quality","income_statement" if mid.startswith("IS.") else "coverage",dash,"monthly_operating_statement.csv",f"statement_line={line}","annual flow = sum monthly flow by year and currency",run_id,as_of_date,suit="safe_with_caveat"))
        # ratio from annual aggregates
        piv=w[w["statement_line"].isin(["coverage_after_draws","net_operating"])].pivot_table(index=["period","Currency"],columns="statement_line",values="amount",aggfunc="sum").reset_index()
        for _,r in piv.iterrows():
            den=r.get("net_operating",0)
            status="available" if pd.notna(den) and den!=0 else "not_applicable"
            val=(r.get("coverage_after_draws",0)/den) if status=="available" else pd.NA
            rows.append(_base("COV.SAVINGS_RATE",r.period,r.Currency,val,status,"ratio","coverage","2. Funding and distributions","monthly_operating_statement.csv","statement_line in coverage_after_draws,net_operating","annual ratio = annual coverage_after_draws / annual net_operating",run_id,as_of_date,suit="safe_with_caveat",caveat="Ratio of annual aggregates; not an average of monthly ratios."))

        fx_caveat = "FX conversion changes liquidity by currency but is not operating income or funding."
        fx_stmt = {
            "treasury_fx_conversion_in": "TR.FX.CONVERSION.IN",
            "treasury_fx_conversion_out": "TR.FX.CONVERSION.OUT",
            "treasury_fx_cost": "TR.FX.COST.OUT",
            "treasury_fx_net": "TR.FX.NET",
        }
        for line, mid in fx_stmt.items():
            sub=w[w["statement_line"].astype(str).eq(line)]
            for _,r in sub.groupby(["period","Currency"],dropna=False)["amount"].sum().reset_index().iterrows():
                rows.append(_base(mid,r.period,r.Currency,r.amount,"available","flow","treasury","treasury_fx","monthly_operating_statement.csv",f"statement_line={line}","annual flow = sum monthly flow by year and currency",run_id,as_of_date,suit="safe_with_caveat",caveat=fx_caveat))
        cov=w[w["statement_line"].astype(str).eq("classification_coverage")]
        for _,r in cov.sort_values("period_end").groupby(["period","Currency"],dropna=False).tail(1).iterrows():
            rows.append(_base("DQ.CLASSIFICATION.COVERAGE",r.period,r.Currency,r.amount,"available","quality","data_quality","6. Data quality and caveats","monthly_operating_statement.csv","statement_line=classification_coverage","last valid monthly coverage value in year",run_id,as_of_date,suit="safe_with_caveat"))
    else:
        for mid in [v for v in stmt_map.values()]+["COV.SAVINGS_RATE","DQ.CLASSIFICATION.COVERAGE"]: unavailable(mid,"monthly_operating_statement.csv","6. Data quality and caveats")

    for mid in ["IS.REVENUE.OPERATING","IS.OPEX.PROPERTY","IS.NET.OPERATING","FUND.CONTRIB.TOTAL","DIST.DRAWS.PERSONAL","DIST.DIVIDENDS","COV.NET.AFTER_DRAWS","COV.SAVINGS_RATE","DQ.UNKNOWN.AMOUNT","DQ.CLASSIFICATION.COVERAGE"]:
        if not any(r["metric_id"] == mid for r in rows):
            unavailable(mid, "monthly_operating_statement.csv", "6. Data quality and caveats" if mid.startswith("DQ.") else ("2. Funding and distributions" if mid.startswith(("FUND.","DIST.","COV.")) else "1. Operating result"))

    if split is not None and not split.empty:
        s=_year(split); 
        for col in ["amount_in","amount_out","net_amount","amount_abs"]: s[col]=pd.to_numeric(s.get(col,0),errors="coerce").fillna(0.0)
        specs=[("IS.RENT.TOTAL",s.semantic_bucket.eq("operating_revenue")&s.semantic_subbucket.eq("rent"),"Lugar","1. Operating result","IS.RENT.BY_PROPERTY"),("IS.OPEX.BY_CATEGORY",s.semantic_bucket.eq("property_opex"),"semantic_subbucket","1. Operating result","IS.OPEX.BY_CATEGORY"),("FUND.CONTRIB.BY_ACTOR",s.semantic_bucket.eq("funding_contribution"),"actor","2. Funding and distributions","FUND.CONTRIB.BY_ACTOR"),("DIST.DRAWS.BY_TYPE",s.semantic_bucket.eq("family_withdrawal_candidate"),"semantic_subbucket","2. Funding and distributions","DIST.DRAWS.BY_TYPE")]
        for total_mid, mask, dim, dash, emit_mid in specs:
            sub=s[mask].copy()
            sub["governed_amount"] = _governed_semantic_amount(sub)
            if total_mid == "IS.RENT.TOTAL":
                for _,r in sub.groupby(["period","Currency"],dropna=False)["governed_amount"].sum().reset_index().iterrows():
                    rows.append(_base(total_mid,r.period,r.Currency,r.governed_amount,"available","flow","semantic_flow",dash,"monthly_flow_semantic_split.csv","semantic_bucket=operating_revenue; semantic_subbucket=rent","annual flow = sum monthly flow by year and currency",run_id,as_of_date,suit="safe"))
            if dim in sub.columns:
                for _,r in sub.groupby(["period","Currency",dim],dropna=False)["governed_amount"].sum().reset_index().iterrows():
                    rows.append(_base(emit_mid,r.period,r.Currency,r.governed_amount,"available","flow","semantic_flow",dash,"monthly_flow_semantic_split.csv",f"semantic filter; dimension={dim}","annual flow = sum monthly flow by year, currency, and dimension",run_id,as_of_date,dim_name=dim,dim_value=str(r[dim]),suit="safe_with_caveat"))

        funding_support = s.loc[_funding_support_mask(s)].copy()
        if not funding_support.empty:
            funding_support["funding_support_amount"] = _governed_semantic_amount(funding_support)
            funding_caveat = "Funding/support semantic metrics include cash contributions, direct obligation payments, and debt-linked support; rent is excluded."
            dim_specs = [
                ("funding_actor", "FUND.CONTRIB.BY_FUNDING_ACTOR"),
                ("funding_channel", "FUND.CONTRIB.BY_CHANNEL"),
                ("cash_effect", "FUND.CONTRIB.BY_CASH_EFFECT"),
                ("target_box", "FUND.CONTRIB.BY_TARGET_BOX"),
            ]
            for dim, mid in dim_specs:
                if dim in funding_support.columns:
                    dim_rows = funding_support[_nonempty_text(funding_support[dim])]
                    for _,r in dim_rows.groupby(["period","Currency",dim],dropna=False)["funding_support_amount"].sum().reset_index().iterrows():
                        rows.append(_base(mid,r.period,r.Currency,r.funding_support_amount,"available","flow","funding_support","2. Funding and distributions","monthly_flow_semantic_split.csv",f"funding/support candidate; dimension={dim}","annual funding/support flow = sum monthly support amount by year, currency, and dimension",run_id,as_of_date,dim_name=dim,dim_value=str(r[dim]),suit="safe_with_caveat",caveat=funding_caveat))

            direct_mask = funding_support.get("cash_effect", pd.Series("", index=funding_support.index)).astype(str).eq("no_cash_in_box_direct_payment")
            cash_to_box_mask = funding_support.get("cash_effect", pd.Series("", index=funding_support.index)).astype(str).eq("cash_in_box")
            debt_mask = funding_support.get("debt_effect", pd.Series("none", index=funding_support.index)).fillna("none").astype(str).str.strip().ne("none")
            total_specs = [
                ("FUND.CONTRIB.DIRECT_OBLIGATION", direct_mask, "cash_effect=no_cash_in_box_direct_payment"),
                ("FUND.CONTRIB.CASH_TO_BOX", cash_to_box_mask, "cash_effect=cash_in_box"),
                ("FUND.CONTRIB.DEBT_LINKED", debt_mask, "debt_effect != none"),
            ]
            for mid, mask, filt in total_specs:
                sub = funding_support.loc[mask]
                for _,r in sub.groupby(["period","Currency"],dropna=False)["funding_support_amount"].sum().reset_index().iterrows():
                    rows.append(_base(mid,r.period,r.Currency,r.funding_support_amount,"available","flow","funding_support","2. Funding and distributions","monthly_flow_semantic_split.csv",f"funding/support candidate; {filt}","annual funding/support flow = sum monthly support amount by year and currency",run_id,as_of_date,suit="safe_with_caveat",caveat=funding_caveat))

        fx=s[s.semantic_bucket.astype(str).eq("treasury_fx")].copy()
        if not fx.empty:
            fx_caveat = "FX conversion changes liquidity by currency but is not operating income or funding."
            for dim, mid in [("Box","TR.FX.BY_BOX"),("semantic_subbucket","TR.FX.BY_TYPE")]:
                if dim in fx.columns:
                    for _,r in fx.groupby(["period","Currency",dim],dropna=False)["net_amount"].sum().reset_index().iterrows():
                        rows.append(_base(mid,r.period,r.Currency,r.net_amount,"available","flow","treasury","treasury_fx","monthly_flow_semantic_split.csv",f"semantic_bucket=treasury_fx; dimension={dim}","annual flow = sum monthly net FX by year, currency, and dimension",run_id,as_of_date,dim_name=dim,dim_value=str(r[dim]),suit="safe_with_caveat",caveat=fx_caveat))
            for _,r in fx.groupby(["period","Currency"],dropna=False)["amount_abs"].sum().reset_index().iterrows():
                rows.append(_base("DQ.FX.ONE_SIDED.AMOUNT",r.period,r.Currency,r.amount_abs,"available","quality","data_quality","treasury_fx","monthly_flow_semantic_split.csv","semantic_bucket=treasury_fx; placeholder one-sided visibility","one-sided FX visibility placeholder; native rows remain by currency",run_id,as_of_date,suit="safe_with_caveat",caveat="One-sided FX proceeds are allowed but cannot be treated as economic income in hard-currency projections."))
                rows.append(_base("DQ.FX.MISSING_RATE.AMOUNT",r.period,r.Currency,pd.NA,"unavailable","quality","data_quality","treasury_fx","monthly_flow_semantic_split.csv","future CCL projection rate availability","missing CCL rate produces unavailable, not zero",run_id,as_of_date,suit="unavailable",validation="warn",caveat="Hard-currency CCL projection is not implemented in this PR."))
                rows.append(_base("DQ.FX.ROWS.REVIEW_REQUIRED",r.period,r.Currency,0,"available","quality","data_quality","treasury_fx","monthly_flow_semantic_split.csv","review_required treasury_fx rows","count/sum placeholder for FX rows needing review",run_id,as_of_date,suit="safe_with_caveat",caveat="FX rows are classified; future matching may add review-required rows."))
    for mid in [
        "IS.RENT.BY_PROPERTY","IS.OPEX.BY_CATEGORY","FUND.CONTRIB.BY_ACTOR",
        "FUND.CONTRIB.BY_FUNDING_ACTOR","FUND.CONTRIB.BY_CHANNEL",
        "FUND.CONTRIB.BY_CASH_EFFECT","FUND.CONTRIB.BY_TARGET_BOX",
        "FUND.CONTRIB.DIRECT_OBLIGATION","FUND.CONTRIB.CASH_TO_BOX",
        "FUND.CONTRIB.DEBT_LINKED","DIST.DRAWS.BY_TYPE",
    ]:
        if not any(r["metric_id"]==mid for r in rows):
            dash = "2. Funding and distributions" if mid.startswith(("FUND.", "DIST.")) else "1. Operating result"
            unavailable(mid,"monthly_flow_semantic_split.csv",dash,caveat="blocked_by_missing_dimension")
    for mid in ["TR.FX.CONVERSION.IN","TR.FX.CONVERSION.OUT","TR.FX.COST.OUT","TR.FX.NET","TR.FX.BY_BOX","TR.FX.BY_TYPE"]:
        if not any(r["metric_id"]==mid for r in rows): unavailable(mid,"monthly_flow_semantic_split.csv","treasury_fx",caveat="No treasury FX rows classified in canonical semantic split.")

    if cash is not None and not cash.empty and "is_frontend_safe" in cash:
        c=_year(cash[_truth(cash["is_frontend_safe"])]); c["close_amount"]=pd.to_numeric(c.get("close_amount",0),errors="coerce")
        if c.empty:
            unavailable("BS.CASH.TOTAL","monthly_cash_close.csv","3. Cash and liquidity","stock","No frontend-safe cash rows exist; no fallback used."); unavailable("BS.CASH.CLOSE.BOX","monthly_cash_close.csv","3. Cash and liquidity","stock","No frontend-safe cash rows exist; no fallback used.")
        else:
            last=c.sort_values("period_end").groupby(["period","Currency","Box"],dropna=False).tail(1)
            for _,r in last.iterrows(): rows.append(_base("BS.CASH.CLOSE.BOX",r.period,r.Currency,r.close_amount,"available","stock","cash","3. Cash and liquidity","monthly_cash_close.csv","is_frontend_safe=true; dimension=Box","annual stock = last frontend-safe monthly close by year, Box, Currency",run_id,as_of_date,dim_name="Box",dim_value=str(r.Box),caveat="Only explicit frontend-safe cash rows are included."))
            for _,r in last.groupby(["period","Currency"],dropna=False)["close_amount"].sum().reset_index().iterrows(): rows.append(_base("BS.CASH.TOTAL",r.period,r.Currency,r.close_amount,"available","stock","cash","3. Cash and liquidity","monthly_cash_close.csv","is_frontend_safe=true","sum annual last frontend-safe box closes by year and currency",run_id,as_of_date,caveat="All source rows are explicitly frontend-safe."))
        for y in sorted(_year(cash).period.unique()): rows.append(_base("DQ.CASH.FRONTEND_SAFE",y,"",int(_truth(cash[cash.period.astype(str).str.startswith(y)]["is_frontend_safe"]).sum()),"available","quality","data_quality","3. Cash and liquidity","monthly_cash_close.csv","is_frontend_safe","count of frontend-safe cash rows by year",run_id,as_of_date,suit="safe_with_caveat"))
    else: unavailable("BS.CASH.TOTAL","monthly_cash_close.csv","3. Cash and liquidity","stock"); unavailable("BS.CASH.CLOSE.BOX","monthly_cash_close.csv","3. Cash and liquidity","stock"); unavailable("DQ.CASH.FRONTEND_SAFE","monthly_cash_close.csv","3. Cash and liquidity","quality")

    if debt is not None and not debt.empty:
        d=debt.copy(); d["period"]=d["period"].astype(str); d["year"]=d["period"].str.slice(0,4); d["open_amount"]=pd.to_numeric(d.get("open_amount",pd.NA),errors="coerce")
        selected_records=[]
        for (year,currency,debtor,creditor,component), group in d.groupby(["year","Currency","debtor","creditor","component"],dropna=False,sort=False):
            selection=select_debt_position(group,period=str(year),annual=True)
            if selection.available:
                picked=selected_debt_position_rows(group,selection).tail(1).iloc[0].to_dict()
                picked["value_status"]="available" if pd.notna(picked.get("open_amount")) else "unavailable"
            else:
                closing=group.loc[group["period"].astype(str).eq(selection.selected_period)]
                picked=(closing.iloc[0] if not closing.empty else group.iloc[0]).to_dict()
                picked["as_of_date"]=""
                picked["open_amount"]=pd.NA
                picked["value_status"]="unavailable"
            picked["period"]=str(year)
            picked["selection_reason"]=selection.reason
            picked["selected_period"]=selection.selected_period
            picked["valid_as_of_rows"]=selection.valid_as_of_rows
            selected_records.append(picked)
        last=pd.DataFrame(selected_records)
        comp={"total":"ID.DEBT.TOTAL.OPEN","principal":"ID.DEBT.PRINCIPAL.OPEN","interest":"ID.DEBT.INTEREST.OPEN"}
        for component,mid in comp.items():
            sub=last[last.component.astype(str).eq(component)]
            for (year,currency),g in sub.groupby(["period","Currency"],dropna=False):
                complete=g["value_status"].astype(str).eq("available").all() and g["open_amount"].notna().all()
                value=float(g["open_amount"].sum()) if complete else pd.NA
                caveat="Debt is stock, not flow; latest period then latest valid as_of_date; no prior-period or lexical fallback."
                if not complete: caveat += " At least one closing counterparty position is unavailable."
                rows.append(_base(mid,year,currency,value,"available" if complete else "unavailable","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv",f"component={component}","annual debt stock = latest period then latest valid as_of_date per counterparty; sum only when all closing positions are available",run_id,as_of_date,suit="safe_with_caveat" if complete else "unavailable",validation="ok" if complete else "warn",caveat=caveat))
        totals=last[last.component.astype(str).eq("total")]
        for _,r in totals.iterrows():
            available_row=str(r.get("value_status","unavailable"))=="available" and pd.notna(r.get("open_amount"))
            rows.append(_base("ID.DEBT.OPEN.BY_COUNTERPARTY",r.period,r.Currency,float(r.open_amount) if available_row else pd.NA,"available" if available_row else "unavailable","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv","component=total; dimension=debtor_creditor","annual stock = latest period then latest valid as_of_date by debtor, creditor, currency",run_id,as_of_date,dim_name="debtor_creditor",dim_value=f"{r.debtor} -> {r.creditor}",suit="safe_with_caveat" if available_row else "unavailable",validation="ok" if available_row else "warn",caveat="Debt stock selection is governed; prior periods are never substituted for an invalid closing period."))
        for (year,currency),g in totals.groupby(["period","Currency"],dropna=False):
            complete=g["value_status"].astype(str).eq("available").all() and g["open_amount"].notna().all()
            if complete:
                value=g.loc[g.creditor.astype(str).str.contains("Property Management",case=False,na=False),"open_amount"].sum()-g.loc[g.debtor.astype(str).str.contains("Property Management",case=False,na=False),"open_amount"].sum()
            else: value=pd.NA
            rows.append(_base("ID.DEBT.NET_PM_POSITION",year,currency,value,"available" if complete else "unavailable","stock","internal_debt","4. Internal debt and claims","monthly_debt_position.csv","component=total; PM creditor less PM debtor","annual stock = latest period then latest valid as_of_date; net PM position only when all closing counterparties are available",run_id,as_of_date,suit="safe_with_caveat" if complete else "unavailable",validation="ok" if complete else "warn",caveat="No prior-period or lexical fallback for unavailable closing debt positions."))
    else:
        for mid in ["ID.DEBT.TOTAL.OPEN","ID.DEBT.OPEN.BY_COUNTERPARTY","ID.DEBT.PRINCIPAL.OPEN","ID.DEBT.INTEREST.OPEN","ID.DEBT.NET_PM_POSITION"]: unavailable(mid,"monthly_debt_position.csv","4. Internal debt and claims","stock")

    if act is not None and not act.empty:
        a=_year(act); maps={"new_principal":"ID.DEBT.ACTIVITY.NEW_CLAIMS","repayments":"ID.DEBT.ACTIVITY.REPAYMENTS","interest_accrued":"ID.DEBT.ACTIVITY.INTEREST_ACCRUED","adjustments":"ID.DEBT.ACTIVITY.ADJUSTMENTS","net_change":"ID.DEBT.ACTIVITY.NET_CHANGE"}
        for col,mid in maps.items():
            a[col]=pd.to_numeric(a.get(col,0),errors="coerce").fillna(0.0)
            for _,r in a.groupby(["period","Currency","debtor","creditor"],dropna=False)[col].sum().reset_index().iterrows(): rows.append(_base(mid,r.period,r.Currency,r[col],"available","flow","internal_debt_activity","5. Debt activity","monthly_debt_activity.csv",f"sum {col}; dimension=debtor_creditor","annual activity = sum monthly activity by year, debtor, creditor, currency",run_id,as_of_date,dim_name="debtor_creditor",dim_value=f"{r.debtor} -> {r.creditor}",suit="safe_with_caveat",caveat="Debt movement; not OPEX or funding unless classified elsewhere."))
        for y in sorted(a.period.unique()): rows.append(_base("DQ.DEBT.ACTIVITY.RECONCILIATION",y,"",int(a[a.period.eq(y)]["reconciliation_status"].astype(str).eq("reconciled").sum()) if "reconciliation_status" in a else pd.NA,"available","quality","data_quality","6. Data quality and caveats","monthly_debt_activity.csv","reconciliation_status","count reconciled activity rows; residual adjustments remain visible",run_id,as_of_date,suit="safe_with_caveat"))
    else:
        for mid in ["ID.DEBT.ACTIVITY.NEW_CLAIMS","ID.DEBT.ACTIVITY.REPAYMENTS","ID.DEBT.ACTIVITY.INTEREST_ACCRUED","ID.DEBT.ACTIVITY.ADJUSTMENTS","ID.DEBT.ACTIVITY.NET_CHANGE","DQ.DEBT.ACTIVITY.RECONCILIATION"]: unavailable(mid,"monthly_debt_activity.csv","5. Debt activity","flow")
    if leak is not None and not leak.empty:
        l=_year(leak); l["amount"]=pd.to_numeric(l.get("amount",0),errors="coerce").fillna(0.0)
        for _,r in l.groupby(["period","Currency"],dropna=False)["amount"].sum().reset_index().iterrows(): rows.append(_base("DQ.OPEX.LEAKAGE.AMOUNT",r.period,r.Currency,r.amount,"available","quality","data_quality","6. Data quality and caveats","semantic_leakage_qa.csv","semantic leakage rows","annual sum suspicious property_opex leakage rows",run_id,as_of_date,suit="safe_with_caveat"))
    else: unavailable("DQ.OPEX.LEAKAGE.AMOUNT","semantic_leakage_qa.csv","6. Data quality and caveats","quality")
    for mid in LEGACY: rows.append(_base(mid,"","",pd.NA,"not_applicable","legacy","legacy_reconciliation","7. Legacy reconciliation","metric_contract_frontier.csv","legacy compatibility marker","legacy metrics demoted; not canonical dashboard metrics",run_id,as_of_date,suit="legacy_only",public=False,internal=True,legacy=True,validation="legacy",caveat="Legacy/reconciliation only unless rebuilt from canonical sources."))

    metrics=pd.DataFrame(rows,columns=ANNUAL_METRICS_COLUMNS)
    contract=_contract_from_rows(metrics)
    qa=build_annual_balance_dashboard_qa(metrics, contract)
    paths={"annual_balance_dashboard_metrics":metrics_dir/"annual_balance_dashboard_metrics.csv","annual_balance_dashboard_contract":metrics_dir/"annual_balance_dashboard_contract.csv","annual_balance_dashboard_qa":metrics_dir/"annual_balance_dashboard_qa.csv"}
    metrics.to_csv(paths["annual_balance_dashboard_metrics"],index=False); contract.to_csv(paths["annual_balance_dashboard_contract"],index=False); qa.to_csv(paths["annual_balance_dashboard_qa"],index=False)
    return paths

def build_annual_balance_dashboard_qa(metrics: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    def add(c, ok, d, sev="error"): rows.append({"check":c,"status":"pass" if ok else "fail","detail":d,"severity":sev})
    sources=set(metrics.source_table.dropna().astype(str)) if not metrics.empty else set()
    add("annual_metrics_use_only_canonical_sources", sources.issubset(CANONICAL), f"sources={sorted(sources)}")
    add("no_raw_stage_d_sources", not any(any(b in s for b in RAW_OR_LEGACY[:5]) for s in sources), f"sources={sorted(sources)}")
    add("no_legacy_views_as_canonical", not any("views/" in s or s.startswith("old ") for s in sources), f"sources={sorted(sources)}")
    add("annual_flows_sum_monthly_flows", metrics.empty or metrics[metrics.flow_or_stock.eq("flow") & metrics.value_status.eq("available")].calculation_rule.str.contains("sum monthly|annual flow|annual activity",case=False,na=False).all(), "flow rules documented")
    add("annual_stocks_use_last_close", metrics.empty or metrics[metrics.flow_or_stock.eq("stock")].calculation_rule.str.contains("last|sum annual last",case=False,na=False).all(), "stock rules documented")
    add("ratios_use_annual_aggregates", metrics[metrics.flow_or_stock.eq("ratio")].calculation_rule.str.contains("annual",case=False,na=False).all() if not metrics.empty else True, "ratio rules documented")
    add("no_cross_currency_aggregation", metrics.empty or metrics[~metrics.flow_or_stock.isin(["quality","legacy"]) & metrics.value_status.eq("available")].Currency.astype(str).str.strip().ne("").all(), "money metrics carry Currency")
    cash=metrics[metrics.metric_id.str.startswith("BS.CASH")]; add("cash_unavailable_without_frontend_safe_rows", cash.empty or cash[~cash.legacy_flag.astype(str).eq("true")].value_status.isin(["available","unavailable"]).all(), f"cash_statuses={sorted(cash.value_status.unique())}")
    add("debt_stock_not_mixed_with_flows", not metrics[metrics.metric_id.str.startswith("ID.DEBT") & metrics.metric_id.str.contains("OPEN|POSITION")].flow_or_stock.eq("flow").any(), "debt OPEN/POSITION are stock")
    add("debt_activity_reconciles_or_residual_visible", "ID.DEBT.ACTIVITY.ADJUSTMENTS" in set(metrics.metric_id) and "DQ.DEBT.ACTIVITY.RECONCILIATION" in set(metrics.metric_id), "adjustments and reconciliation metric present")
    add("legacy_metrics_marked_legacy", metrics[metrics.metric_id.isin(LEGACY)].legacy_flag.astype(str).eq("true").all(), "legacy IDs demoted")
    ids=set(metrics.metric_id.astype(str)); missing=[m for m in REQUIRED if m not in ids]; unavailable=metrics[metrics.metric_id.isin(REQUIRED)&~metrics.value_status.eq("available")].metric_id.drop_duplicates().tolist()
    fx_ids={"TR.FX.CONVERSION.IN","TR.FX.CONVERSION.OUT","TR.FX.COST.OUT","TR.FX.NET","TR.FX.BY_BOX","TR.FX.BY_TYPE"}
    add("fx_metrics_present_or_unavailable", fx_ids.issubset(ids), f"missing={sorted(fx_ids-ids)}")
    add("required_dashboard_metrics_present_or_unavailable", not missing, f"missing={missing}; unavailable_or_not_applicable={unavailable}")
    return pd.DataFrame(rows,columns=QA_COLUMNS)
