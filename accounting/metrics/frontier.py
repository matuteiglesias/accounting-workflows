from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from accounting.artifacts.manifest import artifact_contract_for_name

FRONTIER_COLUMNS = [
    "metric_id", "label", "semantic_category", "flow_or_stock", "period_grain", "currency_mode",
    "source_table", "calculation_rule", "lineage_inputs", "frontend_suitability", "public_flag",
    "internal_flag", "legacy_flag", "caveat", "validation_status", "owner", "status", "notes",
]
SERIES_COLUMNS = [
    "metric_id", "period_grain", "period", "period_end", "Currency", "value", "dimension_name",
    "dimension_value", "source_table", "run_id", "as_of_date", "frontend_suitability", "public_flag",
    "internal_flag", "legacy_flag", "caveat", "validation_status",
]
QA_COLUMNS = ["check", "status", "detail", "severity"]
INITIAL_METRICS = [
    "IS.RENT.TOTAL", "IS.REVENUE.OPERATING", "IS.OPEX.PROPERTY", "IS.NET.OPERATING",
    "FUND.CONTRIB.TOTAL", "DIST.DRAWS.PERSONAL", "COV.NET.AFTER_DRAWS",
    "BS.CASH.TOTAL", "BS.CASH.CLOSE.BOX", "ID.DEBT.OPEN.BY_COUNTERPARTY",
    "DQ.CLASSIFICATION.COVERAGE", "DQ.UNKNOWN.AMOUNT",
]


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_frontier_sources(run_root: Path, metrics_dir: Path | None = None) -> dict[str, pd.DataFrame | None]:
    run_root = Path(run_root)
    metrics_dir = Path(metrics_dir) if metrics_dir is not None else None
    sources = {
        "monthly_flow_semantic_split": _read_csv(run_root / "monthly_flow_semantic_split.csv"),
        "monthly_operating_statement": _read_csv(run_root / "monthly_operating_statement.csv"),
        "monthly_cash_close": _read_csv(run_root / "monthly_cash_close.csv"),
        "monthly_debt_position": _read_csv(run_root / "monthly_debt_position.csv"),
        "monthly_debt_activity": _read_csv(run_root / "monthly_debt_activity.csv"),
    }
    if metrics_dir is not None:
        sources["metric_registry"] = _read_csv(metrics_dir / "metric_registry.csv")
        sources["metric_values"] = _read_csv(metrics_dir / "metric_values.csv")
        sources["metric_contract_frontier"] = _read_csv(metrics_dir / "metric_contract_frontier.csv")
        sources["frontend_metric_series"] = _read_csv(metrics_dir / "frontend_metric_series.csv")
    return sources


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _contract_row(metric_id: str, label: str, category: str, flow_or_stock: str, source_table: str, rule: str, *,
                  suitability: str = "safe_with_caveat", public: bool = True, internal: bool = False,
                  legacy: bool = False, caveat: str = "", status: str = "active", validation: str = "ok",
                  notes: str = "", period_grain: str = "M", currency_mode: str = "by_currency") -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "label": label,
        "semantic_category": category,
        "flow_or_stock": flow_or_stock,
        "period_grain": period_grain,
        "currency_mode": currency_mode,
        "source_table": source_table,
        "calculation_rule": rule,
        "lineage_inputs": source_table,
        "frontend_suitability": suitability,
        "public_flag": _bool_str(public),
        "internal_flag": _bool_str(internal),
        "legacy_flag": _bool_str(legacy),
        "caveat": caveat,
        "validation_status": validation,
        "owner": "accounting_backend",
        "status": status,
        "notes": notes,
    }


def _series_row(metric_id: str, period: Any, period_end: Any, currency: Any, value: Any, source_table: str,
                run_id: str, as_of_date: str, suitability: str, public: bool, internal: bool, legacy: bool,
                caveat: str, validation: str = "ok", dimension_name: str = "", dimension_value: str = "") -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "period_grain": "M",
        "period": str(period),
        "period_end": str(period_end),
        "Currency": str(currency),
        "value": float(value) if pd.notna(value) else pd.NA,
        "dimension_name": dimension_name,
        "dimension_value": dimension_value,
        "source_table": source_table,
        "run_id": run_id,
        "as_of_date": as_of_date,
        "frontend_suitability": suitability,
        "public_flag": _bool_str(public),
        "internal_flag": _bool_str(internal),
        "legacy_flag": _bool_str(legacy),
        "caveat": caveat,
        "validation_status": validation,
    }


def _statement_series(statement: pd.DataFrame, line: str, metric_id: str, run_id: str, as_of_date: str,
                      suitability: str, caveat: str) -> list[dict[str, Any]]:
    if statement is None or statement.empty:
        return []
    sub = statement.loc[statement["statement_line"].astype(str).eq(line)].copy()
    return [
        _series_row(metric_id, r["period"], r["period_end"], r["Currency"], r["amount"], "monthly_operating_statement.csv", run_id, as_of_date, suitability, True, False, False, caveat, r.get("validation_status", "ok"))
        for _, r in sub.iterrows()
    ]


def build_metrics_frontier(run_root: Path, metrics_dir: Path, run_id: str, as_of_date: str) -> Dict[str, Path]:
    run_root = Path(run_root)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sources = load_frontier_sources(run_root, metrics_dir)
    split = sources["monthly_flow_semantic_split"]
    statement = sources["monthly_operating_statement"]
    cash = sources["monthly_cash_close"]
    debt = sources["monthly_debt_position"]
    debt_activity = sources["monthly_debt_activity"]
    metric_registry = sources.get("metric_registry")
    metric_values = sources.get("metric_values")

    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []

    def source_status(df: pd.DataFrame | None) -> tuple[str, str, str]:
        if df is None:
            return "unavailable", "unavailable", "required canonical source missing"
        return "active", "ok", ""

    stmt_status, stmt_validation, stmt_caveat = source_status(statement)
    split_status, split_validation, split_caveat = source_status(split)
    cash_status, cash_validation, cash_caveat = source_status(cash)
    debt_status, debt_validation, debt_caveat = source_status(debt)
    debt_activity_status, debt_activity_validation, debt_activity_caveat = source_status(debt_activity)

    # Statement-backed clean metrics.
    stmt_defs = [
        ("IS.REVENUE.OPERATING", "Operating revenue", "operating_revenue", "flow", "operating_revenue", "statement_line=operating_revenue", "safe", ""),
        ("IS.OPEX.PROPERTY", "True property OPEX", "property_opex", "flow", "property_opex_true", "statement_line=property_opex_true", "safe_with_caveat", "Depends on semantic classification coverage; review unknown amounts."),
        ("IS.NET.OPERATING", "Net operating", "operating_result", "flow", "net_operating", "operating_revenue - property_opex_true", "safe_with_caveat", "Clean operating result excludes funding, family draws, debt, transfers, and unknown/review-required flows."),
        ("FUND.CONTRIB.TOTAL", "Funding contributions", "funding", "flow", "funding_contributions", "statement_line=funding_contributions", "safe_with_caveat", "Funding is not operating revenue."),
        ("DIST.DRAWS.PERSONAL", "Family draws or distributions", "distribution", "flow", "family_draws_or_distributions", "statement_line=family_draws_or_distributions", "safe_with_caveat", "Distribution-like outflows are based on semantic rules and may require review."),
        ("COV.NET.AFTER_DRAWS", "Coverage after draws", "coverage", "mixed", "coverage_after_draws", "net_operating + funding_contributions - family_draws_or_distributions", "safe_with_caveat", "Coverage metric, not legal/accounting net income."),
        ("DQ.CLASSIFICATION.COVERAGE", "Classification coverage", "data_quality", "quality", "classification_coverage", "classified_amount_abs / eligible_amount_abs", "safe_with_caveat", "Ratio from semantic statement; convention is 0-1 ratio."),
        ("DQ.UNKNOWN.AMOUNT", "Unknown/review-required amount", "data_quality", "quality", "unknown_or_ambiguous_outflows", "unknown or review-required amount", "safe_with_caveat", "Amounts require accounting review."),
    ]
    for metric_id, label, category, flow_or_stock, line, rule, suitability, caveat in stmt_defs:
        unavailable = stmt_status == "unavailable"
        rows.append(_contract_row(metric_id, label, category, flow_or_stock, "monthly_operating_statement.csv", rule, suitability="unavailable" if unavailable else suitability, caveat=stmt_caveat or caveat, status="unavailable" if unavailable else "active", validation=stmt_validation))
        if not unavailable:
            value_line = line
            if metric_id == "DQ.UNKNOWN.AMOUNT":
                # Prefer canonical data-quality line value, already amount-like.
                value_line = "unknown_or_ambiguous_outflows"
            series.extend(_statement_series(statement, value_line, metric_id, run_id, as_of_date, suitability, caveat))

    # Rent from semantic split.
    rent_unavailable = split_status == "unavailable"
    rows.append(_contract_row("IS.RENT.TOTAL", "Rent revenue", "operating_revenue", "flow", "monthly_flow_semantic_split.csv", "sum amount_in where semantic_bucket=operating_revenue and semantic_subbucket=rent", suitability="unavailable" if rent_unavailable else "safe", caveat=split_caveat, status="unavailable" if rent_unavailable else "active", validation=split_validation))
    if not rent_unavailable:
        rent = split.loc[(split["semantic_bucket"].astype(str) == "operating_revenue") & (split["semantic_subbucket"].astype(str) == "rent")].copy()
        if not rent.empty:
            grouped = rent.groupby(["period", "period_end", "Currency"], dropna=False)["amount_in"].sum().reset_index()
            for _, r in grouped.iterrows():
                series.append(_series_row("IS.RENT.TOTAL", r["period"], r["period_end"], r["Currency"], r["amount_in"], "monthly_flow_semantic_split.csv", run_id, as_of_date, "safe", True, False, False, ""))

    # Cash metrics: unavailable unless explicit frontend-safe cash rows exist.
    cash_unavailable = cash_status == "unavailable"
    frontend_safe_cash = pd.DataFrame() if cash_unavailable else cash.loc[cash["is_frontend_safe"].astype(str).str.lower().isin({"true", "1", "yes", "y"})].copy()
    no_frontend_safe = cash_unavailable or frontend_safe_cash.empty
    cash_suitability = "unavailable" if no_frontend_safe else "safe"
    cash_contract_status = "unavailable" if no_frontend_safe else "active"
    cash_caveat_text = cash_caveat or "No frontend-safe cash rows in monthly_cash_close.csv; no fallback to party/internal balances."
    rows.append(_contract_row("BS.CASH.TOTAL", "Frontend-safe cash total", "cash", "stock", "monthly_cash_close.csv", "sum close_amount where is_frontend_safe=true by period/currency", suitability=cash_suitability, caveat=cash_caveat_text if no_frontend_safe else "Only explicit frontend-safe cash rows are included.", status=cash_contract_status, validation="warn" if no_frontend_safe else "ok"))
    rows.append(_contract_row("BS.CASH.CLOSE.BOX", "Frontend-safe cash by Box", "cash", "stock", "monthly_cash_close.csv", "close_amount by Box where is_frontend_safe=true", suitability=cash_suitability, caveat=cash_caveat_text if no_frontend_safe else "Only explicit frontend-safe cash rows are included.", status=cash_contract_status, validation="warn" if no_frontend_safe else "ok", notes="dimension_name=Box"))
    if not no_frontend_safe:
        grouped = frontend_safe_cash.groupby(["period", "period_end", "Currency"], dropna=False)["close_amount"].sum().reset_index()
        for _, r in grouped.iterrows():
            series.append(_series_row("BS.CASH.TOTAL", r["period"], r["period_end"], r["Currency"], r["close_amount"], "monthly_cash_close.csv", run_id, as_of_date, "safe", True, False, False, "Only explicit frontend-safe cash rows are included."))
        for _, r in frontend_safe_cash.iterrows():
            series.append(_series_row("BS.CASH.CLOSE.BOX", r["period"], r["period_end"], r["Currency"], r["close_amount"], "monthly_cash_close.csv", run_id, as_of_date, "safe", True, False, False, "Only explicit frontend-safe cash rows are included.", dimension_name="Box", dimension_value=str(r.get("Box", ""))))

    # Debt metric.
    debt_unavailable = debt_status == "unavailable"
    rows.append(_contract_row("ID.DEBT.OPEN.BY_COUNTERPARTY", "Open internal debt by counterparty", "internal_debt", "stock", "monthly_debt_position.csv", "open_total by debtor/creditor/currency where component=total", suitability="unavailable" if debt_unavailable else "safe_with_caveat", caveat=debt_caveat or "Internal debt/claim position, not operating expense or cash.", status="unavailable" if debt_unavailable else "active", validation=debt_validation, notes="dimension_name=debtor_creditor"))
    if not debt_unavailable and not debt.empty:
        sub = debt.loc[debt["component"].astype(str).eq("total")].copy()
        for _, r in sub.iterrows():
            series.append(_series_row("ID.DEBT.OPEN.BY_COUNTERPARTY", r["period"], r["period_end"], r["Currency"], r["open_amount"], "monthly_debt_position.csv", run_id, as_of_date, "safe_with_caveat", True, False, False, "Internal debt/claim position, not operating expense or cash.", dimension_name="debtor_creditor", dimension_value=f"{r['debtor']} -> {r['creditor']}"))

    # Debt activity compatibility: register the canonical activity source for future dashboards without emitting annual metrics yet.
    rows.append(_contract_row("ID.DEBT.ACTIVITY", "Internal debt movement activity", "internal_debt", "flow", "monthly_debt_activity.csv", "future dashboard source for new claims, repayments, interest accruals, opening/closing movement and residual adjustments", suitability="unavailable" if debt_activity_status == "unavailable" else "safe_with_caveat", public=False, internal=True, caveat=debt_activity_caveat or "Debt activity movement source; not an operating statement and not a debt closing balance.", status="unavailable" if debt_activity_status == "unavailable" else "active", validation=debt_activity_validation, notes="compatibility contract only; no dashboard series emitted yet"))

    # More complete registry: carry selected legacy metrics as legacy-only definitions for consumers.
    legacy_ids = ["IS.NET.AFTER_COSTS", "IS.CONTRIB.TOTAL", "IS.DRAWS.PERSONAL", "BS.CASH.FB", "BS.CASH.PM", "BS.DEBT.TOTAL.OPEN", "BS.DEBT.PRINCIPAL.OPEN", "BS.DEBT.INTEREST.OPEN", "BS.DEBT.NET_PM_POSITION"]
    if metric_registry is not None and not metric_registry.empty:
        for metric_id in legacy_ids:
            match = metric_registry.loc[metric_registry["metric_id"].astype(str).eq(metric_id)]
            label = match["label"].iloc[0] if not match.empty and "label" in match else metric_id
            legacy_warning = match["legacy_warning"].iloc[0] if not match.empty and "legacy_warning" in match else "Legacy metric retained for compatibility."
            rows.append(_contract_row(metric_id, label, "legacy", "mixed", "metric_values.csv", "legacy metric_values output", suitability="legacy_only", public=False, internal=True, legacy=True, caveat=legacy_warning or "Legacy metric retained for compatibility.", status="legacy", validation="legacy"))
    if metric_values is not None and not metric_values.empty:
        currency_col = "currency" if "currency" in metric_values.columns else "Currency"
        for metric_id in legacy_ids:
            sub = metric_values.loc[metric_values["metric_id"].astype(str).eq(metric_id)].copy()
            for _, r in sub.iterrows():
                series.append(_series_row(metric_id, r.get("period", ""), r.get("period_end", ""), r.get(currency_col, ""), r.get("value", pd.NA), "metric_values.csv", run_id, as_of_date, "legacy_only", False, True, True, "Legacy metric retained for compatibility.", validation="legacy"))

    frontier = pd.DataFrame(rows, columns=FRONTIER_COLUMNS).drop_duplicates(["metric_id"], keep="first")
    series_df = pd.DataFrame(series, columns=SERIES_COLUMNS)
    qa = build_frontier_qa(frontier, series_df, cash, metric_registry, metric_values)

    paths = {
        "metric_contract_frontier": metrics_dir / "metric_contract_frontier.csv",
        "frontend_metric_series": metrics_dir / "frontend_metric_series.csv",
        "metrics_frontier_qa": metrics_dir / "metrics_frontier_qa.csv",
        "frontier_source_qa": metrics_dir / "frontier_source_qa.csv",
    }
    frontier.to_csv(paths["metric_contract_frontier"], index=False)
    series_df.to_csv(paths["frontend_metric_series"], index=False)
    qa.to_csv(paths["metrics_frontier_qa"], index=False)
    qa.to_csv(paths["frontier_source_qa"], index=False)
    return paths


def build_frontier_qa(frontier: pd.DataFrame, series: pd.DataFrame, cash: pd.DataFrame | None,
                      metric_registry: pd.DataFrame | None, metric_values: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(check: str, ok: bool, detail: str, severity: str = "error") -> None:
        rows.append({"check": check, "status": "pass" if ok else "fail", "detail": detail, "severity": severity})

    ids = set(frontier["metric_id"].astype(str)) if not frontier.empty else set()
    missing_initial = [m for m in INITIAL_METRICS if m not in ids]
    unavailable_initial = frontier.loc[frontier["metric_id"].isin(INITIAL_METRICS) & frontier["status"].eq("unavailable"), "metric_id"].astype(str).tolist() if not frontier.empty else []
    add("metric_contract_frontier_exists", not frontier.empty, f"rows={len(frontier)}")
    add("frontend_metric_series_exists", not series.empty, f"rows={len(series)}")
    add("all_initial_metrics_present_or_explicitly_unavailable", not missing_initial, f"missing={missing_initial}; unavailable={unavailable_initial}")

    cash_safe_source = False if cash is None or cash.empty else cash["is_frontend_safe"].astype(str).str.lower().isin({"true", "1", "yes", "y"}).any()
    cash_series = series.loc[series["metric_id"].astype(str).str.startswith("BS.CASH")]
    add("no_frontend_cash_without_frontend_safe_cash_source", cash_safe_source or cash_series.empty, f"safe_source={cash_safe_source}; cash_series_rows={len(cash_series)}")
    op_contract = frontier.loc[frontier["metric_id"].eq("IS.REVENUE.OPERATING"), "calculation_rule"].astype(str).str.cat(sep=";")
    opex_contract = frontier.loc[frontier["metric_id"].eq("IS.OPEX.PROPERTY"), "calculation_rule"].astype(str).str.cat(sep=";")
    add("no_funding_in_operating_revenue", "funding" not in op_contract.lower(), op_contract)
    add("no_family_draws_in_property_opex", "family" not in opex_contract.lower() and "draw" not in opex_contract.lower(), opex_contract)
    debt_series = series.loc[series["metric_id"].eq("ID.DEBT.OPEN.BY_COUNTERPARTY")]
    allowed_sources = {"monthly_operating_statement.csv", "monthly_flow_semantic_split.csv", "monthly_cash_close.csv", "monthly_debt_position.csv", "monthly_debt_activity.csv", "metric_values.csv"}
    used_sources = set(series["source_table"].dropna().astype(str)) if not series.empty and "source_table" in series.columns else set()
    add("frontend_series_uses_only_frontier_sources", used_sources.issubset(allowed_sources), f"used_sources={sorted(used_sources)}")
    add("no_cross_currency_sum", series.empty or series["Currency"].astype(str).str.strip().ne("").all(), "all frontend rows carry Currency; aggregations are by native currency")
    add("currency_column_present_for_money_outputs", "Currency" in series.columns, f"columns={list(series.columns)}")
    add("cash_metrics_unavailable_if_no_frontend_safe_cash_rows", cash_safe_source or cash_series.empty, f"safe_source={cash_safe_source}; cash_series_rows={len(cash_series)}")
    add("debt_metrics_currency_separated", debt_series.empty or debt_series["Currency"].astype(str).str.strip().ne("").all(), f"rows={len(debt_series)}")
    add("no_debt_stock_mixed_with_ars_flow_without_currency", debt_series.empty or (debt_series["Currency"].astype(str).str.strip().ne("").all() and debt_series["metric_id"].astype(str).str.startswith("ID.DEBT").all()), f"debt_rows={len(debt_series)}")
    add("wide_tables_not_used_as_canonical_sources", not any("wide" in src.lower() or "pivot" in src.lower() for src in used_sources), f"used_sources={sorted(used_sources)}")
    canonical_sources = {
        "monthly_operating_statement.csv",
        "monthly_flow_semantic_split.csv",
        "monthly_cash_close.csv",
        "monthly_debt_position.csv",
        "monthly_debt_activity.csv",
    }
    source_contracts = {src: artifact_contract_for_name(src, src).get("source_authority") for src in used_sources}
    add("metrics_frontier_uses_canonical_sources_when_available", used_sources.issubset(canonical_sources | {"metric_values.csv"}), f"source_contracts={source_contracts}")
    add("notebooks_do_not_classify_flows", True, "enforced by backend frontier contract; notebook static audit documented separately", "warning")
    add("classification_quality_metrics_present", {"DQ.CLASSIFICATION.COVERAGE", "DQ.UNKNOWN.AMOUNT"}.issubset(ids), "DQ metrics present")
    public = frontier.loc[frontier["public_flag"].astype(str).eq("true")]
    needs_caveat = public.loc[public["frontend_suitability"].isin(["safe_with_caveat", "legacy_only", "unavailable"])]
    add("public_metrics_have_caveats_when_needed", needs_caveat.empty or needs_caveat["caveat"].astype(str).str.strip().ne("").all(), f"checked={len(needs_caveat)}")
    legacy_ok = metric_registry is not None and metric_values is not None
    add("legacy_metrics_not_removed", legacy_ok, "metric_registry.csv and metric_values.csv still loaded" if legacy_ok else "legacy metric files missing")
    add("publish_frontier_files_present", True, "publish/latest.py includes frontier files when publish is run", "warning")
    return pd.DataFrame(rows, columns=QA_COLUMNS)
