from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.professional.drilldown import build_professional_flow_drilldowns
from accounting.professional.render_linked_digest import build_professional_linked_digest


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_professional_flow_drilldown_reconciles_and_links(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "Lugar": "", "actor": "Property Management", "counterparty": "Vendor", "payer": "PM", "receiver": "Vendor", "channel": "", "cash_path": "Pagos:Servicios", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_in": 0, "amount_out": 100, "net_amount": -100, "amount_abs": 100, "n_tx": 1, "classification_status": "classified", "classification_confidence": "high", "review_required": False, "source_table": "ledger_canonical.csv", "source_tx_ids_sample": "tx1", "rule_ids": "R003", "notes": ""},
    ])
    _write(run / "classification_audit.csv", [
        {"tx_id": "tx1", "period": "2026-01", "Currency": "ARS", "amount": 100, "Box": "Property Management", "payer": "PM", "receiver": "Vendor", "actor": "Property Management", "counterparty": "Vendor", "cash_path": "Pagos:Servicios", "semantic_bucket": "property_opex", "semantic_subbucket": "services"},
    ])
    _write(tables / "monthly_tables_flow_subbucket_all_measures.csv", [
        {"measure": "amount_out", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "2026-01": 100},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    assert len(index) == 1
    assert index.iloc[0]["status"] == "ok"
    assert index.iloc[0]["matched_rows"] == 1
    assert abs(float(index.iloc[0]["residual"])) <= 1e-6
    assert (pack / index.iloc[0]["detail_csv_relpath"]).exists()
    assert (pack / index.iloc[0]["detail_html_relpath"]).exists()

    digest = build_professional_linked_digest(repo, pack)
    text = digest.read_text(encoding="utf-8")
    assert "class='drilldown'" in text
    assert "monthly_tables_flow_subbucket_all_measures" in text


def test_professional_flow_residual_warning_not_linked(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_in": 0, "amount_out": 90, "net_amount": -90, "amount_abs": 90, "n_tx": 1, "source_tx_ids_sample": "tx1"},
    ])
    _write(tables / "monthly_tables_flow_subbucket_all_measures.csv", [
        {"measure": "amount_out", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "2026-01": 100},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    assert index.iloc[0]["status"] == "residual_warning"

    digest = build_professional_linked_digest(repo, pack)
    assert "class='drilldown'" not in digest.read_text(encoding="utf-8")


def test_acceptance_scope_major_cells_have_details_and_caveats(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    split_rows = [
        {"period": "2026-01", "Currency": "ARS", "Box": "Household", "semantic_bucket": "family_withdrawal_candidate", "semantic_subbucket": "personal_expense", "amount_in": 0, "amount_out": 20, "net_amount": -20, "amount_abs": 20, "n_tx": 1, "source_tx_ids_sample": "draw1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Family Business", "payer": "Tenant", "receiver": "FB", "actor": "Family Business", "counterparty": "Tenant", "semantic_bucket": "operating_revenue", "semantic_subbucket": "rent", "amount_in": 70, "amount_out": 0, "net_amount": 70, "amount_abs": 70, "n_tx": 1, "source_tx_ids_sample": "fb1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_in": 0, "amount_out": 100, "net_amount": -100, "amount_abs": 100, "n_tx": 1, "source_tx_ids_sample": "pm1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Household", "semantic_bucket": "funding_contribution", "semantic_subbucket": "family_or_tenant_contribution", "amount_in": 55, "amount_out": 0, "net_amount": 55, "amount_abs": 55, "n_tx": 1, "source_tx_ids_sample": "hh1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "taxes", "amount_in": 0, "amount_out": 33, "net_amount": -33, "amount_abs": 33, "n_tx": 1, "source_tx_ids_sample": "opex1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "cash_path": "Cambio:FX", "payer": "FX", "receiver": "PM", "semantic_bucket": "treasury_fx", "semantic_subbucket": "fx_conversion_proceeds", "amount_in": 200, "amount_out": 0, "net_amount": 200, "amount_abs": 200, "n_tx": 1, "source_tx_ids_sample": "fx1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "unknown", "semantic_subbucket": "review_required", "amount_in": 0, "amount_out": 0, "net_amount": -7, "amount_abs": 7, "n_tx": 1, "source_tx_ids_sample": "unk1"},
    ]
    _write(run / "monthly_flow_semantic_split.csv", split_rows)
    _write(run / "classification_audit.csv", [
        {"tx_id": r["source_tx_ids_sample"], "period": r["period"], "Currency": r["Currency"], "amount": r.get("amount_in", 0) or r.get("amount_out", 0) or r.get("net_amount", 0), "Box": r["Box"], "payer": r.get("payer", ""), "receiver": r.get("receiver", ""), "actor": r.get("actor", r["Box"]), "counterparty": r.get("counterparty", ""), "cash_path": r.get("cash_path", ""), "semantic_bucket": r["semantic_bucket"], "semantic_subbucket": r["semantic_subbucket"]}
        for r in split_rows
    ])
    _write(tables / "monthly_tables_draws_by_box_amount_out.csv", [{"Currency": "ARS", "Box": "Household", "2026-01": 20}])
    _write(tables / "monthly_tables_fb_bridge_matrix.csv", [{"Currency": "ARS", "metric": "rent_or_revenue_in", "2026-01": 70}])
    _write(tables / "monthly_tables_pm_stress_matrix.csv", [{"Currency": "ARS", "metric": "property_opex_out", "2026-01": 133}])
    _write(tables / "monthly_tables_household_bridge_matrix.csv", [{"Currency": "ARS", "metric": "funding_in", "2026-01": 55}])
    _write(tables / "monthly_tables_opex_by_type_amount_out.csv", [{"Currency": "ARS", "Box": "Property Management", "semantic_subbucket": "taxes", "2026-01": 33}])
    _write(tables / "monthly_tables_fx_treasury_compact.csv", [{"measure": "amount_in", "Currency": "ARS", "2026-01": 200}])
    _write(tables / "monthly_tables_unknown_review_net_matrix.csv", [{"Currency": "ARS", "2026-01": -7}])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    expected_tables = {
        "monthly_tables_draws_by_box_amount_out",
        "monthly_tables_fb_bridge_matrix",
        "monthly_tables_pm_stress_matrix",
        "monthly_tables_household_bridge_matrix",
        "monthly_tables_opex_by_type_amount_out",
        "monthly_tables_fx_treasury_compact",
        "monthly_tables_unknown_review_net_matrix",
    }
    ok = index[index["status"].eq("ok")]
    assert expected_tables.issubset(set(ok["table_id"]))
    for _, row in ok.iterrows():
        assert (pack / row["detail_csv_relpath"]).exists()
        html_path = pack / row["detail_html_relpath"]
        assert html_path.exists()
        text = html_path.read_text(encoding="utf-8")
        assert "Displayed value" in text
        assert "Matched sum" in text
        assert "Residual" in text
        assert "Filters" in text
        assert "Source artifact" in text
        assert "Relevant rows" in text
    fb = ok[ok["table_id"].eq("monthly_tables_fb_bridge_matrix")].iloc[0]
    assert "FB-related" in fb["caveat"]

    digest = build_professional_linked_digest(repo, pack)
    text = digest.read_text(encoding="utf-8")
    assert digest.exists()
    assert text.count("class='drilldown'") >= len(expected_tables)


def test_missing_currency_is_unsupported_to_prevent_cross_currency_sum(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"
    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_in": 0, "amount_out": 90, "net_amount": -90, "amount_abs": 90, "n_tx": 1},
        {"period": "2026-01", "Currency": "USD", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_in": 0, "amount_out": 10, "net_amount": -10, "amount_abs": 10, "n_tx": 1},
    ])
    _write(tables / "monthly_tables_flow_subbucket_all_measures.csv", [
        {"measure": "amount_out", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "2026-01": 100},
    ])
    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    assert index.iloc[0]["status"] == "unsupported"
    assert "cross-currency" in index.iloc[0]["filter_json"]


def test_derived_statement_and_annual_drilldowns_ignore_caller_box_scope(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BOXES", "Household")
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "Lugar": "CABA", "semantic_bucket": "operating_revenue", "semantic_subbucket": "rent", "amount_in": 120, "amount_out": 0, "net_amount": 120, "amount_abs": 120, "n_tx": 1, "source_tx_ids_sample": "rent1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "Lugar": "CABA", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_in": 0, "amount_out": 30, "net_amount": -30, "amount_abs": 30, "n_tx": 1, "source_tx_ids_sample": "opex1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "treasury_fx", "semantic_subbucket": "fx_conversion_proceeds", "cash_path": "Cambio:FX", "amount_in": 15, "amount_out": 0, "net_amount": 15, "amount_abs": 15, "n_tx": 1, "source_tx_ids_sample": "fx1"},
    ])
    _write(run / "classification_audit.csv", [
        {"tx_id": "rent1", "period": "2026-01", "Currency": "ARS", "amount": 120, "Box": "Property Management", "semantic_bucket": "operating_revenue", "semantic_subbucket": "rent"},
        {"tx_id": "opex1", "period": "2026-01", "Currency": "ARS", "amount": 30, "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services"},
        {"tx_id": "fx1", "period": "2026-01", "Currency": "ARS", "amount": 15, "Box": "Property Management", "semantic_bucket": "treasury_fx", "semantic_subbucket": "fx_conversion_proceeds", "cash_path": "Cambio:FX"},
    ])
    _write(run / "monthly_operating_statement.csv", [
        {"period": "2026-01", "Currency": "ARS", "statement_line": "operating_revenue", "amount": 120, "source_table": "monthly_flow_semantic_split.csv", "source_filter": "semantic_bucket=operating_revenue"},
        {"period": "2026-01", "Currency": "ARS", "statement_line": "property_opex_true", "amount": 30, "source_table": "monthly_flow_semantic_split.csv", "source_filter": "semantic_bucket=property_opex"},
        {"period": "2026-01", "Currency": "ARS", "statement_line": "net_operating", "amount": 90, "source_table": "monthly_flow_semantic_split.csv", "source_filter": "operating_revenue - property_opex_true"},
    ])
    _write(run / "annual_balance_dashboard_metrics.csv", [
        {"metric_id": "IS.REVENUE.OPERATING", "period": "2026", "Currency": "ARS", "value": 120, "flow_type": "flow", "source_table": "monthly_operating_statement.csv", "source_filter": "statement_line=operating_revenue", "calculation_rule": "annual flow = sum monthly flow by year and currency"},
        {"metric_id": "IS.RENT.BY_PROPERTY", "period": "2026", "Currency": "ARS", "value": 120, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "source_filter": "semantic_bucket=operating_revenue; semantic_subbucket=rent", "calculation_rule": "annual flow = sum monthly flow by year, currency, and dimension", "dimension_name": "Lugar", "dimension_value": "CABA"},
        {"metric_id": "BS.CASH.TOTAL", "period": "2026", "Currency": "ARS", "value": 999, "flow_type": "stock", "source_table": "monthly_cash_close.csv", "source_filter": "is_frontend_safe=true", "calculation_rule": "annual stock"},
    ])

    _write(tables / "monthly_tables_operating_statement_matrix.csv", [
        {"Currency": "ARS", "statement_line": "operating_revenue", "2026-01": 120},
        {"Currency": "ARS", "statement_line": "net_operating", "2026-01": 90},
    ])
    _write(tables / "monthly_tables_operating_statement_matrix_ars.csv", [
        {"Currency": "ARS", "statement_line": "operating_revenue", "2026-01": 120},
    ])
    _write(tables / "overview_balance_dashboard.csv", [
        {"Currency": "ARS", "metric_id": "IS.REVENUE.OPERATING", "2026": 120},
        {"Currency": "ARS", "metric_id": "BS.CASH.TOTAL", "2026": 999},
    ])
    _write(tables / "income_operating_statement.csv", [
        {"Currency": "ARS", "metric_id": "IS.RENT.BY_PROPERTY", "dimension_name": "Lugar", "dimension_value": "CABA", "2026": 120},
    ])
    _write(tables / "cash_annual_box_flow_bridge_wide.csv", [
        {"Currency": "ARS", "Box": "Property Management", "semantic_bucket": "treasury_fx", "measure": "net_amount", "line": "fx_flow_bridge", "2026": 15},
        {"Currency": "ARS", "Box": "Property Management", "line": "cash close", "2026": 999},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    ok_tables = set(index[index["status"].eq("ok")]["table_id"])
    assert "monthly_tables_operating_statement_matrix" in ok_tables
    assert "monthly_tables_operating_statement_matrix_ars" in ok_tables
    assert "overview_balance_dashboard" in ok_tables
    assert "income_operating_statement" in ok_tables
    assert "cash_annual_box_flow_bridge_wide" in ok_tables
    cash_stock = index[index["filter_json"].str.contains("cash/stock", na=False)]
    assert not cash_stock.empty
    assert set(cash_stock["status"]) == {"unsupported"}

    net = index[index["filter_json"].str.contains("net_operating", na=False)].iloc[0]
    net_html = (pack / net["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "Formula" in net_html
    assert "Component rows" in net_html
    annual = index[index["table_id"].eq("overview_balance_dashboard") & index["status"].eq("ok")].iloc[0]
    annual_html = (pack / annual["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "Annual metric row" in annual_html
    assert "Monthly source rows" in annual_html

    digest = build_professional_linked_digest(repo, pack)
    text = digest.read_text(encoding="utf-8")
    assert "monthly_tables_operating_statement_matrix" in text
    assert text.count("class='drilldown'") >= 5


def test_funding_drilldowns_use_stable_metric_contracts_and_debt_evidence(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    split_rows = [
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "funding_contribution",
            "semantic_subbucket": "tenant_cash_support",
            "funding_actor": "Inquilino",
            "funding_channel": "tenant_to_box",
            "target_box": "Property Management",
            "cash_effect": "cash_in_box",
            "debt_effect": "none",
            "amount_in": 50,
            "amount_out": 0,
            "net_amount": 50,
            "amount_abs": 50,
            "source_tx_ids_sample": "tenant_cash",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "property_opex",
            "semantic_subbucket": "taxes",
            "funding_actor": "Inquilino",
            "funding_channel": "tenant_direct_tax_payment",
            "target_box": "Property Management",
            "obligation_box": "Property Management",
            "cash_effect": "no_cash_in_box_direct_payment",
            "debt_effect": "none",
            "amount_in": 0,
            "amount_out": 30,
            "net_amount": -30,
            "amount_abs": 30,
            "source_tx_ids_sample": "tenant_tax",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Family Business",
            "semantic_bucket": "funding_contribution",
            "semantic_subbucket": "fb_support",
            "funding_actor": "Alejandro",
            "funding_channel": "family_business_contribution",
            "target_box": "Family Business",
            "cash_effect": "cash_in_box",
            "debt_effect": "none",
            "amount_in": 70,
            "amount_out": 0,
            "net_amount": 70,
            "amount_abs": 70,
            "source_tx_ids_sample": "ale_fb",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Family Business",
            "semantic_bucket": "funding_contribution",
            "semantic_subbucket": "fb_support",
            "funding_actor": "Primos",
            "funding_channel": "family_business_contribution",
            "target_box": "Family Business",
            "cash_effect": "cash_in_box",
            "debt_effect": "none",
            "amount_in": 40,
            "amount_out": 0,
            "net_amount": 40,
            "amount_abs": 40,
            "source_tx_ids_sample": "primos_fb",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "funding_contribution",
            "semantic_subbucket": "household_pm_support",
            "funding_actor": "Household",
            "funding_channel": "household_to_pm",
            "source_box": "Household",
            "target_box": "Property Management",
            "cash_effect": "cash_in_box",
            "debt_effect": "none",
            "amount_in": 20,
            "amount_out": 0,
            "net_amount": 20,
            "amount_abs": 20,
            "source_tx_ids_sample": "hh_pm",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "debt_movement",
            "semantic_subbucket": "principal",
            "funding_actor": "Matías",
            "funding_channel": "debt_creation",
            "target_box": "Property Management",
            "cash_effect": "cash_in_box",
            "debt_effect": "creates_debt",
            "linked_debt_id": "D-1",
            "amount_in": 90,
            "amount_out": 0,
            "net_amount": 90,
            "amount_abs": 90,
            "source_tx_ids_sample": "matias_debt",
        },
    ]
    _write(run / "monthly_flow_semantic_split.csv", split_rows)
    _write(run / "classification_audit.csv", [
        {
            "tx_id": row["source_tx_ids_sample"],
            "period": row["period"],
            "Currency": row["Currency"],
            "Box": row["Box"],
            "semantic_bucket": row["semantic_bucket"],
            "semantic_subbucket": row["semantic_subbucket"],
            "funding_actor": row.get("funding_actor", ""),
            "funding_channel": row.get("funding_channel", ""),
            "target_box": row.get("target_box", ""),
            "obligation_box": row.get("obligation_box", ""),
            "cash_effect": row.get("cash_effect", ""),
            "debt_effect": row.get("debt_effect", ""),
        }
        for row in split_rows
    ])
    _write(run / "monthly_debt_activity.csv", [
        {"period": "2026-01", "Currency": "ARS", "linked_debt_id": "D-1", "debtor": "Property Management", "creditor": "Matías", "new_principal": 90},
    ])
    _write(run / "monthly_debt_position.csv", [
        {"period": "2026-01", "Currency": "ARS", "linked_debt_id": "D-1", "debtor": "Property Management", "creditor": "Matías", "open_amount": 90},
    ])
    _write(run / "annual_balance_dashboard_metrics.csv", [
        {"metric_id": "FUND.CONTRIB.BY_CHANNEL", "period": "2026", "Currency": "ARS", "value": 30, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "dimension_name": "funding_channel", "dimension_value": "tenant_direct_tax_payment"},
        {"metric_id": "FUND.CONTRIB.BY_CHANNEL", "period": "2026", "Currency": "ARS", "value": 50, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "dimension_name": "funding_channel", "dimension_value": "tenant_to_box"},
        {"metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR", "period": "2026", "Currency": "ARS", "value": 70, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "dimension_name": "funding_actor", "dimension_value": "Alejandro"},
        {"metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR", "period": "2026", "Currency": "ARS", "value": 40, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "dimension_name": "funding_actor", "dimension_value": "Primos"},
        {"metric_id": "FUND.CONTRIB.BY_CHANNEL", "period": "2026", "Currency": "ARS", "value": 20, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "dimension_name": "funding_channel", "dimension_value": "household_to_pm"},
        {"metric_id": "FUND.CONTRIB.DEBT_LINKED", "period": "2026", "Currency": "ARS", "value": 90, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv"},
        {"metric_id": "ID.DEBT.TOTAL.OPEN", "period": "2026", "Currency": "ARS", "value": 90, "flow_type": "stock", "source_table": "monthly_debt_position.csv"},
    ])
    _write(tables / "overview_balance_dashboard.csv", [
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.BY_CHANNEL", "dimension_name": "funding_channel", "dimension_value": "tenant_direct_tax_payment", "2026": 30},
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.BY_CHANNEL", "dimension_name": "funding_channel", "dimension_value": "tenant_to_box", "2026": 50},
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR", "dimension_name": "funding_actor", "dimension_value": "Alejandro", "2026": 70},
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR", "dimension_name": "funding_actor", "dimension_value": "Primos", "2026": 40},
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.BY_CHANNEL", "dimension_name": "funding_channel", "dimension_value": "household_to_pm", "2026": 20},
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.DEBT_LINKED", "2026": 90},
        {"Currency": "ARS", "metric_id": "ID.DEBT.TOTAL.OPEN", "2026": 90},
    ])
    _write(tables / "cash_annual_box_flow_bridge_wide.csv", [
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.BY_CHANNEL", "dimension_name": "funding_channel", "dimension_value": "tenant_direct_tax_payment", "2026": 30},
        {"Currency": "ARS", "metric_id": "FUND.CONTRIB.BY_CHANNEL", "dimension_name": "funding_channel", "dimension_value": "tenant_to_box", "2026": 50},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    ok = index[index["status"].eq("ok")]

    assert len(ok[ok["filter_json"].str.contains("tenant_direct_tax_payment", na=False)]) == 2
    assert len(ok[ok["filter_json"].str.contains("tenant_to_box", na=False)]) == 2
    assert not ok[ok["filter_json"].str.contains("Alejandro", na=False)].empty
    assert not ok[ok["filter_json"].str.contains("Primos", na=False)].empty
    assert not ok[ok["filter_json"].str.contains("household_to_pm", na=False)].empty

    direct = ok[ok["filter_json"].str.contains("tenant_direct_tax_payment", na=False)].iloc[0]
    direct_html = (pack / direct["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "no_cash_in_box_direct_payment" in direct_html

    debt = ok[ok["filter_json"].str.contains("FUND.CONTRIB.DEBT_LINKED", na=False)].iloc[0]
    assert debt["lineage_level"] == "debt_linked_support_with_debt_evidence"
    debt_html = (pack / debt["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "Debt activity rows" in debt_html
    assert "Debt position rows" in debt_html

    debt_stock = index[index["filter_json"].str.contains("monthly_debt_position.csv", na=False)]
    assert not debt_stock.empty
    assert set(debt_stock["status"]) == {"ok"}
    assert set(debt_stock["lineage_level"]) == {"annual_to_monthly_debt_position"}


def test_overview_annual_metric_matching_normalizes_float_periods_and_blank_dimensions(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2023-01", "Currency": "ARS", "semantic_bucket": "funding_contribution", "amount_in": 2006220, "amount_out": 0, "net_amount": 2006220, "source_tx_ids_sample": "fund1"},
    ])
    _write(run / "classification_audit.csv", [
        {"tx_id": "fund1", "period": "2023-01", "Currency": "ARS", "semantic_bucket": "funding_contribution", "amount": 2006220},
    ])
    _write(run / "monthly_debt_position.csv", [
        {"period": "2025-12", "Currency": "USD", "component": "total", "open_amount": 15234.7},
    ])
    _write(run / "annual_balance_dashboard_metrics.csv", [
        {"metric_id": "FUND.CONTRIB.TOTAL", "period": 2023.0, "Currency": "ARS", "value": 2006220, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "dimension_name": "", "dimension_value": ""},
        {"metric_id": "ID.DEBT.TOTAL.OPEN", "period": 2025.0, "Currency": "USD", "value": 15234.7, "flow_type": "stock", "source_table": "monthly_debt_position.csv", "dimension_name": "", "dimension_value": ""},
    ])
    _write(tables / "overview_balance_dashboard.csv", [
        {"label": "Funding / aportes", "Currency": "ARS", "metric_id": "FUND.CONTRIB.TOTAL", "dimension_name": "", "dimension_value": "", "2023": 2006220},
        {"label": "Deuda total abierta", "Currency": "USD", "metric_id": "ID.DEBT.TOTAL.OPEN", "dimension_name": "", "dimension_value": "", "2025": 15234.7},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    overview = index[index["table_id"].eq("overview_balance_dashboard")].copy()

    funding = overview[overview["filter_json"].str.contains("FUND.CONTRIB.TOTAL", na=False)].iloc[0]
    assert funding["status"] == "ok"
    assert funding["matched_value_sum"] == 2006220
    assert funding["residual"] == 0

    debt = overview[overview["filter_json"].str.contains("ID.DEBT.TOTAL.OPEN", na=False)].iloc[0]
    assert debt["status"] == "ok"
    assert debt["matched_value_sum"] == 15234.7
    assert debt["residual"] == 0
    assert debt["lineage_level"] == "annual_to_monthly_debt_position"


def test_fast_mode_skips_tables_over_100_cells(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    rows = [
        {
            "measure": "amount_out",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "property_opex",
            "semantic_subbucket": f"services_{i}",
            "2026-01": 1,
        }
        for i in range(101)
    ]
    _write(tables / "monthly_tables_flow_subbucket_all_measures.csv", rows)

    paths = build_professional_flow_drilldowns(repo, pack, run, fast=True)
    index = pd.read_csv(paths["index"])
    qa = pd.read_csv(paths["qa"])
    manifest = pd.read_json(paths["manifest"], typ="series")

    assert index.empty
    warnings = qa[
        qa["table_id"].eq("monthly_tables_flow_subbucket_all_measures")
        & qa["check"].eq("table_cell_limit")
    ]
    assert len(warnings) == 1
    assert warnings.iloc[0]["status"] == "warning"
    assert "Table has too many cells to afford triggering drilldowns" in warnings.iloc[0]["detail"]
    assert "cells=101" in warnings.iloc[0]["detail"]
    assert "limit=100" in warnings.iloc[0]["detail"]
    assert bool(manifest["fast"]) is True
    assert int(manifest["table_cell_limit"]) == 100


def test_default_mode_skips_tables_over_500_cells(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    rows = [
        {
            "measure": "amount_out",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "property_opex",
            "semantic_subbucket": f"services_{i}",
            "2026-01": 1,
        }
        for i in range(501)
    ]
    _write(tables / "monthly_tables_flow_subbucket_all_measures.csv", rows)

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    qa = pd.read_csv(paths["qa"])
    manifest = pd.read_json(paths["manifest"], typ="series")

    assert index.empty
    warnings = qa[
        qa["table_id"].eq("monthly_tables_flow_subbucket_all_measures")
        & qa["check"].eq("table_cell_limit")
    ]
    assert len(warnings) == 1
    assert warnings.iloc[0]["status"] == "warning"
    assert "Table has too many cells to afford triggering drilldowns" in warnings.iloc[0]["detail"]
    assert "cells=501" in warnings.iloc[0]["detail"]
    assert "limit=500" in warnings.iloc[0]["detail"]
    assert bool(manifest["fast"]) is False
    assert int(manifest["table_cell_limit"]) == 500


def test_annual_human_labels_match_stable_metric_ids(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2023-01", "Currency": "ARS", "semantic_bucket": "funding_contribution", "semantic_subbucket": "owner_contribution", "amount_in": 2006220, "amount_out": 0, "net_amount": 2006220, "amount_abs": 2006220, "n_tx": 1, "source_tx_ids_sample": "fund1"},
    ])
    _write(run / "classification_audit.csv", [
        {"tx_id": "fund1", "period": "2023-01", "Currency": "ARS", "amount": 2006220, "semantic_bucket": "funding_contribution", "semantic_subbucket": "owner_contribution"},
    ])
    _write(run / "annual_balance_dashboard_metrics.csv", [
        {"metric_id": "FUND.CONTRIB.TOTAL", "period": "2023", "Currency": "ARS", "value": 2006220, "flow_type": "flow", "source_table": "monthly_flow_semantic_split.csv", "source_filter": "semantic_bucket=funding_contribution"},
    ])
    _write(tables / "overview_balance_dashboard.csv", [
        {"Currency": "ARS", "metric": "Funding / aportes", "2023": 2006220},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    assert index.iloc[0]["status"] == "ok"
    assert "FUND.CONTRIB.TOTAL" in index.iloc[0]["filter_json"] or "funding_contribution" in index.iloc[0]["filter_json"]
    html = (pack / index.iloc[0]["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "Annual metric row" in html
    assert "Semantic rows" in html


def test_annual_debt_stock_links_to_monthly_debt_position(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_debt_position.csv", [
        {"period": "2026-12", "Currency": "USD", "debtor": "Property Management", "creditor": "Matias", "component": "total", "open_amount": 50, "open_total": 50, "open_principal": 40, "open_interest": 10},
    ])
    _write(run / "annual_balance_dashboard_metrics.csv", [
        {"metric_id": "BS.DEBT.TOTAL.OPEN", "period": "2026", "Currency": "USD", "value": 50, "flow_type": "stock", "source_table": "monthly_debt_position.csv", "source_filter": "component=total"},
    ])
    _write(tables / "overview_balance_dashboard.csv", [
        {"Currency": "USD", "metric_id": "BS.DEBT.TOTAL.OPEN", "metric": "Deuda total abierta", "2026": 50},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    assert index.iloc[0]["status"] == "ok"
    assert index.iloc[0]["lineage_level"] == "annual_to_monthly_debt_position"
    html = (pack / index.iloc[0]["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "Debt position rows" in html


def test_annual_formula_ratio_rows_build_formula_pages(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "annual_balance_dashboard_metrics.csv", [
        {"metric_id": "IS.REVENUE.OPERATING", "period": "2026", "Currency": "ARS", "value": 1000, "flow_type": "flow", "source_table": "monthly_operating_statement.csv"},
        {"metric_id": "IS.OPEX.PROPERTY", "period": "2026", "Currency": "ARS", "value": 250, "flow_type": "flow", "source_table": "monthly_operating_statement.csv"},
        {"metric_id": "IS.NET.OPERATING", "period": "2026", "Currency": "ARS", "value": 750, "flow_type": "flow", "source_table": "monthly_operating_statement.csv"},
        {"metric_id": "DIST.DRAWS.PERSONAL", "period": "2026", "Currency": "ARS", "value": 300, "flow_type": "flow", "source_table": "monthly_operating_statement.csv"},
        {"metric_id": "FUND.CONTRIB.TOTAL", "period": "2026", "Currency": "ARS", "value": 100, "flow_type": "flow", "source_table": "monthly_operating_statement.csv"},
        {"metric_id": "COV.NET.AFTER_DRAWS", "period": "2026", "Currency": "ARS", "value": 550, "flow_type": "mixed", "source_table": "monthly_operating_statement.csv"},
    ])
    _write(tables / "overview_balance_dashboard.csv", [
        {"Currency": "ARS", "metric": "Margen operativo", "2026": 0.75},
        {"Currency": "ARS", "metric": "OPEX / renta", "2026": 0.25},
        {"Currency": "ARS", "metric": "Retiros / resultado operativo", "2026": 0.4},
        {"Currency": "ARS", "metric": "Cobertura después de funding y retiros", "2026": 550},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    assert set(index["status"]) == {"ok"}
    assert set(index["lineage_level"]) == {"annual_formula_components"}
    for relpath in index["detail_html_relpath"]:
        html = (pack / relpath).read_text(encoding="utf-8")
        assert "Formula" in html
        assert "Component annual rows" in html


def test_annual_debt_stock_spanish_labels_match_id_metric_contracts(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_debt_position.csv", [
        {"period": "2026-12", "Currency": "USD", "debtor": "Property Management", "creditor": "Matias", "component": "total", "open_amount": 50, "open_total": 50, "open_principal": 40, "open_interest": 10},
        {"period": "2026-12", "Currency": "USD", "debtor": "Property Management", "creditor": "Matias", "component": "principal", "open_amount": 40, "open_total": 50, "open_principal": 40, "open_interest": 0},
        {"period": "2026-12", "Currency": "USD", "debtor": "Property Management", "creditor": "Matias", "component": "interest", "open_amount": 10, "open_total": 50, "open_principal": 0, "open_interest": 10},
    ])
    _write(run / "annual_balance_dashboard_metrics.csv", [
        {"metric_id": "ID.DEBT.TOTAL.OPEN", "period": "2026", "Currency": "USD", "value": 50, "flow_type": "stock", "source_table": "monthly_debt_position.csv", "source_filter": "component=total"},
        {"metric_id": "ID.DEBT.PRINCIPAL.OPEN", "period": "2026", "Currency": "USD", "value": 40, "flow_type": "stock", "source_table": "monthly_debt_position.csv", "source_filter": "component=principal"},
        {"metric_id": "ID.DEBT.INTEREST.OPEN", "period": "2026", "Currency": "USD", "value": 10, "flow_type": "stock", "source_table": "monthly_debt_position.csv", "source_filter": "component=interest"},
    ])
    _write(tables / "overview_balance_dashboard.csv", [
        {"Currency": "USD", "metric": "Deuda total abierta", "2026": 50},
        {"Currency": "USD", "metric": "Principal abierto", "2026": 40},
        {"Currency": "USD", "metric": "Interés abierto", "2026": 10},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    assert set(index["status"]) == {"ok"}
    assert set(index["lineage_level"]) == {"annual_to_monthly_debt_position"}
    assert set(index["filter_json"].str.extract(r'"component": "([^"]+)"', expand=False)) == {"total", "principal", "interest"}


def test_cash_annual_bridge_net_debt_movement_uses_signed_net_amount(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "debt_movement", "semantic_subbucket": "repayment", "funding_actor": "Matías", "funding_channel": "debt_settlement", "cash_effect": "cash_out_box", "debt_effect": "settles_debt", "amount_in": 0, "amount_out": 100, "net_amount": -100, "amount_abs": 100, "n_tx": 1, "source_tx_ids_sample": "debt-repay"},
        {"period": "2026-02", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "debt_movement", "semantic_subbucket": "principal", "funding_actor": "Matías", "funding_channel": "debt_creation", "cash_effect": "non_cash_support", "debt_effect": "creates_debt", "amount_in": 0, "amount_out": 0, "net_amount": 40, "amount_abs": 40, "n_tx": 1, "source_tx_ids_sample": "debt-create"},
    ])
    _write(tables / "cash_annual_box_flow_bridge_wide.csv", [
        {"Currency": "ARS", "Box": "Property Management", "line": "Movimiento neto de deuda", "metric_id": "FUND.CONTRIB.DEBT_LINKED", "2026": -60},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    row = index.iloc[0]
    assert row["status"] == "ok"
    assert row["measure"] == "net_amount"
    assert row["matched_value_sum"] == -60
    assert row["residual"] == 0
    assert "movimiento_neto_deuda" in row["filter_json"]


def test_diagnostic_box_level_uses_existing_previous_close_without_box_flag(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_cash_close.csv", [
        {"period": "2024-03", "Currency": "ARS", "Box": "Property Management", "close_amount": 180000},
        {"period": "2024-04", "Currency": "ARS", "Box": "Property Management", "close_amount": 180030, "source_table": "box_balance_time_long.freq=M.csv"},
        {"period": "2024-02", "Currency": "USD", "Box": "Property Management", "close_amount": -210},
        {"period": "2024-03", "Currency": "USD", "Box": "Property Management", "close_amount": -20, "source_table": "box_balance_time_long.freq=M.csv"},
    ])
    _write(tables / "monthly_tables_diagnostic_box_level_matrix.csv", [
        {"Currency": "ARS", "Box": "Property Management", "metric": "diagnostic_box_level", "2024-04": 30},
        {"Currency": "USD", "Box": "Property Management", "metric": "diagnostic_box_level", "2024-03": 190},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    assert set(index["status"]) == {"ok"}
    assert set(index["matched_value_sum"]) == {30.0, 190.0}
    assert set(index["residual"]) == {0.0}
    for relpath in index["detail_html_relpath"]:
        html = (pack / relpath).read_text(encoding="utf-8")
        assert "box_level_fallback_reason" in html

def test_debt_position_drilldown_uses_latest_snapshot_not_month_sum(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_debt_position.csv", [
        {"period": "2025-03", "as_of_date": "2025-03-19", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 8804.2, "open_principal": 8804.2, "open_interest": 104.0, "open_total": 8908.2},
        {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 8726.2, "open_principal": 8726.2, "open_interest": 0.0, "open_total": 8726.2},
        {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "Alejandro", "creditor": "PM", "component": "principal", "open_amount": 100.0, "open_principal": 100.0, "open_interest": 0.0, "open_total": 100.0},
        {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "Hector", "creditor": "MI", "component": "principal", "open_amount": 200.0, "open_principal": 200.0, "open_interest": 0.0, "open_total": 200.0},
        {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "PM", "creditor": "Primos", "component": "principal", "open_amount": 300.0, "open_principal": 300.0, "open_interest": 0.0, "open_total": 300.0},
    ])
    _write(tables / "monthly_tables_debt_position_matrix.csv", [
        {"measure": "open_principal", "Currency": "USD", "pair": "PM → MI", "metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR", "dimension_name": "funding_actor", "dimension_value": "Matías", "funding_actor": "Matías", "2025-03": 8726.2},
        {"measure": "open_principal", "Currency": "USD", "pair": "Alejandro → PM", "2025-03": 100.0},
        {"measure": "open_principal", "Currency": "USD", "pair": "Hector → MI", "2025-03": 200.0},
        {"measure": "open_principal", "Currency": "USD", "pair": "PM → Primos", "2025-03": 300.0},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    debt_rows = index[index["table_id"].eq("monthly_tables_debt_position_matrix")].copy()
    row = debt_rows[debt_rows["filter_json"].str.contains('"pair": "PM → MI"', na=False)].iloc[0]
    assert row["status"] == "ok"
    assert float(row["matched_value_sum"]) == 8726.2
    assert int(row["matched_rows"]) == 1
    assert float(row["residual"]) == 0.0
    assert "FUND.CONTRIB" not in row["filter_json"]
    assert "funding_actor" not in row["filter_json"]
    assert "Matías" not in row["filter_json"]

    detail_csv = (pack / row["detail_csv_relpath"]).read_text(encoding="utf-8")
    detail_html = (pack / row["detail_html_relpath"]).read_text(encoding="utf-8")
    assert "2025-03-31" in detail_csv
    assert "2025-03-19" not in detail_csv
    assert "Selected monthly close snapshot" in detail_html
    assert "All candidate snapshots in period" in detail_html
    assert "2025-03-19" in detail_html

    singles = debt_rows[debt_rows["filter_json"].str.contains("Alejandro → PM|Hector → MI|PM → Primos", na=False)]
    assert len(singles) == 3
    assert set(singles["status"]) == {"ok"}
    assert set(singles["matched_rows"].astype(int)) == {1}
    assert singles["residual"].astype(float).abs().sum() == 0.0


def test_annual_companion_tables_have_drilldowns_and_digest_section(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_cash_close.csv", [
        {"period": "2026-12", "Currency": "ARS", "Box": "Property Management", "close_amount": 10},
    ])
    _write(run / "monthly_flow_semantic_split.csv", [
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "funding_contribution", "funding_actor": "Tenants", "funding_channel": "tenant_to_box", "cash_effect": "cash_in_box", "target_box": "Property Management", "amount_in": 50, "amount_abs": 50, "net_amount": 50, "source_tx_ids_sample": "fund1"},
    ])
    _write(run / "classification_audit.csv", [
        {"tx_id": "fund1", "period": "2026-01", "Currency": "ARS", "semantic_bucket": "funding_contribution", "funding_actor": "Tenants", "funding_channel": "tenant_to_box", "cash_effect": "cash_in_box", "amount": 50},
    ])
    _write(run / "monthly_debt_position.csv", [
        {"period": "2026-12", "as_of_date": "2026-12-31", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "open_principal": 70, "open_interest": 7, "open_total": 77},
    ])
    _write(run / "monthly_debt_activity.csv", [
        {"period": "2026-01", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "repayments": 10, "new_principal": 0, "net_change": -10},
        {"period": "2026-02", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "repayments": 15, "new_principal": 0, "net_change": -15},
    ])

    _write(tables / "annual_cash_close_by_box_wide.csv", [
        {"metric_id": "CASH.CLOSE.BY_BOX", "line_id": "cash.pm", "Box": "Property Management", "Currency": "ARS", "2026": 10},
    ])
    _write(tables / "annual_funding_by_actor_channel_wide.csv", [
        {"metric_id": "FUND.CONTRIB.BY_CHANNEL", "line_id": "fund.tenant", "Currency": "ARS", "funding_actor": "Tenants", "funding_channel": "tenant_to_box", "cash_effect": "cash_in_box", "target_box": "Property Management", "beneficiary_box": "", "obligation_box": "", "2026": 50},
    ])
    _write(tables / "annual_debt_stock_by_pair_wide.csv", [
        {"metric_id": "DEBT.STOCK.BY_PAIR.OPEN_TOTAL", "line_id": "debt.stock", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "component": "open_total", "2026": 77},
    ])
    _write(tables / "annual_debt_activity_by_pair_wide.csv", [
        {"metric_id": "DEBT.ACTIVITY.REPAYMENT.BY_PAIR", "line_id": "debt.repay", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "activity_type": "repayments", "2026": 25},
    ])

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    expected = {
        "annual_cash_close_by_box_wide",
        "annual_funding_by_actor_channel_wide",
        "annual_debt_stock_by_pair_wide",
        "annual_debt_activity_by_pair_wide",
    }
    ok = index[index["status"].eq("ok")]
    assert expected.issubset(set(ok["table_id"]))

    digest = build_professional_linked_digest(repo, pack)
    text = digest.read_text(encoding="utf-8")
    assert "Annual management companion tables" in text
    for table_id in expected:
        assert table_id in text
