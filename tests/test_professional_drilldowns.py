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
    _write(tables / "monthly_tables_fx_treasury_compact.csv", [{"Currency": "ARS", "2026-01": 200}])
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


def test_derived_statement_and_annual_drilldowns_link_supported_flows(tmp_path: Path) -> None:
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
