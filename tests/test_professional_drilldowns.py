from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from accounting.professional.drilldown import build_professional_flow_drilldowns
from accounting.professional.render_linked_digest import build_professional_linked_digest


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _cash_row(*, period: str, amount: float, account_id: str) -> dict[str, object]:
    as_of_date = str(pd.Period(period, freq="M").end_time.date())
    return {
        "period": period,
        "period_end": as_of_date,
        "as_of_date": as_of_date,
        "Box": "Property Management",
        "party": "",
        "account_id": account_id,
        "account_name": account_id,
        "Currency": "ARS",
        "close_amount": amount,
        "source_table": "validated_cash_close.csv",
        "source_date": as_of_date,
        "source_type": "bank_statement",
        "source_reference": "professional-integration-fixture",
        "validation_status": "validated",
        "validated_by": "controller",
        "position_type": "cash_close",
        "cash_suitability": "frontend_safe",
        "is_frontend_safe": True,
        "caveat": "synthetic fixture",
        "notes": "",
        "n_source_rows": 1,
        "calculation_rule": "fixture",
    }


def test_professional_flow_drilldown_reconciles_and_links(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "",
                "actor": "Property Management",
                "counterparty": "Vendor",
                "payer": "PM",
                "receiver": "Vendor",
                "cash_path": "Pagos:Servicios",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "amount_in": 0,
                "amount_out": 100,
                "net_amount": -100,
                "amount_abs": 100,
                "n_tx": 1,
                "source_tx_ids_sample": "tx1",
            }
        ],
    )
    _write(
        run / "classification_audit.csv",
        [
            {
                "tx_id": "tx1",
                "period": "2026-01",
                "Currency": "ARS",
                "amount": 100,
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
            }
        ],
    )
    _write(
        tables / "monthly_tables_flow_subbucket_all_measures.csv",
        [
            {
                "measure": "amount_out",
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "2026-01": 100,
            }
        ],
    )

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


def test_professional_flow_residual_warning_is_not_linked(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "amount_in": 0,
                "amount_out": 90,
                "net_amount": -90,
                "amount_abs": 90,
                "n_tx": 1,
                "source_tx_ids_sample": "tx1",
            }
        ],
    )
    _write(
        tables / "monthly_tables_flow_subbucket_all_measures.csv",
        [
            {
                "measure": "amount_out",
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "2026-01": 100,
            }
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    assert index.iloc[0]["status"] == "residual_warning"
    digest = build_professional_linked_digest(repo, pack)
    assert "class='drilldown'" not in digest.read_text(encoding="utf-8")


def test_current_mixed_surface_writes_details_without_semantic_leakage(tmp_path: Path) -> None:
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
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "cash_path": "Cambio:FX", "semantic_bucket": "treasury_fx", "semantic_subbucket": "fx_conversion_proceeds", "amount_in": 200, "amount_out": 0, "net_amount": 200, "amount_abs": 200, "n_tx": 1, "source_tx_ids_sample": "fx1"},
        {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "unknown", "semantic_subbucket": "review_required", "amount_in": 0, "amount_out": 0, "net_amount": -7, "amount_abs": 7, "n_tx": 1, "source_tx_ids_sample": "unk1"},
    ]
    _write(run / "monthly_flow_semantic_split.csv", split_rows)
    _write(
        run / "classification_audit.csv",
        [
            {
                "tx_id": row["source_tx_ids_sample"],
                "period": row["period"],
                "Currency": row["Currency"],
                "amount": row.get("amount_in", 0) or row.get("amount_out", 0) or row.get("net_amount", 0),
                "Box": row["Box"],
                "semantic_bucket": row["semantic_bucket"],
                "semantic_subbucket": row["semantic_subbucket"],
            }
            for row in split_rows
        ],
    )
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
        html = (pack / row["detail_html_relpath"]).read_text(encoding="utf-8")
        assert "Displayed value" in html
        assert "Matched sum" in html
        assert "Residual" in html
        assert "Source artifact" in html


def test_missing_currency_fails_closed_to_prevent_cross_currency_sum(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {"period": "2026-01", "Currency": "ARS", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_out": 90, "net_amount": -90, "amount_abs": 90},
            {"period": "2026-01", "Currency": "USD", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "amount_out": 10, "net_amount": -10, "amount_abs": 10},
        ],
    )
    _write(
        tables / "monthly_tables_flow_subbucket_all_measures.csv",
        [
            {"measure": "amount_out", "Box": "Property Management", "semantic_bucket": "property_opex", "semantic_subbucket": "services", "2026-01": 100}
        ],
    )
    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    assert index.iloc[0]["status"] == "unsupported"
    assert "cross-currency" in index.iloc[0]["filter_json"]


@pytest.mark.parametrize(
    ("fast", "row_count", "limit"),
    [(True, 101, 100), (False, 501, 500)],
)
def test_table_size_guards_are_preserved(
    tmp_path: Path, fast: bool, row_count: int, limit: int
) -> None:
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
            "semantic_subbucket": f"services_{idx}",
            "2026-01": 1,
        }
        for idx in range(row_count)
    ]
    _write(tables / "monthly_tables_flow_subbucket_all_measures.csv", rows)

    paths = build_professional_flow_drilldowns(repo, pack, run, fast=fast)
    index = pd.read_csv(paths["index"])
    qa = pd.read_csv(paths["qa"])
    manifest = pd.read_json(paths["manifest"], typ="series")
    assert index.empty
    warning = qa[
        qa["table_id"].eq("monthly_tables_flow_subbucket_all_measures")
        & qa["check"].eq("table_cell_limit")
    ].iloc[0]
    assert warning["status"] == "warning"
    assert f"cells={row_count}" in warning["detail"]
    assert f"limit={limit}" in warning["detail"]
    assert bool(manifest["fast"]) is fast
    assert int(manifest["table_cell_limit"]) == limit


def test_debt_position_drilldown_uses_latest_valid_snapshot_not_month_sum(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_debt_position.csv",
        [
            {"period": "2025-03", "as_of_date": "2025-03-19", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_principal": 8804.2, "open_interest": 104.0, "open_total": 8908.2},
            {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_principal": 8726.2, "open_interest": 0.0, "open_total": 8726.2},
        ],
    )
    _write(
        tables / "monthly_tables_debt_position_matrix.csv",
        [{"measure": "open_principal", "Currency": "USD", "pair": "PM → MI", "2025-03": 8726.2}],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    row = index[index["table_id"].eq("monthly_tables_debt_position_matrix")].iloc[0]
    assert row["status"] == "ok"
    assert float(row["matched_value_sum"]) == 8726.2
    assert int(row["matched_rows"]) == 1
    assert float(row["residual"]) == 0.0
    detail = pd.read_csv(pack / row["detail_csv_relpath"])
    assert list(detail["as_of_date"]) == ["2025-03-31"]


def test_diagnostic_box_level_presentation_route_is_retired(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_cash_close.csv",
        [
            {"period": "2024-03", "Currency": "ARS", "Box": "Property Management", "close_amount": 180000},
            {"period": "2024-04", "Currency": "ARS", "Box": "Property Management", "close_amount": 180030, "source_table": "box_balance_time_long.freq=M.csv"},
        ],
    )
    _write(
        tables / "monthly_tables_diagnostic_box_level_matrix.csv",
        [{"Currency": "ARS", "Box": "Property Management", "metric": "diagnostic_box_level", "2024-04": 30}],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    assert "monthly_tables_diagnostic_box_level_matrix" not in set(index["table_id"])


def test_current_annual_companion_tables_reconcile_and_link(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(run / "monthly_cash_close.csv", [_cash_row(period="2026-12", amount=10, account_id="bank-a")])
    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "funding_contribution",
                "semantic_subbucket": "family_or_tenant_contribution",
                "funding_actor": "Tenants",
                "funding_channel": "tenant_to_box",
                "cash_effect": "cash_in_box",
                "target_box": "Property Management",
                "debt_effect": "none",
                "amount_in": 50,
                "amount_out": 0,
                "amount_abs": 50,
                "net_amount": 50,
                "source_tx_ids_sample": "fund1",
            }
        ],
    )
    _write(
        run / "classification_audit.csv",
        [
            {
                "tx_id": "fund1",
                "period": "2026-01",
                "Currency": "ARS",
                "semantic_bucket": "funding_contribution",
                "semantic_subbucket": "family_or_tenant_contribution",
                "funding_actor": "Tenants",
                "funding_channel": "tenant_to_box",
                "cash_effect": "cash_in_box",
                "amount": 50,
            }
        ],
    )
    _write(
        run / "monthly_debt_position.csv",
        [
            {
                "period": "2026-12",
                "as_of_date": "2026-12-31",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "Matías",
                "pair": "PM → Matías",
                "component": "total",
                "open_principal": 70,
                "open_interest": 7,
                "open_total": 77,
            }
        ],
    )
    _write(
        run / "monthly_debt_activity.csv",
        [
            {"period": "2026-01", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "activity_type": "repayment", "repayments": 10, "new_principal": 0, "interest_accrued": 0, "adjustments": 0, "net_change": -10},
            {"period": "2026-02", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "activity_type": "repayment", "repayments": 15, "new_principal": 0, "interest_accrued": 0, "adjustments": 0, "net_change": -15},
        ],
    )

    _write(
        tables / "annual_cash_close_by_box_wide.csv",
        [{"metric_id": "CASH.CLOSE.BY_BOX", "line_id": "cash.pm", "Box": "Property Management", "Currency": "ARS", "2026": 10}],
    )
    _write(
        tables / "annual_funding_by_actor_channel_wide.csv",
        [{"metric_id": "FUND.CONTRIB.BY_CHANNEL", "line_id": "fund.tenant", "Currency": "ARS", "funding_actor": "Tenants", "funding_channel": "tenant_to_box", "cash_effect": "cash_in_box", "target_box": "Property Management", "beneficiary_box": "", "obligation_box": "", "2026": 50}],
    )
    _write(
        tables / "annual_debt_stock_by_pair_wide.csv",
        [{"metric_id": "DEBT.STOCK.BY_PAIR.OPEN_TOTAL", "line_id": "debt.stock", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "component": "open_total", "2026": 77}],
    )
    _write(
        tables / "annual_debt_activity_by_pair_wide.csv",
        [{"metric_id": "DEBT.ACTIVITY.REPAYMENT.BY_PAIR", "line_id": "debt.repay", "Currency": "USD", "debtor": "PM", "creditor": "Matías", "pair": "PM → Matías", "activity_type": "repayments", "2026": 25}],
    )

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
