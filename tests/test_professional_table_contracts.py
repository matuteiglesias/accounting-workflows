from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.professional.table_contracts import enrich_professional_table, enrich_professional_table_contracts


def test_enrich_professional_table_adds_stable_funding_contract_columns() -> None:
    df = pd.DataFrame(
        [
            {"Currency": "ARS", "metric": "Funding / aportes", "2026": 100},
            {"Currency": "ARS", "metric": "Inquilinos directo a pagar impuestos", "2026": 30},
            {"Currency": "ARS", "metric": "Matías funding", "2026": 20},
            {"Currency": "ARS", "metric": "Household funding PM", "2026": 10},
        ]
    )

    out = enrich_professional_table(df, "overview_balance_dashboard")

    for col in ["metric_id", "line_id", "dimension_name", "dimension_value", "funding_channel", "funding_actor", "cash_effect"]:
        assert col in out.columns

    funding = out[out["metric"].eq("Funding / aportes")].iloc[0]
    assert funding["metric_id"] == "FUND.CONTRIB.TOTAL"
    assert funding["line_id"].startswith("overview_balance_dashboard")

    direct_tax = out[out["metric"].eq("Inquilinos directo a pagar impuestos")].iloc[0]
    assert direct_tax["metric_id"] == "FUND.CONTRIB.BY_CHANNEL"
    assert direct_tax["dimension_name"] == "funding_channel"
    assert direct_tax["dimension_value"] == "tenant_direct_tax_payment"
    assert direct_tax["funding_channel"] == "tenant_direct_tax_payment"
    assert direct_tax["funding_actor"] == "Inquilino"
    assert direct_tax["cash_effect"] == "no_cash_in_box_direct_payment"

    matias = out[out["metric"].eq("Matías funding")].iloc[0]
    assert matias["metric_id"] == "FUND.CONTRIB.BY_FUNDING_ACTOR"
    assert matias["dimension_name"] == "funding_actor"
    assert matias["dimension_value"] == "Matías"

    hh = out[out["metric"].eq("Household funding PM")].iloc[0]
    assert hh["metric_id"] == "FUND.CONTRIB.BY_CHANNEL"
    assert hh["dimension_value"] == "household_to_pm"
    assert hh["funding_actor"] == "Household"


def test_enrich_professional_table_contracts_rewrites_csv(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    tables.mkdir()
    path = tables / "overview_balance_dashboard.csv"
    pd.DataFrame([{"Currency": "ARS", "metric": "Inquilinos a la caja", "2026": 50}]).to_csv(path, index=False)

    written = enrich_professional_table_contracts(tables)

    assert written == [path]
    out = pd.read_csv(path)
    row = out.iloc[0]
    assert row["metric_id"] == "FUND.CONTRIB.BY_CHANNEL"
    assert row["dimension_name"] == "funding_channel"
    assert row["dimension_value"] == "tenant_to_box"
    assert row["funding_channel"] == "tenant_to_box"
    assert row["cash_effect"] == "cash_in_box"


def test_enrich_professional_table_does_not_contaminate_debt_position_matrix() -> None:
    df = pd.DataFrame(
        [
            {
                "Currency": "ARS",
                "metric": "open_total",
                "row_label": "FUND.CONTRIB.BY_FUNDING_ACTOR",
                "2026-06": 100,
            }
        ]
    )

    out = enrich_professional_table(df, "monthly_tables_debt_position_matrix")

    assert list(out.columns) == list(df.columns)
    assert "metric_id" not in out.columns


def test_enrich_professional_table_adds_overview_annual_metric_contracts() -> None:
    df = pd.DataFrame(
        [
            {"Currency": "ARS", "metric": "Funding / aportes", "2026": 100},
            {"Currency": "ARS", "metric": "Retiros / gasto personal", "2026": 80},
            {"Currency": "ARS", "metric": "Dividendos", "2026": 20},
            {"Currency": "ARS", "metric": "Deuda total abierta", "metric_id": "FUND.CONTRIB.DEBT_LINKED", "2026": 300},
            {"Currency": "ARS", "metric": "Principal abierto", "2026": 250},
            {"Currency": "ARS", "metric": "Interés abierto", "2026": 50},
        ]
    )

    out = enrich_professional_table(df, "overview_balance_dashboard")

    expected = {
        "Funding / aportes": "FUND.CONTRIB.TOTAL",
        "Retiros / gasto personal": "DIST.DRAWS.PERSONAL",
        "Dividendos": "DIST.DIVIDENDS",
        "Deuda total abierta": "ID.DEBT.TOTAL.OPEN",
        "Principal abierto": "ID.DEBT.PRINCIPAL.OPEN",
        "Interés abierto": "ID.DEBT.INTEREST.OPEN",
    }
    got = dict(zip(out["metric"], out["metric_id"], strict=True))
    assert got == expected

    debt = out[out["metric"].eq("Deuda total abierta")].iloc[0]
    assert debt["dimension_name"] == ""
    assert debt["funding_channel"] == ""


def test_enrich_professional_table_keeps_cash_bridge_net_debt_signed() -> None:
    df = pd.DataFrame(
        [
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "line": "Movimiento neto de deuda",
                "metric_id": "FUND.CONTRIB.DEBT_LINKED",
                "2026": -60,
            }
        ]
    )

    out = enrich_professional_table(df, "cash_annual_box_flow_bridge_wide")
    row = out.iloc[0]

    assert row["metric_id"] == ""
    assert row["dimension_name"] == ""
    assert row["funding_channel"] == ""
