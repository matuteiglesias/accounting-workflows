from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.marts.semantic import build_semantic_outputs
from accounting.metrics.annual import build_annual_balance_dashboard


def test_semantic_outputs_add_explicit_funding_dimensions(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        [
            {
                "tx_id": "rent",
                "Date": "2026-01-01",
                "amount": 100.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA",
                "payer": "Inq",
                "receiver": "PM",
                "Flujo": "Cobros",
                "Tipo": "Renta",
                "Detalle": "tenant rent",
            },
            {
                "tx_id": "tenant_cash",
                "Date": "2026-01-02",
                "amount": 50.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA",
                "payer": "Inq",
                "receiver": "PM",
                "Flujo": "Contribucion",
                "Tipo": "Contribuciones",
                "Detalle": "Inquilino a la caja",
            },
            {
                "tx_id": "tenant_tax",
                "Date": "2026-01-03",
                "amount": 30.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA",
                "payer": "Inq",
                "receiver": "Tax authority",
                "Flujo": "Pagos",
                "Tipo": "Impuestos",
                "Detalle": "Inquilino directo a pagar impuestos",
            },
            {
                "tx_id": "ale_fb",
                "Date": "2026-01-04",
                "amount": 70.0,
                "Currency": "ARS",
                "Box": "Family Business",
                "Lugar": "Tigre",
                "payer": "Alejandro",
                "receiver": "FB",
                "Flujo": "Contribucion",
                "Tipo": "Contribuciones",
                "Detalle": "Alejandro funding FB",
            },
            {
                "tx_id": "primos_fb",
                "Date": "2026-01-04",
                "amount": 40.0,
                "Currency": "ARS",
                "Box": "Family Business",
                "Lugar": "Tigre",
                "payer": "Primos",
                "receiver": "FB",
                "Flujo": "Contribucion",
                "Tipo": "Contribuciones",
                "Detalle": "Primos funding FB",
            },
            {
                "tx_id": "hh_pm",
                "Date": "2026-01-04",
                "amount": 20.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA",
                "payer": "HH",
                "receiver": "PM",
                "Flujo": "Contribucion",
                "Tipo": "Contribuciones",
                "Detalle": "Household funding PM",
            },
            {
                "tx_id": "debt",
                "Date": "2026-01-05",
                "amount": 90.0,
                "Currency": "ARS",
                "Box": "Property Management",
                "Lugar": "CABA",
                "payer": "Matias",
                "receiver": "PM",
                "Flujo": "Transfer",
                "Tipo": "Prestamo",
                "Detalle": "Matias funding deuda",
            },
        ]
    )

    paths = build_semantic_outputs(ledger, tmp_path)
    audit = pd.read_csv(paths["classification_audit"])
    monthly = pd.read_csv(paths["monthly_flow_semantic_split"])

    expected_columns = {
        "funding_actor",
        "funding_channel",
        "source_box",
        "target_box",
        "beneficiary_box",
        "obligation_box",
        "payment_channel",
        "cash_effect",
        "debt_effect",
        "linked_debt_id",
    }
    assert expected_columns.issubset(audit.columns)
    assert expected_columns.issubset(monthly.columns)

    rent = audit[audit["tx_id"].eq("rent")].iloc[0]
    assert rent["semantic_bucket"] == "operating_revenue"
    assert rent["semantic_subbucket"] == "rent"
    assert pd.isna(rent["funding_actor"]) or rent["funding_actor"] == ""
    assert pd.isna(rent["funding_channel"]) or rent["funding_channel"] == ""
    assert rent["cash_effect"] == "cash_in_box"
    assert rent["debt_effect"] == "none"

    tenant_cash = audit[audit["tx_id"].eq("tenant_cash")].iloc[0]
    assert tenant_cash["funding_actor"] == "Inquilino"
    assert tenant_cash["funding_channel"] == "tenant_to_box"
    assert tenant_cash["cash_effect"] == "cash_in_box"
    assert tenant_cash["target_box"] == "Property Management"

    tenant_tax = audit[audit["tx_id"].eq("tenant_tax")].iloc[0]
    assert tenant_tax["semantic_bucket"] == "property_opex"
    assert tenant_tax["funding_actor"] == "Inquilino"
    assert tenant_tax["funding_channel"] == "tenant_direct_tax_payment"
    assert tenant_tax["cash_effect"] == "no_cash_in_box_direct_payment"
    assert tenant_tax["obligation_box"] == "Property Management"

    fb = audit[audit["tx_id"].eq("ale_fb")].iloc[0]
    assert fb["funding_actor"] == "Alejandro"
    assert fb["funding_channel"] == "family_business_contribution"
    assert fb["beneficiary_box"] == "Family Business"

    primos = audit[audit["tx_id"].eq("primos_fb")].iloc[0]
    assert primos["funding_actor"] == "Primos"
    assert primos["funding_channel"] == "family_business_contribution"

    hh_pm = audit[audit["tx_id"].eq("hh_pm")].iloc[0]
    assert hh_pm["funding_actor"] == "Household"
    assert hh_pm["funding_channel"] == "household_to_pm"
    assert hh_pm["target_box"] == "Property Management"

    debt = audit[audit["tx_id"].eq("debt")].iloc[0]
    assert debt["semantic_bucket"] == "debt_movement"
    assert debt["funding_actor"] == "Matías"
    assert debt["funding_channel"] == "debt_creation"
    assert debt["debt_effect"] == "creates_debt"

    assert not monthly[
        monthly["funding_actor"].eq("Matías")
        & monthly["target_box"].eq("Property Management")
        & monthly["debt_effect"].eq("creates_debt")
    ].empty
    assert not monthly[
        monthly["funding_actor"].eq("Inquilino")
        & monthly["funding_channel"].eq("tenant_direct_tax_payment")
        & monthly["cash_effect"].eq("no_cash_in_box_direct_payment")
    ].empty
    assert not monthly[
        monthly["funding_channel"].eq("tenant_to_box")
        & monthly["cash_effect"].eq("cash_in_box")
    ].empty
    assert not monthly[
        monthly["funding_actor"].isin(["Alejandro", "Primos"])
        & monthly["target_box"].eq("Family Business")
    ].empty
    assert not monthly[
        monthly["funding_actor"].eq("Household")
        & monthly["funding_channel"].eq("household_to_pm")
    ].empty


def test_annual_dashboard_emits_dimensioned_funding_metrics(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        [
            {"tx_id": "tenant_cash", "Date": "2026-01-02", "amount": 50.0, "Currency": "ARS", "Box": "Property Management", "Lugar": "CABA", "payer": "Inq", "receiver": "PM", "Flujo": "Contribucion", "Tipo": "Contribuciones", "Detalle": "Inquilino a la caja"},
            {"tx_id": "tenant_tax", "Date": "2026-01-03", "amount": 30.0, "Currency": "ARS", "Box": "Property Management", "Lugar": "CABA", "payer": "Inq", "receiver": "Tax authority", "Flujo": "Pagos", "Tipo": "Impuestos", "Detalle": "Inquilino directo a pagar impuestos"},
            {"tx_id": "ale_fb", "Date": "2026-01-04", "amount": 70.0, "Currency": "ARS", "Box": "Family Business", "Lugar": "Tigre", "payer": "Alejandro", "receiver": "FB", "Flujo": "Contribucion", "Tipo": "Contribuciones", "Detalle": "Alejandro funding FB"},
            {"tx_id": "hh_pm", "Date": "2026-01-04", "amount": 20.0, "Currency": "ARS", "Box": "Property Management", "Lugar": "CABA", "payer": "HH", "receiver": "PM", "Flujo": "Contribucion", "Tipo": "Contribuciones", "Detalle": "Household funding PM"},
            {"tx_id": "debt", "Date": "2026-01-05", "amount": 90.0, "Currency": "ARS", "Box": "Property Management", "Lugar": "CABA", "payer": "Matias", "receiver": "PM", "Flujo": "Transfer", "Tipo": "Prestamo", "Detalle": "Matias funding deuda"},
        ]
    )
    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"
    build_semantic_outputs(ledger, run_root)
    paths = build_annual_balance_dashboard(run_root, metrics_dir, run_id="test", as_of_date="2026-07-12")
    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])

    expected_metric_ids = {
        "FUND.CONTRIB.BY_FUNDING_ACTOR",
        "FUND.CONTRIB.BY_CHANNEL",
        "FUND.CONTRIB.BY_CASH_EFFECT",
        "FUND.CONTRIB.BY_TARGET_BOX",
        "FUND.CONTRIB.DIRECT_OBLIGATION",
        "FUND.CONTRIB.CASH_TO_BOX",
        "FUND.CONTRIB.DEBT_LINKED",
    }
    available = metrics[metrics["value_status"].eq("available")]
    assert expected_metric_ids.issubset(set(available["metric_id"]))

    def value(metric_id: str, dim_name: str = "", dim_value: str = "") -> float:
        rows = available[available["metric_id"].eq(metric_id)]
        if dim_name:
            rows = rows[rows["dimension_name"].eq(dim_name) & rows["dimension_value"].eq(dim_value)]
        return float(rows["value"].sum())

    assert value("FUND.CONTRIB.BY_FUNDING_ACTOR", "funding_actor", "Inquilino") == 80.0
    assert value("FUND.CONTRIB.BY_CHANNEL", "funding_channel", "tenant_direct_tax_payment") == 30.0
    assert value("FUND.CONTRIB.BY_CASH_EFFECT", "cash_effect", "no_cash_in_box_direct_payment") == 30.0
    assert value("FUND.CONTRIB.BY_TARGET_BOX", "target_box", "Family Business") == 70.0
    assert value("FUND.CONTRIB.DIRECT_OBLIGATION") == 30.0
    assert value("FUND.CONTRIB.CASH_TO_BOX") == 230.0
    assert value("FUND.CONTRIB.DEBT_LINKED") == 90.0
