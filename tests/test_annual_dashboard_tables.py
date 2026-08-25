from __future__ import annotations

import pandas as pd

from accounting.professional.annual_dashboard_tables import (
    build_annual_cash_close_by_box,
    build_annual_funding_by_actor_channel,
)


def test_annual_cash_close_uses_latest_month_not_sum() -> None:
    cash = pd.DataFrame([
        {"period": "2025-01", "Box": "Property Management", "Currency": "ARS", "metric": "cash_close", "value": 100},
        {"period": "2025-12", "Box": "Property Management", "Currency": "ARS", "metric": "cash_close", "value": 250},
        {"period": "2025-06", "Box": "Family Business", "Currency": "ARS", "metric": "cash_close", "value": 40},
    ])

    long_df, wide_df = build_annual_cash_close_by_box(cash)

    pm = long_df[(long_df["period"].eq("2025")) & (long_df["Box"].eq("Property Management"))].iloc[0]
    assert pm["value"] == 250
    assert pm["selected_month"] == "2025-12"
    assert pm["metric_id"] == "CASH.CLOSE.BY_BOX"
    assert wide_df.loc[wide_df["Box"].eq("Property Management"), "2025"].iloc[0] == 250


def test_annual_funding_preserves_actor_channel_and_cash_effect() -> None:
    flow = pd.DataFrame([
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Property Management",
            "semantic_bucket": "funding_contribution", "funding_actor": "Tenant A",
            "funding_channel": "tenant_to_box", "cash_effect": "cash_in_box", "target_box": "Property Management",
            "amount_in": 100, "net_amount": 100,
        },
        {
            "period": "2026-02", "Currency": "ARS", "Box": "Property Management",
            "semantic_bucket": "funding_contribution", "funding_actor": "Tenant A",
            "funding_channel": "tenant_direct_tax_payment", "cash_effect": "no_cash_in_box_direct_payment",
            "obligation_box": "Property Management", "amount_in": 30, "net_amount": 30,
        },
    ])

    long_df, wide_df = build_annual_funding_by_actor_channel(flow)

    assert set(long_df["funding_channel"]) == {"tenant_to_box", "tenant_direct_tax_payment"}
    direct = long_df[long_df["funding_channel"].eq("tenant_direct_tax_payment")].iloc[0]
    assert direct["cash_effect"] == "no_cash_in_box_direct_payment"
    assert direct["value"] == 30
    assert "2026" in wide_df.columns


def test_annual_funding_excludes_non_funding_actor_rows() -> None:
    flow = pd.DataFrame([
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Property Management",
            "semantic_bucket": "property_opex", "actor": "Property Management",
            "funding_actor": "", "funding_channel": "", "amount_in": 0, "net_amount": -100,
        },
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Property Management",
            "semantic_bucket": "funding_contribution", "actor": "Matías",
            "amount_in": 50, "net_amount": 50,
        },
    ])

    long_df, _ = build_annual_funding_by_actor_channel(flow)

    assert not long_df.empty
    assert set(long_df["funding_actor"]) == {"Matías"}
    assert long_df[long_df["metric_id"].eq("FUND.CONTRIB.BY_FUNDING_ACTOR")].iloc[0]["value"] == 50


def test_annual_funding_produces_required_metric_views_and_value_rules() -> None:
    flow = pd.DataFrame([
        {
            "period": "2026-01", "Currency": "ARS", "Box": "Property Management",
            "semantic_bucket": "funding_contribution", "payer": "Inq",
            "funding_channel": "tenant_to_box", "target_box": "Property Management",
            "amount_in": 100, "amount_abs": 100, "net_amount": 100,
        },
        {
            "period": "2026-02", "Currency": "ARS", "Box": "Property Management",
            "semantic_bucket": "funding_contribution", "payer": "Inq",
            "funding_channel": "tenant_direct_service_payment",
            "cash_effect": "no_cash_in_box_direct_payment", "obligation_box": "Property Management",
            "amount_in": 0, "amount_out": 30, "amount_abs": 30, "net_amount": -30,
        },
        {
            "period": "2026-03", "Currency": "USD", "Box": "Property Management",
            "semantic_bucket": "funding_contribution", "actor": "Matías",
            "funding_channel": "named_actor_support", "debt_effect": "debt_settlement",
            "amount_in": 20, "amount_abs": 20, "net_amount": 20,
        },
    ])

    long_df, _ = build_annual_funding_by_actor_channel(flow)

    expected = {
        "FUND.CONTRIB.BY_FUNDING_ACTOR",
        "FUND.CONTRIB.BY_CHANNEL",
        "FUND.CONTRIB.BY_CASH_EFFECT",
        "FUND.CONTRIB.BY_TARGET_BOX",
        "FUND.CONTRIB.DIRECT_OBLIGATION",
        "FUND.CONTRIB.CASH_TO_BOX",
        "FUND.CONTRIB.DEBT_LINKED",
    }
    assert expected.issubset(set(long_df["metric_id"]))
    assert "Tenants" in set(long_df["funding_actor"])
    direct = long_df[long_df["metric_id"].eq("FUND.CONTRIB.DIRECT_OBLIGATION")].iloc[0]
    assert direct["value"] == 30
    assert direct["cash_effect"] == "no_cash_in_box_direct_payment"
    assert "amount_in for cash-to-box" in direct["calculation_rule"]
