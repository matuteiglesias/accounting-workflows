from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from accounting.contracts.funding_support import (
    FUNDING_SUPPORT_SPECS_VERSION,
    classify_funding_support,
)
from accounting.marts.semantic import build_semantic_outputs
from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.professional.annual_dashboard_tables import (
    build_annual_funding_by_actor_channel,
)


def _support_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period":"2026-01","Currency":"ARS","semantic_bucket":"funding_contribution","semantic_subbucket":"family_or_tenant_contribution",
                "funding_actor":"Inquilino","funding_channel":"tenant_to_box","cash_effect":"cash_in_box","debt_effect":"none","target_box":"Property Management",
                "amount_in":50.0,"amount_out":0.0,"amount_abs":50.0,"source_tx_ids_sample":"core-1",
            },
            {
                "period":"2026-01","Currency":"ARS","semantic_bucket":"property_opex","semantic_subbucket":"taxes",
                "funding_actor":"Inquilino","funding_channel":"tenant_direct_tax_payment","cash_effect":"no_cash_in_box_direct_payment","debt_effect":"none","target_box":"Property Management",
                "amount_in":0.0,"amount_out":30.0,"amount_abs":30.0,"source_tx_ids_sample":"direct-1",
            },
            {
                "period":"2026-01","Currency":"ARS","semantic_bucket":"debt_movement","semantic_subbucket":"principal",
                "funding_actor":"Matías","funding_channel":"debt_creation","cash_effect":"cash_in_box","debt_effect":"creates_debt","target_box":"Property Management",
                "amount_in":90.0,"amount_out":0.0,"amount_abs":90.0,"source_tx_ids_sample":"debt-1",
            },
            {
                "period":"2026-01","Currency":"ARS","semantic_bucket":"property_opex","semantic_subbucket":"services",
                "funding_actor":"","funding_channel":"","cash_effect":"cash_out_of_box","debt_effect":"none","target_box":"Property Management",
                "payer":"tenant direct funding support debt words only","amount_in":0.0,"amount_out":999.0,"amount_abs":999.0,"source_tx_ids_sample":"ordinary-opex",
            },
        ]
    )


def test_support_contract_has_small_explicit_membership_and_governed_amounts() -> None:
    members = classify_funding_support(_support_rows())

    assert FUNDING_SUPPORT_SPECS_VERSION == "funding_support_specs_v1"
    assert set(members["support_kind"]) == {
        "core_contribution",
        "direct_obligation_payment",
        "debt_linked_support",
    }
    assert set(members["source_member_ids"]) == {"core-1", "direct-1", "debt-1"}
    values = dict(zip(members["source_member_ids"], members["support_amount"], strict=True))
    assert values == {"core-1": 50.0, "direct-1": 30.0, "debt-1": 90.0}
    assert "ordinary-opex" not in set(members["source_member_ids"])
    assert dict(zip(members["source_member_ids"], members["support_measure"], strict=True)) == {
        "core-1": "amount_in",
        "direct-1": "amount_out",
        "debt-1": "amount_abs",
    }


def test_overlapping_support_semantics_fail_closed() -> None:
    row = _support_rows().iloc[[1]].copy()
    row["debt_effect"] = "creates_debt"
    with pytest.raises(ValueError, match="multiple governed support kinds"):
        classify_funding_support(row)


def test_professional_funding_builder_uses_explicit_support_not_label_blob_inference() -> None:
    long_df, _ = build_annual_funding_by_actor_channel(_support_rows(), ["2026"])

    assert not long_df.empty
    assert long_df["source_filter"].astype(str).str.contains(FUNDING_SUPPORT_SPECS_VERSION).all()
    assert not long_df["source_filter"].astype(str).str.contains("label", case=False).any()
    assert float(
        long_df.loc[
            long_df["metric_id"].eq("FUND.CONTRIB.DIRECT_OBLIGATION"), "value"
        ].sum()
    ) == 30.0
    assert float(
        long_df.loc[long_df["metric_id"].eq("FUND.CONTRIB.DEBT_LINKED"), "value"].sum()
    ) == 90.0
    assert float(
        long_df.loc[long_df["metric_id"].eq("FUND.CONTRIB.CASH_TO_BOX"), "value"].sum()
    ) == 140.0
    assert 999.0 not in set(pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0))


def test_core_funding_metric_stays_narrow_while_broader_support_is_explicit(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        [
            {"tx_id":"tenant_cash","Date":"2026-01-02","amount":50.0,"Currency":"ARS","Box":"Property Management","Lugar":"CABA","payer":"Inq","receiver":"PM","Flujo":"Contribucion","Tipo":"Contribuciones","Detalle":"Inquilino a la caja"},
            {"tx_id":"tenant_tax","Date":"2026-01-03","amount":30.0,"Currency":"ARS","Box":"Property Management","Lugar":"CABA","payer":"Inq","receiver":"Tax authority","Flujo":"Pagos","Tipo":"Impuestos","Detalle":"Inquilino directo a pagar impuestos"},
            {"tx_id":"debt","Date":"2026-01-05","amount":90.0,"Currency":"ARS","Box":"Property Management","Lugar":"CABA","payer":"Matias","receiver":"PM","Flujo":"Transfer","Tipo":"Prestamo","Detalle":"Matias funding deuda"},
        ]
    )
    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"
    build_semantic_outputs(ledger, run_root)
    paths = build_annual_balance_dashboard(
        run_root, metrics_dir, run_id="funding-support-test", as_of_date="2026-01-31"
    )
    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    ars_2026 = metrics[
        pd.to_numeric(metrics["period"], errors="coerce").eq(2026)
        & metrics["Currency"].astype(str).eq("ARS")
        & metrics["value_status"].eq("available")
    ]

    core = ars_2026[ars_2026["metric_id"].eq("FUND.CONTRIB.TOTAL")]
    assert len(core) == 1
    assert float(core.iloc[0]["value"]) == 50.0

    direct = ars_2026[ars_2026["metric_id"].eq("FUND.CONTRIB.DIRECT_OBLIGATION")]
    debt = ars_2026[ars_2026["metric_id"].eq("FUND.CONTRIB.DEBT_LINKED")]
    cash = ars_2026[ars_2026["metric_id"].eq("FUND.CONTRIB.CASH_TO_BOX")]
    assert float(direct.iloc[0]["value"]) == 30.0
    assert float(debt.iloc[0]["value"]) == 90.0
    assert float(cash.iloc[0]["value"]) == 140.0
    assert all(FUNDING_SUPPORT_SPECS_VERSION in value for value in direct["source_filter"])
