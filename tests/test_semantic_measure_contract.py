from __future__ import annotations

import csv
from pathlib import Path

import pytest

from accounting.contracts.semantic_measures import (
    SEMANTIC_MEASURE_REGISTRY_V1,
    SEMANTIC_MEASURE_REGISTRY_VERSION,
    resolve_semantic_measure,
)


ROOT = Path(__file__).resolve().parents[1]
CHARACTERIZATION = ROOT / "docs" / "semantic_measure_authorities_20260819.csv"


@pytest.mark.parametrize(
    ("bucket", "subbucket", "measure"),
    [
        ("operating_revenue", "rent", "amount_in"),
        ("property_opex", "taxes", "amount_out"),
        ("property_opex", "services", "amount_out"),
        ("property_opex", "maintenance", "amount_out"),
        ("property_opex", "legal", "amount_out"),
        ("funding_contribution", "family_or_tenant_contribution", "amount_in"),
        ("family_withdrawal_candidate", "personal_expense", "amount_out"),
        ("family_withdrawal", "dividend", "amount_out"),
        ("debt_movement", "principal", "amount_abs"),
        ("debt_movement", "repayment", "amount_abs"),
        ("internal_transfer", "transfer", "amount_abs"),
        ("treasury_fx", "fx_conversion_proceeds", "amount_in"),
        ("treasury_fx", "fx_conversion_outflow", "amount_out"),
        ("treasury_fx", "fx_cost_or_spread", "amount_out"),
    ],
)
def test_registry_resolves_approved_semantic_measures(
    bucket: str, subbucket: str, measure: str
) -> None:
    assert resolve_semantic_measure(bucket, subbucket) == measure


@pytest.mark.parametrize(
    ("bucket", "subbucket"),
    [
        ("", ""),
        ("unknown", "ambiguous"),
        ("treasury_fx", ""),
        ("treasury_fx", "unapproved_future_fx"),
        ("future_bucket", "future_subbucket"),
    ],
)
def test_registry_fails_closed_for_unknown_semantics(
    bucket: str, subbucket: str
) -> None:
    assert resolve_semantic_measure(bucket, subbucket) is None


def test_registry_is_versioned_and_immutable() -> None:
    assert SEMANTIC_MEASURE_REGISTRY_VERSION == "semantic_measure_registry_v1"
    assert len(SEMANTIC_MEASURE_REGISTRY_V1) == 10
    with pytest.raises(TypeError):
        SEMANTIC_MEASURE_REGISTRY_V1[("future_bucket", "*")] = "amount_in"  # type: ignore[index]


def test_registry_matches_characterized_native_authority() -> None:
    with CHARACTERIZATION.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    characterized = {
        row["semantic_concept"]: resolve_semantic_measure(
            row["semantic_bucket"],
            row["semantic_subbucket"].replace("*", "representative_subbucket"),
        )
        for row in rows
        if row["semantic_concept"] not in {"unknown_fx", "review_required"}
    }
    expected = {
        row["semantic_concept"]: row["native_statement_measure"]
        for row in rows
        if row["semantic_concept"] not in {"unknown_fx", "review_required"}
    }
    assert characterized == expected


def test_atomic_consumers_are_wired_to_registry() -> None:
    migrated_consumers = [
        ROOT / "accounting" / "marts" / "semantic.py",
        ROOT / "accounting" / "management" / "usd_ccl_flows.py",
        ROOT / "accounting" / "metrics" / "annual.py",
        ROOT / "accounting" / "professional" / "drilldown.py",
    ]
    for consumer in migrated_consumers:
        assert "contracts.semantic_measures" in consumer.read_text(encoding="utf-8")
