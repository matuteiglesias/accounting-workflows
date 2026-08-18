from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.diagnostics.amount_direction import build_amount_direction_diagnostic
from accounting.marts.semantic import (
    build_monthly_operating_statement_from_split,
    build_semantic_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
AMOUNT_FIXTURE = ROOT / "fixtures" / "semantic_amount_direction_fixture.csv"
FX_FIXTURE = ROOT / "fixtures" / "semantic_fx_precedence_fixture.csv"


def _characterize(row: pd.Series, output_dir: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    paths = build_semantic_outputs(pd.DataFrame([row]), output_dir)
    audit = pd.read_csv(paths["classification_audit"]).iloc[0]
    split = pd.read_csv(paths["monthly_flow_semantic_split"])
    statement, _ = build_monthly_operating_statement_from_split(split)
    return audit, split, statement


@pytest.mark.parametrize(
    ("case_id", "direction", "amount_in", "amount_out", "net_amount", "amount_abs", "line", "contribution"),
    [
        ("box_receives_positive", "in", 100, 0, 100, 100, "operating_revenue", 100),
        ("box_receives_negative", "in", -100, 0, -100, 100, "operating_revenue", -100),
        ("box_pays_positive", "out", 0, 100, -100, 100, "property_opex_true", 100),
        ("box_pays_negative", "out", 0, -100, 100, 100, "property_opex_true", -100),
        ("internal_box_transfer", "internal", 0, 0, 0, 100, "internal_transfers", 100),
        ("neither_semantic_fallback", "in", 100, 0, 100, 100, "operating_revenue", 100),
        ("zero_box_payment", "out", 0, 0, 0, 0, "property_opex_true", 0),
    ],
)
def test_amount_direction_executable_characterization(
    tmp_path: Path,
    case_id: str,
    direction: str,
    amount_in: float,
    amount_out: float,
    net_amount: float,
    amount_abs: float,
    line: str,
    contribution: float,
) -> None:
    row = pd.read_csv(AMOUNT_FIXTURE).set_index("case_id").loc[case_id]
    audit, split, statement = _characterize(row, tmp_path / case_id)

    assert audit["direction"] == direction
    assert split.iloc[0]["amount_in"] == amount_in
    assert split.iloc[0]["amount_out"] == amount_out
    assert split.iloc[0]["net_amount"] == net_amount
    assert split.iloc[0]["amount_abs"] == amount_abs
    assert statement.set_index("statement_line").loc[line, "amount"] == contribution


FX_EXPECTED = {
    "fx_plus_rent": ("in", "operating_revenue", "rent", "R001_rent_collections", "operating_revenue"),
    "fx_plus_taxes": ("out", "property_opex", "taxes", "R002_property_taxes", "property_opex_true"),
    "fx_plus_maintenance": ("out", "property_opex", "maintenance", "R004_property_maintenance", "property_opex_true"),
    "fx_plus_funding": ("in", "funding_contribution", "family_or_tenant_contribution", "R006_contribution", "funding_contributions"),
    "fx_plus_withdrawal": ("out", "family_withdrawal_candidate", "personal_expense", "R011_personal_expense_text", "family_draws_or_distributions"),
    "fx_plus_loan": ("in", "debt_movement", "principal", "R007_debt_principal", "debt_movements"),
    "fx_plus_repayment": ("out", "debt_movement", "repayment", "R008_debt_repayment", "debt_movements"),
    "fx_plus_interest": ("out", "debt_movement", "interest", "R009_debt_interest", "debt_movements"),
    "clean_fx_proceeds": ("in", "treasury_fx", "fx_conversion_proceeds", "R014_fx_conversion_proceeds", "treasury_fx_conversion_in"),
    "clean_fx_outflow": ("out", "treasury_fx", "fx_conversion_outflow", "R014_fx_conversion_proceeds", "treasury_fx_conversion_out"),
    "explicit_fx_cost": ("out", "treasury_fx", "fx_cost_or_spread", "R015_fx_cost_or_spread", "treasury_fx_cost"),
}


@pytest.mark.parametrize("case_id", FX_EXPECTED)
def test_fx_overlap_rule_precedence_is_characterized(tmp_path: Path, case_id: str) -> None:
    row = pd.read_csv(FX_FIXTURE).set_index("case_id").loc[case_id]
    audit, _, statement = _characterize(row, tmp_path / case_id)
    direction, bucket, subbucket, rule_id, inclusion = FX_EXPECTED[case_id]

    assert (audit["direction"], audit["semantic_bucket"], audit["semantic_subbucket"], audit["rule_id"]) == (
        direction,
        bucket,
        subbucket,
        rule_id,
    )
    nonzero_lines = set(statement.loc[statement["amount"].ne(0), "statement_line"])
    assert inclusion in nonzero_lines


def test_amount_direction_diagnostic_writes_only_explicit_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostic"
    paths = build_amount_direction_diagnostic(AMOUNT_FIXTURE, output_dir, examples_per_group=1)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    matrix = pd.read_csv(paths["direction_sign_matrix"])

    assert summary["row_count"] == 7
    assert summary["amount_sign_counts"] == {
        "invalid": 0,
        "negative": 2,
        "positive": 4,
        "zero": 1,
    }
    assert summary["party_direction_counts"] == {
        "internal": 1,
        "neither": 1,
        "payer_is_box": 3,
        "receiver_is_box": 2,
    }
    assert set(paths.values()) == set(output_dir.iterdir())
    assert int(matrix["row_count"].sum()) == 7


def test_amount_direction_diagnostic_rejects_nonlocal_input(tmp_path: Path) -> None:
    from accounting.diagnostics.amount_direction import _local_path

    with pytest.raises(ValueError, match="local filesystem path"):
        _local_path("https://example.test/ledger.csv", "--ledger")
