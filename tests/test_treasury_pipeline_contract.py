from __future__ import annotations

from pathlib import Path

from accounting.artifacts.manifest import artifact_contract_for_name


def test_treasury_artifact_contracts_are_explicit():
    flow = artifact_contract_for_name("monthly_box_treasury_flow.csv")
    assert flow["artifact_role"] == "canonical_source"
    assert flow["accounting_nature"] == "flow"
    assert flow["grain"] == "monthly"
    assert flow["currency_policy"] == "by_currency"
    assert flow["source_authority"] == "source_of_truth_for_treasury_flow"

    accountability = artifact_contract_for_name("monthly_cash_accountability.csv")
    assert accountability["artifact_role"] == "canonical_source"
    assert accountability["accounting_nature"] == "mixed"
    assert accountability["currency_policy"] == "by_currency"
    assert accountability["source_authority"] == "derived_treasury_accountability"

    qa = artifact_contract_for_name("monthly_cash_accountability_qa.csv")
    assert qa["artifact_role"] == "qa"
    assert qa["frontend_suitability"] == "internal_only"


def test_post_debt_pipeline_materializes_accountability():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "accounting.marts.debt" in makefile
    assert "accounting.marts.treasury --run-root" in makefile
    assert makefile.index("accounting.marts.debt") < makefile.index(
        "accounting.marts.treasury --run-root"
    )
    assert "monthly_cash_accountability.csv" in makefile


def test_treasury_contract_does_not_move_accounting_logic_into_reports():
    design = Path(
        "notes/monthly_treasury_accountability_design_20260820.md"
    ).read_text(encoding="utf-8")
    assert "Economic attribution does not imply cash movement" in design
    assert "No HTML/report logic" in design
