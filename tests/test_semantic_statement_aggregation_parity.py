from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from accounting.marts.semantic import build_monthly_operating_statement_from_split


FROZEN = Path(__file__).resolve().parents[1] / "fixtures"

def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "period": "2026-01",
        "period_end": "2026-01-31",
        "Currency": "ARS",
        "semantic_bucket": "unknown",
        "semantic_subbucket": None,
        "amount_in": 0,
        "amount_out": 0,
        "net_amount": 0,
        "amount_abs": 0,
        "n_tx": 1,
        "review_required": False,
        "Box": "Caja A",
        "tx_id": "duplicate-looking-id",
    }
    row.update(overrides)
    return row


def test_cached_statement_matches_frozen_base_on_adversarial_synthetic_split() -> None:
    split = pd.DataFrame(
        [
            _row(semantic_bucket="operating_revenue", semantic_subbucket="rent", amount_in="bad", amount_abs=0, n_tx=2),
            _row(semantic_bucket="property_opex", semantic_subbucket="taxes", amount_out=-25, net_amount=25, amount_abs=25, n_tx=3),
            _row(semantic_bucket="property_opex", semantic_subbucket="services", amount_out=40, net_amount=-40, amount_abs=40, n_tx=4),
            _row(semantic_bucket="family_withdrawal_candidate", semantic_subbucket="personal_expense", amount_out=10, net_amount=-10, amount_abs=10, n_tx=5),
            _row(semantic_bucket="family_withdrawal", semantic_subbucket="personal_expense", amount_out=7, net_amount=-7, amount_abs=7, n_tx=6, Box="Caja B"),
            _row(semantic_bucket="treasury_fx", semantic_subbucket="fx_conversion_proceeds", amount_in=100, net_amount=100, amount_abs=100, n_tx=7),
            _row(semantic_bucket="treasury_fx", semantic_subbucket="unclassified_fx_residual", net_amount=3, amount_abs=3, n_tx=8),
            _row(semantic_bucket="unknown", semantic_subbucket=None, amount_out=0, amount_abs=12, n_tx=9, review_required=True),
            _row(semantic_bucket=None, semantic_subbucket=None, amount_abs=5, n_tx=10),
            _row(period_end="2026-01-30", amount_abs=2, n_tx=11),
            _row(Currency="USD", amount_abs=4, n_tx=12),
        ]
    )

    actual_statement, actual_qa = build_monthly_operating_statement_from_split(split)
    assert actual_statement.to_csv(index=False, lineterminator="\n").encode() == (
        FROZEN / "semantic_statement_adversarial_expected.csv"
    ).read_bytes()
    assert actual_qa.to_csv(index=False, lineterminator="\n").encode() == (
        FROZEN / "semantic_statement_adversarial_qa_expected.csv"
    ).read_bytes()


def test_selected_unsupported_measure_still_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import accounting.marts.semantic as semantic

    split = pd.DataFrame([_row(semantic_bucket="property_opex", semantic_subbucket="unapproved_category", amount_out=5, amount_abs=5)])
    monkeypatch.setattr(semantic, "resolve_semantic_measure", lambda bucket, subbucket: None)
    with pytest.raises(ValueError, match="No approved semantic measure"):
        build_monthly_operating_statement_from_split(split)


def test_unselected_unknown_category_does_not_trigger_measure_resolution() -> None:
    split = pd.DataFrame([_row(semantic_bucket="unknown", semantic_subbucket="unapproved_category", amount_abs=5)])
    statement, qa = build_monthly_operating_statement_from_split(split)
    assert len(statement) == 23
    assert not qa.empty


def test_nullable_string_category_keys_preserve_other_and_unknown_behavior() -> None:
    split = pd.DataFrame([
        _row(semantic_bucket="operating_revenue", semantic_subbucket="rent", amount_in=10, amount_abs=10, n_tx=1),
        _row(semantic_bucket=pd.NA, semantic_subbucket=pd.NA, amount_abs=3, review_required=True, n_tx=2),
        _row(semantic_bucket="property_opex", semantic_subbucket=pd.NA, amount_out=5, amount_abs=5, n_tx=3),
    ])
    split[["semantic_bucket", "semantic_subbucket"]] = split[["semantic_bucket", "semantic_subbucket"]].astype("string")
    statement, qa = build_monthly_operating_statement_from_split(split)
    amounts = statement.set_index("statement_line")["amount"]
    assert amounts["operating_revenue"] == 10
    assert amounts["property_opex_true"] == 5
    assert amounts["other_property_opex"] == 5
    assert amounts["unknown_or_ambiguous_outflows"] == 3
    assert statement.loc[statement.statement_line.eq("property_opex_true"), "n_tx"].iat[0] == 3
    assert len(qa) == 26
