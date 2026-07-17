import pandas as pd

from accounting.marts.semantic import build_monthly_operating_statement_from_split
from accounting.scope import (
    HOUSEHOLD_BOXES,
    PROPERTY_BUSINESS_BOXES,
    property_business_scope_mask,
)


def _row(box: str, bucket: str, amount_in: float = 0, amount_out: float = 0, **extra):
    return {
        "period": "2026-01",
        "period_end": "2026-01-31",
        "Currency": "ARS",
        "Box": box,
        "semantic_bucket": bucket,
        "semantic_subbucket": extra.pop("semantic_subbucket", ""),
        "amount_in": amount_in,
        "amount_out": amount_out,
        "net_amount": amount_in - amount_out,
        "amount_abs": amount_in + amount_out,
        "n_tx": 1,
        "review_required": False,
        **extra,
    }


def test_property_business_scope_uses_box_or_explicit_attribution():
    df = pd.DataFrame(
        [
            {"Box": "Family Business"},
            {"Box": "Property Management"},
            {"Box": "Household"},
            {"Box": "Household", "target_box": "Property Management"},
            {"Box": "Household", "beneficiary_box": "Family Business"},
            {"Box": "Household", "obligation_box": "Property Management"},
        ]
    )

    assert PROPERTY_BUSINESS_BOXES == {"Family Business", "Property Management"}
    assert HOUSEHOLD_BOXES == {"Household"}
    assert property_business_scope_mask(df).tolist() == [True, True, False, True, True, True]


def test_operating_statement_excludes_unattributed_household_opex():
    split = pd.DataFrame(
        [
            _row("Property Management", "operating_revenue", amount_in=500, semantic_subbucket="rent"),
            _row("Property Management", "property_opex", amount_out=100, semantic_subbucket="services"),
            _row("Household", "property_opex", amount_out=900, semantic_subbucket="services"),
            _row(
                "Household",
                "property_opex",
                amount_out=50,
                semantic_subbucket="taxes",
                obligation_box="Property Management",
            ),
        ]
    )

    statement, _ = build_monthly_operating_statement_from_split(split)
    amounts = statement.set_index("statement_line")["amount"]

    assert amounts["property_opex_true"] == 150
    assert amounts["net_operating"] == 350
    assert amounts["services"] == 100
    assert amounts["taxes"] == 50
