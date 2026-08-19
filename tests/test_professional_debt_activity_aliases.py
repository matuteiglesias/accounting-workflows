from __future__ import annotations

import pandas as pd

from accounting.contracts.debt_position_activity import DEBT_ACTIVITY_SPECS_V1
from accounting.professional.debt_activity_executor import _resolve_row_spec


def test_all_v1_debt_activity_specs_are_reachable_from_professional_views() -> None:
    aliases = {
        "new_principal": "debt.activity.new_claim",
        "interest_accrued": "debt.activity.interest_accrual",
        "repayments": "debt.activity.repayment",
        "adjustments": "debt.activity.adjustment",
        "net_change": "debt.activity.net_change",
    }

    resolved_ids = set()
    for view_token, expected_spec_id in aliases.items():
        monthly = _resolve_row_spec(pd.Series({"measure": view_token}))
        annual = _resolve_row_spec(pd.Series({"activity_type": view_token}))
        assert monthly is not None
        assert annual is not None
        assert monthly.spec_id == expected_spec_id
        assert annual.spec_id == expected_spec_id
        resolved_ids.add(monthly.spec_id)

    assert resolved_ids == set(DEBT_ACTIVITY_SPECS_V1)


def test_uncontracted_settlements_view_does_not_resolve_to_v1_activity_spec() -> None:
    assert _resolve_row_spec(pd.Series({"activity_type": "settlements"})) is None
    assert _resolve_row_spec(pd.Series({"measure": "settlements"})) is None
