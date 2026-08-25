from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from accounting.contracts.annual_flow_membership import build_annual_flow_membership
from accounting.professional.drilldown import build_professional_flow_drilldowns

EXPECTATIONS = (
    Path(__file__).parent
    / "fixtures"
    / "professional_regression"
    / "expected_cells.csv"
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _paths(tmp_path: Path, case_id: str) -> tuple[Path, Path, Path]:
    repo = tmp_path / case_id
    return (
        repo,
        repo / "out" / "run" / "accounting" / "latest",
        repo / "out" / "professional_pack" / "latest",
    )


def _expected(case_id: str) -> pd.DataFrame:
    df = pd.read_csv(EXPECTATIONS, keep_default_na=False)
    return df[df["case_id"].eq(case_id)].copy()


def _match(index: pd.DataFrame, expected: pd.Series) -> pd.Series:
    rows = index[index["table_id"].astype(str).eq(str(expected["table_id"]))].copy()
    rows = rows[rows["period"].astype(str).eq(str(expected["period"]))]
    rows = rows[rows["Currency"].astype(str).eq(str(expected["currency"]))]

    needle = str(expected.get("filter_contains", "")).strip()
    if needle:
        rows = rows[
            rows["filter_json"]
            .fillna("")
            .astype(str)
            .str.contains(needle, regex=False)
        ]

    box = str(expected.get("box", "")).strip()
    if box:
        rows = rows[
            rows["row_context_json"]
            .fillna("")
            .astype(str)
            .str.contains(box, regex=False)
            | rows["filter_json"]
            .fillna("")
            .astype(str)
            .str.contains(box, regex=False)
        ]

    assert len(rows) == 1, (
        expected["expectation_id"],
        rows[
            [
                "table_id",
                "period",
                "Currency",
                "filter_json",
                "row_context_json",
            ]
        ].to_dict("records"),
    )
    return rows.iloc[0]


def _assert_case(pack: Path, index: pd.DataFrame, case_id: str) -> None:
    expected = _expected(case_id)
    assert not expected.empty

    for _, spec in expected.iterrows():
        actual = _match(index, spec)
        assert actual["status"] == spec["drilldown_status"], spec["expectation_id"]
        assert (
            abs(float(actual["display_value"]) - float(spec["displayed_value"]))
            <= 1e-6
        )
        assert (
            abs(float(actual["matched_value_sum"]) - float(spec["matched_total"]))
            <= 1e-6
        )
        if actual["status"] == "ok":
            assert abs(float(actual["residual"])) <= 1e-6

        detail_path = pack / str(actual["detail_csv_relpath"])
        assert detail_path.exists(), spec["expectation_id"]
        members = [
            member
            for member in str(spec["source_member_ids"]).split(";")
            if member
        ]
        if members:
            detail = pd.read_csv(detail_path).fillna("").astype(str)
            text = "\n".join(
                detail.astype(str).agg("|".join, axis=1).tolist()
            )
            for member in members:
                assert member in text, (spec["expectation_id"], member, text)


def test_expectation_ledger_covers_required_semantic_risks() -> None:
    expected = pd.read_csv(EXPECTATIONS, keep_default_na=False)
    assert set(expected["semantic_purpose"]) == {
        "monthly rent",
        "native-currency monthly rent",
        "property OPEX excluding Household",
        "personal draw",
        "annual rent flow",
        "annual property OPEX flow",
        "annual personal draw flow",
        "core funding contribution",
        "broader direct-obligation support",
        "broader debt-linked support",
        "valid debt stock",
        "all-invalid debt as_of fails closed",
        "independent debt repayment activity",
        "annual debt activity sums flows",
        "validated cash only",
        "validated cash unavailable no fallback",
        "governed derived metric",
        "FX currency-total route",
        "FX Box by currency route",
    }
    assert {"ARS", "USD"}.issubset(set(expected["currency"]))
    assert set(expected["value_status"]) == {"available", "unavailable"}


def test_professional_corpus_monthly_and_annual_flows(tmp_path: Path) -> None:
    repo, run, pack = _paths(tmp_path, "flows")
    tables = pack / "tables"

    split = [
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Family Business",
            "semantic_bucket": "operating_revenue",
            "semantic_subbucket": "rent",
            "amount_in": 70,
            "amount_out": 0,
            "net_amount": 70,
            "amount_abs": 70,
            "n_tx": 1,
            "source_tx_ids_sample": "rent_ars",
        },
        {
            "period": "2026-01",
            "Currency": "USD",
            "Box": "Family Business",
            "semantic_bucket": "operating_revenue",
            "semantic_subbucket": "rent",
            "amount_in": 10,
            "amount_out": 0,
            "net_amount": 10,
            "amount_abs": 10,
            "n_tx": 1,
            "source_tx_ids_sample": "rent_usd",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "property_opex",
            "semantic_subbucket": "services",
            "amount_in": 0,
            "amount_out": 100,
            "net_amount": -100,
            "amount_abs": 100,
            "n_tx": 1,
            "source_tx_ids_sample": "opex_pm",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Household",
            "semantic_bucket": "household_expense",
            "semantic_subbucket": "services",
            "amount_in": 0,
            "amount_out": 999,
            "net_amount": -999,
            "amount_abs": 999,
            "n_tx": 1,
            "source_tx_ids_sample": "hh_expense",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Household",
            "semantic_bucket": "family_withdrawal_candidate",
            "semantic_subbucket": "personal_expense",
            "amount_in": 0,
            "amount_out": 20,
            "net_amount": -20,
            "amount_abs": 20,
            "n_tx": 1,
            "source_tx_ids_sample": "draw_hh",
        },
    ]
    split_df = pd.DataFrame(split)
    _write(run / "monthly_flow_semantic_split.csv", split)
    build_annual_flow_membership(split_df).to_csv(
        run / "annual_flow_membership.csv", index=False
    )
    _write(
        run / "classification_audit.csv",
        [
            {
                "tx_id": row["source_tx_ids_sample"],
                "period": row["period"],
                "Currency": row["Currency"],
                "Box": row["Box"],
                "semantic_bucket": row["semantic_bucket"],
                "semantic_subbucket": row["semantic_subbucket"],
                "amount": row["amount_abs"],
            }
            for row in split
        ],
    )
    _write(
        run / "annual_balance_dashboard_metrics.csv",
        [
            {
                "metric_id": "IS.RENT.TOTAL",
                "period": "2026",
                "Currency": "ARS",
                "value": 70,
                "value_status": "available",
                "flow_type": "flow",
                "source_table": "monthly_flow_semantic_split.csv",
            },
            {
                "metric_id": "IS.OPEX.PROPERTY",
                "period": "2026",
                "Currency": "ARS",
                "value": 100,
                "value_status": "available",
                "flow_type": "flow",
                "source_table": "monthly_flow_semantic_split.csv",
            },
            {
                "metric_id": "DIST.DRAWS.PERSONAL",
                "period": "2026",
                "Currency": "ARS",
                "value": 20,
                "value_status": "available",
                "flow_type": "flow",
                "source_table": "monthly_flow_semantic_split.csv",
            },
        ],
    )

    # Generic flow matrix rows remain useful diagnostics, but these two rent
    # rows now carry an explicit governed identity rather than relying on the
    # historical FB presentation bridge.
    _write(
        tables / "monthly_tables_flow_subbucket_all_measures.csv",
        [
            {
                "drilldown_cell_id": "flow.rent.total",
                "measure": "amount_in",
                "Currency": "ARS",
                "semantic_bucket": "operating_revenue",
                "semantic_subbucket": "rent",
                "2026-01": 70,
            },
            {
                "drilldown_cell_id": "flow.rent.total",
                "measure": "amount_in",
                "Currency": "USD",
                "semantic_bucket": "operating_revenue",
                "semantic_subbucket": "rent",
                "2026-01": 10,
            },
        ],
    )
    _write(
        tables / "monthly_tables_opex_by_type_amount_out.csv",
        [
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_subbucket": "services",
                "2026-01": 100,
            }
        ],
    )
    _write(
        tables / "monthly_tables_draws_by_box_amount_out.csv",
        [{"Currency": "ARS", "Box": "Household", "2026-01": 20}],
    )
    _write(
        tables / "overview_balance_dashboard.csv",
        [
            {"Currency": "ARS", "metric_id": "IS.RENT.TOTAL", "2026": 70},
            {
                "Currency": "ARS",
                "metric_id": "IS.OPEX.PROPERTY",
                "2026": 100,
            },
            {
                "Currency": "ARS",
                "metric_id": "DIST.DRAWS.PERSONAL",
                "2026": 20,
            },
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    _assert_case(pack, index, "flows")

    opex = _match(
        index,
        _expected("flows").query("expectation_id == 'monthly_opex'").iloc[0],
    )
    detail = pd.read_csv(pack / opex["detail_csv_relpath"]).fillna("")
    assert "hh_expense" not in detail.astype(str).to_string()
    if "Box" in detail.columns:
        assert "Household" not in set(detail["Box"].astype(str))

    annual_rows = index[index["period"].astype(str).eq("2026")]
    assert set(annual_rows["lineage_level"]) == {"annual_governed_membership"}


def test_professional_corpus_funding_support_distinction(tmp_path: Path) -> None:
    repo, run, pack = _paths(tmp_path, "funding")
    tables = pack / "tables"

    split = [
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "funding_contribution",
            "semantic_subbucket": "tenant_cash_support",
            "funding_actor": "Inquilino",
            "funding_channel": "tenant_to_box",
            "target_box": "Property Management",
            "cash_effect": "cash_in_box",
            "debt_effect": "none",
            "amount_in": 50,
            "amount_out": 0,
            "net_amount": 50,
            "amount_abs": 50,
            "source_tx_ids_sample": "fund_cash",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "property_opex",
            "semantic_subbucket": "taxes",
            "funding_actor": "Inquilino",
            "funding_channel": "tenant_direct_tax_payment",
            "target_box": "Property Management",
            "obligation_box": "Property Management",
            "cash_effect": "no_cash_in_box_direct_payment",
            "debt_effect": "none",
            "amount_in": 0,
            "amount_out": 30,
            "net_amount": -30,
            "amount_abs": 30,
            "source_tx_ids_sample": "support_direct",
        },
        {
            "period": "2026-01",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "debt_movement",
            "semantic_subbucket": "principal",
            "funding_actor": "Matías",
            "funding_channel": "debt_creation",
            "target_box": "Property Management",
            "cash_effect": "cash_in_box",
            "debt_effect": "creates_debt",
            "linked_debt_id": "D-1",
            "amount_in": 90,
            "amount_out": 0,
            "net_amount": 90,
            "amount_abs": 90,
            "source_tx_ids_sample": "support_debt",
        },
    ]
    _write(run / "monthly_flow_semantic_split.csv", split)
    _write(
        run / "classification_audit.csv",
        [
            {
                "tx_id": row["source_tx_ids_sample"],
                "period": row["period"],
                "Currency": row["Currency"],
                "Box": row["Box"],
                "semantic_bucket": row["semantic_bucket"],
                "semantic_subbucket": row["semantic_subbucket"],
                "funding_actor": row.get("funding_actor", ""),
                "funding_channel": row.get("funding_channel", ""),
                "target_box": row.get("target_box", ""),
                "obligation_box": row.get("obligation_box", ""),
                "cash_effect": row.get("cash_effect", ""),
                "debt_effect": row.get("debt_effect", ""),
            }
            for row in split
        ],
    )
    _write(
        run / "annual_balance_dashboard_metrics.csv",
        [
            {
                "metric_id": "FUND.CONTRIB.TOTAL",
                "period": "2026",
                "Currency": "ARS",
                "value": 50,
                "value_status": "available",
            },
            {
                "metric_id": "FUND.CONTRIB.BY_CHANNEL",
                "period": "2026",
                "Currency": "ARS",
                "dimension_name": "funding_channel",
                "dimension_value": "tenant_direct_tax_payment",
                "value": 30,
                "value_status": "available",
            },
            {
                "metric_id": "FUND.CONTRIB.DEBT_LINKED",
                "period": "2026",
                "Currency": "ARS",
                "value": 90,
                "value_status": "available",
            },
        ],
    )
    _write(
        tables / "overview_balance_dashboard.csv",
        [
            {"Currency": "ARS", "metric_id": "FUND.CONTRIB.TOTAL", "2026": 50},
            {
                "Currency": "ARS",
                "metric_id": "FUND.CONTRIB.BY_CHANNEL",
                "dimension_name": "funding_channel",
                "dimension_value": "tenant_direct_tax_payment",
                "2026": 30,
            },
            {
                "Currency": "ARS",
                "metric_id": "FUND.CONTRIB.DEBT_LINKED",
                "2026": 90,
            },
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    _assert_case(pack, index, "funding")

    core = _match(index, _expected("funding").query("expectation_id == 'core_funding'").iloc[0])
    direct = _match(index, _expected("funding").query("expectation_id == 'direct_support'").iloc[0])
    debt_linked = _match(index, _expected("funding").query("expectation_id == 'debt_linked_support'").iloc[0])
    assert "core_contribution" in core["filter_json"]
    assert "direct_obligation_payment" in direct["filter_json"]
    assert "debt_linked_support" in debt_linked["filter_json"]


def test_professional_corpus_debt_stock_and_activity(tmp_path: Path) -> None:
    repo, run, pack = _paths(tmp_path, "debt")
    tables = pack / "tables"

    _write(
        run / "monthly_debt_position.csv",
        [
            {
                "period": "2025-03",
                "as_of_date": "2025-03-31",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 850,
                "open_principal": 850,
                "open_interest": 20,
                "open_total": 870,
            },
            {
                "period": "2025-04",
                "as_of_date": "not-a-date-z",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 700,
                "open_principal": 700,
                "open_interest": 0,
                "open_total": 700,
            },
        ],
    )
    _write(
        run / "monthly_debt_activity.csv",
        [
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "new_principal": 0,
                "interest_accrued": 0,
                "repayments": 180,
                "adjustments": 0,
                "net_change": 0,
            },
            {
                "period": "2025-04",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "new_principal": 0,
                "interest_accrued": 0,
                "repayments": 170,
                "adjustments": 0,
                "net_change": 0,
            },
        ],
    )
    _write(
        tables / "monthly_tables_debt_position_matrix.csv",
        [
            {
                "measure": "open_principal",
                "Currency": "USD",
                "pair": "PM → MI",
                "2025-03": 850,
                "2025-04": 700,
            }
        ],
    )
    _write(
        tables / "monthly_tables_debt_activity_matrix.csv",
        [
            {
                "measure": "repayments",
                "Currency": "USD",
                "pair": "PM → MI",
                "2025-03": 180,
                "2025-04": 170,
            }
        ],
    )
    _write(
        tables / "annual_debt_activity_by_pair_wide.csv",
        [
            {
                "metric_id": "DEBT.ACTIVITY.REPAYMENT.BY_PAIR",
                "line_id": "debt.repayment",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayments",
                "2025": 350,
            }
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    _assert_case(pack, index, "debt")

    invalid = _match(
        index,
        _expected("debt").query("expectation_id == 'debt_position_invalid'").iloc[0],
    )
    assert json.loads(invalid["filter_json"])["valid_as_of_rows"] == 0


def _cash_row(
    *,
    period: str,
    amount: float,
    position_type: str,
    as_of_date: str,
    account_id: str = "",
    safe: bool = False,
) -> dict[str, object]:
    return {
        "period": period,
        "period_end": as_of_date,
        "as_of_date": as_of_date,
        "Box": "Property Management",
        "party": "",
        "account_id": account_id,
        "account_name": account_id,
        "Currency": "ARS",
        "close_amount": amount,
        "source_table": (
            "validated_cash_close.csv"
            if position_type == "cash_close"
            else "box_balance_time_long.freq=M.csv"
        ),
        "source_date": as_of_date,
        "source_type": (
            "bank_statement"
            if position_type == "cash_close"
            else "inferred_box_motor"
        ),
        "source_reference": "synthetic-professional-corpus",
        "validation_status": "validated" if safe else "",
        "validated_by": "controller" if safe else "",
        "position_type": position_type,
        "cash_suitability": "frontend_safe" if safe else "safe_with_caveat",
        "is_frontend_safe": safe,
        "caveat": "synthetic fixture",
        "notes": "",
        "n_source_rows": 1,
        "calculation_rule": "fixture",
    }


def test_professional_corpus_cash_available_and_unavailable(tmp_path: Path) -> None:
    repo, run, pack = _paths(tmp_path, "cash")
    tables = pack / "tables"

    _write(
        run / "monthly_cash_close.csv",
        [
            _cash_row(
                period="2026-01",
                amount=70,
                position_type="cash_close",
                as_of_date="2026-01-31",
                account_id="bank-a",
                safe=True,
            ),
            _cash_row(
                period="2026-01",
                amount=30,
                position_type="cash_close",
                as_of_date="2026-01-31",
                account_id="bank-b",
                safe=True,
            ),
            _cash_row(
                period="2026-01",
                amount=100,
                position_type="inferred_box_motor",
                as_of_date="2026-01-31",
            ),
            _cash_row(
                period="2026-02",
                amount=70,
                position_type="cash_close",
                as_of_date="2026-02-28",
                account_id="bank-a",
                safe=True,
            ),
            _cash_row(
                period="2026-02",
                amount=30,
                position_type="cash_close",
                as_of_date="not-a-date",
                account_id="bank-b",
                safe=True,
            ),
            _cash_row(
                period="2026-02",
                amount=100,
                position_type="inferred_box_motor",
                as_of_date="2026-02-28",
            ),
        ],
    )
    _write(
        tables / "monthly_tables_cash_close_matrix.csv",
        [
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "metric": "cash_close",
                "2026-01": 100,
                "2026-02": 100,
            }
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    _assert_case(pack, index, "cash")

    available = _match(
        index,
        _expected("cash").query("expectation_id == 'cash_available'").iloc[0],
    )
    assert set(
        pd.read_csv(pack / available["detail_csv_relpath"])["account_id"]
    ) == {"bank-a", "bank-b"}

    unavailable = _match(
        index,
        _expected("cash").query("expectation_id == 'cash_unavailable'").iloc[0],
    )
    assert unavailable["status"] == "unsupported"
    assert float(unavailable["matched_value_sum"]) == 0.0


def test_professional_corpus_governed_derived_metric(tmp_path: Path) -> None:
    repo, run, pack = _paths(tmp_path, "derived")
    tables = pack / "tables"

    _write(
        run / "annual_balance_dashboard_metrics.csv",
        [
            {
                "metric_id": "IS.REVENUE.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 1000,
                "value_status": "available",
            },
            {
                "metric_id": "IS.RENT.TOTAL",
                "period": "2026",
                "Currency": "ARS",
                "value": 800,
                "value_status": "available",
            },
            {
                "metric_id": "IS.OPEX.PROPERTY",
                "period": "2026",
                "Currency": "ARS",
                "value": 200,
                "value_status": "available",
            },
        ],
    )
    _write(
        tables / "overview_balance_dashboard.csv",
        [
            {
                "Currency": "ARS",
                "metric": "OPEX / renta",
                "derived_metric_id": "derived.opex_to_rent",
                "2026": 0.25,
            }
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    _assert_case(pack, index, "derived")
    assert _match(index, _expected("derived").iloc[0])["lineage_level"] == (
        "governed_derived_formula"
    )


def test_professional_corpus_fx_total_and_box_grains(tmp_path: Path) -> None:
    repo, run, pack = _paths(tmp_path, "fx")
    tables = pack / "tables"

    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Property Management",
                "cash_path": "Cambio:FX",
                "payer": "FX",
                "receiver": "PM",
                "semantic_bucket": "treasury_fx",
                "semantic_subbucket": "fx_conversion_proceeds",
                "amount_in": 200,
                "amount_out": 0,
                "net_amount": 200,
                "amount_abs": 200,
                "n_tx": 1,
                "source_tx_ids_sample": "fx_pm",
            }
        ],
    )
    _write(
        run / "classification_audit.csv",
        [
            {
                "tx_id": "fx_pm",
                "period": "2026-01",
                "Currency": "ARS",
                "amount": 200,
                "Box": "Property Management",
                "cash_path": "Cambio:FX",
                "semantic_bucket": "treasury_fx",
                "semantic_subbucket": "fx_conversion_proceeds",
            }
        ],
    )
    _write(
        tables / "monthly_tables_fx_treasury_amount_in.csv",
        [{"Currency": "ARS", "2026-01": 200}],
    )
    _write(
        tables / "cash_annual_box_flow_bridge_wide.csv",
        [
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "treasury_fx",
                "semantic_subbucket": "fx_conversion_proceeds",
                "measure": "net_amount",
                "line": "fx_flow_bridge",
                "2026": 200,
            }
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    _assert_case(pack, pd.read_csv(paths["index"]), "fx")
