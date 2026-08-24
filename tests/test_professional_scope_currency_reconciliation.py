from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.professional import drilldown as professional


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_representative_hh_pack_does_not_leak_household_into_property_opex(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest_HH"
    pack = repo / "out" / "professional_pack" / "latest_HH"
    tables = pack / "tables"

    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "amount_in": 0.0,
                "amount_out": 30.0,
                "net_amount": -30.0,
                "amount_abs": 30.0,
                "source_tx_ids_sample": "pm-opex",
            },
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Household",
                "semantic_bucket": "household_expense",
                "semantic_subbucket": "services",
                "amount_in": 0.0,
                "amount_out": 999.0,
                "net_amount": -999.0,
                "amount_abs": 999.0,
                "source_tx_ids_sample": "hh-expense",
            },
        ],
    )
    _write(
        tables / "monthly_tables_operating_statement_matrix.csv",
        [
            {
                "statement_line": "property_opex_true",
                "Currency": "ARS",
                "2026-01": 30.0,
            }
        ],
    )

    paths = professional.build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    row = index[
        index["table_id"].eq("monthly_tables_operating_statement_matrix")
    ].iloc[0]
    assert row["status"] == "ok"
    assert float(row["matched_value_sum"]) == 30.0
    assert str(row["lineage_level"]).startswith("governed_atomic_flow")

    detail = pd.read_csv(pack / row["detail_csv_relpath"])
    if "Box" in detail.columns:
        assert "Household" not in set(detail["Box"].fillna("").astype(str))


def test_representative_fbpm_pack_reconciles_native_currencies_and_derived_metrics(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest_FBPM"
    pack = repo / "out" / "professional_pack" / "latest_FBPM"
    tables = pack / "tables"

    split_rows: list[dict[str, object]] = []
    for currency, rent, opex in [("ARS", 120.0, 30.0), ("USD", 12.0, 3.0)]:
        split_rows.extend(
            [
                {
                    "period": "2026-01",
                    "Currency": currency,
                    "Box": "Family Business",
                    "semantic_bucket": "operating_revenue",
                    "semantic_subbucket": "rent",
                    "amount_in": rent,
                    "amount_out": 0.0,
                    "net_amount": rent,
                    "amount_abs": rent,
                },
                {
                    "period": "2026-01",
                    "Currency": currency,
                    "Box": "Property Management",
                    "semantic_bucket": "property_opex",
                    "semantic_subbucket": "services",
                    "amount_in": 0.0,
                    "amount_out": opex,
                    "net_amount": -opex,
                    "amount_abs": opex,
                },
            ]
        )
    _write(run / "monthly_flow_semantic_split.csv", split_rows)
    _write(
        tables / "monthly_tables_operating_statement_matrix.csv",
        [
            {
                "statement_line": "rent_revenue",
                "Currency": "ARS",
                "2026-01": 120.0,
            },
            {
                "statement_line": "property_opex_true",
                "Currency": "ARS",
                "2026-01": 30.0,
            },
            {
                "statement_line": "rent_revenue",
                "Currency": "USD",
                "2026-01": 12.0,
            },
            {
                "statement_line": "property_opex_true",
                "Currency": "USD",
                "2026-01": 3.0,
            },
        ],
    )

    annual_rows: list[dict[str, object]] = []
    for currency, revenue, rent, opex, net, funding, draws, coverage in [
        ("ARS", 120.0, 120.0, 30.0, 90.0, 10.0, 20.0, 80.0),
        ("USD", 12.0, 12.0, 3.0, 9.0, 1.0, 2.0, 8.0),
    ]:
        for metric_id, value in [
            ("IS.REVENUE.OPERATING", revenue),
            ("IS.RENT.TOTAL", rent),
            ("IS.OPEX.PROPERTY", opex),
            ("IS.NET.OPERATING", net),
            ("FUND.CONTRIB.TOTAL", funding),
            ("DIST.DRAWS.PERSONAL", draws),
            ("COV.NET.AFTER_DRAWS", coverage),
        ]:
            annual_rows.append(
                {
                    "metric_id": metric_id,
                    "period": "2026",
                    "Currency": currency,
                    "value": value,
                    "value_status": "available",
                }
            )
    _write(run / "annual_balance_dashboard_metrics.csv", annual_rows)
    _write(
        tables / "overview_balance_dashboard.csv",
        [
            {
                "Currency": "ARS",
                "metric": "Margen operativo",
                "2026": 0.75,
            },
            {"Currency": "ARS", "metric": "OPEX / renta", "2026": 0.25},
            {
                "Currency": "ARS",
                "metric": "Cobertura después de funding y retiros",
                "2026": 80.0,
            },
            {
                "Currency": "USD",
                "metric": "Margen operativo",
                "2026": 0.75,
            },
            {"Currency": "USD", "metric": "OPEX / renta", "2026": 0.25},
            {
                "Currency": "USD",
                "metric": "Cobertura después de funding y retiros",
                "2026": 8.0,
            },
        ],
    )

    paths = professional.build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    monthly = index[
        index["table_id"].eq("monthly_tables_operating_statement_matrix")
    ]
    assert len(monthly) == 4
    assert set(monthly["status"]) == {"ok"}
    assert set(monthly["Currency"]) == {"ARS", "USD"}
    assert all(
        str(value).startswith("governed_atomic_flow")
        for value in monthly["lineage_level"]
    )

    annual = index[index["table_id"].eq("overview_balance_dashboard")]
    assert len(annual) == 6
    assert set(annual["status"]) == {"ok"}
    assert set(annual["Currency"]) == {"ARS", "USD"}
    assert set(annual["lineage_level"]) == {
        "governed_derived_formula",
        "governed_source_value_with_formula_reconciliation",
    }
