from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from accounting.professional.drilldown_wave4_base import _ORIGINAL_SPEC_FOR_CELL
from accounting.professional.table_contracts import enrich_professional_table

from accounting.professional import drilldown as professional
from accounting.professional.table_contracts import enrich_professional_table


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "diagnostics" / "final_compatibility_reachability_20260819.csv"
CENSUS = ROOT / "diagnostics" / "semantic_authority_census_20260819.csv"
WAVE3 = ROOT / "diagnostics" / "atomic_flow_reachability_20260819.csv"


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_final_compatibility_inventory_is_exhaustive_and_evidence_classified() -> None:
    rows = _csv(INVENTORY)
    assert len(rows) >= 20
    allowed = {
        "REQUIRED_COMPATIBILITY",
        "MODERN_REACHABLE_BUG",
        "DEAD",
        "UPSTREAM_FIX_REQUIRED",
    }
    assert {row["classification"] for row in rows} <= allowed
    assert all(row["evidence"].strip() for row in rows)
    assert all(row["immediate_action"].strip() for row in rows)

    # The final audit found no accidental modern-reachable legacy override.
    # Remaining modern deferrals are explicit contract/upstream work, not silent bugs.
    assert not [row for row in rows if row["classification"] == "MODERN_REACHABLE_BUG"]

    dead = {row["route_id"] for row in rows if row["classification"] == "DEAD"}
    assert dead == {
        "legacy.cellspec.draws_by_box",
        "legacy.cellspec.draws_by_type_amount_out",
        "legacy.cellspec.opex_by_type_amount_out",
    }


def test_dead_classification_is_backed_by_prior_wave3_reachability_evidence() -> None:
    wave3 = _csv(WAVE3)
    safe = {
        row["case_id"]
        for row in wave3
        if row["safe_to_delete"].strip().lower() == "true"
    }
    assert safe == {
        "draws_by_box_monthly",
        "draws_by_type_monthly",
        "opex_by_box_category_monthly",
    }

    dead_to_case = {
        "legacy.cellspec.draws_by_box": "draws_by_box_monthly",
        "legacy.cellspec.draws_by_type_amount_out": "draws_by_type_monthly",
        "legacy.cellspec.opex_by_type_amount_out": "opex_by_box_category_monthly",
    }
    final_dead = {
        row["route_id"]
        for row in _csv(INVENTORY)
        if row["classification"] == "DEAD"
    }
    assert {dead_to_case[route] for route in final_dead} == safe


def test_proven_dead_atomic_cellspec_routes_are_physically_pruned() -> None:
    cases = [
        (
            "monthly_tables_draws_by_box_amount_out",
            {"Currency": "ARS", "Box": "Household", "2026-01": 10},
            "flow.draws.by_box",
        ),
        (
            "monthly_tables_draws_by_type_amount_out",
            {
                "Currency": "ARS",
                "semantic_subbucket": "personal_expense",
                "2026-01": 10,
            },
            "flow.draws.by_type",
        ),
        (
            "monthly_tables_opex_by_type_amount_out",
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "semantic_subbucket": "services",
                "2026-01": 10,
            },
            "flow.property_opex.by_box_category",
        ),
    ]

    for table_id, raw, governed_id in cases:
        row = enrich_professional_table(pd.DataFrame([raw]), table_id).iloc[0]
        assert row["drilldown_cell_id"] == governed_id
        assert _ORIGINAL_SPEC_FOR_CELL(table_id, row) is None


def test_semantic_authority_census_has_one_production_authority_per_core_concept() -> None:
    rows = _csv(CENSUS)
    expected = {
        "property_opex_membership",
        "funding_contribution_membership",
        "family_draws_membership",
        "fx_atomic_measure",
        "debt_position",
        "debt_activity",
        "validated_cash_position",
        "inferred_box_control",
        "derived_formula_definition",
    }
    assert {row["concept"] for row in rows} == expected
    assert all(int(row["authority_count"]) == 1 for row in rows)
    assert all(row["compatibility_can_override"] == "no" for row in rows)
    assert all(row["status"].startswith("GOVERNED") for row in rows)


def test_legacy_modules_are_still_reachable_and_must_not_be_deleted_wholesale() -> None:
    drilldown = (ROOT / "accounting" / "professional" / "drilldown.py").read_text(encoding="utf-8")
    wave4 = (ROOT / "accounting" / "professional" / "drilldown_wave4_base.py").read_text(encoding="utf-8")
    annual = (ROOT / "accounting" / "metrics" / "annual.py").read_text(encoding="utf-8")
    companion = (ROOT / "accounting" / "professional" / "annual_dashboard_tables.py").read_text(encoding="utf-8")

    assert "drilldown_wave4_base as _base" in drilldown
    assert "drilldown_legacy as _legacy" in wave4
    assert "annual_legacy as _legacy" in annual
    assert "annual_dashboard_tables_legacy as _legacy" in companion


def test_governed_identity_cannot_fall_back_to_legacy_execution(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("governed identity reached legacy execution fallback")

    # Derived-table path metadata is allowed to retain the historical CellSpec
    # identity. What must never fall back is the actual matched-value execution.
    atomic_row = enrich_professional_table(
        pd.DataFrame(
            [
                {
                    "statement_line": "property_opex_true",
                    "Currency": "ARS",
                    "2026-01": 30.0,
                }
            ]
        ),
        "monthly_tables_operating_statement_matrix",
    ).iloc[0]
    assert atomic_row["drilldown_cell_id"] == "flow.property_opex.total"
    monkeypatch.setattr(professional._base, "_ORIGINAL_BUILD_DERIVED_CELL", explode)
    atomic_result = professional._base._build_derived_cell(
        table_id="monthly_tables_operating_statement_matrix",
        row=atomic_row,
        period="2026-01",
        display_value=30.0,
        split=pd.DataFrame(
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
                }
            ]
        ),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=pd.DataFrame(),
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert atomic_result[0] == "ok"
    assert str(atomic_result[3]).startswith("governed_atomic_flow")

    # Derived metric: modern schema + stable derived ID must also execute governed.
    monkeypatch.setattr(professional, "_ORIGINAL_BUILD_DERIVED_CELL", explode)
    annual = pd.DataFrame(
        [
            {
                "metric_id": "IS.NET.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 75.0,
                "value_status": "available",
            },
            {
                "metric_id": "IS.REVENUE.OPERATING",
                "period": "2026",
                "Currency": "ARS",
                "value": 100.0,
                "value_status": "available",
            },
        ]
    )
    result = professional._build_derived_cell(
        table_id="overview_balance_dashboard",
        row=pd.Series(
            {
                "Currency": "ARS",
                "derived_metric_id": "derived.operating_margin",
            }
        ),
        period="2026",
        display_value=0.75,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=annual,
        cash_close=pd.DataFrame(),
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert result[0] == "ok"
    assert result[3] == "governed_derived_formula"


def test_representative_hh_pack_does_not_leak_household_into_property_opex(tmp_path: Path) -> None:
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


def test_representative_fbpm_pack_reconciles_native_currencies_and_derived_metrics(tmp_path: Path) -> None:
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
            {"statement_line": "rent_revenue", "Currency": "ARS", "2026-01": 120.0},
            {"statement_line": "property_opex_true", "Currency": "ARS", "2026-01": 30.0},
            {"statement_line": "rent_revenue", "Currency": "USD", "2026-01": 12.0},
            {"statement_line": "property_opex_true", "Currency": "USD", "2026-01": 3.0},
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
            {"Currency": "ARS", "metric": "Margen operativo", "2026": 0.75},
            {"Currency": "ARS", "metric": "OPEX / renta", "2026": 0.25},
            {"Currency": "ARS", "metric": "Cobertura después de funding y retiros", "2026": 80.0},
            {"Currency": "USD", "metric": "Margen operativo", "2026": 0.75},
            {"Currency": "USD", "metric": "OPEX / renta", "2026": 0.25},
            {"Currency": "USD", "metric": "Cobertura después de funding y retiros", "2026": 8.0},
        ],
    )

    paths = professional.build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    monthly = index[index["table_id"].eq("monthly_tables_operating_statement_matrix")]
    assert len(monthly) == 4
    assert set(monthly["status"]) == {"ok"}
    assert set(monthly["Currency"]) == {"ARS", "USD"}
    assert all(str(x).startswith("governed_atomic_flow") for x in monthly["lineage_level"])

    annual = index[index["table_id"].eq("overview_balance_dashboard")]
    assert len(annual) == 6
    assert set(annual["status"]) == {"ok"}
    assert set(annual["Currency"]) == {"ARS", "USD"}
    assert set(annual["lineage_level"]) == {
        "governed_derived_formula",
        "governed_source_value_with_formula_reconciliation",
    }
