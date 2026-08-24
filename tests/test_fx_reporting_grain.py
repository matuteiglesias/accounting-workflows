from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from accounting.contracts.fx_reporting_grain import (
    FX_REPORTING_GRAIN_COLUMN,
    FX_REPORTING_GRAIN_VERSION,
    producer_fx_reporting_grain,
    resolve_fx_reporting_spec,
    validate_fx_row_grain,
)
from accounting.professional.drilldown import build_professional_flow_drilldowns
from accounting.professional.fx_reporting_executor import execute_fx_reporting_cell
from accounting.professional.fx_reporting_metadata import enrich_fx_reporting_grain_tables


def _split() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"period":"2026-01","Currency":"ARS","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_proceeds","amount_in":100.0,"amount_out":0.0,"net_amount":100.0,"amount_abs":100.0,"source_tx_ids_sample":"fx-in-pm"},
            {"period":"2026-01","Currency":"ARS","Box":"Family Business","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_proceeds","amount_in":40.0,"amount_out":0.0,"net_amount":40.0,"amount_abs":40.0,"source_tx_ids_sample":"fx-in-fb"},
            {"period":"2026-01","Currency":"ARS","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_outflow","amount_in":0.0,"amount_out":50.0,"net_amount":-50.0,"amount_abs":50.0,"source_tx_ids_sample":"fx-out-pm"},
            {"period":"2026-01","Currency":"ARS","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_cost_or_spread","amount_in":0.0,"amount_out":5.0,"net_amount":-5.0,"amount_abs":5.0,"source_tx_ids_sample":"fx-cost-pm"},
            {"period":"2026-01","Currency":"USD","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_proceeds","amount_in":7.0,"amount_out":0.0,"net_amount":7.0,"amount_abs":7.0,"source_tx_ids_sample":"fx-in-usd"},
        ]
    )


def test_grain_is_resolved_from_producer_identity_not_box_nullability() -> None:
    dedicated_missing_box = pd.Series({"Currency":"ARS","metric":"fx_conversion_proceeds"})
    assert producer_fx_reporting_grain(
        "monthly_tables_fx_treasury_all_measures", dedicated_missing_box
    ) == "box_currency"

    statement = pd.Series(
        {"Currency":"ARS","statement_line":"treasury_fx_conversion_in"}
    )
    assert producer_fx_reporting_grain(
        "monthly_tables_operating_statement_matrix", statement
    ) == "currency_total"

    unrelated_without_box = pd.Series({"Currency":"ARS"})
    assert producer_fx_reporting_grain("some_other_table", unrelated_without_box) is None


def test_contract_supports_four_measures_at_both_explicit_grains() -> None:
    for grain in ("currency_total", "box_currency"):
        rows = [
            ("TR.FX.CONVERSION.IN", "treasury_fx_conversion_in", "conversion_in", "amount_in"),
            ("TR.FX.CONVERSION.OUT", "treasury_fx_conversion_out", "conversion_out", "amount_out"),
            ("TR.FX.COST.OUT", "treasury_fx_cost", "fx_cost", "amount_out"),
            ("TR.FX.NET", "treasury_fx_net", "net", "net_amount"),
        ]
        for metric_id, statement_line, kind, measure in rows:
            table_id = (
                "monthly_tables_operating_statement_matrix"
                if grain == "currency_total"
                else "monthly_tables_fx_treasury_all_measures"
            )
            row = pd.Series(
                {
                    "metric_id": metric_id if grain == "currency_total" else "",
                    "statement_line": statement_line if grain == "currency_total" else "",
                    "metric": {
                        "conversion_in":"fx_conversion_proceeds",
                        "conversion_out":"fx_conversion_outflow",
                        "fx_cost":"fx_cost_or_spread",
                        "net":"fx_net",
                    }[kind] if grain == "box_currency" else "",
                    "Currency":"ARS",
                    "Box":"Property Management" if grain == "box_currency" else "",
                    FX_REPORTING_GRAIN_COLUMN: grain,
                }
            )
            spec = resolve_fx_reporting_spec(table_id, row)
            assert spec is not None
            assert spec.measure_kind == kind
            assert spec.measure_id == measure
            assert spec.grain == grain
            assert validate_fx_row_grain(row, spec) == (True, "")


def test_incompatible_grains_fail_closed() -> None:
    box_missing = pd.Series(
        {
            "metric":"fx_conversion_proceeds",
            "Currency":"ARS",
            FX_REPORTING_GRAIN_COLUMN:"box_currency",
        }
    )
    spec = resolve_fx_reporting_spec("monthly_tables_fx_treasury_all_measures", box_missing)
    assert spec is not None
    valid, reason = validate_fx_row_grain(box_missing, spec)
    assert not valid and "requires explicit Box" in reason

    total_with_box = pd.Series(
        {
            "metric_id":"TR.FX.CONVERSION.IN",
            "statement_line":"treasury_fx_conversion_in",
            "Currency":"ARS",
            "Box":"Property Management",
            FX_REPORTING_GRAIN_COLUMN:"currency_total",
        }
    )
    spec = resolve_fx_reporting_spec("monthly_tables_operating_statement_matrix", total_with_box)
    assert spec is not None
    valid, reason = validate_fx_row_grain(total_with_box, spec)
    assert not valid and "must not carry Box" in reason


def test_executor_keeps_currency_total_and_box_currency_distinct() -> None:
    split = _split()
    total_row = pd.Series(
        {
            "metric_id":"TR.FX.CONVERSION.IN",
            "statement_line":"treasury_fx_conversion_in",
            "Currency":"ARS",
            FX_REPORTING_GRAIN_COLUMN:"currency_total",
        }
    )
    result = execute_fx_reporting_cell(
        table_id="monthly_tables_operating_statement_matrix",
        row=total_row,
        period="2026-01",
        display_value=140.0,
        split=split,
        tolerance=1e-6,
    )
    assert result is not None
    assert result[0] == "ok"
    assert result[1] == 140.0
    assert set(result[7]["Box"]) == {"Property Management", "Family Business"}

    box_row = pd.Series(
        {
            "metric":"fx_conversion_proceeds",
            "Currency":"ARS",
            "Box":"Property Management",
            FX_REPORTING_GRAIN_COLUMN:"box_currency",
        }
    )
    result = execute_fx_reporting_cell(
        table_id="monthly_tables_fx_treasury_all_measures",
        row=box_row,
        period="2026-01",
        display_value=100.0,
        split=split,
        tolerance=1e-6,
    )
    assert result is not None
    assert result[0] == "ok"
    assert result[1] == 100.0
    assert set(result[7]["Box"]) == {"Property Management"}
    assert set(result[7]["Currency"]) == {"ARS"}


def test_professional_enrichment_and_routing_fail_closed_on_missing_box(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"
    run.mkdir(parents=True)
    tables.mkdir(parents=True)
    _split().to_csv(run / "monthly_flow_semantic_split.csv", index=False)

    pd.DataFrame(
        [
            {"statement_line":"treasury_fx_conversion_in","Currency":"ARS","2026-01":140.0},
        ]
    ).to_csv(tables / "monthly_tables_operating_statement_matrix.csv", index=False)
    pd.DataFrame(
        [
            {"metric":"fx_conversion_proceeds","Currency":"ARS","Box":"Property Management","2026-01":100.0},
            {"metric":"fx_conversion_proceeds","Currency":"ARS","Box":"","2026-01":140.0},
        ]
    ).to_csv(tables / "monthly_tables_fx_treasury_all_measures.csv", index=False)

    enriched = enrich_fx_reporting_grain_tables(tables)
    assert enriched
    dedicated = pd.read_csv(tables / "monthly_tables_fx_treasury_all_measures.csv", keep_default_na=False)
    assert set(dedicated[FX_REPORTING_GRAIN_COLUMN]) == {"box_currency"}
    statement = pd.read_csv(tables / "monthly_tables_operating_statement_matrix.csv", keep_default_na=False)
    assert set(statement[FX_REPORTING_GRAIN_COLUMN]) == {"currency_total"}

    paths = build_professional_flow_drilldowns(repo, pack, run_root=run)
    index = pd.read_csv(paths["index"])

    total = index[
        index["table_id"].eq("monthly_tables_operating_statement_matrix")
        & index["period"].astype(str).eq("2026-01")
    ]
    assert len(total) == 1
    assert total.iloc[0]["status"] == "ok"
    total_filters = json.loads(total.iloc[0]["filter_json"])
    assert total_filters["fx_reporting_grain"] == "currency_total"
    assert total_filters["lineage_version"] == FX_REPORTING_GRAIN_VERSION
    assert float(total.iloc[0]["matched_value_sum"]) == 140.0

    dedicated_rows = index[
        index["table_id"].eq("monthly_tables_fx_treasury_all_measures")
        & index["period"].astype(str).eq("2026-01")
    ].sort_values("display_value")
    assert len(dedicated_rows) == 2
    governed = dedicated_rows[dedicated_rows["display_value"].eq(100.0)].iloc[0]
    incompatible = dedicated_rows[dedicated_rows["display_value"].eq(140.0)].iloc[0]
    assert governed["status"] == "ok"
    assert float(governed["matched_value_sum"]) == 100.0
    assert incompatible["status"] == "unsupported"
    assert float(incompatible["matched_value_sum"]) == 0.0
