from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.professional.drilldown import build_professional_flow_drilldowns


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(index: pd.DataFrame, table_id: str, *, contains: str = "") -> pd.Series:
    rows = index[index["table_id"].eq(table_id)].copy()
    if contains:
        rows = rows[
            rows["row_context_json"].fillna("").astype(str).str.contains(contains, regex=False)
            | rows["filter_json"].fillna("").astype(str).str.contains(contains, regex=False)
        ]
    assert len(rows) == 1, rows[["table_id", "row_context_json", "filter_json"]].to_dict("records")
    return rows.iloc[0]


def test_fx_drilldown_authority_reconciles_explicit_measure_and_grain(tmp_path: Path) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_flow_semantic_split.csv",
        [
            {"period":"2026-01","Currency":"ARS","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_proceeds","amount_in":200,"amount_out":0,"net_amount":200,"amount_abs":200,"n_tx":1,"source_tx_ids_sample":"ars_pm_in"},
            {"period":"2026-01","Currency":"ARS","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_outflow","amount_in":0,"amount_out":70,"net_amount":-70,"amount_abs":70,"n_tx":1,"source_tx_ids_sample":"ars_pm_out"},
            {"period":"2026-01","Currency":"ARS","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_cost_or_spread","amount_in":0,"amount_out":5,"net_amount":-5,"amount_abs":5,"n_tx":1,"source_tx_ids_sample":"ars_pm_cost"},
            {"period":"2026-01","Currency":"ARS","Box":"Family Business","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_proceeds","amount_in":30,"amount_out":0,"net_amount":30,"amount_abs":30,"n_tx":1,"source_tx_ids_sample":"ars_fb_in"},
            {"period":"2026-01","Currency":"USD","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_proceeds","amount_in":9,"amount_out":0,"net_amount":9,"amount_abs":9,"n_tx":1,"source_tx_ids_sample":"usd_pm_in"},
            {"period":"2026-01","Currency":"USD","Box":"Property Management","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_outflow","amount_in":0,"amount_out":4,"net_amount":-4,"amount_abs":4,"n_tx":1,"source_tx_ids_sample":"usd_pm_out"},
        ],
    )
    _write(
        tables / "monthly_tables_fx_treasury_amount_in.csv",
        [
            {"Currency":"ARS","2026-01":230},
            {"Currency":"USD","2026-01":9},
        ],
    )
    _write(
        tables / "monthly_tables_fx_treasury_all_measures.csv",
        [
            {"measure":"amount_out","Currency":"ARS","Box":"Property Management","2026-01":75},
            {"measure":"net_amount","Currency":"ARS","Box":"Property Management","2026-01":125},
            {"measure":"amount_abs","Currency":"ARS","Box":"Property Management","2026-01":275},
            {"Currency":"ARS","Box":"Property Management","drilldown_cell_id":"flow.fx.conversion_proceeds","2026-01":200},
            {"Currency":"ARS","drilldown_cell_id":"flow.fx.conversion_proceeds","2026-01":230},
        ],
    )
    _write(
        tables / "monthly_tables_fx_treasury_compact.csv",
        [{"Currency":"ARS","2026-01":155}],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    amount_in = index[index["table_id"].eq("monthly_tables_fx_treasury_amount_in")].copy()
    assert set(amount_in["Currency"]) == {"ARS", "USD"}
    assert dict(zip(amount_in["Currency"], amount_in["matched_value_sum"].astype(float))) == {
        "ARS": 230.0,
        "USD": 9.0,
    }
    assert set(amount_in["status"]) == {"ok"}

    amount_out = _row(
        index,
        "monthly_tables_fx_treasury_all_measures",
        contains='"measure": "amount_out"',
    )
    assert amount_out["status"] == "ok"
    assert float(amount_out["matched_value_sum"]) == 75.0
    assert float(amount_out["residual"]) == 0.0

    net = _row(
        index,
        "monthly_tables_fx_treasury_all_measures",
        contains='"measure": "net_amount"',
    )
    assert net["status"] == "ok"
    assert float(net["matched_value_sum"]) == 125.0

    absolute = _row(
        index,
        "monthly_tables_fx_treasury_all_measures",
        contains='"measure": "amount_abs"',
    )
    assert absolute["status"] == "ok"
    assert float(absolute["matched_value_sum"]) == 275.0

    governed_box = _row(
        index,
        "monthly_tables_fx_treasury_all_measures",
        contains="flow.fx.conversion_proceeds",
    )
    governed_candidates = index[
        index["table_id"].eq("monthly_tables_fx_treasury_all_measures")
        & index["row_context_json"].fillna("").str.contains("flow.fx.conversion_proceeds", regex=False)
    ].copy()
    assert len(governed_candidates) == 2
    supported = governed_candidates[governed_candidates["status"].eq("ok")].iloc[0]
    unsupported = governed_candidates[governed_candidates["status"].eq("unsupported")].iloc[0]
    assert float(supported["matched_value_sum"]) == 200.0
    assert "Property Management" in supported["row_context_json"]
    assert float(unsupported["matched_value_sum"]) == 0.0
    assert "missing Box" in unsupported["caveat"]

    compact = _row(index, "monthly_tables_fx_treasury_compact")
    assert compact["status"] == "unsupported"
    assert float(compact["matched_value_sum"]) == 0.0
    assert compact["measure"] != "net_amount"
    assert "no explicit recognized measure" in compact["caveat"]
