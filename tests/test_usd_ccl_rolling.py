import json
from accounting.management.usd_ccl_rolling import build_rolling_usd

def test_forward_fill_future_only_and_empty_month(tmp_path):
    observed=tmp_path/"observed.csv"; forecast=tmp_path/"forecast.csv"
    head="Box,period,Currency,reconciliation_status,total_cash_in,total_cash_out,net_cash_flow\n"
    observed.write_text(head+"PM,2026-11,ARS,reconciled,100,20,80\n")
    forecast.write_text(observed.read_text()+"PM,2026-12,ARS,reconciled,200,100,100\n")
    means=tmp_path/"mean.csv";means.write_text("period,ccl_mean_ars_per_usd\n2026-11,2\n")
    daily=tmp_path/"daily.csv";daily.write_text("rate_date,ars_per_usd_ccl\n2026-11-27,4\n2026-11-30,8\n")
    spec=tmp_path/"spec.json";spec.write_text(json.dumps(dict(as_of_date="2026-11-28",observed_as_of_date="2026-11-28",through_date="2027-01-31")))
    rows,meta=build_rolling_usd(observed=observed,forecast=forecast,means=means,daily=daily,projection_manifest=spec,output=tmp_path/"result.csv")
    assert [r["rate"] for r in rows]==["2","4","4"]
    assert [r["net_usd_ccl"] for r in rows]==["40.00","25.00","0.00"]
    assert rows[-1]["coverage"]=="no_rows_in_plan"
    assert meta["last_rate_date"]=="2026-11-27"
