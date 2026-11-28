import json
from pathlib import Path
import pytest
from accounting.reports.treasury_accountability.projection import rolling_horizon, projection_spec
from accounting.cutoff import cutoff_metadata
from accounting.scenarios.runner import digest

@pytest.mark.parametrize(("day","end"),[
    ("2026-11-28","2027-04-30"),("2026-12-01","2027-05-31"),
    ("2027-01-01","2027-06-30"),("2027-09-01","2028-02-29"),
    ("2026-11-01","2027-04-30"),("2026-11-30","2027-04-30")])
def test_six_months_including_current(day,end):
    assert rolling_horizon(day)==end

def test_projection_uses_frozen_evaluation_and_separate_plans(tmp_path):
    p=tmp_path/"20261128T094818Z_FBPM"
    (p/"meta").mkdir(parents=True)
    plan=p/"ledger_planning_snapshot.csv"
    plan.write_text("tx_id,Date,status\na,2026-12-01,planeado\n")
    m={"params":{**cutoff_metadata("2026-10-31"),"planning_evaluation_date":"2026-11-28"},
       "outputs":[{"name":"ledger_planning_snapshot","sha256":digest(plan)}]}
    (p/"meta/stage_A_ingest.json").write_text(json.dumps(m))
    spec,source=projection_spec(p)
    assert spec=={"as_of_date":"2026-11-28","through_date":"2027-04-30","observed_as_of_date":"2026-10-31"}
    assert source==plan
    plan.write_text("tampered")
    with pytest.raises(ValueError,match="differs"):projection_spec(p)
