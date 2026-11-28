"""Default rolling treasury forecast using isolated existing accounting stages."""
from calendar import monthrange
from datetime import date, datetime
import json
from pathlib import Path
import shutil
import pandas as pd
from accounting.cutoff import load_run_cutoff_if_present
from accounting.scenarios.runner import digest, seed_execution
from accounting.support.isolated_execution import isolated_accounting_execution

DEFAULT_MONTHS = 6

def rolling_horizon(evaluation_date: str, months: int = DEFAULT_MONTHS) -> str:
    day = date.fromisoformat(evaluation_date)
    if type(months) is not int or not 1 <= months <= 36:
        raise ValueError("forecast months must be in 1..36")
    index = day.year * 12 + day.month - 1 + months - 1
    year, month0 = divmod(index, 12)
    month = month0 + 1
    return date(year, month, monthrange(year, month)[1]).isoformat()

def projection_spec(run_root: Path) -> tuple[dict, Path]:
    root=Path(run_root)
    manifest=json.loads((root/"meta/stage_A_ingest.json").read_text())
    params=manifest.get("params",{})
    cutoff=load_run_cutoff_if_present(root)
    evaluation=params.get("planning_evaluation_date") or datetime.strptime(root.name[:8],"%Y%m%d").date().isoformat()
    observed=cutoff.date if cutoff else evaluation
    if observed>evaluation:
        raise ValueError("observed cutoff follows forecast evaluation")
    plan=root/"ledger_planning_snapshot.csv"
    if not plan.is_file():
        if cutoff:
            raise ValueError("bounded run lacks planning snapshot; re-ingest to generate default treasury forecast")
        plan=root/"ledger_canonical_all_status.csv"
    if plan.name=="ledger_planning_snapshot.csv":
        matches=[x for x in manifest["outputs"] if x.get("name")=="ledger_planning_snapshot"]
        if len(matches)!=1 or matches[0].get("sha256")!=digest(plan):
            raise ValueError("planning snapshot differs from Stage A manifest")
    return {"as_of_date":evaluation,"through_date":rolling_horizon(evaluation),"observed_as_of_date":observed},plan

def build_default_projection(run_root: Path, out_dir: Path):
    """Recalculate on a separate ledger; export only forecast inputs for the report."""
    spec,plan=projection_spec(run_root)
    repo=Path(__file__).resolve().parents[3]
    root=Path(run_root)
    out_dir.mkdir(parents=True,exist_ok=True)
    files=("ledger_canonical.csv","ledger_canonical_all_status.csv","monthly_cash_accountability.csv")
    before={f:digest(root/f) for f in files}
    with isolated_accounting_execution(repo, temp_parent=Path.home()) as execution:
        seed_execution(root,execution.canonical_root,execution.run_id,"default_rolling_treasury",
                       {"id":"planned_baseline"},projection=spec,projection_source=plan)
        execution.run_stage("run-materialize")
        execution.run_stage("run-debt")
        for name in ("monthly_cash_accountability.csv","monthly_cash_accountability_qa.csv"):
            shutil.copy2(execution.canonical_root/name,out_dir/name)
        projected=pd.read_csv(execution.canonical_root/"ledger_canonical.csv",keep_default_na=False)
        plans=projected.loc[projected.status.eq("planeado")].copy()
        plans.to_csv(out_dir/"selected_plans.csv",index=False)
    if before!={f:digest(root/f) for f in files}:
        raise ValueError("observed run changed during projection")
    # Closed observed months must stay identical; current month may include plans.
    hist=pd.read_csv(root/"monthly_cash_accountability.csv")
    future=pd.read_csv(out_dir/"monthly_cash_accountability.csv")
    key=["Box","Currency","period"]; values=["total_cash_in","total_cash_out","net_cash_flow","closing_control"]
    prior=spec["as_of_date"][:7]
    h=hist.loc[hist.period.lt(prior)].set_index(key)[values].sort_index()
    f=future.loc[future.period.lt(prior)].set_index(key)[values].sort_index()
    pd.testing.assert_frame_equal(h,f,check_exact=False,atol=.005,rtol=0)
    if not plans.empty and not plans.Date.astype(str).between(spec["as_of_date"],spec["through_date"]).all():
        raise ValueError("projected plan outside horizon")
    spec.update(planned_rows=len(plans),planning_source_sha256=digest(plan))
    (out_dir/"projection_manifest.json").write_text(json.dumps(spec,indent=2)+"\n")
    return spec,plan
