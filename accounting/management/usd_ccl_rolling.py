"""Translate rolling treasury flows, keeping observed and assumed rates distinct."""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import csv
import hashlib
import json

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_rolling_usd(*, observed, forecast, means, daily, projection_manifest, output):
    spec=json.loads(Path(projection_manifest).read_text())
    cutoff=spec["observed_as_of_date"]; start=spec["as_of_date"][:7]; end=spec["through_date"][:7]
    rates={r["period"]:Decimal(r["ccl_mean_ars_per_usd"]) for r in read(means)}
    eligible=[r for r in read(daily) if r["rate_date"]<=cutoff]
    latest=max(eligible,key=lambda r:r["rate_date"])
    constant=Decimal(latest["ars_per_usd_ccl"])
    if constant<=0 or not constant.is_finite():raise ValueError("invalid forward-fill rate")
    def lane(path):
        data={}
        for r in read(path):
            if r["Currency"]!="ARS":continue
            key=(r["Box"],r["period"])
            if key in data or r["reconciliation_status"]!="reconciled":raise ValueError("invalid treasury grain or status")
            data[key]=r
        return data
    hist=lane(observed); projected=lane(forecast)
    for k,r in hist.items():
        if k[1]<start and any(abs(Decimal(r[f])-Decimal(projected[k][f]))>Decimal(".005") for f in ("total_cash_in","total_cash_out","net_cash_flow")):raise ValueError("forecast history mismatch")
    boxes=sorted({k[0] for k in hist}); y,m=map(int,start.split("-")); months=[]
    while f"{y:04d}-{m:02d}"<=end:
        months.append(f"{y:04d}-{m:02d}")
        y,m=(y+1,1) if m==12 else (y,m+1)
    result=[]
    for box in boxes:
        for month in months:
            k=(box,month); r=projected.get(k); actual=hist.get(k)
            rate=rates.get(month) if month<=cutoff[:7] else constant
            basis="observed_month_mean" if month<=cutoff[:7] else "forward_fill_last_daily"
            if rate is None: basis="missing_fx"
            row={"Box":box,"period":month,"rate":str(rate) if rate else "","rate_basis":basis,"cash_basis":"observed_plus_plans" if month==start else "planned","coverage":"recorded_rows" if r else "no_rows_in_plan"}
            for field,key in [("total_cash_in","in"),("total_cash_out","out"),("net_cash_flow","net")]:
                v=Decimal(r[field]) if r else Decimal(0)
                row[key+"_ars"]=str(v)
                row[key+"_usd_ccl"]=str((v/rate).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)) if rate else ""
            row["observed_net_usd_ccl"]=str((Decimal(actual["net_cash_flow"])/rate).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)) if actual and rate else ""
            result.append(row)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    manifest={"observed_sha256":digest(observed),"daily_sha256":digest(daily),"means_sha256":digest(means),"as_of_date":spec["as_of_date"],"through_date":spec["through_date"],"rate_assumption":"last_daily_ccl_constant","last_rate_date":latest["rate_date"],"last_rate":str(constant),"rows":len(result),"inputs":{str(p):digest(p) for p in [observed,forecast,means,daily,projection_manifest]},"output_sha256":digest(output)}
    output.with_suffix(".json").write_text(json.dumps(manifest,indent=2))
    return result,manifest

if __name__=="__main__":
    import argparse
    from accounting.reports.treasury_accountability.projection import build_default_projection
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--fx-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    forecast=args.out_dir/"forecast"
    build_default_projection(args.run_root,forecast)
    build_rolling_usd(observed=args.run_root/"monthly_cash_accountability.csv",
        forecast=forecast/"monthly_cash_accountability.csv",means=args.fx_dir/"monthly_ccl_mean.csv",
        daily=args.fx_dir/"ccl_rates_snapshot.csv",projection_manifest=forecast/"projection_manifest.json",
        output=args.out_dir/"rolling.csv")
