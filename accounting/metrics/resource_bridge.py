"""Native-currency economic funding bridge; distinct from physical cash and debt.

The paid semantic audit is the membership authority. No inferred pairing of
transactions; distribution-linked recoveries require a private explicit list
and are reconciled to the allocation-grain debt authority.
"""
from pathlib import Path
import os
import pandas as pd
from accounting.scope import load_run_scope_if_present
from accounting.cutoff import load_run_cutoff_if_present

SOURCE = "monthly_resource_bridge.csv"
MEMBERS = "resource_bridge_membership.csv"
COMPONENTS = ("contribution", "designated_advance", "direct_tenant_support", "clearing_pending", "linked_debt_recovery")
BOXES = {"Family Business": "FB", "Property Management": "PM"}

def build_resource_bridge(audit: pd.DataFrame, pairs: pd.DataFrame | None = None,
                          repayments: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"tx_id","period","Currency","Box","payer","receiver","Flujo","Tipo","semantic_bucket","semantic_subbucket","direction","amount"}
    if not required.issubset(audit):
        raise ValueError("resource bridge: incomplete semantic audit")
    a = audit.fillna("").copy()
    if a.tx_id.duplicated().any():
        raise ValueError("resource bridge: duplicate tx_id")
    a = a[a.Box.isin(BOXES) & ~a.get("tag", pd.Series("",index=a.index)).str.casefold().eq("legacy_inferred_net")].copy()
    a["amount"] = pd.to_numeric(a.amount, errors="raise")
    owner = a.Box.map(BOXES)
    payer, receiver = a.payer.str.upper(), a.receiver.str.upper()
    core = (a.semantic_bucket.eq("funding_contribution") & a.direction.eq("in") &
            receiver.eq(owner) & ~payer.isin(["", "FB", "PM", "FX"]))
    designated = core & a.semantic_subbucket.eq("designated_expense")
    clearing = core & a.Tipo.str.casefold().eq("repago")
    direct = (a.semantic_bucket.eq("property_opex") & a.direction.eq("out") &
              a.semantic_subbucket.isin(["taxes","services"]) & payer.eq("INQ") &
              ~receiver.isin(["FB","PM","HH","INQ",""]))
    masks = {"contribution": core & ~designated & ~clearing,
             "designated_advance": designated, "direct_tenant_support": direct,
             "clearing_pending": clearing}
    members=[]
    for name, mask in masks.items():
        for row in a.loc[mask].itertuples(index=False):
            members.append((row.tx_id,row.period,row.Currency,row.Box,name,row.amount,""))
    if pairs is not None and not pairs.empty:
        if repayments is None:
            raise ValueError("linked recovery requires debt allocations")
        if set(pairs.columns) != {"distribution_tx_id","recovery_tx_id"} or pairs.distribution_tx_id.duplicated().any() or pairs.recovery_tx_id.duplicated().any():
            raise ValueError("invalid recovery pairing contract")
        by_id=a.set_index("tx_id")
        for pair in pairs.itertuples(index=False):
            out, inc=by_id.loc[pair.distribution_tx_id],by_id.loc[pair.recovery_tx_id]
            if not (out.semantic_subbucket=="dividend" and out.direction=="out" and inc.semantic_bucket=="debt_movement"
                    and inc.semantic_subbucket=="repayment" and inc.direction=="in"
                    and out.Currency==inc.Currency and out.Box==inc.Box and out.payer==inc.receiver
                    and out.receiver==inc.payer and abs(out.amount-inc.amount)<.005):
                raise ValueError("linked recovery does not reconcile with distribution")
            allocations=repayments.loc[repayments.repayment_tx_id.eq(pair.recovery_tx_id)]
            if allocations.empty or abs(pd.to_numeric(allocations.allocated_amount).sum()-inc.amount)>.005:
                raise ValueError("linked recovery not fully allocated by debt engine")
            members.append((pair.recovery_tx_id,inc.period,inc.Currency,inc.Box,"linked_debt_recovery",inc.amount,pair.distribution_tx_id))
    m=pd.DataFrame(members,columns=["tx_id","period","Currency","Box","component","amount","linked_distribution_tx_id"])
    if m.tx_id.duplicated().any():
        raise ValueError("resource bridge: duplicate funding membership")
    keys=a[["period","Currency"]].drop_duplicates().sort_values(["period","Currency"])
    monthly=keys.copy()
    for component in COMPONENTS:
        v=m.loc[m.component.eq(component)].groupby(["period","Currency"]).amount.sum().rename(component)
        monthly=monthly.merge(v,on=["period","Currency"],how="left")
    for component in COMPONENTS:
        monthly[component] = pd.to_numeric(monthly[component], errors="raise").fillna(0.0)
    monthly["economic_funding"]=monthly.contribution+monthly.designated_advance+monthly.direct_tenant_support
    return monthly,m

def load_recovery_pairs(run_root: Path) -> pd.DataFrame | None:
    """Use the pinned run input, or snapshot an applicable private registry.

    The registry is local, not Git-tracked. Scope and cutoff prevent future
    events from entering older or unrelated runs. A referenced transaction
    missing from the applicable ledger fails in build_resource_bridge.
    """
    pairpath=run_root/"private_review"/"distribution_recovery_pairs.csv"
    if pairpath.exists():
        return pd.read_csv(pairpath,dtype=str)
    scope=load_run_scope_if_present(run_root)
    cutoff=load_run_cutoff_if_present(run_root)
    if scope is None or cutoff is None:
        return None
    configured=os.environ.get("RESOURCE_RECOVERY_REGISTRY","").strip()
    registry=Path(configured) if configured else Path(__file__).resolve().parents[2]/"private"/"distribution_recovery_registry.csv"
    if not registry.is_file():
        if configured:
            raise FileNotFoundError("configured recovery registry missing")
        return None
    data=pd.read_csv(registry,dtype=str)
    required={"Box","recovery_date","distribution_tx_id","recovery_tx_id"}
    if set(data.columns)!=required:
        raise ValueError("invalid private recovery registry")
    dates=pd.to_datetime(data.recovery_date,format="%Y-%m-%d",errors="raise")
    eligible=data.Box.isin(scope.boxes)&dates.le(pd.Timestamp(cutoff.date))
    pairs=data.loc[eligible,["distribution_tx_id","recovery_tx_id"]]
    pairpath.parent.mkdir(parents=True,exist_ok=True)
    pairs.to_csv(pairpath,index=False)
    return pairs


def write_resource_bridge(run_root: Path, metrics_dir: Path):
    p=run_root/"classification_audit.csv"
    if not p.exists():
        return None
    audit=pd.read_csv(p,keep_default_na=False)
    if not audit.Box.isin(BOXES).any():
        return None
    pairs=load_recovery_pairs(run_root)
    detail=run_root/"monthly_debt_repayment_detail.csv"
    repayments=pd.read_csv(detail,keep_default_na=False) if pairs is not None and detail.exists() else None
    monthly,members=build_resource_bridge(audit,pairs,repayments)
    monthly.to_csv(metrics_dir/SOURCE,index=False)
    members.to_csv(metrics_dir/MEMBERS,index=False)
    return monthly
