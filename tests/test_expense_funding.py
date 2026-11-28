import pandas as pd
import pytest
from accounting.metrics.expense_funding import partition

def fixture():
    common=dict(period="2026-01",Currency="ARS",Box="Property Management",Lugar="CABA",Tipo="Impuestos")
    a=pd.DataFrame([dict(common,tx_id="fund",payer="Inq",semantic_bucket="funding_contribution",semantic_subbucket="designated_expense",direction="in",amount=1505,tag=""),
        dict(common,tx_id="expense",payer="PM",semantic_bucket="property_opex",semantic_subbucket="taxes",direction="out",amount=3014,tag="expense_funding_ids:1")])
    l=pd.DataFrame([{"tx_id":"fund","transaction_id":"1"},{"tx_id":"expense","transaction_id":"2"}])
    f=pd.DataFrame([{"tx_id":"fund","component":"designated_advance","funding_actor":"Inquilino","amount":1505}])
    return a,l,f
def test_explicit_partial_funding_conserves_expense_and_operator():
    a,l,f=fixture();p=partition(a,l,f)
    assert p.groupby("funding_source").amount.sum().to_dict()=={"Inquilinos":1505,"PM":1509}
    assert p.groupby("circuit").amount.sum().to_dict()=={"PM":3014}
def test_overfunding_and_reuse_rejected():
    a,l,f=fixture();f.loc[0,"amount"]=4000
    with pytest.raises(ValueError,match="exceeds"):partition(a,l,f)
    a,l,f=fixture();a=pd.concat([a,a.iloc[[-1]].assign(tx_id="other")])
    with pytest.raises(ValueError,match="reused"):partition(a,l,f)
