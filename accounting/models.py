# src/accounting/models.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from enum import Enum
import pandas as pd

class Currency(str, Enum):
    ARS = "ARS"
    USD = "USD"
    # add as needed

class TxStatus(str, Enum):
    PAGADO = "pagado"
    PENDIENTE = "pendiente"
    CANCELADO = "cancelado"
    # normalize values in ingest

class Money(BaseModel):
    amount: float = Field(..., ge=0)
    currency: Currency = Currency.ARS
    rate: Optional[float] = None  # FX rate to base (if present)

    def to_base(self, base_currency: Currency = Currency.USD) -> float:
        if self.currency == base_currency:
            return self.amount
        if self.rate is None:
            raise ValueError("Missing rate to convert to base currency")
        # assume rate is amount_in_original / rate = base (consistent with current convert_currency)
        return self.amount / self.rate if base_currency == Currency.USD else self.amount * self.rate


class Ledger(BaseModel):
    transactions: List[Transaction]

    def to_dataframe(self, base_currency: Currency = Currency.USD) -> pd.DataFrame:
        rows = []
        for t in self.transactions:
            rows.append({
                "transaction_id": t.transaction_id,
                "Date": t.date,
                "Box": t.box,
                "payer": t.payer,
                "receiver": t.receiver,
                "amount": t.money.amount,
                "Currency": t.money.currency,
                "amount_base": (t.money.to_base(base_currency) if t.money.rate else None),
                "Flujo": t.flujo,
                "Tipo": t.tipo,
                "Lugar": t.lugar,
                "Detalle": t.detalle,
                "issuer": t.issuer,
                "account_id": t.account_id,
                "status": t.status,
                "medio": t.medio,
                "due_date": t.due_date,
                "posted_date": t.posted_date,
                **t.metadata
            })
        df = pd.DataFrame(rows)
        # canonicalize Date index & TimePeriod in reporting layer
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Ledger":
        txs = []
        # enforce deterministic order and dedupe on transaction_id
        seen = set()
        for _, row in df.iterrows():
            tid = str(row.get("transaction_id") or "")
            if tid in seen:
                continue
            seen.add(tid)
            txs.append(Transaction.from_series(row))
        return cls(transactions=txs)

class Transaction(BaseModel):
    transaction_id: str
    date: datetime
    box: Optional[str] = None
    payer: Optional[str] = None
    receiver: Optional[str] = None
    money: Money
    flujo: Optional[str] = None
    tipo: Optional[str] = None
    lugar: Optional[str] = None
    detalle: Optional[str] = None
    issuer: Optional[str] = None
    account_id: Optional[str] = None
    status: Optional[TxStatus] = None
    medio: Optional[str] = None
    due_date: Optional[datetime] = None
    posted_date: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

    @root_validator
    def at_least_one_party(cls, values):
        if not (values.get("payer") or values.get("receiver")):
            raise ValueError("transaction must have at least payer or receiver")
        return values

    @validator("transaction_id", pre=True)
    def normalize_tid(cls, v):
        return str(v).strip()

    @classmethod
    def from_series(cls, s: pd.Series) -> "Transaction":
        # s is a DataFrame row
        money = Money(
            amount=float(s.get("amount") or s.get("Amount") or s.get("Debit") or s.get("Credit") or 0.0),
            currency=(s.get("Currency")),
            rate=s.get("Rate")
        )
        return cls(
            transaction_id=str(s.get("transaction_id") or s.get("id") or ""),
            date=pd.to_datetime(s.get("Date")),
            box=s.get("Box"),
            payer=s.get("payer"),
            receiver=s.get("receiver"),
            money=money,
            flujo=s.get("Flujo"),
            tipo=s.get("Tipo"),
            lugar=s.get("Lugar"),
            detalle=s.get("Detalle"),
            issuer=s.get("issuer"),
            account_id=s.get("account_id"),
            status=(s.get("status") and s.get("status").lower()),
            medio=s.get("medio"),
            due_date=pd.to_datetime(s.get("due_date")) if s.get("due_date") else None,
            posted_date=pd.to_datetime(s.get("posted_date")) if s.get("posted_date") else None,
            metadata={}
        )
