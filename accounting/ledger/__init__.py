"""Canonical ledger ingest package."""

__all__ = ["build_ledger_base"]


def __getattr__(name: str):
    if name == "build_ledger_base":
        from accounting.ledger.ingest import build_ledger_base

        return build_ledger_base
    raise AttributeError(name)
