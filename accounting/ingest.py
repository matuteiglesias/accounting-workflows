"""Compatibility wrapper for canonical ledger ingest.

Prefer ``python -m accounting.ledger.ingest`` and imports from
``accounting.ledger.ingest`` for new code.
"""

from accounting.ledger.ingest import *  # noqa: F401,F403

if __name__ == "__main__":
    from accounting.ledger.ingest import main

    raise SystemExit(main())
