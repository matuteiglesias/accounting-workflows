"""Compatibility wrapper for canonical debt balance views.

Prefer ``python -m accounting.debt.balance_views`` and imports from
``accounting.debt.balance_views`` for new code.
"""

from accounting.debt.balance_views import *  # noqa: F401,F403

if __name__ == "__main__":
    from accounting.debt.balance_views import main

    raise SystemExit(main())
