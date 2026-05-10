"""Compatibility wrapper for the canonical debt resolver.

Prefer ``python -m accounting.debt.resolve`` and imports from
``accounting.debt.resolve`` for new code.
"""

from accounting.debt.resolve import *  # noqa: F401,F403

if __name__ == "__main__":
    from accounting.debt.resolve import main

    raise SystemExit(main())
