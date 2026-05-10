"""Compatibility wrapper for the canonical human report factory.

Prefer ``python -m accounting.human.document`` and imports from
``accounting.human.document`` for new code.
"""

from accounting.human.document import *  # noqa: F401,F403

if __name__ == "__main__":
    from accounting.human.document import main

    raise SystemExit(main())
