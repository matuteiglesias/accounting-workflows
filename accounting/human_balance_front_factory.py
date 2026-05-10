"""Compatibility wrapper for the experimental human front report factory.

Prefer ``python -m accounting.human.front`` and imports from
``accounting.human.front`` for new code.
"""

from accounting.human.front import *  # noqa: F401,F403

if __name__ == "__main__":
    from accounting.human.front import main

    raise SystemExit(main())
