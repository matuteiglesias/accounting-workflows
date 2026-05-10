"""Compatibility wrapper for canonical frontend snapshot publishing.

Prefer ``python -m accounting.publish.latest`` and imports from
``accounting.publish.latest`` for new code.
"""

from accounting.publish.latest import *  # noqa: F401,F403

if __name__ == "__main__":
    from accounting.publish.latest import main

    raise SystemExit(main())
