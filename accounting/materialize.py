"""Compatibility wrapper for canonical Stage D materialization.

Prefer ``python -m accounting.stage_d.materialize`` and imports from
``accounting.stage_d.materialize`` for new code.
"""

from accounting.stage_d.materialize import *  # noqa: F401,F403

if __name__ == "__main__":
    from accounting.stage_d.materialize import main

    raise SystemExit(main())
