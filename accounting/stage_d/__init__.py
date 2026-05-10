"""Canonical Stage D materialization package."""

__all__ = ["materialize_all"]


def __getattr__(name: str):
    if name == "materialize_all":
        from accounting.stage_d.materialize import materialize_all

        return materialize_all
    raise AttributeError(name)
