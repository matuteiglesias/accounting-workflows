"""RUN_ID resolution helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union

_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")  # e.g. 20260109T142110Z


def _infer_run_id_from_path(root_dir: Union[str, Path]) -> Optional[str]:
    """
    Scan the directory and its parents for a folder name that looks like a RUN_ID.
    This lets stages infer run_id when orchestrator didn't pass --run-id.
    """
    p = Path(root_dir).resolve()
    for cand in [p, *p.parents]:
        name = cand.name.strip()
        if _RUN_ID_RE.match(name):
            return name
    return None


def resolve_run_id(
    *,
    mode: str,
    run_id: Optional[str] = None,
    root_dir: Optional[Union[str, Path]] = None,
    env_var: str = "RUN_ID",
    strict: bool = True,
) -> str:
    """
    Canonical RUN_ID resolution.

    Precedence:
      1) explicit run_id argument
      2) environment variable RUN_ID (configurable)
      3) infer from root_dir (or its parents)
      4) if mode == smoke -> "smoke"
      5) else: error (strict) or fallback "untracked"

    In run mode, returning "" is forbidden.
    """
    m = (mode or "").strip().lower()

    rid = (run_id or "").strip()
    if rid and rid.lower() != "none":
        return rid

    env_rid = (os.getenv(env_var) or "").strip()
    if env_rid:
        return env_rid

    if root_dir is not None:
        inferred = _infer_run_id_from_path(root_dir)
        if inferred:
            return inferred

    if m == "smoke":
        return "smoke"

    if strict:
        raise ValueError(
            "run_id missing for non-smoke run. "
            "Pass --run-id, set RUN_ID, or run inside out/run/accounting/<RUN_ID>/"
        )
    return "untracked"
