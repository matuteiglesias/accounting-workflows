"""Environment helpers and project-wide environment constants."""

from __future__ import annotations

import os


def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var {name}. Set it in private/.env (not committed) or export it.")
    return v


ACCOUNT_SHEET_URL = os.getenv("ACCOUNT_SHEET_URL", "").strip()
RENTALS_SHEET_URL = os.getenv("RENTALS_SHEET_URL", "").strip()
SERVICE_ACCOUNT_FILE = os.getenv("ACCOUNT_SA", "").strip()
