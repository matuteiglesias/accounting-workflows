"""Canonical accounting-universe governance helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


BOX_CODES = {
    "Family Business": "FB",
    "Household": "HH",
    "Property Management": "PM",
}
CANONICAL_BOXES = frozenset(BOX_CODES)
PROPERTY_BUSINESS_BOXES = {"Family Business", "Property Management"}
HOUSEHOLD_BOXES = {"Household"}
SCOPE_RULE = "box_exact_membership"
SCOPE_VERSION = 1


@dataclass(frozen=True)
class RunScope:
    boxes: tuple[str, ...]
    codes: tuple[str, ...]
    tag: str
    rule: str
    version: int


def canonical_box_name(value: object) -> str:
    """Return a canonical Box name, accepting only spelling/case normalization."""
    text = str(value).strip()
    by_casefold = {name.casefold(): name for name in CANONICAL_BOXES}
    try:
        return by_casefold[text.casefold()]
    except KeyError as exc:
        raise ValueError(f"Unknown Box {text!r}; expected one of {sorted(CANONICAL_BOXES)}") from exc


def parse_box_scope(value: str | None) -> set[str]:
    """Parse and canonicalize a comma-separated Box selection."""
    parts = [part.strip() for part in (value or "").split(",") if part.strip()]
    if not parts:
        raise ValueError("Box scope must name at least one Box")
    return {canonical_box_name(part) for part in parts}


def canonical_scope_tag(boxes: set[str]) -> str:
    """Return the stable concatenated code tag for a canonical Box universe."""
    canonical = {canonical_box_name(box) for box in boxes}
    if not canonical:
        raise ValueError("Box scope must name at least one Box")
    return "".join(sorted(BOX_CODES[box] for box in canonical))


def scope_metadata(boxes: set[str]) -> dict[str, object]:
    """Return the serializable contract persisted by canonical ingest."""
    canonical = {canonical_box_name(box) for box in boxes}
    ordered = sorted(canonical, key=lambda box: BOX_CODES[box])
    return {
        "scope_boxes": ordered,
        "scope_codes": [BOX_CODES[box] for box in ordered],
        "scope_tag": canonical_scope_tag(canonical),
        "scope_rule": SCOPE_RULE,
        "scope_version": SCOPE_VERSION,
    }


def load_run_scope(run_root: Path) -> RunScope:
    """Load and validate the immutable scope declared by Stage A."""
    manifest_path = Path(run_root) / "meta" / "stage_A_ingest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage A scope manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params") or {}
    required = {"scope_boxes", "scope_codes", "scope_tag", "scope_rule", "scope_version"}
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(f"Stage A scope manifest missing fields: {missing}")

    boxes = tuple(params["scope_boxes"])
    expected = scope_metadata(set(boxes))
    for key in required:
        if params[key] != expected[key]:
            raise ValueError(
                f"Invalid Stage A scope metadata {key}: {params[key]!r} != {expected[key]!r}"
            )
    return RunScope(
        boxes=boxes,
        codes=tuple(params["scope_codes"]),
        tag=str(params["scope_tag"]),
        rule=str(params["scope_rule"]),
        version=int(params["scope_version"]),
    )


def load_run_scope_if_present(run_root: Path) -> RunScope | None:
    """Load a run scope when Stage A metadata exists; support legacy fixtures."""
    manifest_path = Path(run_root) / "meta" / "stage_A_ingest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params") or {}
    scope_keys = {"scope_boxes", "scope_codes", "scope_tag", "scope_rule", "scope_version"}
    if not scope_keys.intersection(params):
        return None
    return load_run_scope(run_root)


def assert_frame_within_scope(
    df: pd.DataFrame,
    scope: RunScope,
    *,
    source: str,
    require_box: bool = False,
) -> None:
    """Fail when a row-level dataframe exceeds its run's declared Box universe."""
    if "Box" not in df.columns:
        if require_box:
            raise ValueError(f"{source} must contain Box to validate run scope {scope.tag}")
        return
    actual = {str(value).strip() for value in df["Box"].dropna() if str(value).strip()}
    foreign = sorted(actual - set(scope.boxes))
    if foreign:
        raise ValueError(
            f"{source} contains Boxes outside run scope {scope.tag}: {foreign}; "
            f"expected subset of {list(scope.boxes)}"
        )


def box_scope_mask(df: pd.DataFrame, boxes: set[str]) -> pd.Series:
    """Return rows whose owning Box belongs to a materialized run's universe."""
    if "Box" not in df.columns:
        # Legacy drilldown fixtures and historical extracts can predate the
        # Box dimension. They are already run-scoped and cannot be narrowed
        # further without silently dropping their entire evidence set.
        return pd.Series(True, index=df.index)
    return df["Box"].astype(str).str.strip().isin(boxes)
