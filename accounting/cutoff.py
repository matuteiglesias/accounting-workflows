"""Immutable temporal-horizon contract for bounded accounting runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path


CUTOFF_RULE = (
    "valid Date <= cutoff_date; missing/unparseable Date retained as anomaly evidence"
)
CUTOFF_VERSION = 1
CUTOFF_KEYS = {"cutoff_date", "cutoff_rule", "cutoff_version"}


@dataclass(frozen=True)
class RunCutoff:
    """Validated Stage A temporal horizon for one materialized run."""

    date: str
    rule: str
    version: int


def normalize_cutoff_date(value: object) -> str:
    """Return a strict ISO calendar date or raise a useful ValueError."""

    text = str(value).strip()
    if not text:
        raise ValueError("cutoff date must not be blank")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid cutoff date {text!r}; expected YYYY-MM-DD"
        ) from exc
    return parsed.isoformat()


def cutoff_metadata(value: object) -> dict[str, object]:
    """Return the serializable Stage A cutoff contract."""

    return {
        "cutoff_date": normalize_cutoff_date(value),
        "cutoff_rule": CUTOFF_RULE,
        "cutoff_version": CUTOFF_VERSION,
    }


def load_run_cutoff(run_root: Path) -> RunCutoff:
    """Load and validate the immutable cutoff declared by Stage A."""

    manifest_path = Path(run_root) / "meta" / "stage_A_ingest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage A cutoff manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params") or {}
    missing = sorted(CUTOFF_KEYS - set(params))
    if missing:
        raise ValueError(f"Stage A cutoff manifest missing fields: {missing}")

    expected = cutoff_metadata(params["cutoff_date"])
    for key in CUTOFF_KEYS:
        if params[key] != expected[key]:
            raise ValueError(
                f"Invalid Stage A cutoff metadata {key}: "
                f"{params[key]!r} != {expected[key]!r}"
            )

    return RunCutoff(
        date=str(params["cutoff_date"]),
        rule=str(params["cutoff_rule"]),
        version=int(params["cutoff_version"]),
    )


def load_run_cutoff_if_present(run_root: Path) -> RunCutoff | None:
    """Load a cutoff when Stage A declares one; support unbounded legacy runs."""

    manifest_path = Path(run_root) / "meta" / "stage_A_ingest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params") or {}
    present = CUTOFF_KEYS.intersection(params)
    if not present:
        return None
    if present != CUTOFF_KEYS:
        missing = sorted(CUTOFF_KEYS - set(params))
        raise ValueError(f"Stage A cutoff manifest missing fields: {missing}")
    return load_run_cutoff(run_root)


def resolve_run_as_of_date(run_root: Path, requested: object | None = None) -> str:
    """Resolve reporting metadata without allowing a second temporal authority."""

    cutoff = load_run_cutoff_if_present(run_root)
    explicit = None
    if requested is not None and str(requested).strip():
        explicit = normalize_cutoff_date(requested)

    if cutoff is not None:
        if explicit is not None and explicit != cutoff.date:
            raise ValueError(
                "Requested as_of_date conflicts with immutable Stage A cutoff: "
                f"{explicit} != {cutoff.date}"
            )
        return cutoff.date

    return explicit or date.today().isoformat()
