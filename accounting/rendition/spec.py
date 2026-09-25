from __future__ import annotations

"""Strict private input contract for one formal account rendition."""

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any

from accounting.cutoff import normalize_cutoff_date
from accounting.scope import canonical_box_name
from accounting.rendition.contract import (
    RENDITION_SPEC_SCHEMA,
    SUPPORTED_LANGUAGES,
    SUPPORTED_PURPOSES,
    assert_no_forbidden_semantics,
)


_TOP_LEVEL_KEYS = {
    "schema",
    "rendition_id",
    "source_run_id",
    "rendidor",
    "period",
    "properties",
    "boxes",
    "recipients",
    "purpose",
    "language",
}
_RENDIDOR_KEYS = {"display_name"}
_PERIOD_KEYS = {"from", "through"}
_PROPERTY_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_.-]*$")


@dataclass(frozen=True)
class RenditionPeriod:
    from_date: str
    through_date: str


@dataclass(frozen=True)
class RenditionSpec:
    schema: str
    rendition_id: str
    source_run_id: str
    rendidor_display_name: str
    period: RenditionPeriod
    properties: tuple[str, ...]
    boxes: tuple[str, ...]
    recipients: tuple[str, ...]
    purpose: str
    language: str


def _strict_keys(mapping: dict[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{path} has unsupported fields: {unknown}")


def _nonblank(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be blank")
    return text


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    items = tuple(_nonblank(item, field=field) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{field} must not contain duplicates")
    return items


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "YAML rendition specs require PyYAML; install pyyaml or use JSON"
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError("rendition spec must be .yaml, .yml, or .json")
    if not isinstance(payload, dict):
        raise ValueError("rendition spec root must be a mapping")
    return payload


def parse_rendition_spec(payload: dict[str, Any]) -> RenditionSpec:
    """Parse a strict accounting.rendition_spec.v1 mapping."""

    assert_no_forbidden_semantics(payload)
    _strict_keys(payload, _TOP_LEVEL_KEYS, path="spec")

    missing = sorted(_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(f"spec missing required fields: {missing}")

    schema = _nonblank(payload["schema"], field="schema")
    if schema != RENDITION_SPEC_SCHEMA:
        raise ValueError(f"unsupported rendition spec schema: {schema!r}")

    rendition_id = _nonblank(payload["rendition_id"], field="rendition_id")
    source_run_id = _nonblank(payload["source_run_id"], field="source_run_id")

    rendidor = payload["rendidor"]
    if not isinstance(rendidor, dict):
        raise ValueError("rendidor must be a mapping")
    _strict_keys(rendidor, _RENDIDOR_KEYS, path="rendidor")
    if set(rendidor) != _RENDIDOR_KEYS:
        raise ValueError("rendidor must contain display_name")
    rendidor_display_name = _nonblank(rendidor["display_name"], field="rendidor.display_name")

    period = payload["period"]
    if not isinstance(period, dict):
        raise ValueError("period must be a mapping")
    _strict_keys(period, _PERIOD_KEYS, path="period")
    if set(period) != _PERIOD_KEYS:
        raise ValueError("period must contain from and through")
    from_date = normalize_cutoff_date(period["from"])
    through_date = normalize_cutoff_date(period["through"])
    if date.fromisoformat(from_date) > date.fromisoformat(through_date):
        raise ValueError(f"period.from must be <= period.through: {from_date} > {through_date}")

    properties = _string_list(payload["properties"], field="properties")
    bad_property_ids = [value for value in properties if not _PROPERTY_ID_RE.fullmatch(value)]
    if bad_property_ids:
        raise ValueError(
            "properties must use stable uppercase property_id tokens "
            f"([A-Z0-9_.-]): {bad_property_ids}"
        )

    raw_boxes = _string_list(payload["boxes"], field="boxes")
    boxes = tuple(canonical_box_name(value) for value in raw_boxes)
    if len(set(boxes)) != len(boxes):
        raise ValueError("boxes collapse to duplicate canonical Box values")

    recipients = _string_list(payload["recipients"], field="recipients")
    purpose = _nonblank(payload["purpose"], field="purpose")
    if purpose not in SUPPORTED_PURPOSES:
        raise ValueError(f"unsupported rendition purpose: {purpose!r}")
    language = _nonblank(payload["language"], field="language")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported rendition language: {language!r}")

    return RenditionSpec(
        schema=schema,
        rendition_id=rendition_id,
        source_run_id=source_run_id,
        rendidor_display_name=rendidor_display_name,
        period=RenditionPeriod(from_date=from_date, through_date=through_date),
        properties=properties,
        boxes=boxes,
        recipients=recipients,
        purpose=purpose,
        language=language,
    )


def load_rendition_spec(path: str | Path) -> RenditionSpec:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return parse_rendition_spec(_load_mapping(source))
