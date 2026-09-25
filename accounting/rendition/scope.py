from __future__ import annotations

"""Reporting-only property scope registry for formal renditions.

A registry row says which exact governed Lugar x Box population belongs to a
stable reporting property_id. It says nothing about title, ownership percentage,
usufruct, beneficiary, creditor, or legal entitlement.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from accounting.scope import canonical_box_name


REQUIRED_COLUMNS = {"property_id", "display_name", "lugar_selector", "box_selector"}
OPTIONAL_COLUMNS = {"optional_notes"}
ALLOWED_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS


@dataclass(frozen=True)
class PropertyScope:
    property_id: str
    display_name: str
    lugar_selector: str
    box_selector: str
    optional_notes: str = ""

    @property
    def selector_signature(self) -> tuple[str, str]:
        return (self.lugar_selector.casefold(), self.box_selector)


class PropertyRegistry:
    def __init__(self, entries: Iterable[PropertyScope]):
        rows = tuple(entries)
        if not rows:
            raise ValueError("property registry must contain at least one property")

        by_id: dict[str, PropertyScope] = {}
        by_selector: dict[tuple[str, str], str] = {}
        for row in rows:
            if row.property_id in by_id:
                raise ValueError(f"duplicate property_id in registry: {row.property_id}")
            signature = row.selector_signature
            other = by_selector.get(signature)
            if other is not None and other != row.property_id:
                raise ValueError(
                    "ambiguous property selector: "
                    f"{row.property_id} and {other} both select "
                    f"Lugar={row.lugar_selector!r}, Box={row.box_selector!r}"
                )
            by_id[row.property_id] = row
            by_selector[signature] = row.property_id
        self._rows = rows
        self._by_id = by_id

    @classmethod
    def from_csv(cls, path: str | Path) -> "PropertyRegistry":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                raise ValueError(f"property registry missing columns: {missing}")
            extra = sorted(columns - ALLOWED_COLUMNS)
            if extra:
                raise ValueError(
                    "property registry contains unsupported columns "
                    f"{extra}; legal/title semantics do not belong in this registry"
                )
            entries: list[PropertyScope] = []
            for row_number, raw in enumerate(reader, start=2):
                property_id = str(raw.get("property_id", "")).strip()
                display_name = str(raw.get("display_name", "")).strip()
                lugar = str(raw.get("lugar_selector", "")).strip()
                box_raw = str(raw.get("box_selector", "")).strip()
                notes = str(raw.get("optional_notes", "") or "").strip()
                if not all((property_id, display_name, lugar, box_raw)):
                    raise ValueError(
                        f"property registry row {row_number} has blank required fields"
                    )
                if any(token in lugar for token in ("*", "?", "[", "]")) or lugar.casefold().startswith("re:"):
                    raise ValueError(
                        f"property registry row {row_number} uses a non-exact Lugar selector; "
                        "v1 accepts exact selectors only"
                    )
                entries.append(
                    PropertyScope(
                        property_id=property_id,
                        display_name=display_name,
                        lugar_selector=lugar,
                        box_selector=canonical_box_name(box_raw),
                        optional_notes=notes,
                    )
                )
        return cls(entries)

    @property
    def entries(self) -> tuple[PropertyScope, ...]:
        return self._rows

    def resolve(self, property_ids: Iterable[str]) -> tuple[PropertyScope, ...]:
        requested = tuple(property_ids)
        unknown = sorted(set(requested) - set(self._by_id))
        if unknown:
            raise ValueError(f"unknown property_id(s) in rendition spec: {unknown}")
        return tuple(self._by_id[property_id] for property_id in requested)
