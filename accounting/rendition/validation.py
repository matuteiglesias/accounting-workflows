from __future__ import annotations

"""Cross-check a rendition spec against one immutable governed accounting run."""

import argparse
from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path

from accounting.cutoff import load_run_cutoff_if_present
from accounting.scope import load_run_scope_if_present
from accounting.rendition.scope import PropertyRegistry
from accounting.rendition.spec import RenditionSpec, load_rendition_spec


@dataclass(frozen=True)
class RenditionValidationResult:
    rendition_id: str
    source_run_id: str
    run_root: str
    period_from: str
    period_through: str
    properties: tuple[str, ...]
    boxes: tuple[str, ...]
    status: str = "pass"


def validate_rendition_context(
    spec: RenditionSpec,
    registry: PropertyRegistry,
    run_root: str | Path,
) -> RenditionValidationResult:
    """Validate Phase-1 scope invariants without computing accounting values."""

    root = Path(run_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if root.name != spec.source_run_id:
        raise ValueError(
            "rendition spec would mix accounting runs: "
            f"source_run_id={spec.source_run_id!r}, run_root={root.name!r}"
        )

    selected = registry.resolve(spec.properties)
    selected_boxes = {entry.box_selector for entry in selected}
    spec_boxes = set(spec.boxes)
    outside_spec = sorted(selected_boxes - spec_boxes)
    if outside_spec:
        raise ValueError(
            "selected properties resolve to Box values outside rendition spec boxes: "
            f"{outside_spec}"
        )

    run_scope = load_run_scope_if_present(root)
    if run_scope is not None:
        foreign_spec_boxes = sorted(spec_boxes - set(run_scope.boxes))
        if foreign_spec_boxes:
            raise ValueError(
                f"rendition spec boxes exceed immutable run scope {run_scope.tag}: "
                f"{foreign_spec_boxes}"
            )
        foreign_property_boxes = sorted(selected_boxes - set(run_scope.boxes))
        if foreign_property_boxes:
            raise ValueError(
                "property registry resolves selected properties outside immutable run scope: "
                f"{foreign_property_boxes}"
            )

    cutoff = load_run_cutoff_if_present(root)
    if cutoff is not None:
        through = date.fromisoformat(spec.period.through_date)
        governed_cutoff = date.fromisoformat(cutoff.date)
        if through > governed_cutoff:
            raise ValueError(
                "rendition period exceeds immutable run cutoff: "
                f"{spec.period.through_date} > {cutoff.date}"
            )

    return RenditionValidationResult(
        rendition_id=spec.rendition_id,
        source_run_id=spec.source_run_id,
        run_root=str(root),
        period_from=spec.period.from_date,
        period_through=spec.period.through_date,
        properties=spec.properties,
        boxes=spec.boxes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate accounting.rendition_spec.v1 against one exact governed run"
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--property-registry", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()

    spec = load_rendition_spec(args.spec)
    registry = PropertyRegistry.from_csv(args.property_registry)
    result = validate_rendition_context(spec, registry, args.run_root)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
