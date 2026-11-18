from __future__ import annotations

"""Record outputs from existing marts in the canonical run artifact log.

This is an adapter to accounting.artifacts.manifest, not another registry.
A retry is idempotent for identical outputs; contradictory provenance fails.
"""

import json
from pathlib import Path
from typing import Mapping

from accounting.artifacts.manifest import append_artifacts, artifact_from_path


def register_mart_outputs(
    run_root: Path,
    outputs: Mapping[str, Path],
    *,
    stage: str,
) -> list[dict]:
    root = Path(run_root).resolve(strict=True)
    if not stage or root.name.startswith("latest"):
        raise ValueError("mart provenance requires an exact run and stage")
    if not outputs:
        raise ValueError("mart provenance requires outputs")

    # Preflight the whole group before appending anything.
    candidates: list[dict] = []
    for name, path in outputs.items():
        product = artifact_from_path(
            name=name, path=Path(path), stage=stage,
            mode="run", run_id=root.name, role="derived",
            root_dir=root,
        )
        candidates.append(product)
    paths = [record["relpath"] for record in candidates]
    if len(paths) != len(set(paths)):
        raise ValueError("multiple output names point to the same artifact")

    log_path = root / "meta" / "artifacts.jsonl"
    existing: dict[str, list[dict]] = {}
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("role") not in ("derived", "output"):
                continue
            existing.setdefault(str(record["relpath"]), []).append(record)
    additions = []
    for candidate in candidates:
        prior = existing.get(candidate["relpath"], [])
        for record in prior:
            if record.get("stage") != stage or record.get("run_id") != root.name:
                raise ValueError(f"conflicting producer for {candidate['relpath']}")
            if record.get("sha256") != candidate["sha256"]:
                raise ValueError(f"same-run output changed: {candidate['relpath']}")
        if not prior:
            additions.append(candidate)
    if additions:
        append_artifacts(root / "meta", additions)
    return additions
