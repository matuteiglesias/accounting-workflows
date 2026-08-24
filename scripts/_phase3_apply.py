from __future__ import annotations

from pathlib import Path


def must_replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing expected block: {label}")
    return text.replace(old, new, 1)


root = Path('.')
materialize_path = root / 'accounting/stage_d/materialize.py'
text = materialize_path.read_text(encoding='utf-8')

text = must_replace(
    text,
    'import argparse\nimport hashlib\nimport json\nimport logging\nimport os\nimport sys\n',
    'import argparse\nimport os\n',
    'obsolete Stage D infrastructure imports',
)

text = must_replace(
    text,
    'from accounting.logging_utils import configure_logging, get_logger\n',
    'from accounting.logging_utils import configure_logging, get_logger\n'
    'from accounting.support.hashing import sha256_file\n'
    'from accounting.support.io import atomic_write_df\n'
    'from accounting.support.partitions import load_partitions_json, save_partitions_json\n',
    'shared support imports',
)

helper_start = text.index('\ndef _atomic_write_csv')
helper_end = text.index('\n\n# -----------------------\n# Per-artifact materializers', helper_start)
safe_hash = '''\n\ndef _safe_sha256(path: Path) -> Optional[str]:
    """Keep Stage D's historical fail-soft metadata behavior over shared hashing."""
    try:
        return sha256_file(path)
    except Exception:
        LOG.exception("Failed to hash file: %s", path)
        return None
'''
text = text[:helper_start] + safe_hash + text[helper_end:]

text = text.replace('_atomic_write_csv(out_df, target)', 'atomic_write_df(out_df, target, index=False)')
text = text.replace('_atomic_write_csv(ldf, ledger_path)', 'atomic_write_df(ldf, ledger_path, index=False)')
text = text.replace('_atomic_write_csv(anomalies, anomalies_path)', 'atomic_write_df(anomalies, anomalies_path, index=False)')
text = text.replace('_sha256_file(', '_safe_sha256(')

artifact_start = text.index('    # outputs esperados\n    out_arts = []\n')
artifact_end = text.index('    known_relpaths = {art["relpath"] for art in [in_art, *out_arts]}', artifact_start)
compact_artifacts = '''    # Stable Stage D artifact inventory. The manifest authority remains
    # accounting.artifacts.manifest; Stage D only declares expected paths.
    out_arts = []
    output_specs = [
        ("per_flow_time_long", out_dir / f"per_flow_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("per_party_time_long", out_dir / f"per_party_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("box_balance_time_long", out_dir / f"box_balance_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("box_flow_balance_time_long", out_dir / f"box_flow_balance_time_long.freq={freq}.csv", "derived", "text/csv"),
        ("loans_time", out_dir / "loans_time.freq=M.csv", "derived", "text/csv"),
        ("daily_cash_position", out_dir / "daily_cash_position.csv", "derived", "text/csv"),
        ("partitions", out_dir / "partitions.json", "meta", "application/json"),
        ("anomalies", out_dir / "anomalies.csv", "derived", "text/csv"),
    ]
    for name, path, role, content_type in output_specs:
        if path.exists():
            out_arts.append(
                artifact_from_path(
                    name=name,
                    path=path,
                    stage="D.materialize",
                    mode=args.mode,
                    run_id=run_id,
                    role=role,
                    root_dir=out_dir,
                    content_type=content_type,
                )
            )

'''
text = text[:artifact_start] + compact_artifacts + text[artifact_end:]
materialize_path.write_text(text, encoding='utf-8')

# Promote Stage D's atomic partition-write behavior into the shared partition helper.
partitions_path = root / 'accounting/support/partitions.py'
parts = partitions_path.read_text(encoding='utf-8')
old_save = '''def save_partitions_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
'''
new_save = '''def save_partitions_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write partition metadata while preserving the established JSON shape."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    tmp.replace(path)
'''
parts = must_replace(parts, old_save, new_save, 'shared partition writer')
partitions_path.write_text(parts, encoding='utf-8')

(root / 'tests/test_phase3_stage_d_shared_infrastructure.py').write_text('''from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.support.io import atomic_write_df
from accounting.support.partitions import load_partitions_json, save_partitions_json


def test_stage_d_delegates_generic_infrastructure_to_shared_support() -> None:
    source = Path("accounting/stage_d/materialize.py").read_text(encoding="utf-8")
    assert "from accounting.support.hashing import sha256_file" in source
    assert "from accounting.support.io import atomic_write_df" in source
    assert "from accounting.support.partitions import load_partitions_json, save_partitions_json" in source
    for forbidden in [
        "def _atomic_write_csv",
        "def _sha256_file",
        "def load_partitions_json",
        "def save_partitions_json",
        "def _write_manifest",
        "import hashlib",
        "import json",
    ]:
        assert forbidden not in source


def test_shared_csv_writer_preserves_stage_d_index_false_shape(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    frame = pd.DataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    atomic_write_df(frame, path, index=False)
    reread = pd.read_csv(path)
    assert list(reread.columns) == ["a", "b"]
    assert reread.to_dict("records") == frame.to_dict("records")


def test_shared_partition_writer_round_trips_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "partitions.json"
    payload = {"freq": "M", "outputs": {"x.csv": {"rows": 3}}}
    save_partitions_json(path, payload)
    assert load_partitions_json(path) == payload
    assert not path.with_suffix(path.suffix + ".tmp").exists()
''', encoding='utf-8')

(root / 'notes/accounting_simplification_phase3_stage_d_20260824.md').write_text('''# Accounting simplification Phase 3 — Stage D shared infrastructure

Date: 2026-08-24  
Base: `90e6f403377d355301670e053d203959cd92cdeb`  
Accounting-policy change: **none**

## Invariant

Stage D must produce the same mechanical materializations, preserve the same orchestration order, and leave semantic/cash mart behavior unchanged. This PR changes generic infrastructure ownership only.

Protected accounting/reporting behavior:

- semantic classification unchanged;
- monthly semantic totals unchanged;
- annual metric totals/statuses unchanged;
- debt semantics unchanged;
- governed cash semantics unchanged;
- Box scope unchanged;
- native currencies remain separate;
- professional values/drilldown membership untouched.

## Before

`accounting/stage_d/materialize.py` locally owned generic implementations for:

- atomic CSV writes;
- SHA-256 file hashing;
- partition JSON load/save;
- a dead local stage-manifest writer;
- repetitive registration of known Stage D artifacts.

At the same time the repository already had `accounting.support.io`, `accounting.support.hashing`, `accounting.support.partitions`, and `accounting.artifacts.manifest`.

## After

Stage D delegates:

- CSV writes -> `accounting.support.io.atomic_write_df(..., index=False)`;
- hashing -> `accounting.support.hashing.sha256_file` behind a tiny fail-soft metadata adapter;
- partition JSON -> `accounting.support.partitions`;
- stage/artifact manifests -> the already-authoritative `accounting.artifacts.manifest` path;
- known artifact registration -> one declarative `output_specs` loop.

The shared partition writer is made atomic so this move does not weaken Stage D's prior write guarantee.

## Deliberate non-changes

This phase does **not** move `build_monthly_cash_close`, `build_semantic_outputs`, or `build_monthly_operating_statement` out of Stage D. Their sequencing is therefore unchanged. Moving semantic/cash orchestration is a later bounded architecture change.

No live latest-pointer implementation was present in current `stage_d/materialize.py`; Phase 1 had already removed obsolete human/latest orchestration elsewhere. Nothing is invented here merely to satisfy the old cleanup checklist.

## Acceptance evidence

Before merge require:

1. `make validate`;
2. `make smoke-full`;
3. exact Phase-0 semantic and annual ARS/USD fixture anchors;
4. Stage D expected artifact inventory and manifest contract still present;
5. source regression proving generic helpers are no longer implemented locally.

Generated smoke outputs are evidence only and are not committed.
''', encoding='utf-8')

# Record the ownership boundary in the current map without changing historical notes.
state_path = root / 'notes/current_state_map.md'
state = state_path.read_text(encoding='utf-8')
needle = '- `accounting.stage_d` / `accounting.marts` own materialized and semantic tables.\n'
replacement = needle + '- `accounting.stage_d` owns Stage-D orchestration and mechanical builders but delegates generic CSV/hash/partition/manifest infrastructure to `accounting.support` / `accounting.artifacts`; semantic/cash sequencing remains unchanged.\n'
state = must_replace(state, needle, replacement, 'current state Stage D ownership')
state_path.write_text(state, encoding='utf-8')
