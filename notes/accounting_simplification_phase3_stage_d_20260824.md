# Accounting simplification Phase 3 — Stage D shared infrastructure

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
