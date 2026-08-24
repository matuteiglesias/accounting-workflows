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

## Completed evidence

The transformed workspace was executed before its product commit:

- Stage-D source size: **929 -> 793 lines** (`-136`, about `-14.6%`);
- Stage-D source bytes: **33,712 -> 30,325**;
- `make validate`: **PASS**;
- pytest: **290 passed, 1 warning**;
- warning: unchanged legacy invalid-`as_of_date` debt compatibility case;
- compile + artifact/source/annual/publish contract checks: **PASS**;
- `make smoke-full`: **PASS**;
- Stage-D fixture row counts unchanged: per-flow `114`, per-party `277`, box balance `0`, box flow balance `0`, loans `0`, daily cash `4,284`;
- full expected Stage-D artifact inventory present, including `partitions.json`, semantic/cash outputs, artifact-contract CSVs, and `meta/stage_D_materialize.json`;
- Stage-D manifest still reports `stage = D.materialize` and the same core materialization relpaths;
- semantic leakage QA: **0 rows**;
- exact Phase-0 semantic classification census: **UNCHANGED**;
- exact Phase-0 ARS/USD annual governed anchors and statuses: **UNCHANGED**;
- governed `BS.CASH.TOTAL` remains `unavailable` for both ARS and USD in the smoke fixture;
- native-currency separation remains intact.

Representative reconciled annual anchors remain ARS revenue `33,075,422`, property OPEX `13,615,938.94`, net operating `19,459,483.06`, funding `1,427,956`, draws `29,412,662`, and USD revenue/net operating `4,180`.

The new source regression also requires that Stage D no longer define its own CSV writer, SHA-256 implementation, partition loader/writer, or dead manifest writer. Shared CSV and partition helpers are behaviorally exercised.

## Evidence boundary

The pre-existing fixture limitation remains: `smoke-full` does not provide governed debt-position/activity source artifacts or a real professional pack. This Phase does not touch those layers and makes no new claim about them.

Generated smoke outputs are evidence only and are not committed.
