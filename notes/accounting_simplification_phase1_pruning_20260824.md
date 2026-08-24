# Accounting simplification Phase 1 — prune empty architecture

Date: 2026-08-24  
Base: `b9c391b159a62b6a08a81a6fb83fab86e1213eac`  
Accounting-policy change: **none**  
Intentional interface change: **retire the alternate human/front presentation stack**

## Invariant

Semantic classification, monthly semantic totals, annual metric values/statuses, debt semantics, validated-cash rules, Box scope, native-currency separation, professional displayed values, and professional drilldown membership remain governed by the Phase-0 baseline. This PR removes alternate presentation/orchestration paths; it does not change an accounting rule.

## Reachability census and disposition

| Candidate | Census | Disposition | Reason |
|---|---|---|---|
| `accounting/publish/snapshot.py` | no repository caller | DELETE | re-export-only “reserved future seam” |
| `accounting/debt/models.py` | no repository caller | DELETE | re-export-only aliases from `debt.resolve` |
| `accounting/debt/rules.py` | no repository caller | DELETE | re-export-only aliases from `debt.resolve` |
| `accounting/config.py` | no repository caller | DELETE | stale parallel config loader; Make/env + stage CLIs are live control plane |
| `accounting/contracts/models.py` | no repository caller | DELETE | unused parallel Pydantic ledger/money model; not canonical ledger authority |
| `accounting/human/reports.py` | no external code caller | DELETE with package | deprecated wrapper over marts |
| `accounting/viz/plots.py` | no live caller; CLI disabled | DELETE with package | obsolete plotting surface |

## `accounting.human` capability disposition

The package had no Python caller outside itself. Make/docs exposed it as a second presentation pipeline.

- `reports.py`: redundant wrapper over current marts — dismissed.
- `tables.py`: adapters over already materialized metric/debt views — current reusable table governance lives in metric outputs and `accounting.professional.table_contracts` — dismissed.
- `compact.py`: uncoupled compact-semester projection with no caller; equivalent analytical inputs remain available to notebooks/professional consumers — dismissed.
- `document.py`: `balance_humano_v2` and a duplicate annual-dashboard projection — governed annual dashboard remains in `accounting.metrics`; richer human presentation belongs to the professional pack — dismissed.
- `front.py`: static HTML factory containing explicit stub messages; no Flask import/runtime exists — dismissed. The professional linked digest is the maintained presentation path.

No formula from these files is promoted to semantic authority. Removing a duplicate formatter is not permission to remove its upstream governed metric.

## Publication migration

Before: `accounting.publish.latest` required matching `human_reports`, metrics, and debt latest roots; copied `balance_human_v2`; and published `accounting_frontend_snapshot.v1` with report/navigation fields.

After: publication requires matching metrics + debt latest roots only and writes `accounting_public_bundle.v1`. It remains packaging-only. The old `build_frontend_snapshot_manifest` Python function name is retained only as an explicit deprecated external-import compatibility alias; repository code does not call it. Removal condition is a zero external-import census.

## Makefile migration

Removed live `human-report`, `run-human*`, `front-report`, `build-report`, `build-front`, and `run-metrics-and-human` surfaces plus `out/human_reports` latest management. `run-full` now ends `... -> metrics -> dashboard -> publish -> release-check`. Professional drilldown/linked-digest targets remain separate because a real professional pack is not fixture-CI input.

## Validation evidence

Validated on the transformed Phase-1 tree before commit:

- `make validate`: **PASS** — 284 tests passed, 1 intentional legacy invalid-`as_of_date` warning; compile and contract validation passed.
- `make smoke-full`: **PASS** — fixture ingest/materialization, semantic/cash wrapper checks, validation, and publication dry-run passed.
- publish contract import smoke: **PASS**, 12 classified files on the new metrics/debt bundle surface.
- exact Phase-0 semantic classification totals: **UNCHANGED** for all ARS/USD fixture buckets.
- semantic leakage QA: **0 rows**, unchanged.
- exact Phase-0 annual governed values/statuses: **UNCHANGED** for ARS/USD revenue, rent, OPEX, net operating, funding, draws, coverage, savings rate, and FX net.
- governed `BS.CASH.TOTAL`: remains **unavailable** for both ARS and USD in the fixture; no inferred/internal fallback was introduced.
- native-currency separation: preserved by the exact ARS/USD annual comparison.

The final transformation commit removed 5,109 lines and added 455 lines across 26 files; the deletions are overwhelmingly retired presentation/compatibility code, not semantic computation.

The pre-existing `smoke-views` null-Box fixture failure frozen in Phase 0 is not modified or reinterpreted by this PR.

## Evidence boundary

The committed fixture still lacks debt-position/activity source artifacts and a real professional pack. Therefore this PR does **not** claim new fixture evidence for debt balances or a whole-pack drilldown status histogram. It does not modify debt calculation, professional execution, or drilldown code; those remain protected by the regression suite and the Phase-0 requirement for dedicated real-pack evidence when a later PR touches those layers.
