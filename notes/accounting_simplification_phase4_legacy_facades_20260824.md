# Accounting simplification Phase 4 — narrow legacy facade surfaces

Date: 2026-08-24  
Base: `429dc407800b6fd975ae483efbb766ecbacd1d11`  
Accounting-policy change: **none**

## Invariant

This phase changes import/export reachability only. It does not alter semantic classification, monthly/annual calculations, debt/cash selection, Box scope, currency separation, displayed professional values, or drilldown membership.

## Facade census

| Facade | Broad static delegated names before | Repo-referenced names | Explicit legacy exports after |
| --- | ---: | ---: | ---: |
| `accounting.metrics.annual` | 24 | 4 | 3 |
| `accounting.metrics.frontier` | 18 | 1 | 0 |
| `accounting.professional.annual_dashboard_tables` | 19 | 5 | 4 |
| `accounting.professional.drilldown_wave4_base` | 141 | 16 | 10 |
| `accounting.professional.drilldown` | 48 | 23 | 18 |

The old `dir(delegate) -> globals()` pattern made every imported helper, constant, private function, and future implementation detail of the delegated module appear on the modern facade. Phase 4 replaces that with a caller-derived explicit compatibility list. Star imports, `dir()`/`vars()` introspection, and dynamic `getattr()` against these facades are treated as blockers and fail the census.

The census resolves absolute and relative imports and propagates transitive public requirements through jointly transformed facades. This exposed compatibility names such as `DEFAULT_TOLERANCE`, `INDEX_FILENAME`, `row_context_id`, status constants, and legacy formula helpers that had previously reached the public drilldown module only through two consecutive broad re-export layers.

Machine-readable caller evidence: `accounting_simplification_phase4_legacy_export_inventory_20260824.csv`.

## Professional drilldown deletion map

| Legacy route family | Governed replacement | Legacy still reachable? | Blocker |
| --- | --- | --- | --- |
| monthly operating revenue / rent flow | FlowCellSpec + semantic_measure_registry_v1 | compatibility only | historical/minimal rows without governed cell identity |
| monthly property OPEX flow | FlowCellSpec + semantic_measure_registry_v1 | compatibility only | historical/minimal rows without governed cell identity |
| monthly personal draws / withdrawal flow | FlowCellSpec + semantic_measure_registry_v1 | compatibility only | historical/minimal rows without governed cell identity |
| annual flow drilldowns (rent/OPEX/draws and other atomic flows) | annual governed metric artifacts; monthly FlowCellSpec membership exists | yes | annual lineage contract: wave4 explicitly refuses monthly recomputation for YEAR_RE rows |
| funding by actor/channel/cash-effect/target-Box | none complete | yes | FundingSupportSpec: professional support is broader than core funding and may include debt-linked/direct-obligation support |
| FX conversion proceeds/outflow/cost | partial FlowCellSpec/semantic measures | yes | FX grain mismatch: current specs require Box while some professional statement rows are total-by-currency |
| monthly and annual debt position | DebtPositionSpec executor | historical/minimal compatibility fallback | legacy source/table schemas remain supported |
| monthly and annual debt activity | DebtActivitySpec executor | historical/minimal compatibility fallback | legacy source/table schemas remain supported |
| monthly and annual validated cash position | cash position executor / governed validated selector | historical/minimal compatibility fallback | legacy cash table shapes remain supported |
| derived metric/formula drilldowns | DerivedMetricSpec executor | yes for historical/unregistered compatibility | legacy table identity and incomplete derived-spec coverage |
| diagnostic Box-level matrix and residual compatibility tables | no single closed contract | yes | diagnostic-specific table routing still lives in legacy router |

The full removal conditions are in `accounting_simplification_phase4_drilldown_deletion_map_20260824.csv`.

## Acceptance evidence

The transformed workspace was validated before the product commit:

- `make validate`: **PASS**;
- pytest: **294 passed, 1 warning**;
- unchanged warning: legacy invalid-`as_of_date` debt compatibility case;
- compilation and artifact/source/annual/publish contract checks: **PASS**;
- `make smoke-full`: **PASS**;
- fixture Stage-D counts remain per-flow `114`, per-party `277`, daily cash `4,284`;
- semantic leakage QA: **0 rows**;
- exact Phase-0 semantic classification census: **UNCHANGED**;
- exact Phase-0 ARS/USD annual governed values/statuses: **UNCHANGED**;
- governed `BS.CASH.TOTAL` remains `unavailable` for ARS and USD in the smoke fixture;
- native-currency separation remains intact.

Representative annual anchors still reconcile exactly: ARS operating revenue `33,075,422`, property OPEX `13,615,938.94`, net operating `19,459,483.06`, funding `1,427,956`, personal draws `29,412,662`, net after draws `-8,525,222.94`, and USD operating revenue/net operating `4,180`.

No generated reports, smoke outputs, professional packs, live ledgers, datasets, caches, or confidential records are committed.

## Evidence boundary

The repository smoke fixture still does not contain governed debt-position/activity source artifacts or a real professional pack. Phase 4 does not alter debt calculation or drilldown membership; later changes to those layers remain subject to the Phase-0 dedicated debt/real-pack reconciliation requirement.

## Deliberate non-change

`accounting/professional/drilldown_legacy.py` is not rewritten in this phase. Its remaining reachability is now explicit: funding support, FX grain, annual lineage, historical/minimal schemas, derived compatibility, and residual diagnostic routing are the blockers to later deletion.
