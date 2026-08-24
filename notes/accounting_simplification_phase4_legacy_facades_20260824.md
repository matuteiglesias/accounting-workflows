# Accounting simplification Phase 4 — narrow legacy facade surfaces

Date: 2026-08-24  
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

The old `dir(delegate) -> globals()` pattern made every imported helper, constant, private function, and future implementation detail of the delegated module appear on the modern facade. Phase 4 replaces that with a caller-derived explicit compatibility list. Star imports, `dir()`/`vars()` introspection, and dynamic `getattr()` against these facades are treated as blockers and fail the transformation.

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

## Deliberate non-change

`accounting/professional/drilldown_legacy.py` is not rewritten in this phase. Its remaining reachability is now explicit: funding support, FX grain, annual lineage, historical/minimal schemas, derived compatibility, and residual diagnostic routing are the blockers to later deletion.
