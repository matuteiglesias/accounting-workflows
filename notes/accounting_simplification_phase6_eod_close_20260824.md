# Accounting simplification Phase 6 — end-of-day architecture close

Date: 2026-08-24  
Consolidated codebase state audited: `49a01f14e0f5681e466efba60c480aff19dbc44f`  
Phase-0 behavioral source baseline: `136c977fe6f78e7373129b1eb218a0da05cb0165`  
Accounting-policy change in Phase 6: **none**

## Executive conclusion

Phases 1–5 are accepted as one consolidated architecture step.

The backend is materially smaller and, more importantly, its authority boundaries are clearer:

- the alternate `accounting.human` / static-front architecture is gone;
- the Flask viewer is downstream and read-only rather than a second accounting engine;
- diagnostics no longer live inside the professional production package;
- Stage D delegates generic I/O / hashing / partition / artifact registration infrastructure;
- broad dynamic legacy re-exports are gone;
- legacy reachability is explicit and machine-censused;
- governed cash selection remains in one authority and report surfaces now share one source-backed projection population.

This is a better baseline for future semantic work. The next work should **not** be generic pruning. It should close the small number of remaining semantic/lineage contracts that keep compatibility code alive, and then delete the corresponding legacy routes.

## 1. Structural before / after

The Phase-0 non-notebook Python census was 88 files and 27,849 physical lines. The post–Phase-5 closure census is exact over `accounting/**/*.py`, excluding `accounting/notebooks/**`.

| Surface | Phase 0 | Post Phase 5 | Delta |
| --- | ---: | ---: | ---: |
| accounting Python files | 88 | **76** | **-12 (-13.6%)** |
| accounting Python LOC | 27,849 | **24,794** | **-3,055 (-11.0%)** |
| `professional` files | 15 | **13** | **-2** |
| `professional` LOC | 8,886 | **8,045** | **-841 (-9.5%)** |
| explicit legacy facade modules | 5 target facades | **5** | count unchanged; behavior narrowed |
| broad dynamic re-export facades | 5 target facades | **0** | **-5** |
| legacy implementation modules (`*_legacy.py`) | 4 | **4** | unchanged; reachability now explicit |
| full regression tests | 280 | **298** | **+18 (+6.4%)** |
| semantic leakage fixture rows | 0 | **0** | unchanged |

The meaningful result is not simply lower LOC. Four legacy implementation modules still exist; Phase 4 deliberately made their reachability visible rather than pretending they had disappeared.

Current explicit compatibility facades:

- `accounting/metrics/annual.py`
- `accounting/metrics/frontier.py`
- `accounting/professional/annual_dashboard_tables.py`
- `accounting/professional/drilldown.py`
- `accounting/professional/drilldown_wave4_base.py`

Current legacy implementation modules:

- `accounting/metrics/annual_legacy.py`
- `accounting/metrics/frontier_legacy.py`
- `accounting/professional/annual_dashboard_tables_legacy.py`
- `accounting/professional/drilldown_legacy.py`

Broad `dir(delegate) -> globals()` re-export behavior: **0**.

## 2. Current package concentration

| Package | Files | LOC |
| --- | ---: | ---: |
| `professional` | 13 | 8,045 |
| `metrics` | 13 | 3,857 |
| `marts` | 6 | 3,763 |
| `contracts` | 6 | 1,154 |
| `diagnostics` | 4 | 1,041 |
| `debt` | 3 | 1,038 |
| root modules | 7 | 913 |
| `ledger` | 2 | 881 |
| `artifacts` | 2 | 860 |
| `stage_d` | 2 | 804 |
| `valuation` | 2 | 687 |
| `publish` | 3 | 462 |
| `core` | 2 | 439 |
| `management` | 3 | 439 |
| `support` | 8 | 411 |

Current largest modules:

1. `professional/drilldown_legacy.py` — 3,912 LOC
2. `marts/build.py` — 1,104
3. `professional/render_linked_digest.py` — 970
4. `ledger/ingest.py` — 870
5. `artifacts/manifest.py` — 859
6. `marts/debt.py` — 805
7. `stage_d/materialize.py` — 793
8. `marts/treasury.py` — 758
9. `metrics/build.py` — 756
10. `debt/resolve.py` — 745

This concentration now tells a more useful story than before: the largest remaining complexity is no longer an alternate frontend or generic orchestration duplication. It is concentrated in compatibility routing, transitional marts/metric construction, and a few substantive domain engines.

## 3. What moved today

### Phase 1 — remove alternate presentation architecture

Removed the entire `accounting.human` package, obsolete visualization surface, stale config/model/re-export seams, and the duplicate human-report Makefile pipeline. Publication became a governed metrics+debt artifact handoff. The Flask review UI remains in the separate `accounting-viewer` repository as a read-only downstream consumer.

### Phase 2 — restore package ownership

Moved forensic funding-lineage and professional-issue characterization out of `accounting.professional` and into `accounting.diagnostics`, with no algorithm change. Diagnostics are now forbidden from importing professional runtime code.

### Phase 3 — deduplicate Stage-D infrastructure

`stage_d/materialize.py` fell from 929 to 793 LOC by removing local CSV/hash/partition/dead-manifest infrastructure and delegating those responsibilities to shared support/artifact authorities. Semantic/cash orchestration was deliberately left unchanged.

### Phase 4 — make compatibility explicit

Replaced broad delegated namespace injection with explicit caller-backed compatibility exports. The most important reduction was `professional.drilldown_wave4_base`: about 141 implicitly visible legacy names became 10 explicit compatibility exports. `metrics.frontier` now has zero legacy compatibility exports.

The professional legacy deletion map is now explicit rather than implicit.

### Phase 5 — one governed cash projection population

Kept `accounting.cash_authority` as the sole validated-cash selector. Added a mechanical source-backed projection seam shared by annual metrics, frontend metric series, and professional annual cash tables. This removed three independent period/year/Currency/Box discovery algorithms and eliminated phantom unavailable Cartesian scopes without changing any available governed cash value.

## 4. End-of-day accounting reconciliation

Phase 6 reran the complete fixture-safe validation against the consolidated state.

### Full gates

- `make validate`: **PASS**
- pytest: **298 passed, 1 warning**
- unchanged warning: known legacy invalid-`as_of_date` debt compatibility case
- compile + artifact/source/annual/publish contract checks: **PASS**
- `make smoke-full`: **PASS** for its documented fixture-safe surface
- targeted accounting/professional invariant bundle: **48 passed**

The targeted bundle explicitly covers:

- property-business / Household scope;
- governed cash selection and shared cash projection;
- professional debt-position contract + integration;
- professional debt-activity contract + integration;
- professional drilldown reconciliation behavior.

### Frozen semantic totals

Exact Phase-0 semantic classification totals remain unchanged.

ARS:

- debt movement / principal: `1,576,761`
- personal withdrawal candidate: `29,412,662`
- core funding contribution: `1,427,956`
- operating rent revenue: `33,075,422`
- property OPEX legal: `147,000`
- property OPEX maintenance: `240,000`
- property OPEX services: `5,801,045.41`
- property OPEX taxes: `7,427,893.53`

USD:

- debt movement / principal: `2,240`
- operating rent revenue: `4,180`

Semantic leakage QA: **0 rows**.

### Frozen annual governed values

ARS:

- `IS.REVENUE.OPERATING = 33,075,422`
- `IS.RENT.TOTAL = 33,075,422`
- `IS.OPEX.PROPERTY = 13,615,938.94`
- `IS.NET.OPERATING = 19,459,483.06`
- `FUND.CONTRIB.TOTAL = 1,427,956`
- `DIST.DRAWS.PERSONAL = 29,412,662`
- `COV.NET.AFTER_DRAWS = -8,525,222.94`
- `COV.SAVINGS_RATE = -0.4381012030850936`

USD:

- revenue/rent/net operating/net after draws: `4,180`
- OPEX/funding/draws: `0`
- savings rate: `1.0`

### Explicit invariants rechecked

- **Household excluded from property-business OPEX:** targeted property-business scope tests pass; semantic leakage remains zero.
- **ARS/USD never co-aggregated:** annual `no_cross_currency_aggregation` QA passes; frozen ARS/USD values remain separate.
- **Funding unchanged:** core ARS funding remains `1,427,956`.
- **Validated cash unchanged in meaning:** smoke fixture still has no governed validated cash; `BS.CASH.TOTAL` remains unavailable for both ARS and USD, with no inferred/internal fallback.
- **Cash reporting population is now source-backed:** `annual_cash_uses_governed_validated_projection`, `annual_cash_never_sums_monthly_positions`, and `annual_cash_scopes_are_source_backed` all pass.
- **Annual flow aggregation:** `annual_flows_sum_monthly_flows` and `ratios_use_annual_aggregates` pass.
- **Debt stock remains separate from flow:** `debt_stock_not_mixed_with_flows` and `debt_activity_reconciles_or_residual_visible` pass.

## 5. Evidence boundaries that remain open

### Debt numeric fixture gap

The committed smoke path still does not build governed `monthly_debt_position.csv` / `monthly_debt_activity.csv`. Therefore Phase 6 does **not** claim a fresh fixture-level numeric debt-stock equality check.

What is established:

- no debt calculation code changed in Phases 1–5;
- debt position/activity contract and integration tests pass in the targeted 48-test bundle;
- annual debt stock/flow-separation QA passes;
- the known invalid-`as_of_date` correctness defect remains present and deliberately unresolved.

Any debt correctness PR must carry its own dedicated numeric position/activity fixture and before/after reconciliation.

### Real professional-pack gap

The repository fixture does not contain a real professional pack, so Phase 6 does **not** fabricate a whole-pack residual/status histogram or claim a fresh production-pack displayed-value reconciliation.

What is established:

- professional calculation/drilldown tests pass;
- governed cash/debt integrations pass;
- Phase 4 did not change membership logic, only export reachability;
- no real-pack data or confidential report outputs are committed.

Any future PR changing professional routing/table identity/drilldown membership still needs either a real-pack before/after run or a deliberately synthetic/sanitized professional corpus fixture.

### Existing smoke-views boundary

The pre-existing `make smoke-views` null-`Box` failure in `marts/build.py` is not fixed or reinterpreted by today's architecture cleanup. It should not be repaired incidentally inside unrelated refactors.

## 6. Reassessed tomorrow frontier

The old roadmap has changed materially because several planned cleanups are already complete.

Done and therefore removed from tomorrow's queue:

- prune alternate human/front architecture;
- move professional diagnostics out of production package;
- deduplicate Stage-D generic infrastructure;
- narrow broad legacy facades;
- consolidate governed cash projection.

The next frontier should be contract- and correctness-led.

### T0 — adopt this consolidated state as the new architecture baseline

Use post–Phase-5 `49a01f14e0f5681e466efba60c480aff19dbc44f` plus this Phase-6 close as the structural reference for future work. Continue to use the Phase-0 fixture numbers as the long-run accounting-behavior anchors until an intentionally semantic PR changes them.

A useful enabling improvement before large professional-route deletion is a **synthetic/sanitized professional regression corpus** that can run a whole-pack-like displayed-value/drilldown/status reconciliation in CI without confidential family records.

### T1 — P0 correctness: unify debt stock authority / invalid `as_of_date`

This is now the clearest first substantive task because it is the only characterized cross-layer correctness contradiction still known.

Required invariant:

```text
monthly debt position
    -> latest valid as_of within period/pair/currency
annual debt stock
    -> closing position, never annual sum
invalid latest period
    -> unavailable, no backfill
professional debt position
    -> same selected population
activity events
    -> remain independently preserved
```

The PR must be treated as an intentional semantic correctness change, not behavior-preserving cleanup. It needs a dedicated numeric debt fixture covering valid, partially invalid, and all-invalid stock cases plus independently known repayments/new claims.

### T2 — first-class `FundingSupportSpec`

This is the largest semantic blocker to deleting funding branches from `drilldown_legacy.py`.

Do **not** equate professional support with the core `funding_contribution` semantic bucket. The contract needs explicit membership/grain for at least actor, channel, cash effect, target Box and debt/direct-obligation support where applicable.

Acceptance must separately prove:

- core Phase-0 funding stays `1,427,956` ARS unless intentionally reclassified;
- broader professional support reconciles to its own governed membership;
- label/text inference disappears from current professional routing where the new ID is available.

### T3 — annual flow membership / lineage contract

Monthly rent/OPEX/draw atomic flows already have governed `FlowCellSpec` identities. Annual drilldowns remain legacy because annual rows must not silently recompute monthly detail and discard annual provenance.

Create a typed annual membership/lineage contract that composes governed monthly membership while preserving annual artifact/source identity.

This should unlock annual rent/OPEX/draw route-family deletion from `drilldown_legacy.py`.

### T4 — explicit FX total-vs-Box grain contract

Phase 4 exposed a real grain mismatch: current governed specs often require Box while some professional FX rows are total-by-currency.

Resolve this with an explicit contract, not a generic nullable-Box helper. Preserve native Currency evidence and keep USD-CCL as a management projection over native facts.

### T5 — shrink `professional/drilldown_legacy.py` route family by route family

The 3,912-LOC legacy router is now the single largest source hotspot, but wholesale rewrite/deletion remains the wrong move.

Use the Phase-4 deletion map and delete branches only when their replacement contract closes:

1. current monthly atomic flows whose supported corpus has stable governed cell IDs;
2. debt/cash historical fallbacks once supported corpus reachability proves them unnecessary;
3. annual flows after T3;
4. funding after T2;
5. FX after T4;
6. derived/diagnostic routes after stable identities or explicit retirement.

Target end state:

```text
professional/drilldown.py
    -> explicit typed router
       -> atomic flow executor
       -> cash executor
       -> debt position executor
       -> debt activity executor
       -> derived executor
       -> bounded compatibility adapter only where still proven necessary
```

### T6 — consolidate the metrics engine

This remains one of the largest structural wins, but today's work makes the prerequisite clearer.

Current `metrics` is still 3,857 LOC, with old registry/builders/derive/`metric_values` machinery coexisting with semantic frontier/annual production.

Before changing it, run a metric-ID consumer census and map each current output to its canonical semantic source. Then target:

```text
semantic/domain marts
      ↓
canonical governed metric facts
      ↓
monthly + annual projections
      ↓
legacy metric_values adapter only where a consumer still proves necessary
```

This should reduce duplicated calculation authority rather than merely move code.

### T7 — retire transitional `marts/build.py` responsibilities

At 1,104 LOC, `marts/build.py` is now the second-largest source module. It should not be attacked just because it is large.

After the metrics consumer/source migration is clearer, census which transitional views remain live. Retire or relocate only the ones whose consumers are gone. Keep the existing null-Box smoke-views defect separate from this cleanup unless a dedicated correctness decision explicitly addresses it.

## 7. Recommended sequencing

The revised sequence is:

```text
new consolidated baseline
        ↓
T1 debt correctness
        ↓
T2 funding support contract
T3 annual lineage contract
T4 FX grain contract
        ↓
route-family legacy drilldown deletion
        ↓
metrics engine consolidation
        ↓
transitional marts/build retirement
```

A synthetic professional regression corpus can be developed as enabling infrastructure alongside T1–T4 and should be in place before aggressive professional-route deletion.

## 8. What not to do next

Do not optimize tomorrow around raw LOC removal.

Specifically avoid:

- deleting `drilldown_legacy.py` wholesale;
- splitting large modules without first clarifying authority;
- introducing generic untyped Currency/Box/period helpers;
- merging stock and flow semantics;
- moving governed cash authority again;
- treating broader funding support as identical to core funding contribution;
- fixing the `smoke-views` null-Box issue incidentally;
- rewriting historical audit documents to hide old architecture;
- committing real professional packs, confidential ledgers, caches, or generated reports as regression evidence.

## 9. Phase-6 acceptance

Today's cleanup can be accepted as a consolidated new codebase state if the following remain true:

- full regression suite green;
- Phase-0 frozen semantic/annual values unchanged;
- Household/property-business scope invariant green;
- native currencies separated;
- governed cash fails closed without validated evidence;
- debt stock/flow semantics remain separated;
- no broad dynamic compatibility exports return;
- future professional/debt semantic changes bring the missing dedicated evidence rather than relying on generic smoke success.

All of those conditions are satisfied by the Phase-6 closure run described above, subject to the explicitly documented debt-fixture and real-professional-pack evidence boundaries.
