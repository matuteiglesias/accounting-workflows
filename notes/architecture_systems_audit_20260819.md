# Architecture / systems-engineering audit — 2026-08-19

## Mandate and safety boundary

This is an engineering baseline and an adversarial investigation, not an
accounting-policy decision or a refactoring plan. It examines the tree at
`a783954` (the merge of PR #52). It does not use live inputs, regenerate an
accounting artifact, publish anything, or change an accounting rule.

The governing sequence remains:

```text
canonical ledger -> materialization -> semantic marts -> debt -> metrics
                 -> human reports -> professional pack -> drilldowns
```

The USD/CCL valuation and management outputs remain additive sidecars. Nothing
in this audit recommends making them canonical transaction truth.

## Method

The census parses every `accounting/**/*.py` file with Python's `ast` module.
LOC counts physical lines; SLOC excludes blank and comment-only lines (it does
not exclude docstrings). Function length is inclusive from `lineno` through
`end_lineno`. The decision count is a deliberately simple review trigger: AST
`if`, conditional expression, loop, `try`, `match`, and Boolean-expression
nodes within a function. It is not cyclomatic complexity.

Static internal-import edges resolve `accounting.*` imports to repository
modules. This understates dependencies expressed as artifact filenames, CLI
composition in the Makefile, and pandas schemas; those are reviewed separately.
Fan-in therefore means “number of Python modules statically importing this
module,” not runtime callers. Artifact-family counts below are conservative
manual groupings of filename literals and reads/writes, not a claim that every
literal is reached in one execution.

## Exact census

The complete Python tree contains **70 files, 23,310 LOC, 19,216 SLOC, 744
functions, and 34 classes**. The four largest packages (`professional`,
`metrics`, `marts`, and `human`) contain 15,399 LOC, or **66.1%** of all Python
LOC. Their 12,744 SLOC are **66.3%** of all Python SLOC.

| package | files | LOC | SLOC | functions |
| --- | ---: | ---: | ---: | ---: |
| `professional` | 7 | 6,414 | 5,156 | 186 |
| `metrics` | 11 | 3,459 | 2,957 | 117 |
| `marts` | 5 | 2,941 | 2,552 | 64 |
| `human` | 6 | 2,585 | 2,079 | 137 |
| `debt` | 5 | 1,042 | 857 | 20 |
| `stage_d` | 2 | 978 | 747 | 18 |
| `notebooks` | 1 | 940 | 749 | 34 |
| `ledger` | 2 | 847 | 679 | 21 |
| `artifacts` | 2 | 828 | 732 | 18 |
| `valuation` | 2 | 687 | 621 | 22 |
| `publish` | 4 | 480 | 392 | 25 |
| `management` | 3 | 445 | 396 | 16 |
| `core` | 2 | 439 | 324 | 9 |
| root modules | 4 | 438 | 337 | 21 |
| `support` | 8 | 372 | 297 | 20 |
| `contracts` | 2 | 142 | 122 | 7 |
| `viz` | 2 | 141 | 108 | 5 |
| `diagnostics` | 2 | 132 | 111 | 4 |

The initial remote estimate omitted, among other surfaces, the notebook helper,
linked-digest renderer, and the now larger drilldown module. The exact current
size of `professional/drilldown.py` is 3,938 LOC rather than 3,499.

### Largest modules and function triggers

| module | LOC | SLOC | functions | longest function | static fan-out / fan-in |
| --- | ---: | ---: | ---: | ---: | ---: |
| `professional/drilldown.py` | 3,938 | 3,182 | 96 | 876 | 3 / 0 |
| `human/front.py` | 1,251 | 1,001 | 57 | 79 | 4 / 0 |
| `marts/build.py` | 1,104 | 852 | 24 | 205 | 6 / 0 |
| `professional/render_linked_digest.py` | 970 | 670 | 23 | 258 | 1 / 0 |
| `stage_d/materialize.py` | 967 | 740 | 17 | 278 | 7 / 1 |
| `notebooks/accounting_reports/_shared.py` | 940 | 749 | 34 | 115 | 0 / 0 |
| `ledger/ingest.py` | 836 | 672 | 20 | 155 | 5 / 2 |
| `artifacts/manifest.py` | 827 | 731 | 18 | 302 | 0 / 6 |
| `marts/debt.py` | 805 | 757 | 11 | 412 | 1 / 0 |
| `metrics/build.py` | 754 | 641 | 18 | 130 | 3 / 2 |
| `debt/resolve.py` | 745 | 620 | 13 | 363 | 2 / 3 |
| `valuation/usd_ccl.py` | 686 | 620 | 22 | 148 | 0 / 1 |
| `marts/semantic.py` | 648 | 580 | 24 | 126 | 0 / 2 |
| `management/usd_ccl_flows.py` | 370 | 332 | 13 | 126 | 0 / 1 |

The `>800 SLOC` trigger fires for drilldown, front, and marts/build. The
`>80 LOC` function trigger fires 38 times. Most consequential examples are:

| function | LOC | decision nodes | assessment |
| --- | ---: | ---: | --- |
| `professional.drilldown.build_professional_flow_drilldowns` | 876 | 63 | accidental orchestration plus duplicated contract knowledge |
| `marts.debt._build_monthly_debt_activity` | 412 | 37 | mixed domain and assembly complexity; investigate boundary |
| `debt.resolve.resolve_repayments` | 363 | 27 | largely essential allocation complexity, but needs characterization |
| `professional.drilldown._build_derived_cell` | 321 | 67 | duplicated presentation/semantic complexity |
| `marts.cash.build_monthly_cash_close` | 320 | 32 | flow/stock boundary in one builder; domain-sensitive |
| `artifacts.manifest.artifact_contract_for_name` | 302 | 39 | accidental filename dispatch; central but open-ended |
| `stage_d.materialize.main` | 278 | 25 | CLI/orchestration complexity rather than accounting complexity |
| `metrics.annual.build_annual_balance_dashboard` | 161 | 71 | dense table/formula dispatch; weak declarative contract signal |
| `metrics.frontier.build_metrics_frontier` | 140 | 59 | contract assembly complexity |
| `professional.table_contracts._infer_metric_contract` | 102 | 56 | presentation infers semantics from names |

## Dependency map

The static Python import graph has 52 resolved internal edges and no cycle.
Its package-level shape is:

```text
support / logging / scope / artifact contracts
       ^          ^             ^
       |          |             |
ledger ----> core |             |
  ^               |             |
  +---- debt ------+             |

stage_d ---> core + marts.cash + marts.semantic
marts.build ---> core + artifact/support contracts
marts.debt ---> debt.resolve

metrics.build/frontier ---> artifact contracts + scope
human ---> metrics.build/views/drilldown + human.tables
professional.drilldown ---> table_contracts + scope
publish.latest ---> artifact/publish/latest-link contracts

management.run ---> valuation.usd_ccl + management.usd_ccl_flows
```

No import cycle is reassuring but incomplete: artifact exchange is the dominant
integration mechanism. Several high-level modules have static fan-in zero
because Make targets or external scripts call them as entrypoints. Conversely,
`stage_d.materialize` has the highest static fan-out (7), followed by
`marts.build` (6) and `ledger.ingest` (5). The graph also exposes naming/layer
ambiguity: Stage D imports semantic and cash mart builders, while `marts.build`
also consumes materialized artifacts. This is not a Python cycle, but the name
“Stage D materialization” spans both mechanical time-series materialization and
semantic/cash construction.

The Makefile has 721 LOC and 58 explicit target declarations. It preserves
consequential command names, but aliases (`run`, `run-all`, `run-accounting`,
`run-accounting-full`, `build-all`) and recursive prerequisites obscure which
stages are live, fixture-safe, or publication-capable. That is control-plane
change amplification, not an accounting-rule defect.

## Hotspot findings

### 1. Professional drilldown — **EXTRACT CONTRACT**, then **SPLIT**

The hypothesis is confirmed. The module performs at least nine distinct jobs:

1. table/cell routing;
2. semantic bucket and subbucket reconstruction;
3. measure selection;
4. source-artifact discovery and fallback;
5. flow, cash-stock, debt-activity, and debt-stock selection;
6. annual formula reconstruction;
7. reconciliation and tolerance evaluation;
8. HTML/CSV serialization; and
9. index, manifest, QA, and diagnostics production.

It directly names at least seven accounting input families: semantic split,
classification audit, operating statement, annual metrics, cash close, debt
activity, and debt position. It therefore crosses semantic, metric, cash-stock,
debt-stock, presentation, and publication-evidence concerns. Its low import
fan-out (3) is misleading; filenames, table IDs, labels, column fallbacks, and
callables embedded in `CellSpec` are the real dependencies.

Concrete evidence of fragile local authority is stronger than file length:

* `FX_TREASURY_TABLE_IDS`, `FX_MEASURES`, and
  `_fx_treasury_measure_for_row` are each defined twice. Python silently uses
  the second function; the first implementation is dead.
* `_spec_for_cell`, `_cash_bridge_line_spec`,
  `_annual_professional_line_spec`, and `_build_derived_cell` encode table IDs,
  source filters, measures, and formulas in separate dispatch surfaces.
* Cash helpers search alternative value columns, while source discovery searches
  alternative run roots. These are compatibility responsibilities, not lineage
  membership itself.
* The 876-line entrypoint loads and normalizes seven source frames, routes every
  table/cell family, writes detail files, reconciles values, and emits indexes.

Classification: **duplicated complexity caused by weak interfaces**, with some
essential domain complexity for flow versus stock reconciliation. Splitting the
file before establishing a membership/reconciliation contract would only move
branches between files.

### 2. Semantic authority and USD/CCL management — **EXTRACT CONTRACT**

`marts.semantic` is not large enough to trigger the module threshold, but it is
policy dense. It owns row classification, funding dimensions, leakage QA,
semantic aggregation, and construction of the operating statement. The
operating-statement builder explicitly selects:

```text
operating_revenue -> amount_in
property_opex -> amount_out
funding_contribution -> amount_in
family withdrawals -> amount_out
debt/internal transfer -> amount_abs
treasury FX -> subbucket-specific in/out, with net composition
```

PR #52 had to add an independent `MEASURE_DIRECTIONS` and
`TREASURY_MEASURE_DIRECTIONS` mapping to `management/usd_ccl_flows.py` so the
sidecar would reproduce those selections. The production change touched one
module and one regression-test module, but the concept now has authorities in
at least semantic statement construction, management projection, professional
drilldown, and table/annual metric dispatch. The apparent two-file change count
therefore understates semantic change amplification.

This is the audit's primary finding: management did not need a new accounting
interpretation; it needed a machine-readable projection of an already-approved
semantic measure. A stable semantic-measure contract should state bucket,
optional subbucket, selected direction/measure, aggregation kind, and eligibility
for each consumer. It must be generated or validated from one authority, not
created by choosing which current mapping is “right” during a refactor.

Classification: **duplicated complexity caused by weak interfaces**. No semantic
mapping should move until Matías approves the contract rows and characterization
proves equality with every existing native output.

### 3. Marts build and Stage D — **SIMPLIFY**, boundary first

`marts/build.py` combines report-folder discovery, schema fallback, period
normalization, flow aggregation, stock-end selection, view construction,
serialization, manifests, QA, and CLI orchestration. It has six static internal
dependencies and many artifact names. Its 205-line `export_views` and 149-line
`main` are orchestration triggers; its aggregation helpers contain justified
Box/Currency invariants.

`stage_d/materialize.py` similarly combines low-level materializers with semantic
and cash mart invocation, artifact-contract emission, partition metadata, latest
link behavior, and a 278-line CLI. It is the only module above the requested
fan-out trigger (>6). Because it handles per-period flows and ending/cumulative
positions, it also fires the “flow and stock in one module” review trigger.

Classification: mostly **accidental orchestration complexity** around essential
flow/stock logic. First make the stage manifest explicit about produced
contracts; only then separate pure builders from command orchestration. Do not
merge flow sums and period-end stock selection into a generic aggregation.

### 4. Metrics and human presentation — **INVESTIGATE** / selective **SIMPLIFY**

Metrics is broad rather than dominated by one file. It contains registry,
derivation, views, annualization, frontier, validation, drilldown, IO, and build
surfaces. Three decision-dense functions (`annual`, `frontier`, and professional
contract inference) suggest that metric metadata is partly declarative and
partly reconstructed from table names and branching code.

Human rendering statically depends on metrics views/drilldowns/builders, while
`human.tables` names many concrete artifact files. This is acceptable for
presentation selection, but aggregation in presentation becomes a reverse-layer
dependency when it decides amount direction, period stock behavior, or semantic
membership. The audit found enough evidence to investigate contract coverage,
not enough to recommend deleting a particular human surface.

Classification: a mix of **essential presentation complexity** and **accidental
dispatch complexity**. Measure the number of metric IDs with a complete registry
entry versus IDs inferred from labels before extracting anything.

### 5. Ledger, debt, valuation, and artifact manifest — mostly **KEEP**

`ledger.ingest` is long but its responsibilities—schema validation,
canonicalization, provenance, scope, and deterministic output—belong near the
canonical boundary. Its CLI can be separated eventually, but there is no
evidence that ledger currently imports presentation policy.

Debt resolution's 363-line allocator and debt mart's flow/stock builders carry
substantial domain complexity. They should be protected with characterization
tests rather than mechanically decomposed. `valuation.usd_ccl` remains a bounded
sidecar and has one static consumer; it does not import canonicalization or
presentation modules.

`artifacts.manifest.artifact_contract_for_name` is a 302-line filename dispatcher
with 39 decision nodes and fan-in 6. It is a healthy central authority compared
with duplicating contracts, but its open-ended branch structure is a scaling
warning. Prefer data-driven registrations only if they preserve the current
fail-closed and versioning behavior.

## Cross-cutting review-trigger disposition

| trigger | evidence | disposition |
| --- | --- | --- |
| module >800 SLOC | drilldown, front, marts/build | review warranted; only drilldown proves mixed independent responsibilities |
| function >80 LOC | 38 functions | characterize first; length alone is not a split rule |
| function >12 decisions | many; highest 71/67/63 | declarative dispatch candidate in metrics/professional; domain review in debt |
| internal fan-out >6 | stage_d/materialize = 7 | simplify orchestration boundary |
| selector in >2 layers | direction/measure selection in semantic, management, professional/metrics | extract approved semantic-measure authority |
| >3 artifact families | professional drilldown, marts/build, metrics/build, human/document, publish/latest | strongest coupling in drilldown/build; publishing breadth is expected |
| both flow and stock | Stage D, cash/debt marts, drilldown | keep distinct aggregation semantics; contract them explicitly |
| drilldown reconstructs classification | bucket/subbucket masks and table-ID mappings | replace with membership/reconciliation contract |
| presentation aggregates accounting | annual/professional/human formula surfaces | distinguish governed formula projection from fresh classification |
| repeated `fillna(0)` | semantic, debt, metrics, drilldown, rendering | audit per measure; zero is unsafe for unavailable/review states |
| repeated Currency/Box/period filters | marts, metrics, professional, management | shared row grain/contracts, not a generic untyped filter helper |

## Stable-contract assessment

| proposed contract | current state | recommendation |
| --- | --- | --- |
| canonical transaction | artifact manifest + ingest schemas exist | **KEEP**, tighten only through approved version changes |
| semantic row | classification audit columns exist | **KEEP / formalize version**, preserve provenance and review status |
| semantic measure | embedded in statement, management, and drilldown code | **EXTRACT CONTRACT**; highest leverage |
| flow aggregate | semantic split and operating statement partially provide it | **FORMALIZE**, including additive measure and grain |
| stock snapshot | cash/debt tables implement separate snapshot rules | **FORMALIZE SEPARATELY**; never treat as additive flow |
| valuation | manifest and sidecar bindings are explicit | **KEEP** as additive sidecar |
| presentation | registry/frontier/table contracts are partial | **INVESTIGATE** coverage before consolidation |
| drilldown membership/reconciliation | embedded table-specific callables and branches | **EXTRACT CONTRACT**, with typed flow/stock/formula strategies |

## Failure locality and change amplification

The pipeline generally preserves artifacts between layers, which helps diagnosis,
but semantic meaning can be reintroduced after that boundary. A classification
error propagates visibly through semantic artifacts; a downstream reconstruction
error can instead disagree with native outputs while retaining plausible totals.
PR #52 is the concrete example: valuation stayed bounded and correct, while the
management consumer selected rows by a second direction policy.

Representative change surfaces:

| concept change | current authorities/consumers requiring review | locality risk |
| --- | --- | --- |
| property OPEX direction | semantic statement, management projection, professional drilldown, metric/table mappings, tests | high: plausible double-count or sign error |
| Box scope | scope contract, ingest, Stage D, marts, metrics, professional selectors, publication checks | medium/high: omissions can look like valid zero |
| FX subbucket measure | semantic statement, management direction map, professional FX resolver, annual/table contracts | high: table-dependent disagreement |
| cash/debt period selection | cash/debt marts, annual metrics, professional drilldown fallback | high: summing stock can look numerically valid |

The target metric should be recorded as **semantic edit distance**: for an
approved concept change, count production authorities that encode its membership
or selected measure (tests and consumers that only validate/consume do not count
as authorities). The desired value is one. Today, OPEX and FX measure selection
are at least three.

## At most five bounded interventions

### 1. Extract a versioned semantic-measure contract — **highest leverage**

* **Invariant protected:** native statement measures and management projections
  select identical approved flow directions without making valuation canonical.
* **Evidence:** mappings exist independently in semantic statement construction,
  management projection, and professional drilldown/metric dispatch.
* **Boundary:** a reviewed, versioned mapping of semantic bucket/subbucket to
  measure and aggregation kind, consumed by native and sidecar projections.
* **Simplifies:** management direction maps; portions of statement and drilldown
  measure dispatch. No module is initially deleted.
* **Dependency estimate:** semantic authorities from >=3 to 1; consumers gain one
  contract dependency rather than local mappings.
* **LOC/responsibility estimate:** modest LOC reduction (roughly 50–150), large
  reduction in policy responsibility.
* **Risk:** high if contract rows accidentally redefine policy; mitigate by
  deriving characterization fixtures from current approved outputs.
* **Required evidence:** exact native before/after equality, PR #52 fixture,
  OPEX/funding/draw/FX cases, unknown/review and unavailable valuation cases.

### 2. Introduce a typed drilldown membership/reconciliation contract

* **Invariant protected:** every displayed cell reconciles to governed source
  membership, preserving flow versus snapshot versus formula semantics.
* **Evidence:** seven source families and several parallel dispatch functions in
  one 3,938-line module.
* **Boundary:** table/metric ID -> source contract, grain keys, membership rule,
  measure, aggregation kind, and tolerance. Unsupported rows fail closed.
* **Simplifies:** `_spec_for_cell`, line-spec mappings, source fallbacks, and
  parts of `_build_derived_cell` / the 876-line entrypoint.
* **Dependency estimate:** table-specific routing surfaces from at least four to
  one registry plus three typed executors (flow, stock, formula).
* **LOC/responsibility estimate:** 500–1,000 fewer procedural lines after a later
  migration; responsibilities from nine toward four. Do not promise net LOC
  reduction in the contract-first PR.
* **Risk:** high around annual formulas and stock snapshots.
* **Required evidence:** byte/row-equivalent indexes where deterministic, full
  reconciliation fixtures, missing-source behavior, and explicit stock tests.

### 3. Remove the duplicate FX resolver definition in drilldown

* **Invariant protected:** preserve the currently effective second definition;
  no output or semantic change.
* **Evidence:** duplicated constants and same-named function at two consecutive
  locations; the first is unreachable after module import.
* **Boundary:** one tested resolver.
* **Simplifies:** one module only; eliminates silent shadowing.
* **Dependency estimate:** unchanged; implementation authorities 2 to 1 locally.
* **LOC/responsibility estimate:** about 35–45 lines removed.
* **Risk:** low only after a resolver characterization test confirms which
  behavior PR #52 expects, especially compact-table fallback.
* **Required evidence:** tests for explicit `measure`, metric fallback, each
  single-measure table, compact/all-measures, and unknown measure.

### 4. Separate pure stage builders from CLI/artifact orchestration

* **Invariant protected:** run IDs, hashes, scope, atomic latest handling, and
  exact flow/stock outputs.
* **Evidence:** Stage D fan-out 7 and 278-line CLI; marts build mixes discovery,
  aggregation, exports, QA, and CLI.
* **Boundary:** pure typed builder inputs/outputs plus a thin stage runner that
  owns paths/manifests.
* **Simplifies:** `stage_d.materialize.main`, `materialize_all`,
  `marts.build.export_views`, and `main`.
* **Dependency estimate:** pure builders fan-out toward 1–2; runner retains broad
  control-plane fan-out explicitly.
* **LOC/responsibility estimate:** little initial LOC reduction; reduce each
  module from roughly 6–8 responsibilities to builder versus runner roles.
* **Risk:** medium due to output paths and run metadata.
* **Required evidence:** fixture hashes/schema comparisons, idempotency, run ID,
  artifact-contract QA, and no latest-link mutation during tests.

### 5. Measure registry coverage before simplifying metrics/human dispatch

* **Invariant protected:** all current metric IDs, labels, formulas, and frontend
  suitability remain unchanged.
* **Evidence:** decision-dense annual/frontier/contract inference and concrete
  artifact-name coupling in human tables.
* **Boundary:** an investigation that lists every metric/table ID and whether its
  grain, measure, formula, source, and presentation status are explicit.
* **Simplifies:** nothing in the first PR; it is a decision packet.
* **Dependency estimate:** baseline first; intervention proceeds only if inferred
  authorities can be replaced by existing registry fields.
* **LOC/responsibility estimate:** unknown until coverage is measured.
* **Risk:** low for the inventory, potentially high for later consolidation.
* **Required evidence:** registry-to-produced-artifact coverage and orphan/duplicate
  IDs, without reading live accounting data.

## Decision and completion record

```text
Changed: investigation document only
Accounting rule changed: no
Fixture/test evidence: static AST census; source inspection; existing PR #52 regression history
Commands run: Python stdlib AST census; rg/sed source inspection; git history inspection
Run ID: none
Outputs inspected: source code and git history only; no generated accounting outputs
Live inputs accessed: no
Publication performed: no
Totals/invariants checked: code census totals and static import graph; no accounting totals
Blocked accounting decision: approval of a semantic-measure contract remains with Matías
Next bounded action: intervention 1 decision packet and characterization matrix
```
