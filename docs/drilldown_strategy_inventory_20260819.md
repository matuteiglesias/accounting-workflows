# Residual professional drilldown strategy inventory — 2026-08-19

## Decision this audit supports

This is a static audit of `accounting/professional/drilldown.py` at commit
`6d4b7fa`. It makes **no production change** and does not propose a common
flow/stock/formula `CellSpec`.

The question is narrower: after semantic measure authority moved out of the
drilldown, how much procedural routing remains, and how much of the file exists
only to repeat flow-membership selection?

## Executive answer

`drilldown.py` contains 3,918 physical lines. The inventory attributes **770
source LOC (19.7% of the file)** to atomic-flow membership and routing. This is
the code that repeatedly answers “which semantic rows belong to this cell?”;
it excludes HTML/CSV rendering, source discovery, reconciliation output,
logging, orchestration, cash/debt snapshot selection, formulas, compatibility
fallbacks, and unsupported guards.

That 770-LOC estimate is the relevant upper bound for the next flow-only
intervention. It does **not** justify a mega abstraction spanning flow, stock,
formula, and quality paths. Snapshot selection and formulas have materially
different membership and fallback behavior.

The inventory finds **44 primary executor routes** across all seven
families: 20 atomic-flow, 3 cash-snapshot, 4 debt-snapshot/activity, 3 derived
formula, 3 quality ratio, 5 compatibility fallback, and 6 explicit unsupported
guards. “Route” is defined below; aliases inside one predicate branch are not
double-counted.

No representative professional pack is committed, so the number of runtime
cells exercised by each family is **not measurable from repository-safe
evidence**. Cell counts are data-dependent (`rows × period columns`) and are
therefore reported as `not_measurable_without_a_professional_pack`, rather than
invented from table IDs. The builder already records total runtime cells in its
manifest when an authorized pack is built.

The machine-readable summary is
`diagnostics/drilldown_strategy_inventory_20260819.csv`.

## Taxonomy and counts

| Family | Static routes | Attributable LOC | Source artifacts | Current executor |
|---|---:|---:|---|---|
| `ATOMIC_FLOW` | 20 | 770 | semantic split, classification audit, monthly statement, annual metrics | direct `CellSpec` executor plus flow branches in `_build_derived_cell` |
| `CASH_SNAPSHOT` | 3 | 531 | monthly cash close | cash-control and annual cash companion executors |
| `DEBT_SNAPSHOT` | 4 | 250 | monthly debt position/activity | debt position/activity and annual companion executors |
| `DERIVED_FORMULA` | 3 | 167 | monthly statement, annual metrics, semantic split | annual formula and statement formula branches |
| `QUALITY_RATIO` | 3 | 94 | annual metrics | annual formula executor |
| `COMPATIBILITY_FALLBACK` | 5 | 279 | semantic split and classification audit | annual professional label fallback |
| `UNSUPPORTED` | 6 | 61 | none, or the missing expected artifact | explicit guards in the dispatcher/build loop |

The attributable LOC column is deliberately **non-exhaustive**. The 2,152 LOC
classified above do not include 1,766 shared lines for rendering, serialization,
source location, logging, CLI, comments/blank lines, and the common cell loop.
Shared infrastructure is not evidence for a family-specific abstraction.

## Counting method

### Route count

A route is one statically distinct executor/predicate strategy that can change
membership, snapshot selection, formula components, fallback, or support
status. Multiple spelling aliases handled by the same branch count as one
route. A table ID routed to the same executor but with a different stock or
semantic membership contract counts separately. Runtime cells do not count as
routes.

The 20 `ATOMIC_FLOW` routes break down as follows:

1. fifteen direct semantic-table routes in `_spec_for_cell` (including the
   two box bridge strategies and the five FX table contracts);
2. one operating-statement semantic-lineage route;
3. one annual-metric-to-semantic-detail route;
4. two cash-bridge flow routes (stable funding contract and explicit line
   membership);
5. one annual funding companion route.

The 14 human label alternatives inside `_cash_bridge_line_spec` are predicate
aliases within one cash-bridge executor route. They contribute LOC but are not
inflated into 14 additional routes. Conversely, the five compatibility labels
are counted separately because they only execute after exact annual-row lookup
fails and include a formula fallback with different behavior.

### LOC attribution

LOC uses inclusive physical source ranges at the audited commit, including
comments and blank lines inside an attributable range. A range is assigned to
one family only. Shared orchestration is excluded rather than apportioned.

Atomic-flow attribution covers the predicate helpers, direct `CellSpec`
routing, statement/funding selectors, semantic/rule masks, cash-bridge line
membership, cash-bridge semantic-row routing, and the annual funding companion.
The exact inclusive ranges are 160–247, 268–420, 726–729, 732–733, 736–778,
864–1194, 1445–1571, and 2625–2646; they sum to 770 physical LOC. It does not
count the shared derived dispatcher or compatibility reconstruction executor,
even though both also read semantic rows.

This method answers the architectural question more conservatively than a
search for `semantic_bucket`: it counts procedural membership machinery, not
every diagnostic string or rendering reference to a semantic column.

## Family behavior

### `ATOMIC_FLOW`

**Membership.** Period or year and Currency are the common boundary. Depending
on the route, predicates additionally use Box, semantic bucket/subbucket,
semantic rule IDs, stable funding dimensions, Lugar, actor, cash path, or an FX
mask. Atomic measure selection is now governed elsewhere; this family still
owns procedural membership routing.

**Fallback.** Matching semantic rows are expanded to classification-audit rows
using sampled transaction IDs when available. If audit expansion is empty, the
semantic subset remains the evidence. Unknown route/measure combinations are
unsupported; they do not widen to all flows.

**Tolerance.** Reconciliation uses the caller-provided tolerance, default
`1e-6`.

### `CASH_SNAPSHOT`

**Membership.** Period/year, Currency, Box, metric, and recognized physical
value column. Annual paths select a latest close rather than sum months.

**Fallback.** Cash-control paths contain base-box reconstruction and legacy
value-column recognition. Missing evidence produces empty/error states, not a
semantic-flow reconstruction.

**Tolerance.** Caller tolerance applies to displayed-versus-selected snapshot
reconciliation.

### `DEBT_SNAPSHOT`

This family includes debt activity because it shares a dedicated debt executor
boundary, not because activity is a stock. Debt position selects the latest
`as_of_date`; debt activity sums explicit activity types. Membership uses
Currency, pair/debtor/creditor, component or activity type. Neither path falls
back to generic semantic-flow membership.

### `DERIVED_FORMULA`

Formula routes identify component statement lines or annual metric IDs and
recompute the displayed result. The three routes are operating result,
coverage-after-funding/draws, and the non-ratio annual coverage formula. Their
evidence is component rows, not one “matching” transaction.

### `QUALITY_RATIO`

The three ratio routes are operating margin, OPEX/rent, and draws/operating
result. They use annual metric components. Division returns zero when the
denominator is within `DEFAULT_TOLERANCE`; the displayed residual still uses
the caller tolerance.

### `COMPATIBILITY_FALLBACK`

Five presentation-label routes reconstruct rent total, rent by property, OPEX
total, OPEX category, or net operating only when no exact annual metric row is
available. Membership may accept legacy rule IDs as well as semantic pairs.
Unknown labels are unsupported. This is intentionally classified apart from
primary atomic flow: deleting it requires compatibility evidence, not a new
generic flow selector.

### `UNSUPPORTED`

Six guard families cover unknown table/line/measure, missing Currency, missing
source, stock accidentally entering a flow path, oversized tables, and missing
annual source rows. These paths explicitly skip, error, or mark unsupported;
none silently broadens membership.

## Source-artifact touch matrix

| Artifact | Flow | Cash | Debt | Formula/ratio | Compatibility |
|---|:---:|:---:|:---:|:---:|:---:|
| `monthly_flow_semantic_split.csv` | yes | no | no | component evidence | yes |
| `classification_audit.csv` | lineage expansion | no | no | no | lineage expansion |
| `monthly_operating_statement.csv` | statement lineage | no | no | yes | no |
| `annual_balance_dashboard_metrics.csv` | annual source/detail | no | debt overview evidence | yes | absence triggers fallback |
| `monthly_cash_close.csv` | no | yes | no | no | no |
| `monthly_debt_position.csv` | debt-linked evidence only | no | yes | no | no |
| `monthly_debt_activity.csv` | debt-linked evidence only | no | yes | no | no |

## Recommended next bounded action

If an authorized fixture-safe professional pack becomes available, run the
existing builder once and join its drilldown index back to this route inventory
to replace `observed_cells` with empirical counts. Until then, a next PR should
target only the 770-LOC `ATOMIC_FLOW` membership surface. It should not unify
cash snapshots, debt snapshots, formulas, ratios, or compatibility fallbacks.

## Accounting and execution statement

- Accounting rules changed: **none**.
- Production code changed: **none**.
- Generated accounting outputs edited: **none**.
- Live inputs accessed: **none**.
- Publication performed: **none**.
- Accounting decision blocked: **none**; empirical cell frequency remains
  unavailable without an approved pack fixture.
