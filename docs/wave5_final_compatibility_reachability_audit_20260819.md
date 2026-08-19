# Wave 5 final compatibility / reachability audit — 2026-08-19

## Purpose

PR19 is the final STOP audit for the semantic-authority migration. It does **not** treat a file or function as dead merely because its name contains `legacy`. It asks whether each compatibility path is reachable, whether it can override a governed identity, and what evidence is required before deletion.

The accounting invariant is:

> A modern cell with a governed identity must execute through that authority. Compatibility may adapt historical metadata or historical schemas, but it may not redefine a governed membership, measure, stock selector, activity selector, cash population, or derived formula.

A historical `CellSpec` may still supply derived-table path/filename identity so existing drilldown addresses remain stable. That is metadata compatibility, not matched-value authority: when a governed ID is present, actual accounting execution must bypass the legacy derived executor.

Pipeline scope remains:

`ledger canonicalization → materialization → semantic marts → debt → metrics → human reports → professional pack → drilldowns`.

No ownership, legal, governance, or family-intention conclusion is inferred from these accounting classifications.

## Evidence inspected

The audit traces the current stacked PR16–PR18 branch through:

- `accounting/professional/drilldown.py`;
- `accounting/professional/drilldown_wave4_base.py`;
- `accounting/professional/drilldown_legacy.py`;
- `accounting/professional/annual_dashboard_tables.py` and `_legacy.py`;
- `accounting/metrics/annual.py` and `annual_legacy.py`;
- `accounting/professional/table_contracts.py`;
- `accounting/professional/derived_metric_metadata.py`;
- the FlowCellSpec, debt, cash/control, and DerivedMetricSpec executors;
- the prior Wave 3 reachability inventory `diagnostics/atomic_flow_reachability_20260819.csv`.

PR19 adds:

- `diagnostics/final_compatibility_reachability_20260819.csv`;
- `diagnostics/semantic_authority_census_20260819.csv`;
- final architecture/reachability regressions;
- representative HH and FB/PM fixture packs.

## Classification result

The final inventory contains 24 explicit surfaces:

| Classification | Count | Meaning |
|---|---:|---|
| `REQUIRED_COMPATIBILITY` | 16 | Still reachable for historical schema, diagnostic/presentation surface, public API, or preserved lineage. Not a production authority override. |
| `UPSTREAM_FIX_REQUIRED` | 5 | Modern specialized surface cannot be removed safely without first defining the missing contract/grain/identity. |
| `DEAD` | 3 | Prior reachability evidence proves the legacy branch cannot be selected after modern producer enrichment. |
| `MODERN_REACHABLE_BUG` | 0 | No accidental modern governed identity was found falling through to a competing legacy execution authority. |

The machine-readable source is `diagnostics/final_compatibility_reachability_20260819.csv`.

## The three DEAD branches

The only immediate pruning candidates are legacy `CellSpec` branches for:

1. `monthly_tables_draws_by_box_amount_out`;
2. `monthly_tables_draws_by_type_amount_out`;
3. `monthly_tables_opex_by_type_amount_out`.

This is not a new judgment. Wave 3 already marked the corresponding cases `safe_to_delete=true`, and regression tests prove producer enrichment always supplies respectively:

- `flow.draws.by_box`;
- `flow.draws.by_type`;
- `flow.property_opex.by_box_category`.

PR19 records them as `DEAD`. It does **not** rewrite the 147k-character `drilldown_legacy.py` monolith merely to erase three branches while that file is still the active orchestration host. Their deletion is mechanical physical cleanup and has no semantic prerequisite.

## Why the legacy files cannot be deleted wholesale

Three legacy-named modules are still directly reachable:

- public `drilldown.py` imports `drilldown_wave4_base`, which imports and patches `drilldown_legacy` and reuses its index/detail/HTML/QA orchestration;
- `accounting/metrics/annual.py` delegates pre-PR15B non-cash annual construction to `annual_legacy`;
- `accounting/professional/annual_dashboard_tables.py` preserves non-cash companion builders from `annual_dashboard_tables_legacy` while overriding governed cash.

Therefore file-name based deletion would be wrong. Future cleanup may physically re-home these functions, but that is a mechanical architecture task, not a license to change accounting semantics.

## Required compatibility boundary

Compatibility is now conceptually lateral rather than authoritative:

```text
legacy / presentation metadata
        |
        v
CompatibilityAdapter
        |
        +--> governed identity, only when provable
        |
        +--> explicit compatibility path otherwise

stable governed identity
        |
        v
contract registry
        |
        v
specialized executor
        |
        v
DrilldownResult / professional evidence
```

Human labels remain in metadata adapters only. `DerivedMetricSpec` and its executor are label-free. Compatibility metric inference cannot opt a row into FlowCellSpec execution unless stable producer metadata independently proves the ID.

PR19 explicitly monkeypatches the legacy **execution** fallback to raise and proves governed atomic/derived cells still reconcile. Thus legacy path metadata cannot become a competing accounting authority.

## Explicit upstream-fix families

Five inventory entries are not classified as bugs because the deferral is explicit and contract insufficiency is known.

### Funding support dimensions

`flow.funding_contribution.by_actor`, `.by_channel`, `.by_cash_effect`, and `.by_target_box` can represent direct-obligation and debt-linked support broader than plain atomic `funding_contribution`. Removing compatibility requires a dedicated support contract, not a looser FlowCellSpec filter.

### FX atomic professional grain

Known FX atomic measures are governed by `semantic_measure_registry_v1`, but some professional total-by-currency rows do not carry the `Box` grain required by the existing FlowCellSpec. Removing the compatibility route requires resolving that grain contract explicitly.

### `TR.FX.NET`

Net FX is specialized treasury composition, not one atomic semantic measure and not a generic scalar DerivedMetricSpec formula. It needs a dedicated specialized executor/contract if the legacy route is to disappear.

### `ID.DEBT.NET_PM_POSITION`

Net PM position depends on debtor/creditor identity and direction. It cannot be reconstructed by pretending one DebtPositionSpec scalar is sufficient.

### bridge `net_flow`

FB/PM/HH and annual cash bridges combine heterogeneous governed flow rows. This is a bridge identity problem, not a generic ratio/subtraction formula.

These deferrals are visible and bounded. None is allowed to change the authority of the already-governed core concepts.

## Representative-pack validation

No confidential family pack or generated private report is committed or read by CI. PR19 therefore uses deterministic synthetic packs that reproduce the two scopes relevant to the migration.

### HH fixture

The fixture contains:

- ARS Property Management OPEX = 30;
- ARS Household expense = 999.

The modern `property_opex_true` professional cell reconciles to **30**, with `governed_atomic_flow` lineage, and its detail does not contain Household. This is an adversarial semantic-leakage test, not merely a pipeline-success test.

### FB/PM fixture

The fixture contains native ARS and USD rent/OPEX flows plus annual scalar authorities. It verifies:

- monthly rent/OPEX drilldowns reconcile separately in ARS and USD;
- no cross-currency aggregation occurs;
- annual operating margin and OPEX/rent use governed component authorities;
- coverage keeps source authority with formula reconciliation;
- monthly and annual routes expose governed lineage types.

These fixtures are representative regression packs, not claims about current family totals.

## Final semantic-authority census

`diagnostics/semantic_authority_census_20260819.csv` records one production authority for each core concept:

| Concept | Authority |
|---|---|
| Property OPEX membership | FlowCellSpec + semantic measure registry |
| Funding contribution membership | FlowCellSpec + semantic measure registry |
| Family draws membership | FlowCellSpec + semantic measure registry |
| FX atomic measure/direction | semantic measure registry |
| Debt position | DebtPositionSpec |
| Debt activity | DebtActivitySpec |
| Validated cash position | ValidatedCashPositionSpec |
| Inferred box control | InferredBoxControlSpec |
| Derived formula definition | DerivedMetricSpec |

Every row has `authority_count=1` and `compatibility_can_override=no`.

The FX row is marked `GOVERNED_WITH_GRAIN_DEFERRED`: atomic measure direction has one authority, while professional total-by-currency adaptation remains explicit follow-up work.

## Wave 5 DONE criteria

PR19 closes only if the regression suite demonstrates all of the following together:

- atomic flows reconcile;
- debt positions remain stock snapshots and reconcile;
- debt activity remains flow activity and reconciles;
- validated cash and inferred control remain disjoint;
- derived formulas reconcile component-by-component;
- native currencies remain separate;
- Household cannot leak back into property OPEX through the professional layer;
- missing derived inputs do not become invented zeroes;
- a governed identity cannot call its legacy **execution** fallback;
- professional drilldowns expose governed lineage type;
- all remaining compatibility routes are explicitly classified;
- only prior-evidence `DEAD` branches are candidates for immediate deletion.

CI success alone is not sufficient; the representative pack and semantic-authority assertions are part of the gate.

Final validation on the stacked branch: **281 passed, 1 pre-existing warning**. The warning is the previously characterized `pd.to_datetime` warning in the legacy debt-position helper.

## Physical architecture after closure

The desired authority boundary is now materially present:

```text
semantic_measure_registry
        |
        +---- FlowCellSpec ----------------> governed atomic-flow executor
        |
        +---- DebtPositionSpec ------------> debt-position executor
        |
        +---- DebtActivitySpec ------------> debt-activity executor
        |
        +---- ValidatedCashPositionSpec ---> cash-position executor
        |
        +---- InferredBoxControlSpec ------> control selector / period delta
        |
        +---- DerivedMetricSpec -----------> derived formula executor
                                                |
                                                v
                                         DrilldownResult
                                                |
                                      professional evidence
```

The remaining compatibility monolith still hosts orchestration and historical adapters, but it no longer has permission to override a stable governed execution identity.

## Recommended post-Wave-5 cleanup

After the stacked PR is merged, cleanup should be mechanical and separately reviewable:

1. prune the three proven-dead legacy CellSpec branches;
2. optionally squash/re-home facade/orchestration code without changing output contracts;
3. deprecate old metric aliases only after usage evidence from notebooks/external consumers;
4. tackle the five `UPSTREAM_FIX_REQUIRED` specialized families only as separate contracts, not as opportunistic fallback deletion.

No new generic semantic migration is required for the governed core.
