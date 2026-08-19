# Wave 4 closure audit — governed position and activity lineage

Date: 2026-08-19

This audit is the STOP gate after PR15B. It asks whether the position/activity authorities introduced in PR12–PR15B are now real production boundaries and whether cash headline populations reconcile across layers.

## Final validation

GitHub Actions `accounting-ci` on the complete PR15B implementation passed:

```text
246 passed
1 pre-existing warning
compile, contract, and regression validation ok
```

The warning is the already-characterized `pd.to_datetime` warning in the legacy debt-position helper. PR15B does not change that path.

## Gate matrix

| gate | target | evidence | final status |
| --- | --- | --- | --- |
| FlowCellSpec production consumer | governed atomic-flow executor active | public facade + preserved governed flow base | **PASS** |
| DebtPositionSpec production consumer | snapshot executor active | `accounting/professional/debt_position_executor.py` | **PASS** |
| DebtActivitySpec production consumer | sum-flow executor active | `accounting/professional/debt_activity_executor.py` | **PASS** |
| ValidatedCashPositionSpec production consumer | shared runtime selector used by metrics + professional | `accounting/cash_authority.py` and PR15B facades | **PASS** |
| InferredBoxControlSpec governed primitive | independent selector exists and is not headline eligible | `select_inferred_box_control_period` | **PASS** |
| monthly cash population == annual cash population | same account-snapshot primitive | adversarial fixture = 100 monthly / 100 annual | **PASS** |
| validated + inferred never additive | inferred excluded from headline | adversarial fixture + selector tests | **PASS** |
| internal balance never headline cash | internal rows excluded but retained as evidence | adversarial fixture + drilldown sections | **PASS** |
| stocks never annual-summed | annual cash/debt use closing snapshot rules | cash/debt regressions | **PASS** |
| debt activity never snapshot-selected | activity executor has no as-of selection | PR14 architecture regression | **PASS** |
| diagnostic box level unchanged in PR15B | remains derived-formula legacy path | fixture remains 40 | **PASS / DEFERRED TO WAVE 5** |

## Accounting/reporting invariant

A successful pipeline run is not sufficient. For modern cash artifacts, all affected layers now identify the same source population:

```text
validated account snapshots
!= inferred box control
!= internal party balances
```

`BS.CASH.TOTAL`, `BS.CASH.CLOSE.BOX`, the annual cash companion, and professional cash drilldowns reconcile against the same selected account snapshots.

## Source trace

The canonical cash mart remains `monthly_cash_close.csv`. PR15B does not alter its three populations. Consumption is centralized in `accounting.cash_authority`:

- headline cash: `cash.position.validated`;
- inferred control: `cash.control.inferred_box_motor`;
- internal balance: excluded evidence only.

No upstream source row is reclassified or deleted.

## Before/after effect

The versioned synthetic adversarial case is recorded in `diagnostics/wave4_cash_authority_change_20260819.csv`.

Deliberate reporting change:

```text
monthly cash: 200 -> 100
annual cash:  250 -> 100
```

The difference is entirely population selection:

- inferred 100 is no longer added to monthly cash;
- inferred 100 and internal 50 are no longer added to annual cash;
- validated Bank A 70 + Bank B 30 remain.

The same fixture proves the inferred 100 is still independently available as governed control evidence.

## Layer reconciliation result

For the modern fixture, all migrated headline layers reconcile:

```text
cash_authority selector
= metrics frontier monthly BS.CASH
= annual dashboard BS.CASH
= annual companion cash close
= professional monthly drilldown
= professional annual drilldown
= 100 ARS
```

The selector also fails closed on duplicate latest account/as-of rows and on any candidate account lacking a valid as-of snapshot.

## Semantic leakage checks

The professional cash drilldown exposes selected validated accounts plus excluded inferred-control and internal-balance rows. Missing validated evidence remains unavailable; it is never replaced by zero or inferred control.

`monthly_tables_diagnostic_box_level_matrix` is not a cash-position headline and is deliberately deferred to Wave 5. PR15B proves its characterized fixture remains 40, so cash migration did not silently change the period-delta formula.

## Annual/multiple-account correction

PR15B also closes a distinct downstream leakage risk: prior annual cash code could select one frontend-safe row per Box/year and thereby undercount when multiple validated accounts coexisted. Annual cash now uses the same account-level selector as monthly cash and sums selected account snapshots only.

## Migration scaffolding

Pre-change implementations remain copied byte-for-byte as compatibility modules. This audit does **not** treat smaller public facades as net LOC reduction. Compatibility code remains reachable for historical schemas and untouched non-cash behavior. Physical pruning is deferred to the final migration audit after Wave 5.

## Closure decision

All Wave 4 gates pass on the synthetic adversarial fixture and the full repository regression suite.

**Wave 4 status: DONE.**

Safe next bounded action: Wave 5 PR16 — characterize derived/formula authority without changing production formulas.
