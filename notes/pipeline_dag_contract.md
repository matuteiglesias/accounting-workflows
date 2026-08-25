# Makefile pipeline DAG contract

Status: current authority
Last reviewed: 2026-08-25

This document defines orchestration only. Accounting classification, cash selection, debt position/activity, annual aggregation and FX membership remain governed by their dedicated contracts.

## Intended DAG

```text
doctor
validate
smoke-core
smoke-full

run-canonical                         [live]
  run-ingest -> run-materialize

run-debt RUN_ID=<exact-run-id>        [replayable]
  debt.resolve -> debt.balance_views
               -> monthly_debt_position / monthly_debt_activity
               -> treasury accountability

run-metrics RUN_ID=<exact-run-id>     [replayable]
  accounting.metrics.build
    -> metric frontier + annual dashboard + source contracts

run-reports RUN_ID=<exact-run-id>     [replayable]
  governed metrics + treasury
    -> annual_management HTML/PDF
    -> treasury_accountability HTML/PDF
    -> report_catalog.json

run-full                              [live ordered composite]
  run-canonical
    -> run-debt
    -> run-metrics
    -> run-reports
    -> atomic latest alignment
    -> publish-latest + publish-reports
    -> release-check

professional-drilldowns / professional-linked-digest
  existing governed professional pack -> traceable presentation/drilldowns
```

There is no `run-marts` generic views stage, `run-debt-views` sub-pipeline, `run-dashboard` assertion alias, `views_sanity.json` gate, generic `metric_values` registry engine, or standalone human-report accounting stage.

## Target contracts

| Target | Stage | Required inputs | Produced artifacts | Main invariant | Live env required |
|---|---|---|---|---|---|
| `doctor` | static | source tree | compile status | command modules import/compile | no |
| `validate` | static/contracts/tests | source tree | validation status | declared contracts and regressions pass | no |
| `smoke-core` | fixture canonical | ledger fixture | canonical/materialized semantic + cash artifacts | offline canonical path works | no |
| `smoke-full` | fixture product | smoke-core | validate + publish dry-run | fixture-safe product gates | no |
| `run-ingest` | live source | Google Sheet credentials | canonical ledger | source access is explicit | yes |
| `run-materialize` | materialization | existing exact-run canonical ledger | governed semantic + cash artifacts | no live ingest side effect | no |
| `run-canonical` | live canonical | Google Sheet credentials | canonical ledger + governed materialization | one run identity across ingest/materialize | yes |
| `run-debt` | debt + treasury | existing exact-run all-status canonical ledger | debt resolution + position/activity + treasury | stock and movement remain distinct | no |
| `run-metrics` | governed metrics | existing exact run | frontier, annual dashboard, flow membership, contracts | canonical sources only | no |
| `run-reports` | report product | exact-run treasury + metrics | HTML/PDF/catalog/manifests | presentation only; same run identity | no |
| `publish-latest` | packaging | aligned latest metrics/debt | public machine bundle | no computation or legacy metric revival | no |
| `publish-reports` | document packaging | aligned latest report bundle | public report bundle | documents only | no |
| `release-check` | release | public bundle | readiness status | cash/debt/currency/publication checks | no |
| `run-full` | full live | live env | aligned/published machine + report bundles | ordered canonical -> debt -> metrics -> reports -> publish | yes |

## Exact-run identity contract

`RUN_ID` is the exact directory identity shared by:

```text
out/run/accounting/<RUN_ID>/
out/debt_resolution/<RUN_ID>/
out/metrics/<RUN_ID>/
out/reports/<RUN_ID>/
```

When creating a new live run, the Makefile derives `RUN_ID` from `RUN_STAMP` and the canonical scope tag. For replay, callers pass `RUN_ID` directly. Downstream stage targets must not manufacture a second run identity or pull live inputs.

## Latest pointer contract

After a complete run, `_update_latest` preflights and aligns one run identity across:

```text
out/run/accounting/latest_<SCOPE_TAG>
out/debt_resolution/latest_<SCOPE_TAG>
out/metrics/latest_<SCOPE_TAG>
out/reports/latest_<SCOPE_TAG>
```

No light/core partial-latest target belongs to the supported command surface. A focused exact-run stage does not move latest pointers.

## Change rule

A Makefile change is not validated merely because commands execute. Any change that affects an accounting layer must also reconcile the relevant totals/scope/currency and professional drilldowns. Structural pruning may retire obsolete compatibility outputs when their underlying governed facts remain available.
