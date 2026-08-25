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

run-canonical
  run-ingest -> run-materialize

run-debt-views
  run-canonical -> debt.resolve -> debt.balance_views
                -> monthly_debt_position / monthly_debt_activity
                -> treasury accountability

metrics-from-run / run-metrics
  existing RUN_OUT
    -> accounting.metrics.build
    -> metric frontier + annual dashboard + source contracts

run-dashboard
  asserts annual dashboard metrics, contract, and QA outputs

publish-latest
  scope-qualified latest debt/metrics -> public/accounting/latest_<SCOPE_TAG>

release-check
  public bundle -> dashboard-readiness validation

professional-drilldowns / professional-linked-digest
  existing governed professional pack -> traceable presentation/drilldowns
```

There is no `run-marts` generic views stage, `views_sanity.json` gate, generic `metric_values` registry engine, or standalone human-report accounting stage.

## Target contracts

| Target | Stage | Required inputs | Produced artifacts | Main invariant | Live env required |
|---|---|---|---|---|---|
| `doctor` | static | source tree | compile status | command modules import/compile | no |
| `validate` | static/contracts/tests | source tree | validation status | declared contracts and regressions pass | no |
| `smoke-core` | fixture canonical | ledger fixture | canonical/materialized semantic + cash artifacts | offline canonical path works | no |
| `smoke-full` | fixture product | smoke-core | validate + publish dry-run | fixture-safe product gates | no |
| `run-canonical` | live canonical | Google Sheet credentials | canonical ledger + governed materialization | no downstream reclassification stage | yes |
| `run-debt-views` | debt | canonical all-status ledger | debt resolution + position/activity + treasury | stock and movement remain distinct | yes when upstream live |
| `metrics-from-run` | governed metrics | existing canonical run | frontier, annual dashboard, flow membership, contracts | canonical sources only | no |
| `run-metrics-live` | live metrics | live canonical/debt path | governed metric artifacts | same authority as existing-run metrics | yes |
| `run-dashboard` | dashboard gate | metrics directory | assertions only | annual artifacts exist | no |
| `publish-latest` | packaging | latest metrics/debt | public bundle | no computation or legacy metric revival | no |
| `release-check` | release | public bundle | readiness status | cash/debt/currency/publication checks | no |
| `run-full` | full live | live env | public bundle | canonical -> debt -> governed metrics -> publish | yes |

## Latest pointer contract

After a complete run, `_update_latest` aligns one run identity across:

```text
out/run/accounting/latest_<SCOPE_TAG>
out/debt_resolution/latest_<SCOPE_TAG>
out/metrics/latest_<SCOPE_TAG>
```

The primary scope may also maintain the documented compatibility `latest` pointer. No `human_reports/latest` or views-stage latest pointer belongs to the current spine.

## Change rule

A Makefile change is not validated merely because commands execute. Any change that affects an accounting layer must also reconcile the relevant totals/scope/currency and professional drilldowns. Structural pruning may retire obsolete compatibility outputs when their underlying governed facts remain available.
