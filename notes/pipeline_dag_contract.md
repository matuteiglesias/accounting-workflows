# Makefile pipeline DAG contract

This document defines the intended Makefile control-plane surface. It documents orchestration only; it does not change accounting formulas, semantic rules, cash/debt logic, or metric definitions.

## Intended DAG

```text
doctor
validate
smoke-core
smoke-full

run-canonical
  run-ingest -> run-materialize -> run-marts -> run-debt -> run-debt-views

metrics-from-run / run-metrics
  existing RUN_OUT -> accounting.metrics.build -> frontier + annual dashboard artifacts

run-dashboard
  asserts annual dashboard metrics, contract, and QA outputs

run-human
  metrics + canonical/report-safe sources -> human report

publish-latest
  latest producer outputs -> public/accounting/latest package

release-check
  public/accounting/latest -> dashboard readiness validation
```

## Target contracts

| Target | Stage | Command | Required inputs | Produced artifacts | QA artifacts | Contract assumptions | Live env required |
|---|---|---|---|---|---|---|---|
| `doctor` | environment/static | `python -m py_compile ...` | Python and repo files | compile status | none | command modules import/compile | no |
| `validate` | static/contracts | `doctor`, `make help`, `scripts/check_contracts.py` | source tree | validation status | temp artifact/source contract QA | emitted contracts match declared vocabularies | no |
| `smoke-core` | fixture core | `smoke-ingest` + `accounting.stage_d.materialize` | `fixtures/ledger_fixture.csv` | smoke ingest, Stage D semantic/cash artifacts | materialize checks, `semantic/cash artifact presence checks` | fixture path exercises offline core | no |
| `smoke-full` | fixture product | `smoke-core validate publish --dry-run` | fixture core | dry-run publish manifest on stdout | validate checks | full fixture debt/human publish is a documented follow-up | no |
| `run-canonical` | live canonical | `run-debt-views` | Google Sheet credentials and sheet URL | run root, marts, debt wrappers | ingest/materialize/view/debt checks | canonical backend stops before metrics/human/publish | yes |
| `metrics-from-run` | metrics | `_run_metrics_action` | existing `RUN_OUT` with canonical artifacts | metric registry, metric values, frontier, annual dashboard outputs | validation report, source contract QA | consumes existing canonical artifacts only | no |
| `run-metrics-live` | metrics orchestration | `run-debt-views _run_metrics_action` | Google Sheet credentials | metrics outputs | metrics QA | live orchestration retained separately | yes |
| `run-dashboard` | dashboard | file assertions | metrics dir | annual dashboard metrics/contract/QA | annual dashboard QA | dashboard outputs are produced by metrics build | no |
| `run-human` | human | `accounting.human.document` | `RUN_OUT`, metrics dir | human HTML/report manifest | story manifest | does not recompute semantics | no |
| `publish-latest` | publish | `accounting.publish.latest` | latest symlinks for run/debt/metrics/human | public bundle | publish contract QA | packaging only, not release readiness | no |
| `release-check` | release | `scripts/check_release.py` | `public/accounting/latest` | readiness status | printed checks | public bundle is dashboard-ready | no |
| `run-full` | full live | `run-canonical -> run-metrics -> run-dashboard -> run-human -> publish-latest -> release-check` | live env | public bundle | release-check | full live release path | yes |

## Latest symlink contract

After human report generation, `_update_latest` updates all producer latest pointers consistently:

```text
out/run/accounting/latest
out/debt_resolution/latest
out/metrics/latest
out/human_reports/latest
```
