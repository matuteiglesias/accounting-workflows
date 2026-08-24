---
id: notes/accounting_spine_runbook
title: "Accounting spine runbook"
sidebar_label: "Accounting spine runbook"
---

# Accounting spine runbook

Status: current authority
Last reviewed: 2026-08-24

## Official path

The supported live order is:

1. ingest / canonical ledger
2. materialization + semantic marts
3. debt resolution/views
4. governed metrics + annual dashboard
5. artifact publication

`make run-full` is the canonical composite. The retired `human_reports` producer is not a stage.

## Key outputs

### Canonical run root — `out/run/accounting/<RUN_ID>/`
- `ledger_canonical.csv`
- `ledger_canonical_all_status.csv`
- semantic/materialized monthly artifacts and QA

### Debt — `out/debt_resolution/<RUN_ID>/`
- resolved debt evidence and status reconciliation
- governed debt stock/activity inputs consumed downstream

### Metrics — `out/metrics/<RUN_ID>/`
- `metric_registry.csv`
- `metric_values.csv`
- `validation_report.csv`
- `build_manifest.json`
- `metric_contract_frontier.csv`
- `frontend_metric_series.csv`
- `annual_balance_dashboard_metrics.csv`
- `annual_balance_dashboard_contract.csv`
- `annual_balance_dashboard_qa.csv`
- governed metric views/drilldowns

### Publication — `public/accounting/latest_<SCOPE_TAG>/`
- `manifest.json` (`accounting_public_bundle.v1`)
- `artifact_contracts.csv`
- `publish_contract_qa.csv`
- classified governed metric/debt artifacts

Publication is packaging only. It does not require `human_reports` and does not own a web application.

### Professional pack / drilldowns
A real professional pack is generated/maintained outside fixture CI. `accounting.professional.drilldown` and `accounting.professional.render_linked_digest` are the supported richer human-facing surfaces. They must reconcile to displayed values and may not invent accounting semantics.

## Fixture validation

```bash
make smoke-core
make smoke-full
make validate
```

`smoke-full` is fixture-safe and includes publication dry-run. As frozen in the Phase-0 baseline, fixture debt and real professional-pack execution require separate evidence when affected.

## Logging
Operational logs are evidence about execution, not substitutes for CSV/JSON contracts. A successful process exit is insufficient: validate totals, scope, currency grain, status, and affected drilldowns.
