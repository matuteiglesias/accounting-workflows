---
id: notes/accounting_spine_runbook
title: "Accounting spine runbook"
sidebar_label: "Accounting spine runbook"
---

# Accounting spine runbook

Status: current authority
Last reviewed: 2026-08-25

## Official path

The supported live order is:

1. ingest / canonical ledger
2. materialization + semantic and governed cash artifacts
3. debt resolution, position/activity and treasury accountability
4. governed metric frontier + annual dashboard
5. artifact publication
6. professional presentation/drilldowns over those governed artifacts

`make run-full` is the canonical composite. There is no generic `run-marts` views stage and no parallel `metric_values` registry engine.

## Key outputs

### Canonical run root — `out/run/accounting/<RUN_ID>/`

- `ledger_canonical.csv`
- `ledger_canonical_all_status.csv`
- `monthly_flow_semantic_split.csv`
- `monthly_operating_statement.csv`
- `monthly_operating_statement_qa.csv`
- `semantic_leakage_qa.csv`
- `monthly_cash_close.csv`
- `monthly_cash_close_qa.csv`
- Stage-D diagnostic/materialized evidence where needed for audit

Materialization owns these facts. No downstream generic views layer is allowed to reclassify them.

### Debt — `out/debt_resolution/<RUN_ID>/` and canonical run root

- resolved debt evidence and status reconciliation
- debt balance source artifacts
- `monthly_debt_position.csv` + QA in the canonical run root
- `monthly_debt_activity.csv` + QA in the canonical run root
- treasury/cash-accountability artifacts in the canonical run root

Debt position is stock authority; debt activity is movement authority. Do not sum monthly stock positions into annual debt.

### Metrics — `out/metrics/<RUN_ID>/`

- `build_manifest.json`
- `metric_contract_frontier.csv`
- `frontend_metric_series.csv`
- `metrics_frontier_qa.csv`
- `frontier_source_qa.csv`
- `annual_balance_dashboard_metrics.csv`
- `annual_balance_dashboard_contract.csv`
- `annual_balance_dashboard_qa.csv`
- `annual_flow_membership.csv`
- `artifact_contracts.csv`
- `source_contract_qa.csv`

The retired `metric_registry.csv`, `metric_values.csv`, generic Q/Y statements, and `metric_views/*` are not current accounting products.

### Publication — `public/accounting/latest_<SCOPE_TAG>/`

- `manifest.json` (`accounting_public_bundle.v1`)
- `artifact_contracts.csv`
- `publish_contract_qa.csv`
- classified governed metric/debt artifacts

Publication is packaging only. It does not recalculate accounting semantics.

### Professional pack / drilldowns

A real professional pack is generated/maintained outside fixture CI. `accounting.professional.drilldown` and `accounting.professional.render_linked_digest` are the supported richer human-facing surfaces. They must reconcile to displayed values and may not invent accounting semantics.

## Fixture validation

```bash
make smoke-core
make smoke-full
make validate
```

`smoke-full` is fixture-safe and includes publication dry-run. Fixture debt and real professional-pack execution require separate evidence when affected.

## Accounting checks after structural changes

A successful process exit is insufficient. Validate at least:

- rent/OPEX/funding/draws by year and native currency;
- Household exclusion from property OPEX;
- validated cash only, with no inferred/internal fallback;
- debt closing stock versus additive activity flow;
- explicit FX measure/grain and no cross-currency sum;
- requested Box scope through publication and drilldowns;
- professional drilldown membership for every affected current table family.

## Logging

Operational logs are evidence about execution, not substitutes for CSV/JSON contracts. A green pipeline must still be reconciled for totals, scope, currency grain, status and affected drilldowns.
