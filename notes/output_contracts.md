---
id: notes/output_contracts
title: "Accounting Backend Output Contracts"
sidebar_label: "Accounting Backend Output Contracts"
---

# Accounting Backend Output Contracts

Status: current authority
Last reviewed: 2026-08-25

## Purpose

Downstream consumers should depend on governed accounting artifacts rather than arbitrary intermediates, legacy report files, or a second metric-classification engine.

## Contract summary

| Contract | Authority | Typical path | Consumer role |
|---|---|---|---|
| canonical ledger | `accounting.ledger.ingest` | `out/run/accounting/<run_id>/ledger_canonical.csv` | transaction evidence |
| semantic monthly facts | `accounting.marts.semantic` via materialization | `monthly_flow_semantic_split.csv`, `monthly_operating_statement.csv` | operating/funding/distribution/FX truth |
| governed cash close | cash mart/authority via materialization | `monthly_cash_close.csv` | validated cash only |
| debt position/activity | `accounting.marts.debt` + debt authorities | `monthly_debt_position.csv`, `monthly_debt_activity.csv` | debt stock and movement truth |
| metric frontier | `accounting.metrics.frontier` | `out/metrics/<run_id>/metric_contract_frontier.csv`, `frontend_metric_series.csv` | current monthly frontend contract |
| annual dashboard | `accounting.metrics.annual` | `annual_balance_dashboard_metrics.csv`, `annual_balance_dashboard_contract.csv` | annual governed facts |
| annual flow membership | annual membership contract | `annual_flow_membership.csv` | drilldown/lineage evidence |
| public bundle | `accounting.publish.latest` | `public/accounting/latest_<scope>/` | packaged downstream handoff |
| professional drilldowns | `accounting.professional.drilldown` | professional/drilldown output roots | human traceability |

The retired `metric_values.csv`, `metric_registry.csv`, generic Q/Y statements, generic metric views and generic marts/views outputs are not current contracts.

## Canonical ledger

Producer: `accounting.ledger.ingest`.

Required accounting fields include stable transaction identity, date, amount, native `Currency`, parties, status, `Box`, source provenance and the ledger classification inputs. The canonical ledger is evidence, not permission for downstream report code to invent new classifications.

Validation expectations:

- required accounting fields exist;
- dates and amounts parse;
- native currency is explicit;
- source provenance is retained;
- requested Box scope is preserved;
- anomalies are visible rather than silently coerced.

## Semantic monthly facts

Primary artifacts:

```text
monthly_flow_semantic_split.csv
monthly_operating_statement.csv
monthly_operating_statement_qa.csv
semantic_leakage_qa.csv
classification_audit.csv
classification_audit_summary.csv
```

These artifacts own the current operating/funding/distribution/treasury semantic split. Stage-D `per_*`, box-balance and daily-position artifacts may remain useful diagnostic/materialized evidence, but they are not frontend truth and must not be used to reclassify flows in reports.

Core invariants:

- operating revenue excludes family funding;
- property OPEX excludes Household/personal/distribution semantics;
- funding remains distinct from operating revenue;
- personal draws/distributions remain distinct from OPEX;
- treasury FX remains outside operating income/funding;
- native currencies are never silently summed together.

## Governed cash close

Primary artifacts:

```text
monthly_cash_close.csv
monthly_cash_close_qa.csv
```

Cash reporting uses validated account snapshots only when the governed cash schema supports them. Internal party balances and inferred box-control balances are separate evidence populations and are never a headline fallback.

Annual cash is a closing stock: select the last governed period in the year and apply the same validated-account snapshot primitive. Never sum monthly cash positions.

## Debt contracts

Resolution evidence under `out/debt_resolution/<run_id>/` includes:

```text
debt_open_items.csv
debt_allocations.csv
debt_repayment_events.csv
debt_resolution_timeline.csv
debt_status_reconciliation.csv
```

Canonical downstream debt facts in the accounting run root include:

```text
monthly_debt_position.csv
monthly_debt_position_qa.csv
monthly_debt_activity.csv
monthly_debt_activity_qa.csv
```

`monthly_debt_position` is stock authority. `monthly_debt_activity` is additive movement authority. Annual debt stock selects the latest valid closing position; it is not a sum of monthly positions. Invalid or incomplete position grain fails closed under the debt position authority.

## Metric frontier

Producer: `accounting.metrics.frontier`.

Primary outputs:

```text
metric_contract_frontier.csv
frontend_metric_series.csv
metrics_frontier_qa.csv
frontier_source_qa.csv
```

The frontier consumes only the governed monthly semantic/cash/debt artifacts. It does not load a generic metric registry or `metric_values.csv` compatibility universe.

Expected checks include:

- current metric IDs are present or explicitly unavailable;
- all money rows carry native `Currency`;
- only canonical frontier sources are used;
- no legacy-only series are injected;
- validated cash has no inferred/internal fallback;
- FX metrics remain explicit treasury facts.

## Annual dashboard

Producer: `accounting.metrics.annual`.

Primary outputs:

```text
annual_balance_dashboard_metrics.csv
annual_balance_dashboard_contract.csv
annual_balance_dashboard_qa.csv
annual_flow_membership.csv
```

Flows aggregate monthly governed facts by year and native currency. Stocks select governed closing positions. Ratios are computed from annual aggregates rather than averaged monthly ratios unless an explicit contract says otherwise.

The annual dashboard exposes current operating, property OPEX, funding/support, distribution, coverage, validated cash, debt stock/activity, data-quality and treasury FX families. Historical mixed metrics such as funding-inclusive `IS.INCOME.TOTAL` are not targets for parity.

## Publication contract

Producer: `accounting.publish.latest`.

The public bundle is a packaging boundary, not a calculation layer. Current classes are:

- `public_contract` — explicit frontend contracts;
- `canonical_dashboard` — governed/report-safe facts;
- `internal_diagnostic` — internal evidence and QA;
- `unsafe_for_frontend` — evidence that must not be displayed as dashboard fact.

Retired generic metric statements are not published. Release QA checks that they remain absent.

## Professional presentation / drilldowns

Professional tables and linked drilldowns are downstream projections of governed facts. They may format, select and explain; they may not silently decide accounting membership.

Required invariants when a backend change touches their source families:

- displayed value reconciles to drilldown membership;
- explicit currency and Box grain are preserved;
- FX cells use explicit measure/grain authority;
- validated cash shows the selected validated population and does not add inferred/internal evidence;
- debt stock and debt activity remain distinct;
- unsupported or ambiguous rows fail closed rather than defaulting to another measure.

## Compatibility rule

Historical audits and old file names may remain in repository documentation as evidence of migration history. They do not create a current runtime contract. A removed convenience output should only be recreated if a real consumer needs it and the projection is best owned at the report layer; it is not a reason to restore a generic backend views or metric engine.
