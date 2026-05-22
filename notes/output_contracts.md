---
id: notes/output_contracts
title: "Accounting Backend Output Contracts"
sidebar_label: "Accounting Backend Output Contracts"
---

# Accounting Backend Output Contracts

Status: authority draft
Last reviewed: 2026-05-10

## Purpose

This document names the stable and emerging output contracts in the accounting backend. Downstream consumers should depend on these artifacts instead of arbitrary intermediate files.

## Stability vocabulary

| Stability | Meaning |
|---|---|
| stable | Safe for downstream modules to depend on |
| current | Used today but still evolving |
| experimental | Future-facing or incomplete |
| legacy | Historical; avoid new dependencies |
| unknown | Needs runtime verification |

## Contract summary

| Contract | Level | Stability | Producer | Typical path |
|---|---:|---|---|---|
| `ledger_canonical` | 1 | stable candidate | `accounting.ledger.ingest` | `out/run/accounting/<run_id>/ledger_canonical.csv` |
| `stage_d_materialized_views` | 2 | current/stable candidate | `accounting.stage_d.materialize` | `out/run/accounting/<run_id>/*.csv` |
| `metric_values` | 3 | stable | `accounting.metrics.build` | `out/metrics/<run_id>/metric_values.csv` |
| `metric_registry` | 3 | stable | `accounting.metrics.registry` via `accounting.metrics.build` | `out/metrics/<run_id>/metric_registry.csv` |
| `validation_report` | 3 | stable candidate | `accounting.metrics.validate` via `accounting.metrics.build` | `out/metrics/<run_id>/validation_report.csv` |
| `metric_views` | 3 | current | `accounting.metrics.views` via `accounting.metrics.build` | `out/metrics/<run_id>/metric_views/*` |
| `metric_drilldown` | 3/4 | current | `accounting.metrics.drilldown` via `accounting.metrics.build` | `out/metrics/<run_id>/metric_drilldown/*` |
| `debt_open_items` | 3 | current/canonical | `accounting.debt.resolve` | `out/debt_resolution/<run_id>/debt_open_items.csv` |
| `debt_allocations` | 3 | current/canonical | `accounting.debt.resolve` | `out/debt_resolution/<run_id>/debt_allocations.csv` |
| `debt_repayment_events` | 3 | current/canonical | `accounting.debt.resolve` | `out/debt_resolution/<run_id>/debt_repayment_events.csv` |
| `debt_balance_views` | 3 | current | `accounting.debt.balance_views` | `out/debt_resolution/<run_id>/debt_balance_*.csv` |
| `human_table_specs` | 4 | current/stable seam | `accounting.human.tables` | generated in human/front report outputs |
| `human_balance_report` | 4 | current/canonical | `accounting.human.document` | `out/human_reports/<run_id>/balance_human_v2/*` |
| `frontend_snapshot_manifest` | 5 | current | `accounting.publish.latest` | `public/accounting/latest/manifest.json` |

## `ledger_canonical`

Producer: `accounting.ledger.ingest`.

Consumers: `accounting.stage_d.materialize`, debt resolution, metric/drilldown builders, report evidence loaders.

Required columns:

```text
tx_id
Date
amount
Currency
payer
receiver
Flujo
Tipo
status
Box
source_file
source_row
ingest_ts
notes
```

Common optional columns:

```text
amount_cents
base_amount
Detalle
Lugar
medio
tag
```

Validation expectations:

- Required accounting fields exist before materialization.
- `Date` parses as date/datetime.
- `amount` parses as numeric.
- `Currency` is non-empty.
- `tx_id` is stable and non-empty when required.
- Ingest anomalies are visible and not silently ignored.

## `stage_d_materialized_views`

Producer: `accounting.stage_d.materialize`.

Consumers: `accounting.views`, metrics, human tables, report factories.

Required representative artifacts:

```text
per_flow_time_long.freq=*.csv
per_party_time_long.freq=*.csv
daily_cash_position.csv
box_balance_time_long.freq=*.csv
box_flow_balance_time_long.freq=*.csv
```

Validation expectations:

- Required Stage D files exist for the requested frequency.
- Period/date columns parse cleanly.
- Amount columns remain numeric and currency-aware.
- View/report layers treat these as authoritative over legacy report files.

## `metric_values`

Producer: `accounting.metrics.build`.

Consumers: human tables, report factories, validation, statement views, and frontend surfaces.

Required columns:

```text
metric_id
period_grain
period
currency
value
run_id
as_of_date
source_layer
build_status
build_detail
```

Validation expectations:

- Schema is normalized by the metrics I/O layer.
- Duplicate keys are checked across metric, period, currency, run, and as-of dimensions.
- `metric_id` exists in `metric_registry.csv`.
- `value` is numeric.
- `build_status` defaults to `ok` when not set.

## `metric_registry`

Producer: `accounting.metrics.registry` via `accounting.metrics.build`.

Consumers: metric validation, labels, ordering, grouping, statement views, reports, and frontend display.

Required columns:

```text
metric_id
statement
section
label
agg_rule
is_leaf
source_layer
builder_key
parent_metric_id
display_code
sort_key
currency_mode
status
notes
```

Validation expectations:

- `metric_id` is unique.
- Active leaf metrics have a `builder_key`.
- Derived metrics have enough metadata to explain construction.
- `status` defaults to active.
- `currency_mode` defaults to by-currency behavior unless specified otherwise.

## `validation_report`

Producer: `accounting.metrics.validate` via `accounting.metrics.build`.

Consumers: human report, QA, support checks, and future release gates.

Required columns:

```text
level
check_name
message
n_rows
```

Validation expectations:

- The file may be empty or have zero rows when clean.
- `level` distinguishes warnings and errors.
- Error rows should fail or degrade the build unless explicitly waived.

## Debt contracts

Producer: `accounting.debt.resolve`.

Consumers: `accounting.debt.balance_views`, metrics, human tables, reports, and frontend snapshot packaging.

Required primary artifacts:

```text
debt_open_items.csv
debt_allocations.csv
debt_repayment_events.csv
debt_resolution_timeline.csv
debt_status_reconciliation.csv
```

Important `debt_open_items.csv` columns:

```text
debt_id
source_tx_id
opened_at
debtor
creditor
currency
item_type
original_amount
open_amount
detalle
lugar
issuer
ledger_status
engine_status
closed_at
```

Validation expectations:

- `opened_at` parses as a date.
- Parties, currency, and item type are non-empty.
- Amounts parse as numeric.
- `engine_status` is explicit.
- Reconciliation output is inspected when ledger status and engine status diverge.

## `debt_balance_views`

Producer: `accounting.debt.balance_views`.

Consumers: metrics and human reports.

Required artifacts:

```text
debt_balance_daily.csv
debt_balance_monthly.csv
debt_balance_quarterly.csv
debt_balance_yearly.csv
```

Validation expectations:

- Inputs come from `debt_open_items.csv` or an equivalent resolved debt contract.
- Period/date fields parse cleanly.
- Open balances remain numeric and currency-aware.

## `human_table_specs`

Producer: `accounting.human.tables`.

Consumers: current human document factory and future front report factory.

Expected structure:

```text
slug
title
builder_key
group
notes
enabled_by_default
```

Validation expectations:

- Every generated table has a corresponding spec.
- Empty tables are hidden, marked partial, or included with an explicit note.
- Report factories do not invent new accounting semantics outside table builders/contracts.
- Table slugs remain stable because report blocks depend on them.

## `human_balance_report`

Producer: `accounting.human.document`.

Consumers: humans and `accounting.publish.latest`.

Required representative artifacts:

```text
balance_humano_v2.html
story_manifest.json
report.css
```

Validation expectations:

- The report is built from metric, debt, drilldown, and human-table contracts.
- The story manifest identifies report items and provenance.
- Missing/partial upstream data is visible rather than silently hidden.

## `frontend_snapshot_manifest`

Producer: `accounting.publish.latest`.

Consumer: accounting viewer/static frontend.

Required representative location:

```text
public/accounting/latest/manifest.json
```

Expected v1 manifest fields include:

```text
schema_name
built_at
source_run_id
status
source_paths
files
metrics
debt
reports
```

Compatibility fields such as `surface_id`, `published_at_utc`, `publish_mode`, `run_id`, `as_of_date`, `months`, `include_statuses`, `report`, and `navigation` may also be present for older consumers.

Governance expectations:

- Snapshot exposes only frontend-safe artifacts.
- Snapshot does not expose credentials or private local source paths.
- Snapshot contains enough provenance to trace back to the source run.
- Frontend reads this snapshot instead of arbitrary producer internals.
