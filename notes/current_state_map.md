# Accounting Backend Current State Map

## Purpose

The accounting backend is an artifact-producing pipeline that turns household/property accounting records into canonical ledgers, materialized views, metric values, debt-resolution artifacts, human-readable balance tables/reports, and frontend-ready outputs.

A good one-line purpose:

```text
accounting-backend converts messy accounting records into validated financial views and human-readable balance artifacts.
```

The system is not just a calculator. It is becoming a small accounting intelligence pipeline:

```text
source ledger
  → canonical ledger
  → materialized analytical views
  → metric registry + metric values
  → debt resolution
  → human balance tables/reports
  → frontend/public handoff
```

The clearest internal contract today is the canonical ledger produced by `ingest.build_ledger_base(...)`. The module docstring explicitly defines canonical columns such as `tx_id`, `Date`, `amount`, `Currency`, `payer`, `receiver`, `Flujo`, `Tipo`, `status`, `Box`, `source_file`, `source_row`, and `ingest_ts`, and notes that ingest anomalies are attached in `DataFrame.attrs["anomalies"]`. 

---

## Pipeline spine

Current inferred spine:

```text
input source
  Google Sheet / fixture CSV / local financial records
    ↓
ingest.py
  canonical ledger DataFrame
    ↓
materialize.py
  ledger_canonical.csv
  per_flow_time_long.freq=*.csv
  per_party_time_long.freq=*.csv
  daily_cash_position.csv
  manifest / partitions
    ↓
views.py
  report/view tables from materialized Stage D artifacts
    ↓
metrics_views.py + metrics_builders.py + build_metric_values.py
  metric views
  metric_values.csv
  metric_registry.csv
  validation_report.csv
  metric drilldown artifacts
    ↓
resolve_internal_debt_v2.py
  debt open items
  allocations
  repayment events
  timeline/reconciliation
    ↓
build_debt_balance_views.py
  debt balance views over time
    ↓
human_balance_tables.py
  human-facing table specs and generated tables
    ↓
human_balance_document_factory.py / human_balance_front_factory.py
  HTML / narrative / report surfaces
    ↓
publish_latest.py / accounting-viewer
  frontend-ready snapshot
```

The strongest architectural center is this:

```text
canonical ledger
  → materialized views
  → metrics / debt / human balance outputs
```

The weakest part, based on the files seen so far, is that multiple reporting/front modules overlap: `human_balance_document_factory.py` looks active, while `human_balance_front_factory.py` explicitly calls itself a stub architecture for a gradual migration away from a legacy monolith.  

---

## Artifact levels

A first accounting-specific artifact ladder could be:

### Level 0: Source / working inputs

```text
Google Sheet
fixture CSVs
local input CSVs
raw ledger-like rows
```

Purpose: source material, not stable downstream contract.

Producers/owners:

```text
external sheets
manual exports
fixtures
```

Consumers:

```text
ingest.py
```

---

### Level 1: Canonical accounting records

```text
ledger_canonical.csv
canonical ledger DataFrame
ingest anomalies
```

Purpose: normalized accounting facts.

The canonical ledger contract is already described in `ingest.py`: stable columns, normalized names, source tracking, and attached anomalies. 

Potential future contracts:

```text
ledger_entry.v1
ingest_anomaly.v1
```

---

### Level 2: Materialized analytical views

```text
per_flow_time_long.freq=<freq>.csv
per_party_time_long.freq=<freq>.csv
daily_cash_position.csv
box_balance_time_long.freq=<freq>.csv
box_flow_balance_time_long.freq=<freq>.csv
views/*
meta/stage_D_materialize.json
```

`materialize.py` explicitly presents itself as the materialization layer for CSV outputs and exports functions like `materialize_per_flow`, `materialize_per_party`, `materialize_daily_cash`, `materialize_loans`, and `materialize_all`. It writes CSV files plus manifest/partition metadata. 

---

### Level 3: Metrics and debt-resolution buses/views

```text
metric_values.csv
metric_registry.csv
validation_report.csv
metric_views/*
metric_drilldown/*
debt_open_items.csv
debt_allocations.csv
debt_repayment_events.csv
debt_balance_monthly/quarterly/yearly views
```

`metrics_io.py` defines the metric values schema, with columns like `metric_id`, `period_grain`, `period`, `currency`, `value`, `run_id`, `as_of_date`, `source_layer`, `build_status`, and `build_detail`. 

`metrics_registry.py` defines a metric registry schema with `metric_id`, statement/section/label fields, aggregation rules, leaf status, source layer, builder key, parent metric, display code, currency mode, and status. 

`resolve_internal_debt_v2.py` has a richer debt engine model, including `OpenItem`, `Allocation`, `RepaymentEvent`, `TimelineEvent`, and `StatusReconciliation`. 

---

### Level 4: Human balance/report surfaces

```text
human tables
HTML reports
storypack
front-facing balance report
methodology sections
drilldown-linked tables
```

`human_balance_tables.py` defines a public `HumanTableSpec` model and a default table registry with table specs such as `cash_snapshot`, `debt_snapshot`, `income_statement_monthly_last6`, `debt_balance_monthly_last12`, and validation reports. 

`human_balance_document_factory.py` builds the current human report around `balance_human_v2`, loading metric views, drilldown artifacts, human tables, and rendering report HTML. 

`human_balance_front_factory.py` is a front-oriented report builder stub intended to reuse `human_balance_tables` and support a gradual migration away from the legacy balance monolith. 

---

### Level 5: Frontend/public handoff

```text
public/accounting/*
accounting-viewer/accounting_surface/data/*
accounting-viewer/public/accounting/latest/*
```

This layer was visible in the earlier tree, but the specific publishing module was not uploaded in this batch. Treat as likely but not fully mapped yet.

Potential future contract:

```text
accounting_front_snapshot.v1
```

---

## Main modules

| Module                              | Role                                                                                                                               | Inputs                                                            | Outputs                                                                                                    | Status                                                                                                                                                                           |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                         | Configuration loader for project paths, inputs, output dirs, ETL options, finance defaults, manifests, and validation requirements | YAML/JSON config, environment defaults                            | `Config` object and path/options map                                                                       | Active support module. `Config` includes `out_dir`, fixtures, Google Sheet settings, `freq`, `dry_run`, `force`, base currency, FX table, manifest names, and required headers.  |
| `ingest.py`                         | Source ingest and canonical ledger builder                                                                                         | Google Sheet, fixture CSV/parquet, source ledger rows             | Canonical ledger DataFrame, anomalies in `attrs["anomalies"]`                                              | Core contract module. Strong docstring and clear canonical schema.                                                                                                               |
| `core_timeseries.py`                | Pure time-series primitives                                                                                                        | Canonical ledger DataFrames                                       | Aggregated per-flow, per-party, daily cash, loan time-series structures                                    | Active utility layer. It explicitly says functions are pure, deterministic, and no-I/O.                                                                                          |
| `materialize.py`                    | Materialization layer                                                                                                              | Canonical ledger DataFrame                                        | CSV artifacts, manifest, partitions                                                                        | Core Level 2 producer. Writes per-flow, per-party, daily cash, loans, etc.                                                                                                       |
| `views.py`                          | Report/view loading and transformation layer                                                                                       | Stage D materialized outputs, optional legacy report artifacts    | Higher-level views and reports                                                                             | Active but somewhat bridging/compatibility-heavy. It says Stage D artifacts are source-of-truth and legacy report artifacts are best-effort only.                                |
| `metrics_views.py`                  | Metric-oriented view builders and loaders                                                                                          | `ledger_canonical.csv`, materialized views                        | Metric view tables such as income statement, rent rollups, flow rollups, draws discipline                  | Active view layer. Loads ledger and standardizes columns/status/periods.                                                                                                         |
| `metrics_io.py`                     | Metric value schema and I/O helpers                                                                                                | Metric DataFrames                                                 | Normalized `metric_values` schema                                                                          | Core contract module for metrics.                                                                                                                                                |
| `metrics_registry.py`               | Metric registry model and normalization                                                                                            | `MetricSpec` records / registry DataFrame                         | Normalized registry DataFrame                                                                              | Core contract module for metric definitions.                                                                                                                                     |
| `metrics_builders.py`               | Leaf metric builders                                                                                                               | `MetricsContext` with ledger/views/debt data                      | Leaf metric value frames                                                                                   | Active builder layer. Uses `MetricsContext` and aggregation helpers.                                                                                                             |
| `metrics_derive.py`                 | Derived metric formulas                                                                                                            | Existing metric values                                            | Sum/subtract-derived metric values                                                                         | Active derived-metric layer. Provides reusable formula builders such as `derive_sum_components` and `derive_formula_subtract`.                                                   |
| `metrics_validate.py`               | Metric validation layer                                                                                                            | Metric values, registry                                           | Validation issues/report DataFrames                                                                        | Active validation module. Checks uniqueness, registry IDs, leaf builder keys, known metric IDs, etc.                                                                             |
| `build_metric_values.py`            | Orchestrator for metric outputs                                                                                                    | Latest run root, materialized views, debt outputs, registry/specs | `metric_values.csv`, `metric_registry.csv`, validation report, metric views, drilldown artifacts, manifest | Core metrics orchestration module. It imports builders, derived metrics, registry, validations, and metric views.                                                                |
| `metric_drilldown.py`               | Drilldown artifact builder for selected metrics                                                                                    | Ledger and metric views                                           | Drilldown detail files, index, manifest                                                                    | Active support/reportability module. Supports metrics like rent total, opex total, and personal draws.                                                                           |
| `resolve_internal_debt.py`          | Older internal debt resolver                                                                                                       | Ledger built from `build_ledger_base`                             | Open items, allocations, repayment events                                                                  | Likely superseded by v2. Simpler model with `OpenItem`, `Allocation`, `RepaymentEvent`.                                                                                          |
| `resolve_internal_debt_v2.py`       | Current internal debt engine                                                                                                       | Canonical ledger debt slice                                       | Open items, allocations, repayment events, timeline, reconciliation                                        | Likely canonical debt resolver. It adds richer fields and explicit rule version `interest_first_fifo_full_only_skip_if_insufficient_v2`.                                         |
| `build_debt_balance_views.py`       | Debt balance view builder                                                                                                          | `debt_open_items.csv` or equivalent open items                    | Debt open balances over dates/periods                                                                      | Active downstream debt view module. Normalizes required open-item columns and builds time-span views.                                                                            |
| `human_balance_tables.py`           | Registry and builder layer for human-facing tables                                                                                 | Metric views, debt views, validation/drilldown context            | Human table specs and generated table artifacts                                                            | Core human-report table layer. Defines default table registry.                                                                                                                   |
| `human_balance_document_factory.py` | Current human balance report builder                                                                                               | Human tables, metric views, drilldowns                            | HTML/report document for `balance_human_v2`                                                                | Active current report factory.                                                                                                                                                   |
| `human_balance_front_factory.py`    | Front-oriented human balance report builder                                                                                        | Human tables and context                                          | Future/narrative front blocks/profiles                                                                     | Stub/transitional. Explicitly says implementation is intentionally stubbed and meant to migrate away from legacy monolith.                                                       |

---

## Entrypoints

Entrypoints are partially inferred from module names and imports. We need the Makefile for certainty.

| Command                                                   | Purpose                                                              | Status                                                                                     |
| --------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `python -m accounting.ingest ...` or direct module CLI    | Build canonical ledger from fixture or Google Sheet                  | Probable. `ingest.py` imports `argparse`, but exact CLI not fully inspected from snippet.  |
| `python -m accounting.materialize ...`                    | Materialize canonical ledger into CSV views                          | Probable core. `materialize.py` defines materialization functions and imports `argparse`.  |
| `python -m accounting.resolve_internal_debt_v2 ...`       | Run canonical debt resolution engine                                 | Probable canonical debt command. v2 has richer rule version and reconciliation model.      |
| `python -m accounting.resolve_internal_debt ...`          | Older debt resolution path                                           | Legacy/compat candidate. Simpler predecessor to v2.                                        |
| `python -m accounting.build_debt_balance_views ...`       | Build historical debt balance views from open items                  | Active downstream debt command.                                                            |
| `python -m accounting.build_metric_values ...`            | Build metric registry, metric values, validations, views, drilldowns | Core metrics command. Strong candidate for canonical metrics entrypoint.                   |
| `python -m accounting.human_balance_document_factory ...` | Build current human balance report/document                          | Active report command.                                                                     |
| `python -m accounting.human_balance_front_factory ...`    | Build newer front-oriented report                                    | Stub/transitional, not yet canonical.                                                      |
| `make ...` targets                                        | Actual canonical orchestration                                       | Unknown until Makefile inspected.                                                          |

Strong recommendation: create an `entrypoints.md` once the Makefile is inspected. The current module-level entrypoints are plausible, but the canonical commands should come from Make targets or the runbook.

---

## Output surfaces

| Path                                                                | Meaning                                                        | Producer                                             | Consumer                                        |
| ------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| `out/<run_id>/ledger_canonical.csv` or similar                      | Canonical ledger export                                        | `ingest.py` / pipeline wrapper                       | `materialize.py`, metrics/views, audit          |
| `out/<run_id>/per_flow_time_long.freq=<freq>.csv`                   | Flow/time aggregate                                            | `materialize.materialize_per_flow`                   | `views.py`, `metrics_views.py`, metric builders |
| `out/<run_id>/per_party_time_long.freq=<freq>.csv`                  | Party/time aggregate                                           | `materialize.materialize_per_party`                  | `views.py`, metrics                             |
| `out/<run_id>/daily_cash_position.csv`                              | Daily cash position                                            | `materialize.materialize_daily_cash`                 | Metrics, human balance                          |
| `out/<run_id>/meta/stage_D_materialize.json`                        | Materialization manifest                                       | `materialize.py`                                     | Diagnostics, downstream stage resolution        |
| `out/<run_id>/views/*`                                              | Higher-level derived/report views                              | `views.py` / metrics view builders                   | Metrics, reports                                |
| `out/<run_id>/metric_values.csv`                                    | Canonical metric values                                        | `build_metric_values.py`                             | Human balance, frontend, validation             |
| `out/<run_id>/metric_registry.csv`                                  | Metric definitions and display metadata                        | `metrics_registry.py` / `build_metric_values.py`     | Metrics validation, reports, frontend           |
| `out/<run_id>/validation_report.csv`                                | Metric/report validation issues                                | `metrics_validate.py` / `build_metric_values.py`     | Human balance, QA                               |
| `out/<run_id>/metric_views/*`                                       | Income/rent/flow/draws metric views                            | `build_metric_values.py` / `metrics_views.py`        | Human tables, reports                           |
| `out/<run_id>/metric_drilldown/*`                                   | Drilldown detail/index/manifest                                | `metric_drilldown.py`                                | Human report links and evidence                 |
| `out/debt_resolution/<run_id>/*`                                    | Debt open items, allocations, repayment events, reconciliation | `resolve_internal_debt_v2.py`                        | Debt balance views, metrics, reports            |
| `out/debt_resolution/<run_id>/debt_open_items.csv`                  | Normalized open debt items                                     | `resolve_internal_debt_v2.py`                        | `build_debt_balance_views.py`                   |
| `out/<run_id>/debt_balance_*` or `out/debt_resolution/<run_id>/...` | Debt balances over time                                        | `build_debt_balance_views.py`                        | Metrics, human balance                          |
| `out/human_reports/<run_id>/*`                                      | Human-readable HTML/report outputs                             | `human_balance_document_factory.py`                  | User/human review                               |
| `out/front/<run_id>/*`                                              | Front-oriented balance artifacts                               | `human_balance_front_factory.py` or frontend factory | `accounting-viewer`                             |
| `public/accounting/latest/*`                                        | Published frontend-ready accounting snapshot                   | `publish_latest.py` likely                           | `accounting-viewer`                             |

Some exact paths remain inferred. `build_metric_values.py` looks for run roots and required metric-view files, and has helper logic for debt candidate directories around `debt_resolution/<run_id>`. 

---

## Known drifts / ambiguities

### 1. `resolve_internal_debt.py` vs `resolve_internal_debt_v2.py`

There are two debt engines. `resolve_internal_debt.py` has the simpler `OpenItem`, `Allocation`, and `RepaymentEvent` model. 

`resolve_internal_debt_v2.py` adds richer fields, timeline events, reconciliation, ledger/engine status, issuer, and a named rule version. 

Likely decision:

```text
resolve_internal_debt_v2.py = canonical
resolve_internal_debt.py = legacy/compat/reference
```

This should be stated explicitly.

---

### 2. `human_balance_document_factory.py` vs `human_balance_front_factory.py`

`human_balance_document_factory.py` appears to be the current functioning report builder around `balance_human_v2`. 

`human_balance_front_factory.py` explicitly describes itself as a stub architecture and says rendering/block logic can be filled in later by Codex. 

Likely decision:

```text
human_balance_document_factory.py = current report factory
human_balance_front_factory.py = next/front architecture, not yet canonical
```

Do not let both be treated equally in docs.

---

### 3. Materialized views are source-of-truth for views, but legacy reports still exist

`views.py` says Stage D artifacts are the source of truth for Views, while legacy report artifacts like `fondos_report.csv` and `renta_*.csv` are best-effort only and must never be required. 

This is good, but it should be promoted into docs:

```text
Legacy reports are optional compatibility inputs, not canonical artifacts.
```

---

### 4. Metric system is better structured than the surrounding pipeline

The metrics layer has clear schema modules:

```text
metrics_io.py
metrics_registry.py
metrics_validate.py
metrics_derive.py
metrics_builders.py
build_metric_values.py
```

This is relatively mature. The clearest contracts are `metric_values.csv` and `metric_registry.csv`.  

The rest of the system should probably imitate this pattern.

---

### 5. Artifact directories may be too implicit

From uploaded files, many modules infer candidate paths dynamically. For example, `build_metric_values.py` searches for run roots and debt candidate dirs. 

This is pragmatic, but docs should clarify:

```text
Which output directory is canonical?
Which fallback paths are legacy compatibility?
Which path should a frontend consume?
```

---

### 6. No clearly named artifact ladder yet

Unlike `media_monitor`, this project does not yet seem to have explicit artifact levels.

A first doctrine should name:

```text
canonical ledger
materialized views
metric/debt analytical artifacts
human/report artifacts
frontend snapshots
```

---

### 7. Human-facing reporting is powerful but at risk of becoming monolithic

`human_balance_tables.py` is a good decomposition point: table specs and table builders. 

`human_balance_document_factory.py` and `human_balance_front_factory.py` should not accumulate too much business logic. The report factories should compose tables/blocks, not define accounting semantics.

---

## Next documentation actions

### 1. Create this file

```text
src/notes/current_state_map.md
```

Use this response as the first draft.

---

### 2. Create `src/notes/artifact_ladder.md`

Suggested structure:

```markdown
# Accounting Artifact Ladder

## Level 0 — Source inputs
Google Sheets, fixtures, raw ledger rows.

## Level 1 — Canonical ledger
`ledger_canonical.csv`, ingest anomalies.

## Level 2 — Materialized views
`per_flow_time_long`, `per_party_time_long`, `daily_cash_position`, Stage D manifest.

## Level 3 — Metric and debt analytical artifacts
`metric_values.csv`, `metric_registry.csv`, validation reports, metric views, debt resolution outputs.

## Level 4 — Human/report surfaces
Human tables, HTML reports, front report blocks.

## Level 5 — Frontend/public snapshot
`public/accounting/latest/*`, viewer-ready artifacts.
```

---

### 3. Create `src/notes/entrypoints.md`

This needs the Makefile.

Minimum table:

```markdown
| Command | Layer | Purpose | Status |
|---|---|---|---|
```

Classify each command as:

```text
canonical
support
legacy
experimental
```

---

### 4. Create `src/notes/module_inventory.md`

Use a table like:

```markdown
| Module | Layer | Status | Canonical outputs | Notes |
|---|---|---|---|---|
```

Especially mark:

```text
resolve_internal_debt.py = legacy candidate
resolve_internal_debt_v2.py = canonical candidate
human_balance_front_factory.py = stub/future
human_balance_document_factory.py = current
```

---

### 5. Create `src/notes/output_contracts.md`

Start with the stable contracts:

```text
canonical ledger
metric_values.csv
metric_registry.csv
validation_report.csv
debt_open_items.csv
human tables
frontend snapshot
```

For each:

```markdown
## metric_values.csv

Producer:
Consumer:
Required columns:
Validation:
Example path:
Stability:
```

---

### 6. Inspect Makefile next

The next evidence needed is:

```bash
cd ~/repos/accounting-backend/src
sed -n '1,260p' Makefile
```

Without the Makefile, we can understand the architecture but not the authoritative command surface.

---

## Short diagnosis

This project is not as chaotic as it may feel. It already has several good seams:

```text
ingest canonical ledger
materialize Stage D outputs
metric registry/value schema
debt resolver v2
human table specs
human report factory
```

The main weakness is that the architecture is implicit. The pipeline has grown into layers, but the docs do not yet name those layers.

The first upgrade should be documentation and authority labeling, not refactor.

