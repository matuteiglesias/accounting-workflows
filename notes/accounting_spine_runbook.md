# Accounting spine runbook

## Official path

The supported local execution order is:

1. `make run-ingest`
2. `make run-materialize`
3. `make run-views`
4. `make run-metrics`
5. `make run-human-balance`

`run-accounting` is just the wrapper target for the same happy path. Legacy storypack / compile targets are not part of the supported path.

## Stage outputs

### 1. `run-ingest`
Writes to `out/run/accounting/<RUN_ID>/`.

Required outputs:
- `ledger_canonical.csv`
- ingest-side metadata/check files produced by `accounting.ingest`

### 2. `run-materialize`
Writes to the same run root: `out/run/accounting/<RUN_ID>/`.

Required outputs:
- `per_flow_time_long.freq=M.csv` (or current `FREQ`)
- `per_party_time_long.freq=M.csv` (or current `FREQ`)
- `daily_cash_position.csv`

### 3. `run-views`
Writes to `out/run/accounting/<RUN_ID>/views/`.

Required outputs:
- `views/views_sanity.json`
- `views/v_contributions_monthly.csv`
- `views/v_opex_category_monthly.csv`

Optional / transitional inputs:
- `out/run/accounting/<RUN_ID>/reports/fondos_report.csv`
- `out/run/accounting/<RUN_ID>/reports/renta_*.csv`

These legacy report artifacts are best-effort only and are not part of the required pipeline contract.

### 4. `run-metrics`
Writes to `out/metrics/<RUN_ID>/`.

Required contract outputs:
- `metric_registry.csv`
- `metric_values.csv`
- `validation_report.csv`
- `build_manifest.json`
- `metric_views/income_statement_monthly_last6.csv`
- `metric_views/rent_rollup_by_place_m_last6.csv`
- `metric_views/rent_rollup_by_detail_m_last6.csv`
- `metric_views/flow_type_rollup_m_last6.csv`
- `metric_views/draws_discipline_monthly_last6.csv`
- `metric_views/metric_views_manifest.csv`

Optional outputs:
- `metric_values.parquet`
- `metric_values_q_wide.csv`
- `metric_values_y_wide.csv`
- `income_statement_q.csv`
- `income_statement_y.csv`
- `balance_cash_q.csv`
- `balance_cash_y.csv`

### 5. `run-human-balance`
Reads from:
- run root: `out/run/accounting/<RUN_ID>/`
- metrics root: `out/metrics/<RUN_ID>/`

Required inputs from metrics:
- `metric_registry.csv`
- `metric_values.csv`
- `validation_report.csv`
- `build_manifest.json`
- every file listed under `metric_views/` above

Writes to `out/human_reports/<RUN_ID>/balance_human_v2/`.

Required outputs:
- `balance_humano_v2.html`
- `story_manifest.json`
- `tables/` CSV exports
- `html/` per-section HTML exports

## Smoke checklist

Exact command sequence:

```bash
make run-ingest
make run-materialize
make run-views
make run-metrics
make run-human-balance
```

At the end, confirm these paths exist for the active `<RUN_ID>`:

```text
out/run/accounting/<RUN_ID>/ledger_canonical.csv
out/run/accounting/<RUN_ID>/views/views_sanity.json
out/metrics/<RUN_ID>/metric_values.csv
out/metrics/<RUN_ID>/metric_views/income_statement_monthly_last6.csv
out/human_reports/<RUN_ID>/balance_human_v2/balance_humano_v2.html
```
