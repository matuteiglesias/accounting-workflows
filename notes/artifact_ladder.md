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
