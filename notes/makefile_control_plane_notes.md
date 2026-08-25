# Makefile control-plane notes

Status: current implementation notes
Last reviewed: 2026-08-25

## Target families

The public Make surface is intentionally small:

- fixture/static validation: `doctor`, `validate`, `smoke-*`;
- live source/composite: `run-ingest`, `run-canonical`, `run-full`, `run-env`;
- exact-run stages: `run-materialize`, `run-debt`, `run-metrics`, `run-reports`;
- sidecars: USD-CCL valuation/management flow targets;
- publication: `publish-latest`, `publish-reports`, `release-check`;
- professional presentation: drilldowns and linked digest;
- maintenance: `clean-derived`.

Internal implementation targets are underscore-prefixed and are not part of the user-facing command contract.

## Alias policy

Compatibility aliases are not retained merely for convenience. Historical names such as `metrics`, `publish`, `build-all`, `run-accounting*`, `run-debt-views`, `run-dashboard`, `*-from-run`, and light/downstream shortcut targets are retired.

A compatibility name may be introduced only when a concrete external caller cannot migrate immediately. It must have a documented caller and removal condition. Otherwise update the caller to the canonical target.

## Exact-run replay

`RUN_ID` selects the shared exact-run identity across canonical, debt, metrics, and reports roots. Downstream targets do not re-ingest live data:

```bash
make run-materialize RUN_ID=<id>
make run-debt RUN_ID=<id>
make run-metrics RUN_ID=<id>
make run-reports RUN_ID=<id>
```

This replaces the former distinction between `run-metrics` and `metrics-from-run`, or `run-reports` and `reports-from-run`.

## Live orchestration

`run-canonical` is explicitly live and orders `run-ingest` before `run-materialize`. `run-full` orders all supported stages and only then aligns latest pointers and publishes. The sequence is encoded in recipes rather than a broad unordered prerequisite list so `make -j` cannot accidentally run accounting stages concurrently.

## Smoke model

`smoke-core` is the CI-safe offline core. It uses the ledger fixture and validates ingest, materialization, semantic outputs, and cash artifacts.

`smoke-full` adds static/contract validation and a publication dry-run. It does not use private Google Sheets.

## Release readiness

Focused exact-run stages never move latest pointers. `_update_latest` is the single supported latest-alignment operation and includes canonical, debt, metrics, and reports. `publish-latest` and `publish-reports` package the already-aligned handoffs; `release-check` validates the public machine bundle.
