# Makefile control-plane notes

## Target families

The Makefile is organized around these families: doctor, validate, smoke, fixture, live, canonical, metrics, dashboard, human, publish, release, legacy, diagnostic, and cleanup.

## Compatibility policy

Existing user-facing aliases remain available. Ambiguous legacy names are labeled in `make help` and routed to explicit targets where practical:

- `metrics` -> `run-metrics` -> `metrics-from-run`
- `human-report` -> `run-human-report` -> `run-human`
- `publish` -> `publish-latest`
- `build-all` -> `run-full` (which includes `release-check`)

## Smoke model

`smoke-core` is the CI-safe offline core. It uses the ledger fixture and validates ingest, materialize, semantic outputs, and cash wrapper artifacts.

`smoke-full` is intentionally fixture-safe and currently partial: it runs `smoke-core`, static/contract validation, and publish dry-run. Fixture debt data and a fully fixture-compatible human/publish bundle are follow-up work; smoke-full does not use private Google Sheets.

## Metrics isolation

`metrics-from-run` is the stable target for rebuilding metrics from an existing canonical `RUN_OUT`. It does not depend on live ingest or debt resolution. Use `run-metrics-live` when live upstream orchestration is desired.

## Release readiness

`publish-latest` only packages latest artifacts. `release-check` validates the public dashboard surface and fails with a clear suggestion when the public bundle is absent.
