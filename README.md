# Accounting workflows (src)

Python pipeline for ledger ingestion -> canonicalization -> materialization -> views/reports.

## Structure
- `accounting/`: main library + CLI scripts
- `fixtures/`: small sample inputs for smoke / tests
- `templates/`: HTML templates
- `tests/`: tests (if present)
- `notes/`: runbook / contracts

## Quickstart
Preferred interface:
- `make -C accounting smoke-ingest`
- `make -C accounting run_all` (if meaningful)

## Repo hygiene
- Generated outputs are not tracked (`out/`, `accounting/out/`, etc.)
- Local secrets are kept in `private/` and never committed.
