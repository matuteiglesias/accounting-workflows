# Accounting workflows

Python pipeline for ledger ingestion -> canonicalization -> materialization -> views -> metrics -> human balance output.

## Official run path
Run the accounting spine from the repository root with these Make targets:

1. `make run-ingest`
2. `make run-materialize`
3. `make run-views`
4. `make run-metrics`
5. `make run-human-balance`

`make run-accounting` is the happy-path wrapper and resolves to `run-human-balance`. The legacy storypack / compile branch is not part of the official flow anymore.

## Runbook
See `notes/accounting_spine_runbook.md` for the per-stage outputs, required files, and a concise smoke checklist.

## Repo hygiene
- Generated outputs are not tracked (`out/`, `accounting/out/`, etc.)
- Local secrets are kept in `private/` and never committed.
