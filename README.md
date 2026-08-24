# Accounting workflows

Python pipeline for ledger ingestion, canonicalization, materialization, semantic marts, debt resolution, governed metrics/dashboards, professional-pack drilldowns, and artifact publication.

## Official command surface

Run commands from the repository root.

### Fixture and validation path

```bash
make smoke-core
make smoke-full
make validate
```

- `smoke-core` exercises fixture ingest and materialization with semantic and cash checks.
- `smoke-full` adds repository validation and a publication dry-run.
- `validate` runs compilation, contract checks, and the regression suite without private credentials.

### Live canonical core

```bash
make run-canonical
```

`run-canonical` resolves to `run-marts`, whose dependency chain performs live ingest, materialization, and semantic-mart generation for one timestamped run.

### Full live and publication path

```bash
make run-full
```

`run-full` runs the canonical core, debt views, governed metrics, annual-dashboard assertions, artifact publication, and the release-readiness check.  The retired `accounting.human` report stack is not a live pipeline stage.

`make run-accounting` and `make run-accounting-full` are compatibility aliases for `run-full`.

For bounded operation on an existing run, use the focused targets exposed by `make help`, including `metrics-from-run`, `run-dashboard`, and `publish-latest`.

### Human-facing / professional presentation

The repository no longer owns a standalone Flask/front application or a parallel `human_reports` producer. Human-facing work is layered over governed artifacts:

```bash
make professional-drilldowns
make professional-linked-digest
```

These operate on an existing professional pack. The linked digest is presentation-only and does not recalculate accounting semantics. Notebook/report consumers should likewise read governed metric/debt artifacts rather than introducing a second accounting engine.

## Runbook
See `notes/accounting_spine_runbook.md` for the per-stage outputs and smoke checklist.

## Publication contract
See `notes/public_bundle_contract.md` for the consumer-safe artifact handoff.

## Documentation compass
Use `notes/documentation_compass.md` as the role-based guide to current docs.

## Repo hygiene
- Generated outputs are not tracked (`out/`, `accounting/out/`, etc.).
- Local secrets are kept in `private/` and never committed.
- Historical audits may mention retired module paths; they are evidence, not live command authority.

## Logging convention
Operational Python entrypoints use `YYYY-MM-DDTHH:MM:SSZ LEVEL [stage] message`. Keep `journalctl` as the operational log source of truth and retain per-run CSV/JSON/HTML artifacts under the governed run, metrics, professional-pack, drilldown, and publication roots rather than duplicating logs into report artifacts.
