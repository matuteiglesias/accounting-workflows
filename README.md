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

- `smoke-core` exercises fixture ingest and governed materialization with semantic and cash checks.
- `smoke-full` adds repository validation and a publication dry-run.
- `validate` runs compilation, contract checks, and the regression suite without private credentials.

### Live canonical core

```bash
make run-canonical
```

`run-canonical` resolves directly to live ingest plus governed materialization. Materialization emits the semantic split, monthly operating statement, semantic QA, and governed cash-close artifacts. There is no separate generic views stage.

### Full live and publication path

```bash
make run-full
```

`run-full` runs canonical materialization, debt resolution/position/activity, governed frontier and annual metrics, artifact publication, and the release-readiness check.

The current spine is:

```text
ledger ingest
  -> materialization / semantic + cash facts
  -> debt position + activity / treasury
  -> governed frontier + annual dashboard
  -> publication
  -> professional reports / drilldowns
```

The retired generic `metric_values`/registry engine and the old `accounting.marts.build` views layer are not live pipeline stages.

`make run-accounting` and `make run-accounting-full` are compatibility aliases for `run-full`.

For bounded operation on an existing run, use the focused targets exposed by `make help`, including `metrics-from-run`, `run-dashboard`, and `publish-latest`.

### Human-facing / professional presentation

The repository does not own a parallel human-report accounting engine. Human-facing work is layered over governed artifacts:

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
