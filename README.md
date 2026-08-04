# Accounting workflows

Python pipeline for ledger ingestion, canonicalization, materialization, semantic marts, debt resolution, metrics, dashboards, human reports, and release publication.

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

`run-full` runs the canonical core, debt views, metrics, dashboard assertions, the human report, publication packaging, and the release readiness check.

`make run-accounting` and `make run-accounting-full` are compatibility aliases for `run-full`. They therefore include publication and release-check side effects; they are not aliases for the human-report stage alone.

For bounded operation on an existing run, use the focused targets exposed by `make help`, including `metrics-from-run`, `run-dashboard`, `run-human`, and `publish-latest`.

The legacy storypack / compile branch is not part of the canonical flow.

## Runbook
See `notes/accounting_spine_runbook.md` for the per-stage outputs, required files, and a concise smoke checklist.

## Documentation compass
Use `notes/documentation_compass.md` as the role-based guide to choose the right docs (operators, developers, analysts, and agents).

## Operations playbook
For stability-first incident response (human + agent workflow), see `notes/human_agent_playbook.md`.

## Repo hygiene
- Generated outputs are not tracked (`out/`, `accounting/out/`, etc.)
- Local secrets are kept in `private/` and never committed.

## Logging convention
The pipeline uses a single Python `logging` convention across stage entrypoints and helpers:

- Format: `YYYY-MM-DDTHH:MM:SSZ LEVEL [stage] message`
- Levels used operationally: `INFO`, `WARNING`, `ERROR`
- Normal logs go to stderr, which keeps them visible in terminals, `make`, wrapper shells, and `journalctl` under systemd
- Output artifacts stay as files under each run directory; logs are not used as a replacement for CSV/JSON manifests or sanity reports
- Ad-hoc dataframe shape / sample spam is available only behind DEBUG

Enable extra debug output only when needed:

```bash
ACCOUNTING_DEBUG=1 make run-materialize
# or
ACCOUNTING_LOG_LEVEL=DEBUG python -m accounting.views --reports-dir ... --write-dir ...
```

Recommended operations flow:

```bash
make run-accounting
journalctl --user -u accounting-spine-live.service -n 200 --no-pager
journalctl --user -u accounting-spine-live.service --since "2026-03-18 00:00:00"
```

Keep `journalctl` as the operational log source of truth and retain per-run CSV/JSON/HTML artifacts under `out/run/accounting/<RUN_ID>/`, `out/metrics/<RUN_ID>/`, and `out/human_reports/<RUN_ID>/`. A separate per-run log file is not enabled by default because it would duplicate journal storage without adding much decision value for the current manual/systemd workflow.
