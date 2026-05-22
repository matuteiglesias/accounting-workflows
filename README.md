# Accounting workflows

Python pipeline for ledger ingestion -> canonicalization -> materialization -> views -> metrics -> human balance output.

## Official run path
Run the accounting spine from the repository root with these Make targets:

1. `make run-ingest`
2. `make run-materialize`
3. `make run-views`
4. `make run-metrics`
5. `make run-human-report`

`make run-accounting` is the happy-path wrapper and resolves to `run-human-report`. The legacy storypack / compile branch is not part of the official flow anymore.

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
The pipeline now uses a single Python `logging` convention across stage entrypoints and helpers:

- Format: `YYYY-MM-DDTHH:MM:SSZ LEVEL [stage] message`
- Levels used operationally: `INFO`, `WARNING`, `ERROR`
- Normal logs go to stderr, which keeps them visible in terminals, `make`, wrapper shells, and `journalctl` under systemd
- Output artifacts stay as files under each run directory; logs are not used as a replacement for CSV/JSON manifests or sanity reports
- Ad-hoc dataframe shape / sample spam was moved behind DEBUG

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

Recommendation on per-run logs: keep `journalctl` as the operational source of truth and keep per-run CSV/JSON/HTML artifacts in `out/run/accounting/<RUN_ID>/`, `out/metrics/<RUN_ID>/`, and `out/human_reports/<RUN_ID>/`. A separate per-run log file is not enabled by default because it would duplicate journal storage without adding much decision value for the current manual/systemd workflow.
