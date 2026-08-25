# Accounting workflows

Python pipeline for ledger ingestion, canonicalization, materialization, semantic marts, debt resolution, governed metrics/dashboards, human report rendering, professional-pack drilldowns, and artifact publication.

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

`run-full` runs canonical materialization, debt resolution/position/activity and treasury, governed frontier and annual metrics, governed human reports, latest alignment, machine-artifact publication, human-report publication, and the release-readiness check.

The current spine is:

```text
ledger ingest
  -> materialization / semantic + cash facts
  -> debt position + activity / treasury
  -> governed frontier + annual dashboard
  -> governed human reports (HTML -> PDF)
  -> publication
  -> professional evidence / drilldowns
```

The retired generic `metric_values`/registry engine and the old `accounting.marts.build` views layer are not live pipeline stages.

`make run-accounting` and `make run-accounting-full` are compatibility aliases for `run-full`.

For bounded operation on an existing run, use the focused targets exposed by `make help`, including `metrics-from-run`, `reports-from-run`, `run-dashboard`, `publish-latest`, and `publish-reports`.

### Governed human reports

Finished human documents are generated under:

```text
out/reports/<RUN_ID>/
```

The current product bundle contains:

- `annual_management/report.html` and `report.pdf`, rendered from the governed annual dashboard CSV/contract/QA artifacts;
- `treasury_accountability/report.html` and `report.pdf`, rendered from the governed monthly cash-accountability mart;
- `report_catalog.json`, which exposes document-discovery metadata only.

Build the reports for the selected run with:

```bash
make run-reports
```

or, when the exact run and metrics artifacts already exist:

```bash
make reports-from-run RUN_STAMP=<existing stamp>
```

PDF is derived from the same HTML using headless Chromium/Chrome. Set `REPORT_BROWSER_BIN=/path/to/chromium` when browser auto-discovery is insufficient.

Publish only the finished document surface with:

```bash
make publish-reports
```

This writes `public/reports/latest_<SCOPE>/`. Accounting CSVs are not part of that publication contract; the downstream viewer consumes the report catalog plus HTML/PDF documents rather than metric/debt schemas.

See `notes/report_bundle_contract.md` for the exact report boundary and provenance rules.

### Professional evidence / drilldowns

The repository does not own a parallel human-report accounting engine. Professional evidence remains layered over governed artifacts:

```bash
make professional-drilldowns
make professional-linked-digest
```

These operate on an existing professional pack. The linked digest is presentation-only and does not recalculate accounting semantics. Report and notebook consumers must likewise read governed metric/debt/treasury artifacts rather than introducing a second accounting engine.

## Runbook
See `notes/accounting_spine_runbook.md` for the per-stage outputs and smoke checklist.

## Publication contracts
See `notes/public_bundle_contract.md` for the consumer-safe machine artifact handoff and `notes/report_bundle_contract.md` for the finished human-report handoff.

## Documentation compass
Use `notes/documentation_compass.md` as the role-based guide to current docs.

## Repo hygiene
- Generated outputs are not tracked (`out/`, `accounting/out/`, etc.).
- Local secrets are kept in `private/` and never committed.
- Historical audits may mention retired module paths; they are evidence, not live command authority.

## Logging convention
Operational Python entrypoints use `YYYY-MM-DDTHH:MM:SSZ LEVEL [stage] message`. Keep `journalctl` as the operational log source of truth and retain per-run CSV/JSON/HTML/PDF artifacts under the governed run, metrics, reports, professional-pack, drilldown, and publication roots rather than duplicating logs into source-controlled artifacts.
