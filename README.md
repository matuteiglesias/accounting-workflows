# Accounting workflows

Python pipeline for ledger ingestion, canonicalization, materialization, semantic marts, debt resolution, governed metrics, human report rendering, professional-pack drilldowns, and artifact publication.

## Official command surface

Run commands from the repository root. `make help` is the executable command authority; the repository deliberately has no compatibility-alias command layer.

### Fixture and validation path

```bash
make smoke-core
make smoke-full
make validate
```

- `smoke-core` exercises fixture ingest and governed materialization with semantic and cash checks.
- `smoke-full` adds repository validation and a publication dry-run.
- `validate` runs compilation, contract checks, and the regression suite without private credentials.

### Live source path

```bash
make run-canonical
```

`run-canonical` is the explicit live source operation: it performs live ingest and then governed materialization for one generated `RUN_ID`. Materialization emits the semantic split, monthly operating statement, semantic QA, and governed cash-close artifacts. There is no separate generic views stage.

### Exact-run stage replay

Downstream stages do not silently re-ingest live inputs. Select an existing exact run and execute only the required stage:

```bash
make run-materialize RUN_ID=<exact-run-id>
make run-debt       RUN_ID=<exact-run-id>
make run-metrics    RUN_ID=<exact-run-id>
make run-reports    RUN_ID=<exact-run-id>
```

`run-debt` owns the complete debt stage: resolution, balance views, monthly position/activity marts, and treasury accountability. `run-metrics` also asserts the governed annual dashboard outputs; there is no separate dashboard command. `run-reports` consumes the already-produced treasury and metrics artifacts for that exact run.

### Full live and publication path

```bash
make run-full
```

`run-full` is the ordered live composite:

```text
run-canonical
  -> run-debt
  -> run-metrics
  -> run-reports
  -> atomic latest alignment
  -> publish-latest + publish-reports
  -> release-check
```

For automation that keeps credentials in an env file, `make run-env` loads `ENV_FILE` (default `private/accounting.env`) and delegates to `run-full`.

The retired generic `metric_values`/registry engine, old `accounting.marts.build` views layer, notebook report stack, and historical Make aliases are not live pipeline stages.

### Governed human reports

Finished human documents are generated under:

```text
out/reports/<RUN_ID>/
```

The current product bundle contains:

- `annual_management/report.html` and `report.pdf`, rendered from the governed annual dashboard CSV/contract/QA artifacts;
- `treasury_accountability/report.html` and `report.pdf`, rendered from the governed monthly cash-accountability mart;
- `report_catalog.json`, which exposes document-discovery metadata only.

PDF is derived from the same HTML using headless Chromium/Chrome. Set `REPORT_BROWSER_BIN=/path/to/chromium` when browser auto-discovery is insufficient.

Publish only the finished document surface with:

```bash
make publish-reports
```

This writes `public/reports/latest_<SCOPE>/`. Accounting CSVs are not part of that publication contract; the downstream viewer consumes the report catalog plus HTML/PDF documents rather than metric/debt schemas.

See `notes/report_bundle_contract.md` for the exact report boundary and provenance rules.

### Professional evidence / drilldowns

Professional evidence remains layered over governed artifacts:

```bash
make professional-drilldowns
make professional-linked-digest
```

These operate on an existing professional pack. The linked digest is presentation-only and does not recalculate accounting semantics.

## Source-tree contract

The import root is intentionally the top-level `accounting/` package. This wave does **not** introduce a parallel `src/accounting` tree. Runtime Python belongs under `accounting/`; fixtures, reference policies, scripts, tests, documentation, and historical diagnostics stay in their dedicated top-level roots. Notebook/report presentation artifacts do not belong inside the runtime package.

See `notes/repository_tree_contract.md` for the governed root/path classification.

## Runbook
See `notes/accounting_spine_runbook.md` for the per-stage outputs and smoke checklist.

## Publication contracts
See `notes/public_bundle_contract.md` for the consumer-safe machine artifact handoff and `notes/report_bundle_contract.md` for the finished human-report handoff.

## Documentation compass
Use `notes/documentation_compass.md` as the role-based guide to current docs.

## Repo hygiene
- Generated outputs are not tracked (`out/`, `accounting/out/`, etc.).
- Local secrets are kept in `private/` and never committed.
- Historical audits may mention retired module paths or commands; they are evidence, not live authority.
- New compatibility aliases require a concrete external caller and an explicit removal condition; otherwise use the canonical target or module directly.

## Logging convention
Operational Python entrypoints use `YYYY-MM-DDTHH:MM:SSZ LEVEL [stage] message`. Keep `journalctl` as the operational log source of truth and retain per-run CSV/JSON/HTML/PDF artifacts under the governed run, metrics, reports, professional-pack, drilldown, and publication roots rather than duplicating logs into source-controlled artifacts.
