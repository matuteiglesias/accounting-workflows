# Accounting workflows

Python pipeline for ledger ingestion, canonicalization, materialization, semantic marts, debt resolution, governed metrics, human report rendering, professional-pack drilldowns, documentation-ready factual projections, and artifact publication.

## Official command surface

Run commands from the repository root. `make help` is the executable accounting command authority; the repository deliberately has no compatibility-alias command layer. The docs-input bridge also exposes one bounded exact-run Python entrypoint and is not part of live ingestion or publication.

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

A separate downstream docs-input projection can be built from one exact run without re-ingesting, moving latest pointers, publishing, or changing accounting authority:

```bash
python -m accounting.docs_input.build \
  --run-root out/run/accounting/<exact-run-id> \
  --out-dir out/docs_input/<exact-run-id>
```

Optional inputs are explicit:

```bash
# Add professional drilldown evidence coverage.
--pack-dir out/professional_pack/<pack>

# Add a private prospective reserve schedule without entering the ledger.
--commitments private/accounting/treasury_commitments.csv
```

The bridge emits `acct.docs-input@1` factual projections only. It does not infer legal responsibility, ownership, entitlement, fairness, or a distributable balance.

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

The docs-input stage is deliberately outside this live/publication composite. It is generated only when a downstream documentation/professional consumer needs the factual bridge.

For automation that keeps credentials in an env file, `make run-env` loads `ENV_FILE` (default `private/accounting.env`) and delegates to `run-full`.

The retired generic `metric_values`/registry engine, old `accounting.marts.build` views layer, notebook report stack, and historical Make aliases are not live pipeline stages.

### Governed human reports

Finished human documents are generated under:

```text
out/reports/<RUN_ID>/
```

The current product bundle contains core documents for:

- `annual_management/report.html` and `report.pdf`, rendered from the governed annual dashboard CSV/contract/QA artifacts;
- `treasury_accountability/report.html` and `report.pdf`, rendered from the governed monthly cash-accountability mart;
- `treasury_transaction_tape/report.html` and `report.pdf` when the governed transaction-detail source is available;
- `debt_accountability/report.html` and `report.pdf` from governed debt position/activity, repayment, and unresolved-allocation inputs;
- the available specialized governed reports declared in `accounting/reports/specialized/spec.py`, each rendered only when its named source view is available;
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

### Documentation-ready factual bridge

`accounting.docs_input` packages a small, generated exact-run surface for downstream internal documentation and professional reasoning:

```text
out/docs_input/<RUN_ID>/
  docs_input_manifest.json
  accounting_evidence_coverage.csv
  actor_property_cost_support_detail.csv
  unresolved_allocation_detail.csv
  required_reserve_schedule.csv        # only when commitments are supplied
  required_reserve_schedule_qa.csv     # paired with the schedule
```

The bridge consumes existing governed authorities and explicit identities such as `tx_id`/`source_tx_id`; it does not create a second views engine. Missing evidence is explicit. Unresolved allocation remains unresolved rather than becoming debt. Prospective commitments remain outside historical ledger/OPEX/debt/cash facts, and scenario rows do not enter required reserve totals.

The implementation program, invariants, stop conditions, and reconciliation matrix are recorded in `notes/docs_input_bridge_program_20260917.md`.

## Source-tree contract

The import root is intentionally the top-level `accounting/` package. This wave does **not** introduce a parallel `src/accounting` tree. Runtime Python belongs under `accounting/`; fixtures, reference policies, scripts, tests, documentation, and historical diagnostics stay in their dedicated top-level roots. Notebook/report presentation artifacts do not belong inside the runtime package.

See `notes/repository_tree_contract.md` for the governed root/path classification.

## Runbook
See `notes/accounting_spine_runbook.md` for the per-stage outputs and smoke checklist.

## Publication contracts
See `notes/public_bundle_contract.md` for the consumer-safe machine artifact handoff and `notes/report_bundle_contract.md` for the finished human-report handoff. Docs-input remains internal/generated by default and is not added to either public publication contract.

## Documentation compass
Use `notes/documentation_compass.md` as the role-based guide to current docs.

## Repo hygiene
- Generated outputs are not tracked (`out/`, `accounting/out/`, etc.).
- Local secrets and prospective commitment inputs are kept in `private/` and never committed.
- Historical audits may mention retired module paths or commands; they are evidence, not live authority.
- New compatibility aliases require a concrete external caller and an explicit removal condition; otherwise use the canonical target or module directly.

## Logging convention
Operational Python entrypoints use `YYYY-MM-DDTHH:MM:SSZ LEVEL [stage] message`. Keep `journalctl` as the operational log source of truth and retain per-run CSV/JSON/HTML/PDF artifacts under the governed run, metrics, reports, professional-pack, drilldown, docs-input, and publication roots rather than duplicating logs into source-controlled artifacts.
