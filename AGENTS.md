# AGENTS.md — Accounting Workflows

## Mission

Maintain the canonical accounting transformation and reporting pipeline without weakening provenance, business-rule, privacy, or publication boundaries.

This repository turns approved accounting inputs into canonical ledger, materialized views, debt outputs, metrics, human reports, professional packs, drilldowns, and publication bundles. It is not the raw-document intake system, the browser viewer, or the documentation authority.

## Authority boundary

Matías owns accounting semantics, scope choices, classification policy, materiality decisions, debt interpretation, publication approval, and any correction to source records.

Agents may:

- implement an explicitly approved rule or contract change;
- improve fixture-safe validation, deterministic generation, diagnostics, and tests;
- repair a reproduced pipeline defect;
- prepare a decision packet for ambiguous accounting meaning.

Agents must not independently:

- reinterpret a transaction, debt, withdrawal, transfer, owner, property, box, currency, status, or reporting period;
- change inclusion/exclusion policy to make totals look cleaner;
- edit generated CSV, JSON, HTML, reports, latest links, or public bundles as a substitute for fixing the pipeline;
- run live ingestion, publish, or clean derived outputs without explicit authorization;
- copy private documents, ledgers, credentials, service-account files, or `.env` contents into the repository;
- move intake, viewer, or docs responsibilities into this codebase.

## Source and generated-data rules

Canonical business logic and transformations live in code, configuration, and tested rules.

Generated outputs under `out/`, `public/accounting/latest/`, reports, professional packs, drilldowns, and `latest` links are evidence from a run. Do not hand-edit them.

When a generated number is wrong:

1. identify the exact upstream input and rule;
2. reproduce the discrepancy in the smallest safe fixture;
3. change source logic or approved configuration;
4. rerun the bounded stage;
5. verify invariants and downstream propagation;
6. record the run ID and output paths.

Do not commit large, private, or source accounting datasets for convenience.

## Execution modes

Treat smoke and live modes as different safety classes.

Safe default checks:

```bash
make doctor
make validate
make smoke-core
make smoke-full
```

These commands still require review of their documented output and prerequisites, but they are intended to avoid live private ingestion.

Live or consequential commands require explicit task authorization and the correct private environment:

```bash
make run-canonical
make run-full
make publish-latest
make release-check
```

`make clean-derived` deletes generated outputs. Do not run it unless cleanup is explicitly requested and the target paths have been inspected.

Do not hide live execution behind a new generic alias.

## Business-rule changes

Any change affecting classifications, scopes, currencies, debt, flows, metrics, totals, publication, or human interpretation must include:

- the problem and affected output;
- the old and new rule;
- a fixture or regression test;
- before/after evidence;
- downstream tables and reports affected;
- migration or re-run implications;
- an explicit statement of what remains unchanged.

Unknown accounting meaning is a stop condition, not an implementation choice.

## Repository boundaries

- `accounting-doc-triage` owns document intake and candidate metadata.
- this repository owns canonical transformations and reporting calculations;
- `accounting-viewer` owns read-only browser navigation over an approved packaged snapshot;
- `accounting-docs` owns published operating and contract guidance.

Read `SYSTEM.yaml` before altering interfaces. `projects` is a portfolio projection and must never become a runtime dependency.

## Change discipline

- Prefer a bounded stage repair over a full-pipeline rewrite.
- Preserve run IDs, provenance, idempotency, and timestamped outputs.
- Keep latest-link updates atomic and explicit.
- Do not add cloud, database, dashboard, orchestration, or framework infrastructure without a demonstrated requirement.
- Avoid repository-wide formatting during accounting-rule work.
- Never claim a live run, source refresh, publication, or number validation that did not occur.

## Completion report

```text
Changed:
Accounting rule changed:
Fixture/test evidence:
Commands run:
Run ID:
Outputs inspected:
Live inputs accessed:
Publication performed:
Totals/invariants checked:
Blocked accounting decision:
Next bounded action:
```
