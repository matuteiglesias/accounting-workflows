# Accounting docs-input bridge program — 2026-09-17

Status: approved implementation specification  
Branch: `feat/docs-input-bridge`  
Owner: Accounting Workflows  
Consumer class: internal documentation / Family Strategy Lab / professional review  
Contract target: `artifact:acct.docs-input@1`

## Purpose

Materialize a small, governed, read-only bridge from existing accounting authorities into documentation-ready factual inputs.

This program exists because the backend already owns most of the transaction, support, cost, debt, treasury, evidence, and drilldown facts needed by higher-level legal, strategy, governance, and psychology documents. The missing work is primarily explicit projection, packaging, reconciliation, and provenance — not a second accounting engine.

The bridge must make it easier to answer factual questions such as:

- which governed transactions in a defined population have approved supporting evidence;
- who paid, supported, or physically moved funds for a property-related cost;
- which cost allocations remain unresolved rather than being silently promoted to debt or legal liability;
- which prospective obligations are known or estimated for future periods;
- which accounting populations are available for downstream actor/property dossiers without requiring downstream code to reclassify ledger rows.

The bridge must never decide legal ownership, legal liability, entitlement, moral fairness, negotiation position, psychological motive, or who should administer family assets.

## Architectural invariant

Preserve the production spine:

```text
ledger canonicalization
  -> materialization
  -> semantic marts
  -> debt
  -> metrics
  -> human reports
  -> professional pack
  -> drilldowns
  -> docs-input projection
```

`docs-input` is a downstream projection. It may select, join through explicit governed identities, aggregate, annotate provenance, and fail closed. It may not:

- classify ledger rows independently;
- infer a legal debtor, creditor, beneficiary, owner, administrator, heir, usufructuary, or bearer;
- infer cash from economic attribution;
- infer evidence relations by similarity of amount/date/text;
- convert an unresolved allocation into debt;
- net legal positions between actors;
- manufacture a distributable balance;
- treat a treasury control balance as validated liquidity;
- combine ARS and USD without a separately governed valuation artifact;
- move document custody, OCR, raw-document intake, viewer behavior, or public documentation authority into this repository.

## Governing source contracts

The implementation must consume existing authorities rather than recreate them.

Primary sources:

- `ledger_canonical.csv` — transaction identity/provenance authority;
- `classification_audit.csv` / `monthly_flow_semantic_split.csv` — semantic membership and actor/property dimensions;
- `stakeholder_settlement_detail.csv` / `monthly_stakeholder_support.csv` — explicit support/settlement authority;
- `cost_allocation_gaps.csv` — unresolved economic-burden projection, explicitly not debt;
- `monthly_cash_accountability.csv` / `monthly_box_treasury_flow.csv` — governed treasury/control populations;
- `monthly_debt_position.csv` / `monthly_debt_activity.csv` / repayment detail — governed debt populations;
- professional drilldown detail plus `acct.transaction-evidence@1` sidecar — evidence-link coverage;
- annual governed metrics only where an annual scalar or comparison is needed.

Explicit identity joins are allowed through governed keys such as `tx_id`, `source_tx_id`, `settlement_case_id`, `physical_payment_id`, and existing exact-run identity. Heuristic joins are prohibited.

## Output contract

Exact-run generated root:

```text
out/docs_input/<RUN_ID>/
```

Initial contract:

```text
docs_input_manifest.json
accounting_evidence_coverage.csv
actor_property_cost_support_detail.csv
unresolved_allocation_detail.csv
required_reserve_schedule.csv        # only when a prospective commitments input is present
required_reserve_schedule_qa.csv     # paired with the schedule
```

The generated output is not source-controlled data. Only code, tests, schemas/specs, and small fixtures belong in Git.

### `docs_input_manifest.json`

Schema id: `acct.docs-input@1`.

Minimum fields:

- `schema_version`;
- `source_run_id`;
- `scope_tag` when available;
- `generated_at_utc`;
- `artifacts[]` with logical name, relative path, row count, SHA-256, grain, accounting authority statement, and downstream caveat;
- optional prospective-input provenance with presence, schema version, row count, and SHA-256;
- `accounting_authority_changed=false`;
- `legal_interpretation_included=false`.

The manifest is a packaging/provenance surface, not a metric registry.

## Artifact 1 — accounting evidence coverage

### Question

For a defined transaction-detail population, what proportion has approved supporting evidence, candidate evidence, or no linked evidence?

### Source rule

Reuse the existing `acct.transaction-evidence@1` contract and professional drilldown transaction membership. Evidence linkage must be by explicit `tx_id` only.

### Minimum grain

One row per explicit drilldown/report population and currency, with optional Box/property/actor dimensions only where already present in the source detail.

Minimum columns:

```text
population_id
population_label
source_detail_path
Currency
Box
property
actor
transaction_rows
approved_evidence_rows
candidate_evidence_rows
missing_evidence_rows
coverage_pct
source_tx_ids
```

### Invariants

- `approved + candidate + missing == transaction_rows`;
- coverage percentage is `approved / transaction_rows` only;
- candidate evidence is never counted as approved proof;
- zero-row populations have explicit zero coverage and are not treated as fully evidenced;
- evidence status cannot change accounting values or membership;
- duplicated `tx_id` rows inside the source population must be counted according to the source grain, but a separate distinct-ID count may be emitted if useful; do not silently deduplicate financial membership.

## Artifact 2 — actor × property × cost/support detail

### Question

For property-related governed costs/support cases, what cost was recognized, which property/Box was involved, who physically paid where known, who was recognized as a support actor where governed, what amount was recognized, and what evidence/allocation status exists?

### Source rule

Compose `stakeholder_settlement_detail.csv` with `classification_audit.csv` only through explicit source identities. Do not infer a settlement relation from matching amount/date/provider.

### Minimum columns

```text
settlement_case_id
source_tx_id
Date
period
Currency
property
Box
obligation_box
expense_category
gross_cost
stakeholder_actor
actor_role
recognized_support
settlement_mode
cash_path
physical_payment_id
physical_payer
physical_payee
payment_method
evidence_ref
evidence_status
allocation_status
allocation_basis
obligation_period
settlement_period
debt_origin
```

Optional additive fields may be preserved where already governed.

### Invariants

- recognized support reconciles to the governed stakeholder settlement detail;
- gross cost is not multiplied by multi-leg joins;
- responsibility-mirror/allocation-component legs do not become paid/support amounts unless their existing contract says so;
- physical payer is only populated from governed physical evidence/metadata;
- `property`/`Lugar` is descriptive and never interpreted as ownership;
- no `legal_obligor`, `legal_debtor`, `legal_creditor`, `legal_owner`, or equivalent field is created.

## Artifact 3 — unresolved allocation detail

### Question

Which governed economic-burden items remain unresolved as to allocation?

### Source rule

Project `cost_allocation_gaps.csv` without altering its semantic boundary.

The output may enrich rows with approved evidence-link status through `source_tx_id`, but must not add a debt or legal bearer.

### Minimum columns

Preserve at least:

```text
source_tx_id
Date
period
Currency
amount
property
description
economic_scope
accounting_nature
debt_effect
allocation_status
asserted_bearer
source_file
source_row
evidence_status
```

### Invariants

- every source gap appears exactly once;
- total amount by native currency reconciles exactly to `cost_allocation_gaps.csv`;
- `debt_effect` remains `none` unless the upstream authority changes through a separately approved accounting rule;
- `asserted_bearer` remains blank unless a governed source explicitly supplies a non-legal allocation assertion under a future approved contract;
- the docs-input layer never promotes unresolved allocation to established debt.

## Artifact 4 — prospective commitments and required reserve schedule

This is the only genuinely new data family in this program.

### Purpose

Represent future known/estimated property obligations without pretending they have already occurred in the ledger.

### Input boundary

Optional private input:

```text
private/accounting/treasury_commitments.csv
```

or an explicitly configured path.

Input schema version: `treasury_commitments.v1`.

Minimum columns:

```text
commitment_id
due_date
property
Box
Currency
expected_amount
amount_status
commitment_status
obligation_category
source_ref
notes
```

Allowed `amount_status` values:

- `known`;
- `estimated`;
- `range_midpoint` only if a separate range contract is later approved.

Allowed `commitment_status` values:

- `contracted`;
- `known_due`;
- `approved_plan`;
- `scenario`.

Initial reserve authority includes only `contracted`, `known_due`, and `approved_plan`. `scenario` rows remain visible but are excluded from required-reserve totals.

### Output

`required_reserve_schedule.csv` keeps one row per commitment plus cumulative reserve information by native currency and due month.

Minimum derived fields:

```text
commitment_id
due_date
period
property
Box
Currency
expected_amount
amount_status
commitment_status
obligation_category
reserve_included
required_reserve_amount
cumulative_required_reserve
source_ref
notes
```

### Invariants

- prospective commitments never enter the canonical ledger, OPEX, debt, cash, or historical metrics merely by existing;
- native currencies remain separate;
- scenario rows are not required reserve;
- duplicate `commitment_id` fails closed;
- invalid dates/amounts/statuses fail closed;
- reserve totals equal the included commitment rows exactly;
- missing input is a supported no-op, not zero obligations;
- no `distribution_capacity` value is produced in v1.

### Distribution-capacity stop condition

Do not compute:

```text
validated cash - required reserve = distributable balance
```

unless a later contract explicitly establishes both the liquidity authority and legal/administrative basis for that calculation.

If validated cash is unavailable, downstream documentation may state that reserve obligations are known while distribution capacity remains unavailable. The docs-input layer must not turn treasury control balances into liquidity.

## Artifact 5 — docs-input manifest / consumer bridge

The manifest is the only stable cross-repository interface in this program.

Family Strategy Lab, accounting-docs, or other consumers may read this manifest and named artifacts. They must not depend on arbitrary internal CSVs or reinterpret accounting membership.

The bridge may later support property or stakeholder dossiers by composition. Those dossiers are downstream documentation products and are explicitly not new accounting authorities.

## Implementation waves

### W0 — spec and boundary lock

Deliverables:

- this specification;
- branch from current `main`;
- confirm `SYSTEM.yaml` already declares `acct.docs-input@1`;
- no runtime behavior change.

Exit condition: scope/non-goals accepted and encoded before code changes.

### W1 — docs-input package + manifest skeleton

Implement a small `accounting.docs_input` package with:

- deterministic output root handling;
- SHA-256 / row-count metadata;
- manifest writer;
- CLI entrypoint that consumes an exact run and optional professional pack/evidence paths;
- no integration into `run-full` yet unless tests demonstrate the stage is safe and exact-run only.

Exit tests:

- exact-run identity preserved;
- relative paths cannot escape bundle root;
- manifest contains no accounting values beyond artifact metadata;
- no source artifact mutation.

### W2 — evidence coverage

Implement the evidence-coverage projection by reusing existing evidence relations and drilldown transaction detail.

Exit tests:

- approved/candidate/missing partition reconciles;
- malformed or partial evidence sidecars fail closed under the existing evidence contract;
- missing evidence sidecar produces an explicit unavailable/no-evidence state without changing accounting facts;
- duplicate evidence relations remain governed by the existing evidence loader.

### W3 — actor/property/cost/support + unresolved allocations

Implement both factual projections.

Exit tests:

- settlement amounts reconcile to source contract;
- no multiplied gross cost from joins;
- unresolved allocation row count and currency totals reconcile exactly;
- no debt/legal fields are inferred;
- `status=X` exclusions and existing semantic scope remain unchanged;
- drilldown/source transaction IDs remain traceable.

### W4 — prospective commitments + required reserve

Implement optional input parser, provenance, validation, schedule, and QA.

Exit tests:

- missing input is supported;
- duplicate IDs / invalid statuses / mixed invalid amounts fail closed;
- scenario rows excluded from required reserve;
- reserve schedule reconciles by month/currency;
- no canonical or historical artifact changes when the prospective sidecar is added/removed.

### W5 — command integration, contracts, and docs refresh

Add one explicit exact-run command such as:

```text
make run-docs-input RUN_ID=<exact-run-id>
```

The command must not ingest live data, move latest pointers, publish, or mutate reports.

Then refresh:

- `SYSTEM.yaml` verification/interface note if needed;
- `notes/output_contracts.md`;
- `notes/report_bundle_contract.md` only if the report boundary needs a clarification;
- `README.md` current report surface, which currently understates transaction-tape/debt/specialized coverage;
- documentation compass/runbook if the new command is part of supported operator surface.

Exit tests:

- `make validate` passes;
- relevant docs-input tests pass independently;
- no live inputs required;
- generated outputs are ignored/untracked;
- current report/drilldown contracts remain unchanged unless explicitly documented.

## Reconciliation matrix

Every implementation wave must report these checks where applicable:

| Layer | Required check |
|---|---|
| canonical ledger | no source rows or classifications changed |
| semantic materialization | no changed totals/membership unless explicitly approved — expected delta for this program is zero |
| stakeholder support | docs-input recognized support equals governed source membership |
| unresolved cost allocation | row count and native-currency totals exactly equal source gap artifact |
| debt | no gap item promoted into debt; debt position/activity unchanged |
| treasury | no economic-only/support row promoted to physical cash; control/validated-cash semantics unchanged |
| annual metrics | unchanged |
| human reports | unchanged except later documentation/catalog text if explicitly refreshed |
| professional drilldowns | membership and displayed values unchanged; evidence coverage is passive enrichment/projection only |
| docs-input | every artifact has provenance, row count, hash, grain, caveat, and deterministic reconciliation |

## Before/after measurement rule

For W1–W4, the expected accounting before/after effect is:

```text
canonical ledger totals            delta = 0
semantic bucket totals             delta = 0
debt stock/activity                delta = 0
treasury physical cash             delta = 0
annual governed metrics            delta = 0
professional displayed values      delta = 0
```

The only additions are new downstream factual projections and, in W4, an optional prospective non-ledger dataset.

Any non-zero delta in the existing accounting pipeline is a stop condition and must be explained before continuation.

## Privacy and publication

- `out/docs_input/` is generated/internal by default;
- prospective commitment inputs remain private and uncommitted;
- evidence documents are not copied into docs-input;
- manifests may contain evidence IDs/relative references but not raw private document contents;
- nothing in this program implies public publication;
- downstream privacy/access policy remains owned outside this repository.

## Anti-bloat rules

Do not create:

- a second generic views framework;
- a second metric registry;
- a new database/API/service;
- one renderer per factual artifact;
- a legal allocation engine;
- a psychology/fairness score;
- actor rankings or accountability scores;
- automatic property ownership inference;
- automatic entitlement/net-distribution calculations.

Prefer small projection functions over new frameworks.

## Stop conditions

Stop and return a decision packet rather than guessing if implementation requires deciding any of the following:

- whether an accounting cost is legally borne by a specific person;
- whether rent legally belongs to a particular actor beyond already-governed accounting scope;
- whether an unresolved item is legally a debt;
- whether a reserve creates legal authority to withhold/distribute funds;
- whether a person has a duty to render accounts for a population not already defined by the input contract;
- whether two transactions should be linked without an explicit governed identity;
- whether a prospective item should be recognized historically.

## Definition of done

The program is complete when:

1. `acct.docs-input@1` exists as a real generated exact-run bridge rather than only a declaration in `SYSTEM.yaml`;
2. evidence coverage is materialized from explicit `tx_id` relations;
3. actor/property/cost/support detail is available without legal inference;
4. unresolved allocations are exposed without promotion to debt;
5. optional prospective commitments can produce a governed required-reserve schedule;
6. no distribution-capacity number is fabricated when validated liquidity/authority is unavailable;
7. all new outputs have provenance and reconciliation tests;
8. existing accounting, debt, treasury, metrics, reports, and drilldowns reconcile with zero semantic drift;
9. operator/documentation surfaces accurately describe the current backend;
10. no private/generated artifacts are committed.

## Completion report template

```text
Wave:
Invariant protected:
Changed:
Accounting rule changed: no / yes (explain)
Source records traced:
New semantic classification: none / approved change
Before/after existing accounting totals:
New docs-input outputs:
Reconciliations checked:
Drilldown leakage checked:
Tests / CI:
Live inputs accessed:
Publication performed:
Blocked accounting/legal decision:
Next bounded wave:
```
