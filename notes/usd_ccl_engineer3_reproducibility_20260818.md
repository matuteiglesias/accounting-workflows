# USD/CCL Engineer 3 review — reliability and reproducibility

**Status:** investigation only; no implementation or accounting-rule change.
**Mandate:** make every USD/CCL result explainable two years later from immutable evidence.

## Decision

Treat CCL rates as a local, immutable, independently hashed input; treat the
valuation sidecar as deterministic derived evidence; and use a valuation
manifest to bind ledger bytes, rate bytes, policy bytes, code identity, match
outcomes, and output bytes.

> **No network lookup during an accounting run.** Acquisition and approval of
> CCL observations is an external process. `accounting-workflows` consumes a
> concrete versioned snapshot and never a URL, remote API, or mutable `latest`.

## 1. Invariant being protected

For every reported USD/CCL value, the evidence chain must remain reconstructable:

```text
reported value
  -> contributing projected rows
  -> exact valuation-sidecar bytes
  -> exact canonical-ledger bytes and population
  -> exact rate-snapshot bytes
  -> immutable valuation-policy definition
  -> code/implementation identity
```

Protected invariants:

1. no network access during valuation;
2. identical ledger, rates, policy and implementation produce byte-identical
   sidecar rows;
3. corrected rates create a new valuation, not a historical-ledger mutation;
4. missing, stale, invalid, duplicate, ambiguous or unsupported observations do
   not become zero or implicit fallback;
5. each projected row records the actual observation date and match method;
6. mutable `latest` pointers are operational convenience, never provenance;
7. match/quality counters are exclusive and reconcile to a named source
   population.

## 2. Current behavior found in code

### 2.1 Current FX input loses evidence

`_load_fx_rates` accepts CSV/Parquet, uses the general ledger column normalizer,
requires `Date` and `Currency`, chooses the first alias among `rate_to_base`,
`rate`, `fx`, or `tc`, coerces values, and returns only date, currency, and
rate. It does not preserve source, series, quote direction, observation ID,
publication/acquisition time, snapshot version, timezone, coverage or revision.

Invalid dates/currencies are dropped and invalid rates become null. Because this
happens before validation, raw input counts and rejection reasons cannot be
reconstructed reliably.

`_attach_base_amount` performs exact `(Date, Currency)` matching with
`validate="m:1"`. Duplicate rate keys therefore fail only incidentally during
merge, without an artifact-level duplicate diagnostic. There is no
previous-available policy, business calendar, age limit, match status, or
explicit stale rejection.

### 2.2 FX remains coupled to ingest

`build_ledger_base` attaches `base_amount` inside canonical construction, and
the stable fingerprint includes all columns except `ingest_ts`. Rate changes can
therefore change ledger fingerprint without transaction changes—the sidecar
decision correctly rejects extending this path.

### 2.3 Make does not declare rate input

The Python CLI exposes `--fx-rates` and `--base-currency`, but live
`run-ingest` does not pass them. Normal Make runs omit rates, while ad hoc direct
invocation can consume them without making the dependency or input identity
visible. A future valuation stage must be explicit and must not become a
dependency of `run-canonical` or `run-full`.

### 2.4 Existing manifests hash outputs, not rate inputs

The ingest manifest records only an FX-present boolean and base currency; its
input list does not bind rate path/SHA/source/series/coverage/policy.
`artifact_from_path` already computes full-file SHA, size, row count, structure,
and contract for outputs, so hashing machinery is reusable for local inputs.

Generic stage manifests allow extra properties, but that flexibility does not
validate valuation-specific counter equations or provenance. `artifacts.jsonl`
and stage manifests also add current timestamps, making them useful audit
indexes but intentionally non-byte-deterministic.

### 2.5 Run IDs and `latest`

Make supplies a scope-qualified timestamp run ID. Path inference recognizes
only bare timestamp directory names, so ad hoc reruns that omit explicit
`--run-id` can lose the intended scoped identity.

Latest-link replacement is atomic per link, but multiple bases are updated
sequentially and may point at different generations after interruption. No
historical explanation may cite `latest`; valuation evidence must record a
concrete relative path and verified hash.

## 3. Hidden coupling and failure modes

### 3.1 Filename/version is not immutability

`ccl_2026.csv` can be replaced. The run must compute and verify SHA-256 before
parsing. SHA alone is also insufficient: metadata must state schema, source,
series, quote convention, unit, timezone/date meaning, coverage and acquisition
identity.

### 3.2 Duplicate observation needs a closed definition

Exact duplicates, same-key identical rates, conflicting rates, revisions, and
multiple instruments are different evidence situations. V1 should reject every
duplicate canonical observation key—including identical duplicate rows—unless
an approved revision dimension is part of that key. Silent deduplication
destroys input-quality evidence.

### 3.3 Coercive parsing prevents reconciliation

Record:

```text
rate_raw_observation_count
rate_observation_count              # accepted canonical observations
rate_rejected_observation_count
```

and require `raw = accepted + rejected`. Rejections retain reason/count rather
than disappearing. `rate_min_date` and `rate_max_date` derive only from accepted
observations.

### 3.4 Match counters require exclusive definitions

Recommended source-population partition:

```text
valuation_rows
  = native_usd_identity_rows
  + exact_matches
  + previous_available_matches
  + stale_rejections
  + missing_rates
  + unsupported_currency_rows
  + invalid_native_rows
```

- `exact_matches`: eligible non-USD rows whose observation date equals the
  transaction date;
- `previous_available_matches`: earlier accepted observation within an approved
  age limit;
- `stale_rejections`: earlier candidate exists but exceeds that limit;
- `missing_rates`: no candidate exists; excludes stale;
- `unsupported_currency_rows`: outside controlled currencies;
- native USD uses its own identity count.

Observation count is not matched-ledger-row count: many transactions may reuse
one observation.

### 3.5 Aggregate counters do not explain an individual row

Every sidecar row needs transaction date, applied rate date/value/direction,
match method, age, policy ID, rate source/series, and status. Otherwise “97
previous matches” cannot explain which observation produced USD 8,431.

### 3.6 Deterministic result differs from timestamped execution record

The sidecar needs fixed schema/column order, stable row order, canonical
date/null encoding, explicit decimal precision/rounding, stable newline and
encoding. It must be byte-identical for identical inputs.

The execution manifest may contain `generated_at`, run ID and environment data,
so its bytes may differ. Tests compare sidecar bytes and nonvolatile manifest
semantics while explicitly excluding volatile fields.

### 3.7 Policy and code identities can lie

A free-text `valuation_policy_id` is insufficient. Record the hash of the
resolved immutable policy definition, and require a new ID when its content
changes. Record `code_revision`, whether the worktree was dirty, and an
implementation/schema version. Precision, intermediate/final rounding, and
presentation rounding also belong to the approved policy.

### 3.8 Reruns can overwrite history

Writing into an existing run directory can replace a sidecar while stale
append-only artifact records survive. A valuation stage must refuse overwrite
unless the existing SHA is identical, or use a valuation-instance directory
keyed by ledger/rate/policy/implementation identity.

### 3.9 Scope/population mismatch remains reproducibly wrong

A deterministic valuation against the wrong recognized/all-status or
FBPM/Household ledger is still wrong. Bind exact source path, SHA, row count,
scope tag and status population.

## 4. Disagreements with the existing packet

1. Requested manifest fields alone are insufficient without ledger/output SHA,
   source population, policy hash, schema/code identity, native-USD count and
   rejected/invalid counts.
2. `rate_observation_count` must mean accepted canonical observations; raw and
   rejected counts must also be recorded.
3. A versioned filename is not immutable evidence; hash verification is
   mandatory.
4. `_load_fx_rates` is not an authoritative CCL validator because it coerces,
   drops and erases provenance.
5. Generic manifest validation cannot enforce valuation-specific equations.
6. Timestamped manifests are not deterministic artifacts; determinism belongs
   to the sidecar and nonvolatile semantics.
7. Previous-available matching remains a Matías policy choice, not an assumed
   PR1 convenience.
8. A valuation Make target must remain outside native canonical/full targets.
9. Neither rate input nor recorded provenance may use `latest`.

## 5. Recommended architectural choice

Create an explicit valuation stage consuming:

```text
immutable canonical ledger
+ immutable local CCL rate snapshot
+ immutable resolved valuation policy
```

and producing:

```text
ledger_valuation_usd_ccl.csv
valuation_manifest.json
valuation_validation.csv or .json
```

### Rate snapshot rows

```text
rate_date
source_currency
target_currency
rate_to_target
rate_source
rate_series
observation_id
published_at
acquired_at
```

Companion metadata records schema version, source/series, quote convention,
timezone, snapshot ID/creation, raw/accepted/rejected counts, min/max date and
acquisition reference. The accounting run computes the rate-data SHA itself; it
does not trust an unverified hash inside metadata.

### Valuation manifest

```text
stage=V.usd_ccl_valuation
run_id
generated_at
valuation_schema_version
valuation_policy_id
valuation_policy_sha256
source_ledger_artifact
source_ledger_sha256
source_ledger_rows
source_scope_tag
source_status_population
rate_artifact
rate_artifact_sha256
rate_source
rate_series
rate_observation_count
rate_raw_observation_count
rate_rejected_observation_count
rate_min_date
rate_max_date
valuation_rows
native_usd_identity_rows
exact_matches
previous_available_matches
stale_rejections
missing_rates
unsupported_currency_rows
invalid_native_rows
valuation_artifact
valuation_artifact_sha256
code_revision
code_dirty
implementation_id
generated_by_network_access=false
```

The offline assertion is useful evidence, but enforcement comes from accepting
local paths only and having no HTTP/client dependency in valuation code.

Define a content-derived `valuation_id` from source-ledger SHA, rate SHA, policy
SHA, valuation-schema version and implementation ID. Operational `run_id`
answers when/where; `valuation_id` answers what computation occurred.

## 6. Minimum PR boundary

PR1 contains only:

1. a versioned synthetic CCL snapshot;
2. an explicit rate-schema validator;
3. an immutable policy fixture/config with ID and hash;
4. a deterministic sidecar generator reading an existing fixture canonical
   ledger read-only;
5. valuation manifest and validation report;
6. an explicit `smoke-usd-ccl-valuation`-style target outside native run targets;
7. derived-valuation artifact contracts;
8. fail-closed duplicate, invalid quote, ledger/rate hash mismatch, counter,
   overwrite and no-network validation.

It excludes live acquisition, mutable rate `latest`, canonical changes,
semantic/metric/report outputs, cash/debt valuation, publication, and any
previous-available policy that Matías has not approved.

## 7. Tests that must fail before and pass afterward

### Rate artifact

- identical and conflicting duplicate keys fail before matching;
- nonpositive, infinite, nonnumeric, missing-date and unsupported-quote rows do
  not disappear silently;
- raw = accepted + rejected, and accepted observations define min/max dates;
- changed bytes under the same name produce a new SHA/valuation identity;
- expected-SHA mismatch fails before parsing.

### Network isolation

- URL/URI inputs are rejected;
- valuation succeeds with network APIs monkeypatched to raise;
- Make declares an explicit local snapshot and no remote/default path.

### Matching and counts

- native USD increments identity only;
- exact ARS increments exact only;
- approved previous match records actual date/age and previous only;
- stale candidate is unavailable and increments stale only;
- no candidate increments missing only;
- unsupported currency increments unsupported only;
- exclusive categories equal sidecar population;
- observation count differs correctly from ledger matches when rates are reused.

### Provenance binding

- changing ledger, rate or policy independently changes/rekeys the valuation;
- changing policy content under the same ID fails;
- recognized/all-status and scope mismatches fail;
- source path is concrete and never `latest`.

### Determinism/reruns

- identical inputs produce byte-identical sidecar;
- ledger/rate row order does not change output;
- dates, nulls, decimals, newlines and encoding are fixed;
- existing identical valuation is accepted without mutation;
- conflicting existing bytes fail rather than overwrite;
- volatile manifest fields may differ, but nonvolatile provenance and sidecar
  SHA remain identical.

### Native isolation

- canonical bytes/SHA/fingerprint and native smoke outputs remain identical;
- native targets never require rates;
- fixture valuation updates no `latest` link;
- sidecar/manifest are not classified as canonical transaction sources.

## 8. Decisions that genuinely require Matías

1. authoritative CCL series/instrument/source;
2. quote convention and authoritative direction;
3. observation-date meaning;
4. whether previous-available is allowed, calendar and maximum age;
5. behavior before/after available history;
6. historical corrections/restatement retention and labels;
7. revision/duplicate observation key and selection authority;
8. calculation, intermediate, output and presentation precision/rounding;
9. recognized versus all-status population;
10. unavailable versus explicitly partial aggregates;
11. whether caveated stale valuation is ever allowed;
12. snapshot/sidecar/manifest retention;
13. external acquisition/approval owner;
14. immutable policy governance and restatement threshold;
15. acceptance of content-derived `valuation_id` alongside `run_id`.

## Evidence map

- FX loading/matching and ingest manifest: `accounting/ledger/ingest.py`.
- Artifact hashing, generic schemas, timestamped indexes/manifests:
  `accounting/artifacts/manifest.py`.
- Run identity inference: `accounting/support/run_id.py`.
- Atomic but mutable latest links: `accounting/support/latest.py`.
- Live/smoke command plumbing: `Makefile`.
- Repository transformation-versus-intake boundary: `SYSTEM.yaml`.

## Completion record

```text
Changed: Engineer 3 reliability/reproducibility decision packet; main investigation strengthened with offline immutable snapshot, counter, identity and rerun requirements.
Accounting rule changed: None.
Fixture/test evidence: Existing code/contracts inspected; proposed contract tests only.
Run ID: N/A.
Outputs inspected: Source-controlled code and notes only.
Live inputs accessed: No.
Publication performed: No.
Totals/invariants checked: Static provenance/control-plane trace only; no live totals claimed.
Blocked accounting decision: Series, quote/date/fallback/revisions/precision/population/partials/retention/acquisition owner/policy governance/valuation identity.
Next bounded action: Matías approves the reliability policy decisions before fixture-only valuation PR1.
```
