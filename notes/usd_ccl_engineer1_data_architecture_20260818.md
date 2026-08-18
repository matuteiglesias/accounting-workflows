# USD/CCL Engineer 1 review — data architecture and contracts

**Status:** investigation only; no implementation or accounting rule change.
**Mandate:** decide whether USD/CCL valuation belongs in canonical ledger columns or a valuation sidecar.

## Decision

Choose **B**:

```text
ledger_canonical.csv                 # unchanged canonical transaction truth
ledger_valuation_usd_ccl.csv         # derived, versioned valuation evidence
valuation_manifest.json              # source/rate/policy/output identity
```

The sidecar is superior because a corrected CCL history is a revaluation, not a
transaction mutation. There is one important blocker to resolve in its
contract: supplied `tx_id` values are normalized but not rejected when
duplicated, so “exactly 1:1 by `tx_id`” is not yet an established invariant.
The first PR must fail closed on duplicate keys or use a separately approved
governed row identity.

## 1. Invariant being protected

Transaction identity and native accounting truth must not change when only the
valuation interpretation changes:

- both canonical ledger files retain exact native values, schema, bytes, and
  artifact identity;
- a CCL correction does not change native materializations/reconciliations or
  the fingerprint used to decide whether native ledger business content moved;
- ledger identity, rate-snapshot identity, valuation-policy identity, and
  valuation-output identity remain independently addressable;
- a valuation join never drops or multiplies canonical rows.

Both ledgers are formally `canonical_source`, transaction-grain,
`source_of_truth` artifacts. The recognized ledger is the canonical
transaction source, while the all-status ledger is scoped normalized evidence
for debt resolution. Mixing revisable external valuation into either artifact
would weaken that boundary.

## 2. Current behavior found in code

### Canonical construction and serialization

`build_ledger_base` normalizes source facts, scope, status, parties and
provenance. It retains supplied `tx_id` values and generates IDs only for
missing ones. Its returned schema is permissive: preferred columns are followed
by all remaining columns. Consequently any ingest-level projection becomes
canonical serialized content.

The CLI builds the scoped all-status frame first, filters the recognized frame
from it, and writes both `ledger_canonical_all_status.csv` and
`ledger_canonical.csv`. An ingest-level CCL correction would therefore change
both artifacts even though no transaction changed.

### Fingerprints and hashes

`build_stable_ledger_snapshot` calls `build_ledger_base`, removes only
`ingest_ts`, converts every other column to strings, sorts rows, and sorts **all
columns** alphabetically. `compute_ledger_fingerprint` hashes that complete CSV
serialization. `base_amount` already participates; future `amount_usd_ccl` and
`fx_rate_*` fields would participate too. The fingerprint API and CLI probe
also forward `fx_rates_path` and `base_currency`.

A narrower `compute_source_hash` exists and defaults to `tx_id`, `Date`, and
`amount_cents`. The coexistence of whole-dataframe fingerprint, selected-column
source hash, and full-file artifact SHA means “ledger hash” is currently
ambiguous and must be named precisely in a valuation contract.

`artifact_from_path` hashes the entire file and records size, rows, structure,
contract, and timestamps. A robust design distinguishes:

1. native-ledger business fingerprint;
2. canonical-ledger byte SHA;
3. rate-snapshot SHA;
4. valuation-sidecar SHA.

### Consumer behavior

Stage D records the canonical ledger byte hash and treats it as input. Several
native consumers explicitly select `amount` and `Currency`, so extra columns
often do not change arithmetic. That is not complete isolation: metric
drilldowns export full matching canonical rows, artifact structure hints expose
all columns, and consumers watching hashes/fingerprints observe schema or rate
changes. Debt resolution also has a path that calls `build_ledger_base`
directly.

## 3. Hidden coupling and failure modes

### A. Projection fields inside canonical ledgers

| Failure mode | Consequence |
|---|---|
| Corrected rate changes whole-dataframe fingerprint | revaluation masquerades as source-ledger mutation |
| Both recognized and all-status files are reserialized | two canonical source hashes change unnecessarily |
| Transaction truth and revisable interpretation share authority | `source_of_truth` boundary becomes mixed |
| Full-row exports inherit fields automatically | projected drilldown schema appears without deliberate contract |
| Consumers watch hashes/schema rather than selected values | downstream invalidation is broader than arithmetic dependency |
| All-status rows receive implicit valuation | recognition/scope purpose of valuation is unclear |

The advantages—no join and easy propagation—are implementation conveniences,
not stronger accounting invariants.

### B. Canonical ledger plus sidecar

| Failure mode | Required protection |
|---|---|
| duplicate supplied `tx_id` | assert non-null/unique and merge with `validate="1:1"`, or approve a stronger key |
| wrong source population | declare recognized vs all-status population explicitly |
| stale sidecar joined to another run | bind exact source SHA, business fingerprint, row count, and path |
| hidden incomplete join | equal counts plus empty anti-joins and one status row per source row |
| multiple valuation versions | bind one policy and one rate snapshot per artifact/version |
| valuation stage rewrites ingest outputs | consume an already-written canonical artifact read-only |

The sidecar hypothesis survives. Its hard problem is identity/cardinality, not
performance. An unchecked merge would be less safe than canonical columns, so
the 1:1 contract is mandatory.

## 4. Disagreements with the original packet

1. Additive canonical columns do **not** preserve ledger identity: values may be
   native-unchanged while bytes, schema, artifact SHA, stable fingerprint,
   structure hints, and exported drilldown rows change.
2. The original packet put CCL fields on the canonical ledger too early and
   understated the significance of the whole-dataframe fingerprint.
3. `tx_id` was treated as a ready sidecar key without proving uniqueness of
   supplied IDs.
4. It did not separate business fingerprint, canonical byte SHA, rate SHA, and
   valuation SHA rigorously enough.
5. Natural propagation across a canonical boundary is a hazard, not a benefit;
   an explicit valued view makes activation intentional.
6. Existing `base_amount` is evidence of the coupling to avoid, not the
   architectural template to extend.

## 5. Recommended architectural choice

Create `ledger_valuation_usd_ccl.csv` as **derived valuation evidence**, not a
canonical transaction source. It represents one valuation result per row of one
named canonical population, under one policy and one immutable rate snapshot.

The valuation manifest should record at minimum:

```text
source_ledger_relpath
source_ledger_sha256
source_ledger_business_fingerprint
source_ledger_row_count
source_population
valuation_policy_id
rate_artifact_sha256
valuation_artifact_sha256
```

The row contract includes the approved immutable join key plus
`amount_usd_ccl`, applied rate/date/source/policy, conversion status, and any
rate-age/source-reference evidence. Missing/unsupported conversions still get a
row; omission must never mean “unavailable.”

Correct historical rates should produce:

| Identity/output | Expected result |
|---|---|
| canonical ledger bytes/SHA | unchanged |
| native business fingerprint | unchanged |
| rate artifact SHA | changed |
| valuation sidecar SHA | changed |
| projected management output | changed |
| native output | unchanged |

## 6. Minimum PR boundary

PR1 is contracts, synthetic fixtures, validation, and a sidecar generator only:

1. define the sidecar artifact contract and valuation manifest;
2. define a versioned synthetic rate snapshot;
3. read an existing `ledger_canonical.csv` read-only;
4. validate source key cardinality;
5. emit exactly one status-bearing row per canonical row;
6. record and verify independent ledger/rate/valuation identities;
7. prove deterministic reruns and rate-correction isolation;
8. optionally add a new explicit fixture-only Make target.

It must not alter ingest, either canonical ledger, `base_amount`, Stage D,
semantic/debt/cash marts, metrics, reports, drilldowns, publication, or live
Make targets. It must not fetch rates over the network or consume live data.

## 7. Tests that must fail before and pass afterward

| Test | Acceptance condition |
|---|---|
| rate correction leaves canonical bytes unchanged | exact bytes/SHA/schema of both canonical ledgers are identical across rate V1/V2 |
| native business fingerprint isolation | ledger fingerprint unchanged while rate and valuation SHA change |
| exact sidecar coverage | source and sidecar counts equal; keys non-null/unique; both anti-joins empty; `validate="1:1"` succeeds |
| duplicate supplied ID fails closed | no deduplication, suffixing, row multiplication, or partial sidecar |
| wrong-ledger binding fails | SHA/fingerprint mismatch stops before totals |
| population binding is explicit | recognized sidecar cannot claim all-status coverage |
| deterministic valuation | same ledger/policy/rates produce byte-identical sidecar content |
| independent identity verification | recomputed ledger/rate/valuation hashes match manifest |
| unavailable rows retain cardinality | null projected amount plus explicit status, never dropped/zero |
| canonical schema isolation | no CCL fields appear in either canonical ledger |
| native drilldown isolation | existing drilldowns still sum native `amount` from canonical ledger |
| artifact authority | ledger remains source of truth; sidecar is derived valuation evidence |

## 8. Decisions that genuinely require Matías

1. Is `tx_id` formally unique in each canonical population, and should duplicate
   supplied IDs fail ingest or only valuation? If not, approve a stronger
   governed row identity.
2. Does v1 value only recognized `ledger_canonical.csv` (recommended), the
   all-status population, or separate artifacts for both?
3. Approve the sidecar's artifact-role/source-authority vocabulary.
4. Approve a new native-business fingerprint or a native allow-list for the
   existing fingerprint; do not silently redefine it.
5. Decide whether `base_amount` remains legacy, is deprecated, or later moves
   to the valuation architecture.
6. Decide whether corrected rates create a retained valuation version under the
   same accounting run or a distinct valuation run ID.
7. Decide whether exact source SHA, business fingerprint, or both bind a
   valuation to its canonical population (recommend both).
8. Approve the rule that unavailable/unsupported rows remain present in the
   sidecar (recommended).

## Evidence map

- Canonical ledger artifact authority: `accounting/artifacts/manifest.py`,
  `artifact_contract_for_name`.
- Construction, supplied/generated IDs, all-status/recognized writes, stable
  snapshot and fingerprint: `accounting/ledger/ingest.py`,
  `_build_tx_id`, `build_ledger_base`, `build_stable_ledger_snapshot`,
  `compute_ledger_fingerprint`, `main`.
- Narrow source hash: `accounting/support/hashing.py`, `compute_source_hash`.
- File hashes/structure hints: `accounting/artifacts/manifest.py`,
  `artifact_from_path`.
- Canonical read-through/hash behavior: `accounting/stage_d/materialize.py`.
- Explicit native ledger measure selection: `accounting/metrics/views.py`.
- Full-row native drilldown exports: `accounting/metrics/drilldown.py`.
- Direct ingest reuse by debt: `accounting/debt/resolve.py`.

## Completion record

```text
Changed: Engineer 1 architecture decision packet; original investigation recommendation corrected to sidecar-first.
Accounting rule changed: None.
Fixture/test evidence: Existing code/tests inspected; proposed contract tests only because implementation is forbidden.
Run ID: N/A.
Outputs inspected: Source-controlled code and notes only.
Live inputs accessed: No.
Publication performed: No.
Totals/invariants checked: Static trace only; no live totals claimed.
Blocked accounting decision: Join-key authority, valued population, fingerprint vocabulary, sidecar authority, version retention.
Next bounded action: Matías approves the eight decisions, then a fixture-only sidecar PR may begin.
```
