# USD/CCL PR3 — explicit offline existing-run capability

Date: 2026-08-18

## Approved policy

`ccl_txn_prev_available_v1` is a management flow-valuation policy, not a native
accounting rule. Native USD uses identity. ARS uses an exact transaction-date
observation when present; otherwise it uses only the most recent prior
observation when no more than five calendar days old. The stage never selects a
future observation, interpolates, extrapolates past five days, or substitutes
zero. Missing history, stale rates, invalid amounts, and unsupported currencies
remain unavailable with explicit statuses.

## Reference artifact

`reference/fx/ccl_ars_usd.csv` is deliberately repo-tracked reference input, not
generated accounting output. It contains only the approved header because no
authoritative rate source or observations were supplied in this task. An agent
must not invent or fetch those accounting inputs. Matías must populate and commit
the file before an authoritative run. Every run records its exact SHA, source,
series, observation count, and date coverage.

## Existing-run boundary

The two explicit Make targets resolve `ledger_canonical.csv` and, for management
flows, `classification_audit.csv` below one supplied `RUN_ROOT`. They never invoke
ingest, canonical materialization, publication, latest links, human/professional
reports, cash stocks, or debt stocks.

Valuations are written below:

```text
<RUN_ROOT>/valuations/usd_ccl/<valuation_id>/
```

The ID binds ledger SHA, rate SHA, policy SHA, schema version, and implementation
identity. Reordering rate rows retains deterministic sidecar bytes but changes
the ID because the reviewed input artifact SHA changed. A corrected rate snapshot
therefore creates a sibling valuation and never overwrites the earlier evidence.

## Reconciliation evidence

The manifest and validation evidence include identity, exact, previous-available,
stale, missing-history, exact-policy-missing, unsupported, and invalid counters;
ledger/rate date bounds; maximum applied rate age; and valued-row totals. The
status counters partition every valuation row.

`valuation_coverage_by_year.csv` provides annual rows, valued rows, exact and
previous matches, stale and missing counts, unsupported/invalid counts, and a
coverage ratio. Missing values remain NA in the sidecar and flow components.

## Synthetic acceptance evidence

Tests cover Friday exact matching, Saturday/Sunday/holiday fallback, a five-day
accepted observation, a six-day stale rejection, no future selection, dates
before history, rate correction identity, shuffled observations, source SHA
binding, content-addressed reruns, and current management quarantine rules.

No live accounting data or external FX source was accessed.

## Exact authorized command after updating reference data

After populating and committing `reference/fx/ccl_ars_usd.csv`, Matías should run:

```bash
make run-usd-ccl-management-flows \
  RUN_ROOT=out/<YOUR_EXACT_RUN> \
  CCL_RATES=reference/fx/ccl_ars_usd.csv
```

Then inspect the printed content-addressed directory, especially
`valuation_manifest.json`, `valuation_validation.json`,
`valuation_coverage_by_year.csv`, `management/management_usd_ccl_flow_audit.csv`,
and `management/monthly_management_usd_ccl_components.csv`.
