# USD/CCL PR2 — projected flow eligibility and reconciliation

Date: 2026-08-18

## Boundary and invariant

This fixture-only stage joins canonical rows, the existing native semantic audit,
and the USD/CCL valuation sidecar by an identical unique `tx_id` population. The
valuation manifest must bind the supplied ledger and sidecar SHA-256 values, and
the semantic audit's native identity fields must match the ledger row. It
does not modify canonical amounts, native classification, semantic precedence,
valuation evidence, cash/debt stocks, publication, or live runs.

The protected invariant is local fail-closed completeness: an ineligible row
makes only its own `period × Box × semantic bucket × semantic subbucket ×
valuation policy` component unavailable. It cannot invalidate an unrelated cell,
and it cannot silently contribute to a headline value.

## Old and new behavior

Old behavior: there was no projected semantic-flow eligibility artifact.

New additive behavior:

- valid positive and zero rows in the approved v1 flow buckets are eligible;
- negative native amounts retain their signed native and projected values in the
  audit but are `review_required/negative_native_amount`;
- credible FX evidence classified earlier as operating revenue, property OPEX,
  funding, withdrawal, or debt is `review_required/fx_semantic_overlap`;
- missing or unsupported valuations remain traceable as
  `unavailable_valuation`;
- unknown or native review-required semantics remain review-required;
- clean FX rows are eligible only as gross Treasury components;
- non-approved buckets are visible but excluded from v1 components.

No `abs()`, sign flip, native reclassification, FX pairing, economic FX net,
closing-stock valuation, or global reporting selector is introduced.

## Outputs

`management_usd_ccl_flow_audit.csv` is a tx-grain drilldown bridge preserving
native amount/direction/semantics beside projected amount, valuation status,
eligibility, and exclusion reason.

`monthly_management_usd_ccl_components.csv` is grouped locally by period, Box,
semantic component, and valuation policy. `value_usd_ccl` and
`reportable_value_usd_ccl` are blank unless every contributing row is eligible.
`available_value_usd_ccl` is an explicitly diagnostic subtotal of eligible rows;
it must not be presented as a complete figure. Row counters explain why a cell
is incomplete.

Both contracts are internal, derived evidence. Neither is canonical truth or an
implicitly publishable frontend artifact.

## Fixture characterization

The ten-row fixture produces:

| Characterization | Rows |
| --- | ---: |
| eligible | 7 |
| review_required_negative | 1 |
| review_required_fx_overlap | 1 |
| unavailable_valuation | 1 |

The complete Property Management rent cell reconciles to USD 110: ARS 120,000 at
ARS 1,200/USD, a zero row, and native USD 10. Property Management OPEX, funding,
draw, and clean FX reconcile to USD 1, USD 2, USD 1, and USD 1 respectively.
The negative Family Business OPEX, overlapping FX/rent, and missing-rate funding
cells are locally incomplete and have blank reportable values.

## Test and rerun implications

Tests prove identical tx coverage, source byte preservation, signed negative
traceability, local NA behavior, positive component reconciliation, clean FX
Treasury isolation, artifact authority, and deterministic output bytes. The
stage is available only through its explicit fixture smoke target and is not a
dependency of canonical, full, live ingest, latest-link, or publication targets.

No live ledger was accessed. A later read-only real-ledger characterization
requires explicit authorization and must not change this eligibility policy.
