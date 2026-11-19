# Debt mart selection optimization — 2026-11-19

## Scope and observation

Pinned read-only debt input: `out/debt_resolution/20261118T210059Z_FBPM` (125 open items, 82 allocations, 260 monthly balance source rows). The monthly balance source has 185 unique `(period, debtor, creditor, Currency)` scopes after snapshot deduplication, each with exactly one candidate row. Profiled `accounting.marts.debt` took 10.38 seconds under cProfile; 185 calls to the common `select_debt_position` authority accounted for 2.69 s cumulative, with another 0.88 s in `selected_debt_position_rows`. Profiled times include instrumentation overhead.

## Bounded changes

1. Add a one-row fast path **inside the existing shared selection authority** (`accounting/debt/position_authority.py`). Preserve calendar scope, annual/monthly selection, invalid date fail-closed reasons, selected positions and all public fields. Multi-row selection remains unchanged.
2. Index the already-grouped open-item counts once in `accounting/marts/debt.py`, instead of rescanning the counts DataFrame for each monthly row. Null keys retain unmatched/zero behavior. No changes to amounts, position, activity, repayment detail, QA rules, or output schemas.

## Measured effect (one process, unprofiled, same source/output path)

| Version | Four timed observations, seconds | Median |
| --- | --- | ---: |
| Before | 5.369, 4.961, 4.283, 4.420 | 4.69 s |
| After singleton selector only | 3.736, 3.136, 3.358, 4.177 | 3.55 s |
| After both changes | 2.413, 2.662, 2.615, 2.366 | 2.51 s |

Observed medians: **2.18 s less per mart invocation** (~46% reduction), about **4.35 seconds potential** for two scenario variants if conditions are similar. This is a local, first-pass comparison, not a controlled end-to-end speedup or a guarantee under machine contention. PDF and other stages have shown large variability and are excluded from this inference.

## Parity gates

Five outputs match the original CSV bytes, including QA (comparison uses the same private output path in both runs):

- `monthly_debt_position.csv`: `35bbf08c8d46a5293475c28c2bb807fabdb5e80a97be9bec3014ecbdced3e94f`
- `monthly_debt_position_qa.csv`: `e111be035d416bac083b4a24641a0fc7344676eca17a740143ba3a6a9e18fef6`
- `monthly_debt_repayment_detail.csv`: `948b7b2a9dda70e4c8c6aaececc4ea119c571745e9459f85b3b089559ea1393e`
- `monthly_debt_activity.csv`: `e4f3d4feb3803ab7fc375e82a42c475d0453058ba53db98cb43cd90c33ae8c9e`
- `monthly_debt_activity_qa.csv`: `eccf4f3cdc57d5596ca48d80c9ac35302832a1f961230d05db9a21977360f8a6`

Differential comparison of singleton selection against baseline Git version: **350 cases, 0 mismatches**. Focused related tests: **47 passed** (existing expected Pandas warnings). Diagnostic profile: `~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/chino_debt_mart_profile_20261119.txt`. Raw timings and SHA records: `~/.cache/accounting_debt_mart_bench_20261119/`.

No real `out`, `latest`, source ledger, published artifacts or business scenarios were changed. End-to-end scenario was not rerun for this narrow performance change; its current HTML/PDF report still pertains to historical October 2026 and is not evidence of the future nonpayment scenario impact.

## Decision

Keep both narrow improvements; do not create a new data mart, shared cache, or parallel calculation pipeline. Any additional debt optimization should first measure a *new* exclusive hotspot and beat the cost of tests, review and future maintenance. Do not infer changes to the scenario's total 128–144 second wall time without a separate controlled end-to-end comparison.
