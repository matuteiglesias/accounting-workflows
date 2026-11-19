# Stage D monthly operating statement — senior engineering design decision
Date: 2026-11-19
Status: **design reviewed, implementation NOT started**
Branch context: `refactor/cash-retirement-integration` (post validated-cash decommission)

## Inputs and reviewers
Three independent Codex Luna/medium, read-only reviews:
- accounting semantics: `/home/matias/.cache/cash-retirement-agents/stageD_finance_review_result.md`
- performance: `/home/matias/.cache/cash-retirement-agents/stageD_performance_review_result.md`
- integration, QA, release: `/home/matias/.cache/cash-retirement-agents/stageD_release_review_result.md`

Profile evidence: `~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/cash_retirement_stageD_profile_20261119.txt`.

## Decision (3/3 reviewer consensus)
Implement **Packet A**: one cached, *execution-local* aggregate per `(period, period_end, Currency)`, keyed by `(semantic_bucket, semantic_subbucket)`, inside `accounting/marts/semantic.py::build_monthly_operating_statement_from_split()`. Retain the existing public function, output schema, source split and QA writer, statement-line order, and all economic formula definitions. Use `resolve_semantic_measure` as the single measure authority: never sum generic `amount`, `net_amount` or `amount_abs` indiscriminately.

This is a local elimination of repeated pandas filters/groupbys, **not** a new mart, caching system, monetary model or dependency planner.

Current function around lines 571–727 issues 1,638 calls to `atomic_sum` and 1,729 calls to `ntx` across roughly 91 monthly/currency statement scopes. The cProfile observation was 33.021 s total (profiled Stage D), 15.263 s within the statement builder, including 6.291 s cumulative `atomic_sum` and 5.521 s cumulative `ntx`. These nested profiler times are **not additive** and cannot predict wall-clock speedup.

### Algorithm specification

Within each of the already sorted `(period, period_end, Currency)` groups, after current coercions and coverage computation:

1. Group existing split rows once by `semantic_bucket,semantic_subbucket`, with `dropna=False, sort=True`; aggregate the four current amount columns and `n_tx`. Avoid deduplicating split rows.
2. For each present category pair, resolve the **registry-approved measure** using `resolve_semantic_measure`. Retain its current fail-closed behavior for unapproved pairs **when selected**. Do not eagerly reject every unrelated/unused unknown category unless tests prove it has already been rejected; otherwise this optimization silently changes supported behavior.
3. Preserve current per-line masks as category-pair predicates/lookup definitions, not new economic rules. Use the cached sums and counts to populate the **same** existing `add_row(...)` calls and derived formulas.
4. Preserve the current raw-row semantics for:
   - `unknown_mask = bucket=="unknown" OR review_required`;
   - absolute unknown/review amount, coverage and zero-eligible ratio;
   - `unknown_out = sum(amount_out)`, falling back to absolute unknown when the sum equals zero;
   - `fx_other` residual `net_amount` and FX net;
   - all statement `source_filter`, labels, caveats, statuses, and line order.
5. Return identical statement and `build_monthly_operating_statement_qa` outputs. Avoid new durable intermediate outputs.

**Important economic constraints:** `amount_in` for revenue/funding, `amount_out` for OPEX/draws, `amount_abs` for debt/transfers, and FX subbucket-specific measures. Signed negative inflows/outflows must remain signed. `Box` is deliberately **not** a statement grouping key; it exists in the upstream split for other downstream products. Never combine currencies or lose `period_end`.

## Acceptance gates for Packet A

1. Existing golden fixture test `tests/test_semantic_statement_contract_parity.py` produces the exact expected **CSV bytes**, including row and column order, not only equal totals; the QA file must match as well.
2. Add focused synthetic tests for mixed currencies, two period ends within one period, null bucket/subbucket, malformed amounts, negative in/out, zero denominator, unknown/review outflow fallback, FX residual subbucket, both family withdrawal buckets, duplicate-looking rows, unsupported selected semantic pair, and exact `n_tx` counts.
3. Differential-test old/new implementation on deterministic randomized synthetic splits. Compare complete statement/QA rows and unknown/review coverage, not only headline values.
4. Compare the generated `monthly_operating_statement.csv` and QA on a **read-only pinned exact-run source** with output written only in an isolated temporary directory; record SHA-256 and row counts. If serialized floating-point order differs, do **not** silently regenerate the golden fixture. Investigate separately and stop for explicit review.
5. Run focused downstream suites: `test_semantic_statement_contract_parity`, `test_semantic_measure_contract`, `test_semantic_characterization`, `test_property_business_scope`, `test_management_semantic_contract_parity`, `test_governed_metrics_build_boundary`, `test_professional_atomic_measure_contract`, and appropriate report/QA tests.
6. Benchmark unprofiled **statement-only** and **Stage D** before/after on the same pinned input, same machine, separate isolated temp roots; use several repeats if time permits, report medians and raw timings. Do not mix cProfile elapsed time with real wall-clock.
7. Absolutely no writes to `out`, `latest`, source spreadsheets or public bundles; keep downstream report input schemas unchanged. Use a separate commit/PR for the optimization.

**Stop conditions:** any changed category/sign, loss of unknown/review evidence, new currency/Box amalgamation, missing QA, changed source filters/caveats/line count, unexpected float or sample-row drift, or no meaningful speed gain under matched test conditions.

## New issues surfaced by release review: separate correctness backlog

These are important but **not bundled with the optimization**, because they change failure semantics or accounting interpretation:

- **QA gate mismatch:** Make checks `monthly_operating_statement_qa.csv` existence but does not enforce `status=fail` as a hard error. Create a separate QA-release hardening packet with an explicit existing-fixture pass/fail analysis before turning it on.
- **Malformed numeric coercion:** `errors="coerce"` followed by zero fill can hide invalid source values. Add an explicit diagnostic count or fail policy separately after establishing historical impact.
- **Insufficient reconciliation checks:** QA currently checks availability and known line/quality flags, not full identities such as revenue − OPEX = net operating, coverage = net operating + funding − draws, or exact reconciliation of detailed OPEX. Add focused tests first; choose product-level QA severity separately.
- **Duplicate `tx_id` and provenance ordering:** upstream sample IDs and representative texts can depend on input order; `n_tx` is a supplied count, not a unique-tx count. Do not deduplicate split rows. Duplicate-ID detection belongs upstream, separate from statement aggregation.
- **Coverage ambiguity:** eligible absolute amount zero currently yields classification coverage 1.0; unknown-outflow fallback on zero is also intentional current behavior. Preserve initially, document for semantic review separately.
- **Semantic Stage D manifest inventory:** some semantic products are registered through a fallback rather than first-class explicit specs; do not refactor artifacts during this optimization.

## Candidate Packet B (after A proves its impact)

`accounting/marts/treasury.py::_residual_cash_outputs` repeatedly rescans actual and residual inputs by `period, Box, Currency` (~4.274 s cumulative under cProfile; nested in ~5.882 s treasury). A separate bounded preaggregation may reduce that work while retaining source transaction ID evidence, warning shares and all reconciliation/QA thresholds. Do not start B before A is benchmarked.

The review also noted that other semantic outputs perform grouped aggregations and sample selection, but optimization should follow measured residual costs, not speculative work.

## Review status
All three agents operated read-only and produced no code changes or production outputs. One agent's attempted targeted pytest was blocked because its read-only sandbox had no usable temporary directory; therefore **these reviews are designs, not test verification**. The implementing agent must run the gates in a writable isolated worktree.
