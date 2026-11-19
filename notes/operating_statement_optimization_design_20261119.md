# Stage D operating-statement aggregation — engineering review and implementation handoff
Date: 2026-11-19
Status: DESIGN APPROVED for a bounded optimization packet; **no implementation code changed in this review**.
Branch reviewed: `refactor/cash-retirement-integration` after validated-cash retirement.

## Evidence and measurements

- Real two-variant Chino Juan scenario audit after cash-retirement: **144.79 s** total versus **222.42 s** before. Across both variants: Stage D 51.01 s, debt/treasury 47.71 s, metrics 12.88 s, treasury HTML 1.75 s, treasury PDF 30.29 s. These are sequential observations, not controlled trials.
- Focused `cProfile` of Stage D from pinned run `20261118T210059Z_FBPM`: **33.52 s** profiled wall. Cumulative profiled time: `build_monthly_operating_statement` **15.38 s** (`build_monthly_operating_statement_from_split` **15.26 s**), `build_semantic_outputs` **11.66 s** (nested treasury generation **5.88 s**).
- Within the statement, **1,638 `atomic_sum` calls (6.29 s cumulative)**, **1,729 `ntx` calls (5.52 s cumulative)**. Each builds masks/slices; `atomic_sum` also groups selected rows and resolves semantic measures. Approx. 91 reporting groups. The profile adds overhead and nested call times must not be added as independent stages.
- Source profiling evidence: `~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/cash_retirement_stageD_profile_20261119.txt`.
- Three Codex Luna/medium read-only reviews:
  - `~/.cache/cash-retirement-agents/stageD_finance_review_result.md`
  - `~/.cache/cash-retirement-agents/stageD_performance_review_result.md`
  - `~/.cache/cash-retirement-agents/stageD_release_review_result.md`

## Decision: Packet A (first implementation)

One owner: `accounting/marts/semantic.py::build_monthly_operating_statement_from_split`. Leave all upstream classification rules, `build_semantic_outputs`, semantic-measure authority, treasury, metrics and reports unchanged.

**Algorithm**:
1. Preserve numeric coercion, `review_required` normalization, the sorted reporting universe `(period, period_end, Currency)`, the line-emission order and output schema exactly.
2. Within each reporting group `g`, build **one summary grouped by `semantic_bucket, semantic_subbucket` with `dropna=False`**, aggregating `amount_in`, `amount_out`, `net_amount`, `amount_abs`, and especially **`n_tx` (not row count)**. Prefer a local dict keyed by pairs over another layer of infrastructure.
3. Answer existing approved-category `atomic_sum` requests and transaction-count `ntx` requests from these local aggregates. **Call `resolve_semantic_measure` only for the pairs selected by an approved statement line.** Do not eagerly demand approved measures for unknown/residual pairs: the original code permits those through separate calculations, particularly `treasury_fx` residuals.
4. Initially **leave the existing raw-row calculations for `unknown_amount`, `review_amount`, `unknown_out` and `fx_other`** untouched, including `unknown_out == 0` fallback. Preserve existing `n_tx` filters even where they differ from amount filters.
5. Preserve `net_operating`, `coverage_after_draws`, FX net formulas and the single `build_monthly_operating_statement_qa()` call. No rounding or cross-currency conversion. Do not add Box to the statement grain.
6. Keep all state local to the invocation; no global, per-run, or scenario-spanning caches. Do not change generated artifacts beyond identical statement and QA.

**Important accounting nuances (do not “fix” during optimization):**
- Revenue uses `amount_in`; property OPEX and draws use `amount_out`; debt and internal transfer use `amount_abs`; approved FX subbuckets use different measures; residual FX uses `net_amount`.
- Negative signed inflows/outflows are intentionally preserved by characterized tests.
- Source split contains finer Box/party/actor dimensions, but statement intentionally rolls them up to month/currency.
- Some distribution details include both withdrawal buckets in monetary amounts but `ntx` references only `family_withdrawal_candidate`. `other_property_opex` currently counts broader property OPEX. Preserve these until a separate semantics review.
- `review_required` rows contribute to the unknown/coverage mask **but can also contribute to operating totals if their category is operating revenue/OPEX**. This conflicts with some caveat language; characterize and raise a separate accounting decision. Do not silently drop them.
- Some unknown, null or FX residual categories have no approved measure and must not trigger new errors merely because a precomputation is eager. Existing fail-closed behavior applies when an unsupported semantic pair is actually passed through an approved-measure statement line.
- The statement uses `n_tx` as a sum of already aggregated transaction counts; duplicate-looking split rows must not be deduplicated.

## Characterization and acceptance gates

- Existing byte-level anchor: `tests/test_semantic_statement_contract_parity.py`, fixtures `semantic_measure_statement_input.csv` and `semantic_measure_statement_expected.csv`. Must remain **byte-identical** (column order, row order, source_filter strings, caveats, suitability and float serialization).
- Existing tests to run: `tests/test_semantic_measure_contract.py`, `test_semantic_characterization.py`, `test_semantic_statement_contract_parity.py`, `test_property_business_scope.py`, `test_management_semantic_contract_parity.py`, relevant professional atomic measure and governed metrics tests.
- New adversarial synthetic differential tests against frozen old implementation: mixed currencies, multiple Boxes, duplicate-looking rows, multiple categories in each bucket, null bucket/subbucket, unsupported semantic categories, nonstandard FX residual, negative amounts, malformed numeric values, `review_required` true/false, zero eligible abs / 100% coverage, `unknown_out == 0` fallback, duplicate `n_tx`, different `period_end` for one month, statement line ordering. Compare **all** amounts, counts, QA, lineage, caveats and frontend suitability.
- Pinned exact-run statement and QA generation in two isolated temp directories: compare serialized CSV bytes and SHA256 where order is contractual; otherwise compare keyed rows and state any non-byte deterministic differences explicitly. Do not write to production `out` or advance `latest`.
- Minimal downstream regression: frontier, annual, professional and affected report input tables retain source/value/availability parity. Do not require PDF byte equality; avoid full PDF rendering unless a consumer contract actually changed.

## Benchmark / decision gate

Before implementation, measure direct function wall time on a real copied semantic split under current base; after optimization use the **same loaded data**. 5 warm repetitions and a few independent cold-process repetitions, using median and variability, plus peak RSS. Compare one full Stage D run in isolated temp roots (same inputs) only after statement parity passes. A 2-variant scenario benchmark is optional; avoid wasting minutes if the statement-only speedup is negligible.

A meaningful initial acceptance criterion is **at least ~30% reduction in the unprofiled statement-builder median** *and* no significant regression in Stage D total or memory; this is a decision threshold, not a promised benefit. If total Stage D saving is below ~3 s in a representative run, stop optimizations here and reassess the next constraint rather than chasing microseconds.

## Packet B (deferred)

`accounting/marts/treasury.py::_residual_cash_outputs` had 4.27 s cumulative profiled time. It loops over period × Box × Currency and repeatedly scans actual and residual data. A separate, output-parity-gated grouped-aggregation patch could eliminate repeated totals while keeping bounded top-transaction evidence. **Do not include this in Packet A.**

## Separate integrity issue

Release reviewer found that Makefile currently checks that `monthly_operating_statement_qa.csv` exists but does not fail when its `status` contains `fail`. This is a P1 failure-semantics gate and should be addressed separately, keeping accounting formulas unchanged. Other possible future semantic/quality concerns (missing numeric coerced to zero, ambiguous outflow fallback, duplicate transaction IDs, classification caveat wording) require explicit design decisions; do not bundle them into a performance optimization.

## Guardrails

No changes to the retired validated-cash authority, no alternate financial implementation, no `latest` or publication, no production runs overwritten, no private source data committed. Keep this as one small PR. The three engineering reviews were read-only; Python pytest attempts in read-only agent sandbox failed due temporary-directory permissions, **not test failures**. Tests are to be run during the implementation packet in a proper writable isolated workspace.
