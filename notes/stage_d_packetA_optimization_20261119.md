# Stage D Packet A — operating statement aggregation optimization
Date: 2026-11-19
Branch: `perf/staged-semantic-statement-aggregation`
Status: **implementation and parity gates PASS**, unmerged/publish untouched at time of writing.

## What changed

`accounting/marts/semantic.py::build_monthly_operating_statement_from_split` reuses a single per-`(period, period_end, Currency)` category-pair `groupby` summary. It materializes the small result into ordered Python tuple records once per group and answers the existing amount/transaction-count requests from those records rather than repeatedly creating pandas boolean masks and grouped slices.

The semantic measure `resolve_semantic_measure` remains authoritative and is called **only for selected pairs**. No changes were made to financial formulas, source filters, statement-line definitions, classification, signs, unknown/review logic, residual FX handling, or QA production.

The tuple cache normalizes only category lookup keys containing `NaN`/`pd.NA` to `None` to preserve selection semantics for nullable-string pandas dtypes. Raw accounting data and numeric measures remain unchanged.

## Pinned reference and parity

Source exact run: `20261118T210059Z_FBPM`.
Source `monthly_flow_semantic_split.csv`: **660 rows**.
Statement output: **2,093 rows**; QA **26 rows**.

Both the optimized statement and QA are byte-identical to the previous implementation and the pinned official CSVs:

| Artifact | SHA-256, before and after |
| --- | --- |
| `monthly_operating_statement.csv` | `dbeeec7c87ada6478174bd8e02c9cfb65dd15d977c0124b5cc1fd67c7b16eb9e` |
| `monthly_operating_statement_qa.csv` | `0de57b68edf6890a9b3554b9c5b44dd78e070e3208dee73cb18bd1d9058c1445` |

A complete Stage D execution in **two independent disposable workspaces** produced **34 CSV products in both versions, with 34/34 byte-identical**. Temporary directories were removed afterward.

The adversarial synthetic reference is frozen as CSV fixtures independent of Git history:
`fixtures/semantic_statement_adversarial_{expected,qa_expected}.csv`.
It checks full statement and QA serialization for mixed currencies/period ends, null categories, review, signed values, FX residual, family withdrawal asymmetries, malformed numeric inputs, supplied `n_tx`, and source-facing fields. A separate nullable-string `pd.NA` test covers approved categories with null subbuckets; its full output was also compared independently with the original implementation during acceptance.

## Timings — unprofiled, same machine and pinned input

Statement-only, alternating versions and 3 timing cycles after warmup:
- Original: **5.26868, 5.61897, 6.44262 seconds**; median **5.61897 s**.
- Optimized: **1.07916, 1.31756, 1.73144 seconds**; median **1.31756 s**.
- **Observed median speedup: 4.265×** (~4.30 s saved per statement calculation).

Full Stage D, sequential isolated invocations:
- Original **23.46 seconds**, user CPU **18.17 s**, peak RSS **125,448 KB**.
- Optimized **15.28 seconds**, user CPU **11.62 s**, peak RSS **125,340 KB**.
- **Observed wall-time reduction: 8.18 s (34.9%); 1.535× faster**.

The cached-label `pd.NA` handling was added after this full Stage D benchmark; its direct pinned SHA comparison and separate differential regression passed. No significant numerical or structural behavior was changed by that final guard.

Data-only audit records (outside repo):
- `~/.cache/cash-retirement-agents/stageD_packetA_sequential_benchmark.json`
- `~/.cache/cash-retirement-agents/stageD_packetA_fullstage_benchmark.json`
- `~/.cache/cash-retirement-agents/stageD_packetA_baseline.json`

All times are first-pass measurements, not confidence intervals; do not extrapolate them as guaranteed latency under host contention. cProfile cumulative timings from the earlier audit are **not** used as direct wall-clock benchmark evidence.

## Constraints / follow-up

- Nothing touched production `out`, `latest`, external sheets, or publishing.
- Full run and PDF bundling not re-executed for this narrow performance packet.
- Stage D's QA-file-existence check does not ensure all QA rows pass; making QA `status=fail` a hard failure is a **separate correctness-policy change**, not part of this optimization.
- The next profiler candidate remains `accounting/marts/treasury.py::_residual_cash_outputs` (~4.27 seconds cumulative profiled). Re-profile only after real scenario runs demonstrate a remaining bottleneck.
