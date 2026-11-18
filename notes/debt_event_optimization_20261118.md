# Debt event-day optimization: benchmark and parity (2026-11-18)

## Scope

Module: `accounting.debt.balance_views.build_debt_balance_daily`.
Change: **compute debt stock only on economic event dates**, carry cached amounts across no-event calendar days, **continue to emit the same daily table**. This is NOT a switch to monthly-only output or a change in debt economics.

An event is:
- an obligation opening;
- an allocation date if repayment allocations exist;
- a closing date if no allocation history was passed;
- the first requested date, so custom `start_date` reconstructs any carried position.

The branch for allocations deliberately follows original behavior: if allocations are supplied, historical closing flags are not used to compute current principal.

## Source and fixed parity target

Existing exact-run source:
`out/debt_resolution/20261118T210059Z_FBPM/`
- 125 debt open items;
- 82 allocation rows;
- 1,270 days in the 2023-05-11–2026-10-31 computation;
- 42 unique obligation-opening dates, 16 allocation dates, 56 unique event days;
- 1,214 no-event days: **95.6% of days do not recompute debt-group balances**.

The output still has **7,842 daily rows**, as before.

| File | Exact SHA256 reference = optimized output |
| --- | --- |
| `debt_balance_daily.csv` | `4030378afb213193473486760406a85780e02240e4d6acd97750a8d016f3fa3e` |
| `debt_balance_monthly.csv` | `13167b9a735ef199405a11949372fcb5756455863a49d7bf7a2b7284088728ca` |
| `debt_balance_quarterly.csv` | `a1cd6edfa5b7cc2c2f07260a6f1871d3e63ad9a33b69747394ea6f83e0f50a65` |
| `debt_balance_yearly.csv` | `1cebce70119c0b7b1e6e0cd59b8fd866157a762241c575395705584603d78040` |

All four files were byte-identical; the benchmark used the exact same source and cutoff and wrote only to separate disposable temp directories.

## Unprofiled comparison (same machine, sequential runs)

| Version | Wall time | CPU user | Peak RSS |
| --- | ---: | ---: | ---: |
| Previous daily recomputation | 50.92 s | 35.72 s | 118564 KB |
| Event-day recomputation | **5.26 s** | 3.34 s | 116344 KB |

**Observed acceleration: 9.68×**, or **45.66 seconds saved for this module invocation**. Conditions may vary across future runs; these are first-pass single-shot timings, not a confidence interval. The previous 143.85-second cProfile run was instrumented and under possible contention and must not be used as a fair baseline.

## Tests and hard boundaries

- `tests/test_debt_balance_event_carry.py`: verifies recomputation only on three event dates across 90 days, carried balances, close and interest preservation.
- Existing `test_partial_debt_allocation`, `test_cutoff_pipeline_contract`, `test_canonical_debt_status_contract` and debt position authority tests passed in the first bounded suite (**21 tests**). Additional relevant regression suite (**19 tests**) passed afterward.
- No original ledger/mart output changed; production `latest` not advanced.
- Exact-byte comparison acts as the golden reference for this concrete dataset, not a general proof for all possible inputs.
- The scenario runner's existing opt-in timing audit is the next measurement instrument. A new two-variant scenario is running to identify the **next** end-to-end bottleneck, rather than optimizing on assumptions.

## Next decision

Use the new scenario-runner audit to rank `run-materialize`, `run-debt` (now optimized), `run-metrics`, treasury HTML and PDF by **exclusive stage wall time**. Do not sum nested Python subprocess times on top of parent Make stage times. After stage ranking, inspect the slowest remaining module and verify output parity before any second code change.

## End-to-end scenario re-audit (measured after event-day fix)

Opt-in audit generated on the existing Chino Juan YAML (two variants) using pinned base `20261118T210059Z_FBPM`, code with debt-event optimization. Files in:
`~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/chino_timing_after_event_debt_20261118.{json,html}`.

**Total measured wall time: 222.416 s (3m42s)**, with two fully isolated runs, all standard marts and metrics, the original treasury HTML and its PDF, and automatic temporary cleanup. OS `time` reported 222.87 s. The previous pilot's ~12 minutes was a rough wall estimate without this instrumentation, and should not be treated as a controlled before/after speedup measurement.

| Exclusive stage | Base planeada (s) | Chino no paga (s) |
| --- | ---: | ---: |
| Seed/transform cloned ledger | 0.05 | 0.09 |
| Stage D materialization | 29.45 | 24.14 |
| Debt resolution + debt/treasury marts | 44.73 | 22.66 |
| Metrics | 39.39 | 23.33 |
| Treasury HTML | 2.10 | 0.97 |
| Treasury PDF | 25.22 | 9.07 |
| **Total variant** | **142.09** | **80.28** |

Parent Make-stage times **include** nested subprocesses. Within `run-debt`, the now-optimized debt balance subprocess was **4.60 s / 3.02 s**. The remaining debt stage invokes other legitimate producers, e.g. `accounting.marts.debt` (13.60 s / 4.89 s) and `accounting.marts.treasury` (11.55 s / 6.36 s). Times are variable across variants, likely affected by warm cache and concurrent host activity: don't promise a fixed latency or add nested numbers to parent totals.

**Post-run checks:** four HTML/PDF files only in the new outbox; both HTML files byte-identical to pinned historical treasury report (future scenario was only `planeado`); no leftover `accounting-isolated-*` workspaces; pinned base `20261118T210059Z_FBPM` remained the source. During the work, the global `latest_FBPM` pointer was observed moving to `20261118T230135Z_FBPM` from a concurrent unrelated execution; do **not** say it stayed unchanged globally or attribute the movement to this runner without evidence.

## Candidate #2: eliminate repeated validated-cash selection inside metrics

**Evidence from a targeted `cProfile` of `accounting.metrics.build` using the same pinned real run (temporary output directory):**

- Profiled total **61.82 s wall / 59.39 s cumulative build**, with profiler overhead; independent unprofiled metrics run **42.05 s wall**.
- `build_metrics_frontier`: **32.01 s cumulative**; `iter_validated_monthly_cash_positions`: **30.65 s**, invoking `select_validated_cash_period` **348 times** (29.24 s cumulative).
- `build_annual_balance_dashboard`: **27.29 s cumulative**; `_rewrite_cash_metrics`: **21.34 s**; `iter_validated_annual_cash_positions`: **21.32 s**, invoking `select_validated_cash_year` **30 times** (21.07 s).
- `cash_authority._text` is called **7,504 times** (25.28 s cumulative); eligibility-mask and scope filtering repeatedly normalize the same small DataFrame columns.

**Second intervention recommended (not implemented):** pre-normalize and validate the cash input **once per metrics execution**, then share its eligible, source-backed, as-of-indexed representation across the existing monthly and annual selection primitives. Preserve the authority in `accounting/cash_authority.py`, the source-backed scope semantics in `accounting/cash_projection.py`, and the existing monthly/annual calculation contracts. Do not introduce a second cash model, a scenario cache, or a parallel materialization pipeline. First build a fixture-based characterization test for duplicate snapshots, invalid latest snapshots, incomplete periods, mixed currencies and missing validation; then compare the existing output series/marts.

**Parity qualification:** unprofiled metrics rebuilds of the pinned real run produced byte-identical `frontend_metric_series.csv`, `metric_contract_frontier.csv`, and `annual_flow_membership.csv`. `annual_balance_dashboard_metrics.csv` had the **same 466 data rows and contents but one annual 2024 USD classification-coverage row appeared in a different order**; the unified text diff showed one removed/reinserted *identical* line. Compare annual metrics by sorted record identity in future tests until that pre-existing nondeterministic row ordering is separately corrected; do not confuse row-order jitter with changed accounting values.

**Why not the other candidates first?** Stage D is ~20–29 s, and remaining debt+treasury is ~23–45 s but split among multiple producers. HTML is ~1–2 s; PDF is 9–25 s but is an explicitly requested output. `accounting.metrics.build` is the **largest individual Python module** in both variants (36.86 s / 21.99 s measured by lightweight timing) and exhibits a specific repeated-work signature. That makes a bounded shared-cash-preparation pass the clearest second code intervention.

**Gate before changing #2:** maintain `cash.position.validated` fail-closed behavior and exact currency/Box/period scope; ensure neither inferred box control nor unverified cash enters headline metrics. Tests in `tests/test_cash_position_governed.py` and `tests/test_cash_projection_shared.py`, plus sorted-row/metric-value parity against pinned run, should all pass.
