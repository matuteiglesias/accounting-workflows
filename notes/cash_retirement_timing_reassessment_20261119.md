# Accounting cash-retirement performance reassessment — 2026-11-19

## Scope and evidence

The two measurements use the same scenario `chino_juan_no_paga_desde_2026_12`, its two variants (`base_planeada`, `chino_no_paga`), and the same pinned reference run `20261118T210059Z_FBPM`.

- Before cash-retirement: `~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/chino_timing_after_event_debt_20261118.json`
- After cash-retirement (integration): `~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/cash_retirement_recheck_20261119T011158Z.json`
- Comparative visual audit: `~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/cash_retirement_before_after_20261119.html`
- Stage D profile: `~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/cash_retirement_stageD_profile_20261119.txt`

**Do not** sum `python_subprocess` timing into parent `make_stage` timing. These are single observed runs with variable cache and process load, not statistical benchmarks.

## Measured wall time (two-variant scenario)

| Exclusive stage | Before, s | After, s | Reduction, s |
| --- | ---: | ---: | ---: |
| Stage D materialization | 53.59 | 51.01 | 2.58 |
| Debt and treasury | 67.39 | 47.71 | 19.68 |
| Metrics | 62.72 | 12.88 | **49.84** |
| Treasury HTML | 3.08 | 1.75 | 1.33 |
| Treasury PDF | 34.29 | 30.29 | 4.00 |
| **End to end** | **222.42** | **144.79** | **77.63 (34.9%)** |

Baseline variant: 142.09 → 78.13 s. Chino nonpayment variant: 80.28 → 66.62 s. The cash-retirement migration reduced the measured metric stage by **79.5%** across both variants. A separate, unprofiled pinned metrics build took about 3.74 s, versus ~22–37 s for the metrics subprocess before retirement.

**Correctness:** Both scenario variants completed offline stages and produced treasury HTML/PDF in the scenario end-to-end acceptance run. Exactly eight planned Chino rents were suppressed (ARS 1,600,000 and USD 1,520 across four months); paid-only records were not altered. Old `BS.CASH.*` and `DQ.CASH.FRONTEND_SAFE` metrics were retired without silently remapping their identities. Tests of remaining old metric rows (1,397 monthly and 431 annual) passed after excluding explicitly retired and newly introduced controls and normalizing boolean CSV casing. The report's historical cutoff still precedes the hypothetical months, so its historical totals are expected not to react to the planned suppression.

## Stage D call profile — candidate for next optimization

A **separate isolated** Stage D execution with the pinned canonical ledger (no write to production) took 33.52 s under `cProfile`; the accounting Stage D subprocess in the scenario timing audit took ~22.43 s / 18.38 s unprofiled. Do not compare profiled and unprofiled durations as equivalent benchmarks.

Key cumulative times in the profile:
- `build_monthly_operating_statement`: **15.38 s**, of which `build_monthly_operating_statement_from_split` 15.26 s.
- `build_semantic_outputs`: **11.66 s**, including a nested `build_monthly_box_treasury_flow` 5.88 s and `_residual_cash_outputs` 4.27 s.
- 1,638 `atomic_sum` calls (6.29 s cumulative), 1,729 `ntx` calls (5.52 s cumulative), and thousands of repeated pandas indexing operations during the operating-statement build.
- `daily_cash_position` outputs ~52,530 rows; its isolated log interval was ~2 seconds in this run, so it is not the first candidate despite its output size.

**Proposed next intervention:** Characterize and optimize `build_monthly_operating_statement_from_split` by pre-aggregating repeated source-backed period/category filters, leaving the existing semantic classification, financial formulas, QA checks, public mart schema, and control evidence untouched. Before coding: write fixture cases covering missing categories, currency separation, Box boundaries, inferred/realized statuses, empty buckets, and mixed cash effects; compare the complete existing operating-statement CSV and QA against a pinned exact run. Stop if a single economically meaningful value changes.

A second candidate is the semantic treasury-flow pipeline, but first identify its independent time after the operating-statement optimization. Direct PDF speed improvements are lower leverage than the repeated semantic filtering and also risk diverging from the shared report renderer.

## Delivery and remaining constraints

- Scenario runner still executes all standard mats/marts and metrics, rendering only requested treasury HTML/PDF; no request-specific DAG.
- No new global result cache, altered cash guarantees, or automatic publication.
- Optional observed balances remain a future, scoped evidence feature; they are not prerequisites for movement-based controls.
- Exact-run data and historical manifests untouched. Diagnostics live **outside** the HTML/PDF scenario outbox.
- Stage D profiling was performed with input copies under a disposable `/tmp` directory; the directory was deleted at exit.
- Property rendition's last `validated_cash_status` dependency was removed with its own targeted tests. The new opening-basis output no longer carries an obsolete validation-status column.
- The broader `accounting-docs` repository and historical report explanations may still contain old terminology and should be reviewed separately; this does not affect the currently exercised calculation pipeline.
