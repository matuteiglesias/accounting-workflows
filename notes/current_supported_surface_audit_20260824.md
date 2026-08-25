# Current supported surface / deletion permission audit

Status: read-only architecture census
Baseline: `main` at `06697ab8a1dd6cc2dc99488942e534cf781d397d`
Branch: `audit/current-supported-surface`

Machine-readable sheet: `notes/current_supported_surface_deletion_permission_20260824.csv`.

## Accounting invariant

Deletion is permitted only when current accounting authority, current reporting capability, diagnostic evidence, lineage/reconciliation, Box scope, currency separation, cash/debt semantics, and drilldown traceability remain available. Historical compatibility is not itself an accounting authority.

## Main finding

The current system is smaller than the compatibility code suggests, but the largest `*_legacy.py` files cannot yet be deleted wholesale. Modern facades still delegate baseline/orchestration work to them. The safe strategy is therefore to remove zero-consumer outputs and compatibility branches first, then migrate the few remaining transitive dependencies.

A table ID is not by itself proof of legacy meaning. Current and pre-governed packs can reuse the same ID; the modern distinction is often stable cell identity/source schema. Delete fallback branches only when the current supported producer guarantees the governed identity/schema.

## Current supported professional router surface

The current drilldown router declares these IDs:

- `monthly_tables_flow_bucket_all_measures`
- `monthly_tables_flow_subbucket_all_measures`
- `monthly_tables_draws_by_box_amount_out`
- `monthly_tables_draws_by_type_amount_out`
- `monthly_tables_fb_bridge_matrix`
- `monthly_tables_pm_stress_matrix`
- `monthly_tables_household_bridge_matrix`
- `monthly_tables_opex_by_type_amount_out`
- `monthly_tables_fx_treasury_compact`
- `monthly_tables_unknown_review_net_matrix`
- `monthly_tables_draws_by_type_net_amount`
- `monthly_tables_fx_treasury_all_measures`
- `monthly_tables_fx_treasury_amount_in`
- `monthly_tables_fx_treasury_amount_out`
- `monthly_tables_fx_treasury_net_amount`
- `monthly_tables_operating_statement_matrix`
- `monthly_tables_operating_statement_matrix_ars`
- `overview_balance_dashboard`
- `income_operating_statement`
- `cash_annual_box_flow_bridge_wide`
- `monthly_tables_cash_close_matrix`
- `monthly_tables_debt_activity_matrix`
- `monthly_tables_diagnostic_box_level_matrix`
- `monthly_tables_debt_position_matrix`
- `annual_cash_close_by_box_wide`
- `annual_funding_by_actor_channel_wide`
- `annual_debt_stock_by_pair_wide`
- `annual_debt_activity_by_pair_wide`

Current governed/reporting core: statement/dashboard tables, validated cash, debt position/activity, annual companions, and stable atomic draws/OPEX detail. Generic flow matrices, Box/bridge matrices, unknown-review and broad FX variants are diagnostic/compatibility surfaces and must not be mistaken for accounting authority.

## Viewer/publication result

The active `accounting-viewer` reads the publication manifest, story metadata, metrics build manifest, and manifest-selected debt tables. No current viewer load of `metric_values.csv`, `metric_registry.csv`, or `metric_views/*` was found.

The publisher exposes governed public-contract/canonical-dashboard artifacts. `income_statement_y.csv` and `balance_cash_y.csv` are copied only under `legacy_reconciliation`; release validation checks their classification if present but does not require them.

## Metrics result

`metric_values.csv` and `metric_registry.csv` remain runtime-internal dependencies, not viewer dependencies. `accounting.metrics.frontier` still reads them for frontier QA and the old metrics build still uses registry/leaf-builder machinery. They therefore require one metrics consolidation migration before deletion.

`views/v_contributions_monthly.csv` and `views/v_opex_category_monthly.csv` are hard-read by `metrics.build.load_context` and used by the legacy `IS.CONTRIB.TOTAL` / `IS.OPEX.TOTAL` leaf builders. Their accounting meanings already have governed replacements elsewhere, but the files cannot be deleted until those old leaf builders are retired/migrated.

## `accounting.marts.build` result

Actual current pipeline dependencies from this module are narrow:

- `views_sanity.json`: execution gate/observability.
- `v_contributions_monthly.csv`: old metrics leaf-builder input.
- `v_opex_category_monthly.csv`: old metrics leaf-builder input.

`party_balance_detailed.csv` is loaded optionally by metrics but no downstream `ctx.party_balance_detailed` consumer was found. No current repo/viewer consumer was found for `v_cashflow_monthly.csv`, `renta_pivot.party_currency.csv`, `fondos_wide.csv`, party-balance wide outputs, `balance_by_flujo_tipo.currency_safe.csv`, `consolidated_balance.currency_safe.csv`, or `upcoming_90.raw.csv`.

Legacy `fondos_report.csv` and `renta_*.csv` inputs are explicitly best-effort fallbacks. The current Makefile creates `reports/` only as an optional legacy-loader anchor; they are not required inputs.

## Diagnostic Box matrix

`monthly_tables_diagnostic_box_level_matrix` is diagnostic presentation over Box-level evidence, not validated cash authority. Its presentation/drilldown route can be retired now while retaining `box_balance_time_long` / `monthly_cash_close` / cash QA evidence. This also removes one important reason the old cash-control helper remains reachable.

## Legacy modules

### `accounting.metrics.annual_legacy.py`

Keep for now. The current annual facade still calls its baseline builder and helpers, then rewrites governed funding/support and cash. Public compatibility constants `ANNUAL_CONTRACT_COLUMNS`, `ANNUAL_METRICS_COLUMNS`, and `QA_COLUMNS` have a production caller in `scripts/check_contracts.py`.

### `accounting.metrics.frontier_legacy.py`

Keep for now. The current frontier facade delegates the non-cash frontier and QA helpers to it. The facade intentionally exposes no public legacy compatibility symbols.

### `accounting.professional.annual_dashboard_tables_legacy.py`

Defer whole-file deletion. The modern module still uses its generic table helpers/fallbacks. The explicit debt/write compatibility exports are repo-test-only and can be removed once tests call governed/current producers instead.

### `accounting.professional.drilldown_legacy.py`

Keep the module chassis for now, but prune it incrementally. Current production still uses its orchestrator, source discovery/read helpers, renderer/detail-writing machinery, CLI, and residual fallback router. Public `DEFAULT_TOLERANCE`, `INDEX_FILENAME`, and `row_context_id` reach the linked digest. Most underscore compatibility exports and status constants are test-only. Phase-4 facade pruning already removed the broad accidental export surface; the remaining opportunity is branch/route deletion, not another export sweep.

## DELETE NOW

- `fondos_report.csv` loader/fallback.
- `renta_*.csv` loader/fallback.
- zero-consumer marts presentation outputs listed above, while retaining Stage-D/canonical evidence.
- `legacy_reconciliation` publication copies of `income_statement_y.csv` / `balance_cash_y.csv`.
- `monthly_tables_diagnostic_box_level_matrix` presentation/drilldown route, retaining underlying evidence.
- pre-governed/minimal cash/debt fallback branches once the supported professional-input precondition is stated as the existing governed schema; their current direct evidence is compatibility tests, not a current producer.
- test-only public aliases in `annual_dashboard_tables_legacy` when the corresponding tests are modernized.

## KEEP — CURRENTLY REQUIRED

- canonical semantic, cash, debt, annual metric, lineage, reconciliation and publication artifacts.
- current governed professional statement/dashboard/cash/debt/annual companion routes.
- stable current atomic draws/OPEX drilldowns.
- `views_sanity.json`.
- the four `*_legacy.py` files as physical modules until their transitive runtime dependencies are migrated; do not confuse this with semantic authority.
- FX diagnostic/reporting capability until its total-vs-Box grain contract is closed.

## DEFER — NEEDS ONE SPECIFIC MIGRATION

- `metric_values.csv` / `metric_registry.csv`: migrate frontier QA + old leaf/derive/statement generation.
- `metric_views/*`: remove Makefile assertions and move supported notebooks to governed artifacts.
- `v_contributions_monthly.csv` / `v_opex_category_monthly.csv`: retire the old contribution/OPEX leaf builders first.
- legacy statement generation (`income_statement_*`, `balance_cash_*`, `balance_debt_*`): migrate any remaining supported notebook/reconciliation consumer to annual/frontier artifacts.
- diagnostic generic/bridge tables as linked-digest routes: split diagnostics from the presentation allowlist before deleting their drilldown routing.
- legacy derived-label formulas: require stable `derived_metric_id` for every supported derived row.
- funding/FX residual drilldown compatibility: complete typed funding/FX grain execution before deleting fallback routes.
- whole-file deletion of `annual_legacy.py`, `frontier_legacy.py`, `annual_dashboard_tables_legacy.py`, and especially `drilldown_legacy.py`: first extract/rehome the still-current baseline/orchestration helpers.

## Expected deletion shape

The largest immediate low-risk win is in `accounting.marts.build` and old input/output compatibility, not in the accounting authority layers. The largest eventual LOC win is `drilldown_legacy.py` (~3.9k LOC gross), but deleting it early would only move or break current orchestration. The correct sequence is route retirement -> supported-input hardening -> metrics consolidation -> orchestration extraction -> legacy module deletion.
