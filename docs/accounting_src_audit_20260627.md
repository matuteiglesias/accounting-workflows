# Accounting source architecture audit — 2026-06-27

## 1. Executive summary

The repository is organized as a Python accounting spine, but the package is rooted at `accounting/` rather than `src/accounting/`. The official path is documented as ingesting a Google Sheet or fixture into `ledger_canonical.csv`, materializing period/cash/party/box CSVs, building views, resolving internal debt, building a hard-coded metric registry and metric values, producing metric-view CSVs and drilldowns, then rendering a current human HTML report and publishing a frontend snapshot.

The current design already contains a canonical ledger, a metric registry, metric values, validation reports, human report tables, and publish/export layers. However, several accounting contracts are currently embedded in Python code rather than in external registry/config files. The biggest conceptual mismatch is that the current `IS` namespace explicitly includes contributions (`IS.CONTRIB.TOTAL`) and derives `IS.INCOME.TOTAL = IS.RENT.TOTAL + IS.CONTRIB.TOTAL`, so operating income is mixed with family funding in the current Income Statement contract. This audit does not change that behavior.

## 2. File/module map

| Path | Purpose | Reads | Writes | Key functions/classes | Accounting concepts implemented | Confidence |
|---|---|---|---|---|---|---|
| `README.md` | Top-level run documentation and official pipeline description. | N/A | N/A | N/A | Ingest → canonicalization → materialization → views → metrics → human balance output. | High |
| `Makefile` | Main orchestration surface for smoke/run modes. | env vars, fixture, Google Sheet, run artifacts | `out/run/accounting/<run_id>`, `out/metrics/<run_id>`, `out/debt_resolution/<run_id>`, `out/human_reports/<run_id>` | `run-ingest`, `run-materialize`, `run-views`, `run-debt`, `run-debt-views`, `run-metrics`, `run-human-report`, `publish-latest` | Pipeline sequencing, required artifacts, checks, latest symlinks. | High |
| `accounting/config.py` | Dataclass config loader with YAML/JSON/env overrides. | `accounting/config.yaml`, legacy `src/accounting/config.yaml`, env | stdout summary only | `Config`, `load_config`, `summary` | Output dirs, required report files, fixture, Google Sheet settings. | Medium |
| `accounting/contracts/models.py` | Pydantic/domain models. | N/A | N/A | `Currency`, `TxStatus`, `Money`, `Ledger`, `Transaction` | Ledger/transaction contract concepts. | Medium |
| `accounting/ledger/ingest.py` | Source ingest and canonical ledger construction. | fixture CSV/parquet or Google Sheet tab | `ledger_canonical.csv`, `ledger_anomalies.csv`, `manifest.json`, stage manifest | `read_sheet_to_df`, `build_ledger_base`, `build_stable_ledger_snapshot`, `compute_ledger_fingerprint`, `main` | Canonical long ledger, column normalization, money/date coercion, party mapping, optional FX, anomalies, tx IDs. | High |
| `accounting/core/timeseries.py` | Shared time-series aggregation utilities. | DataFrames | DataFrames | `aggregate_per_flow`, `expand_party_rows`, `aggregate_per_party`, `compute_daily_cash_position`, `compute_loans_time` | Flow aggregation, party expansion, daily cash stocks, loan time series. | High |
| `accounting/stage_d/materialize.py` | Stage D CSV materialization. | `ledger_canonical.csv`, optional loan register | per-flow, per-party, daily cash, box balance, box flow balance, loans, manifest/partitions | `materialize_per_flow`, `materialize_per_party`, `materialize_daily_cash`, `materialize_box_balance_time_long`, `materialize_box_flow_balance_time_long`, `materialize_all` | Classified/normalized analytical tables, cash balance, box-level flow/balance. | High |
| `accounting/views.py` | View layer from Stage D/legacy reports to decision-grade CSV marts. | Stage D outputs, ledger, optional legacy `reports/` files | `views/*.csv`, `views_sanity.json`, stage manifest | `load_reports_folder`, `build_v_cashflow_monthly`, `build_v_contributions_monthly`, `build_v_opex_category_monthly`, `build_party_timeseries_view`, `export_views` | Cashflow views, contribution mart, opex mart, party balances, sanity checks. | High |
| `accounting/debt/resolve.py` | Internal debt resolver. | canonical ledger or Google Sheet | `debt_open_items.csv`, allocations, repayment events, timeline, reconciliation | `OpenItem`, `Allocation`, `RepaymentEvent`, `TimelineEvent`, `build_open_items`, `build_repayments`, `resolve_repayments`, `main` | Internal debt contracts, open items, repayments, borrower/lender status. | High |
| `accounting/debt/balance_views.py` | Debt stock view builder. | `debt_open_items.csv` | daily/monthly/quarterly/yearly debt balances, manifest | `build_debt_balance_daily`, `_last_snapshot_by_period`, `write_outputs`, `main` | Open principal/interest/total balances by debtor/creditor/currency. | High |
| `accounting/metrics/registry.py` | In-code metric registry definition and CSV/parquet registry I/O. | optional registry CSV/parquet when loaded | `metric_registry.csv` when saved by build | `MetricSpec`, `default_metric_specs_v1`, `registry_from_specs`, `normalize_registry`, `load_registry`, `save_registry` | Metric contract fields, IS/BS metric IDs, agg rules, source layers, builder keys. | High |
| `accounting/metrics/builders.py` | Leaf metric builder functions. | `MetricsContext`: ledger, per-flow, daily cash, views, debt balances | metric-value DataFrames | `build_is_rent_total`, `build_is_contrib_total`, `build_is_opex_total`, `build_is_draws_personal`, `build_bs_cash_*`, `build_bs_debt_*`, `BUILDER_REGISTRY` | Flow metrics, stock metrics, debt metrics, hard-coded filters for rent/contributions/opex/draws/cash/debt. | High |
| `accounting/metrics/derive.py` | Derived metric formulas. | metric values DataFrame | metric values DataFrame | `derive_sum_components`, `derive_formula_subtract`, `derive_default_v1` | `IS.INCOME.TOTAL`, net-after-costs, net-post-draws, BS cash total. | High |
| `accounting/metrics/build.py` | Metric pipeline CLI and exports. | run root, views, debt artifacts | `metric_registry.csv`, `metric_values.csv`, validation, build manifest, wide/statement/metric-view/drilldown outputs | `load_context`, `select_builder_keys`, `build_wide_views`, `build_statement_views`, `build_metric_view_exports`, `main` | Metric contract production, statement exports, monthly narrative views, debt/cash/contrib/opex rollups. | High |
| `accounting/metrics/validate.py` | Metric QA checks. | registry and metric values | `validation_report.csv` | `check_metric_values_unique`, `check_registry_metric_ids_unique`, `check_leaf_builder_keys_present`, `check_metric_ids_known`, `check_sum_identity`, `check_formula_subtract_identity` | Uniqueness, registry coverage, builder coverage, sum/formula identities. | High |
| `accounting/metrics/views.py` | Metrics-oriented rollup builders from ledger. | `ledger_canonical.csv` | DataFrames consumed by `build_metric_view_exports` | `build_income_statement_monthly_last6`, `build_flow_rollup_last_n_months`, `build_draws_discipline_monthly_last6` | Last-N-month income, rent, flow, and draw rollups; noise floor/status filtering. | High |
| `accounting/metrics/drilldown.py` | Per-metric detail drilldown artifacts. | ledger, per-flow M, opex view, metric values | `metric_drilldowns/metric_drilldown_index.csv`, detail CSVs, manifest | `build_metric_drilldown_artifacts`, `drilldown_lookup`, `supported_metric_ids` | Source traceability from metric values to ledger/view details. | Medium |
| `accounting/metrics/io.py` | Metric value schema and table I/O helpers. | CSV/parquet | CSV/parquet | `MetricsContext`, `ensure_metric_values_schema`, `build_metric_frame`, `concat_metric_frames` | Metric value contract columns and context object. | High |
| `accounting/human/tables.py` | Registry of human report tables and table builders. | metrics dir files and metric views | report table DataFrames | `HumanTableSpec`, `default_human_table_specs_v1`, `load_human_tables_context`, table builders | Human narrative table set, liquidity/debt/income/flows/QA table inventory. | High |
| `accounting/human/document.py` | Current human HTML report renderer. | metric registry, metric values, validation, metric views, drilldowns | `balance_humano_v2.html`, table CSV/HTML fragments, `story_manifest.json` | `build_human_balance_report`, `build_summary_kpis`, `ensure_required_metric_views`, `main` | Human report UI, KPI cards, drilldown links, manifest. | High |
| `accounting/human/front.py` | Front-oriented experimental renderer. | run root, metrics dir, metric views, drilldowns | front report artifacts | `Front*` dataclasses, `load_front_data_context`, narrative block functions | Presentation/narrative layer; should not own accounting calculations. | Medium |
| `accounting/human/reports.py` | Thin compatibility wrapper. | CLI args | delegates outputs | `main` | Human report entrypoint alias. | Medium |
| `accounting/publish/latest.py` | Snapshot/publish layer. | latest run/metrics/human/debt artifacts | `public/accounting/latest` snapshot/manifest | `publish_report`, `publish_selected_files`, `publish_metrics`, `publish_debt`, `build_surface_manifest`, `main` | Frontend/export handoff. | High |
| `accounting/publish/manifest.py` | Frontend snapshot manifest builder. | artifact paths | manifest object | `build_frontend_snapshot_manifest` | Export metadata. | Medium |
| `accounting/viz/plots.py` | Legacy/ad hoc plots. | view CSVs such as `renta_pivot.csv` | image files via matplotlib | `plot_renta_series`, `plot_fondos_heatmap`, `plot_party_balance`, `main` | Visualization only. | Medium |
| `accounting/artifacts/manifest.py` | Stage/artifact manifest support. | artifact files | stage manifests and `artifacts.jsonl` | `artifact_from_path`, `write_stage_manifest`, `append_artifacts` | Artifact inventory, schemas/hints, hashes. | High |
| `scripts/check_ingest.py`, `scripts/check_materialize.py` | Makefile checks. | run outputs | CLI status/errors | script main logic | Data quality gates for ingest/materialize. | Medium |
| `notes/*` | Existing architecture/runbook docs. | N/A | N/A | N/A | Current-state, contracts, runbooks, evidence maps. | Medium |

## 3. Data flow map

Current apparent pipeline:

```text
fixture CSV/parquet or Google Sheet "C. Long Ledger"
  -> accounting.ledger.ingest.build_ledger_base / build_stable_ledger_snapshot
  -> out/run/accounting/<RUN_ID>/ledger_canonical.csv
  -> accounting.stage_d.materialize.materialize_all
  -> per_flow_time_long.freq=M.csv
  -> per_party_time_long.freq=M.csv
  -> daily_cash_position.csv
  -> box_balance_time_long.freq=M.csv
  -> box_flow_balance_time_long.freq=M.csv
  -> accounting.views.export_views
  -> views/v_cashflow_monthly.csv
  -> views/v_contributions_monthly.csv
  -> views/v_opex_category_monthly.csv
  -> views/party_balance_*.csv and views_sanity.json
  -> accounting.debt.resolve + accounting.debt.balance_views
  -> out/debt_resolution/<RUN_ID>/debt_*.csv and debt_balance_*.csv
  -> accounting.metrics.build
  -> out/metrics/<RUN_ID>/metric_registry.csv
  -> out/metrics/<RUN_ID>/metric_values.csv
  -> validation_report.csv, build_manifest.json, statement/wide views, metric_views/*.csv, metric_drilldowns/*
  -> accounting.human.document
  -> out/human_reports/<RUN_ID>/balance_human_v2/balance_humano_v2.html, tables, html fragments, story_manifest.json
  -> accounting.publish.latest
  -> public/accounting/latest snapshot for frontend/export consumers
```

The canonical ledger is already the audit source for most downstream layers, but some debt logic can read the Google Sheet directly and metric-view exports re-load the ledger for rollups. Legacy report loading still exists as optional fallback in `accounting.views`.

## 4. Metric registry audit

### Registry location and fields

The active default registry is code-defined in `accounting/metrics/registry.py` by `default_metric_specs_v1()`. It can be normalized/loaded/saved as CSV or parquet using `load_registry()` and `save_registry()`. Registry columns are:

`metric_id`, `statement`, `section`, `label`, `agg_rule`, `is_leaf`, `source_layer`, `builder_key`, `parent_metric_id`, `display_code`, `sort_key`, `currency_mode`, `status`, `notes`.

### Namespaces currently used

Only `IS.*` and `BS.*` are active in the default registry. There are no active `CF.*`, `ID.*`, `FUND.*`, `DIST.*`, `COV.*`, or `HUMAN.*` metric namespaces in the current registry. Some desired concepts exist as sections or report/table groups rather than namespaces, especially contributions, debt, cash/debt coverage, and human/QA tables.

### Active metric IDs and treatment

| Metric ID | Statement/section | Label | Agg rule | Leaf? | Source/builder | Current treatment |
|---|---|---:|---|---:|---|---|
| `IS.RENT.TOTAL` | IS/RENT | Renta total | `sum_components` | no | derived; no active builder in registry | Derived flow; comments show historical CABA/Torcuato leaf metrics are disabled, but builder `build_is_rent_total` exists and is not selected by the registry. |
| `IS.CONTRIB.TOTAL` | IS/CONTRIB | Contribuciones totales | `sum` | yes | `v_contributions_monthly` / `build_is_contrib_total` | Flow/funding metric inside IS. |
| `IS.INCOME.TOTAL` | IS/INCOME | Ingresos totales | `sum_components` | no | derived | Derived flow: rent + contributions. Red flag: mixes operating rent with contributions/funding. |
| `IS.OPEX.TOTAL` | IS/OPEX | Costos operativos totales | `sum` | yes | `v_opex_category_monthly` / `build_is_opex_total` | Flow expense/outflow. |
| `IS.NET.AFTER_COSTS` | IS/RESULT | Neto después de costos | `formula` | no | derived | Derived result: income total - opex total. Because income includes contributions, this is not pure operating result. |
| `IS.DRAWS.PERSONAL` | IS/DRAWS | Retiros personales | `sum` | yes | ledger / `build_is_draws_personal` | Flow distribution/withdrawal-like metric inside IS. |
| `IS.NET.POST_DRAWS` | IS/RESULT | Neto después de retiros | `formula` | no | derived | Derived result after personal draws; currently still in IS. |
| `BS.CASH.FB` | BS/CASH | Fondos FB al cierre | `last` | yes | daily cash / `build_bs_cash_fb` | Stock, last-of-period. |
| `BS.CASH.PM` | BS/CASH | Fondos PM al cierre | `last` | yes | daily cash / `build_bs_cash_pm` | Stock, last-of-period. |
| `BS.CASH.TOTAL` | BS/CASH | Activos líquidos totales | `sum_components` | no | derived | Derived stock from FB + PM cash. |
| `BS.DEBT.PM_TO_MI.OPEN` | BS/DEBT | Deuda PM con MI abierta | `last` | yes | debt balance / builder | Stock exposure. |
| `BS.DEBT.PM_TO_PRIMOS.OPEN` | BS/DEBT | Deuda PM con Primos abierta | `last` | yes | debt balance / builder | Stock exposure. |
| `BS.CLAIM.ALE_TO_PM.OPEN` | BS/CLAIM | Crédito PM contra Alejandro abierto | `last` | yes | debt balance / builder | Stock/claim exposure. |
| `BS.DEBT.PRINCIPAL.OPEN` | BS/DEBT | Principal deuda PM abierta | `last` | yes | debt balance / builder | Stock exposure. |
| `BS.DEBT.INTEREST.OPEN` | BS/DEBT | Interés deuda PM abierta | `last` | yes | debt balance / builder | Stock exposure. |
| `BS.DEBT.TOTAL.OPEN` | BS/DEBT | Deuda total PM abierta | `sum` | yes | debt balance / builder | Stock built from period balance rows. Registry says `sum`, but builder aggregates periodic rows by summing counterparties/items; not a cross-period sum. This agg rule is ambiguous for a stock metric. |
| `BS.DEBT.NET_PM_POSITION` | BS/DEBT | Posición neta PM frente a deuda | `formula` | yes | debt balance / builder | Derived stock formula implemented as a leaf builder; registry `is_leaf=True` conflicts with formula-like derivation. |

### Derived formulas currently implemented

`derive_default_v1()` implements:

* `IS.INCOME.TOTAL = IS.RENT.TOTAL + IS.CONTRIB.TOTAL`.
* `IS.NET.AFTER_COSTS = IS.INCOME.TOTAL - IS.OPEX.TOTAL`.
* `IS.NET.POST_DRAWS = IS.NET.AFTER_COSTS - IS.DRAWS.PERSONAL`.
* `BS.CASH.TOTAL = BS.CASH.FB + BS.CASH.PM`.

`BS.DEBT.NET_PM_POSITION = BS.DEBT.TOTAL.OPEN - BS.CLAIM.ALE_TO_PM.OPEN` is implemented in `builders.py`, not in `derive.py`.

### Flags

* **`IS.*` includes financing-like items:** `IS.CONTRIB.TOTAL` is a contribution/funding metric and is summed into `IS.INCOME.TOTAL`; `IS.DRAWS.PERSONAL` is withdrawal/distribution-like and included in IS net-after-draws.
* **Operating result is not cleanly separated:** current `IS.NET.AFTER_COSTS` is after `IS.INCOME.TOTAL`, which includes contributions.
* **Missing/disabled rent leaves:** registry comments show disabled `IS.RENT.CABA` and `IS.RENT.TORCUATO`; exports still list many IDs not present in the active registry, including `IS.RENT.CABA`, `IS.CONTRIB.MATIAS`, `IS.OPEX.TAX`, `IS.DIVIDENDS`, etc.
* **Stock/flow ambiguity:** cash and most debt stocks correctly use `last`, but `BS.DEBT.TOTAL.OPEN` uses `sum` while representing an open stock; the sum is across counterparties/items at a period, not across time.
* **Formula/leaf ambiguity:** `BS.DEBT.NET_PM_POSITION` is `agg_rule=formula` but `is_leaf=True` with a builder.
* **Formula missing metrics risk:** `IS.INCOME.TOTAL` references `IS.RENT.TOTAL`, but active registry marks `IS.RENT.TOTAL` as derived/no-builder and its component leaves are commented out. If no external value is produced for rent total, formula validation may fill missing components as zero rather than fail hard.
* **Sign conventions partly implicit:** contributions are absolute payer-side amounts; opex uses positive `amount_out`; draws use ledger `amount` directly after text matching and may depend on input sign.

## 5. Accounting concept map

| Concept | Current implementation | File(s) | Notes |
|---|---|---|---|
| Long ledger | Canonical CSV built from fixture/Google Sheet with normalized columns, stable tx IDs, anomalies. | `accounting/ledger/ingest.py`, Make `run-ingest` | Already central, but debt resolver can also read Sheet directly. |
| Metric registry | Code-defined `MetricSpec` list, emitted as `metric_registry.csv`. | `accounting/metrics/registry.py`, `accounting/metrics/build.py` | Not currently external YAML/CSV source of truth. |
| Income statement | `IS.*` metrics plus statement exports and monthly last-6 view. | `accounting/metrics/registry.py`, `accounting/metrics/builders.py`, `accounting/metrics/derive.py`, `accounting/metrics/views.py`, `accounting/metrics/build.py` | Includes contributions and draws; not pure operating result. |
| Operating result | `IS.NET.AFTER_COSTS` and monthly P&L tables. | `accounting/metrics/derive.py`, `accounting/human/tables.py` | Polluted by `IS.CONTRIB.TOTAL` through `IS.INCOME.TOTAL`. |
| Contributions / funding | View mart and IS metric. | `accounting/views.py`, `accounting/metrics/builders.py`, `accounting/metrics/build.py`, `accounting/human/tables.py` | Semantically funding, but currently placed under `IS.CONTRIB.*`. |
| Cash flow | Stage D box balance and `v_cashflow_monthly`; flow rollup metric views. | `accounting/stage_d/materialize.py`, `accounting/views.py`, `accounting/metrics/build.py` | No `CF.*` metric namespace yet. |
| Balance sheet / close | `BS.CASH.*`, `BS.DEBT.*`, cash/debt snapshots. | `accounting/metrics/registry.py`, `accounting/metrics/builders.py`, `accounting/debt/balance_views.py`, `accounting/human/tables.py` | Proxy balance sheet exists for cash/debt only. |
| Internal debt | Open item resolver and balance views. | `accounting/debt/resolve.py`, `accounting/debt/balance_views.py`, `accounting/metrics/builders.py` | No `ID.*` namespace; debt appears under `BS.DEBT.*`. |
| Deposits / guarantee deposits | Search terms appear only in debt/rent/classification context; no explicit metric namespace found. | likely ledger `Tipo`/`Detalle` only; no dedicated implementation found | Needs confirmation with real data. |
| Human report | HTML document with table registry, KPIs, drilldown links, story manifest. | `accounting/human/document.py`, `accounting/human/tables.py`, `accounting/human/front.py` | Human UI exists as generated HTML, not notebooks. |
| Frontend/export layer | Publish latest snapshot and experimental front report. | `accounting/publish/latest.py`, `accounting/publish/manifest.py`, `accounting/human/front.py` | Mostly presentation/export, but front narrative code should remain non-accounting. |
| Data quality checks | Ingest/materialize scripts, view sanity, metric validations, human QA tables, artifact manifests. | `scripts/check_*.py`, `accounting/views.py`, `accounting/metrics/validate.py`, `accounting/artifacts/manifest.py`, `accounting/human/tables.py` | Good start; limited business-rule/accounting-policy gates. |

## 6. Outputs currently produced

| Output path/name | Producer | Likely consumer | Type |
|---|---|---|---|
| `out/run/accounting/<RUN_ID>/ledger_canonical.csv` | `accounting.ledger.ingest` | all downstream accounting stages | Analytical/audit source |
| `out/run/accounting/<RUN_ID>/ledger_anomalies.csv` | `accounting.ledger.ingest` | QA/operator | QA |
| `out/run/accounting/<RUN_ID>/manifest.json` and `meta/stage_A_ingest.json` | ingest/artifact manifest | operators/publish | QA/manifest |
| `per_flow_time_long.freq=<FREQ>.csv` | `accounting.stage_d.materialize` | views/metrics | Analytical |
| `per_party_time_long.freq=<FREQ>.csv` | materialize | views | Analytical |
| `daily_cash_position.csv` | materialize | metrics/cash snapshots | Analytical stock |
| `box_balance_time_long.freq=<FREQ>.csv` | materialize | `accounting.views` | Analytical cashflow/stock |
| `box_flow_balance_time_long.freq=<FREQ>.csv` | materialize | opex/cashflow views | Analytical |
| `loans_time_long.freq=<FREQ>.csv` | materialize | possible loan views | Analytical |
| `meta/stage_D_materialize.json`, `partitions.json` | materialize | QA/operators | Manifest/QA |
| `views/v_cashflow_monthly.csv` | `accounting.views.export_views` | metrics/human | Analytical |
| `views/v_contributions_monthly.csv` | views | metrics/human | Analytical funding view |
| `views/v_opex_category_monthly.csv` | views | metrics/human | Analytical opex view |
| `views/renta_pivot.party_currency.csv` | views | notebooks/legacy consumers | Analytical/presentation |
| `views/fondos_wide.csv` | views | notebooks/legacy consumers | Analytical/presentation |
| `views/party_balance_detailed.csv` | views | metrics/human | Analytical |
| `views/party_balance_net_wide.party_currency.csv` and `party_balance_cum_wide.party_currency.csv` | views | notebooks/legacy consumers | Analytical/presentation |
| `views/balance_by_flujo_tipo.currency_safe.csv` | views | QA/analysis | Analytical |
| `views/consolidated_balance.currency_safe.csv` | views | QA/analysis | Analytical; can hide zero-sum party effects |
| `views/upcoming_90.raw.csv` | views | operators/human review | QA/raw convenience |
| `views/views_sanity.json` | views | Make checks/operators | QA |
| `out/debt_resolution/<RUN_ID>/debt_open_items.csv` | `accounting.debt.resolve` | debt balance views/metrics | Analytical debt contract |
| `debt_allocations.csv`, `debt_repayment_events.csv`, `debt_resolution_timeline.csv`, `debt_status_reconciliation.csv` | debt resolver | QA/audit | Analytical/QA |
| `debt_balance_daily.csv`, `debt_balance_monthly.csv`, `debt_balance_quarterly.csv`, `debt_balance_yearly.csv`, `debt_balance_manifest.json` | `accounting.debt.balance_views` | metrics/human | Analytical stocks/manifest |
| `out/metrics/<RUN_ID>/metric_registry.csv` | `accounting.metrics.build` | human report/export | Accounting contract |
| `metric_values.csv` | metrics build | human report/export/notebooks | Analytical contract |
| `validation_report.csv` | metrics validation | human QA | QA |
| `build_manifest.json` | metrics build | human/report/publish | Manifest |
| `metric_values_y_wide.csv`, `metric_values_q_wide.csv` | metrics build | analysts/export | Analytical/presentation |
| `income_statement_y.csv`, `income_statement_q.csv`, `balance_cash_y.csv`, `balance_cash_q.csv`, `balance_debt_y.csv`, `balance_debt_q.csv` | metrics build | human/front/export | Presentation/statement |
| `metric_views/*.csv` including income, rent, flow, draws, debt, cash, contrib, opex rollups | metrics build | human tables/front | Presentation/narrative inputs |
| `metric_views/metric_views_manifest.csv` | metrics build | human report | Manifest |
| `metric_drilldowns/details/*.csv`, `metric_drilldown_index.csv`, `metric_drilldown_manifest.json` | metrics drilldown | human report/notebooks | Audit/QA |
| `out/human_reports/<RUN_ID>/balance_human_v2/balance_humano_v2.html` | `accounting.human.document` | human reviewers | Narrative/frontend |
| `out/human_reports/<RUN_ID>/balance_human_v2/tables/*.csv` and `html/*.html` | human document | reviewers/front | Narrative/presentation |
| `story_manifest.json` | human document | publish/front | Manifest |
| `public/accounting/latest/*` | `accounting.publish.latest` | frontend/static consumers | Export/frontend |
| Plot image files | `accounting.viz.plots` | ad hoc users | Presentation |

## 7. Desired architecture comparison

Desired target:

```text
ledger_canonical.csv
  -> metric_registry.csv / yaml
  -> metric_values_monthly.csv
  -> metric_values_quarterly.csv
  -> metric_values_yearly.csv
  -> audit tables
  -> notebooks/accounting/*.ipynb
  -> optional frontend/export artifacts
```

Current state comparison:

* `ledger_canonical.csv`: exists and is central.
* `metric_registry.csv / yaml`: CSV output exists, but default source is Python code, not an external CSV/YAML contract.
* `metric_values_monthly.csv`: **missing as a canonical output**. Current `metric_values.csv` contains Q/Y from builders; monthly views exist under `metric_views/` and ledger-derived views but not as canonical monthly metric values.
* `metric_values_quarterly.csv` and `metric_values_yearly.csv`: not separate canonical long files; current output is one long `metric_values.csv` plus optional wide Q/Y exports.
* Audit tables: partial via anomalies, views sanity, validation report, drilldown index/details, artifact manifests.
* `notebooks/accounting/*.ipynb`: no current notebook folder/files found in the inspected tree. Existing generated HTML is the primary review UI.
* Optional frontend/export artifacts: exists via human HTML and `publish/latest.py`.

### Desired notebooks

| Desired notebook | Current support | Existing inputs | Missing inputs | Logic to reuse | Do not duplicate |
|---|---|---|---|---|---|
| `00_metric_registry_audit.ipynb` | Mostly supported | `metric_registry.csv`, `metric_values.csv`, `validation_report.csv`, `build_manifest.json` | External registry source, monthly values, namespace policy table | `accounting.metrics.registry`, `accounting.metrics.validate`, human QA table builders | Registry normalization and validation formulas. |
| `01_cash_position_snapshot.ipynb` | Supported for cash only | `daily_cash_position.csv`, `BS.CASH.*`, `cash_position_monthly_last12.csv`, `balance_cash_*.csv` | Broader assets/liabilities registry if needed | `materialize_daily_cash`, cash builders, human cash tables | Cash stock last-of-period logic. |
| `02_operating_result.ipynb` | Partially supported but semantically risky | rent/opex metric views, `IS.*` metrics | Clean operating namespace excluding `FUND.*`/contrib and `DIST.*`/draws | rent/opex rollups, income statement table builders | Current `IS.INCOME.TOTAL` if it remains mixed with contributions. |
| `03_cashflow_and_funding.ipynb` | Partially supported | `v_cashflow_monthly.csv`, `v_contributions_monthly.csv`, flow rollups | `CF.*`, `FUND.*`, `DIST.*` metrics as contracts | views cashflow/contrib builders, flow rollup views | Funding classification/sign logic. |
| `04_internal_debt.ipynb` | Strong support | debt open items, balance views, `BS.DEBT.*`, debt metric views | `ID.*` namespace if desired | debt resolver, balance views, debt human tables | Open-item allocation logic. |
| `05_balance_close_proxy.ipynb` | Partially supported | `BS.CASH.*`, `BS.DEBT.*`, cash/debt snapshots | Full close pack/BS asset/liability/equity contract | cash/debt builders, cash-vs-debt table | Existing stock aggregation semantics. |
| `06_human_monthly_report.ipynb` | Inputs exist, UI currently HTML | metric views, human table CSVs, drilldowns, story manifest | Notebook scaffold and narrative cells | human table builders and report manifest | HTML renderer calculations; notebooks should call shared table builders. |
| `07_semester_close_pack.ipynb` | Partially supported | Q/Y metric values, debt/cash statements, validation/drilldown | semester-specific period logic, close checklist, review signoffs | metric values, validation, human QA tables | Existing metric build and drilldown logic. |

## 8. Risks / red flags

1. **Operating income mixed with funding:** `IS.CONTRIB.TOTAL` is under `IS` and included in `IS.INCOME.TOTAL`; this directly conflicts with the desired clean split between operating result and family financing.
2. **Draws/distributions in IS:** `IS.DRAWS.PERSONAL` and `IS.NET.POST_DRAWS` are in `IS`, whereas desired architecture calls for `DIST.*` distributions/withdrawals.
3. **No `CF.*`, `ID.*`, `FUND.*`, `DIST.*`, `COV.*`, `HUMAN.*` namespaces:** These concepts exist as views/report groups, but not as the accounting metric contract.
4. **Registry is hard-coded in Python:** The exported CSV is an output, not the source contract. This makes business review harder and couples accounting policy to code deploys.
5. **Duplicated/competing metric definitions:** Export ID lists include metrics not present in the active registry, and rent leaves are commented out while related builders/export IDs remain.
6. **Stock/flow aggregation rules are incomplete:** `last` vs `sum` is present, but there is no explicit metric type (`flow`, `stock`, `derived`, `manual`) field; `BS.DEBT.TOTAL.OPEN` uses `sum` despite stock semantics.
7. **Derived formulas tolerate missing components:** Sum/formula derivation fills missing components with zero, which can mask absent rent/debt components unless validation flags are promoted to hard failures.
8. **Sign conventions are spread across layers:** contributions become absolute payer-side amounts in views; opex is positive `amount_out`; party views trust signed materialized amounts; draws text-match ledger rows and sum raw `amount`.
9. **Business logic in report/view code:** `accounting.views`, `accounting.metrics.views`, `accounting.human.tables`, and `accounting.human.document` contain classification, filtering, and narrative table decisions. Some are analytical, not pure presentation.
10. **Debt source path split:** debt can be resolved directly from Sheet instead of only from canonical ledger. This may weaken ledger-as-audit-source discipline.
11. **Frontend/notebook UI mismatch:** HTML human report is current primary review UI; notebooks are not present, so review/narrative ownership is not notebook-centered yet.
12. **Generated artifacts are numerous:** Many CSV/HTML/JSON outputs exist without a single external canonical metric contract and without separate monthly/quarterly/yearly long metric files.

## 9. Recommended next steps

1. **Safe documentation/audit step:** Keep this audit report and add a generated artifact inventory from a representative successful run, including exact columns and sample row counts for each output.
2. **Smallest code clarification:** Add registry documentation (not behavior changes) that labels each current metric as flow/stock/derived/manual and explicitly warns that `IS.CONTRIB.TOTAL` is funding-like and currently mixed into `IS.INCOME.TOTAL`.
3. **Smallest notebook scaffold:** Create `notebooks/accounting/00_metric_registry_audit.ipynb` that reads existing `metric_registry.csv`, `metric_values.csv`, and `validation_report.csv` only; no metric changes.
4. **Defer risky refactors:** Do not rename metric IDs or split namespaces until the current outputs are reproducible and stakeholders approve the target taxonomy (`IS`, `CF`, `BS`, `ID`, `FUND`, `DIST`, `COV`, `HUMAN`).

## Terminal audit summary

```text
AUDIT SUMMARY
- Current architecture: Python package rooted at accounting/ with Makefile-driven ingest -> materialize -> views -> debt -> metrics -> human report -> publish pipeline.
- Main metric source: default_metric_specs_v1() in accounting/metrics/registry.py, exported to out/metrics/<RUN_ID>/metric_registry.csv.
- Main report outputs: metric_views CSVs and out/human_reports/<RUN_ID>/balance_human_v2/balance_humano_v2.html, plus publish/latest frontend snapshot.
- Biggest mismatch: IS.INCOME.TOTAL currently includes IS.CONTRIB.TOTAL, mixing operating income with family funding.
- Recommended first change: document metric type/namespace intent and audit generated registry/metric values before changing business logic.
```
