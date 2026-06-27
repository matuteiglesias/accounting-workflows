# Backend simplification and metrics frontier audit — 2026-06-27

## 1. Executive summary

The backend has the right broad layers — ingest, Stage D materialization, marts/views, debt resolution, metrics, human reports, and publish — but the reporting semantics are currently split across too many places. The highest-impact simplification is **not** to remove legacy outputs; it is to add a small canonical monthly semantic layer that notebooks, metric builders, human reports, and public snapshots can all consume.

### Opinionated answers to the definition-of-done questions

| Question | Recommended answer |
|---|---|
| What should be the monthly source of truth for economic flows? | `monthly_flow_semantic_split.csv`, produced from `ledger_canonical.csv` plus an explicit semantic classification rules table. It should be long-format, monthly, by currency/box/semantic bucket, with source transaction counts and coverage flags. |
| What should be the monthly source of truth for cash close? | `monthly_cash_close.csv`, produced from `daily_cash_position.csv` as last-observed monthly close by box/party/currency, with explicit caveat columns distinguishing real cash from internal balance artifacts. |
| What should be the monthly source of truth for debt position? | `monthly_debt_position.csv`, a light wrapper over debt balance monthly outputs by debtor/creditor/currency/component. Keep the current debt engine intact; just expose a cleaner consumption contract. |
| Where should true opex vs family withdrawals be classified? | In a backend-owned semantic classifier immediately after canonical ledger / before metrics. Do not classify this in notebooks, wide human report tables, or frontend code. |
| Which outputs should notebooks consume? | The canonical monthly semantic tables plus `frontend_metric_series.csv` for chart-ready public metrics. Notebooks may inspect legacy tables for evidence, but they should not choose sources ad hoc. |
| Which metrics should frontends consume? | A curated `metrics frontier`: clean revenue, true property opex, operating result, funding, distributions/draws, after-draws coverage, cash close by box, debt open by counterparty, actor netting support, and data-quality metrics, each with lineage/caveats. |
| What 3–5 changes should be prioritized next? | 1) Add classification rules + `classification_audit.csv`; 2) add `monthly_flow_semantic_split.csv`; 3) add `monthly_operating_statement.csv` and rebuild monthly metrics from it; 4) add `monthly_cash_close.csv`; 5) add `metric_contract_frontier.csv` + `frontend_metric_series.csv`. |

### Key finding

The architecture is currently **artifact-rich but contract-thin**. There are many useful outputs, but the stable backend contract for monthly business concepts is not explicit enough. As a result, metrics, human reports, notebooks, and public snapshots each carry part of the semantic burden.

## 2. Current backend map

### Ledger and Stage D

* `accounting/ledger/ingest.py` canonicalizes source columns, money/date fields, parties, status, `Flujo`, `Tipo`, `Box`, and provenance into `ledger_canonical.csv`.
* `accounting/stage_d/materialize.py` writes the first analytical artifacts:
  * `per_flow_time_long.freq=<freq>.csv`: period/box/currency/`Flujo`/`Tipo` amounts.
  * `per_party_time_long.freq=<freq>.csv`: payer/receiver-expanded party balances.
  * `box_balance_time_long.freq=<freq>.csv`: box motor close-like cumulative balance using inferred `BoxParty`.
  * `box_flow_balance_time_long.freq=<freq>.csv`: box motor by `Flujo`/`Tipo`.
  * `loans_time.freq=M.csv`.
  * `daily_cash_position.csv`.

### Marts / views

* `accounting/marts/build.py` loads Stage D artifacts and optional legacy report artifacts. It builds accountant-facing views such as contributions, opex category views, party balances, and sanity reports.
* The marts loader already says Stage D artifacts are source of truth and legacy report artifacts are best-effort only. This is the right direction.

### Debt

* `accounting/debt/resolve.py` filters the ledger to `Prestamo`, `Interes`, and `Repago`, then resolves open items and repayment allocations using explicit rule versioning.
* `accounting/debt/balance_views.py` converts open items into daily/monthly/quarterly/yearly open balances by debtor/creditor/currency/component.
* Debt should not be collapsed yet. It is a specialized engine with distinct business rules.

### Metrics

* `accounting/metrics/registry.py` defines a metric registry with legacy and desired namespace metadata. It already acknowledges semantic tension: contributions are funding, draws are distributions, some legacy IS metrics are coverage metrics, and debt may belong in an internal-debt namespace.
* `accounting/metrics/builders.py` builds leaf metrics from Stage D/mart/debt sources, mostly quarterly/yearly for `metric_values.csv`.
* `accounting/metrics/views.py` builds extra last-6/last-12 views directly from `ledger_canonical.csv` and other artifacts. These are useful for humans but duplicate some metric logic.
* `accounting/metrics/build.py` orchestrates metrics, writes wide quarterly/yearly reports, statement views, metric drilldowns, and metric view exports.

### Human reports

* `accounting/human/tables.py` treats metrics and metric views as a report source and assembles many human-facing tables.
* This layer should stay presentation/report-support oriented. It should not become the semantic source of truth.

### Publish

* `accounting/publish/latest.py` copies a small bundle from `out/human_reports/latest`, `out/metrics/latest`, and `out/debt_resolution/latest` to `public/accounting/latest`.
* The publish surface is intentionally small, but its selected metric files are report/table oriented rather than a stable frontend metric series contract.

### Makefile

* The `Makefile` encodes the canonical pipeline: ingest → materialize → debt → debt views → metrics → human report → publish.
* It also carries a lot of operational knowledge: latest symlink behavior, run IDs, debt flags, metric view options, smoke/run variants, legacy compatibility targets, and public publishing.

## 3. Current artifact map

The requested generated directories (`out/run/accounting/latest/`, `out/metrics/latest/`, `out/debt_resolution/latest/`, and `public/accounting/latest/`) were not present in this checkout at audit time. Evidence below is therefore based on producer code and documented expected outputs rather than current local generated files.

### Expected accounting run artifacts

| Artifact | Producer | Current role | Notes |
|---|---|---|---|
| `ledger_canonical.csv` | `accounting.ledger.ingest` / Stage D write-through | canonical transaction base | Right source for audit detail and semantic classification. |
| `per_flow_time_long.freq=M.csv` | Stage D | monthly flow rollup by `Flujo`/`Tipo` | Useful but too raw for accountant semantics; no family draw vs true opex split. |
| `per_party_time_long.freq=M.csv` | Stage D | party-level signed movement | Useful for internal balances, but can be confused with real cash. |
| `box_balance_time_long.freq=M.csv` | Stage D | cumulative box motor | Not safe as real cash close without caveats; depends on BoxParty inference. |
| `box_flow_balance_time_long.freq=M.csv` | Stage D | box flow decomposition | Good audit input, not frontier contract. |
| `daily_cash_position.csv` | Stage D | daily cash/balance-like position | Candidate input to `monthly_cash_close.csv`; needs real-cash/internal-artifact labeling. |
| `views/v_contributions_monthly.csv` | marts | contributions monthly | Semantically funding, not operating income. |
| `views/v_opex_category_monthly.csv` | marts | opex category monthly | Current OPEX source; needs stricter semantic rules. |

### Expected metrics artifacts

| Artifact | Producer | Current role | Recommendation |
|---|---|---|---|
| `metric_values.csv` | metrics | compact metric contract | Keep, but do not ask it to carry every intermediate monthly table. Add monthly frontier series. |
| `metric_registry.csv` | metrics | metric metadata | Good base for frontier; needs public/internal flags, source table, caveats, grain. |
| `metric_values_y_wide.csv`, `metric_values_q_wide.csv` | metrics | report convenience | Legacy/report support only, not sources. |
| `income_statement_y.csv`, `income_statement_q.csv` | metrics | wide statement views | Report support only. |
| `metric_views/income_statement_monthly_last6.csv` | metrics.views | report-oriented monthly view | Useful but should be rebuilt from `monthly_operating_statement.csv`. |
| `metric_views/flow_type_rollup_m_last6.csv` | metrics.views | exploratory drilldown | Report support, not canonical. |
| `metric_views/draws_discipline_monthly_last6.csv` | metrics.views | draw discipline view | Needs canonical draw classification. |

### Expected debt artifacts

| Artifact | Producer | Current role | Recommendation |
|---|---|---|---|
| `debt_open_items.csv` | debt resolver | resolved debt items | Keep as debt engine detail. |
| `debt_repayment_events.csv` | debt resolver | repayment allocation results | Keep. |
| `debt_status_reconciliation.csv` | debt resolver | reconciliation | Keep. |
| `debt_balance_monthly.csv` | debt balance views | stock by debtor/creditor/currency/item type | Wrap as `monthly_debt_position.csv`. |
| `debt_balance_quarterly.csv`, `debt_balance_yearly.csv` | debt balance views | period-close stocks | Keep for reports/legacy metrics. |

## 4. Excessive complexity / duplication findings

### Finding 1 — Monthly metric logic is split across builders and views

`metric_values.csv` is built through registry-selected builders, while monthly last-6 views are separately rebuilt in `accounting/metrics/views.py` from the ledger. This duplicates calculation concepts such as rent, contributions, opex, and net after costs.

**Why it matters:** accountants and notebooks may audit monthly charts, but the official metric contract is mostly Q/Y. That creates a gap where monthly charts are report views rather than contract-backed metrics.

**Simplification:** make monthly semantic tables canonical, then build both metric values and metric views from them.

### Finding 2 — Wide/pivot report tables are too close to source selection

The metrics layer writes wide Q/Y statements and last-6 wide monthly tables. Those are convenient for humans, but wide tables are poor backend sources because months become columns and metric semantics are implicit in row labels.

**Simplification:** keep wide outputs for compatibility, but declare long monthly tables and `frontend_metric_series.csv` as the preferred source for automation.

### Finding 3 — Similar rollups exist in marts, metrics views, and human tables

Opex by category, contributions by party, income statement, flow rollups, cash position, and debt counterparty summaries appear in multiple layers. The repeated rollups are not all wrong, but today they are mixed with semantic selection.

**Simplification:** split the responsibilities:

1. canonical monthly semantic tables decide what a thing means;
2. metric builders aggregate metrics;
3. human tables pivot/format/report;
4. notebooks visualize and inspect, not classify.

### Finding 4 — Makefile exposes too much implicit data lineage

The Makefile is useful but encodes where latest links point, which debt directories are expected, which statuses are included, and which metric views are produced. This is operationally practical but not a semantic contract.

**Simplification:** add a generated `artifact_contracts.csv` or extend manifests so consumers can discover canonical outputs and their roles without reading the Makefile.

### Finding 5 — Path/source lookup for debt is more complex than it should be

Metrics debt loading searches multiple candidate paths around the run root. This is understandable during migration but fragile.

**Simplification:** write debt output locations into the accounting run manifest or pass an explicit debt directory to metrics. Keep candidate lookup as fallback only.

### Finding 6 — Box cash/balance concepts are overloaded

`daily_cash_position.csv`, `box_balance_time_long.freq=M.csv`, and `per_party_time_long.freq=M.csv` can all look like “cash” to analysts, but only some are cash-position candidates; others are internal movement/balance views.

**Simplification:** define `monthly_cash_close.csv` and `monthly_actor_netting_base.csv` separately, with caveat fields.

## 5. Semantic mismanagement findings

### `IS.OPEX.TOTAL`

Current OPEX is tied to `Flujo == Pagos` in monthly views and to `v_opex_category_monthly` for builders. That risks treating all payments as operating costs, including family/informal withdrawals or repayment-like movements.

**Required correction:** classify true property operating costs using a semantic bucket such as `property_opex`, with rule evidence from `Flujo`, `Tipo`, `Detalle`, payer/receiver, `Box`, and possibly tags.

### `IS.NET.OPERATING`

The registry already marks this as a desired clean operating metric: revenue minus true opex. This should become the primary operating result.

**Required correction:** build it monthly from `monthly_operating_statement.csv`, not as a shadow/derived metric over legacy income concepts.

### `IS.NET.AFTER_COSTS`

This legacy metric depends on `IS.INCOME.TOTAL`, which mixes rent and contributions. It is therefore a coverage metric, not a clean operating result.

**Recommendation:** keep it for compatibility, but mark it legacy/not-frontier or present it as `COV.NET.AFTER_COSTS_LEGACY` in documentation.

### `IS.DRAWS.PERSONAL`

The registry says this is semantically distribution/draws, not income-statement opex. Current report view detection is text-pattern based across `Tipo`, `Detalle`, `tag`, and `Lugar`, which is too fragile.

**Required correction:** canonical classification should assign `family_withdrawal` / `distribution_like_outflow` at transaction level, then monthly roll up to `DIST.DRAWS.PERSONAL`.

### `FUND.CONTRIB.TOTAL`

The registry already acknowledges that `IS.CONTRIB.TOTAL` should migrate to funding. This should not be part of operating revenue.

**Required correction:** produce funding/contributions from `monthly_flow_semantic_split.csv` with actor/counterparty support.

### `BS.CASH.*`

Cash metrics currently source from `daily_cash_position`; separate from `box_balance_time_long` and party balances. That separation is good, but it is not made explicit enough for users.

**Required correction:** canonical `monthly_cash_close.csv` should specify `position_type` (`cash_close`, `internal_balance`, `inferred_box_motor`) and `cash_suitability`.

### Debt metrics

Debt metrics are built from debt balance views, which is appropriate. The problem is mostly namespace and source clarity: these are internal balances/claims by counterparty, not operating cash or expense.

**Required correction:** expose monthly debt position by counterparty/component and use it for both Q/Y metrics and frontend debt charts.

### `daily_cash_position.csv`

This is close to cash close, but it is daily and by party/box/currency. It needs a monthly close wrapper with `as_of_date`, completeness flags, and caveats.

### `box_balance_time_long.freq=M.csv`

This is a box motor/cumulative net table, not necessarily cash. It uses inferred BoxParty from box names in current code. This is useful for reconciliation, but risky as frontend cash.

### `per_flow_time_long.freq=M.csv`

This is a raw accounting flow rollup by `Flujo`/`Tipo`. It should remain a low-level artifact, not a semantic frontier.

### `metric_values.csv`

This is a good compact metric contract, but it currently emphasizes Q/Y and mixes legacy and desired semantics. It should be supplemented by a monthly frontend/report frontier table rather than overloaded.

## 6. Source-of-truth findings

### Keep as canonical foundations

* `ledger_canonical.csv`: transaction-level foundation.
* debt resolver outputs: source of truth for debt resolution and repayments.
* `debt_balance_monthly.csv`: source of truth for open debt position until wrapped.

### Promote to canonical monthly contracts

* New `monthly_flow_semantic_split.csv`.
* New `monthly_operating_statement.csv`.
* New `monthly_cash_close.csv`.
* New `monthly_debt_position.csv`.
* New `monthly_actor_netting_base.csv`.

### Demote to report/support or legacy source

* `income_statement_monthly_last6.csv`: report support only after canonical monthly statement exists.
* `metric_values_*_wide.csv`, `income_statement_y.csv`, `income_statement_q.csv`: presentation/report support.
* `box_balance_time_long.freq=M.csv`: reconciliation/internal motor source, not cash close.
* `per_flow_time_long.freq=M.csv`: raw rollup, not semantic source.

## 7. Recommended intermediate tables

### 7.1 `monthly_flow_semantic_split.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Canonical monthly economic-flow source of truth. Separates operating revenue, property opex, family withdrawals, funding, debt movements, transfers/internal, and unknown/unclassified. |
| Grain | `period`, `currency`, `box`, `semantic_bucket`, `semantic_subbucket`, optional `counterparty`, optional `actor`. |
| Primary key | `period,currency,box,semantic_bucket,semantic_subbucket,counterparty,actor`. |
| Columns | `period`, `period_end`, `currency`, `box`, `semantic_bucket`, `semantic_subbucket`, `direction`, `amount_in`, `amount_out`, `net_amount`, `n_tx`, `source_tx_ids_sample`, `rule_id`, `classification_confidence`, `classification_status`. |
| Source inputs | `ledger_canonical.csv`; classification rules table; optional `per_flow_time_long.freq=M.csv` for reconciliation. |
| Producer | New module under `accounting/marts/semantic.py` or `accounting/stage_d/semantic.py`. Prefer marts if rules are analytical/business classification rather than raw materialization. |
| Role | Backend canonical. |

### 7.2 `monthly_operating_statement.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Clean monthly operating statement: operating revenue, true property opex, net operating, funding, distributions/draws, after-draws coverage. |
| Grain | `period`, `currency`, `metric_line`. |
| Primary key | `period,currency,metric_line`. |
| Columns | `period`, `currency`, `metric_line`, `label`, `semantic_category`, `amount`, `source_table`, `source_filter`, `n_tx`, `caveat`. |
| Source inputs | `monthly_flow_semantic_split.csv`. |
| Producer | `accounting/metrics` or `accounting/marts`; prefer `marts` for canonical table, metrics consumes it. |
| Role | Backend canonical/report support. |

### 7.3 `monthly_cash_close.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Monthly cash close by box/party/currency with explicit suitability flags. |
| Grain | `period`, `box`, `party`, `currency`, `position_type`. |
| Primary key | `period,box,party,currency,position_type`. |
| Columns | `period`, `as_of_date`, `box`, `party`, `currency`, `close_amount`, `source_table`, `source_date`, `position_type`, `cash_suitability`, `is_frontend_safe`, `caveat`, `n_source_rows`. |
| Source inputs | `daily_cash_position.csv`; optionally `box_balance_time_long.freq=M.csv` for reconciliation only. |
| Producer | `accounting/marts/cash.py` or `accounting/stage_d/materialize.py` wrapper. |
| Role | Backend canonical. |

### 7.4 `monthly_debt_position.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Stable monthly debt/claim stock by counterparty and component. |
| Grain | `period`, `debtor`, `creditor`, `currency`, `component`. |
| Primary key | `period,debtor,creditor,currency,component`. |
| Columns | `period`, `as_of_date`, `debtor`, `creditor`, `currency`, `component`, `open_amount`, `open_principal`, `open_interest`, `open_total`, `source_rule_version`, `n_open_items`, `caveat`. |
| Source inputs | `debt_balance_monthly.csv`, `debt_open_items.csv`, debt manifest. |
| Producer | `accounting/debt/balance_views.py` or a small `accounting/marts/debt.py` wrapper. |
| Role | Backend canonical for consumption; debt engine remains source of computation. |

### 7.5 `monthly_actor_netting_base.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Supports internal actor netting without confusing it with cash. Shows claims/contributions/draws/transfers by party. |
| Grain | `period`, `currency`, `actor`, `counterparty`, `semantic_bucket`. |
| Primary key | `period,currency,actor,counterparty,semantic_bucket`. |
| Columns | `period`, `currency`, `actor`, `counterparty`, `semantic_bucket`, `amount_in`, `amount_out`, `net_amount`, `n_tx`, `source_table`, `caveat`. |
| Source inputs | `ledger_canonical.csv`, `per_party_time_long.freq=M.csv`, semantic classifier. |
| Producer | `accounting/marts/semantic.py`. |
| Role | Backend canonical for internal analysis; likely not public by default. |

### 7.6 `metric_contract_frontier.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Curated metric contract for frontends/reports, with lineage and caveats. |
| Grain | One row per metric definition. |
| Primary key | `metric_id`. |
| Columns | `metric_id`, `label`, `semantic_category`, `flow_or_stock`, `grain`, `currency_mode`, `source_table`, `calculation_rule`, `frontend_suitability`, `public_flag`, `internal_flag`, `caveats`, `owner`, `status`. |
| Source inputs | `metric_registry.csv` plus manual frontier annotations. |
| Producer | `accounting/metrics`. |
| Role | Frontend/public contract metadata. |

### 7.7 `frontend_metric_series.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Long chart-ready metric series for public/frontends and professional reports. |
| Grain | `metric_id`, `period_grain`, `period`, `currency`, optional dimensions. |
| Primary key | `metric_id,period_grain,period,currency,dimension_name,dimension_value`. |
| Columns | `metric_id`, `period_grain`, `period`, `period_end`, `currency`, `value`, `dimension_name`, `dimension_value`, `source_table`, `run_id`, `as_of_date`, `caveat`, `frontend_suitability`. |
| Source inputs | monthly canonical tables and existing `metric_values.csv` for Q/Y compatibility. |
| Producer | `accounting/metrics` and copied by `accounting/publish`. |
| Role | Frontend/public artifact. |

### 7.8 `classification_audit.csv`

| Attribute | Recommendation |
|---|---|
| Purpose | Make accountant/analyst feedback auditable and actionable. Shows how each transaction or rolled group was classified. |
| Grain | Transaction-level preferred; optional grouped summary by rule. |
| Primary key | `tx_id` for transaction table; `rule_id,semantic_bucket` for summary. |
| Columns | `tx_id`, `Date`, `Currency`, `amount`, `Box`, `Flujo`, `Tipo`, `Detalle`, `payer`, `receiver`, `semantic_bucket`, `semantic_subbucket`, `rule_id`, `rule_version`, `confidence`, `classification_status`, `warning`, `review_required`. |
| Source inputs | `ledger_canonical.csv`, rules table. |
| Producer | semantic classifier. |
| Role | Backend canonical/audit support. |

## 8. Recommended metrics frontier

The frontier should be small, stable, monthly-capable, and explicit about whether a metric is frontend-safe. Existing legacy IDs can be kept; new IDs should clarify semantics.

| metric_id | Label | Semantic category | Flow/stock | Grain | Currency handling | Source table | Calculation rule | Caveats | Frontend suitability |
|---|---|---|---|---|---|---|---|---|---|
| `IS.REVENUE.OPERATING` | Operating rent/revenue | operating revenue | flow | M/Q/Y | native currency, no cross-currency sum | `monthly_operating_statement.csv` | sum `metric_line=operating_revenue` | Currently aliases rent unless other operating revenue is classified. | high |
| `IS.RENT.TOTAL` | Rent collected | operating revenue | flow | M/Q/Y | by currency | `monthly_flow_semantic_split.csv` | sum `semantic_bucket=operating_revenue` and `semantic_subbucket=rent` | Legacy ID; okay if defined strictly. | high |
| `IS.OPEX.PROPERTY` | True property operating costs | property opex | flow | M/Q/Y | by currency | `monthly_flow_semantic_split.csv` | sum outflows where `semantic_bucket=property_opex` | Requires rule review to exclude family withdrawals/debt. | high after classification audit |
| `IS.OPEX.TOTAL` | Property operating costs, legacy total | property opex | flow | Q/Y legacy, M supported | by currency | `monthly_operating_statement.csv` | alias of `IS.OPEX.PROPERTY` after migration | Legacy meaning currently broader; annotate version. | medium until migration |
| `IS.NET.OPERATING` | Net operating result | clean operating result | flow | M/Q/Y | by currency | `monthly_operating_statement.csv` | operating revenue - true property opex | Excludes funding, draws, debt principal/repayments. | high |
| `FUND.CONTRIB.TOTAL` | Funding/contributions | funding | flow | M/Q/Y | by currency | `monthly_flow_semantic_split.csv` | sum `semantic_bucket=funding_contribution` | Not operating revenue. | high with caveat |
| `DIST.DRAWS.PERSONAL` | Family withdrawals/distributions | distribution | flow | M/Q/Y | by currency | `monthly_flow_semantic_split.csv` | sum outflows where `semantic_bucket=family_withdrawal` | Needs rule evidence; do not infer only from text pattern. | high after classification audit |
| `COV.NET.AFTER_DRAWS` | Coverage after funding and draws | coverage | flow | M/Q/Y | by currency | `monthly_operating_statement.csv` | net operating + funding - draws | Coverage, not GAAP net income. | high with caveat |
| `BS.CASH.CLOSE.BOX` | Cash close by box | cash position | stock | M/Q/Y close | by currency; dimension=`box` | `monthly_cash_close.csv` | last cash close in period by box/currency | Only frontend-safe where `position_type=cash_close` and `is_frontend_safe=true`. | high if flags pass |
| `BS.CASH.TOTAL` | Total liquid cash | cash position | stock | M/Q/Y close | by currency; no FX unless explicit | `monthly_cash_close.csv` | sum frontend-safe close amounts by period/currency | Do not include internal balance artifacts. | high |
| `ID.DEBT.OPEN.BY_COUNTERPARTY` | Open debt by counterparty | internal debt | stock | M/Q/Y close | by currency; dimensions debtor/creditor | `monthly_debt_position.csv` | open_total by debtor/creditor/currency | Internal position, not expense/cash. | high |
| `ID.DEBT.PRINCIPAL.OPEN` | Open principal | internal debt | stock | M/Q/Y close | by currency | `monthly_debt_position.csv` | sum `open_principal` | Component of debt. | high |
| `ID.DEBT.INTEREST.OPEN` | Open interest | internal debt | stock | M/Q/Y close | by currency | `monthly_debt_position.csv` | sum `open_interest` | Component of debt. | high |
| `ID.NET.PM_POSITION` | PM net debt/claim position | actor netting | stock | M/Q/Y close | by currency | `monthly_debt_position.csv` + actor netting if needed | PM liabilities minus PM claims | Sign convention must be documented. | medium-high |
| `ACTOR.NET.FUNDING_DRAWS` | Actor funding less draws | actor netting | flow | M/Q/Y | by currency; actor dimension | `monthly_actor_netting_base.csv` | funding contributions - personal draws by actor | Internal analysis, likely not public. | internal only |
| `DQ.CLASSIFICATION.COVERAGE` | Classification coverage | data quality | flow/quality | M | percent plus counts | `classification_audit.csv` | classified tx count / eligible tx count | Must show denominator and unknown amount. | high |
| `DQ.UNKNOWN.AMOUNT` | Unclassified amount | data quality | flow/quality | M | by currency | `classification_audit.csv` | sum amount where status unknown/review required | Drives analyst feedback. | high |
| `DQ.CASH.FRONTEND_SAFE_COVERAGE` | Cash frontend-safe coverage | data quality | stock/quality | M | percent/count | `monthly_cash_close.csv` | safe cash rows / cash close rows | Prevents internal balances from being charted as cash. | high |

## 9. Frontend/reporting support design

### Public contract layout

Publish should include:

* `metrics/metric_contract_frontier.csv`
* `metrics/frontend_metric_series.csv`
* `metrics/classification_audit_summary.csv`
* `metrics/monthly_operating_statement.csv` if professional reports need source rows
* `cash/monthly_cash_close.csv` or a filtered frontend-safe subset
* `debt/monthly_debt_position.csv`

Do not publish transaction-level `classification_audit.csv` unless privacy-reviewed.

### Metric metadata fields to add

Add to frontier/registry:

* `public_flag`
* `frontend_suitability` (`safe`, `safe_with_caveat`, `internal_only`, `legacy_only`)
* `source_table`
* `source_grain`
* `calculation_rule`
* `lineage_inputs`
* `caveat`
* `validation_status`
* `owner`

### Professional report support

Professional reports should use:

1. `monthly_operating_statement.csv` for the clean economic statement;
2. `monthly_cash_close.csv` for cash close;
3. `monthly_debt_position.csv` for debt schedules;
4. `classification_audit_summary.csv` for review agenda;
5. wide tables only as display derivatives.

### Notebook support

Notebooks should import one backend source-selection helper that returns the canonical tables. The notebook should not choose between `daily_cash_position`, `box_balance_time_long`, and `per_party_time_long` on its own.

Recommended helper:

```python
from accounting.metrics.frontier import load_frontier_sources
sources = load_frontier_sources(run_root_or_latest=True)
```

Returned keys:

* `monthly_flow_semantic_split`
* `monthly_operating_statement`
* `monthly_cash_close`
* `monthly_debt_position`
* `frontend_metric_series`
* `metric_contract_frontier`
* `classification_audit_summary`

## 10. Priority simplification plan

### Priority 1 — Add semantic classification rules and audit output

Create an explicit rules table and classifier. The rules should map combinations of `Flujo`, `Tipo`, `Detalle`, `payer`, `receiver`, `Box`, and optional tags to semantic buckets.

**Outputs:** `classification_audit.csv`, `classification_audit_summary.csv`.

**Why now:** This directly addresses opex vs withdrawals and revenue vs funding.

### Priority 2 — Add `monthly_flow_semantic_split.csv`

Produce long monthly flow rows by semantic bucket and currency.

**Outputs affected:** future metrics, notebooks, operating statement, coverage charts.

**Why now:** It becomes the monthly source of truth for economic flows.

### Priority 3 — Add `monthly_operating_statement.csv`

Build clean monthly statement lines from semantic flows, including clean operating result and coverage after funding/draws.

**Outputs affected:** `metric_values.csv`, `income_statement_monthly_last6.csv`, human reports, frontend charts.

**Why now:** It stops duplicated P&L logic in metrics views and notebooks.

### Priority 4 — Add `monthly_cash_close.csv`

Wrap daily cash into monthly close with explicit flags and caveats. Do not use box motor as cash unless explicitly marked inferred/reconciliation.

**Outputs affected:** `BS.CASH.*`, frontend cash charts, public snapshots.

**Why now:** It prevents internal balances from being charted as real cash.

### Priority 5 — Add metrics frontier artifacts

Generate `metric_contract_frontier.csv` and `frontend_metric_series.csv`; publish them.

**Outputs affected:** frontend snapshot, reports, notebooks.

**Why now:** It gives consumers one stable contract while preserving legacy outputs.

## 11. Risks and things not to simplify yet

### Do not collapse debt logic yet

Debt resolution has explicit rule versioning, allocation behavior, status reconciliation, and principal/interest separation. It is complex but justified. Add a consumption wrapper instead of rewriting the engine.

### Do not remove legacy metrics yet

Legacy metrics such as `IS.NET.AFTER_COSTS`, `IS.CONTRIB.TOTAL`, and `IS.DRAWS.PERSONAL` are likely used by reports or snapshots. Keep them and annotate them.

### Do not move presentation logic into core modules

Human report formatting and wide table output should remain in human/report layers. The core should produce long, stable tables.

### Do not treat notebooks as source of truth

Notebooks should be feedback tools. Backend tables should absorb validated accountant feedback.

### Do not overbuild a framework

A CSV-backed rules table plus a deterministic classifier is sufficient for the next step. Avoid a DSL or workflow engine unless rules become unmanageable.

## 12. Suggested follow-up implementation tasks

1. **Create `accounting/marts/semantic.py`** with a first-pass rules-table classifier.
   * Add `semantic_bucket`, `semantic_subbucket`, `rule_id`, `confidence`, and `review_required`.
   * Write `classification_audit.csv` and `monthly_flow_semantic_split.csv`.

2. **Create `accounting/marts/cash.py`** to write `monthly_cash_close.csv` from `daily_cash_position.csv`.
   * Include `position_type`, `cash_suitability`, `is_frontend_safe`, and `caveat`.

3. **Create `accounting/marts/debt.py` or extend debt balance views** to write `monthly_debt_position.csv`.
   * Preserve existing debt outputs.
   * Add rule version and source manifest metadata.

4. **Update metrics builders to consume canonical monthly tables.**
   * Build Q/Y values by aggregating monthly canonical rows.
   * Keep existing wide outputs.

5. **Add `metric_contract_frontier.csv` and `frontend_metric_series.csv`.**
   * Start with the frontier metrics listed in this report.
   * Add public/internal suitability and caveats.

6. **Update publish bundle.**
   * Copy frontier artifacts to `public/accounting/latest`.
   * Keep existing files for compatibility.

7. **Update notebooks.**
   * Replace direct source-selection logic with canonical source loading.
   * Make notebooks show classification warnings and drilldowns rather than classify flows themselves.

8. **Add validation checks.**
   * Unknown semantic amount by month/currency below threshold.
   * OPEX excludes family withdrawals.
   * Cash frontend-safe rows exist before publishing cash metrics.
   * Debt monthly position reconciles to debt open items.

