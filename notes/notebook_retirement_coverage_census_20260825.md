# Notebook retirement coverage census — 2026-08-25

Status: pre-retirement evidence for PR #99

## Purpose

This census records what the retiring `accounting/notebooks/` layer was trying to provide and where that capability now lives in the governed accounting system.

The deletion criterion is **not** byte-for-byte or chart-for-chart equivalence. A notebook is safe to retire when it no longer owns accounting authority and its useful question can either:

1. already be answered through governed artifacts/reports/drilldowns; or
2. be named explicitly as a presentation/analysis gap that can later be rebuilt on top of governed artifacts without reviving notebook-owned semantics.

Coverage labels used below:

- **REPLACED** — the notebook's main human question has a current governed report/product path.
- **COVERED** — the underlying facts, contracts and drilldowns are governed; the old notebook presentation may differ.
- **PARTIAL** — governed facts exist, but a distinctive analytical projection or human view is not yet standardized.
- **RETIRED BY DESIGN** — the old diagnostic or fallback should not be recreated because the hardened model supersedes it.

## Current governed replacement surface

### Exact-run facts

`run-canonical` / `run-debt` materialize the authoritative monthly evidence under one exact `RUN_ID`:

- `monthly_flow_semantic_split.csv`
- `monthly_operating_statement.csv`
- `monthly_cash_close.csv`
- `monthly_debt_position.csv`
- `monthly_debt_activity.csv`
- `monthly_cash_accountability.csv`
- their corresponding QA artifacts
- canonical ledger / all-status evidence and debt-resolution artifacts

### Governed metrics

`run-metrics` produces:

- `metric_contract_frontier.csv`
- `frontend_metric_series.csv`
- `annual_balance_dashboard_metrics.csv`
- `annual_balance_dashboard_contract.csv`
- `annual_balance_dashboard_qa.csv`
- `annual_flow_membership.csv`
- `artifact_contracts.csv`
- `source_contract_qa.csv`
- build/frontier QA manifests

### Human documents

`run-reports` currently produces two exact-run document products:

- `annual_management/report.html` + PDF
- `treasury_accountability/report.html` + PDF

The annual management report now has six governed pages:

1. executive summary;
2. operation and portfolio;
3. application of result;
4. cash and treasury;
5. debt and internal credits;
6. control and quality.

Its exact metric selectors cover rent, property OPEX, operating result, funding, personal draws, dividends, post-draw result, savings/retention, validated cash, property rent breakdowns, OPEX categories, operating ratios, debt relations/stock/activity and data-quality metrics.

The treasury report covers the monthly cash-accountability bridge by Box/Currency: opening control, rent/funding/debt/FX/internal-transfer inflows, taxes/services/maintenance/legal/draws/dividends/debt/FX/internal-transfer outflows, total in/out, net flow, closing control, unknowns and direct non-cash support.

### Professional evidence

`accounting.professional` retains governed drilldown executors for annual flows, validated cash, debt position/activity, funding/support and derived metrics. `professional-drilldowns` and `professional-linked-digest` remain presentation/reconciliation operations over an existing professional pack.

Cash is materially stronger than in the notebook era: `cash.position.validated` and `cash.control.inferred_box_motor` are separate authorities. Validated cash selects the latest valid snapshot per `Box/account_id`; inferred box control and internal party balances are excluded from cash headlines and never used as fallback.

---

## Notebook-by-notebook census

| Notebook | What it was for | Current governed path | Coverage after deletion | Residual worth preserving |
|---|---|---|---|---|
| `00_shared_loader_and_contracts.ipynb` (root) | Find latest artifacts, normalize annual metrics, inventory metric IDs/dimensions, readiness/QA, export a shared professional contract. | Build manifests; `artifact_contracts.csv`; `source_contract_qa.csv`; annual dashboard contract/QA; release check; report manifests/catalog. | **COVERED** | No single human-readable “all artifacts / all dimensions” inventory page. Low-cost convenience, not accounting authority. |
| `accounting_reports/00_shared_loader_and_contracts.ipynb` | Later/executed version of the same shared read contract, including monthly/debt source inventory. | Same governed contracts plus exact-run monthly marts and debt artifacts. | **COVERED** | Same convenience inventory gap. The two `00` notebooks are overlapping generations, not independent accounting products. |
| `01_executive_brief_one_page.ipynb` | 3-minute brief: 5–7 KPIs, five findings, caveats and support references over rent/cost/margin/draws/funding/cash/debt. | `annual_management` page 1 plus pages 2–6; report cells/validation/manifest; professional drilldowns for evidence. | **REPLACED / PARTIAL** | The current report is stronger structurally but does not reproduce the notebook's free-form conditional “five findings” narrative generator exactly. A future brief profile could sit on top of the annual report cells. |
| `02_operating_result_report.ipynb` | Long monthly history, canonical monthly operating table, administration-regime map, regime comparison, 2026-H1 close, OPEX support, plots and narrative. | `monthly_operating_statement.csv`; `frontend_metric_series.csv`; annual management operation page; treasury/accountability monthly data; annual report labels current year as H1/YTD when appropriate. | **PARTIAL** | **Real gap:** standardized longitudinal operating report, manual/semi-manual administration-regime annotations/comparisons, and its time-series figures. Regime semantics were intentionally external to accounting and should remain annotations, not inferred core facts. |
| `03_compact_semester_tables.ipynb` | ≤10×10 semester tables: semester overview, big-number digest, a few ratios, 2026-H1 close, claims/support. | Monthly governed artifacts and annual metrics contain the source facts; current annual report provides compact annual/YTD views. | **PARTIAL** | **Real gap:** a governed semester projection and compact semester renderer. This is recoverable without notebooks if period aggregation explicitly respects flow vs stock semantics. |
| `04_monthly_metric_series_audit.ipynb` | Generic review of every monthly metric for gaps, spikes, sign changes, zero runs, level shifts; anomaly queue; individual/group plots. | `frontend_metric_series.csv`; frontier/source/semantic QA checks. | **PARTIAL — important gap** | **Real gap:** generic heuristic time-series anomaly census and review queue. Current QA is semantic/contractual, not an all-series statistical scanner. This is a good candidate for a small governed diagnostics product. |
| `accounting_reports/01_balance_dashboard_overview.ipynb` | Annual executive picture combining operation, funding/distributions, position, debt and extended QA; ARS fund bridge. | `annual_management` pages 1–6 + annual dashboard contract/QA + treasury report. | **REPLACED** | Some notebook-specific display diagnostics (e.g. hidden-label collision inspection) need not be a report feature; retain only if they remain useful as static validation. |
| `accounting_reports/02_cash_and_liquidity.ipynb` | Annual validated cash vs governance-flow bridge; FB drain audit; 2024 focus; Household/PM/FB comparison. | `cash_authority.py`; governed `monthly_cash_close`; annual cash metrics; `monthly_cash_accountability`; treasury report; annual management cash/treasury page; cash drilldown executor. | **COVERED / PARTIAL** | Core cash semantics are **better covered now**. Residual views: bespoke FB-drain analysis, fixed 2024 focus and cross-scope Household/PM/FB comparison. Household is not part of the default FB/PM report scope and should only return if intentionally supported. |
| `accounting_reports/03_income_rent_and_operations.ipynb` | “Does the property operation sustain itself?” Rent total/by property, property OPEX, taxes/services/maintenance/legal, operating margin, OPEX/rent, classification QA. | Annual metrics + `annual_management` operation/portfolio page + quality page + monthly operating statement. | **REPLACED** | Specific editorial emphasis on “2024 pressure” is narrative, not a missing accounting capability. |
| `accounting_reports/04_debt_open_items_and_reconciliation.ipynb` | Who owes whom, how much/why, stock vs flow, engine-vs-ledger mismatch, repayments, residual adjustments, action list for source fixes. | `run-debt` debt resolution artifacts; `monthly_debt_position/activity`; annual debt page; debt position/activity executors and drilldowns. | **COVERED / PARTIAL** | Debt authority and evidence are covered. **Residual:** a compact remediation/action-list UI over `debt_status_reconciliation.csv` for ledger corrections is not a first-class current report. |
| `accounting_reports/06_monthly_dynamics_bar_charts.ipynb` | Broad monthly chart pack: operation, semantic buckets, draws by Box, FB rent vs withdrawals, PM stress, Household, OPEX, FX, unknowns, debt, cash and diagnostic levels. | All authoritative monthly inputs still exist; treasury report presents a governed monthly bridge. | **PARTIAL — presentation gap** | **Real gap:** no equivalent broad monthly chart gallery. Rebuild only from governed facts; do not revive diagnostic balances as cash truth. |
| `accounting_reports/07_monthly_dynamics_tables.ipynb` | Matrix equivalent of 06: rows=metric/scope/currency/category, columns=months; clean CSV exports; monthly package index. | Authoritative monthly marts + `frontend_metric_series.csv` + treasury report. | **PARTIAL — presentation gap** | **Real gap:** generic governed month-column matrices / CSV export package. The old diagnostic `box_balance_*` matrix is **RETIRED BY DESIGN**; if a control matrix is useful, use explicit inferred-control authority separately from validated cash. |
| `accounting_reports/annual_balance_dashboard.ipynb` | Large annual management dashboard: operating statement, funding/distributions, settlement channels, position, debt movement, actor claims, quality, metric appendix. | Directly superseded by governed annual dashboard metrics + six-page `annual_management` report; treasury report handles payment/cash-accountability channels; professional drilldowns provide evidence. | **REPLACED** | Any actor/channel detail omitted from the fixed annual document can be added as a governed report section without restoring notebook logic. |
| `accounting_reports/cash_position_eda.ipynb` | EDA distinguishing daily party balances from box-level motor; key invariant: internal party balances ≠ box cash close. | `cash_authority.py` + `cash_position_executor.py` explicitly partition validated cash, inferred control, internal balances and other exclusions. | **REPLACED / RETIRED BY DESIGN** | The semantic discovery is now an enforced runtime invariant. Free-form EDA plots are gone, which is acceptable; the old party/box motor should not regain cash authority. |

### Planned notebook that never existed

The shared-loader notebook mentioned a proposed `05_family_human_storypack.ipynb`, but there is no such file in the retiring tree. Its absence is therefore **not** coverage lost by PR #99.

---

## Capability-level conclusion

### Fully or materially covered by the hardened system

The notebook layer no longer owns unique authority for:

- artifact/readiness contracts;
- annual operating result and portfolio breakdown;
- funding vs operating-income separation;
- draws/dividends/application of result;
- annual validated cash;
- validated-cash vs inferred-control separation;
- monthly cash-accountability / treasury bridge;
- debt stock and debt activity;
- principal/interest/repayment distinctions;
- annual debt relations;
- native-currency separation;
- unknown/review-required and quality signals;
- source lineage and drilldown evidence;
- HTML/PDF human-report generation.

Deleting notebooks does **not** delete those capabilities. Most are now safer because they are contract-backed and tested rather than reconstructed ad hoc in cells.

### Useful analytical projections not yet standardized

These are genuine residuals and should remain visible after notebook deletion:

1. **Semester projection / compact semester report**  
   Need an explicit flow-vs-stock-aware period projector, then a small table renderer. Never implement “semester = generic sum” for stocks.

2. **Administration-regime annotations and comparisons**  
   The regime map is external contextual metadata. If revived, keep it as an explicit annotation input layered over governed monthly facts, not an inferred accounting classification.

3. **Generic monthly metric anomaly review**  
   Produce a machine-readable anomaly census from `frontend_metric_series.csv`: missing periods, abrupt level changes, sign flips, long zero runs, unusual spikes, plus a review queue. Heuristics must only say “review”, never “wrong”.

4. **Generic monthly matrix package**  
   A reusable projection from governed monthly facts to month-column matrices would recover much of Notebook 07 cheaply. It should expose source/measure/grain and keep currencies separate.

5. **Monthly chart gallery**  
   Optional presentation over the same governed matrix/series layer. This is lower priority than the machine-readable monthly review/matrix products.

6. **Debt remediation view**  
   A compact action list derived from `debt_status_reconciliation.csv` would recover the most operationally useful piece of the old debt notebook that is not prominent in the current report.

7. **Household cross-scope analysis, if still intended**  
   Old notebooks sometimes compared Household, Family Business and Property Management. Current standard reports intentionally center the FB/PM supported scope. Do not silently broaden the scope; add Household only through an explicit scope decision.

8. **One-page conditional narrative profile**  
   If useful, generate a short brief from governed annual report cells/QA rather than maintaining a second accounting/reporting stack.

---

## Things that should *not* be recovered

Notebook retirement is also an opportunity to prevent old ambiguity from returning. Do not recreate:

- `metric_values.csv` / legacy metric-registry fallback as semantic authority;
- generic column guessing as a substitute for explicit contracts;
- `box_balance_*` or party-level balances as validated cash;
- fallback from unavailable validated cash to inferred control;
- notebook-owned OPEX/funding/debt classification;
- cross-currency totals;
- independent notebook `latest` path conventions;
- report code that silently widens the supported scope.

The old notebook EDA was valuable because it discovered several of these distinctions. The hardened system should preserve the **conclusions and contracts**, not the exploratory implementation.

## Merge assessment

From this census, deleting `accounting/notebooks/` in PR #99 is reasonable **without loss of unique accounting semantics**.

What is being removed is mainly:

- duplicated readers/loaders;
- exploratory and presentation code;
- notebook-local narrative generators;
- broad monthly matrices/charts;
- heuristic anomaly review;
- a few bespoke contextual comparisons.

The meaningful residuals are listed above as explicit extension candidates. They can be rebuilt as small consumers of the governed spine rather than kept alive as a parallel notebook architecture.
