# Metric engine decommission gates

Baseline: `main` at `ea769177a99992b6626c83ebd641ca65c72fb7b5`.

## Accounting invariant

Deletion is allowed only when the current governed stack preserves the accounting facts and traceability that matter: native-currency separation, property-vs-Household scope, narrow core funding vs broader support, validated cash only, closing debt stock vs additive debt activity, explicit FX measure/grain, and professional drilldown traceability.

Historical compatibility is not an accounting authority. Numerical parity is required only where the semantic identity is intentionally unchanged. A deliberate semantic correction must be documented as retirement rather than forced back to the old number.

## Decision rule

A legacy metric is migrated only if it expresses a useful accounting quantity that cannot be recovered by a small projection of a governed artifact.

- Different legacy metric ID: not a reason to preserve it.
- Old notebook/presentation convenience: not a reason to preserve it.
- Rejected mixed semantics: explicitly retire; do not reproduce.
- Unique useful accounting fact: migrate the smallest governed gap before deletion.

The final census is `notes/metric_capability_census_20260825.csv`.

## Phase gates

### A. Frontier cut

`accounting.metrics.frontier` must be runnable from only:

- `monthly_flow_semantic_split.csv`
- `monthly_operating_statement.csv`
- `monthly_cash_close.csv`
- `monthly_debt_position.csv`
- `monthly_debt_activity.csv`

It must not load or emit legacy compatibility from `metric_registry.csv` or `metric_values.csv`.

Required QA:

- frontier sources are a subset of the canonical five above;
- no legacy-only series are injected;
- cash projection remains `cash.position.validated`, with no inferred/internal fallback;
- all monetary series carry Currency;
- FX remains native-currency and current semantic scope.

### B. Metrics engine decommission

The old generic registry/leaf/derive/view engine may be deleted when:

1. frontier is independent from it;
2. annual dashboard remains independent from it;
3. publisher/reporting consumers use annual/frontier/governed monthly artifacts;
4. no current supported consumer requires `metric_values.csv`, `metric_registry.csv`, generic statement exports, or `metric_views/*`;
5. any tests that encode accounting meaning are moved to governed artifacts rather than retained merely to keep the old engine alive.

A retained `accounting.metrics.build` is acceptable only as a thin current orchestrator for frontier + annual + manifest/source-contract handoff.

### C. Old views-stage removal

`accounting/marts/build.py` may be deleted after old contribution/OPEX leaf consumers disappear.

Do not delete governed marts:

- `accounting/marts/semantic.py`
- `accounting/marts/cash.py`
- `accounting/marts/debt.py`
- `accounting/marts/treasury.py`

The Makefile must no longer require `run-marts`, `_run_views_action`, `views_sanity.json`, or legacy reports/view directories for the canonical/debt/metrics path.

## Reconciliation matrix

| Family | Required post-change evidence |
| --- | --- |
| Rent | governed annual sum by year × Currency; semantic rent membership only |
| Property OPEX | governed annual sum by year × Currency; Household excluded |
| Funding | narrow `FUND.CONTRIB.TOTAL` remains distinct from broader support |
| Draws/dividends | governed semantic membership; no raw text rediscovery |
| Cash | latest valid governed account snapshots; no monthly stock sum; no inferred/internal fallback |
| Debt stock | latest valid closing position; no lexical invalid date and no prior-period backfill |
| Debt activity | additive movement flow remains separate from stock |
| FX | explicit governed semantics and native-currency grain; no cross-currency sum |
| Scope | outputs stay within the requested Box scope |
| Publication | public contract and canonical dashboard remain complete |
| Drilldowns | supported professional cells continue to resolve governed membership |

## Explicitly retired semantics

The following are not migration targets:

- `IS.INCOME.TOTAL` mixing rent and family funding;
- `IS.NET.AFTER_COSTS` derived from that funding-inclusive income;
- `IS.NET.POST_DRAWS` built on the same legacy coverage identity;
- raw-regex draws classification;
- `daily_cash_position` as reported cash;
- generic quarterly metric universe with no current supported consumer.

If a future report needs a quarterly or rolling presentation, derive it at the report layer from governed monthly facts rather than restoring a second metric universe.
