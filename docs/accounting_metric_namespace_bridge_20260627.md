# Accounting metric namespace bridge — 2026-06-27

## Why this change exists

The accounting pipeline already produces useful metric contracts and reports, but the legacy registry uses mostly `IS.*` and `BS.*` names. The audit found that some legacy `IS.*` metrics combine operating activity with family funding or distributions. This pass makes those semantics explicit without renaming or removing existing metrics.

## Current legacy problem

The legacy formulas remain available for compatibility:

```text
IS.INCOME.TOTAL = IS.RENT.TOTAL + IS.CONTRIB.TOTAL
IS.NET.AFTER_COSTS = IS.INCOME.TOTAL - IS.OPEX.TOTAL
IS.NET.POST_DRAWS = IS.NET.AFTER_COSTS - IS.DRAWS.PERSONAL
```

The issue is that `IS.CONTRIB.TOTAL` is funding, not operating revenue. Therefore `IS.INCOME.TOTAL` and metrics derived from it are coverage-style legacy views, not clean operating-income metrics. The registry now carries `legacy_warning` text on those legacy metrics.

## Desired namespace semantics

| Namespace | Intended meaning |
|---|---|
| `IS.*` | Operating result / economic operation. |
| `CF.*` | Cash movement. |
| `BS.*` | Closing stocks / proxy balance sheet. |
| `ID.*` | Internal debt explanation. |
| `FUND.*` | Family contributions / funding. |
| `DIST.*` | Draws / distributions. |
| `COV.*` | Human coverage view that may intentionally combine operating, funding, and distributions. |
| `HUMAN.*` | Narrative / QA / presentation metrics. |

## Added shadow metrics

These metrics are additive and do not replace legacy metric IDs:

| Shadow metric | Definition | Purpose |
|---|---|---|
| `IS.REVENUE.TOTAL` | `IS.RENT.TOTAL` | Clean operating revenue probe. |
| `IS.NET.OPERATING` | `IS.REVENUE.TOTAL - IS.OPEX.TOTAL` | Clean operating result excluding family funding. |
| `FUND.CONTRIB.TOTAL` | `IS.CONTRIB.TOTAL` | Funding namespace alias for family contributions. |
| `DIST.DRAWS.PERSONAL` | `IS.DRAWS.PERSONAL` | Distribution namespace alias for personal draws. |
| `COV.NET.AFTER_DRAWS` | `IS.NET.OPERATING + FUND.CONTRIB.TOTAL - DIST.DRAWS.PERSONAL` | Coverage view that combines clean operating result, funding, and draws explicitly. |

## Registry metadata added

The registry output now includes:

```text
metric_type
economic_role
namespace_target
migration_status
legacy_warning
```

These columns make metric intent inspectable in `metric_registry.csv` and give future notebooks a stable place to filter operating, funding, distribution, cash, debt, claim, coverage, and QA metrics.

## Compatibility guarantee

This is an additive bridge only:

* Existing metric IDs remain available, including `IS.CONTRIB.TOTAL`, `IS.INCOME.TOTAL`, `IS.NET.AFTER_COSTS`, `IS.DRAWS.PERSONAL`, and `IS.NET.POST_DRAWS`.
* Existing statement export ID lists and human report inputs were not removed.
* Legacy metrics that can mislead future operating-result analysis now carry semantic metadata and warnings.

## Bridge CSV

The current-to-desired mapping is documented in:

```text
docs/accounting_metric_bridge_current_to_desired.csv
```

The bridge table identifies the current metric, desired metric/namespace, proposed action, priority, risk, confidence, and notes.

## Deferred

* Full namespace migration and renaming.
* External YAML/CSV registry as the source of truth.
* Notebook UI scaffold.
* Full `CF.*`, `BS.*`, `ID.*`, `COV.*`, and `HUMAN.*` expansion.
* Any business-logic changes to legacy reports/frontends.
