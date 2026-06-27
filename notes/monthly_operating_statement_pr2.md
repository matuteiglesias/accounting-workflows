# Monthly operating statement PR 2

## Source input

`monthly_operating_statement.csv` is built only from `monthly_flow_semantic_split.csv`, which is produced by the conservative semantic classifier. If the semantic split is missing or has the wrong schema, the operating statement builder fails clearly instead of falling back to legacy wide report tables.

## Statement lines

The first backend-owned statement includes these canonical lines:

- `operating_revenue`
- `rent_revenue`
- `property_opex_true`
- `taxes`
- `services`
- `maintenance`
- `legal`
- `other_property_opex`
- `net_operating`
- `funding_contributions`
- `family_draws_or_distributions`
- `coverage_after_draws`
- `unknown_or_ambiguous_outflows`
- `classification_coverage`
- `debt_movements`
- `internal_transfers`

## Calculation rules

The clean operating result is:

```text
net_operating = operating_revenue - property_opex_true
```

Funding contributions, family draws/distributions, debt movements, internal transfers, and unknown/review-required flows are visible as separate lines and are excluded from `net_operating`.

`coverage_after_draws` is a coverage-style cash line:

```text
coverage_after_draws = net_operating + funding_contributions - family_draws_or_distributions
```

`classification_coverage` is a ratio:

```text
classified_amount_abs / eligible_amount_abs
```

## Caveats

The statement is canonical but conservative. Unknown and review-required amounts remain visible and should be reviewed before using the statement for decision-grade external reporting. `family_withdrawal_candidate` rows are separated from OPEX but still require accounting review.

## Legacy outputs unchanged

This PR does not delete, rename, or rebuild legacy metrics/views such as `IS.NET.AFTER_COSTS`, `IS.CONTRIB.TOTAL`, `IS.DRAWS.PERSONAL`, `income_statement_monthly_last6.csv`, or `metric_values.csv`. Those remain report-support artifacts and may still include legacy coverage-like semantics.

## Later integration

Future PRs should wire these clean statement lines into the metrics registry, human reports, and frontend contracts with explicit metric IDs such as `IS.REVENUE.OPERATING`, `IS.OPEX.PROPERTY`, `IS.NET.OPERATING`, `FUND.CONTRIB.TOTAL`, `DIST.DRAWS.PERSONAL`, `COV.NET.AFTER_DRAWS`, `DQ.CLASSIFICATION.COVERAGE`, and `DQ.UNKNOWN.AMOUNT`.
