# Specialized governed reports — Round 2

Round 2 adds controller-facing and stakeholder-facing documents over authorities that already exist in the accounting spine. It does **not** add accounting logic to `accounting-viewer`, create a parallel metric engine, or reinterpret ledger rows in the report layer.

```text
governed run artifact
    -> specialized professional view
    -> fixed report recipe
    -> self-contained HTML
    -> PDF
    -> report catalog
    -> read-only viewer
```

## Reports and authorities

| Report | Governing artifact | Measure / grain | Primary invariant |
|---|---|---|---|
| support by target Box | `monthly_stakeholder_support.csv` | recognized support / year × currency × target Box | target-Box attribution is not physical cash |
| prior-period clearing | `monthly_stakeholder_support.csv` | recognized clearing / settlement grain | obligation period and settlement period remain separate |
| physical inflows by Box | `monthly_cash_accountability.csv` | `total_cash_in` / year × currency × Box | constructive paths never manufacture cash |
| physical outflows by Box | `monthly_cash_accountability.csv` | `total_cash_out` / year × currency × Box | direct third-party payments are not Box cash-out |
| accountability balance | `family_business_accountability_cycles.csv` | latest governed cycle components | opening + receipts - distributions - uses - transfers = closing |
| Mar–Aug / Sep–Feb cycles | `family_business_accountability_cycles.csv` | one closing position per governed six-month cycle | accountability balance is not validated cash |
| open debt positions | `monthly_debt_position.csv` | latest available `component=total` stock | stock is selected at close, never summed through months |
| debt activity | `monthly_debt_activity.csv` | new principal / interest / repayment / adjustment flows | activity reconciles opening to closing; stocks are excluded from flow aggregation |
| repayment allocations | `monthly_debt_repayment_detail.csv` | repayment-to-obligation allocation | allocated + leftover = repayment at event grain; repayment totals are not repeated per target |

## Interpretation boundaries

- `Box` remains an accounting/control scope, not a bank account or legal person.
- `support by target Box` answers where recognized support was applied. It does not answer where physical money moved or who legally owed the underlying cost.
- `prior_period_clearing` is reported only when the governed settlement authority already says `settlement_nature=prior_period_clearing`. The report does not infer clearing from dates, amounts, actors, providers or debt rows.
- Physical inflow/outflow reports use the treasury authority only. Stakeholder support is not promoted into physical cash.
- A positive accountability closing balance is described as a balance to render/account for. A negative balance is described as a deficit of rendition. Neither becomes a claim about missing cash, appropriation or legal debt.
- Family Business accountability cycles use the existing six-month authority anchored on 1 March. The report does not manufacture a new cycle definition.
- Open debt positions consume the latest governed stock and fail closed if the latest relation is unavailable; they never backfill a prior month to make the report look complete.
- Debt activity aggregates only governed activity flows. Adjustments keep their direction explicit rather than being silently converted into ordinary claims or repayments.
- Repayment-allocation reports display `allocated_amount`; repeated event-level `repayment_amount` is reconciliation metadata, not an additive report measure.
- Raw/internal debt identifiers remain trace-only and are not intended as visible document content.

## Validation gates

The fixture-safe Round-2 regressions require:

1. FBPM support excludes Household as a target Box while preserving Household or other actors as participant dimensions where applicable.
2. Prior-period clearing preserves both obligation period and settlement period.
3. `total_cash_in - total_cash_out = net_cash_flow` where the governed treasury row exposes all three measures.
4. Every rendered accountability cycle satisfies its governed control equation.
5. Open-debt reporting uses only the latest period and excludes zero/closed relations; an unavailable latest position blocks reporting rather than causing prior-period fallback.
6. Debt activity reconciles opening + new principal + interest - repayments + adjustments to closing whenever the source position is available.
7. Every repayment event satisfies `sum(allocated_amount) + leftover_amount = repayment_amount`.
8. Native currencies stay separate.
9. Specialized traces reconcile to their rendered view populations.
10. Every document declares what it establishes and what it does not establish.

## Publication boundary

The backend may catalog the HTML/PDF when a governed population exists and report validation passes. Internal traces, validations and manifests remain backend artifacts. The viewer continues to discover and serve finished documents from the catalog; it does not receive selectors, formulas, debt rules, Box rules, cycle logic, or settlement semantics.

This round does not perform live ingestion, source correction, latest-pointer changes, publication, or viewer deployment. A private exact-run render at the frozen cutoff remains the final product QA gate before these reports are treated as live-stable.
