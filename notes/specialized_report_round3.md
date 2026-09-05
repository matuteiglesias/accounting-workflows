# Specialized governed reports — Round 3

Round 3 broadens the administration folder using only already-governed artifacts. It keeps the same boundary:

```text
governed artifact -> professional view -> fixed recipe -> self-contained HTML -> PDF -> catalog -> read-only viewer
```

No report in this round changes ledger membership, debt semantics, cash authority, Box meaning, currency policy, or legal interpretation.

## Reports added

| Report | Authority | Main invariant |
|---|---|---|
| monthly OPEX evolution | `monthly_flow_semantic_split.csv` + `IS.OPEX.PROPERTY` | monthly sum reconciles to governed annual OPEX |
| maintenance/conservation by property | semantic split + `IS.OPEX.BY_CATEGORY[maintenance]` | descriptive location only; no legal bearer inference |
| legal costs by property/reference | semantic split + `IS.OPEX.BY_CATEGORY[legal]` | descriptive grouping only; no legal-strategy inference |
| support by obligation category | `monthly_stakeholder_support.csv` | recognized support != legal obligation or physical cash |
| support by funding channel | `monthly_stakeholder_support.csv` | channel is governed metadata, not reconstructed bank routing |
| support by settlement nature | `monthly_stakeholder_support.csv` | settlement nature is consumed, never inferred by the report |
| physical inflows by category | `monthly_box_treasury_flow.csv` + `monthly_cash_accountability.csv` | actual-cash categories reconcile to `total_cash_in` |
| physical outflows by category | same | actual-cash categories reconcile to `total_cash_out` |
| cash residuals | `monthly_cash_accountability.csv` | diagnostics remain visible; report does not reclassify residuals |

## Review boundary

Accounting/controller review requires exact population, native currency separation, and reconciliation to the upstream total where an independent authority exists.

Legal/governance review requires every report to say what it establishes and what it does not establish. In particular, maintenance/legal/support/residual classifications do not determine ownership, liability, entitlement, wrongdoing, debt enforceability, or final actor net balances.

Administrator review favors one practical question, a small KPI strip, one chart, one table, and fixed Spanish narrative.

QA requires chart/table/trace membership parity, unavailable-vs-zero preservation, no Household target-Box leakage into FBPM support totals, and no constructive support promoted into physical cash.

## Still held back

The following remain intentionally unbuilt pending stronger authority or a separate design decision:

- vacancy inference;
- extraordinary incidents;
- arrears/refinanced-cost reporting;
- designated/pass-through pools as a dedicated report;
- debt economic origin where origin metadata is incomplete;
- legal/accounting net balance by actor;
- historical administration-regime conclusions;
- USD-CCL mixed with native books.

Composite property and stakeholder dossiers remain a later weaving layer over stabilized specialized reports; they must not become new accounting authorities.
