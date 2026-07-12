# Professional funding semantics audit

Diagnostic-only report for contributions / funding / support / direct payments / debt-linked flows.

## Current semantic conclusions

1. `funding_contribution` is currently visible, but actor/channel/cash-effect/debt dimensions are incomplete unless supplied upstream.
2. Direct obligation payments must not be treated as simple cash inflows.
3. Professional labels need explicit metric IDs and dimensions before renderer wiring.

## Classification problems

No candidate rows found in available artifacts.

## Answers to required questions

1. Current `funding_contribution` rows are rows explicitly classified as funding plus candidate funding/support rows detected by text, actor, non-rent PM/FB inflows, direct obligation payments, or debt hints.
2. Non-rent PM inflows are rows where target/Box is Property Management, amount_in is positive, and the semantic bucket is not operating rent; see `funding_lineage_audit.csv`.
3. Non-rent FB inflows are rows where target/Box/party evidence indicates Family Business and the flow is not rent; see `funding_lineage_audit.csv`.
4. Direct obligation payments are candidates with tax/service wording and tenant/family actor evidence; these are flagged with `is_direct_obligation_payment` when detectable.
5. Rows invisible in annual metrics are those without a stable annual `metric_id` or those represented only as generic OPEX/debt without funding dimensions.
6. Rows visible in semantic split but lost in professional tables are rows whose dimensions collapse to generic `funding_in` or `FUND.CONTRIB.TOTAL`.
7. Dashboard labels needing explicit mappings include Funding / aportes, Matías funding, Inquilinos directo a pagar impuestos, Inquilinos a la caja, Alejandro funding, Primos funding, Héctor funding, Household funding PM, Retiros / gasto personal, Dividendos, and Cobertura después de funding y retiros.
8. Needed subbuckets/channels include cash_to_box, tenant_to_box, tenant_direct_tax_payment, tenant_direct_service_payment, household_to_pm, family_business_contribution, named_actor_support, debt_creation, and debt_settlement.
9. Needed dimensions include funding_actor, funding_channel, source_box, target_box, beneficiary_box, obligation_box, cash_effect, debt_effect, and linked_debt_id.
10. Debt-affecting cases are rows with debt/deuda/prestamo/repago/settlement evidence or future linked debt IDs.
11. Unsupported flow drilldowns should become supported for funding totals and funding by actor/channel/cash effect.
12. Stock/debt lineage should be used for debt balances, settlements, and debt-linked funding rows.

## Prioritized implementation plan

### Patch 1 — semantic classifier / rule IDs
Files: `accounting/marts/semantic.py`. Expected behavior: classify contribution/support channels explicitly. Risks: historical migration and double counting. Acceptance: classification audit shows specific funding rule IDs.

### Patch 2 — monthly_flow_semantic_split dimensions
Files: `accounting/marts/semantic.py`. Expected behavior: propagate funding_actor, funding_channel, target_box, obligation_box, cash_effect, and debt fields. Risks: changed aggregation grain. Acceptance: monthly split has explicit funding dimensions.

### Patch 3 — annual_balance_dashboard_metrics generation
Files: `accounting/metrics/annual.py`. Expected behavior: annual funding totals by channel/actor/cash effect. Risks: new IDs need frontend mapping. Acceptance: annual metrics include dimensioned FUND rows.

### Patch 4 — professional table labels / metric IDs
Files: professional table builders/exporters. Expected behavior: rows carry stable metric IDs. Risks: notebook/code drift. Acceptance: labels no longer drive semantics.

### Patch 5 — drilldown mapping for funding labels
Files: `accounting/professional/drilldown.py`. Expected behavior: drilldowns filter by metric ID plus semantic dimensions. Risks: legacy rows may remain unsupported. Acceptance: funding rows produce supported lineage.

### Patch 6 — debt linkage / stock lineage
Files: `accounting/debt/resolve.py`, `accounting/marts/semantic.py`, `accounting/professional/drilldown.py`. Expected behavior: debt-linked support routes to debt activity/position. Risks: double-counting debt and funding. Acceptance: linked_debt_id connects funding rows to debt evidence.

### Patch 7 — tests and QA checks
Files: `tests/`. Expected behavior: fixtures cover rent, cash funding, direct obligation payments, HH→PM, FB support, and debt. Risks: insufficient real-ledger coverage. Acceptance: CLI and acceptance script pass.
