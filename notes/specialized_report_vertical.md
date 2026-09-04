# Specialized governed reports

This vertical keeps one authority boundary:

```text
governed artifact → professional view → report HTML → PDF → catalog → viewer
```

The viewer remains document discovery/delivery only. It never consumes accounting CSVs, rebuilds metrics, or decides membership.

Recipes in `accounting/reports/specialized/spec.py` declare the practical question, audience, view, scope, period/currency policy, section plan, and caveat. Accounting selection lives in explicit backend professional views under `accounting/reports/specialized/views.py`. Renderers only format those views.

## Current stable/pilot surface

The first four specialized reports remain the initial pilots:

- `pm_tax_accountability`
- `pm_services_accountability`
- `stakeholder_support`
- `distributions_by_recipient`

Round 1 adds the next low-ambiguity administrative views where the canonical semantic split already provides authority:

- `rent_by_property`
- `rent_monthly_evolution`
- `opex_by_category`
- `taxes_by_property`
- `services_by_property`
- `distributions_vs_rent`

`distributions_vs_rent` is explicitly a side-by-side comparison of two governed populations. It does not net them or assert that distributions derive from same-period rent.

## Opportunity census / program backlog

| Family | Candidate report | Governing artifact/view | Audience | Readiness | Missing authority / caution | Priority |
|---|---|---|---|---|---|---|
| Production | Rent by property/source | `monthly_flow_semantic_split.csv` / governed rent membership | stakeholder | ready | location is descriptive, not ownership | P1 |
| Production | Monthly rent evolution | semantic split / governed rent membership | stakeholder | ready | no automatic vacancy inference | P1 |
| Production | Annual rent comparison | annual rent metric / semantic split | stakeholder | ready-next | native currencies separate | P2 |
| Production | Vacancy / months without rent | not yet singular | management | blocked | absence of rent is not automatically vacancy | HOLD |
| Costs | OPEX by category | semantic split / property OPEX | stakeholder | ready | no legal bearer inference | P1 |
| Costs | Taxes by property | semantic split / taxes | stakeholder | ready | direct payment != Box cash | P1 |
| Costs | Services by property | semantic split / services | stakeholder | ready | direct payment != Box cash | P1 |
| Costs | Maintenance/conservation | semantic split / maintenance | management | ready-next | review source coverage | P2 |
| Costs | Extraordinary incidents | settlement/expense metadata | management | partial | needs stable incident nature/coverage | HOLD |
| Costs | Arrears/refinanced costs | status semantics | management | partial | recognition versus settlement policy still needs review | HOLD |
| Stakeholders | Who paid/supported | `monthly_stakeholder_support.csv` | stakeholder | pilot | support != physical payer | P1 |
| Stakeholders | Who covered taxes | governed tax/service payment/support views | stakeholder | pilot | Box-funded and actor-funded sources remain distinct | P1 |
| Stakeholders | Who covered services | governed tax/service payment/support views | stakeholder | pilot | same | P1 |
| Stakeholders | Support by target Box | stakeholder support mart | management | ready-next | do not sum chained Boxes globally | P2 |
| Stakeholders | Prior-period clearing | settlement nature + obligation/settlement period | management | ready-next | debt/support double counting prohibited | P2 |
| Stakeholders | Designated funding/pass-through | settlement/detail evidence | management | partial | Lote 32 and other pools may lack 1:1 expense linkage | HOLD |
| Distribution | Distributions by recipient | governed distribution membership | stakeholder | pilot | not entitlement/final custody | P1 |
| Distribution | Distributions by year | distribution professional view | stakeholder | ready-next | native currencies separate | P2 |
| Distribution | Distributions versus rent | governed distribution + rent views | stakeholder | ready | comparison only; no netting | P1 |
| Treasury | Physical inflows by Box | `monthly_box_treasury_flow.csv` | management | ready-next | physical cash only | P2 |
| Treasury | Physical outflows by Box | `monthly_box_treasury_flow.csv` | management | ready-next | physical cash only | P2 |
| Treasury | Accountability balance | `monthly_cash_accountability.csv` | stakeholder | ready-next | control balance != validated cash | P2 |
| Treasury | Mar-Aug / Sep-Feb cycles | `family_business_accountability_cycles.csv` | stakeholder | ready-next | cycle dates are administrative review cuts | P2 |
| Treasury | `other_cash_*` residuals | residual audit/QA | controller | ready-internal | diagnostic only | INTERNAL |
| Debt | Open positions | `monthly_debt_position.csv` | stakeholder | core/ready | accounting debt facts, legal characterization separate | P2 |
| Debt | Debt activity | `monthly_debt_activity.csv` | stakeholder | core/ready | stock and flow remain separate | P2 |
| Debt | Repayment allocations | repayment detail | management | ready-next | sum allocated amount, not repeated repayment amount | P2 |
| Debt | Economic origin of debt | debt origin metadata + source trace | management | partial | only where origin is explicitly governed | P2/HOLD |
| Control | Review-required | semantic/treasury QA | controller | ready-internal | not stakeholder-facing by default | INTERNAL |
| Control | Semantic coverage | semantic coverage/QA | controller | ready-internal | quality view, not accounting result | INTERNAL |
| Control | Validated cash vs accounting control | cash close + accountability | management | ready-next | unavailable must remain unavailable | P2 |
| Valuation | USD-CCL reading | derived valuation sidecar | management | separate family | never replace native books | P3 |
| Historical | Administration regime changes | documentary chronology | management | narrative only | do not infer regime from transaction patterns alone | HOLD |
| Composite | Property dossier | composition of stable specialized views | stakeholder | later | must preserve authority per block | P3 |
| Composite | Stakeholder dossier | support/distribution/debt blocks | management | later | no automatic legal netting | P3 |

## Agent SOP

For one specialized report:

1. State one practical stakeholder/administrative question.
2. Identify an existing governed artifact and exact professional-view selector.
3. Declare scope, native currency policy, period basis, measure, and dimensions.
4. Perform four reviews:
   - accounting/controller: population, denominator, reconciliation;
   - legal/governance: fact versus legal/disputed inference;
   - administrator/stakeholder: usefulness and clarity;
   - QA: table/chart/trace membership, unavailable-vs-zero, no duplication.
5. Record what the report establishes and what it does not establish.
6. Compose fixed Spanish narrative fragments plus governed dynamic values.
7. Render self-contained canonical HTML and derive PDF from that HTML.
8. Catalog only finished HTML/PDF. Keep traces/manifests/CSV internal.

Unknown accounting or legal meaning is a stop condition.

## Anti-bloat rules

A new report should normally require:

```text
one governed view/selector
+ one small recipe
+ reusable presentation blocks
+ reconciliation tests
```

Do not create one renderer per report, a chart-specific accounting metric, a new semantic registry, a CMS, a client-side accounting API, or a legal conclusion from accounting classifications.

## Review rounds

Round 1 focuses on production/cost/distribution views with mature semantic authority. The recommended next bounded round is:

- support by target Box;
- prior-period clearing;
- physical inflows/outflows by Box;
- accountability balance and cycles;
- debt positions/activity/repayment allocations.

Composite property/actor dossiers should wait until their component reports have stabilized so they remain weaving surfaces rather than new authorities.
