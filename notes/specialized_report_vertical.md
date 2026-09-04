# Specialized governed reports

This vertical keeps one authority boundary:

```text
governed artifact → professional view → report HTML → PDF → catalog → viewer
```

Recipes in `accounting/reports/specialized/spec.py` declare the practical
question and caveat. They do not classify transactions or calculate ledger
membership. Views are selected before rendering and carry their own source
and period semantics.

## Initial opportunity census

| Family | Governed authority | First report | Measure | Natural view | Caveat |
|---|---|---|---|---|---|
| Costs | `classification_audit.csv` | PM taxes | identified tax applications | actor/year table + pie | recognized application is not PM cash or legal liability |
| Costs | `classification_audit.csv` | PM services | identified service applications | actor/year table + pie | direct payment and Box cash remain distinct |
| Support | `monthly_stakeholder_support.csv` | stakeholder support | recognized amount | actor/year table + pie | target Box scoped; not a physical cash total |
| Distributions | governed distribution membership / annual metrics | distributions by recipient | registered distribution | recipient/year table + pie | does not establish final custody or legal entitlement |

Other supported opportunities remain candidates: rent by Box, OPEX by category,
funding composition, debt origin, property dossiers, actor dossiers, treasury
controls, valuation, and controller QA. They are not added as new authorities by
this change.

## Agent SOP

1. State one practical stakeholder question.
2. Identify the governed artifact and exact selector.
3. Declare scope, currency, period basis, measure, and dimensions.
4. State what the report establishes and does not establish.
5. Build a small professional view; never aggregate raw ledger in the renderer.
6. Compose fixed Spanish narrative fragments, metrics, chart/table, and method note.
7. Reconcile source, denominator, membership, and drilldown.
8. Render canonical HTML and derive PDF from that HTML.
9. Catalog only the finished HTML/PDF; keep traces/manifests internal.

Unknown accounting or legal meaning is a stop condition.
