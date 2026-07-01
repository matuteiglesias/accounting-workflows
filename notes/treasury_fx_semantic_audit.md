# Treasury FX semantic audit

1. Semantic rules are defined in `accounting/marts/semantic.py` in `SEMANTIC_RULES`; executable row matching lives in `_classify_row`.
2. `R999_unknown_review_required` is the final fallback returned by `_classify_row` when no conservative rule matches; it sets `semantic_bucket=unknown`, `semantic_subbucket=review_required`, low confidence, and `review_required=True`.
3. FX matching fields available after Stage D/materialization and semantic preparation include `Flujo`, `Tipo`, `payer`, `receiver`, `Box`, inferred `actor`, inferred `counterparty`, `Detalle`, `notes`, plus `cash_path` and `channel` when present or synthesized.
4. `monthly_flow_semantic_split.csv` is the first aggregated output carrying `semantic_bucket` and `semantic_subbucket`; `classification_audit.csv` carries row-level classifications earlier in the semantic mart.
5. Annual `DQ.UNKNOWN.AMOUNT` is sourced from `monthly_operating_statement.csv`, specifically the `unknown_or_ambiguous_outflows` statement line.
6. Annual operating statement metrics are sourced from `monthly_operating_statement.csv` lines including `operating_revenue`, `property_opex_true`, `net_operating`, funding, draws, and coverage.
7. Unknown amounts are not included in operating revenue, property OPEX, or net operating result; they are surfaced separately as data quality / review-required amounts.
8. Legacy OPEX views remain compatibility artifacts and can be broader than canonical OPEX. Canonical property OPEX is restricted to `semantic_bucket=property_opex`; FX must not be classified there.
9. Safest minimal classification: `Cambio:FX` is `treasury_fx / fx_conversion_proceeds` or outflow depending party direction, and `Costo Operativo:FX` is `treasury_fx / fx_cost_or_spread`. Both are excluded from income, funding, draws, debt, and property OPEX.
10. Downstream metrics needed: `TR.FX.CONVERSION.IN`, `TR.FX.CONVERSION.OUT`, `TR.FX.COST.OUT`, `TR.FX.NET`, `TR.FX.BY_BOX`, `TR.FX.BY_TYPE`, plus QA visibility for one-sided FX, missing future rates, and review-required FX rows.
