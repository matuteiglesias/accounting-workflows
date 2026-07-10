# Professional derived drilldown audit

PR 2 adds explanation drilldowns for derived professional tables. These pages reconcile first to the statement/annual source artifact and only then show semantic/classification rows when the source contract permits it.

| table_id | source artifact | supported rows | unsupported rows | lineage kind | expected filters |
|---|---|---|---|---|---|
| `monthly_tables_operating_statement_matrix` | `monthly_operating_statement.csv` | flow statement lines such as `operating_revenue`, `rent_revenue`, `property_opex_true`, `net_operating`, `funding_contributions`, `family_draws_or_distributions`, `unknown_or_ambiguous_outflows`, `treasury_fx_*` | missing `Currency`, missing source rows, unmapped lines | direct or composite | `period`, `Currency`, `statement_line`; composite pages include component rows |
| `monthly_tables_operating_statement_matrix_ars` | `monthly_operating_statement.csv` | same as monthly operating matrix, generally ARS rows | non-reconciled or unmapped rows | direct or composite | `period`, `Currency`, `statement_line` |
| `overview_balance_dashboard` | `annual_balance_dashboard_metrics.csv` | annual flow/quality metrics whose source is `monthly_operating_statement.csv` or `monthly_flow_semantic_split.csv` | stock/cash/debt metrics, missing annual source row | annual explanation | `year`, `Currency`, `metric_id`, optional dimensions |
| `income_operating_statement` | `annual_balance_dashboard_metrics.csv` | annual income/flow rows and supported semantic dimensions | stock/cash/debt metrics, unsupported ratios without formula source | annual explanation | `year`, `Currency`, `metric_id`, optional `dimension_name`/`dimension_value` |
| `cash_annual_box_flow_bridge_wide` | `monthly_flow_semantic_split.csv` | annual flow bridge rows by year/currency/box/semantic/cash path | validated cash levels, cash close, diagnostic box balance | semantic flow bridge | `year`, `Currency`, `Box`, optional semantic bucket/subbucket/cash path, `measure` |

Rules preserved:

- Annual/dashboard stock and cash levels are not treated as flow ledger drilldowns.
- Ratios and formula lines link to explanation pages, not flat ledger-row pages.
- ARS and USD are never summed together; missing `Currency` remains unsupported.
- Public bundle files are read only when needed as source artifacts; no public bundle output is written by these builders.
