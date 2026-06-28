# Backend pruning source inventory

Canonical/reporting contract after pruning:

`ledger_canonical.csv -> classification_audit.csv / monthly_flow_semantic_split.csv -> monthly_operating_statement.csv -> frontend_metric_series.csv and compact/report presentations`.

Cash contract: `daily_cash_position.csv -> monthly_cash_close.csv`; only `is_frontend_safe=true` rows may feed frontend cash metrics.

Debt contract: debt engine outputs -> `monthly_debt_position.csv`; debt remains a currency-aware stock, outside operating flows.

| Artifact | Grain / columns inspected | Classification | Contract note |
|---|---|---|---|
| `out/run/accounting/latest/ledger_canonical.csv` | transaction ledger; Date, amount, Currency, Flujo, Tipo, parties | canonical_source | Only transaction-level source for semantic classification. |
| `out/run/accounting/latest/classification_audit.csv` | transaction classification with semantic bucket/rule | debug_evidence | Audit evidence and QA input, not a report source. |
| `out/run/accounting/latest/monthly_flow_semantic_split.csv` | month, Currency, Box, semantic bucket/subbucket, amount_in/out | canonical_source | Canonical semantic monthly split; safe source for statement detail metrics. |
| `out/run/accounting/latest/monthly_operating_statement.csv` | month, Currency, statement_line, amount | canonical_source | Authoritative operating monthly source. |
| `out/run/accounting/latest/monthly_cash_close.csv` | month, Currency, Box, close_amount, `is_frontend_safe` | canonical_source | Cash only when frontend-safe rows exist; no fallback reconstruction. |
| `out/run/accounting/latest/monthly_debt_position.csv` | month, Currency, debtor/creditor/component/open_amount | canonical_source | Currency-aware stock source for debt frontier/reporting. |
| `out/run/accounting/latest/per_flow_time_long.freq=M.csv` | monthly raw flow/type rollup | debug_evidence; unsafe_for_frontend | Raw evidence only; not semantic reporting source. |
| `out/run/accounting/latest/box_balance_time_long.freq=M.csv` | inferred box motor balances | debug_evidence; unsafe_for_frontend | Reconciliation/inferred motor only; not real cash. |
| `out/run/accounting/latest/per_party_time_long.freq=M.csv` | actor/internal balance rollup | debug_evidence; unsafe_for_frontend | Internal balance evidence only; not real cash. |
| `out/metrics/latest/metric_contract_frontier.csv` | metric contract, suitability, source_table | canonical_source | Thin frontend/reporting frontier. |
| `out/metrics/latest/frontend_metric_series.csv` | monthly metric rows with Currency/source_table | canonical_source | Frontend-safe series; consumers should use this instead of wide views. |
| `out/metrics/latest/frontier_source_qa.csv` | QA checks for source contract | debug_evidence | Fails/warns for unsafe sources, cash fallback, currency issues. |
| `out/metrics/latest/metric_views/income_statement_monthly_last6.csv` | last-six wide presentation | report_presentation; legacy_compatibility | Now must be derived from `monthly_operating_statement.csv`; not canonical. |
| `out/metrics/latest/metric_views/*wide*` | pivot/wide tables | report_presentation; deprecated_candidate | Human presentation only; automation must not source from these. |
| `out/professional_pack/latest/operating_result/operating_result_monthly.csv` | month-level operating report | report_presentation | Allowed only if built from canonical monthly statement. |
| `out/professional_pack/latest/compact_tables/*.csv` | semester/report compact tables | report_presentation | Must be built from one canonical monthly source and grouped by Currency. |
| `out/debt_resolution/latest/*` | debt matching and balance evidence | debug_evidence | Debt engine source evidence; reporting uses `monthly_debt_position.csv`. |
| `public/accounting/latest/**` | published latest snapshot | report_presentation | Published copy of contract/frontier/report artifacts; not source of truth. |
| `accounting/notebooks/*.ipynb` | analyst presentation notebooks | report_presentation; deprecated_candidate | Must not classify flows or choose raw Stage D sources except diagnostics. |
| `accounting/stage_d/materialize.py` | materialization orchestrator | canonical_source | Owns canonical build sequence. |
| `accounting/marts/semantic.py` | semantic classifier and operating statement | canonical_source | Owns conservative classification, statement, leakage QA. |
| `accounting/metrics/frontier.py` | frontend metric contract/series | canonical_source | Owns source suitability and frontend availability. |
| `accounting/human/*` | document/front/table rendering | report_presentation | Must consume frontend/report-safe tables only. |
| `accounting/publish/latest.py` | copy latest artifacts | legacy_compatibility | Should publish frontier QA aliases and not imply raw artifacts are canonical. |
| `Makefile` | workflow runner | legacy_compatibility | Existing targets preserved; smoke checks should include frontier and operating statement artifacts. |
