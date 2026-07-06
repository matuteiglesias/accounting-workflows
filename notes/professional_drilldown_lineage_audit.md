# Professional drilldown lineage audit

This audit captures the current legacy drilldown shape and the PR 1 professional target shape.

## Legacy metric drilldowns

Existing metric drilldowns are produced by `accounting.metrics.drilldown.build_metric_drilldown_artifacts`, called from the metrics build pipeline. The legacy index is keyed by `(metric_id, period_grain, period, currency)` and supports only `IS.RENT.TOTAL`, `IS.OPEX.TOTAL`, and `IS.DRAWS.PERSONAL`.

The legacy index columns are:

```text
run_id, metric_id, period_grain, period, currency, source_table, filter_json,
detail_csv_relpath, detail_html_relpath, matched_rows, matched_value_sum,
target_metric_value, difference_vs_target, status
```

## Semantic source contracts

`classification_audit.csv` is the transaction-level semantic classification audit. The semantic mart defines these columns:

```text
tx_id, Date, period, period_end, Currency, amount, Box, Lugar, payer, receiver,
Flujo, Tipo, Detalle, semantic_bucket, semantic_subbucket, direction,
direction_source, direction_confidence, actor, counterparty, channel, cash_path,
rule_id, rule_version, classification_confidence, classification_status,
review_required, warning, notes
```

`monthly_flow_semantic_split.csv` is the monthly semantic reconciliation layer. The semantic mart defines these columns:

```text
period, period_end, Currency, Box, Lugar, actor, counterparty, payer, receiver,
channel, cash_path, semantic_bucket, semantic_subbucket, amount_in, amount_out,
net_amount, amount_abs, n_tx, classification_status, classification_confidence,
review_required, source_table, source_tx_ids_sample, rule_ids, notes
```

## PR 1 supported table families

The first professional scope covers monthly wide tables generated under `out/professional_pack/latest/tables`:

```text
monthly_tables_flow_bucket_all_measures.csv
monthly_tables_flow_subbucket_all_measures.csv
monthly_tables_draws_by_box_amount_out.csv
monthly_tables_draws_by_type_amount_out.csv
monthly_tables_fb_bridge_matrix.csv
monthly_tables_pm_stress_matrix.csv
monthly_tables_household_bridge_matrix.csv
monthly_tables_opex_by_type_amount_out.csv
monthly_tables_fx_treasury_compact.csv
monthly_tables_unknown_review_net_matrix.csv
```

Cells can reconcile exactly when their row context maps to a period/currency/box/semantic filter over `monthly_flow_semantic_split.csv` and the resulting sum equals the displayed cell within tolerance. Cells can descend to `classification_audit.csv` when the same semantic filter applies at transaction grain or `source_tx_ids_sample` exposes tx ids.

Unsupported or non-reconciled cells must remain non-clickable. The QA file records whether a cell was `ok`, `empty`, `residual_warning`, `unsupported`, or `error`.
