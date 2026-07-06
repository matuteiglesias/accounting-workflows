# Professional clickable drilldowns

This PR adds a post-notebook professional drilldown layer for monthly flow tables.

## Contract

The builder reads professional wide tables from `out/professional_pack/latest/tables`, reconciles monthly cells against `monthly_flow_semantic_split.csv`, and writes:

```text
out/professional_pack/latest/drilldown/professional_drilldown_index.csv
out/professional_pack/latest/drilldown/professional_drilldown_manifest.json
out/professional_pack/latest/drilldown/professional_drilldown_qa.csv
out/professional_pack/latest/drilldown/details/*.csv
out/professional_pack/latest/drilldown/details/*.html
```

A cell is linkable only when:

```text
status == ok
matched_rows > 0
abs(residual) <= tolerance
```

The linked digest renderer is intentionally passive: it reads the index and renders links only for already-approved cells. It does not classify transactions, infer cash, invent debt, or sum currencies.

## Commands

```bash
python -m accounting.professional.drilldown \
  --repo-root . \
  --pack out/professional_pack/latest \
  --run-root out/run/accounting/latest

python -m accounting.professional.render_linked_digest \
  --repo-root . \
  --pack out/professional_pack/latest
```


## Derived / statement drilldowns

PR 2 extends the same index/detail/linking contract to derived tables. Derived pages reconcile first to `monthly_operating_statement.csv` or `annual_balance_dashboard_metrics.csv`; semantic and classification rows are shown as supporting layers when available. Stock/cash rows remain unsupported rather than being linked as flow ledger drilldowns.
