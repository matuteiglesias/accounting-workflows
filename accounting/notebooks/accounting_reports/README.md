# Governed accounting reports — minimal split

Copy into the repo:

```bash
mkdir -p accounting/notebooks/accounting_reports

cp _shared.py accounting/notebooks/accounting_reports/_shared.py
cp 01_balance_dashboard_overview.ipynb accounting/notebooks/accounting_reports/
cp 03_income_rent_and_operations.ipynb accounting/notebooks/accounting_reports/
cp 04_debt_open_items_and_reconciliation.ipynb accounting/notebooks/accounting_reports/
```

Run from repo root or from inside `accounting/notebooks/accounting_reports`.

Expected outputs:

```text
out/professional_pack/latest/html/
out/professional_pack/latest/markdown/
out/professional_pack/latest/tables/
out/professional_pack/latest/qa/
```

Design rules:

- Do not recalculate backend core logic.
- Do not infer cash.
- Do not sum ARS + USD.
- Keep `Currency` next to line/metric.
- Treat unavailable as `s/d`, never as zero.
- Keep debt separate from OPEX.
- Keep funding separate from operating revenue.
