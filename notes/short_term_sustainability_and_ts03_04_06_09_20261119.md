# Second wave of time-series charts and short-term cash-flow report

**2026-11-19 | Implemented, locally tested; not published to Viewer.**

## What was added

- **TS03** — Monthly rent recognized in the existing specialized **`rent_monthly_evolution`** report. One governed time-series chart per currency (ARS and USD separate), alongside the original summary/bars/table. The current incomplete month is excluded from the prominent plot but the existing report table retains its own governed source view.
- **TS04** — Physical monthly inflows and outflows, per Box and currency, in the existing **`treasury_accountability`** report, above its already existing net-flow/cumulative-control chart and transaction table. Most recent twelve complete months; physical cash movements include non-operating funding/debt and are not rental or savings measures.
- **TS06** — Monthly operating result versus recorded personal draws/distributions in **`annual_management`** page 3. It fills a previously unused design slot. Difference between two series **does not mean bank savings**, distributable money, legal entitlement, or actor responsibility.
- **TS09** — Separate, private report: `accounting/reports/short_term_sustainability/render.py`. **Six historical months (May–October 2026), then six forward months (November 2026–April 2027).** November counts as **first forward month** but is **current/partial**: only recorded transactions through the evaluation date, NOT a forecast of the entire month. December–March are explicit planned ledger cash movements. **April is unplanned**, rendered missing, not zero. Reuses the same `TimeSeriesSpec` renderer (now supports a declared monthly domain and a distinct current/partial phase).

## TS09 report sections

For **PM ARS**, **FB ARS**, and **PM USD** (where full historical monthly population exists):

1. Monthly cash inflows/outflows, visually distinguishing observed, current partial, and future planned.
2. Monthly net movement, including possible deficits.
3. Accounting control position, **never represented as real cash/bank balance**.
4. Monthly table of inflows, outflows, **Box-paid operating costs (OPEX)**, net, accounting control, and completeness phase.
5. Conditional liquidity buffer: maximum monthly-flow cash deficit from **December onward** through the last consecutively planned month, using zero **incremental** starting liquidity after November. Already-paid November receipts are **not** recounted as future funding.
6. A distinct **unavailable** section for unverified overdue bills, without setting them to zero.

TS09 uses exclusively an isolated `accounting.policy_inputs.v1` export pinned to the exact same all-status base hash. No mixed accounting snapshots, no direct editor in Viewer, no changes to historical paid ledger, no automatic bank reconciliation.

**Interpretation of PM ARS example:** With 2026-11-19 evaluation and plans through 2027-03, incremental liquidity needed from December's *monthly planned net flows* is **ARS 4,126,451**. This excludes an unknown bank/cash opening balance, intramonth due dates, amounts whose payer is another party, omitted future obligations and overdue bills. **Six-month sustainability is not established** because 2027-04 has no current plans and November itself is partial.

## Private previews

Source runs:
- Existing historical reports: `20261119T043349Z_FBPM`, after accounting paid-only materialization. Source snapshot is older than the freshly ingested future-tax plan; not for new forward cash forecast.
- TS09 future plans: fresh `20261119T061512Z_FBPM` (978 paid + 52 valid planned through March 2027), isolated materialization with output through April 2027.

Previews on desktop:

```text
~/Desktop/MEMOS/10_accounting_family/08_scenarios/chart_previews/
  20261119T043349Z_FBPM_TS03_TS04_TS06/
    rent_monthly_evolution/report.{html,pdf}
    treasury_accountability/report.{html,pdf}
    annual_management/report.{html,pdf}
  20261119T061512Z_FBPM_TS09_6plus6/
    report.{html,pdf}
    time_series_trace.csv
    monthly_sustainability_table.csv
    report_manifest.json
```

Phase-A source used by TS09:
`~/Desktop/MEMOS/10_accounting_family/08_scenarios/policy_inputs/20261119T061512Z_FBPM_6m_sustainability/`.

Reproduce TS09:

```bash
cd ~/repos/accounting-workflows
PYTHONPATH=. python3 -m accounting.reports.short_term_sustainability.render \
  --run-root out/run/accounting/20261119T061512Z_FBPM \
  --policy-root ~/Desktop/MEMOS/10_accounting_family/08_scenarios/policy_inputs/20261119T061512Z_FBPM_6m_sustainability \
  --out-dir /path/to/private/preview \
  --as-of-date 2026-11-19 \
  --browser-bin /usr/bin/google-chrome
```

To refresh the policy-input source after the ledger changes, run `scripts.run_policy_inputs` with `start_month=2026-05`, `as_of_date=2026-11-19`, `through_date=2027-04-30`, an **explicit new output directory**, and the separately documented provisional agreement mapping. Do not use a stale output against a changed base hash.

## Quality boundary and further work

- 67 focused tests passed (TS series, Phase A, reports, catalog/publication contracts). SVG/PDF rendered with Chrome and visually checked. TS06 annual report now requires 7 A4 pages; Treasury preview 10 A4-landscape pages; specialized rent 3 pages; TS09 private 8 A4-landscape pages.
- Original tables, pie charts and table-adjacent microcharts were retained.
- No daily or weekly modeling; only months and years.
- **No evidence added** on overdue bills, actual bank accounts, physical custody, whether a cost is legally shared, or the remaining 2027 rent/tax plans.
- New private TS09 report is **not in the Viewer publication catalog**. It must stay private until decision-quality planning evidence and approval. Once future coverage is complete and reviewed, adding its final HTML/PDF as a cataloged report is a separate, small change.
- Remaining TS05 already exists; TS08 and TS10 still require stronger actor attribution; no general metric-calculation DSL was introduced.
