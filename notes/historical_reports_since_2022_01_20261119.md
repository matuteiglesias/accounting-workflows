# Historical reports: full monthly horizon from January 2022

**2026-11-19 · Implemented and locally verified · NOT published.**

## Decision and invariant

For **historical line/bar series and corresponding monthly tables**, begin the
visible calendar at **2022-01**, not the last 12/24 months or only the current
year. This is a display/reporting policy; it does NOT backfill paid transactions,
change contract authority, invent observed debt/costs, or imply bank liquidity.

Single source: `accounting.reports.time_series.HISTORY_START_MONTH = "2022-01"`.
Treasury `START_PERIOD` reuses this. Charts use years as sparse x-axis ticks,
not 60 overlapping monthly labels, while the tables retain the actual monthly
grain. No data is converted across native currencies.

## Implemented report changes

| Report | Monthly plot(s) | Monthly table(s) | Missing/history rule |
| --- | --- | --- | --- |
| Treasury Accountability | Existing net movement + control chart (full 2022–present), TS04 Box cash in/out (removed 12m tail). | Treasury 2022–cutoff already existed; preserved and added muted pre-inception status | Before first recorded Box month, don't draw a 0-cash point/line; zero-origin control in tables is explicitly a convention. |
| Rent Monthly Evolution | TS03 2022–last fully closed month, per currency; source results may start later | `all_history_months`: 2022–cutoff, explicit every month, "Sin registro" where governed source absent | No invented rent amount; historical 2026 month partial stays in table but not the historical chart. |
| OPEX Monthly Evolution | New TS11 using the SAME SVG component, 2022–last fully closed month | `all_history_months`: Jan 2022–cutoff, explicitly missing months | Jan 2022 OPEX absent in checked source; do NOT silently claim zero operating cost. |
| Annual Management | TS06 monthly net operating vs recorded distributions, now Jan 2022–last completed month (no 12m tail) | Annual management's annual tables were already 2022 onward | No quarterly/day granularity, no mixing of cash net and operating result. |
| Debt Accountability | TS07 position chart now has Jan 2022 axis (no 24m tail) | New monthly stock appendix from Jan 2022, per currency with blank pre-2023 balances | First governed debt position **May 2023**; Jan 2022–Apr 2023 are **16 months of "Sin registro"**, not verified zero debt. |
| Short-term Sustainability TS09 | **Unchanged:** 6 historical + 6 forward months, with Nov 2026 current/partial at forward position 1 | **Unchanged:** 6+6 decision window | Explicit exception, not a full historical report. |

The specialized reports keep their source-governed selections and internal
lineage. They do not reclassify rows. For monthly evolution, the new full-range
tables replace the redundant little bars with a large time-series panel plus
the exhaustive 2022–cutoff table.

## Known source coverage on checked run

Pinned existing materialization: `20261119T043349Z_FBPM`.

- FB ARS treasury: 2022-01–2026-11.
- PM ARS treasury: recorded starting 2023-05; history before that appears with muted calendar convention.
- PM USD treasury: recorded from 2023-11; no claims for earlier months.
- Monthly rent ARS: source from 2022-01; rent USD from 2024-03.
- OPEX monthly ARS: source from 2022-02, with missing months.
- Debt stock USD: source from 2023-05.
- TS01/TS02 annual comparisons already span 2022–2026, same-completed-month windows.

## Rendered, reviewed, PRIVATE previews

Directory (locally on authorized desktop):

```text
~/Desktop/MEMOS/10_accounting_family/08_scenarios/chart_previews/
  20261119T043349Z_FBPM_full_history_2022/
    treasury_accountability/report.{html,pdf}
    annual_management/report.{html,pdf}
    rent_monthly_evolution/report.{html,pdf}
    opex_monthly_evolution/report.{html,pdf}
    debt_accountability/report.{html,pdf}
```

PDF validation: Treasury 15 landscape A4 pages, Annual 7 portrait A4,
Rent 7 letter pages, OPEX 4 letter pages, Debt 10 letter pages.
First-page charts inspected visually; no overlapping x labels. Table/text
extraction confirms 2022-01 entries or explicitly no-source statuses.
These previews use a pinned old run, NOT new forecast plans or a production
publication and are NOT in Accounting Viewer.

## Execution/testing

- `tests/test_report_history_start_2022.py` guards the full-history policy and
  no-fabricated-zero distinction.
- Existing annual, specialized, treasury, debt and SVG tests also cover
  source/population contracts.
- No ledger mutation, no Viewer/deployment publish, no new data mart or
  daily/weekly granularity.
- Future short-term scenario reports remain 6+6. Annual tables remain
  annual, but anchored on reporting year 2022 rather than implying monthly
  rows that do not exist.
