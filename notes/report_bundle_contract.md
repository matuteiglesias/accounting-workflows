# Human report bundle contract

Status: current implementation contract for the report product boundary.

## Purpose

`accounting.reports` turns governed accounting artifacts into finished human documents. It is presentation-only: it may select, order, format, validate, and render governed facts, but it may not classify ledger rows, infer cash, reinterpret debt, combine currencies, or create a second accounting engine.

## Boundaries

```text
governed marts / debt / treasury / annual metrics
                    ↓
             accounting.reports
                    ↓
          HTML -> PDF / report manifests
                    ↓
             report_catalog.json
                    ↓
              accounting-viewer
```

HTML is the canonical presentation render. PDF must be derived from that same HTML rather than rendered independently from accounting CSVs.

The viewer boundary is document discovery and delivery. Accounting CSV schemas must not become a viewer runtime API.

## Current reports

### Annual management

Inputs:

- `annual_balance_dashboard_metrics.csv`
- `annual_balance_dashboard_contract.csv`
- `annual_balance_dashboard_qa.csv`

Outputs:

- `annual_management/report.html`
- `annual_management/report.pdf`
- `annual_management/report_manifest.json`
- internal trace/validation CSVs retained under `out/reports`

The renderer uses exact metric selectors and must preserve native-currency separation, stock/flow authority, unavailable-vs-zero semantics, and governed derived results.

### Treasury accountability

Inputs:

- `monthly_cash_accountability.csv`
- `monthly_cash_accountability_qa.csv`

Outputs:

- `treasury_accountability/report.html`
- `treasury_accountability/report.pdf`
- `treasury_accountability/report_manifest.json`
- internal trace/validation CSVs retained under `out/reports`

The renderer presents the governed zero-origin cash control and physical movement reconciliation. It must not relabel inferred control as validated liquidity.

## Schemas

- `accounting_report_manifest.v1`: provenance for one rendered report, including source run, scope, as-of date, logical source fingerprints, output fingerprints, and validation status.
- `accounting_report_catalog.v1`: document-discovery metadata only. It contains report IDs, titles, descriptions, period labels, ordering, and relative HTML/PDF/manifest paths. It contains no metric IDs or accounting values.

## Generated paths

Exact-run product root:

```text
out/reports/<RUN_ID>/
```

Full-run latest pointers include the report surface only after the exact-run report bundle exists and all producer targets have been preflighted:

```text
out/run/accounting/latest_<SCOPE>
out/debt_resolution/latest_<SCOPE>
out/metrics/latest_<SCOPE>
out/reports/latest_<SCOPE>
```

`accounting.support.latest` must preflight every requested target before moving the first pointer. Focused metric-only/light commands retain their narrower latest contracts and do not require reports.

Generated reports are evidence from a run and must not be hand-edited or committed as a substitute for fixing code or source artifacts.

## Publication

Machine accounting artifacts and finished human documents are separate publication contracts:

```text
public/accounting/latest_<SCOPE>/   # governed machine artifact handoff
public/reports/latest_<SCOPE>/      # finished report documents
```

`public/reports` contains only `report_catalog.json` plus catalog-referenced HTML, PDF, and report manifests. Source accounting CSVs, report trace CSVs, caches, and confidential raw evidence must not be copied into the viewer-facing document surface.

Publication requires a PDF for every cataloged report. The downstream viewer should consume `report_catalog.json` and the finished documents only.

## Commands

```bash
make run-reports
make reports-from-run RUN_STAMP=<existing stamp>
make publish-reports
```

PDF rendering uses a local Chromium/Chrome executable. `REPORT_BROWSER_BIN` can explicitly select it when auto-discovery is insufficient.

The full live path is:

```text
run-debt-views
  -> run-dashboard
  -> run-reports
  -> latest preflight/alignment including reports
  -> publish-latest
  -> publish-reports
  -> release-check
```

Historical cutoff runs remain protected by the existing latest guard: building an exact-run report is allowed, but moving ordinary latest pointers to a cutoff/backfill requires the existing deliberate override semantics.

## Invariants

- report paths are relative and cannot escape their bundle root;
- report IDs are unique inside a catalog;
- report inputs for one bundle must share one exact run identity;
- reports preserve source run/scope/as-of provenance;
- unavailable values remain distinguishable from zero;
- ARS and USD remain separate native-currency books unless a separately governed valuation artifact is explicitly selected;
- report generation does not mutate governed source artifacts or the machine `public/accounting` publication contract;
- PDF is derived from the rendered HTML;
- viewer-facing report publication contains no accounting CSVs;
- latest pointers are not mutated until all requested producer targets exist.
