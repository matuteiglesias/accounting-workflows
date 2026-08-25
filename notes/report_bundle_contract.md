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
          HTML / PDF / report manifests
                    ↓
             report_catalog.json
                    ↓
              accounting-viewer
```

The viewer boundary is document discovery and delivery. Accounting CSV schemas must not become a viewer runtime API.

## Schemas

- `accounting_report_manifest.v1`: provenance for one rendered report, including source run, scope, as-of date, source fingerprints, output fingerprints, and validation status.
- `accounting_report_catalog.v1`: document-discovery metadata only. It contains report IDs, titles, descriptions, period labels, ordering, and relative HTML/PDF/manifest paths. It contains no metric IDs or accounting values.

## Generated paths

The intended product root is `out/reports/<RUN_ID>/`. Generated reports are evidence from a run and must not be hand-edited or committed as a substitute for fixing code or source artifacts.

## Invariants

- report paths are relative and cannot escape their bundle root;
- report IDs are unique inside a catalog;
- reports preserve source run/scope/as-of provenance;
- unavailable values remain distinguishable from zero;
- ARS and USD remain separate native-currency books unless a separately governed valuation artifact is explicitly selected;
- report generation does not mutate governed source artifacts or the machine `public/accounting` publication contract.
