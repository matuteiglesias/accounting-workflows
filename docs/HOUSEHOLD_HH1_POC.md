# Household HH1 — scope-switched reporting POC (2026-11-19)

> **Actualización 19/11/2026:** Claudia (CA) registró dos aportes recibidos/aprobados de ARS 150.000 (29/08 y 04/11). El **run vigente para el POC de octubre** es `20261119T222524Z_HH`, con saldo virtual final **−2.009.394,01 ARS**. Los importes y el run `20261119T211126Z_HH` descritos debajo son el *estado histórico previo al registro*. Ver `docs/HOUSEHOLD_PRODUCT_DISCOVERY_20261119.md` para la conciliación vigente.

## Finding

The production accounting backend already has an explicit Box-universe selector:
- `BOXES="Family Business,Property Management"` → `FBPM`
- `BOXES="Household"` → `HH`

Scope ownership is exact `Box` membership, **not** the payer/receiver name. E.g. `HH -> PM` remains in HH only if its source ledger row has `Box=Household`; the PM mirror remains FBPM. Stage A writes a scope manifest, downstream stages preserve it and use scope-qualified output roots. Never compute HH by filtering published FBPM tables: rerun from source.

### 2026-11-19 proof of computation

Source: current owner-controlled accounting sheet (unchanged by HH1). No live sheet changes were made for this POC.

`20261119T211126Z_HH`, cutoff `2026-10-31`, ingested with `BOXES=Household`, no latest pointer updates. All stages to governed metrics passed:

- Stage A: 264 canonical HH-only transactions; 263 ARS and 1 USD, from 2023-01-05 to 2026-10-26.
- Stage B: 1 scoped debt obligation, no repayment allocations; stage C: monthly treasury 47 rows (HH only); governed metrics: 188 annual rows.
- ARS Jan–Oct 2026: 10 reconciled monthly treasury rows; cash entries 1,177,500; cash out 2,897,813.43; 40 expense-classified transactions. The `closing_control` can be negative because the source ledger has not reconciled all beginning balances; do **not** present it as audited bank cash.
- A full `accounting.reports.build` dry probe (HTML only) failed at `annual_management`: **no comparable rent denominators for cost ratio**. This is a genuine incompatibility, not bad HH data. Existing pack validation also assumes `FBPM` specifically and bans a Household reporting page. Do not bypass those guards globally.

### Compatibility decision

Do **not** force the 26 existing FBPM catalog reports to use HH or synthesize undefined denominators. The scope-aware HH1 producer at `accounting/reports/household_poc.py` deliberately does not execute the patrimonial report bundle; it records **SKIPPED_INCOMPATIBLE** for the 4 core and 28 specialized registered source report definitions. There are 28 registered specialized candidates, not all are included in the 26-report FBPM release; the recorded 32 decisions refer to the registry, **not 32 released documents**.

A future detailed compatibility matrix can enable those renderers whose semantics genuinely apply to HH (e.g. cash accountability) through explicit per-report capabilities. The compatibility gate is per report, not a blanket HH on/off, but this POC supports only the five chart views below.

### New HH1 bundle

The backend writes `out/reports/20261119T211126Z_HH/HH1/`:
- `household_catalog.json` with source run ID, scope, cutoff, 5 view entries, chart SHA-256, explicit skipped reports, and validation totals.
- `charts/*.svg`: **exactly five** self-contained charts, precomputed on the producer side:
  1. Entradas y salidas mensuales ARS (cash accountability, 2026).
  2. Aportes de financiación por mes ARS (cash accountability, 2026).
  3. Salidas de caja por categoría ARS (cash accountability, 2026).
  4. Gastos reconocidos por mes ARS (semantic audit, 2026).
  5. Conteo de movimientos por tipo semántico (semantic audit, 2026).

Pseudonym mapping or additional household actor inference **not attempted**. Some services are categorized `no_cash_in_box_direct_payment`, but gross classified expenses and treasury outflows can have distinct economic meaning even when numerically equal. Frontend must preserve this caveat; no unverified owner-specific financial story.

Regenerate:
```bash
python3 -m accounting.reports.household_poc \
  --run-root out/run/accounting/20261119T211126Z_HH \
  --out-dir out/reports/<new-HH-run-id>/HH1 \
  --cutoff 2026-10-31
```
For another real HH run, use a new exact run ID and an empty destination. This module refuses mixed-Box sources, inconsistent month input, unreconciled treasury, future dates, duplicate tx IDs, incompatible scope and overwritten bundles.

The consumer `accounting-viewer/scripts/build_hh1.py` accepts only this catalog and hash-verified five SVGs. It copies them under `/HH1/`, builds a basic five-tab static page, and leaves the root `/reports/` library untouched. No CSV, account history or private notes are copied. The frontend is a static consumer, not another accounting engine.

### Separate publication gates

1. Verify origin source and `HH` scope in the source-owned catalog and the five charts.
2. Run targeted backend scope tests and viewer tests, then build the existing root viewer separately from approved FBPM reports.
3. Add HH1 to a *clean* staged directory; verify expected file list, no extra private files, no CSV and that the root remains unchanged.
4. If previewing on Vercel, use a clean isolated root and keep the full-domain Vercel Authentication protection. A Vercel `--skip-domain` preview is not a production promotion.
5. Do not move `latest_FBPM`, `latest_HH`, or publish `public/reports/latest_HH` as if all 26 reports were compatible.
6. Independent S1 and L0 work in `accounting-viewer` is concurrent; do not stage those changes in HH1 commits.

### Hand-off

Potential extension for another agent: a report contract `supported_scopes: [FBPM, HH]` at each report, with explicit capability checks (e.g. `requires_rent_denominator`, `requires_property_opex`, `requires_pm_actor`), stable nonfatal SKIP logging, and a scope-aware report bundle catalog. This should be deferred until HH1's five charts are accepted; avoid replicating 26 renderer forks.
