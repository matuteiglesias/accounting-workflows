# Artifact map and source-of-truth guide

This is a **navigation map**, not a new report registry and not a snapshot certification. Inspect the exact present files before using them. No personal accounting numbers or private document copies belong here.

## Repository map

**Governed reports and charts — here:**
- `accounting/reports/build.py` — entrypoint and report orchestration.
- `accounting/reports/time_series.py` and `accounting/reports/charts.py` — reusable chart rendering.
- `accounting/reports/annual_management/`, `treasury_accountability/`, `treasury_transaction_tape/`, `debt_accountability/` — main report families.
- `accounting/reports/specialized/` — rent/OPEX/property/recipient/category reports.
- `accounting/reports/payment_evidence.py` — payment-proof report. Evidence approval and links originate upstream; report is not an evidence approver.
- `accounting/reports/short_term_sustainability/` — conditional short-term policy view; not long-history plot.
- `accounting/reports/private_bundle.py` — compose private approved evidence report with ordinary catalog, preserving attachment hashes/allowlist.
- `accounting/reports/catalog.py`, `manifest.py`, `pdf.py`, `publish.py` — product contracts and rendering/publication gates.

**Source contracts and audited guidance:**
- `AGENTS.md` and `SYSTEM.yaml` — mandatory.
- `notes/report_bundle_contract.md` — HTML/PDF, catalog, private evidence and publication boundary.
- `notes/report_product_ownership_20261118.md` — source/producer attribution and its known gaps; avoid inventing another registry.
- `notes/historical_reports_since_2022_01_20261119.md` — Jan-2022 history policy and explicit missingness.
- `notes/short_term_sustainability_and_ts03_04_06_09_20261119.md` — TS03/04/06/09 and partial/planned limits; consult newer note when available.
- `notes/g02_chino_payment_custody_and_distribution_policy_20261119.md` — **historical example of a false narrative induced by modeling an FX sale that did not occur.** This is a reasoning lesson, not permission to share private case details publicly.

**Viewer / delivery (separate repo):**
- `accounting-viewer/src/editorial.json` (may be untracked/in development).
- `accounting-viewer/src/index.template.html`, `viewer.css`, `scripts/build_site.py`, `scripts/accept_site.py`.
- The editorial reading journey: **Panorama → Administración y tesorería → Rendición de cuentas → Patrimonio y decisiones**. A report may appear in multiple chapters through one catalog entry. Respect existing worktree modifications.
- A report outside the approved catalog may exist as a private/local preview; do not say it is deployed.

## Questions and report portfolio

| Question / consumer | Existing product(s) | Common failure to guard |
| --- | --- | --- |
| What produces income and what does the operation cost? | Annual management, rent by property/month, OPEX by category/month | Nominal ARS ≠ real growth; partial year, availability and missing months matter. |
| How does money move, and what is actually available? | Treasury Accountability, transaction tape, physical inflow/outflow breakdowns | Zero-origin control ≠ physically reconciled cash; stock ≠ flow. |
| What distributions and obligations are documented? | Distributions, accountability cycles, debt positions/repayments | Recorded movement ≠ legal entitlement; payer and obligated person differ. |
| What backs up a paid movement? | `pm_payment_evidence` report, approved private evidence bundle | Missing proof ≠ unpaid. Boleta ≠ payment proof. Historic evidence cutoff may be older than current ledger/report cutoff. |
| What is sustainable in coming months? | TS09, G02 private scenario report | Hypothetical funding ≠ actual liquidity; distinguish current partial / planned / not planned. |
| What can stakeholders understand or decide? | Viewer editorial + selected report exhibits; governance A0–A17 | Avoid publishing legal strategy or importing confidential conclusions into a public view. |

## Tested visualization invariants

- **History policy:** Jan 2022 calendar for historical monthly charts/tables; real data can start later, and missing is not zero. TS09 explicitly retains 6 months history + 6 prospective.
- **Native currencies:** ARS and USD controls must be separate. FX requires evidence of an actual currency conversion, not a convenient story about an equivalent amount.
- **Partial periods:** current partial month is distinguishable from last complete month and forecasts; planned months without scheduled data are not zero.
- **Transaction integrity:** figures, tables and report trace rows should agree for one exact source run. No double counting mirrored actors/Boxes.
- **Reporter design:** HTML and PDF are two views of the same generated report; verify actual rendered PDF pages, not just unit tests or screenshot of template.
- **Privacy and temporal claims:** viewer report cutoff, proof snapshot cutoff, private forecast date and active deployed catalog can differ. Every exhibit must state which one it uses.

## Useful checks (read first, run only in safe fixture mode)

- `python3 -m pytest -q tests/test_time_series_reporting.py tests/test_report_charts.py tests/test_report_history_start_2022.py`
- `python3 -m pytest -q tests/test_annual_management_report.py tests/test_treasury_accountability_report.py tests/test_payment_evidence_report.py`
- `python3 -m pytest -q tests/test_report_publication_boundary.py tests/test_private_bundle.py tests/test_report_product_ownership_audit.py`
- In the Viewer repo, `make test`, `scripts/build_site.py --help`, `scripts/accept_site.py --help` (read the commands before writing outputs).

Do not require all suites for every small copy change. Select tests based on the affected source, then smoke/inspect the **exact changed artifact**.

## Private context boundary

This repo is public. Keep stakeholder specifics, meeting plans, legal correspondence, bank/custody details and real report paths in `ops/report-director/private/`, which is ignored by Git. The directory should point to private materials, not duplicate them. If the new thread cannot access those files, work from sanitized report/public artifacts and state the scope limitation; **do not fabricate private context**.
