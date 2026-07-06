# Professional drilldown merge readiness

## Branch state

- Branch: `work`
- Commit range audited: `a4de327...HEAD`
- Recent commits in scope:
  - `aa6d0a2 Add professional flow drilldowns`
  - `205013a Harden professional drilldown acceptance coverage`
  - `e82e2a7 Add derived professional drilldowns`
- Files changed summary from `git diff --stat a4de327...HEAD`:
  - `Makefile`
  - `accounting/marts/semantic.py`
  - `accounting/professional/__init__.py`
  - `accounting/professional/drilldown.py`
  - `accounting/professional/render_linked_digest.py`
  - `notes/professional_clickable_drilldowns.md`
  - `notes/professional_derived_drilldown_audit.md`
  - `notes/professional_drilldown_lineage_audit.md`
  - `tests/test_professional_drilldowns.py`

## Commands run

- `git status --short`
- `git branch --show-current`
- `git log --oneline -10`
- `git diff --stat a4de327...HEAD`
- `git diff --check a4de327...HEAD`
- `python -m compileall accounting/professional accounting/marts/semantic.py`
- `python -m pytest -q`
- `make smoke`
- `python -m accounting.professional.drilldown --help >/tmp/dd_help`
- `python -m accounting.professional.render_linked_digest --help >/tmp/digest_help`
- `if [ -n "${ACCOUNT_SHEET_URL:-}" ]; then make run-all; else echo 'ACCOUNT_SHEET_URL not set; run-all environment-blocked'; fi`
- `rg -n "professional_drilldown" public/accounting/latest || true` when public bundle exists; current environment has no `public/accounting/latest`.
- `rg -n "ledger_canonical" public/accounting/latest || true` when public bundle exists; current environment has no `public/accounting/latest`.
- `rg -n "classification_audit" public/accounting/latest || true` when public bundle exists; current environment has no `public/accounting/latest`.
- `git ls-files out public | head`
- `git diff a4de327...HEAD -- accounting/marts/semantic.py`

## Results

- `git diff --check`: pass.
- Python compile: pass.
- Unit tests: pass, `5 passed`.
- `make smoke`: pass.
- CLI help for both professional modules: pass.
- `make run-all`: environment-blocked because `ACCOUNT_SHEET_URL` is not set in this agent environment.
- Real-pack build: pending local execution because this environment does not contain `out/professional_pack/latest` or `out/run/accounting/latest`.
- Public leakage check: no public bundle exists in this environment; no generated `out/` or `public/` files are tracked by git.
- Semantic diff review: limited to dtype-safe FX validation text aggregation in `_build_validation_rows`; no classification rule changes.

## Generated artifact paths

When run against a real professional pack, the builder writes:

```text
out/professional_pack/latest/drilldown/professional_drilldown_index.csv
out/professional_pack/latest/drilldown/professional_drilldown_manifest.json
out/professional_pack/latest/drilldown/professional_drilldown_qa.csv
out/professional_pack/latest/drilldown/details/*.csv
out/professional_pack/latest/drilldown/details/*.html
out/professional_pack/latest/digest/accounting_professional_pack_digest_linked.html
```

## Supported table IDs

PR 1 flow tables:

```text
monthly_tables_flow_bucket_all_measures
monthly_tables_flow_subbucket_all_measures
monthly_tables_draws_by_box_amount_out
monthly_tables_draws_by_type_amount_out
monthly_tables_fb_bridge_matrix
monthly_tables_pm_stress_matrix
monthly_tables_household_bridge_matrix
monthly_tables_opex_by_type_amount_out
monthly_tables_fx_treasury_compact
monthly_tables_unknown_review_net_matrix
```

PR 2 derived/statement tables:

```text
monthly_tables_operating_statement_matrix
monthly_tables_operating_statement_matrix_ars
overview_balance_dashboard
income_operating_statement
cash_annual_box_flow_bridge_wide
```

## Unsupported categories and caveats

- Unsupported cells remain plain numbers in the linked digest.
- Missing `Currency` is unsupported to avoid cross-currency aggregation.
- Stock/cash/debt rows are not treated as flow ledger drilldowns.
- Cash close, validated cash levels, and diagnostic box balances are unsupported in the cash annual bridge.
- `FB-related` is broader than `Box = Family Business`; the caveat is shown in the relevant detail rows.
- FX fallback detection remains in place until `treasury_fx` semantics are fully stable.
- Professional drilldowns are internal/professional pack artifacts only; they are not written into the public bundle.

## Merge readiness conclusion

Synthetic acceptance, compile, smoke, CLI, and safety checks pass in this environment. Real-pack drilldown generation and `make run-all` remain pending for Matías's local/live environment because required live artifacts and `ACCOUNT_SHEET_URL` are not available here.
