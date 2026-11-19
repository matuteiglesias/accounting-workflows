# Phase A: Monthly policy input series (2026-11-19)

Status: validated exploratory derivative, NOT production-distribution authority.

## Source and execution

- Canonical evidence: **Contab Torcuato / C. Long Ledger**.
- Pinned fresh ingestion: `20261119T061512Z_FBPM`, all-status SHA-256 `15836867b3d777e31f438ff586b753d185bf28c82e538f9a6831fb020d9e7b7f`.
- Horizon: observed through 2026-11-19; planned through 2027-03-31.
- Source to scenario: existing `accounting.scenarios.projection.build_projection_ledger` admits paid rows and explicit future planned rows only; all other statuses stay out.
- Main accounting materializer, debt and metrics stages run in isolated workspace; all temporary stages are removed after export. No scenario operation or distribution is applied.
- Collector attribution from **G. Governance Flows** rows 10–15 (owner reported, unverified, distinct from legal entitlement and physical custody).
- Local agreement proposal mapping: `notes/fb_primos_alejandro_attribution_20261119.yaml`. It uses ledger **tags**, not cross-property tax-payer guesses.
- Output: `~/Desktop/MEMOS/10_accounting_family/08_scenarios/policy_inputs/20261119T061512Z_FBPM_pre_policy/`.

## Output contract

1. `monthly_policy_inputs.csv`: each month × Box × currency; existing monthly `rent_in`, `net_cash_flow`, separately computed pre-distribution cash net and operating surplus, eligible scenario payout bases (zero-floor 100%, 50%), next 1/2/3 calendar months of planned cash operating costs and a cash-flow-based reserve requirement for each horizon.
2. `monthly_agreement_settlement_inputs.csv`: Primos and Alejandro reported direct-collector incomes; specifically excluded Taipe income; unassigned rental lanes, candidate expenses financed by named actors and costs recorded to other parties; equal-share settlement only when complete.
3. `policy_input_sources.csv`: each included direct cash planned-cost and attributed-rent transaction ID with its group.
4. `policy_inputs_manifest.json`: pinned input hashes, horizon, definitions, provenance and unresolved-month count.

## Financial semantics and safeguards

- Monthly `net_cash_flow` can include funding, transfers, debt and distributions; it is **not** automatically distributable income.
- `pre_distribution_cash_net` removes recorded `personal_draws_out` and `dividends_out` from the net *for input-policy purposes*, avoiding circular payout formulas.
- `pre_distribution_operating_surplus = rent_in - taxes_out - services_out - maintenance_out - legal_out`. This deliberately differs from the cash net. Only physically Box-paid amounts enter those treasury `_out` columns.
- `cost_due_next_Nm` sums *explicit future plans* classified by the existing treasury backend as physical cash expenses, due in the **following N whole months**. Does not include modeled unrecorded obligations, debts, third-party payments or legal contingencies.
- The `*_complete` boolean means only that the **calendar horizon is within the requested projection through-date**. It does **not** certify that all future liabilities or rents are known.
- `cashflow_reserve_required_next_Nm` is the largest cash shortfall in the cumulative **projected net cash-flow sequence** for the following N months, starting with zero incremental liquidity. It assumes forecast receipts arrive as scheduled and uses month-grain, not exact due-date timing.
- No opening bank/cash balance is verified; a cumulative FB Box control balance is **not** bank liquidity. FB itself is an aggregate reporting scope, not a physically reconciled common cash box.
- Provisional collection map: Kiosco/Vidriero to Primos; Correa to Alejandro; Técnico split 50/50. Taipe is removed from the proposed split because of the **reported** offset related to the father's wife's use. The offset is **not** a verified debt or quantified market rental.
- Fernando has no reported collector in the inspected governance table. Those months have `unattributed_collections` and no final settlement value.
- Expenses attributed simply to `FB`, `Inq` or other payers are not assigned to Primos/Alejandro by guess. Such months have `unallocated_candidate_expenses_paid` and no final settlement value.
- `illustrative_transfer_before_unallocated_costs` is *never* a legally payable amount. Even a non-null `settlement_primos_to_alejandro` is conditional on the owner-reported arrangement and does not prove signed agreement, entitlement, or actual personal custody.
- Money stays separated by currency and Box. No netting FB versus PM is permitted without explicit, traceable counterpart transactions.

## Reproduce

```bash
cd ~/repos/accounting-workflows
PYTHONPATH=. python3 -m scripts.run_policy_inputs \
  --base-run-id 20261119T061512Z_FBPM \
  --as-of-date 2026-11-19 --through-date 2027-03-31 \
  --start-month 2026-09 \
  --agreement notes/fb_primos_alejandro_attribution_20261119.yaml \
  --out-dir /path/to/new_empty_output_directory
```

No new distributions were written to the source ledger; policies consume these Phase A outputs in a separate future step.
