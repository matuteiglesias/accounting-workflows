# Semantic classifier PR 1

## Implemented rules

The first-pass classifier is deterministic and conservative. It maps rent collections to `operating_revenue / rent`, explicit tax/service/maintenance/legal types to `property_opex`, explicit contribution flows/types to `funding_contribution`, explicit loan/repayment/interest types to `debt_movement`, and explicit dividend/personal-expense/transfer-expense signals to `family_withdrawal_candidate`.

Anything that does not match those rules is emitted as `unknown / review_required` with low confidence.

## Outputs

The Stage D materialization step now writes these backend-owned semantic artifacts in the accounting run output directory:

- `classification_audit.csv`
- `classification_audit_summary.csv`
- `monthly_flow_semantic_split.csv`
- `classification_validation.csv`

For run-mode executions, the existing latest symlink mechanism makes these available under `out/run/accounting/latest/` after latest is updated.

## Review-required flows

Unknown flows remain explicit rather than being forced into operating costs. The classifier also flags family/informal withdrawal candidates for review when transactions look like personal expenses, dividends, or transfer-to-expense outflows.

## Deliberate non-changes

This PR does not change existing metrics, notebooks, public report outputs, frontend contracts, or legacy view names. The new mart is additive and intended to become the canonical source for future metrics/reporting changes after accounting review.
