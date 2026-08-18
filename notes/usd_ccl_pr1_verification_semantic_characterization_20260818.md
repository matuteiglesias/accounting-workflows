# USD/CCL PR1 closeout and semantic characterization

Date: 2026-08-18

Scope: fixture-safe verification and characterization only

Accounting rule changed: no

Live inputs accessed: no

Publication performed: no

## PR1 verification status

Commit `49be7ff` passes its committed-patch check and all requested fixture-safe
gates after provisioning the repository's documented CI dependencies. The
valuation target remains opt-in. Static inspection found no dependency from
`run-canonical`, `run-full`, ingest, publication, or latest-link targets to
`smoke-usd-ccl-valuation` or `accounting.valuation`. The existing canonical
artifact contract cases are unchanged; the valuation artifacts have separate
`derived_valuation` contracts.

Commands and observed results:

| Command | Result |
| --- | --- |
| `git show --check --oneline 49be7ff` | pass |
| `make doctor` | pass |
| `make validate` | pass; 65-test PR1 baseline, then 85-test characterization suite |
| `make smoke-usd-ccl-valuation OUT=/tmp/accounting-usd-ccl-smoke` | pass; isolated synthetic artifacts |
| `make smoke-core OUT=/tmp/accounting-native-smoke` | pass |
| `make smoke-full OUT=/tmp/accounting-native-smoke-full` | pass; fixture inputs and dry-run publication only |

The native smoke runs emitted existing warnings about unmatched box rows and a
`datetime.utcnow()` deprecation. They did not fail, and this task does not mix
those unrelated issues into valuation work.

## Environment discrepancy root cause

There was no Python-interpreter split:

- `python` and Make's `python3` both resolved through pyenv to Python 3.12.13;
- both initially had `pytest` and neither initially had `pandas`;
- the focused valuation test passed because neither that test nor the valuation
  module imports `pandas`;
- `make validate` exercises the native pipeline, which imports `pandas`.

The repository has no dependency manifest. Its CI explicitly installs
`pandas pytest`, and `notes/environment_bootstrap.md` describes iterative local
dependency installation. Installing `pandas` into the active pyenv interpreter
therefore provisioned the expected environment; no Makefile or dependency-policy
change was justified.

## Native smoke regression status

The optional valuation stage did not alter or become a prerequisite of the
native path:

1. `run-canonical` still depends on the existing marts path.
2. `run-full` still composes the existing canonical/report/package/publish path.
3. ingest has no valuation or rate-artifact requirement.
4. publication has explicit artifact lists and does not select the valuation
   sidecar implicitly.
5. canonical ledger artifacts retain their prior roles and authorities.

This is structural and smoke evidence, not a live-number validation. No live
ledger, rate source, publication bundle, or latest link was accessed or changed.

## Amount semantics

### Documented contract

`accounting/ledger/ingest.py` describes `amount` as a signed amount. In tension
with that description, the operating runbook states that accepted money values
must be nonnegative. These statements do not establish one consistent contract.

### Executable contract

Semantic execution first derives direction from whether `payer` or `receiver`
matches the initials inferred from `Box`; a matching party overrides a rule
default. It then copies the original value, including its sign, into either
`amount_in` or `amount_out`, computes `net_amount = amount_in - amount_out`, and
uses `abs(amount)` only for `amount_abs` and selected non-operating lines.

Therefore the current executable behavior is neither purely “signed cash flow”
nor a safely enforced “nonnegative magnitude plus direction” contract. It is
**original signed value routed by party-derived direction**. A negative outgoing
value becomes negative `amount_out`, positive `net_amount`, and negative OPEX or
draws. This is characterization, not a judgment that the sign should be changed.

### Synthetic fixture evidence

The fixture is deliberately one row per case. “Statement contribution” is the
direct amount on the named line, not a proposed accounting interpretation.

| Case | Current direction | amount_in | amount_out | net_amount | amount_abs | Statement contribution |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Box receives `+100` | in | 100 | 0 | 100 | 100 | operating revenue `+100` |
| Box receives `-100` | in | -100 | 0 | -100 | 100 | operating revenue `-100` |
| Box pays `+100` | out | 0 | 100 | -100 | 100 | property OPEX `+100` |
| Box pays `-100` | out | 0 | -100 | 100 | 100 | property OPEX `-100` |
| Internal Box transfer `+100` | internal | 0 | 0 | 0 | 100 | internal transfers `100` |
| Neither party matches; rent rule | in | 100 | 0 | 100 | 100 | operating revenue `+100` |
| Box pays `0` | out | 0 | 0 | 0 | 0 | property OPEX `0` |

The principal `ledger_fixture.csv` has 307 positive rows, two zero rows, and no
negative rows, which is consistent with magnitude-style use. In contrast,
`ledger_scope_fixture.csv` includes negative mirrored cross-Box rows. Existing
fixture evidence therefore does not permit a repository-wide inference that
negative values are absent or that they are ordinary reversals.

### Read-only diagnostic

`python -m accounting.diagnostics.amount_direction` accepts only an explicit
local `--ledger` and explicit `--output-dir`. It writes:

- `amount_direction_summary.json`;
- `amount_by_box_currency.csv`;
- `direction_sign_matrix.csv`;
- `amount_direction_examples.csv`.

It reports invalid/negative/zero/positive counts, native sums by `Box` and
`Currency`, payer/receiver/internal/neither party matching, the direction/sign
matrix, and bounded examples. It does not classify, normalize, modify, publish,
or overwrite the supplied ledger. It was tested only against the new synthetic
fixture in this task; it was not run against live accounting data.

### Contradictions and decision required from Matías

The documentation, validation language, primary fixture, and scoped mirrored
fixture do not encode one common sign rule. Before any projected semantic figure,
Matías must decide which source cases legitimately carry negative amounts and
whether each represents a reversal, a correction, a mirrored cross-Box entry,
or a normal flow. An engineer must not resolve that accounting meaning with
`abs()` or a sign flip.

## FX precedence

### Current precedence matrix

The table records the current classifier result when credible FX evidence
(`payer/receiver=FX` and/or `cash_path=Cambio:FX`) overlaps another signal.
“Inclusion” names the native statement family reached after classification.

| Synthetic case | Direction | Bucket / subbucket | Rule | Inclusion |
| --- | --- | --- | --- | --- |
| FX + rent | in | operating_revenue / rent | R001 | operating revenue |
| FX + taxes | out | property_opex / taxes | R002 | property OPEX |
| FX + maintenance | out | property_opex / maintenance | R004 | property OPEX |
| FX + contribution | in | funding_contribution / family_or_tenant_contribution | R006 | funding |
| FX + personal withdrawal | out | family_withdrawal_candidate / personal_expense | R011 | distribution candidate |
| FX + loan principal | in | debt_movement / principal | R007 | debt |
| FX + repayment | out | debt_movement / repayment | R008 | debt |
| FX + interest | out | debt_movement / interest | R009 | debt |
| Clean FX proceeds | in | treasury_fx / fx_conversion_proceeds | R014 | Treasury |
| Clean FX outflow | out | treasury_fx / fx_conversion_outflow | R014 | Treasury |
| Explicit FX cost | out | treasury_fx / fx_cost_or_spread | R015 | Treasury |

### Leakage cases

Rent, taxes, maintenance, contribution, personal-expense, and debt rules execute
before FX conversion detection. Consequently, a statement formula that excludes
rows already classified as `treasury_fx` does **not** protect against these
overlaps: the row never reaches the Treasury bucket. The synthetic matrix proves
possible leakage into operating revenue, property OPEX, funding, distributions,
and debt. It does not prove that any such overlap exists in a live ledger.

### Cases clearly safe

With the tested fields, clean proceeds and outflow enter separate gross Treasury
lines; explicit commission/cost evidence enters the Treasury cost line. Those
rows remain outside operating revenue, property OPEX, funding, distributions,
and debt. This finding does not establish pairing, conversion completeness,
spread, realized gain/loss, or an economically meaningful cross-currency net.

### Ambiguous cases and decision required from Matías

An overlap may be bad metadata, a legitimate transaction with two accounting
meanings, or evidence that FX principal should take precedence. Fixtures cannot
choose among those meanings. Matías must approve which overlap combinations are
eligible for management figures and which must fail closed as `review_required`.
Changing native rule precedence is explicitly outside this task.

## Recommendation for PR2

PR2 should not change native semantic classification or build headline reports.
It should consume the canonical ledger, valuation sidecar, and existing semantic
audit to produce a parallel, fixture-only eligibility/reconciliation artifact.
Eligibility should be explicit and additive: native semantics remain untouched;
sign/direction contradictions and rows with both FX evidence and a non-Treasury
classification remain unavailable for projected components until policy approval.

> “PR2 should be a fixture-only projected semantic-flow eligibility layer that rejects sign/direction contradictions and FX-overlap rows pending approval, protecting invariant that only unambiguous native semantic rows enter USD management components, because the characterization evidence shows negative amounts can invert current semantic arithmetic and FX evidence can be preempted by earlier rent, OPEX, funding, withdrawal, and debt rules.”
