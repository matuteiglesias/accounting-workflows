# Docs-input bridge — implementation completion packet

Date: 2026-09-17  
Branch: `feat/docs-input-bridge`  
PR: #108  
Program spec: `notes/docs_input_bridge_program_20260917.md`

## Status

W0–W5 are implemented for the approved v1 boundary.

The backend now exposes `acct.docs-input@1` as a real exact-run generated bridge for internal documentation/professional consumers. The implementation adds no new historical accounting classification and does not enter the live/publication composite.

## Invariant protected

The existing accounting pipeline remains authoritative:

```text
ledger canonicalization
  -> materialization
  -> semantic marts
  -> debt
  -> metrics
  -> human reports
  -> professional pack
  -> drilldowns
  -> docs-input factual projection
```

Docs-input may project already-governed facts and an optional prospective non-ledger commitments input. It may not reinterpret ledger membership, manufacture cash, promote unresolved allocations to debt, assign legal parties, or calculate a distributable balance.

## Source records traced

The implemented bridge reads these existing governed populations as applicable:

- `classification_audit.csv` for explicit `tx_id` → descriptive property/Box lookup;
- `stakeholder_settlement_detail.csv` for governed settlement/support legs;
- `cost_allocation_gaps.csv` for unresolved economic burden;
- professional `professional_drilldown_index.csv` plus its explicit detail CSVs for evidence populations;
- `acct.transaction-evidence@1` sidecar for approved/candidate evidence links by `tx_id` only;
- optional private `treasury_commitments.v1` for future commitments.

No date/amount/text similarity join was introduced.

## New outputs

Generated under:

```text
out/docs_input/<RUN_ID>/
```

V1 outputs:

- `docs_input_manifest.json`;
- `accounting_evidence_coverage.csv`;
- `actor_property_cost_support_detail.csv`;
- `unresolved_allocation_detail.csv`;
- `required_reserve_schedule.csv` when a prospective commitments input is supplied;
- `required_reserve_schedule_qa.csv` alongside the schedule.

Generated artifacts remain untracked/internal by default.

## Accounting rule changed

**No.**

No canonical ledger classification, semantic bucket membership, cash rule, debt rule, annual metric rule, report membership, or professional drilldown membership was changed.

No new semantic classification was introduced.

## Before / after effect on existing accounting

Expected and tested invariant:

```text
canonical ledger totals            delta = 0
semantic bucket totals             delta = 0
debt stock/activity                delta = 0
treasury physical cash             delta = 0
annual governed metrics            delta = 0
professional displayed values      delta = 0
```

The regression fixture explicitly snapshots the governed source CSV bytes before docs-input generation and verifies they remain unchanged afterwards. Full repository `make validate` also remained green.

No live accounting run was executed in this implementation wave, so this packet does not claim a new live numerical reconciliation beyond the repository regression/contract corpus.

## Reconciliations and fail-closed checks

### Evidence coverage

- approved + candidate + missing = transaction rows;
- candidate evidence never counts as approved evidence;
- missing evidence sidecar is explicit rather than silently interpreted as fully unsupported accounting;
- malformed/partial evidence remains governed by the existing evidence contract;
- professional detail paths cannot escape the supplied pack root.

### Actor × property × cost/support

- join to property/Box is only through explicit `source_tx_id` → `tx_id`;
- referenced non-unique transaction identity fails closed;
- settlement-leg row count is preserved;
- support amount is emitted only for existing support leg roles;
- allocation/mirror legs cannot silently become support;
- no `legal_debtor`, `legal_creditor`, `legal_owner`, or equivalent field is created.

### Unresolved allocation

- row count is preserved;
- native-currency totals reconcile exactly to `cost_allocation_gaps.csv`;
- docs-input refuses a source gap that no longer has `debt_effect=none` without upstream review;
- docs-input refuses nonblank bearer assertions at this boundary;
- evidence status may enrich a row but cannot change its accounting nature.

### Prospective reserve schedule

- duplicate commitment IDs fail closed;
- invalid dates, amounts, amount statuses, and commitment statuses fail closed;
- native currencies remain separate;
- `scenario` rows remain visible but contribute zero required reserve;
- required reserve reconciles to included commitments by currency;
- the prospective input does not enter ledger/OPEX/debt/cash/history;
- no `distribution_capacity` field or distributable-balance calculation exists.

## Drilldown semantic-leakage check

Docs-input consumes professional detail populations read-only. It does not select alternative semantic membership or recompute displayed values. Evidence coverage is a passive classification of evidence-link state over the already-built detail rows.

Therefore the new layer cannot reintroduce Household/property OPEX/funding/distribution membership by a parallel selector. The full existing regression suite, including professional/drilldown contracts, remained green.

## Tests / CI

GitHub Actions `accounting-ci` runs `make validate`, which compiles the accounting/scripts/tests tree, checks repository contracts, and runs the full pytest regression suite.

Validated branch heads:

- implementation/tests head `d06744f0...`: CI success;
- documentation-integrated head `79f9de877...`: CI success, workflow run `35281049706`.

No live credentials or private accounting inputs were required by CI.

## Operator surface decision

The program specification allowed an exact-run command **such as** `make run-docs-input`. V1 deliberately exposes the bounded module entrypoint instead:

```bash
python -m accounting.docs_input.build \
  --run-root out/run/accounting/<exact-run-id> \
  --out-dir out/docs_input/<exact-run-id>
```

Reason: repository governance explicitly avoids adding Make aliases/command surface without a concrete operational caller. The module command is exact-run, explicit, testable, and does not alter `run-full`, latest pointers, or publication. A Make target can be added later if an actual operator/automation consumer requires it.

## Documentation refreshed

- `README.md` now documents the factual bridge and the actual wider report surface;
- `notes/output_contracts.md` defines `acct.docs-input@1` as a current downstream contract;
- `notes/documentation_compass.md` directs internal documentation consumers to the bridge while preserving the interpretation firewall;
- `SYSTEM.yaml` states that this repository owns factual projections but not downstream legal/strategy/governance/psychology interpretation.

The public report/public accounting publication contracts remain unchanged.

## Live inputs / publication

Live accounting inputs accessed: **no**.  
Private commitments accessed: **no**; only synthetic fixture data in tests.  
Publication performed: **no**.  
Latest pointers moved: **no**.

## Deliberately unresolved / stop conditions preserved

The bridge does not decide:

- who legally bears an accounting cost;
- whether an unresolved cost is legally a debt;
- whether rent belongs to an actor beyond existing governed accounting scope;
- whether a reserve gives authority to withhold or distribute funds;
- who has a legal duty to render accounts for a particular population;
- whether current liquidity is available when validated cash is unavailable;
- what amount is legally distributable.

Those questions remain for the appropriate legal/governance layer using accounting as evidence.

## Next bounded action

Human review of PR #108. After merge, downstream Family Strategy Lab/docs work can consume `acct.docs-input@1` by manifest rather than reaching into arbitrary accounting internals. A later live exact-run generation should be treated as a separate operational validation and should inspect totals, scope, evidence coverage, unresolved allocations, and reserve-source provenance rather than relying only on command success.
