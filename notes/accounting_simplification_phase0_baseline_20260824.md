# Accounting simplification Phase 0 baseline — 2026-08-24

Status: **behavior-preserving refactor baseline**  
Scope: `accounting/` source simplification program  
Accounting-policy change: **none**  
Production/live input execution: **none**

## Purpose

This note freezes the evidence boundary for the repository-wide accounting source simplification work begun on 2026-08-24.

The goal of the subsequent cleanup PRs is to reduce duplicated code, migration scaffolding, dead compatibility seams, and misplaced responsibilities **without changing accounting meaning or reported values**.

This baseline is deliberately fixture-safe. It does not commit generated accounting reports, live ledgers, confidential records, or publication outputs.

## Frozen source point

The accounting source baseline is the `main` tree at:

```text
136c977fe6f78e7373129b1eb218a0da05cb0165
```

That commit is the merge of PR #73, `test: normalize migration-era governance boundaries`.

The Phase-0 evidence was executed on a branch whose accounting source was identical to that baseline. The temporary workflow used to gather the evidence was removed before this documentation-only PR is intended to merge.

## Evidence runs

| Evidence | Result | Notes |
| --- | --- | --- |
| Repository CI (`accounting-ci`, run 139) | PASS | Normal repository validation on the Phase-0 branch. |
| `make smoke-full` | PASS | Fixture ingest/materialization + semantic/cash wrapper checks + full validation + publish dry-run. |
| Full pytest suite | **280 passed, 1 warning** | Python 3.12.14; warning is the intentionally preserved legacy invalid-`as_of_date` debt compatibility case. |
| Compile + contract validation | PASS | Artifact vocabulary, lookups, source-contract QA, annual schema imports, publish contract imports. |
| Extended `make smoke-views` probe | **FAIL, exit 2** | Existing fixture boundary: `per_party_time_long` contains null `Box`; `marts.build` requires non-null `Box`. Not introduced or repaired by Phase 0. |
| Whole real professional-pack drilldown run | NOT AVAILABLE IN FIXTURE CI | Real professional pack is external/local and is not committed. See professional baseline limitation below. |

Evidence run references:

- normal CI: `https://github.com/matuteiglesias/accounting-workflows/actions/runs/32768010223`
- one-off Phase-0 fixture capture: `https://github.com/matuteiglesias/accounting-workflows/actions/runs/32768010227`

`make smoke-full` currently describes itself as partial: fixture core + validation + publish dry-run are covered; fixture debt/human publication remains a documented follow-up.

## Behavior-preservation contract for today's cleanup PRs

Unless a later PR is explicitly declared to be a semantic/accounting change, all of these are invariants:

| Invariant | Phase-0 verification surface | Requirement for later cleanup |
| --- | --- | --- |
| semantic classification unchanged | `classification_audit.csv`, semantic totals below, semantic leakage QA | Same classified membership and amount totals at governed grain. |
| monthly semantic totals unchanged | `monthly_operating_statement.csv` and monthly spot checks | Same values by period / Currency / statement line. |
| annual metric totals unchanged | direct governed annual-dashboard build and annual totals below | Same values/statuses for available fixture metrics; stock/flow semantics unchanged. |
| debt balances unchanged | debt contract/mart/professional regression tests; fixture smoke lacks debt-position artifacts | Any PR touching debt requires dedicated debt fixture reconciliation in addition to this baseline. |
| validated cash unchanged | cash authority tests; smoke fixture has no validated-cash population | No inferred/internal fallback may become headline cash; cash-changing PRs need validated-cash fixture parity. |
| Box scope unchanged | scope tests + semantic outputs + leakage checks | Household/property-business scope must not widen or be reintroduced downstream. |
| native-currency separation unchanged | ARS/USD outputs remain separate; annual QA `no_cross_currency_aggregation=pass` | No native ARS/USD co-aggregation. |
| professional displayed values unchanged | professional regression suite | PRs touching professional routing additionally require a real-pack before/after run. |
| drilldown membership unchanged | governed atomic/debt/cash/derived professional tests | Same governed membership, grain, display reconciliation, and no semantic leakage. |

A green pipeline run alone is not sufficient. Later PRs must compare the affected totals, statuses, scope, and drilldown behavior.

## Fixture envelope

The deterministic smoke ledger has:

| Artifact | Rows |
| --- | ---: |
| `ledger_canonical_all_status.csv` | 309 |
| `ledger_canonical.csv` | 306 |
| `per_flow_time_long.freq=M.csv` | 114 |
| `per_party_time_long.freq=M.csv` | 277 |
| `box_balance_time_long.freq=M.csv` | 0 |
| `box_flow_balance_time_long.freq=M.csv` | 0 |
| `loans_time.freq=M.csv` | 0 |
| `daily_cash_position.csv` | 4,284 |
| `classification_audit.csv` | 306 |
| `classification_audit_summary.csv` | 77 |
| `classification_validation.csv` | 56 |
| `monthly_flow_semantic_split.csv` | 168 |
| `monthly_operating_statement.csv` | 529 |
| `monthly_operating_statement_qa.csv` | 26 |
| `monthly_cash_close.csv` | 144 |
| `monthly_cash_close_qa.csv` | 15 |
| `monthly_box_treasury_flow.csv` | 129 |
| `monthly_box_treasury_flow_qa.csv` | 3 |
| `semantic_dashboard_coverage.csv` | 18 |
| `semantic_leakage_qa.csv` | **0** |
| `semantic_rule_registry.csv` | 16 |
| `artifact_contracts.csv` | 21 |
| `artifact_contract_qa.csv` | 9 |

Fixture maximum ledger date / direct annual-builder `as_of_date`:

```text
2025-12-25
```

The two Box-motor artifacts are empty because none of the 306 recognized fixture rows has physical Box-party payer/receiver evidence under the current matching rule. Materialization explicitly warns that all 306 rows are dropped from those Box-motor views. This is a fixture property, not permission to weaken the physical-cash rule.

## Generated artifact snapshot

The generated files themselves are intentionally **not committed**. The following hashes are an evidence snapshot from the Phase-0 workflow; row counts and reconciled semantic values are the durable comparison surfaces. Raw hashes may change when artifacts contain generated timestamps or other run metadata.

| Artifact | Rows | SHA-256 snapshot |
| --- | ---: | --- |
| `artifact_contract_qa.csv` | 9 | `24b2ae4afc922abcc76629522dfefb567b813baa5a830427a08a4bc85fa4ed0f` |
| `artifact_contracts.csv` | 21 | `11d0f6b78071c965f69b6c2df0a7886ddb681b94072f1191d1c004155785d5bf` |
| `box_balance_time_long.freq=M.csv` | 0 | `dcb0b3e22db348c01dab495d097e222a132a518252f84102300ef337dad6b9b3` |
| `box_flow_balance_time_long.freq=M.csv` | 0 | `cfca617b3791515606029cdb1e91ad862de7791cb7f06b83cecc7f4b44bdc739` |
| `classification_audit.csv` | 306 | `2d499a46dbfc92570834a9bc9e9fd5b66de9ec217d3201fa6e9c7faefd74390f` |
| `classification_audit_summary.csv` | 77 | `02f4b4ee69db5fdb380844440f39fd1ac2fc1b64c92c3421e1e7a0d025846e2a` |
| `classification_validation.csv` | 56 | `bf23bfcd17e9c49c1398bf159ae2867368c294f705e2450af70a7bbb4ae931e9` |
| `daily_cash_position.csv` | 4,284 | `24457d1d115f84122b7ce0ab99192ecfc24954c09d46678ab925a1b3c6afcad1` |
| `ledger_canonical.csv` | 306 | `8cacddd2ffb7cfb6daeb41f58bcd5ec500928b775e15a515450cd48a07a23525` |
| `ledger_canonical_all_status.csv` | 309 | `424dd65440b4465fb6337bf0dccd0e39e4748b24c550c299d6aa580b6291a465` |
| `loans_time.freq=M.csv` | 0 | `89cc2c2fecc53606c08a7beea6da179765e60637963e3cc7389dcb6c64f1360b` |
| `monthly_box_treasury_flow.csv` | 129 | `3d3eb99130c7985f738cf38e50a14f7e2ae2d9c069e15e246ca30c1381e87d9b` |
| `monthly_box_treasury_flow_qa.csv` | 3 | `af4c0ee857f90a9aface9442a628bd157620ed04a3878a6e1f4aaa0e6a4ff69e` |
| `monthly_cash_close.csv` | 144 | `05f789afd36090240426c65295268d174ed5a3942d77de059f9866d46de96e6d` |
| `monthly_cash_close_qa.csv` | 15 | `f586bf856f891e2d5ec7f1e216cc08737efc71e88880460c48c85e1d59a507bb` |
| `monthly_flow_semantic_split.csv` | 168 | `e4666a17dd43bcb76e559937a2b580e3f81f4ae613d84f03f92067d9d4c5c94b` |
| `monthly_operating_statement.csv` | 529 | `870edbd8b5d0b42fb2381095fbe45965f2f58eca944b8dcc8f5d1634d9fc4b80` |
| `monthly_operating_statement_qa.csv` | 26 | `f2d0f49bb1ed840707caade6d02d90bdb1333517ee18e8138b09660dfab5f3b7` |
| `per_flow_time_long.freq=M.csv` | 114 | `c43c0c59d2f55a7eeedbe2fedfb36472c3e670e21fd5a1dd1203b9d2b5ed845e` |
| `per_party_time_long.freq=M.csv` | 277 | `094744ca600ff82a1650c375769bb6341c873c94428d203afcdfb86eae592597` |
| `semantic_dashboard_coverage.csv` | 18 | `48a0f0c5383791cb5aa2f7fc7f8f89a3949605bbf5d4dee10d31c381e1461315` |
| `semantic_leakage_qa.csv` | 0 | `358d2c43e59c66e806c00c3a229c75169878b4c85b5b003ce6a3928be293fe73` |
| `semantic_rule_registry.csv` | 16 | `d36d6e6a7af05f6f2cb70377d5cca746672a3acedd733b97bccba37a3d491e00` |

Metadata snapshot hashes were also captured in the workflow log for `meta/artifacts.jsonl`, ingest/materialization checks and manifests, and `partitions.json`. They are evidence, not equality gates, because generated metadata can legitimately contain timestamps.

## Semantic-classification baseline

Totals below are fixture amounts at the governed classification grain.

### ARS

| semantic bucket | subbucket | amount | rows |
| --- | --- | ---: | ---: |
| `operating_revenue` | `rent` | 33,075,422.00 | 90 |
| `property_opex` | `taxes` | 7,427,893.53 | 73 |
| `property_opex` | `services` | 5,801,045.41 | 78 |
| `property_opex` | `maintenance` | 240,000.00 | 5 |
| `property_opex` | `legal` | 147,000.00 | 1 |
| `funding_contribution` | `family_or_tenant_contribution` | 1,427,956.00 | 19 |
| `family_withdrawal_candidate` | `personal_expense` | 29,412,662.00 | 11 |
| `debt_movement` | `principal` | 1,576,761.00 | 12 |

### USD

| semantic bucket | subbucket | amount | rows |
| --- | --- | ---: | ---: |
| `operating_revenue` | `rent` | 4,180.00 | 11 |
| `debt_movement` | `principal` | 2,240.00 | 6 |

`semantic_leakage_qa.csv` contains **0 rows**.

These values freeze classification behavior for behavior-preserving cleanup. They must not be used as production/family accounting totals; they are committed-fixture evidence only.

## Monthly statement spot checks

The full monthly table remains generated evidence rather than a committed dataset. These periods provide compact regression anchors across early, mid, and closing-year behavior.

### ARS

| period | operating revenue | property OPEX | net operating | funding | draws | coverage after draws |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-01 | 1,467,524.00 | 824,833.00 | 642,691.00 | 407,956.00 | 1,467,524.00 | -416,877.00 |
| 2025-06 | 2,735,000.00 | 1,372,083.24 | 1,362,916.76 | 0.00 | 2,034,000.00 | -671,083.24 |
| 2025-12 | 0.00 | 371,369.78 | -371,369.78 | 0.00 | 0.00 | -371,369.78 |

### USD representative month

2025-01 has operating revenue / rent / net operating / coverage after draws = **380.00 USD**, with property OPEX, funding and draws all zero.

All native-currency lines remain separate.

## Annual governed-metric baseline

The annual dashboard builder was run directly over the smoke semantic/cash artifacts to freeze annual-flow behavior even though the ordinary smoke path does not build a complete debt/human stack.

### 2025 ARS

| metric | value | status |
| --- | ---: | --- |
| `IS.REVENUE.OPERATING` | 33,075,422.00 | available |
| `IS.RENT.TOTAL` | 33,075,422.00 | available |
| `IS.OPEX.PROPERTY` | 13,615,938.94 | available |
| `IS.NET.OPERATING` | 19,459,483.06 | available |
| `FUND.CONTRIB.TOTAL` | 1,427,956.00 | available |
| `DIST.DRAWS.PERSONAL` | 29,412,662.00 | available |
| `COV.NET.AFTER_DRAWS` | -8,525,222.94 | available |
| `COV.SAVINGS_RATE` | -0.4381012030850936 | available |
| `TR.FX.NET` | 0.00 | available |
| `BS.CASH.TOTAL` | — | unavailable: no governed validated-cash period |

### 2025 USD

| metric | value | status |
| --- | ---: | --- |
| `IS.REVENUE.OPERATING` | 4,180.00 | available |
| `IS.RENT.TOTAL` | 4,180.00 | available |
| `IS.OPEX.PROPERTY` | 0.00 | available |
| `IS.NET.OPERATING` | 4,180.00 | available |
| `FUND.CONTRIB.TOTAL` | 0.00 | available |
| `DIST.DRAWS.PERSONAL` | 0.00 | available |
| `COV.NET.AFTER_DRAWS` | 4,180.00 | available |
| `COV.SAVINGS_RATE` | 1.0 | available |
| `TR.FX.NET` | 0.00 | available |
| `BS.CASH.TOTAL` | — | unavailable: no governed validated-cash period |

The annual QA confirms that annual flows sum monthly flows, ratios use annual aggregates, native currencies remain separate, cash uses the governed validated selector, and monthly cash positions are never summed into annual stock.

Some direct-smoke annual QA rows fail because the smoke envelope deliberately lacks full stock/debt/cash source coverage. Those failures are documented fixture limitations, not evidence that a cleanup PR may alter stock semantics.

## Cash baseline

The smoke fixture does **not** contain validated frontend-cash snapshots.

`monthly_cash_close.csv` has 144 rows, all belonging to the internal-balance/control population in this fixture. They are marked:

```text
position_type = internal_balance
validation_status = not_validated_for_frontend_cash
cash_suitability = internal_only
```

Therefore annual `BS.CASH.TOTAL` is unavailable and the governed producer correctly refuses to fall back to inferred/internal balances.

For later refactors, the durable invariant is not a fixture cash number; it is the population boundary:

```text
validated cash != inferred control != internal balances
fallback_to_inferred = never
```

Any PR changing cash selection or projection must additionally run the dedicated validated-cash fixture tests and reconcile monthly/annual/professional populations.

## Debt baseline

The committed smoke path does not build `monthly_debt_position.csv` / `monthly_debt_activity.csv`, so annual smoke debt metrics are explicitly unavailable rather than fabricated.

This is not evidence that debt is absent: the ledger fixture contains debt-movement rows, and the regression suite exercises debt source, stock/activity, annualization, and professional behavior.

Consequently, the Phase-0 fixture cannot serve as a numeric debt-balance oracle. Any PR touching debt must provide a dedicated before/after debt fixture and preserve:

- native Currency;
- debtor / creditor identity;
- position vs activity separation;
- latest-valid-`as_of_date` stock semantics;
- annual stock as closing position, never monthly sum;
- activity annualization as flow sum.

The known invalid-`as_of_date` production contradiction identified before this baseline remains a separate P0 correctness change and is **not** normalized into the behavior-preserving cleanup envelope.

## Professional drilldown baseline limitation

A current real professional pack is intentionally not stored in the repository or fixture CI. The repository's professional merge-readiness documentation already treats real-pack generation as external/local evidence.

Therefore **there is no trustworthy current whole-pack professional drilldown status histogram to freeze from committed inputs**. Historical status counts are not substituted here because that would create a stale baseline.

What Phase 0 does establish:

- all current professional contract/executor/reconciliation tests are part of the 280-test green suite;
- no behavior-preserving cleanup PR may weaken those tests;
- a PR touching professional routing, table identity, reconciliation, or legacy fallback must additionally run a real professional pack before and after the change;
- that run must compare displayed values, matched drilldown totals, status counts, residual counts, scope, Currency, and semantic membership.

Until real-pack evidence exists for a professional-routing PR, the invariant `professional displayed values / drilldown membership unchanged` is **not closed by fixture CI alone**.

## Pre-existing baseline gaps discovered or reconfirmed

These exist before the simplification work and must not be attributed to later cleanup PRs:

1. **Extended `smoke-views` fails**: `per_party_time_long` contains null `Box`, while `marts.build` requires non-null `Box`; exit code 2.
2. **Fixture Box motors are empty**: no recognized fixture row establishes physical Box-party payer/receiver matching.
3. **Fixture has no validated-cash population**: governed cash headline is unavailable and correctly does not fall back.
4. **Fixture smoke does not build debt position/activity**: debt annual values are unavailable in the direct annual smoke build.
5. **Fixture/CI does not contain a real professional pack**: whole-pack drilldown status counts require external/local evidence.
6. **Legacy invalid debt `as_of_date` compatibility remains characterized**: the one pytest warning belongs to the preserved legacy fail-open behavior pending the separate P0 repair.

## Use of this baseline in subsequent PRs

A behavior-preserving simplification PR should report:

```text
source baseline: 136c977fe6f78e7373129b1eb218a0da05cb0165
concern being simplified: <one named responsibility>
surviving authority: <module / contract>
removed or relocated authority: <module / path>
fixture before/after: reconciled
scope before/after: reconciled
Currency before/after: reconciled
annual/monthly before/after: reconciled where affected
professional/drilldown before/after: reconciled where affected
known Phase-0 gaps: unchanged unless explicitly addressed
```

If a PR intentionally changes an accounting rule or corrects a known defect, it must not claim Phase-0 behavior preservation. It needs a separate semantic-change statement with old rule, new rule, affected records/artifacts, quantified before/after effect, and reconciliation through every affected layer.
