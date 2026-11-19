# Family strategy scenario mining — round 2

**Status:** DRAFT / ideas only; not an executable YAML or verified legal/accounting claim.
**Mined:** 2026-11-19.
**Purpose:** consolidate prior 39 concept ideas with the 47-case economic-consequences register, live governance questions, and original A0–A17 meeting artifacts. No helper-TS selection or scenario implementation in this pass.

## Epistemic and report legend

- **F** — candidate future financial-ledger transformation; **A** — attribution/incidence overlay, without rewriting actual payer; **V** — non-ledger valuation or conditional exposure; **G** — governance/game/evidence state.
- **T** = existing `treasury_accountability`; **D** = existing `debt_accountability`. All other named views are proposed or require renderer/contract verification before use.
- Source `M###` references the **47-case restricted** economic-consequences workbook `Mapa_Impactos_Patrimoniales_Familia_2026-11-17.xlsx`, updated 2026-11-18. M048 is proposed separately.
- `A6`, `A9`, `A13`, `A14`, `A15`, `A17` reference the September original family-meeting artifacts under `~/Desktop/paquete_reunion_gobernanza_familiar_2026-09-09/` and `~/Desktop/segunda_ola_artefactos_gobernanza_familiar_2026-09-09/`.
- Governance Flows: Contab Torcuato spreadsheet, tab `G. Governance Flows` (collector reported, not validated custody).
- The former first-round 39-idea YAML remains a brainstorming predecessor at `notes/family_strategy_scenario_ideas_20261119.yaml`; do not treat the two files as additive independent tasks.
- Historical economic register numbers may predate the 2026-11-19 refreshed ledger. Use exact pinned current source and date for any financial execution.

## Inventory — questions × transformations × reports

### P — Distributions, reserves and treasury rules

| ID | Decision / study | Scenario change (concept only) | Mode | Output | Source / precedent | Critical gate |
|---|---|---|---|---|---|---|
| P01 | Everything available every month | Pay out full positive pre-distribution monthly cash net, preserving Box/currency; contrast with eligible operating surplus. | F | T + funding-gap comparison | M023 M035 M036 / D01 D04 | Cash availability and entitlement separate |
| P02 | Half this month's net | Payout 50% of a selected base, with the other 50% retained; explicit negative-net policy. | F | T + distribution schedule | M036 / D04 | Must choose base and recipients |
| P03 | Distribution on next month's apparent surplus | Issue payment at t from estimated t+1 surplus; test January ABL/February taxes. | F | T + reserve stress | M021 M022 M036 / D02 | Forecasting is not validated cash |
| P04 | 0/30/60/90 day reserve | Apply four horizon rules to recurring bills and known one-off expenses. | F | T + coverage-gap | M035 M036 / D05 | Reserve is neither actual expense nor unilateral right |
| P05 | Due-date-first versus month-end-first | Distribute on day 1, day 15, or month-end and test payment interruptions. | F | T + daily liquidity timeline (proposed) | M018 M021 M035 M036 / D07 | Monthly values alone insufficient |
| P06 | One-time emergency draw versus regular withdrawals | Same total withdrawals distributed in lump or monthly schedule. | F | T + peak-funding requirement | M009 M023 M035 | Use same total and horizon |
| P07 | Distribute only the amount above a policy floor | Compare current cash net against forward obligations and a bounded reserve. | F | T + funding gap | A13 §2.4 / D06 D08 | Requires independently verified opening cash for absolute payout |
| P08 | Pay historical arrears before current distributions | Order settlement of verified overdue charges before hypothetical distribution; test financing consequences. | F | T + arrears timeline | M025 M036 M045 / A6-003 | Cannot recode undocumented arrears as due |

### R — Rental income, contracts and collectors

| ID | Decision / study | Scenario change (concept only) | Mode | Output | Source / precedent | Critical gate |
|---|---|---|---|---|---|---|
| R01 | Chino Juan defaults on December–March rent | Suppress selected planned receipts; distinguish scenario from contractual evidence. | F | T + rent evolution | M011 M017 / I01 | Current contract status and pay direction contested |
| R02 | Remisería defaults at the same time | Suppress remisería planned receipts alone or together with Chino Juan. | F | T + concentration exposure | M012 M017 / I06 | Joint scenario is not sum of independent observed losses |
| R03 | Rent late rather than lost | Shift payment +30 or +60 days; keep nominal total constant. | F | T + funding-gap timing | M011 M012 / I02 | Cash timing distinct from insolvency |
| R04 | Tigre 28 vacant versus successful reletting | Add optional rent *after* vacancy period and letting costs; compare 0/50/100% occupancy months. | F+V | T + vacancy opportunity cost | M013 M030 / X02 | Never count vacancy rent as actual revenue |
| R05 | Lease renewal needs signatures | Stop receipts at contract end; resume only after hypothetical authorized renewal. | F+G | T + contract risk timeline | M011 M012 M017 / A6-004 | Signed authority cannot be assumed |
| R06 | Rent levels vary after contract indexation | Scale planned rent by bounds and stress tenant default response separately. | F | T + rent evolution | A13 §2.2 / I05 | Contract basis and currency must remain explicit |
| R07 | Rent is collected directly by heirs versus one Box | Reroute receipts to reported collectors, then see if PM charges remain payable. | F+A | T + actor funding bridge | A6-006 / M009 M036 / G01 | FB is reporting aggregate; collector ≠ legal owner |
| R08 | Physical collector differs from beneficiary | Retain collector and allocate economic benefit under rival settlement agreements. | A | Actor settlement matrix | G. Governance Flows / A15 / B04 | Custody, collector and legal fruits are separate |
| R09 | Tenant payment and service pass-through | Suppress either tenant contribution or matching AySA payment to model interrupted pass-through. | F | T + service reconciliation | M024 | Two legs may cancel; do not count both as independent rent |

### C — Operating expenses, tax defaults and cash protection

| ID | Decision / study | Scenario change (concept only) | Mode | Output | Source / precedent | Critical gate |
|---|---|---|---|---|---|---|
| C01 | ABL January sensitivity | Stress ARS 4m planned ABL by ±25% and 50% after verifying updated ledger. | F | T + tax schedule | M021 / C01 | Prior register's ARS 3.014m historical proxy is superseded as current plan |
| C02 | Tigre + ARBA February shock | Scale the ARS 600k and 800k PM February plans; keep payment dates visible. | F | T + tax schedule | M022 M035 / C02 | Do not confuse estimates with tax bills |
| C03 | Tigre year-end settlement slips into January | Shift the one consolidated Dec 30 payment, not the F/refinanciado component rows. | F | T + arrears/penalty stress | M025 M045 / C03 | No duplicate bills |
| C04 | Late tax penalties versus timely financing | Choose missed payment path with stated penalty assumptions and enforcement costs. | F+V | T + contingent exposure register | M001–M005 M045 | CABA ABL 2023 historical episode is precedent, not recurrence forecast |
| C05 | Repair early versus emergency repair | Compare routine maintenance now against later larger cost and disrupted rent. | F+V | T + asset-condition comparison | A13 §2.8 / M031 / X02 | Need independent condition and cost range |
| C06 | Expenses funded by tenants versus PM | Reassign future *cash route* for qualifying services/taxes and test Box deficits. | F+A | T + burden/coverage matrix | M024 M037 / A15 | Expense remains same; support may not pass through PM |
| C07 | Professional fees: accountant and counsel | Schedule bounded service fees and compare completion/funding dates. | F | T + administrative-cost report | M027 M028 | Must obtain quotations |
| C08 | Lote 27 legal and carrying cost | Model legal stages, tax carrying cost and contingent costs without promising result. | F+G | T + option-tree cost timeline | M018–M020 M029 | Legal outcome and timing are uncertain |

### S — Cost incidence, responsibility and reciprocal settlement

| ID | Decision / study | Scenario change (concept only) | Mode | Output | Source / precedent | Critical gate |
|---|---|---|---|---|---|---|
| S01 | Primos/Alejandro equal split with Taipe excluded | Calculate collector-side rent minus eligible payer-side expenses, half the net pool, then settlement. | A | Monthly actor settlement + source transaction tape | G. Governance Flows / B04 | Reported 50/50 not proved; explicit Taipe offset |
| S02 | Who really financed FB bills | Allocate FB-typed expenses to possible real payers under an evidence-backed scenario. | A | Cost financing attribution matrix | A6-003 / M037 | FB source is not named person |
| S03 | Who legally bears each tax versus who actually paid | Overlay alternative usufruct/share/deed allocations on immutable paid transactions. | A+G | Cost-responsibility matrix | M037 / A15 | Requires legal/deed review |
| S04 | Proportional versus 50/50 versus actual incidence | Compare alternative actor shares of common cost and rent. | A | Actor incidence delta | A9 / B03 B04 | Specify asset/share and eligible universe |
| S05 | Reimburse personal advances now versus later | Add conditional future reimbursement flows in amounts supported by audited advances. | F+A | T + D + reimbursement exposure | M005 M006 M007 / A9 | Paying a bill does not create automatic claim |
| S06 | 2023 ABL funding waterfall reconstructed | Reconcile payment, contributions, repayments and potential reimbursement; keep historic ledger invariant. | A | Historical funding proof table | M001–M005 | Do not sum total and components twice |
| S07 | USD 2,500 alleged claim: full, partial or rejected | Compare scenario outcomes without promoting allegation to recognized debt. | A+F | Debt report + conditional liquidity | M010 | Disputed claimant, currency and scope |
| S08 | Donation and notarial costs allocation | Allocate historical evidence-backed fees by beneficiary/interest, preserving unassigned items. | A | Cost responsibility and pending-allocation | M006–M008 / A9 | No legal debtor inferred from sheet |
| S09 | Reciprocal accountability among custodians | Compare what each administrator can substantiate only where another's interest was managed. | A+G | Actor-by-actor accountability matrix | A6-008 / RECIPROCAL_ACCOUNTABILITY | Personal lawful usufruct spending excluded from general family audit |

### V — Asset preservation, opportunity cost and alternatives

| ID | Decision / study | Scenario change (concept only) | Mode | Output | Source / precedent | Critical gate |
|---|---|---|---|---|---|---|
| V01 | Sell now versus rent three years | Compare net proceeds plus avoided costs to future operating cash and retained value. | V+F | Hold/sell decision valuation | M040 M041 / X03 | Asset price cannot be added to recurring income |
| V02 | Commercial development vs ordinary property sale | Compare bounded appraisal, development costs, timing and execution risk. | V+G | Development option valuation | M040 M041 M042 | Listing/indicative value is not binding sale offer |
| V03 | Two brokers with different listing prices | Compare probability/timing of sale, commissions and marketing conditions. | V+G | Agent option evaluation | M040 M041 | Higher asking price is not realized price |
| V04 | Lote 27 possession recovery | Model costs and contingent rental use after control is recovered; include failure branch. | V+F+G | Recover/hold/lease option tree | M029–M031 M044 | No automatic possession or rent |
| V05 | Rent-equivalent value of occupied family space | Price opportunity cost under alternative rights/permission premises. | V | Non-cash occupancy comparison | M038 M039 / A1P | Not an enforceable rent claim by itself |
| V06 | Housing security as a binding condition | Compare sale or handover options with relocation and residential continuity constraints. | V+G | Housing continuity/constraint summary | M039 / M048 PROPOSED | No proven threat of eviction |
| V07 | Human admin labor has economic value | Hours × benchmark plus error/rework/time-opportunity cost, separate from reimbursement. | V | Owner labor cost vs replacement | M034 | Economic opportunity cost, not retroactive fee |
| V08 | Cost of waiting on appraisal/agent decision | Model expiry of offers, additional carrying expenses, and risk of professional disengagement. | V+G | Delay and option-preservation timeline | M042 M043 | Lost deal cannot be asserted without evidence |

### G — Governance, negotiation and irreversible choices

| ID | Decision / study | Scenario change (concept only) | Mode | Output | Source / precedent | Critical gate |
|---|---|---|---|---|---|---|
| G01 | Matías disengages now, 30 or 90 days later | Contrast work hours, future bills without assigned payer, documented handover and exposure. | G+F+V | Transition feasibility + T | Strategy handoff / A14 / M032–M034 | No implicit legal right to walk away from duties or retain funds |
| G02 | Family manager versus professional administrator | Add quote-based monthly fees, minimum controls, rework prevention and response times. | G+F+V | T + managed-service comparison | M032 M033 / A13 | Management quote/authority required |
| G03 | Fragmented owners pay taxes by separate rota | Split collection by actor/property and assign explicit bill-handling responsibility. | G+F | T + per-asset funding gap | A6-006 / A14 | Named person, consent and evidence required |
| G04 | No successor accepts responsibility | Stress 30/60/90-day inaction, uncovered obligations and Matías backstop scenarios. | G+F | Continuity risk register | Strategy handoff / A14 | Rescue trap is a behavioral hypothesis |
| G05 | Neutral collection account versus direct tenant payment | Choose traceable channel, controls, fees and timing under valid agreement/authority. | G+F | T + handover acceptance score | A13 A14 / G02 | Legal authorization condition; prudence alone no retention right |
| G06 | Disputed balances are ring-fenced or paid by safe route | Separate physical custody, admitted amounts, disputed allocation and agreed transfer. | G+A | Custody/closing-account register | A6-009 / M009 M010 | No private escrow mislabeling or indefinite hold |
| G07 | Cost-sharing pact with default/escape clause | Test whether each actor pays assigned share; stress one actor's noncompliance. | G+F | Funding/arrears/reciprocity report | A13 / M036 M045 | Requires valid agreement |
| G08 | Cooperative package vs formal institutional route | Compare authorized reserve + administrator + reporting against counsel-supervised alternatives. | G+V | Cooperation-frontier/branch report | COOPERATION_FRONTIER / A6-014 | Outcome not predicted; no invented court powers |
| G09 | Rendition before/after objections | Model audit completion, evidence creation, administrative cost and disputes remaining. | G+A | Rendition status + evidence gaps | M026 M027 / A17 | Never auto-convert silence into approval |
| G10 | Intergenerational house preservation | Compare present ownership and lawful estate-planning alternatives with residence protection. | G+V | Noncash rights/option matrix | A1P | Legal questions; not ledger-flow scenarios |

## What the second pass added beyond the first 39

New or newly sharpened families include: (a) due-date liquidity versus month-end accounting; (b) verified ABL-2023 funding rather than presumed reimbursement; (c) cash collector versus legal income beneficiary; (d) the cost of a successor failing to materialize, including Matías's hours and repeated rescues; (e) compare agent listing prices to executable sale options; (f) sale conditioned on household security; (g) Lote 27 recover/repair/rent option chain; (h) possible loss of commercial collaborators from delay; (i) agreement implementation/defection; (j) chronology of rendición, source evidence and objections; (k) service contributions that offset outgoing bills; (l) rent and tax exposures that should never be summed together automatically.

## Consolidation and anti-double-counting decisions

1. **Not 47 + 39 = 86 independent experiments.** The source register M001–M004 is one tax-default episode with components and outcome. M011/M012/M017 are receipts and overlapping loss scenarios. M035/M036 are a reserve policy and one failure mode, not two expenses.
2. **Cash / allocation / noncash / option** remain separate. A lost rent scenario is not an observed loss, and a speculative asset sale price is not monthly available treasury cash.
3. For Taipe, an informal reported offset is a hypothesis to test and explicitly exclude in the relevant agreed settlement, not proof of a market-rent payment or legal setoff.
4. FB is a reporting aggregation, not automatically a physical treasury account or actor. Primos/Alejandro settlement is provisionally attributed by reported collectors and requires unsupported items visibly unresolved.
5. Ownership/authority, rights to fruits, who collects cash, who pays a bill, who finances that payer and who may claim reimbursement are distinct dimensions.
6. Do not convert an audit claim, an historical account control, or an evidence gap into a paid/owed transaction. Court outcomes and formal rights stay external conditional branches.

## Deferral: helper-TS selection is the next decision, not this mining result

This file intentionally does **not** designate a new TS/metrics engine. After comparing the candidate studies, seek the smallest reusable primitives and projections needed to express many rows; validate repeated use before expanding `accounting.scenarios.policy_inputs` and the scenario runner. An operator is only justified when the same transformation survives at least two real, audited study cases.

## Source pointers (not duplicating confidential evidence)

- Restricted economic consequences workbook: `https://docs.google.com/spreadsheets/d/1WezYuNeBu54MhhYcbq-tzOzfx9aRlpHO/edit` (stored `.xlsx`, file ID verified by Drive search).
- Current Contab Torcuato: `https://docs.google.com/spreadsheets/d/1G92WIwtaayQ0MOSLH3vWph-7PKVseXF4vhiTOi2RVls/edit`.
- FSL repo: `https://github.com/matuteiglesias/family-strategy-lab` (world.v6 status as of 2026-09-24; older analyses remain historical).
- Local notes: `~/Desktop/MEMOS/10_accounting_family/STRATEGY_LAB_THREAD_HANDOFF_2026-11-17.md`, `~/Desktop/MEMOS/10_accounting_family/2026-11-17_economic_consequences_register_POINTER.md`, and September A0–A17.
