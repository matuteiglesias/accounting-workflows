# Time-series graphics across accounting reports and Family Strategy artifacts

**Design study:** 2026-11-19. **Status:** PROPOSED, not implemented / no changes to report output, Viewer, or production ledger.

## Goal and verified interface constraints

One reusable, visually prominent `TimeSeriesSpec → inline SVG` chart component, usable in any **finished HTML report**, with the same design palette as existing report charts. It consumes already-governed metric rows, and never classifies ledger transactions or calculates accounting sums. Existing table-adjacent bars, tiny summaries and pie charts remain unchanged.

- Frequency is **monthly (M)** or **annual (Y)** only. No daily/weekly models, controls or plots.
- Mark financial **flow**, **stock**, **ratio**, **conditional forecast**, or **attribution overlay** in each chart's source semantics.
- Every panel is titled by a human question, labelled with money unit or %, timeframe, scope, source, and status/cutoff.
- Do not mix ARS/USD, do not depict annual YTD as a full year, and do not draw missing values as zeros.
- Observed and forecast values must be visually distinct (line style and separation).
- The Viewer currently owns catalog-driven HTML/PDF navigation, NOT source charts or any CSV/metric calculations. On the inspected local bundle it has four tabs and 26 finished reports (the inspected `dist` source run is 2026-11-02, cutoff 2026-10-31; this is a local snapshot, not a claim about live deployment).
- Existing upstream visuals: `accounting/reports/charts.py::render_pie_svg()` is a governed SVG primitive; `treasury_accountability/render.py::_svg_chart()` is a Treasury-specific **monthly bar + cumulative-line** visualization; annual report render has several empty `chart-slot` sections.
- External images in HTML would need additional viewer publication and acceptance rules: the viewer currently copies catalog-referenced **HTML and PDF** only. **Inline SVG** in finished HTML is the simpler path; the PDF renderer consumes the same finished HTML.

## First 10 high-value time series — placement, not a build queue

| ID | Human question / title | Grain / visual | Existing source / likely helper | Primary report and placement | Viewer tab; A artifact | Readiness |
| --- | --- | --- | --- | --- | --- | --- |
| TS01 | **¿Cómo crecieron los costos de las propiedades?** (costs by tax/service/maintenance/legal categories, bars by year) | Y · stacked bars | `monthly_operating_statement`: `property_opex_true`, `taxes`, `services`, `maintenance`, `legal`; approved annual aggregation or equal Jan–Nov window | `annual_management` p2 operations; `opex_monthly_evolution` supporting detail | Panorama; A2 / A7–A8 | **Ready** for a governed plotting adapter |
| TS02 | **¿Qué parte de los alquileres absorbieron los costos?** | Y · dots + line, % | Same operating mart: annual `property_opex_true / rent_revenue`, preserving same dates and currency | `annual_management` p2, paired with TS01; optional analytic text | Panorama; A2 | **Small derived ratio** |
| TS03 | **¿Cómo evolucionó la recaudación mensual?** FB vs PM (native currency) | M · two lines, optional dots | `monthly_cash_accountability.rent_in` by Box/currency; compare `rent_monthly_evolution` source membership | `rent_monthly_evolution` prominent time-series panel; `annual_management` p2 if space | Panorama; A2/A3 | **Ready** |
| TS04 | **¿Cuánto entró y salió efectivamente por mes?** | M · grouped bars, one Box at a time | `monthly_cash_accountability.total_cash_in/out`, not rent/operating income | `treasury_accountability` above main movement table; related `physical_inflows_by_box` and `physical_outflows_by_box` | Administración y tesorería; A3 | **Ready** |
| TS05 | **¿Cómo cambió el movimiento neto y su control acumulado?** | M · bars + line | `monthly_cash_accountability.net_cash_flow, closing_control`; **reuse** existing `_svg_chart` before replacing it | `treasury_accountability`, already present. Avoid a second duplicate image in same report | Administración y tesorería / Rendición; A3 | **Already visualized**; extraction possible |
| TS06 | **¿Cuánto resultado hubo frente a las distribuciones registradas?** (and hypothetical retained result, without claiming bank savings) | M · grouped bars / 2 lines | `monthly_operating_statement.net_operating`, `family_draws_or_distributions`, optional existing `coverage_after_draws` as a *coverage* measure, **not verified savings** | `annual_management` p3 application; `distributions_vs_rent` with monthly extension | Rendición de cuentas; A2 / A6 | **Small adapter**, semantic contract required |
| TS07 | **¿Cómo cambiaron las posiciones abiertas de deuda?** | M · lines + last dots by relation, native USD or ARS | `monthly_debt_position`, **component=total**, `position_status=available`, with repeated monthly balance as a **stock**; never sum stock over months | `debt_accountability`; `annual_management` p5 currently has an empty relationships slot; `open_debt_positions` is a snapshot supporting the chart | Rendición de cuentas; A2 / A6 | **Ready** |
| TS08 | **¿Quién cubrió impuestos y servicios PM a lo largo del tiempo?** (named actor / Box / tenants as recorded) | Y · stacked bars, actor source dimension | `pm_tax_accountability` and `pm_services_accountability` existing reporting populations; a certified **period × funding actor** projection is needed. `monthly_stakeholder_support.csv` and `stakeholder_settlement_detail.csv` were **empty** in checked 2026-11-19 run; cannot use them as real support series. | `pm_tax_accountability`, `pm_services_accountability` | Administración y tesorería; A7–A9/A15 | **Needs verified actor-period source** |
| TS09 | **¿Qué pagos están programados para los próximos meses y cuándo aprieta la caja?** | M · expenses planned bars; optional separate panel for cash-flow reserve | Phase A `monthly_policy_inputs.planned_cash_operating_cost_due_this_month` and `cashflow_reserve_required_next_{1,2,3}m`; source manifest and pinned snapshot. 3m rolling window **ends when horizon incomplete**. | Dedicated **private decision/scenario report** near a treasury excerpt; do not insert forecasts into observed historical tables without clear labeling | Patrimonio y decisiones (later, approved scenario report only); A13/A14 | **Phase A available**; renderer/report pending |
| TS10 | **¿Quién cobró qué y cuánto cambiaría el reparto 50/50?** | M · two actor lines/dots; unassigned collection shown separately | Phase A `monthly_agreement_settlement_inputs` and `policy_input_sources`; specifically Taipe excluded; inferred collector mapping can be incomplete; no final transfer on incomplete month | **Private settlement/accountability comparison report**; optionally linked from approved governance brief, not the public Viewer | Privado; A6/A9/A15 | **Provisional; evidence gated** |

**Coverage without combinatorial plotting:** ~7 existing-mart candidates + 1 Phase-A forecast + 2 actor-provenance-sensitive studies; 10 visual definitions can support many of the 52 candidate experiments because each scenario produces the same sort of canonical outputs.

## What a broad-coverage “hero” chart should look like

- Standard card roughly 1040×340 (flexible responsive viewBox), comfortable A4 portrait scaling; 6–12 annual/monthly labels at a time with selective year ticks on longer histories.
- At most 2–3 series per panel; a stronger story may use two **paired charts** instead of dual y-axes.
- Three primitives only: `bars`, `line` with optional points, `dots` with optional connector. TS05 existing combo may be retained but it is a preexisting exception, not a requirement for arbitrary secondary axes.
- Subtle vertical grid, consistent blues/slate/accent, high contrast title, readable latest-value label, restrained legend, one-sentence takeaway, one-line provenance/caveat.
- Short annotations only for **documented** events (e.g. contract break, new fee), not invented causality.
- **Never automatically make chart data public** merely because the viewer landing page is public; evidence-based scenario/settlement studies require separate artifact approval.
- Tables and existing tiny bar summaries remain available for exact numbers. The chart is the visual narrative, not a substitute for tabular audit.

## Minimal component contract (conceptual Python interface; not implemented)

```python
@dataclass(frozen=True)
class TimeSeriesSpec:
    chart_id: str
    title: str
    frequency: Literal["M", "Y"]
    mark: Literal["bars", "line", "dots"]
    currency_or_unit: str              # ARS / USD / percent, no mixtures
    scope: str
    series_ids: tuple[str, ...]        # max 3
    observed_through: str | None
    subtitle: str = ""
    interpretation_note: str = ""

# rows supplied by an upstream, governed adapter:
# period | series_id | value | Currency_or_unit | scope | value_status | source
svg, trace = render_timeseries_svg(spec, governed_rows)
html = f'<figure class="ts-card">{svg}<figcaption>...</figcaption></figure>'
```

The renderer handles **only visual formatting, point validation, missing periods, axes, status styling and accessible SVG text**. The caller resolves every metric and aggregation from existing backend marts. Validate duplicate points, time ordering, required months, unit/scope consistency, source cutoff, status segregation, and no valid-data emptiness. The trace output should retain `period, value, metric id, source table/filter, status, scope, unit, chart id`, comparable to existing pie-chart traces.

### Why inline SVG rather than static image links or viewer-specific JavaScript

1. One self-contained `report.html` with inline SVG renders without a browser fetch.
2. Same HTML creates the printable PDF, with sharp vector graphics.
3. Current viewer copies HTML/PDF verbatim and need not know the chart exists. No asset-route or publication contract changes.
4. Optional standalone PNG for email/briefs can be exported **from the same spec/data**, not maintained as an authoritative second chart.
5. Seaborn is not installed in the inspected environment; Matplotlib is. A stable bespoke SVG renderer may be as simple as extending the current governed SVG primitive. Adopt extra plotting dependencies only if fidelity tests justify them.

## A0–A17 / Viewer placement discipline

The A0–A17 naming refers to a **meeting/report suite**, not 18 independent local HTML files. The two recovered September directories contain the local A0, A1, A1P, A6, A9, A13, A14, A15, A17 source documents; index entries A2/A3/A4/A7/A8 point to **existing accounting reports**, and A5 is an evidence index. Do not generate phantom A10/A11/A12/A16 files.

- **A0 / A17**: agendas and minutes should link or reference the *selected* relevant report chart, not become charts galleries.
- **A1 / A1P**: asset ownership/occupancy map and long-term noncash options. A monthly cash chart does not prove rights or occupancy.
- **A2**: best flagship 3–4 charts (TS01/TS02/TS06/TS07 where page space allows).
- **A3 / A4**: TS04/TS05 with existing transaction tape for audit and source records, not invented duplicate data figures.
- **A6 / A9 / A15**: TS08/TS10 only with attribution/provenance labels and dispute classification; private until approved.
- **A13 / A14**: TS09 for future obligations, reserve feasibility and handover continuity, scenario-labelled.
- **Viewer Panorama**: existing `annual_management`, `rent_monthly_evolution`, `opex_monthly_evolution`.
- **Viewer Administración y tesorería**: existing `treasury_accountability`, `pm_tax_accountability`, `pm_services_accountability`.
- **Viewer Rendición**: `distributions_vs_rent`, `debt_accountability`, `open_debt_positions`, `treasury_transaction_tape`.
- **Viewer Patrimonio y decisiones**: currently questions with links to accounting tabs, **not** a scenario calculation surface. Any new scenario report should be an independently approved finished document and catalog entry.

## Concrete first iteration and acceptance

1. **Implement the component only once** upstream at `accounting/reports/time_series.py` (or a tight extension of `charts.py`). Do **not** create a metrics DSL or a duplicate source-mart.
2. Prove it on **TS01 annual OPEX**, using a same-month range across years and separate output for real data versus forecast if any. Pair with TS02 annual OPEX/rent ratio as the second small experiment. Keep fonts, page breaks and SVG legible in HTML + PDF.
3. Insert TS01 in **one** established annual-management page or a dedicated nonpublished preview report. Evaluate HTML/PDF side-by-side; do not refactor all existing renderers at once.
4. Then reuse exactly the same component for **TS07 debt** (line/dots) or **TS04 monthly in/out** (bars). If its visual contract works across two independent semantic populations, only then promote as the shared renderer for more reports.
5. Keep actor attribution and privately scoped scenario variants off general reports until evidence/readiness checks pass.

### Verified illustrative cost comparison

From an inspected Nov-19 accounting materialization, **ARS Jan–Nov** true property OPEX and rent (nominal, no inflation adjustment):

| Year | OPEX Jan–Nov (ARS) | Rent Jan–Nov (ARS) | OPEX / rent |
|---|---:|---:|---:|
| 2022 | 587,266 | 4,336,108 | 13.5% |
| 2023 | 3,794,593 | 8,195,412 | 46.3% |
| 2024 | 4,150,701 | 12,090,812 | 34.3% |
| 2025 | 7,464,445 | 34,845,422 | 21.4% |
| 2026 | 9,040,126 | 55,974,000 | 16.2% |

This is **not** a causal measure of management quality, a constant-price real increase, or proof that all invoices have been recognized. The strongest visual story is the juxtaposition of nominal escalation and the ratio to income, not nominal escalation alone.

## No-op boundaries

The design exercise does not alter: canonical ledger, accounting classifications, debt positions, legitimate rights, scenario YAML operators, existing HTML/PDF reports, the Viewer or deployment. Code implementation is a separate, small task.
