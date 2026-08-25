# Legacy views-stage retirement — 2026-08-25

Stacked base: `refactor/decommission-legacy-metrics-engine` at `c8f553f669662c9002b0d3481d934df6f6911987`.

## Invariant

Removing the generic views layer must not remove underlying governed evidence or reintroduce semantic classification downstream.

The surviving pipeline must still preserve:

- canonical ledger and Stage-D diagnostic evidence;
- monthly semantic split and operating statement;
- Household/property scope separation;
- governed validated cash and cash QA;
- debt stock/activity separation;
- treasury/FX semantics;
- annual governed metrics and flow membership;
- publication and professional drilldown traceability.

## Deleted layer

`accounting/marts/build.py` was a transitional generic views module. Its former outputs included legacy/presentation conveniences such as contribution/OPEX views, cashflow views, rent pivots, party balances, zero-sum/balance presentations and upcoming-payment extracts. Its legacy `fondos_report.csv` and `renta_*.csv` inputs were already best-effort fallbacks rather than accounting authority.

The file is deleted. No replacement generic views framework is introduced.

The governed marts remain:

- `accounting/marts/semantic.py`
- `accounting/marts/cash.py`
- `accounting/marts/debt.py`
- `accounting/marts/treasury.py`

## Control-plane change

Before:

```text
run-ingest
  -> run-materialize
  -> run-marts / views_sanity
  -> debt
  -> generic metrics + governed metrics
  -> dashboard
  -> publish
```

After the stacked deletion program:

```text
run-ingest
  -> run-materialize
  -> debt position/activity + treasury
  -> governed frontier + annual dashboard
  -> publish
  -> professional presentation/drilldowns
```

`run-canonical` now resolves directly to `run-materialize`.

Removed control-plane concepts:

- `run-marts`
- `_run_views_action`
- `smoke-views`
- `_check_views`
- `views_sanity.json`
- reports-directory loader anchors
- views-directory variables
- Makefile assertions for `metric_registry.csv`, `metric_values.csv`, `validation_report.csv` and `metric_views/*`

## Publication boundary

Retired generic metric statement outputs are no longer selected for publication. Release QA now asserts that `metric_registry.csv`, `metric_values.csv`, and old Q/Y income/cash/debt statement projections are absent.

## Semantic non-regression rules

Deletion is not validated by process success alone. The stacked change remains acceptable only when tests/CI demonstrate the current governed invariants and when any stale compatibility-only tests are distinguished from accounting invariants.

In particular:

- do not force old funding-inclusive income parity;
- do not restore Household into property OPEX;
- do not recreate raw-regex draws classification;
- do not replace validated cash with internal/inferred balances;
- do not annual-sum debt stocks;
- do not collapse currencies;
- do not build a new permanent backend view merely because an old convenience CSV disappeared.
