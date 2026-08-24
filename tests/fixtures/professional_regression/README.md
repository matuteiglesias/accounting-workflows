# Synthetic professional regression corpus

This fixture is deliberately synthetic and non-confidential. It is not a claim to reproduce a real family professional pack.

Its purpose is to freeze the professional reporting/drilldown contracts that matter for safe architecture simplification: displayed cell value, status, native currency, Box/grain where applicable, matched drilldown total, and stable source membership.

`expected_cells.csv` is the compact expectation ledger. `tests/test_professional_regression_corpus.py` materializes bounded temporary run/pack inputs through the real professional drilldown entrypoint and checks the ledger.

The cases cover monthly and annual flows, Household/property scope separation, core funding versus broader support, debt stock versus activity, validated cash available/unavailable, a governed derived metric, and current FX total/Box route shapes.

This corpus must remain small enough for normal CI. It must not snapshot HTML, generated professional packs, live ledgers, or confidential records. When a route family is intentionally migrated, update the expectation only if the accounting/reporting contract intentionally changes; a filename or implementation change alone is not a reason to change displayed values or source membership.
