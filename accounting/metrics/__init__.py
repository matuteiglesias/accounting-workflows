"""Governed reporting metrics for the accounting backend.

The package owns two current projections over governed monthly artifacts:

- ``accounting.metrics.frontier`` for monthly frontend-safe metric contracts/series;
- ``accounting.metrics.annual`` for annual dashboard facts and contracts.

``accounting.metrics.build`` is only their orchestration and artifact-handoff CLI.
The retired generic registry / metric_values / leaf-builder universe is not an
accounting authority and no longer lives in this package.
"""
