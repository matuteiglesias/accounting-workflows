from __future__ import annotations

"""Single source-location and dependency contract for specialized reports.

The three implementation modules still own calculations, but not file routing.
This mapping is deliberately static and does not initiate selective execution.
"""

from pathlib import Path

SOURCE_LOCATIONS: dict[str, tuple[str, str]] = {'semantic_audit': ('run', 'classification_audit.csv'),
 'stakeholder_support': ('run', 'monthly_stakeholder_support.csv'),
 'semantic_split': ('run', 'monthly_flow_semantic_split.csv'),
 'annual_metrics': ('metrics', 'annual_balance_dashboard_metrics.csv'),
 'cash_accountability': ('run', 'monthly_cash_accountability.csv'),
 'accountability_cycles': ('run', 'family_business_accountability_cycles.csv'),
 'debt_position': ('run', 'monthly_debt_position.csv'),
 'debt_activity': ('run', 'monthly_debt_activity.csv'),
 'repayment_detail': ('run', 'monthly_debt_repayment_detail.csv'),
 'treasury_flow': ('run', 'monthly_box_treasury_flow.csv')}

VIEW_REQUIREMENTS: dict[str, tuple[str, ...]] = {'pm_tax_by_actor': ('semantic_audit',),
 'pm_services_by_actor': ('semantic_audit',),
 'pm_support_by_actor': ('stakeholder_support',),
 'distributions_by_recipient': ('semantic_audit', 'annual_metrics'),
 'rent_by_property': ('annual_metrics',),
 'rent_monthly_evolution': ('semantic_split', 'annual_metrics'),
 'opex_by_category': ('annual_metrics',),
 'taxes_by_property': ('semantic_split', 'annual_metrics'),
 'services_by_property': ('semantic_split', 'annual_metrics'),
 'distributions_vs_rent': ('semantic_audit', 'annual_metrics'),
 'support_by_target_box': ('stakeholder_support',),
 'prior_period_clearing': ('stakeholder_support',),
 'physical_inflows_by_box': ('cash_accountability',),
 'physical_outflows_by_box': ('cash_accountability',),
 'accountability_balance': ('accountability_cycles',),
 'accountability_cycles': ('accountability_cycles',),
 'open_debt_positions': ('debt_position',),
 'debt_activity': ('debt_activity',),
 'repayment_allocations': ('repayment_detail',),
 'rent_annual_comparison': ('annual_metrics',),
 'distributions_by_year': ('semantic_audit', 'annual_metrics'),
 'opex_monthly_evolution': ('semantic_split', 'annual_metrics'),
 'maintenance_by_property': ('semantic_split', 'annual_metrics'),
 'legal_costs_by_property': ('semantic_split', 'annual_metrics'),
 'support_by_obligation_category': ('stakeholder_support',),
 'support_by_funding_channel': ('stakeholder_support',),
 'support_by_settlement_nature': ('stakeholder_support',),
 'physical_inflows_by_category': ('treasury_flow', 'cash_accountability'),
 'physical_outflows_by_category': ('treasury_flow', 'cash_accountability'),
 'cash_residuals': ('cash_accountability',)}

ROUND1_VIEW_KEYS = frozenset(('pm_tax_by_actor', 'pm_services_by_actor', 'pm_support_by_actor', 'distributions_by_recipient', 'rent_by_property', 'rent_monthly_evolution', 'opex_by_category', 'taxes_by_property', 'services_by_property', 'distributions_vs_rent'))
ROUND2_VIEW_KEYS = frozenset(('support_by_target_box', 'prior_period_clearing', 'physical_inflows_by_box', 'physical_outflows_by_box', 'accountability_balance', 'accountability_cycles', 'open_debt_positions', 'debt_activity', 'repayment_allocations'))
ROUND3_VIEW_KEYS = frozenset(('rent_annual_comparison', 'distributions_by_year', 'opex_monthly_evolution', 'maintenance_by_property', 'legal_costs_by_property', 'support_by_obligation_category', 'support_by_funding_channel', 'support_by_settlement_nature', 'physical_inflows_by_category', 'physical_outflows_by_category', 'cash_residuals'))

# Implemented views without a published REPORT_SPECS consumer. Kept as
# callable internal/legacy views; they are NOT automatically materialized.
UNPUBLISHED_VIEW_KEYS = frozenset({"rent_annual_comparison", "distributions_by_year"})


def path_for(source_key: str, run_root: Path, metrics_dir: Path) -> Path:
    root, filename = SOURCE_LOCATIONS[source_key]
    if root == "run":
        return Path(run_root) / filename
    if root == "metrics":
        return Path(metrics_dir) / filename
    raise ValueError(f"Unsupported report input namespace: {root!r}")


def source_paths_for_view(
    view_key: str,
    run_root: Path,
    metrics_dir: Path,
) -> tuple[tuple[Path, str], ...]:
    return tuple(
        (path_for(key, run_root, metrics_dir), f"{SOURCE_LOCATIONS[key][0]}/{SOURCE_LOCATIONS[key][1]}")
        for key in VIEW_REQUIREMENTS[view_key]
    )
