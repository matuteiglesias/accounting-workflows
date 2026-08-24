from pathlib import Path

annual = Path('accounting/metrics/annual.py')
text = annual.read_text()
text = text.replace(
    'from accounting.contracts.funding_support import (\n',
    'from accounting.contracts.annual_flow_membership import build_annual_flow_membership\nfrom accounting.contracts.funding_support import (\n',
)
old = '''    split_path = run_root / "monthly_flow_semantic_split.csv"
    split = pd.read_csv(split_path) if split_path.exists() else None
    metrics, funding_rewritten = _rewrite_funding_support_metrics(
        metrics, split, run_id=run_id, as_of_date=as_of_date
    )
'''
new = '''    split_path = run_root / "monthly_flow_semantic_split.csv"
    split = pd.read_csv(split_path) if split_path.exists() else None
    annual_membership = build_annual_flow_membership(split) if split is not None else build_annual_flow_membership(pd.DataFrame())
    membership_path = metrics_dir / "annual_flow_membership.csv"
    annual_membership.to_csv(membership_path, index=False)
    paths["annual_flow_membership"] = membership_path
    metrics, funding_rewritten = _rewrite_funding_support_metrics(
        metrics, split, run_id=run_id, as_of_date=as_of_date
    )
'''
if old not in text:
    raise SystemExit('annual split block not found')
annual.write_text(text.replace(old, new))

prof = Path('accounting/professional/drilldown.py')
text = prof.read_text()
text = text.replace(
    'import pandas as pd\n\nfrom accounting.contracts.atomic_flow_drilldowns import (\n',
    'from contextvars import ContextVar\nfrom pathlib import Path\n\nimport pandas as pd\n\nfrom accounting.contracts.atomic_flow_drilldowns import (\n',
)
text = text.replace(
    'from accounting.professional.cash_position_executor import (\n',
    'from accounting.professional.annual_flow_executor import execute_annual_flow_membership\nfrom accounting.professional.cash_position_executor import (\n',
)
marker = '''_ORIGINAL_BUILD_DERIVED_CELL = _base._build_derived_cell
_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS = (
    _base._legacy.enrich_professional_table_contracts
)
'''
replacement = '''_ORIGINAL_BUILD_DERIVED_CELL = _base._build_derived_cell
_ORIGINAL_BUILD_PROFESSIONAL_FLOW_DRILLDOWNS = _base._legacy.build_professional_flow_drilldowns
_ORIGINAL_ENRICH_PROFESSIONAL_TABLE_CONTRACTS = (
    _base._legacy.enrich_professional_table_contracts
)
_CURRENT_ANNUAL_FLOW_MEMBERSHIP: ContextVar[pd.DataFrame | None] = ContextVar(
    "current_annual_flow_membership", default=None
)
'''
if marker not in text:
    raise SystemExit('professional original marker not found')
text = text.replace(marker, replacement)
needle = '''    if table_id == "monthly_tables_cash_close_matrix":
'''
insert = '''    annual_flow_membership = _CURRENT_ANNUAL_FLOW_MEMBERSHIP.get()
    if annual_flow_membership is not None and not annual_flow_membership.empty:
        governed_annual_flow = execute_annual_flow_membership(
            row=row,
            period=period,
            display_value=display_value,
            annual_flow_membership=annual_flow_membership,
            tolerance=tolerance,
        )
        if governed_annual_flow is not None:
            return governed_annual_flow

    if table_id == "monthly_tables_cash_close_matrix":
'''
if needle not in text:
    raise SystemExit('derived executor insertion point not found')
text = text.replace(needle, insert, 1)
old_tail = '''_base._legacy._build_derived_cell = _build_derived_cell
_base._legacy.enrich_professional_table_contracts = _enrich_professional_table_contracts

build_professional_flow_drilldowns = _base._legacy.build_professional_flow_drilldowns
main = _base._legacy.main
'''
new_tail = '''_base._legacy._build_derived_cell = _build_derived_cell
_base._legacy.enrich_professional_table_contracts = _enrich_professional_table_contracts


def build_professional_flow_drilldowns(
    repo_root: Path,
    pack_dir: Path,
    run_root: Path | None = None,
    tables_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    fast: bool = False,
):
    """Run professional drilldowns with optional governed annual lineage.

    The lineage artifact is loaded once at this public boundary and passed to
    annual governed execution through invocation-local context. Historical packs
    without the artifact continue through the characterized compatibility path.
    """

    membership_path = _base._legacy._find_source(
        Path(repo_root), Path(pack_dir), Path(run_root) if run_root is not None else None,
        "annual_flow_membership.csv",
    )
    membership = (
        _base._legacy._read_csv(membership_path)
        if membership_path is not None
        else pd.DataFrame()
    )
    token = _CURRENT_ANNUAL_FLOW_MEMBERSHIP.set(membership)
    try:
        return _ORIGINAL_BUILD_PROFESSIONAL_FLOW_DRILLDOWNS(
            repo_root=repo_root,
            pack_dir=pack_dir,
            run_root=run_root,
            tables_dir=tables_dir,
            tolerance=tolerance,
            fast=fast,
        )
    finally:
        _CURRENT_ANNUAL_FLOW_MEMBERSHIP.reset(token)


main = _base._legacy.main
'''
if old_tail not in text:
    raise SystemExit('professional tail not found')
prof.write_text(text.replace(old_tail, new_tail))
