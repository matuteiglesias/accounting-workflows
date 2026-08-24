from __future__ import annotations

import ast
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path('.')
TARGETS = {
    'accounting/metrics/annual.py': {
        'module': 'accounting.metrics.annual',
        'delegate_module': 'accounting.metrics.annual_legacy',
        'delegate_path': 'accounting/metrics/annual_legacy.py',
        'alias': '_legacy',
        'label': 'metrics.annual',
    },
    'accounting/metrics/frontier.py': {
        'module': 'accounting.metrics.frontier',
        'delegate_module': 'accounting.metrics.frontier_legacy',
        'delegate_path': 'accounting/metrics/frontier_legacy.py',
        'alias': '_legacy',
        'label': 'metrics.frontier',
    },
    'accounting/professional/annual_dashboard_tables.py': {
        'module': 'accounting.professional.annual_dashboard_tables',
        'delegate_module': 'accounting.professional.annual_dashboard_tables_legacy',
        'delegate_path': 'accounting/professional/annual_dashboard_tables_legacy.py',
        'alias': '_legacy',
        'label': 'professional.annual_dashboard_tables',
    },
    'accounting/professional/drilldown_wave4_base.py': {
        'module': 'accounting.professional.drilldown_wave4_base',
        'delegate_module': 'accounting.professional.drilldown_legacy',
        'delegate_path': 'accounting/professional/drilldown_legacy.py',
        'alias': '_legacy',
        'label': 'professional.drilldown_wave4_base',
    },
    'accounting/professional/drilldown.py': {
        'module': 'accounting.professional.drilldown',
        'delegate_module': 'accounting.professional.drilldown_wave4_base',
        'delegate_path': 'accounting/professional/drilldown_wave4_base.py',
        'alias': '_base',
        'label': 'professional.drilldown',
    },
}


def top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == '*':
                    continue
                names.add(alias.asname or alias.name.split('.')[-1])
    return names


def attr_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return '.'.join(reversed(parts))
    return None


class CallerVisitor(ast.NodeVisitor):
    def __init__(self, module: str):
        self.module = module
        self.parent, self.child = module.rsplit('.', 1)
        self.module_aliases: set[str] = set()
        self.symbols: set[str] = set()
        self.dynamic_errors: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == self.module:
            for alias in node.names:
                if alias.name == '*':
                    self.dynamic_errors.append('star import')
                else:
                    self.symbols.add(alias.name)
        elif node.module == self.parent:
            for alias in node.names:
                if alias.name == self.child:
                    self.module_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == self.module and alias.asname:
                self.module_aliases.add(alias.asname)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = attr_chain(node)
        if chain:
            for alias in self.module_aliases:
                prefix = alias + '.'
                if chain.startswith(prefix):
                    remainder = chain[len(prefix):]
                    if remainder:
                        self.symbols.add(remainder.split('.', 1)[0])
            prefix = self.module + '.'
            if chain.startswith(prefix):
                remainder = chain[len(prefix):]
                if remainder:
                    self.symbols.add(remainder.split('.', 1)[0])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {'dir', 'vars'} and node.args:
            chain = attr_chain(node.args[0]) if isinstance(node.args[0], ast.Attribute) else (
                node.args[0].id if isinstance(node.args[0], ast.Name) else None
            )
            if chain in self.module_aliases or chain == self.module:
                self.dynamic_errors.append(f'{node.func.id}() introspection')
        if isinstance(node.func, ast.Name) and node.func.id == 'getattr' and len(node.args) >= 2:
            obj = node.args[0]
            chain = attr_chain(obj) if isinstance(obj, ast.Attribute) else (obj.id if isinstance(obj, ast.Name) else None)
            if chain in self.module_aliases or chain == self.module:
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    self.symbols.add(key.value)
                else:
                    self.dynamic_errors.append('dynamic getattr')
        self.generic_visit(node)


def scan_callers(target_path: str, module: str) -> dict[str, set[str]]:
    callers: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    for path in ROOT.rglob('*.py'):
        rel = path.as_posix()
        if rel == target_path or rel.startswith('.git/') or '/__pycache__/' in rel:
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = CallerVisitor(module)
        visitor.visit(tree)
        for symbol in visitor.symbols:
            callers[symbol].add(rel)
        for err in visitor.dynamic_errors:
            errors.append(f'{rel}: {err} of {module}')
    if errors:
        raise SystemExit('Cannot narrow facade with dynamic callers:\n' + '\n'.join(sorted(errors)))
    return callers


def replace_dynamic_reexport(path: Path, alias: str, compat: list[str]) -> None:
    text = path.read_text(encoding='utf-8')
    old = (
        f'for _name in dir({alias}):\n'
        '    if not _name.startswith("__"):\n'
        f'        globals()[_name] = getattr({alias}, _name)\n'
    )
    if old not in text:
        raise SystemExit(f'expected broad re-export block not found in {path}')
    tuple_lines = ''.join(f'    {name!r},\n' for name in compat)
    assignments = ''.join(f'{name} = {alias}.{name}\n' for name in compat)
    replacement = (
        '# Explicit compatibility surface derived from repository caller census.\n'
        '# Do not broaden this list: every retained legacy symbol must have a caller\n'
        '# or an independently documented compatibility contract/removal condition.\n'
        f'LEGACY_COMPAT_EXPORTS = (\n{tuple_lines})\n\n'
        f'{assignments}'
    )
    path.write_text(text.replace(old, replacement, 1), encoding='utf-8')


inventory_rows: list[dict[str, str]] = []
summary_rows: list[dict[str, object]] = []

for target_path, cfg in TARGETS.items():
    target = ROOT / target_path
    delegate = ROOT / cfg['delegate_path']
    callers = scan_callers(target_path, cfg['module'])
    local_names = top_level_names(target)
    delegate_names = top_level_names(delegate)

    requested = sorted(callers)
    compat = sorted(name for name in requested if name not in local_names)
    missing = [name for name in compat if name not in delegate_names]
    if missing:
        raise SystemExit(
            f"{cfg['module']} caller symbols are neither local nor statically present in "
            f"{cfg['delegate_module']}: {missing}"
        )

    replace_dynamic_reexport(target, cfg['alias'], compat)

    for symbol in requested:
        inventory_rows.append({
            'facade': cfg['module'],
            'delegated_module': cfg['delegate_module'],
            'symbol': symbol,
            'ownership': 'explicit_compat_export' if symbol in compat else 'facade_local',
            'caller_count': str(len(callers[symbol])),
            'callers': ';'.join(sorted(callers[symbol])),
        })

    broad = sorted(name for name in delegate_names if not name.startswith('__'))
    summary_rows.append({
        'facade': cfg['module'],
        'delegated_module': cfg['delegate_module'],
        'broad_static_surface_before': len(broad),
        'repo_referenced_symbols': len(requested),
        'explicit_compat_exports_after': len(compat),
        'facade_local_references': len(requested) - len(compat),
    })

notes = ROOT / 'notes'
notes.mkdir(exist_ok=True)
inv_path = notes / 'accounting_simplification_phase4_legacy_export_inventory_20260824.csv'
with inv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['facade','delegated_module','symbol','ownership','caller_count','callers'])
    writer.writeheader()
    writer.writerows(sorted(inventory_rows, key=lambda r: (r['facade'], r['symbol'])))

summary_path = notes / 'accounting_simplification_phase4_facade_summary_20260824.csv'
with summary_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)

route_rows = [
    {
        'legacy_route_family': 'monthly operating revenue / rent flow',
        'governed_replacement': 'FlowCellSpec + semantic_measure_registry_v1',
        'modern_governed': 'yes when drilldown_cell_id is present',
        'legacy_reachable': 'compatibility only',
        'blocker': 'historical/minimal rows without governed cell identity',
        'removal_condition': 'supported professional corpus contains governed cell IDs for all current monthly revenue rows',
    },
    {
        'legacy_route_family': 'monthly property OPEX flow',
        'governed_replacement': 'FlowCellSpec + semantic_measure_registry_v1',
        'modern_governed': 'yes when drilldown_cell_id is present',
        'legacy_reachable': 'compatibility only',
        'blocker': 'historical/minimal rows without governed cell identity',
        'removal_condition': 'supported professional corpus contains governed cell IDs for all current monthly OPEX rows',
    },
    {
        'legacy_route_family': 'monthly personal draws / withdrawal flow',
        'governed_replacement': 'FlowCellSpec + semantic_measure_registry_v1',
        'modern_governed': 'yes when drilldown_cell_id is present',
        'legacy_reachable': 'compatibility only',
        'blocker': 'historical/minimal rows without governed cell identity',
        'removal_condition': 'supported professional corpus contains governed cell IDs for all current monthly draw rows',
    },
    {
        'legacy_route_family': 'annual flow drilldowns (rent/OPEX/draws and other atomic flows)',
        'governed_replacement': 'annual governed metric artifacts; monthly FlowCellSpec membership exists',
        'modern_governed': 'display metric yes; drilldown intentionally legacy',
        'legacy_reachable': 'yes',
        'blocker': 'annual lineage contract: wave4 explicitly refuses monthly recomputation for YEAR_RE rows',
        'removal_condition': 'dedicated annual membership/lineage contract composes monthly governed measure without discarding annual provenance',
    },
    {
        'legacy_route_family': 'funding by actor/channel/cash-effect/target-Box',
        'governed_replacement': 'none complete',
        'modern_governed': 'no; IDs explicitly deferred',
        'legacy_reachable': 'yes',
        'blocker': 'FundingSupportSpec: professional support is broader than core funding and may include debt-linked/direct-obligation support',
        'removal_condition': 'first-class funding-support membership/grain contract lands and reconciles current support drilldowns',
    },
    {
        'legacy_route_family': 'FX conversion proceeds/outflow/cost',
        'governed_replacement': 'partial FlowCellSpec/semantic measures',
        'modern_governed': 'no; IDs explicitly deferred',
        'legacy_reachable': 'yes',
        'blocker': 'FX grain mismatch: current specs require Box while some professional statement rows are total-by-currency',
        'removal_condition': 'governed FX total/Box grain contract lands with no silent aggregation across missing dimensions',
    },
    {
        'legacy_route_family': 'monthly and annual debt position',
        'governed_replacement': 'DebtPositionSpec executor',
        'modern_governed': 'yes',
        'legacy_reachable': 'historical/minimal compatibility fallback',
        'blocker': 'legacy source/table schemas remain supported',
        'removal_condition': 'corpus/reachability census proves no supported pack requires pre-governed debt-position schema',
    },
    {
        'legacy_route_family': 'monthly and annual debt activity',
        'governed_replacement': 'DebtActivitySpec executor',
        'modern_governed': 'yes',
        'legacy_reachable': 'historical/minimal compatibility fallback',
        'blocker': 'legacy source/table schemas remain supported',
        'removal_condition': 'corpus/reachability census proves no supported pack requires pre-governed debt-activity schema',
    },
    {
        'legacy_route_family': 'monthly and annual validated cash position',
        'governed_replacement': 'cash position executor / governed validated selector',
        'modern_governed': 'yes',
        'legacy_reachable': 'historical/minimal compatibility fallback',
        'blocker': 'legacy cash table shapes remain supported',
        'removal_condition': 'supported packs all carry governed monthly_cash_close schema and compatibility fixtures are retired',
    },
    {
        'legacy_route_family': 'derived metric/formula drilldowns',
        'governed_replacement': 'DerivedMetricSpec executor',
        'modern_governed': 'yes for registered modern metrics',
        'legacy_reachable': 'yes for historical/unregistered compatibility',
        'blocker': 'legacy table identity and incomplete derived-spec coverage',
        'removal_condition': 'all supported derived rows have stable derived_metric_id and governed source/reconciliation contract',
    },
    {
        'legacy_route_family': 'diagnostic Box-level matrix and residual compatibility tables',
        'governed_replacement': 'no single closed contract',
        'modern_governed': 'partial',
        'legacy_reachable': 'yes',
        'blocker': 'diagnostic-specific table routing still lives in legacy router',
        'removal_condition': 'either retire diagnostic presentation table or give it a typed executor with explicit source/grain semantics',
    },
]
route_path = notes / 'accounting_simplification_phase4_drilldown_deletion_map_20260824.csv'
with route_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(route_rows[0]))
    writer.writeheader()
    writer.writerows(route_rows)

md = [
    '# Accounting simplification Phase 4 — narrow legacy facade surfaces',
    '',
    'Date: 2026-08-24  ',
    'Accounting-policy change: **none**',
    '',
    '## Invariant',
    '',
    'This phase changes import/export reachability only. It does not alter semantic classification, monthly/annual calculations, debt/cash selection, Box scope, currency separation, displayed professional values, or drilldown membership.',
    '',
    '## Facade census',
    '',
    '| Facade | Broad static delegated names before | Repo-referenced names | Explicit legacy exports after |',
    '| --- | ---: | ---: | ---: |',
]
for row in summary_rows:
    md.append(
        f"| `{row['facade']}` | {row['broad_static_surface_before']} | {row['repo_referenced_symbols']} | {row['explicit_compat_exports_after']} |"
    )
md += [
    '',
    'The old `dir(delegate) -> globals()` pattern made every imported helper, constant, private function, and future implementation detail of the delegated module appear on the modern facade. Phase 4 replaces that with a caller-derived explicit compatibility list. Star imports, `dir()`/`vars()` introspection, and dynamic `getattr()` against these facades are treated as blockers and fail the transformation.',
    '',
    'Machine-readable caller evidence: `accounting_simplification_phase4_legacy_export_inventory_20260824.csv`.',
    '',
    '## Professional drilldown deletion map',
    '',
    '| Legacy route family | Governed replacement | Legacy still reachable? | Blocker |',
    '| --- | --- | --- | --- |',
]
for row in route_rows:
    md.append(
        f"| {row['legacy_route_family']} | {row['governed_replacement']} | {row['legacy_reachable']} | {row['blocker']} |"
    )
md += [
    '',
    'The full removal conditions are in `accounting_simplification_phase4_drilldown_deletion_map_20260824.csv`.',
    '',
    '## Deliberate non-change',
    '',
    '`accounting/professional/drilldown_legacy.py` is not rewritten in this phase. Its remaining reachability is now explicit: funding support, FX grain, annual lineage, historical/minimal schemas, derived compatibility, and residual diagnostic routing are the blockers to later deletion.',
]
(notes / 'accounting_simplification_phase4_legacy_facades_20260824.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

print('Phase 4 caller census and explicit facade surfaces applied.')
for row in summary_rows:
    print(row)
