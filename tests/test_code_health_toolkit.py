"""Read-only infrastructure diagnostics: static graph and status integrity."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import code_health as health


def test_source_inventory_excludes_other_packages_and_outputs(tmp_path: Path):
    source = tmp_path / 'accounting' / 'professional'
    source.mkdir(parents=True)
    (source / '__init__.py').write_text('')
    (source / 'core.py').write_text('import accounting.scope\n')
    (tmp_path / 'accounting' / 'scope.py').write_text('pass\n')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'test_fake.py').write_text('pass\n')
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'generated.py').write_text('pass\n')
    modules, edges = health.internal_imports(tmp_path)
    assert 'accounting.professional.core' in modules
    assert 'accounting.professional' in modules
    assert 'accounting.scope' in edges['accounting.professional.core']
    assert len(modules) == 3


def test_relative_imports_and_true_cycles(tmp_path: Path):
    pkg = tmp_path / 'accounting' / 'reports'
    pkg.mkdir(parents=True)
    (pkg / '__init__.py').write_text('')
    (pkg / 'a.py').write_text('from . import b\n')
    (pkg / 'b.py').write_text('from .a import something\n')
    _, edges = health.internal_imports(tmp_path)
    assert 'accounting.reports.b' in edges['accounting.reports.a']
    assert edges['accounting.reports.b'] == {'accounting.reports.a'}
    assert health.cycles(edges) == [['accounting.reports.a', 'accounting.reports.b']]


def test_package_graph_filter_does_not_delete_machine_readable_edges():
    graph = {'package_sizes': {'a': 1, 'b': 2}, 'package_imports': [
        {'from': 'a', 'to': 'b', 'count': 1},
        {'from': 'b', 'to': 'a', 'count': 2},
    ]}
    dot = health.dot_graph(graph)
    assert '"b" -> "a"' in dot
    assert '"a" -> "b"' not in dot
    assert 'Only links with >=2' in dot


def test_report_does_not_call_missing_or_failed_check_a_pass(tmp_path: Path):
    health.save(tmp_path / 'inspection.json', {'tools': {'complexity': {
        'status': 'missing', 'message': 'Tool unavailable: radon'}}, 'summary': {}})
    health.write_report(tmp_path)
    report = (tmp_path / 'REPORT.md').read_text()
    assert 'NOT RUN' in report
    assert 'MISSING' in report
    assert 'No correctness claim' in report
    health.save(tmp_path / 'checks.json', {'status': 'fail', 'exit_code': 2, 'elapsed_seconds': 1})
    health.write_report(tmp_path)
    assert '**FAIL**: existing `make validate`' in (tmp_path / 'REPORT.md').read_text()


def test_complexity_deduplicates_class_method_and_standalone():
    record = {'type': 'method', 'name': 'do', 'lineno': 4, 'complexity': 15, 'rank': 'C'}
    cls = {'type': 'class', 'name': 'A', 'lineno': 1, 'methods': [record]}
    rows = health.top_complexity({'accounting/test.py': [cls, record]})
    assert len(rows) == 1
    assert rows[0]['name'] == 'A.do'


def test_advisory_analysis_cannot_hide_error_exit(tmp_path: Path, monkeypatch):
    # An unexpected tool traceback is not treated as a Vulture candidate.
    def fake_run(argv, **kwargs):
        if argv[0] == 'vulture':
            return {'status': 'error', 'exit_code': 1, 'stdout': 'Traceback: error',
                    'stderr': '', 'elapsed_seconds': 0.1}
        if argv[0] == 'radon':
            return {'status': 'missing', 'command': argv, 'elapsed_seconds': 0}
        return {'status': 'ok', 'stdout': 'test', 'stderr': '', 'elapsed_seconds': 0}
    monkeypatch.setattr(health, 'run', fake_run)
    monkeypatch.setattr(health, 'structural', lambda root: {
        'module_count': 0, 'module_import_edges': 0, 'package_count': 0,
        'cross_package_import_edges': 0, 'module_cycles': [], 'largest_files': [],
        'highest_fanout': [], 'highest_fanin': []})
    health.inspect(tmp_path, tmp_path / '.reports')
    state = json.loads((tmp_path / '.reports' / 'inspection.json').read_text())
    assert state['tools']['unused_candidates']['status'] == 'error'
    assert state['tools']['complexity']['status'] == 'missing'
