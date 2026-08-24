from __future__ import annotations

import ast
from pathlib import Path

import accounting.diagnostics.funding_lineage as funding_lineage
import accounting.diagnostics.professional_issues as professional_issues


REPO_ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_forensic_tools_live_in_diagnostics_not_professional() -> None:
    assert not (REPO_ROOT / "accounting/professional/funding_lineage_audit.py").exists()
    assert not (REPO_ROOT / "accounting/professional/issue_digest.py").exists()
    assert (REPO_ROOT / "accounting/diagnostics/funding_lineage.py").exists()
    assert (REPO_ROOT / "accounting/diagnostics/professional_issues.py").exists()


def test_moved_diagnostics_keep_their_public_capabilities() -> None:
    assert callable(funding_lineage.build_audit)
    assert callable(funding_lineage.write_outputs)
    assert callable(funding_lineage.main)
    assert callable(professional_issues.build_issue_rows)
    assert callable(professional_issues.build_summary_rows)
    assert callable(professional_issues.main)


def test_diagnostics_do_not_import_professional_runtime_modules() -> None:
    for rel in [
        "accounting/diagnostics/funding_lineage.py",
        "accounting/diagnostics/professional_issues.py",
    ]:
        imports = _imports(REPO_ROOT / rel)
        assert not any(name == "accounting.professional" or name.startswith("accounting.professional.") for name in imports), (rel, imports)
