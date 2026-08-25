from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_matches_canonical_make_aliases() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "run-canonical: run-materialize" in makefile
    assert "run-marts" not in makefile
    assert "views_sanity" not in makefile
    assert "accounting.marts.build" not in makefile
    assert "run-accounting: run-accounting-full" in makefile
    assert "run-accounting-full: run-full" in makefile

    assert "make run-canonical" in readme
    assert "make run-full" in readme
    assert "compatibility aliases for `run-full`" in readme
    assert "There is no separate generic views stage" in readme
    assert "make run-views" not in readme
