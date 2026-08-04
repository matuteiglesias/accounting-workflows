from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_matches_canonical_make_aliases() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "run-canonical: run-marts" in makefile
    assert "run-accounting: run-accounting-full" in makefile
    assert "run-accounting-full: run-full" in makefile

    assert "make run-canonical" in readme
    assert "make run-full" in readme
    assert "compatibility aliases for `run-full`" in readme
    assert "resolves to `run-human-report`" not in readme
    assert "make run-views" not in readme
