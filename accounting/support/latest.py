"""Atomic, scope-qualified latest pointers for accounting producer surfaces."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from accounting.scope import canonical_scope_tag, parse_box_scope

PRIMARY_SCOPE_TAG = "FBPM"


def _replace_symlink(link: Path, target: str) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise ValueError(f"Refusing to replace non-symlink latest path: {link}")
    tmp = link.parent / f".{link.name}.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp, target_is_directory=True)
    os.replace(tmp, link)


def update_scoped_latest(base: Path, target: str, scope_tag: str) -> dict[str, Path]:
    """Update only this scope; plain latest is the stable FBPM compatibility alias."""
    base = Path(base)
    target_path = base / target
    if not target_path.is_dir():
        raise FileNotFoundError(f"Latest target does not exist: {target_path}")
    scoped = base / f"latest_{scope_tag}"
    _replace_symlink(scoped, target)
    updated = {"scoped": scoped}
    if scope_tag == PRIMARY_SCOPE_TAG:
        plain = base / "latest"
        _replace_symlink(plain, target)
        updated["plain"] = plain
    return updated


def update_primary_compatibility_latest(base: Path, target: str, scope_tag: str) -> Path | None:
    """Point plain latest at an existing primary scoped directory only."""
    if scope_tag != PRIMARY_SCOPE_TAG:
        return None
    base = Path(base)
    if not (base / target).is_dir():
        raise FileNotFoundError(f"Compatibility latest target does not exist: {base / target}")
    plain = base / "latest"
    _replace_symlink(plain, target)
    return plain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="append", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--scope-tag")
    parser.add_argument("--boxes")
    args = parser.parse_args()
    tag = args.scope_tag or canonical_scope_tag(parse_box_scope(args.boxes))
    for base in args.base:
        update_scoped_latest(base, args.target, tag)


if __name__ == "__main__":
    main()
