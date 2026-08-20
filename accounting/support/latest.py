"""Atomic, scope-qualified latest pointers for accounting producer surfaces."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from accounting.cutoff import load_run_cutoff_if_present
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


def cutoff_for_latest_target(bases: list[Path], target: str):
    """Return a Stage A cutoff found among target surfaces, before mutating pointers."""
    for base in bases:
        cutoff = load_run_cutoff_if_present(Path(base) / target)
        if cutoff is not None:
            return cutoff
    return None


def assert_latest_target_publishable(
    bases: list[Path],
    target: str,
    *,
    allow_cutoff_latest: bool = False,
) -> None:
    """Protect ordinary latest pointers from silently becoming historical backfills."""
    cutoff = cutoff_for_latest_target(bases, target)
    if cutoff is not None and not allow_cutoff_latest:
        raise ValueError(
            "Refusing to move latest pointers to a historical cutoff run "
            f"({target}, cutoff={cutoff.date}). Use the exact run paths, or pass "
            "--allow-cutoff-latest only for a deliberate publication decision."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="append", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--scope-tag")
    parser.add_argument("--boxes")
    parser.add_argument(
        "--allow-cutoff-latest",
        action="store_true",
        help="Explicitly permit historical cutoff runs to replace latest pointers.",
    )
    args = parser.parse_args()
    tag = args.scope_tag or canonical_scope_tag(parse_box_scope(args.boxes))
    assert_latest_target_publishable(
        args.base,
        args.target,
        allow_cutoff_latest=bool(args.allow_cutoff_latest),
    )
    for base in args.base:
        update_scoped_latest(base, args.target, tag)


if __name__ == "__main__":
    main()
