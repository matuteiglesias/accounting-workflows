from __future__ import annotations

"""Temporary, publication-free execution context for exact-run accounting stages.

This is orchestration only: producers retain their accounting logic.
No live ingest, latest updates, publication, or arbitrary Make targets.
Not an operating-system sandbox; child processes still run as the user.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator


ALLOWED_STAGES = frozenset({
    "run-materialize",
    "run-debt",
    "run-metrics",
    "run-reports",
})


@dataclass(frozen=True)
class IsolatedAccountingExecution:
    repo_root: Path
    workspace: Path
    run_id: str

    @property
    def out_root(self) -> Path:
        return self.workspace / "out"

    @property
    def canonical_root(self) -> Path:
        return self.out_root / "run" / "accounting" / self.run_id

    @property
    def debt_root(self) -> Path:
        return self.out_root / "debt_resolution" / self.run_id

    @property
    def metrics_root(self) -> Path:
        return self.out_root / "metrics" / self.run_id

    @property
    def reports_root(self) -> Path:
        return self.out_root / "reports" / self.run_id

    def make_command(self, target: str, *, boxes: str = "Family Business,Property Management") -> list[str]:
        if target not in ALLOWED_STAGES:
            raise ValueError(f"stage not authorized for isolated execution: {target!r}")
        # The supported report/release contracts currently assume FBPM.
        # Refuse arbitrary scope interpolation into Make's shell recipes.
        if boxes != "Family Business,Property Management":
            raise ValueError("isolated execution currently supports FBPM only")
        return [
            "make", "--no-print-directory", target,
            f"OUT={self.out_root}",
            f"RUN_ID={self.run_id}",
            f"BOXES={boxes}",
        ]

    def run_stage(self, target: str, *, boxes: str = "Family Business,Property Management") -> None:
        # Explicit allowlist excludes run-full, ingest, latest, publish and professional targets.
        subprocess.run(
            self.make_command(target, boxes=boxes),
            cwd=self.repo_root, check=True,
        )

    def export(self, source: Path, destination: Path) -> Path:
        """Preserve one requested deliverable before workspace deletion; no overwrite."""
        source_path = Path(source).resolve(strict=True)
        output_root = self.out_root.resolve(strict=True)
        if not source_path.is_relative_to(output_root) or not source_path.is_file():
            raise ValueError("export source must be a file owned by the isolated execution")
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing overwrite of requested deliverable: {destination}")
        target_parent = destination.parent.resolve(strict=True)
        if target_parent.is_relative_to(self.workspace.resolve(strict=True)):
            raise ValueError("export destination must outlive the temporary workspace")
        # Do not allow exporting directly over any production accounting destination.
        if target_parent.is_relative_to((self.repo_root / "out").resolve()) or target_parent.is_relative_to((self.repo_root / "public").resolve()):
            raise ValueError("export destination cannot be the production output tree")
        shutil.copy2(source_path, destination)
        return destination


@contextmanager
def isolated_accounting_execution(
    repo_root: Path, *, temp_parent: Path | None = None,
) -> Iterator[IsolatedAccountingExecution]:
    repo_root = Path(repo_root).resolve(strict=True)
    if not (repo_root / "Makefile").is_file():
        raise ValueError("expected accounting-workflows checkout")
    parent = None if temp_parent is None else Path(temp_parent).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="accounting-isolated-", dir=parent) as temp:
        workspace = Path(temp).resolve(strict=True)
        if workspace.is_relative_to(repo_root) or repo_root.is_relative_to(workspace):
            raise ValueError("isolated workspace cannot contain or live inside the repository")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_FBPM"
        execution = IsolatedAccountingExecution(repo_root=repo_root, workspace=workspace, run_id=run_id)
        execution.out_root.mkdir()
        yield execution
