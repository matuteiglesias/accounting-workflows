from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


BROWSER_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
)


def resolve_browser(browser_bin: str | Path | None = None) -> Path:
    requested = str(browser_bin or os.environ.get("REPORT_BROWSER_BIN", "")).strip()
    if requested:
        path = Path(requested)
        if path.is_file():
            return path
        resolved = shutil.which(requested)
        if resolved:
            return Path(resolved)
        raise FileNotFoundError(f"Configured report browser not found: {requested}")

    for candidate in BROWSER_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)
    raise FileNotFoundError(
        "No headless Chromium/Chrome executable found. Set REPORT_BROWSER_BIN."
    )


def render_pdf(
    html_path: str | Path,
    pdf_path: str | Path,
    *,
    browser_bin: str | Path | None = None,
    timeout_seconds: int = 90,
) -> Path:
    source = Path(html_path).resolve(strict=True)
    target = Path(pdf_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    browser = resolve_browser(browser_bin)

    command = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={target}",
        source.as_uri(),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "PDF rendering failed "
            f"returncode={result.returncode} stderr={result.stderr.strip()!r}"
        )
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"PDF renderer did not create a non-empty file: {target}")
    with target.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise RuntimeError(f"PDF renderer produced an invalid file header: {target}")
    return target
