from __future__ import annotations

"""Opt-in wall-clock timings for Python subprocesses launched by Make.

Set PY="python3 -m accounting.diagnostics.timed_python" and
ACCOUNTING_TIMINGS_JSONL=<absolute path>. This wrapper is deliberately
inactive for normal production commands and passes through exit status.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from datetime import datetime, timezone


def main() -> int:
    args = sys.argv[1:]
    if not args:
        raise SystemExit("Usage: python -m accounting.diagnostics.timed_python <python args>")
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        code = subprocess.run([sys.executable, *args], check=False).returncode
    except BaseException:
        code = 127
        raise
    finally:
        destination = os.environ.get("ACCOUNTING_TIMINGS_JSONL")
        if destination:
            record = {
                "kind": "python_subprocess",
                "started_utc": started_utc,
                "elapsed_seconds": round(perf_counter() - started, 6),
                "command": args,
                "exit_code": code,
                "parent_pid": os.getpid(),
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with os.fdopen(
                os.open(Path(destination), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600),
                "w",
                encoding="utf-8",
            ) as fp:
                fp.write(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
