from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / "scripts/_phase1_apply.py"
text = old.read_text(encoding="utf-8")
needle = 'make = must_replace(make, human_target_block, replacement_dashboard, "human target block")'
replacement = '''
start = make.find("\\n.PHONY: run-human-report run-human-balance _run_human_balance_action\\n")
end = make.find("\\n# ========================================\\n# CHECKS", start)
if start < 0 or end < 0:
    raise RuntimeError(f"could not locate human target region: start={start} end={end}")
make = (
    make[:start]
    + "\\n.PHONY: run-dashboard\\n"
      "run-dashboard: run-metrics\\n"
      "\\t@test -s \\\"$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv\\\"\\n"
      "\\t@test -s \\\"$(RUN_METRICS_DIR)/annual_balance_dashboard_contract.csv\\\"\\n"
      "\\t@test -s \\\"$(RUN_METRICS_DIR)/annual_balance_dashboard_qa.csv\\\"\\n"
    + make[end:]
)
'''
if needle not in text:
    raise RuntimeError("expected brittle Phase 1 replacement line not found")
old.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

runpy.run_path(str(old), run_name="__main__")

# The first applier deletes itself and the push workflow. Remove all remaining
# temporary orchestration so the final PR contains only product/test/docs work.
for rel in [
    ".github/workflows/phase1-pr.yml",
    ".github/workflows/phase1-pr2.yml",
    "scripts/_phase1_apply2.py",
]:
    path = ROOT / rel
    if path.exists():
        path.unlink()
