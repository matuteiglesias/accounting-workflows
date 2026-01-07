# accounting/storypack_io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _read_csv(p: Path, **kw) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing expected artifact: {p}")
    return pd.read_csv(p, **kw)

def load_run(run_out: str | Path) -> dict[str, object]:
    run_out = Path(run_out)
    out: dict[str, object] = {"run_out": run_out}

    out["ledger"] = _read_csv(run_out / "ledger_canonical.csv")
    out["daily_cash"] = _read_csv(run_out / "daily_cash_position.csv")

    # optional materialize outputs (freq-specific)
    for name in ["per_party_time_long.freq=W.csv", "per_flow_time_long.freq=W.csv", "loans_time.freq=M.csv"]:
        p = run_out / name
        if p.exists():
            out[name.replace(".csv","")] = _read_csv(p)

    # reports
    reports_dir = run_out / "reports"
    out["reports_dir"] = reports_dir
    if reports_dir.exists():
        for fname in ["fondos_report.csv", "renta_PM_ARS.csv", "renta_PM_USD.csv", "renta_FB_ARS.csv"]:
            p = reports_dir / fname
            if p.exists():
                out[f"report::{fname}"] = _read_csv(p)

    # meta
    meta_dir = run_out / "meta"
    out["meta_dir"] = meta_dir
    for fname in ["reports_summary.json", "partitions.json"]:
        p = meta_dir / fname
        if p.exists():
            out[f"meta::{fname}"] = p  # dejalo como Path; lo parseás si querés

    return out


# accounting/storypack_export.py
from __future__ import annotations
from pathlib import Path
import json

def ensure_dirs(write_dir: Path) -> dict[str, Path]:
    figs = write_dir / "figs"
    tables = write_dir / "tables"
    html = write_dir / "html"
    for d in [figs, tables, html, write_dir / "nbs_executed"]:
        d.mkdir(parents=True, exist_ok=True)
    return {"figs": figs, "tables": tables, "html": html}

def write_summary(write_dir: Path, payload: dict) -> None:
    p = write_dir / "storypack_summary.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
