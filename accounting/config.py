# src/accounting/config.py
"""
Configuration loader for accounting pipeline.

Usage:
    from accounting.config import load_config
    cfg = load_config()
    print(cfg.to_paths()["out"])
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import os
from pathlib import Path
import json
import sys

# Try to import yaml if available; otherwise fall back to JSON-only usage
try:
    import yaml  # PyYAML
except Exception:
    yaml = None


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    else:
        # fallback: try json
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            raise RuntimeError(
                "PyYAML not installed and config is not valid JSON. Install pyyaml or use JSON config."
            )


@dataclass
class Config:
    # general
    project_name: str = "RAG_Sync_Accounting"
    env: str = "local"  # local|staging|prod

    # directories
    base_dir: str = "."                 # project root-ish (defaults to repo root)
    out_dir: str = "./out"              # main output directory (artifact CSVs, manifests)
    time_series_subdir: str = "time_series"
    assemble_subdir: str = "assembled"
    reports_subdir: str = "reports"
    fixtures_dir: str = "tests/fixtures"
    logs_dir: str = "./logs"

    # input sources
    fixture_file: str = "tests/fixtures/ledger_fixture.csv"
    service_account_file: Optional[str] = None
    sheet_url: Optional[str] = None

    # ETL options
    freq: str = "W"
    dry_run: bool = False
    force: bool = False

    # finance defaults
    base_currency: str = "ARS"
    fx_table_path: Optional[str] = None   # path to fx table CSV used for revaluation

    # manifest & validation
    manifest_name: str = "manifest.json"
    required_headers: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_txns_parsed.csv": ["date", "source", "account", "payee", "amount", "Currency"],
            "fondos_report.csv": ["Date", "Net"],
            "renta_PM.csv": ["Date"],
        }
    )

    # backup / archive
    backup_dir: str = "./backups"
    archive_dir: str = "./archive"
    backup_retention_days: int = 30

    # logging
    log_level: str = "INFO"

    # stakeholders (optional)
    stakeholders: List[Dict[str, str]] = field(default_factory=list)

    def to_paths(self) -> Dict[str, Path]:
        """
        Return frequently used paths expanded as Path objects.
        """
        base = Path(self.base_dir).expanduser()
        out = Path(self.out_dir).expanduser()
        return {
            "base": base,
            "out": out,
            "time_series": out / self.time_series_subdir,
            "assembled": out / self.assemble_subdir,
            "reports": out / self.reports_subdir,
            "fixtures": Path(self.fixtures_dir).expanduser(),
            "logs": Path(self.logs_dir).expanduser(),
            "backups": Path(self.backup_dir).expanduser(),
            "archive": Path(self.archive_dir).expanduser(),
            "fx_table": Path(self.fx_table_path).expanduser() if self.fx_table_path else None,
        }

    def ensure_dirs(self):
        # create all runtime dirs except fixtures (fixtures expected to be present or used only in dev)
        paths = self.to_paths()
        for k, p in paths.items():
            if p is None:
                continue
            if k in ("fixtures", "fx_table"):
                # do not auto-create fixtures or fx_table files
                continue
            p.mkdir(parents=True, exist_ok=True)


def _env_override(key: str) -> Optional[str]:
    # common env naming: ACCOUNT_<KEY> uppercase underscores
    env_key = "ACCOUNT_" + key.upper().replace(".", "_")
    return os.getenv(env_key)


def load_config(yaml_path: Optional[str] = None) -> Config:
    """
    Load configuration with the following precedence:
      1. config YAML/JSON file (src/accounting/config.yaml by default)
      2. environment variables (ACCOUNT_<KEY>)
      3. dataclass defaults

    Returns a Config dataclass instance.
    """
    # locate default yaml in repo relative to this file
    repo_root = Path(__file__).resolve().parents[1]  # src/accounting -> project root
    if yaml_path:
        yaml_file = Path(yaml_path).expanduser()
    else:
        yaml_file = repo_root / "src" / "accounting" / "config.yaml"

    cfg_data = read_yaml(yaml_file) if yaml_file.exists() else {}

    # overlay explicit environment variable overrides for a small set of keys
    overrides = [
        "base_dir",
        "out_dir",
        "fixture_file",
        "service_account_file",
        "sheet_url",
        "freq",
        "dry_run",
        "force",
        "manifest_name",
        "log_level",
        "base_currency",
        "fx_table_path",
        "archive_dir",
    ]
    for k in overrides:
        val = os.getenv(f"ACCOUNT_{k.upper()}")
        if val is not None:
            if k in ("dry_run", "force"):
                cfg_data[k] = val.lower() in ("1", "true", "yes")
            else:
                cfg_data[k] = val

    # build Config dataclass (respecting defaults)
    cfg = Config(
        project_name=cfg_data.get("project_name", Config.project_name),
        env=cfg_data.get("env", Config.env),
        base_dir=cfg_data.get("base_dir", Config.base_dir),
        out_dir=cfg_data.get("out_dir", Config.out_dir),
        time_series_subdir=cfg_data.get("time_series_subdir", Config.time_series_subdir),
        assemble_subdir=cfg_data.get("assemble_subdir", Config.assemble_subdir),
        reports_subdir=cfg_data.get("reports_subdir", Config.reports_subdir),
        fixtures_dir=cfg_data.get("fixtures_dir", Config.fixtures_dir),
        logs_dir=cfg_data.get("logs_dir", Config.logs_dir),
        fixture_file=cfg_data.get("fixture_file", Config.fixture_file),
        service_account_file=cfg_data.get("service_account_file", Config.service_account_file),
        sheet_url=cfg_data.get("sheet_url", Config.sheet_url),
        freq=cfg_data.get("freq", Config.freq),
        dry_run=bool(cfg_data.get("dry_run", Config.dry_run)),
        force=bool(cfg_data.get("force", Config.force)),
        base_currency=cfg_data.get("base_currency", Config.base_currency),
        fx_table_path=cfg_data.get("fx_table_path", Config.fx_table_path),
        manifest_name=cfg_data.get("manifest_name", Config.manifest_name),
        required_headers=cfg_data.get("required_headers", Config.required_headers),
        backup_dir=cfg_data.get("backup_dir", Config.backup_dir),
        archive_dir=cfg_data.get("archive_dir", Config.archive_dir),
        backup_retention_days=int(cfg_data.get("backup_retention_days", Config.backup_retention_days)),
        log_level=cfg_data.get("log_level", Config.log_level),
        stakeholders=cfg_data.get("stakeholders", Config.stakeholders),
    )

    # sanity checks and warnings
    fpath = Path(cfg.fixture_file).expanduser()
    if not fpath.exists() and cfg.fixture_file and not (cfg.service_account_file or cfg.sheet_url):
        # warn but don't crash: fixture optional if running live against sheets
        print(
            f"[WARN] fixture file not found: {fpath} (ok if you plan to run etl-live with a Google Sheet)", file=sys.stderr
        )

    # create runtime directories
    cfg.ensure_dirs()
    return cfg


# small pretty printer
def summary(cfg: Config) -> str:
    d = asdict(cfg)
    pmap = cfg.to_paths()
    # convert Path values to str for readability
    for k, v in pmap.items():
        d.setdefault("paths", {})[k] = str(v) if v is not None else None
    s = json.dumps(d, indent=2, ensure_ascii=False)
    return s


if __name__ == "__main__":
    c = load_config()
    print(summary(c))
