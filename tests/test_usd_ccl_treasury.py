from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from accounting.management.usd_ccl_treasury import build_monthly_treasury_projection
from accounting.reports.treasury_usd_ccl.render import render_report
from accounting.valuation.ccl_snapshot import CCLSnapshotError, build_snapshot


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_ccl_snapshot_uses_business_day_mean_and_preserves_hash(tmp_path: Path) -> None:
    outputs = build_snapshot(
        [
            {"rate_date": "2026-10-09", "ars_per_usd_ccl": "1611.26", "rate_source": "RAVA", "rate_series": "CCL", "source_reference": "fri"},
            {"rate_date": "2026-10-10", "ars_per_usd_ccl": "1611.26", "rate_source": "RAVA", "rate_series": "CCL", "source_reference": "sat"},
            {"rate_date": "2026-10-08", "ars_per_usd_ccl": "1609.53", "rate_source": "RAVA", "rate_series": "CCL", "source_reference": "thu"},
        ], cutoff_date="2026-10-10", source_name="CCL CANON", source_reference="fixture", output_dir=tmp_path,
    )
    mean = _read(outputs["monthly_mean"])[0]
    assert mean["observations"] == "2"
    assert mean["ccl_mean_ars_per_usd"] == "1610.395000000000"
    assert mean["coverage_status"] == "partial"
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["non_business_day_observations_excluded_from_mean"] == 1
    assert len(manifest["snapshot_sha256"]) == 64


def test_ccl_snapshot_rejects_duplicate_dates(tmp_path: Path) -> None:
    row = {"rate_date": "2026-09-01", "ars_per_usd_ccl": "1", "rate_source": "RAVA", "rate_series": "CCL", "source_reference": "x"}
    with pytest.raises(CCLSnapshotError, match="duplicate CCL observation"):
        build_snapshot([row, row], cutoff_date="2026-10-10", source_name="CCL CANON", source_reference="fixture", output_dir=tmp_path)


def test_exact_run_projection_matches_rounded_checkpoints() -> None:
    root = Path("out/treasury_usd_ccl/20261010T174500Z_FBPM")
    if not (root / "projection/monthly_treasury_usd_ccl.csv").exists():
        pytest.skip("private exact-run artifacts are unavailable")
    rows = _read(root / "projection/monthly_treasury_usd_ccl.csv")
    selected = {(r["period"], r["Box"]): r for r in rows if r["period"] in {"2026-09", "2026-10"}}
    assert float(selected[("2026-09", "Family Business")]["usd_ccl_net_flow"]) == pytest.approx(2443.46, abs=0.01)
    assert float(selected[("2026-09", "Property Management")]["usd_ccl_net_flow"]) == pytest.approx(411.18, abs=0.01)
    assert float(selected[("2026-10", "Family Business")]["usd_ccl_net_flow"]) == pytest.approx(2479.39, abs=0.01)
    assert float(selected[("2026-10", "Property Management")]["usd_ccl_net_flow"]) == pytest.approx(455.59, abs=0.01)


def test_exact_run_usd_ccl_renderer_reconciles_html_without_touching_native_report(tmp_path: Path) -> None:
    root = Path("out/treasury_usd_ccl/20261010T174500Z_FBPM")
    native = Path("out/reports/20261010T174500Z_FBPM/treasury_accountability/report.html")
    required = [
        root / "projection/monthly_treasury_usd_ccl.csv",
        root / "projection/treasury_usd_ccl_aggregates.csv",
        root / "projection/treasury_usd_ccl_cumulative_control.csv",
        root / "projection/treasury_usd_ccl_reconciliation_manifest.json",
        root / "ccl_snapshot_manifest.json",
        native,
    ]
    if not all(path.exists() for path in required):
        pytest.skip("private exact-run/report artifacts are unavailable")
    before = native.read_bytes()
    outputs = render_report(
        monthly_path=required[0], aggregates_path=required[1], cumulative_control_path=required[2],
        projection_manifest_path=required[3], fx_manifest_path=required[4], out_dir=tmp_path,
        run_id="20261010T174500Z_FBPM", cutoff_date="2026-10-10", render_pdf=False,
    )
    html = outputs["html"].read_text(encoding="utf-8")
    assert "Tesorería mensual · USD/CCL" in html
    assert "YTD 2026" in html and "TOTAL SERIE" in html
    assert html.count('data-aggregation="annual"') == 9
    assert html.count('data-aggregation="ytd"') == 2
    assert html.count('data-aggregation="total"') == 2
    aggregates = _read(required[1])
    assert {row["period"] for row in aggregates if row["aggregation"] == "ytd"} == {"2026"}
    assert {row["period"] for row in aggregates if row["aggregation"] == "total"} == {"all"}
    assert json.loads(outputs["manifest"].read_text(encoding="utf-8"))["validation_status"] == "pass"
    assert native.read_bytes() == before
