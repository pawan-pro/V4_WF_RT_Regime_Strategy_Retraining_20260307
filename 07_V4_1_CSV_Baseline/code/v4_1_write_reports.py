"""Write V4.1 comparison, RT gap analysis, and final build summary."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from v4_1_common import REFERENCE_METRICS, TIMING_VARIANTS, V4_1_ROOT, ensure_v4_1_dirs, print_json, write_json


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_rt_gap_analysis(root: Path) -> Path:
    path = root / "manifests" / "v4_1_rt_gap_analysis.md"
    text = f"""# V4.1 RT Implementation Gap Analysis

Generated: {datetime.now(timezone.utc).isoformat()}

V4.1 is a CSV-only research/audit baseline. It does not modify current V4 realtime notebooks, Saxo execution, token refresh, workflow schedules, model registry, or production Delta aliases.

## 1. Macro Carry-Forward Differences

V4.1 freezes ALFRED/FRED first-release values to CSV, assigns explicit assumed release timestamps, converts them to UTC, and joins by availability timestamp. Current RT enrichment consumes canonical historical silver context and live macro capture rows. Porting V4.1 requires matching the carry-forward clock exactly, including pre-market releases becoming usable only from the ETF decision point and UMCSENT using final releases only.

## 2. Macro Timing Assumptions

The research baseline uses deterministic assumptions because ALFRED/FRED does not provide reliable intraday public-availability timestamps. Live jobs can implement the same rules only if a release schedule source supplies event date/time and release identity in advance.

## 3. Live Release Capture Source

ALFRED-only is acceptable for historical backtest reconstruction, but it is not sufficient for live release detection/value capture at the decision timestamp. Finnhub or another economic-calendar/vendor source is still required unless a separate live data-vendor decision replaces it.

## 4. VIX/VIX3M PIT Rule

V4.1 treats daily VIX/VIX3M observation date T as unavailable on T and first usable next trading day at 10:00 ET. RT must implement that same delay before updating VIX-derived features.

## 5. Z-Score Window Alignment

V4.1 uses a 8760-hour rolling window over the CSV-built grid. RT must compute the live row using identical history, forward-fill semantics, and window endpoints.

## 6. Feature Schema Alignment

The output schema remains the 14 production model features plus SPY/QQQ close and timestamp. Schema-compatible columns are required before any model promotion dry-run.

## 7. Model Artifact Promotion

V4.1 model artifacts are research outputs under `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/models/`. Promotion to `model_registry_current` must be a separate controlled action and is intentionally out of scope here.

## 8. Dashboard/Reporting

Dashboards can compare V4.1 artifacts as offline research inputs, but production dashboard notebooks should not be changed until a dry-run proves RT feature equivalence and signal behavior.

## 9. Dry-Run Validation

Before rollout, run a shadow RT job that writes to isolated tables/files only. Required checks: feature parity vs V4.1 CSV grid, signal parity on overlapping timestamps, VIX availability timing, macro availability timing, UMCSENT final-only behavior, and no writes to live execution state.

## 10. Saxo Execution

No Saxo execution logic change is expected if V4.1 still emits regime `0=QQQ`, `1=SPY`, `2=CASH`. Execution, token refresh, and live trading workflows remain untouched by this baseline.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_variant_comparison(root: Path) -> Path:
    rows = [
        {
            "strategy": "Corrected-ALFRED reporting baseline",
            "CAGR": REFERENCE_METRICS["cagr"],
            "Sharpe": REFERENCE_METRICS["sharpe"],
            "Sortino": REFERENCE_METRICS["sortino"],
            "max_drawdown": REFERENCE_METRICS["max_drawdown"],
            "final_equity": REFERENCE_METRICS["final_equity"],
            "period": f"{REFERENCE_METRICS['start_date']} to {REFERENCE_METRICS['end_date']}",
            "source_artifact_path": "previously confirmed corrected-ALFRED reference",
            "methodology_notes": "Reference supplied in task context.",
        }
    ]
    for variant in TIMING_VARIANTS:
        metrics_path = root / "performance" / variant / "v4_1_metrics.json"
        if metrics_path.exists():
            metrics = read_json(metrics_path)["daily_eod_metrics"]
            strategy = metrics["strategy"]
            spy = metrics["spy"]
            rows.append(
                {
                    "strategy": f"V4.1 CSV-only {variant}",
                    "CAGR": strategy["cagr"],
                    "Sharpe": strategy["sharpe"],
                    "Sortino": strategy["sortino"],
                    "max_drawdown": strategy["max_drawdown"],
                    "final_equity": strategy["final_equity"],
                    "period": f"{strategy['start_date']} to {strategy['end_date']}",
                    "source_artifact_path": str(metrics_path),
                    "methodology_notes": "CSV-only grid, monthly walk-forward XGBoost, NY daily-EOD audit metrics.",
                }
            )
            if variant == TIMING_VARIANTS[0]:
                rows.append(
                    {
                        "strategy": "SPY benchmark from V4.1 main variant",
                        "CAGR": spy["cagr"],
                        "Sharpe": spy["sharpe"],
                        "Sortino": spy["sortino"],
                        "max_drawdown": spy["max_drawdown"],
                        "final_equity": spy["final_equity"],
                        "period": f"{spy['start_date']} to {spy['end_date']}",
                        "source_artifact_path": str(metrics_path),
                        "methodology_notes": "Same NY daily-EOD date selection as strategy.",
                    }
                )
    out = root / "performance" / "v4_1_variant_comparison.xlsx"
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_xlsx = Path(tmp.name)
    with pd.ExcelWriter(tmp_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="variant_comparison", index=False)
    shutil.copy2(tmp_xlsx, out)
    tmp_xlsx.unlink(missing_ok=True)
    return out


def write_summary(root: Path, rt_path: Path, comparison_path: Path) -> Path:
    input_hashes = read_json(root / "manifests" / "v4_1_input_hashes.json") if (root / "manifests" / "v4_1_input_hashes.json").exists() else {}
    sensitivity = read_json(root / "performance" / "v4_1_timing_sensitivity_comparison.json") if (root / "performance" / "v4_1_timing_sensitivity_comparison.json").exists() else {}
    main_metrics_path = root / "performance" / TIMING_VARIANTS[0] / "v4_1_metrics.json"
    main_metrics = read_json(main_metrics_path) if main_metrics_path.exists() else {}
    validation_path = root / "grids" / TIMING_VARIANTS[0] / "v4_1_silver_grid_validation.json"
    validation = read_json(validation_path) if validation_path.exists() else {}
    path = root / "manifests" / "v4_1_build_summary.md"
    text = f"""# V4.1 CSV-Only Build Summary

Generated: {datetime.now(timezone.utc).isoformat()}

## Artifact Paths

- Root: `{root}`
- Inputs: `{root / 'inputs'}`
- Main grid: `{root / 'grids' / TIMING_VARIANTS[0] / 'v4_1_silver_grid.csv'}`
- Main signals: `{root / 'signals' / TIMING_VARIANTS[0]}`
- Main performance: `{root / 'performance' / TIMING_VARIANTS[0]}`
- Timing sensitivity: `{root / 'performance' / 'v4_1_timing_sensitivity_comparison.xlsx'}`
- Variant comparison: `{comparison_path}`
- RT gap analysis: `{rt_path}`

## Input Hashes

```json
{json.dumps(input_hashes, indent=2)[:12000]}
```

## Main Grid Validation

```json
{json.dumps(validation, indent=2)[:12000]}
```

## Main Metrics

```json
{json.dumps(main_metrics.get('daily_eod_metrics', {}), indent=2)}
```

## Timing Sensitivity

```json
{json.dumps(sensitivity, indent=2)[:12000]}
```

## Comparison Notes

Mismatches versus the corrected-ALFRED reference should be classified by data differences, timing assumptions, feature-grid differences, training config differences, signal methodology differences, or performance methodology differences. V4.1 uses the production XGBoost parameter set from the current retraining notebook unless the training manifest states otherwise.

## No Production Change Confirmation

This baseline writes only under `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/`. It does not modify `silver_bt_master_grid_1h`, `bt_walkforward_regime_signals`, `model_registry_current`, Saxo execution, Saxo token refresh, live inference notebooks, RT notebooks, dashboard production notebooks, Databricks workflow schedules, or live trading workflows.
"""
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(V4_1_ROOT))
    args = parser.parse_args()
    root = ensure_v4_1_dirs(args.root)
    rt_path = write_rt_gap_analysis(root)
    comparison_path = write_variant_comparison(root)
    summary_path = write_summary(root, rt_path, comparison_path)
    payload = {"rt_gap_analysis": rt_path, "variant_comparison": comparison_path, "build_summary": summary_path}
    write_json(payload, root / "manifests" / "v4_1_report_manifest.json", root)
    print_json(payload)


if __name__ == "__main__":
    main()
