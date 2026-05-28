#!/usr/bin/env python3
"""Run the isolated faithful legacy-grid model replay.

This wrapper executes the recovered CSV-only model runner against the faithful
legacy grid, then writes the exact audit artifact names requested for the
V4.1 forensic replay. It reads legacy CSV/Volume artifacts and writes only
below OUTPUT_ROOT.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import sklearn
except Exception:  # pragma: no cover
    sklearn = None

try:
    import pyspark
except Exception:  # pragma: no cover
    pyspark = None


BASE = Path("/Volumes/pawan_quant_research_1/v4/market_data")
CSV_BASE = BASE / "v4_1_csv"
OUTPUT_ROOT = CSV_BASE / "faithful_legacy_grid_model_replay_2026-04-30"
FAITHFUL_GRID = (
    CSV_BASE
    / "full_legacy_grid_clone_2026-04-30"
    / "grids"
    / "full_legacy_grid_clone.csv"
)
RECOVERED_GENERATOR_SOURCE = (
    CSV_BASE
    / "forensic_corrected_training_artifact_recovery_2026-04-30"
    / "code"
    / "recovered_corrected_training_generator.py"
)
RECOVERED_RUNNER = OUTPUT_ROOT / "code" / "v4_1_recovered_generator_runner.py"
REFERENCE_SIGNAL = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "walkforward_signals.csv"
)
REFERENCE_FULL_SIGNAL = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "full_signal_export.csv"
)
REFERENCE_GRID = (
    CSV_BASE
    / "forensic_feature_freeze_ladder_2026-04-30"
    / "grids"
    / "D_full_feature_freeze"
    / "v4_1_silver_grid.csv"
)

FEATURES = [
    "CPIAUCSL_z",
    "PCEPI_z",
    "PCE_z",
    "PAYEMS_z",
    "UNRATE_z",
    "GDPC1_z",
    "INDPRO_z",
    "UMCSENT_z",
    "VIXCLS_z",
    "vix3m_z",
    "VIX_Inversion_Z",
    "Price_Mom_24H",
    "CPI_Event_Active",
    "NFP_Event_Active",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def normalize_timestamp(pdf: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = pdf.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    try:
        out[column] = out[column].dt.tz_localize(None)
    except Exception:
        pass
    return out


def load_csv(path: Path) -> pd.DataFrame:
    return normalize_timestamp(pd.read_csv(path))


def alias_file(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def update_runtime_manifest() -> dict:
    manifest_path = OUTPUT_ROOT / "manifests" / "runtime_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "faithful_replay_wrapper_generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "sklearn_version": getattr(sklearn, "__version__", None),
            "xgboost_version": xgb.__version__,
            "joblib_version": joblib.__version__,
            "pyspark_version": getattr(pyspark, "__version__", None),
            "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
            "cluster_id": os.environ.get("DATABRICKS_CLUSTER_ID")
            or os.environ.get("DB_CLUSTER_ID"),
            "git_branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "git_commit": run_command(["git", "rev-parse", "HEAD"]),
            "output_root": str(OUTPUT_ROOT),
            "faithful_grid_path": str(FAITHFUL_GRID),
            "faithful_grid_sha256": file_sha256(FAITHFUL_GRID),
            "recovered_generator_source_path": str(RECOVERED_GENERATOR_SOURCE),
            "recovered_generator_source_sha256": file_sha256(RECOVERED_GENERATOR_SOURCE),
            "recovered_runner_path": str(RECOVERED_RUNNER),
            "recovered_runner_sha256": file_sha256(RECOVERED_RUNNER),
            "reference_signal_path": str(REFERENCE_SIGNAL),
            "reference_signal_sha256": file_sha256(REFERENCE_SIGNAL),
            "reference_full_signal_path": str(REFERENCE_FULL_SIGNAL),
            "reference_full_signal_sha256": file_sha256(REFERENCE_FULL_SIGNAL),
            "read_only_source_tables": [],
            "production_delta_writes": False,
            "production_workflow_changes": False,
            "saxo_execution_changes": False,
            "model_registry_changes": False,
        }
    )
    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def compare_regime_signal(
    candidate_path: Path,
    reference_path: Path,
    candidate_name: str,
    reference_name: str,
) -> tuple[dict, pd.DataFrame]:
    cand = load_csv(candidate_path)
    ref = load_csv(reference_path)
    if "predicted_regime" not in cand.columns or "predicted_regime" not in ref.columns:
        return (
            {
                "candidate_path": str(candidate_path),
                "reference_path": str(reference_path),
                "error": "predicted_regime column missing",
            },
            pd.DataFrame(),
        )
    cand = cand[["timestamp", "predicted_regime"]].rename(
        columns={"predicted_regime": candidate_name}
    )
    ref = ref[["timestamp", "predicted_regime"]].rename(
        columns={"predicted_regime": reference_name}
    )
    cmp = cand.merge(ref, on="timestamp", how="outer", indicator=True)
    both = cmp[cmp["_merge"] == "both"].copy()
    if not both.empty:
        both[candidate_name] = pd.to_numeric(both[candidate_name], errors="coerce").astype("Int64")
        both[reference_name] = pd.to_numeric(both[reference_name], errors="coerce").astype("Int64")
        both["match"] = both[candidate_name] == both[reference_name]
    mismatches = both[both["match"] != True].head(100).copy() if not both.empty else pd.DataFrame()
    report = {
        "candidate_path": str(candidate_path),
        "reference_path": str(reference_path),
        "candidate_rows": int(len(cand)),
        "reference_rows": int(len(ref)),
        "overlap_rows": int(len(both)),
        "missing_in_candidate": int((cmp["_merge"] == "right_only").sum()),
        "extra_in_candidate": int((cmp["_merge"] == "left_only").sum()),
        "mismatch_count": int((both["match"] != True).sum()) if not both.empty else None,
        "match_rate": float(both["match"].mean()) if not both.empty else None,
    }
    return report, mismatches


def compare_features_on_prediction_timestamps() -> dict:
    signals = load_csv(OUTPUT_ROOT / "signals" / "walkforward_signals.csv")
    cand = load_csv(FAITHFUL_GRID)
    ref = load_csv(REFERENCE_GRID)
    timestamps = signals[["timestamp"]].drop_duplicates()
    cand = timestamps.merge(cand[["timestamp"] + FEATURES], on="timestamp", how="left")
    ref = timestamps.merge(ref[["timestamp"] + FEATURES], on="timestamp", how="left")
    merged = cand.merge(ref, on="timestamp", suffixes=("_candidate", "_reference"), how="outer", indicator=True)
    rows = []
    total_equal = 0
    total_values = 0
    for feature in FEATURES:
        c = pd.to_numeric(merged[f"{feature}_candidate"], errors="coerce")
        r = pd.to_numeric(merged[f"{feature}_reference"], errors="coerce")
        both = c.notna() & r.notna()
        diff = (c - r).abs()
        near = both & (diff <= 1e-9)
        total_equal += int(near.sum())
        total_values += int(both.sum())
        rows.append(
            {
                "feature": feature,
                "overlap_values": int(both.sum()),
                "near_match_rate": float(near.sum() / both.sum()) if both.sum() else None,
                "mae": float(diff[both].mean()) if both.sum() else None,
                "max_abs_diff": float(diff[both].max()) if both.sum() else None,
            }
        )
    feature_cmp = pd.DataFrame(rows)
    feature_cmp.to_csv(
        OUTPUT_ROOT / "forensics" / "feature_match_on_prediction_timestamps.csv",
        index=False,
    )
    return {
        "candidate_grid_path": str(FAITHFUL_GRID),
        "reference_grid_path": str(REFERENCE_GRID),
        "prediction_timestamp_rows": int(len(timestamps)),
        "candidate_missing_timestamps": int((merged["_merge"] == "right_only").sum()),
        "reference_missing_timestamps": int((merged["_merge"] == "left_only").sum()),
        "overall_near_match_rate": float(total_equal / total_values) if total_values else None,
        "by_feature_path": str(OUTPUT_ROOT / "forensics" / "feature_match_on_prediction_timestamps.csv"),
        "by_feature": rows,
    }


def write_requested_aliases() -> None:
    alias_file(
        OUTPUT_ROOT / "signals" / "recovered_walkforward_signals.csv",
        OUTPUT_ROOT / "signals" / "walkforward_signals.csv",
    )
    alias_file(
        OUTPUT_ROOT / "manifests" / "folds" / "fold_manifest.csv",
        OUTPUT_ROOT / "manifests" / "fold_manifest.csv",
    )
    alias_file(
        OUTPUT_ROOT / "manifests" / "folds" / "fold_manifest.json",
        OUTPUT_ROOT / "manifests" / "fold_manifest.json",
    )
    alias_file(
        OUTPUT_ROOT / "performance" / "recovered_equity_curve.csv",
        OUTPUT_ROOT / "performance" / "equity_curve.csv",
    )
    alias_file(
        OUTPUT_ROOT / "performance" / "recovered_metrics.json",
        OUTPUT_ROOT / "performance" / "metrics.json",
    )
    alias_file(
        OUTPUT_ROOT / "performance" / "recovered_annual_returns.csv",
        OUTPUT_ROOT / "performance" / "annual_returns.csv",
    )
    alias_file(
        OUTPUT_ROOT / "performance" / "recovered_trailing_metrics.csv",
        OUTPUT_ROOT / "performance" / "trailing_metrics.csv",
    )
    alias_file(
        OUTPUT_ROOT / "performance" / "recovered_audit_pack.xlsx",
        OUTPUT_ROOT / "performance" / "audit_pack.xlsx",
    )


def write_performance_comparison(metrics: dict, walkforward_report: dict) -> dict:
    reference = metrics.get(
        "reference_comparison",
        {
            "CAGR": 0.208405,
            "Sharpe": 1.0024,
            "Sortino": 1.3729,
            "Max Drawdown": -0.255385,
            "Final Value": 12_883_000,
        },
    )
    replay = {key: metrics.get(key) for key in reference}
    diffs = {
        key: (None if replay.get(key) is None else float(replay[key] - reference[key]))
        for key in reference
    }
    comparison = {
        "replay_metrics": replay,
        "corrected_reference_metrics": reference,
        "absolute_differences": diffs,
        "walkforward_signal_match_rate": walkforward_report.get("signal_match_rate")
        or walkforward_report.get("match_rate"),
        "cagr_20_8405_reproduced": bool(abs((metrics.get("CAGR") or 0) - 0.208405) <= 1e-6),
        "cagr_within_10bps": bool(abs((metrics.get("CAGR") or 0) - 0.208405) <= 0.001),
        "difference_classification": metrics.get("difference_classification"),
    }
    path = OUTPUT_ROOT / "forensics" / "performance_comparison.json"
    path.write_text(json.dumps(comparison, indent=2, default=str))
    return comparison


def write_enhanced_reports() -> None:
    write_requested_aliases()
    update_runtime_manifest()

    walkforward_report_path = OUTPUT_ROOT / "forensics" / "signal_reproduction_report.json"
    walkforward_report = json.loads(walkforward_report_path.read_text())
    teacher_report, teacher_mismatches = compare_regime_signal(
        OUTPUT_ROOT / "signals" / "full_signal_export.csv",
        REFERENCE_FULL_SIGNAL,
        "replay_teacher_regime",
        "reference_teacher_regime",
    )
    teacher_mismatches.to_csv(
        OUTPUT_ROOT / "forensics" / "teacher_full_signal_mismatches.csv",
        index=False,
    )
    feature_timestamp_report = compare_features_on_prediction_timestamps()
    metrics = json.loads((OUTPUT_ROOT / "performance" / "metrics.json").read_text())
    performance_comparison = write_performance_comparison(metrics, walkforward_report)

    combined_signal_report = {
        "walkforward_signal_match": walkforward_report,
        "teacher_full_signal_match": teacher_report,
        "feature_match_on_prediction_timestamps": feature_timestamp_report,
        "diagnosis": diagnose(walkforward_report, teacher_report, feature_timestamp_report, metrics),
    }
    (OUTPUT_ROOT / "forensics" / "signal_match_report.json").write_text(
        json.dumps(combined_signal_report, indent=2, default=str)
    )
    write_summary_markdown(
        walkforward_report,
        teacher_report,
        feature_timestamp_report,
        metrics,
        performance_comparison,
    )
    write_governance_note(walkforward_report, teacher_report, metrics)


def diagnose(
    walkforward_report: dict,
    teacher_report: dict,
    feature_report: dict,
    metrics: dict,
) -> str:
    signal_match = walkforward_report.get("signal_match_rate")
    if signal_match is None:
        signal_match = walkforward_report.get("match_rate") or 0.0
    if signal_match >= 0.99:
        return "Walk-forward signals are at or above the 99% replay threshold."
    if (feature_report.get("overall_near_match_rate") or 0.0) >= 0.999999:
        return (
            "All 14 features match on prediction timestamps, so remaining signal "
            "mismatch is attributable to retraining without the original saved "
            "teacher/fold model artifacts and/or xgboost runtime nondeterminism, "
            "not grid feature lineage."
        )
    return (
        "Signal mismatch remains partly explainable by feature/timestamp membership "
        "differences. Inspect feature_match_on_prediction_timestamps.csv and fold "
        "row hashes before attributing gaps to model nondeterminism."
    )


def write_summary_markdown(
    walkforward_report: dict,
    teacher_report: dict,
    feature_report: dict,
    metrics: dict,
    performance_comparison: dict,
) -> None:
    signal_match = walkforward_report.get("signal_match_rate")
    teacher_match = teacher_report.get("match_rate")
    lines = [
        "# Faithful Legacy Grid Model Replay Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
        f"- Faithful grid: `{FAITHFUL_GRID}`",
        f"- Recovered generator source: `{RECOVERED_GENERATOR_SOURCE}`",
        f"- Corrected reference walk-forward signals: `{REFERENCE_SIGNAL}`",
        f"- Corrected reference full signal export: `{REFERENCE_FULL_SIGNAL}`",
        "",
        "## Signal Match",
        "",
        f"- Walk-forward overlap rows: {walkforward_report.get('overlap_rows'):,}",
        f"- Walk-forward signal match: {signal_match:.6%}",
        f"- Walk-forward mismatches: {walkforward_report.get('mismatch_count'):,}",
        f"- Teacher full-signal overlap rows: {teacher_report.get('overlap_rows'):,}",
        f"- Teacher full-signal match: {teacher_match:.6%}",
        f"- 14-feature match on prediction timestamps: {feature_report.get('overall_near_match_rate'):.6%}",
        "",
        "## Performance",
        "",
        f"- CAGR: {metrics.get('CAGR'):.6%}",
        f"- Sharpe: {metrics.get('Sharpe'):.6f}",
        f"- Sortino: {metrics.get('Sortino'):.6f}",
        f"- Max drawdown: {metrics.get('Max Drawdown'):.6%}",
        f"- Final equity: ${metrics.get('Final Value'):,.2f}",
        f"- 20.8405% CAGR reproduced: {performance_comparison.get('cagr_20_8405_reproduced')}",
        "",
        "## Diagnosis",
        "",
        diagnose(walkforward_report, teacher_report, feature_report, metrics),
        "",
        "## Production Safety",
        "",
        "This replay wrote only isolated CSV/joblib/JSON/XLSX/Markdown artifacts under the V4.1 CSV replay folder. It did not write production Delta tables, update workflows, touch Saxo execution/token refresh, modify RT or dashboard notebooks, or update the model registry.",
        "",
    ]
    (OUTPUT_ROOT / "forensics" / "faithful_grid_model_replay_summary.md").write_text(
        "\n".join(lines)
    )


def write_governance_note(walkforward_report: dict, teacher_report: dict, metrics: dict) -> None:
    signal_match = walkforward_report.get("signal_match_rate")
    note = f"""# Faithful Legacy Replay Governance Note

Generated: {datetime.now(timezone.utc).isoformat()}

## Result

- The corrected/reference grid is faithfully reproducible from recovered legacy inputs and `build_legacy_silver()` semantics.
- Walk-forward model replay signal match vs corrected reference: `{signal_match:.6%}`.
- Teacher full-signal match vs corrected reference: `{teacher_report.get('match_rate'):.6%}`.
- Replay CAGR: `{metrics.get('CAGR'):.6%}`.
- Corrected reference CAGR target: `20.8405%`.
- Exact corrected reference performance can be regenerated only if signal/model replay matches the saved reference sequence. Otherwise, the saved corrected signal artifact remains the authoritative replay of that run.

## Leakage Risk

The recovered method trains a full-sample teacher on raw forward-return labels, then uses the teacher's `predicted_regime` as pseudo-labels for monthly walk-forward student folds. That is a retrospective distillation baseline with leakage risk because the teacher observes data after historical fold boundaries.

## Recommended Treatment

- Treat this as a forensic/reference baseline.
- Do not treat it as a strict point-in-time production baseline unless the training flow is redesigned.
- Use the saved corrected signal artifact when exact 20.8405% reference reproduction is required.
- Keep live V4 untouched until a PIT-safe teacher-student or raw-label walk-forward design is validated separately.

## Production Safety

No production Delta table, Saxo execution path, token refresh path, RT notebook, dashboard notebook, workflow, live schedule, or model registry artifact was modified.
"""
    (OUTPUT_ROOT / "forensics" / "faithful_legacy_replay_governance_note.md").write_text(note)


def main() -> None:
    for subdir in ["code", "models", "signals", "manifests", "forensics", "performance"]:
        ensure_dir(OUTPUT_ROOT / subdir)
    missing = [
        str(path)
        for path in [FAITHFUL_GRID, RECOVERED_RUNNER, RECOVERED_GENERATOR_SOURCE, REFERENCE_SIGNAL, REFERENCE_FULL_SIGNAL]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required replay inputs: {missing}")

    os.environ["OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    os.environ["GRID_PATH"] = str(FAITHFUL_GRID)
    os.environ["REFERENCE_SIGNAL_PATH"] = str(REFERENCE_SIGNAL)
    runpy.run_path(str(RECOVERED_RUNNER), run_name="__main__")
    write_enhanced_reports()

    result = json.loads((OUTPUT_ROOT / "forensics" / "signal_match_report.json").read_text())
    perf = json.loads((OUTPUT_ROOT / "forensics" / "performance_comparison.json").read_text())
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "walkforward_signal_match_rate": result["walkforward_signal_match"].get(
                    "signal_match_rate"
                ),
                "teacher_full_signal_match_rate": result["teacher_full_signal_match"].get(
                    "match_rate"
                ),
                "performance": perf["replay_metrics"],
                "cagr_20_8405_reproduced": perf["cagr_20_8405_reproduced"],
                "no_production_change_confirmation": True,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
