# Databricks notebook source
# Baseline snapshot for before/after corrected-grid comparison

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"
WF_SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"
SILVER_RT_TABLE = f"{CATALOG}.{SCHEMA}.silver_rt_1h"
RT_SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
PERFORMANCE_ROOT = f"{VOLUME_ROOT}/performance"
FULL_SIGNAL_PATH = f"{VOLUME_ROOT}/bt_fred_1d_regime_signals-v4.csv"
WF_SIGNAL_PATH = f"{VOLUME_ROOT}/bt_walkforward_regime_signals.csv"
LIVE_MODEL_PATH = f"{VOLUME_ROOT}/models/bt_fred_1d_xgboost_regime_live.joblib"
CURRENT_WF_METRICS_PATH = f"{PERFORMANCE_ROOT}/summary/monthly_walkforward_validation.csv"

RT_SAMPLE_ROWS = 72


def serialize(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def to_records(pdf):
    if pdf.empty:
        return []
    normalized = pdf.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(serialize)
    return normalized.to_dict(orient="records")


def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def copy_if_exists(src, dst):
    src_path = Path(src)
    if not src_path.exists():
        return False
    dst_path = Path(dst)
    ensure_dir(str(dst_path.parent))
    shutil.copy2(src_path, dst_path)
    return True


def write_json(path_str, payload):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(json.dumps(payload, indent=2, default=str))


def write_markdown(path_str, content):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(content)


def list_files_recursive(root_path):
    root = Path(root_path)
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return files


def copy_tree_files(src_root, dst_root):
    copied = []
    src = Path(src_root)
    if not src.exists():
        return copied
    for file_info in list_files_recursive(src_root):
        src_path = src / file_info["relative_path"]
        dst_path = Path(dst_root) / file_info["relative_path"]
        ensure_dir(str(dst_path.parent))
        shutil.copy2(src_path, dst_path)
        copied.append(str(dst_path))
    return copied


def latest_registry_row(registry_pdf):
    if registry_pdf.empty:
        return {}
    pdf = registry_pdf.copy()
    if "promoted_ts" in pdf.columns:
        pdf["promoted_ts"] = pd.to_datetime(pdf["promoted_ts"], errors="coerce")
        pdf = pdf.sort_values("promoted_ts", ascending=False, na_position="last")
    row = pdf.iloc[0].to_dict()
    return {key: serialize(value) for key, value in row.items()}


comparison_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
comparison_root_default = (
    f"{VOLUME_ROOT}/comparison/{comparison_date}_baseline_vs_corrected"
)

if "dbutils" in globals():
    dbutils.widgets.text("comparison_date", comparison_date)
    comparison_date = dbutils.widgets.get("comparison_date").strip() or comparison_date
    comparison_root_default = (
        f"{VOLUME_ROOT}/comparison/{comparison_date}_baseline_vs_corrected"
    )
    dbutils.widgets.text("comparison_root", comparison_root_default)
    comparison_root = dbutils.widgets.get("comparison_root").strip() or comparison_root_default
else:
    comparison_root = comparison_root_default

baseline_root = f"{comparison_root}/baseline"
performance_copy_root = f"{baseline_root}/performance_exports"
ensure_dir(baseline_root)
ensure_dir(performance_copy_root)

capture_ts = datetime.now(timezone.utc).isoformat()

registry_pdf = spark.table(MODEL_REGISTRY_TABLE).toPandas()
if "promoted_ts" in registry_pdf.columns:
    registry_pdf["promoted_ts"] = pd.to_datetime(
        registry_pdf["promoted_ts"], errors="coerce"
    )
registry_snapshot_path = f"{baseline_root}/model_registry_current_snapshot.csv"
registry_pdf.to_csv(registry_snapshot_path, index=False)

current_model = latest_registry_row(registry_pdf)
artifact_source = (
    current_model.get("versioned_model_path")
    or current_model.get("model_path")
    or LIVE_MODEL_PATH
)
artifact_copy_path = None
if artifact_source and Path(artifact_source).exists():
    artifact_copy_path = (
        f"{baseline_root}/model_artifacts/{Path(artifact_source).name}"
    )
    copy_if_exists(artifact_source, artifact_copy_path)

full_signal_copy_path = f"{baseline_root}/current_full_signal_export.csv"
copy_if_exists(FULL_SIGNAL_PATH, full_signal_copy_path)

wf_signals_pdf = (
    spark.table(WF_SIGNAL_TABLE)
    .toPandas()
    .sort_values("timestamp")
)
wf_signal_export_path = f"{baseline_root}/current_walkforward_signals.csv"
wf_signals_pdf.to_csv(wf_signal_export_path, index=False)

wf_metrics_export_path = f"{baseline_root}/current_walkforward_metrics.csv"
copy_if_exists(CURRENT_WF_METRICS_PATH, wf_metrics_export_path)

copied_performance_files = copy_tree_files(PERFORMANCE_ROOT, performance_copy_root)
performance_manifest = {
    "captured_at_utc": capture_ts,
    "source_root": PERFORMANCE_ROOT,
    "destination_root": performance_copy_root,
    "copied_file_count": len(copied_performance_files),
    "files": list_files_recursive(PERFORMANCE_ROOT),
}
performance_manifest_path = f"{baseline_root}/current_performance_manifest.json"
write_json(performance_manifest_path, performance_manifest)

recent_silver_pdf = (
    spark.sql(
        f"""
        SELECT *
        FROM {SILVER_RT_TABLE}
        ORDER BY timestamp DESC
        LIMIT {RT_SAMPLE_ROWS}
        """
    )
    .toPandas()
    .sort_values("timestamp")
)

recent_rt_signal_pdf = (
    spark.sql(
        f"""
        SELECT *
        FROM {RT_SIGNAL_TABLE}
        ORDER BY timestamp DESC
        LIMIT {RT_SAMPLE_ROWS}
        """
    )
    .toPandas()
    .sort_values("timestamp")
)

if not recent_silver_pdf.empty:
    recent_silver_pdf["timestamp"] = pd.to_datetime(
        recent_silver_pdf["timestamp"], errors="coerce"
    )
if not recent_rt_signal_pdf.empty:
    recent_rt_signal_pdf["timestamp"] = pd.to_datetime(
        recent_rt_signal_pdf["timestamp"], errors="coerce"
    )

recent_signal_sample = recent_silver_pdf.merge(
    recent_rt_signal_pdf,
    on="timestamp",
    how="outer",
    suffixes=("_silver_rt", "_rt_signal"),
).sort_values("timestamp")
recent_signal_sample_path = f"{baseline_root}/current_recent_signal_sample.csv"
recent_signal_sample.to_csv(recent_signal_sample_path, index=False)

latest_rt_signal_summary = {}
if not recent_rt_signal_pdf.empty:
    latest_row = recent_rt_signal_pdf.sort_values("timestamp").iloc[-1].to_dict()
    latest_rt_signal_summary = {
        key: serialize(value) for key, value in latest_row.items()
    }

current_model_metadata = {
    "captured_at_utc": capture_ts,
    "comparison_root": comparison_root,
    "baseline_root": baseline_root,
    "model_registry_table": MODEL_REGISTRY_TABLE,
    "wf_signal_table": WF_SIGNAL_TABLE,
    "current_model_version": current_model.get("model_version"),
    "current_model_path": current_model.get("model_path"),
    "current_versioned_model_path": current_model.get("versioned_model_path"),
    "current_signal_path": current_model.get("signal_path"),
    "current_wf_signal_table": current_model.get("wf_signal_table"),
    "current_promoted_ts": current_model.get("promoted_ts"),
    "current_train_start": current_model.get("train_start"),
    "current_train_end": current_model.get("train_end"),
    "current_model_artifact_source": artifact_source,
    "current_model_artifact_copy": artifact_copy_path,
    "full_signal_copy_path": full_signal_copy_path if Path(full_signal_copy_path).exists() else None,
    "wf_signal_export_path": wf_signal_export_path,
    "wf_signal_volume_path": WF_SIGNAL_PATH if Path(WF_SIGNAL_PATH).exists() else None,
    "wf_metrics_export_path": wf_metrics_export_path if Path(wf_metrics_export_path).exists() else None,
    "live_model_alias_path": LIVE_MODEL_PATH,
    "latest_rt_signal_summary": latest_rt_signal_summary,
    "performance_manifest_path": performance_manifest_path,
    "performance_copy_root": performance_copy_root,
}
current_model_metadata_path = f"{baseline_root}/current_model_metadata.json"
write_json(current_model_metadata_path, current_model_metadata)

readme_lines = [
    "# Baseline Snapshot",
    "",
    f"- Captured at UTC: `{capture_ts}`",
    f"- Comparison root: `{comparison_root}`",
    f"- Current model version: `{current_model.get('model_version', 'unknown')}`",
    f"- Current model path: `{current_model.get('model_path', LIVE_MODEL_PATH)}`",
    f"- Current versioned model path: `{current_model.get('versioned_model_path', 'N/A')}`",
    f"- Copied model artifact: `{artifact_copy_path or 'not copied'}`",
    f"- Walk-forward signal export: `{wf_signal_export_path}`",
    f"- Walk-forward metrics export: `{wf_metrics_export_path if Path(wf_metrics_export_path).exists() else 'missing'}`",
    f"- Copied performance file count: `{len(copied_performance_files)}`",
]

if latest_rt_signal_summary:
    readme_lines.extend(
        [
            "",
            "## Latest RT Signal",
            "",
            f"- Timestamp: `{latest_rt_signal_summary.get('timestamp', 'N/A')}`",
            f"- Regime: `{latest_rt_signal_summary.get('regime_name', 'N/A')}`",
            f"- Position: `{latest_rt_signal_summary.get('position', 'N/A')}`",
            f"- Model version: `{latest_rt_signal_summary.get('model_version', 'N/A')}`",
            f"- Model path: `{latest_rt_signal_summary.get('model_path', 'N/A')}`",
        ]
    )

write_markdown(f"{baseline_root}/README.md", "\n".join(readme_lines) + "\n")

result = {
    "phase": "baseline",
    "status": "success",
    "comparison_root": comparison_root,
    "baseline_root": baseline_root,
    "captured_at_utc": capture_ts,
    "current_model_version": current_model.get("model_version"),
    "current_model_path": current_model.get("model_path"),
    "current_versioned_model_path": current_model.get("versioned_model_path"),
    "artifacts": {
        "model_registry_current_snapshot": registry_snapshot_path,
        "current_model_metadata": current_model_metadata_path,
        "current_walkforward_signals": wf_signal_export_path,
        "current_walkforward_metrics": wf_metrics_export_path if Path(wf_metrics_export_path).exists() else None,
        "current_full_signal_export": full_signal_copy_path if Path(full_signal_copy_path).exists() else None,
        "current_recent_signal_sample": recent_signal_sample_path,
        "current_performance_manifest": performance_manifest_path,
        "readme": f"{baseline_root}/README.md",
    },
    "performance_file_count": len(copied_performance_files),
    "latest_rt_signal_summary": latest_rt_signal_summary,
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))
