# Databricks notebook source
# Comparison summary for baseline versus corrected-grid retrain

# MAGIC %pip install xgboost==3.1.1

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
SILVER_RT_TABLE = f"{CATALOG}.{SCHEMA}.silver_rt_1h"
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"

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

REGIME_NAME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
REPAIR_WINDOW_START = pd.Timestamp("2026-03-10 00:00:00")
REPAIR_WINDOW_END = pd.Timestamp("2026-04-10 23:59:59")
RECENT_WALKFORWARD_ROWS = 240
RECENT_RT_ROWS = 72


def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def write_json(path_str, payload):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(json.dumps(payload, indent=2, default=str))


def write_text(path_str, content):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(content)


def load_json(path_str):
    return json.loads(Path(path_str).read_text())


def safe_read_csv(path_str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_predict(model, frame, prefix):
    result = frame.copy()
    X = result[FEATURES].fillna(0)
    result[f"{prefix}_predicted_regime"] = model.predict(X).astype(int)
    result[f"{prefix}_regime_name"] = result[f"{prefix}_predicted_regime"].map(
        REGIME_NAME_MAP
    )
    result[f"{prefix}_prob_expansion"] = np.nan
    result[f"{prefix}_prob_slowdown"] = np.nan
    result[f"{prefix}_prob_recession"] = np.nan
    result[f"{prefix}_max_probability"] = np.nan
    try:
        probs = model.predict_proba(X)
        if probs is not None:
            probs = np.asarray(probs)
            if probs.ndim == 2 and probs.shape[1] >= 1:
                if probs.shape[1] > 0:
                    result[f"{prefix}_prob_expansion"] = probs[:, 0]
                if probs.shape[1] > 1:
                    result[f"{prefix}_prob_slowdown"] = probs[:, 1]
                if probs.shape[1] > 2:
                    result[f"{prefix}_prob_recession"] = probs[:, 2]
                result[f"{prefix}_max_probability"] = probs.max(axis=1)
    except Exception:
        pass
    return result


def load_model(path_str):
    return joblib.load(path_str)


def resolve_existing_path(*candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    raise FileNotFoundError("No model artifact path available for comparison")


def normalize_timestamp(pdf, column="timestamp"):
    if column in pdf.columns:
        pdf[column] = pd.to_datetime(pdf[column], errors="coerce")
    return pdf


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
corrected_root = f"{comparison_root}/corrected_retrain"
comparison_dir = f"{comparison_root}/comparison"
ensure_dir(comparison_dir)

baseline_metadata = load_json(f"{baseline_root}/current_model_metadata.json")
corrected_metadata = load_json(f"{corrected_root}/model_metadata.json")

baseline_metrics = safe_read_csv(f"{baseline_root}/current_walkforward_metrics.csv")
corrected_metrics = safe_read_csv(f"{corrected_root}/monthly_walkforward_validation.csv")

for frame in [baseline_metrics, corrected_metrics]:
    if not frame.empty:
        normalize_timestamp(frame, "train_end")
        normalize_timestamp(frame, "test_end")

metrics_diff = baseline_metrics.merge(
    corrected_metrics,
    on=["train_end", "test_end"],
    how="outer",
    suffixes=("_baseline", "_corrected"),
)
if not metrics_diff.empty:
    metrics_diff["accuracy_diff"] = (
        metrics_diff["accuracy_corrected"] - metrics_diff["accuracy_baseline"]
    )
metrics_diff_path = f"{comparison_dir}/metrics_diff.csv"
metrics_diff.to_csv(metrics_diff_path, index=False)

baseline_wf = normalize_timestamp(
    safe_read_csv(f"{baseline_root}/current_walkforward_signals.csv")
)
corrected_wf = normalize_timestamp(
    safe_read_csv(f"{corrected_root}/walkforward_signals.csv")
)

wf_compare = baseline_wf.merge(
    corrected_wf,
    on="timestamp",
    how="outer",
    suffixes=("_baseline", "_corrected"),
)
if not wf_compare.empty:
    wf_compare["predicted_regime_changed"] = (
        wf_compare["predicted_regime_baseline"]
        != wf_compare["predicted_regime_corrected"]
    )
    recent_walkforward_compare = wf_compare.sort_values("timestamp").tail(
        RECENT_WALKFORWARD_ROWS
    )
else:
    recent_walkforward_compare = pd.DataFrame()

baseline_full = normalize_timestamp(
    safe_read_csv(f"{baseline_root}/current_full_signal_export.csv")
)
corrected_full = normalize_timestamp(
    safe_read_csv(f"{corrected_root}/full_signal_export.csv")
)

historical_full_compare = baseline_full.merge(
    corrected_full,
    on="timestamp",
    how="outer",
    suffixes=("_baseline", "_corrected"),
)
if not historical_full_compare.empty:
    historical_full_compare["predicted_regime_changed"] = (
        historical_full_compare["predicted_regime_baseline"]
        != historical_full_compare["predicted_regime_corrected"]
    )
    repair_window_compare = historical_full_compare[
        historical_full_compare["timestamp"].between(
            REPAIR_WINDOW_START, REPAIR_WINDOW_END
        )
    ].copy()
else:
    repair_window_compare = pd.DataFrame()

baseline_model_path = (
    resolve_existing_path(
        baseline_metadata.get("current_model_artifact_copy"),
        baseline_metadata.get("current_versioned_model_path"),
        baseline_metadata.get("current_model_path"),
    )
)
corrected_model_path = resolve_existing_path(corrected_metadata["model_artifact_path"])

baseline_model = load_model(baseline_model_path)
corrected_model = load_model(corrected_model_path)

repair_window_pdf = (
    spark.sql(
        f"""
        SELECT timestamp, regime, {", ".join(FEATURES)}
        FROM {SILVER_TABLE}
        WHERE timestamp BETWEEN timestamp('{REPAIR_WINDOW_START.strftime("%Y-%m-%d %H:%M:%S")}')
          AND timestamp('{REPAIR_WINDOW_END.strftime("%Y-%m-%d %H:%M:%S")}')
        ORDER BY timestamp
        """
    )
    .toPandas()
)
repair_window_pdf = normalize_timestamp(repair_window_pdf)

recent_rt_pdf = (
    spark.sql(
        f"""
        SELECT timestamp, {", ".join(FEATURES)}
        FROM {SILVER_RT_TABLE}
        ORDER BY timestamp DESC
        LIMIT {RECENT_RT_ROWS}
        """
    )
    .toPandas()
    .sort_values("timestamp")
)
recent_rt_pdf = normalize_timestamp(recent_rt_pdf)

repair_baseline = safe_predict(baseline_model, repair_window_pdf.copy(), "baseline")
repair_corrected = safe_predict(corrected_model, repair_window_pdf.copy(), "corrected")
repair_model_compare = repair_baseline.merge(
    repair_corrected[
        [
            "timestamp",
            "corrected_predicted_regime",
            "corrected_regime_name",
            "corrected_prob_expansion",
            "corrected_prob_slowdown",
            "corrected_prob_recession",
            "corrected_max_probability",
        ]
    ],
    on="timestamp",
    how="inner",
)
if not repair_model_compare.empty:
    repair_model_compare["scope"] = "repair_window_bt"
    repair_model_compare["regime_changed"] = (
        repair_model_compare["baseline_predicted_regime"]
        != repair_model_compare["corrected_predicted_regime"]
    )
    repair_model_compare["max_probability_diff"] = (
        repair_model_compare["corrected_max_probability"]
        - repair_model_compare["baseline_max_probability"]
    )

recent_rt_baseline = safe_predict(baseline_model, recent_rt_pdf.copy(), "baseline")
recent_rt_corrected = safe_predict(corrected_model, recent_rt_pdf.copy(), "corrected")
recent_rt_compare = recent_rt_baseline.merge(
    recent_rt_corrected[
        [
            "timestamp",
            "corrected_predicted_regime",
            "corrected_regime_name",
            "corrected_prob_expansion",
            "corrected_prob_slowdown",
            "corrected_prob_recession",
            "corrected_max_probability",
        ]
    ],
    on="timestamp",
    how="inner",
)
if not recent_rt_compare.empty:
    recent_rt_compare["scope"] = "recent_rt"
    recent_rt_compare["regime_changed"] = (
        recent_rt_compare["baseline_predicted_regime"]
        != recent_rt_compare["corrected_predicted_regime"]
    )
    recent_rt_compare["max_probability_diff"] = (
        recent_rt_compare["corrected_max_probability"]
        - recent_rt_compare["baseline_max_probability"]
    )

walkforward_signal_sample = recent_walkforward_compare.copy()
if not walkforward_signal_sample.empty:
    walkforward_signal_sample["scope"] = "walkforward_recent"
    walkforward_signal_sample = walkforward_signal_sample[
        [
            "scope",
            "timestamp",
            "regime_baseline",
            "predicted_regime_baseline",
            "predicted_regime_corrected",
            "predicted_regime_changed",
        ]
    ]

signal_diff_frames = []
if not walkforward_signal_sample.empty:
    signal_diff_frames.append(walkforward_signal_sample)

if not repair_model_compare.empty:
    signal_diff_frames.append(
        repair_model_compare[
            [
                "scope",
                "timestamp",
                "regime",
                "baseline_predicted_regime",
                "baseline_regime_name",
                "baseline_prob_expansion",
                "baseline_prob_slowdown",
                "baseline_prob_recession",
                "baseline_max_probability",
                "corrected_predicted_regime",
                "corrected_regime_name",
                "corrected_prob_expansion",
                "corrected_prob_slowdown",
                "corrected_prob_recession",
                "corrected_max_probability",
                "regime_changed",
                "max_probability_diff",
            ]
        ]
    )

if not recent_rt_compare.empty:
    signal_diff_frames.append(
        recent_rt_compare[
            [
                "scope",
                "timestamp",
                "baseline_predicted_regime",
                "baseline_regime_name",
                "baseline_prob_expansion",
                "baseline_prob_slowdown",
                "baseline_prob_recession",
                "baseline_max_probability",
                "corrected_predicted_regime",
                "corrected_regime_name",
                "corrected_prob_expansion",
                "corrected_prob_slowdown",
                "corrected_prob_recession",
                "corrected_max_probability",
                "regime_changed",
                "max_probability_diff",
            ]
        ]
    )

signal_diff_sample = (
    pd.concat(signal_diff_frames, ignore_index=True, sort=False)
    if signal_diff_frames
    else pd.DataFrame()
)

if not signal_diff_sample.empty and "regime_changed" in signal_diff_sample.columns:
    signal_diff_sample = signal_diff_sample[
        signal_diff_sample["regime_changed"].fillna(False)
        | signal_diff_sample["max_probability_diff"].abs().fillna(0).gt(0.01)
    ].copy()

if signal_diff_sample.empty and not recent_rt_compare.empty:
    signal_diff_sample = recent_rt_compare.tail(10).copy()

signal_diff_sample_path = f"{comparison_dir}/signal_diff_sample.csv"
signal_diff_sample.to_csv(signal_diff_sample_path, index=False)

mean_accuracy_diff = (
    float(metrics_diff["accuracy_diff"].mean())
    if not metrics_diff.empty and "accuracy_diff" in metrics_diff.columns
    else 0.0
)
walkforward_diff_count = (
    int(recent_walkforward_compare["predicted_regime_changed"].sum())
    if not recent_walkforward_compare.empty
    else 0
)
repair_window_diff_count = (
    int(repair_model_compare["regime_changed"].sum())
    if not repair_model_compare.empty
    else 0
)
recent_rt_diff_count = (
    int(recent_rt_compare["regime_changed"].sum())
    if not recent_rt_compare.empty
    else 0
)

latest_rt_baseline_regime = None
latest_rt_corrected_regime = None
if not recent_rt_compare.empty:
    latest_rt_row = recent_rt_compare.sort_values("timestamp").iloc[-1]
    latest_rt_baseline_regime = latest_rt_row["baseline_regime_name"]
    latest_rt_corrected_regime = latest_rt_row["corrected_regime_name"]

if not corrected_metadata.get("registry_unchanged", False) or not corrected_metadata.get(
    "live_alias_unchanged", False
):
    recommendation = "defer"
    recommendation_reason = "Safety assertions failed during non-promoting retrain."
elif mean_accuracy_diff > 0 and (
    repair_window_diff_count > 0 or recent_rt_diff_count > 0 or walkforward_diff_count > 0
):
    recommendation = "promote after approval"
    recommendation_reason = (
        "Corrected retrain improves walk-forward accuracy and changes model outputs in the inspected windows."
    )
elif abs(mean_accuracy_diff) < 1e-12 and repair_window_diff_count == 0 and recent_rt_diff_count == 0:
    recommendation = "no promotion needed"
    recommendation_reason = (
        "Corrected retrain does not materially change measured outcomes versus the frozen baseline."
    )
else:
    recommendation = "defer"
    recommendation_reason = (
        "Comparison results are mixed or need human review before any promotion decision."
    )

summary_lines = [
    "# Baseline vs Corrected Comparison",
    "",
    "## Model Metadata",
    "",
    f"- Baseline model version: `{baseline_metadata.get('current_model_version', 'unknown')}`",
    f"- Baseline model path: `{baseline_metadata.get('current_model_path', 'unknown')}`",
    f"- Corrected model version: `{corrected_metadata.get('model_version', 'unknown')}`",
    f"- Corrected model artifact: `{corrected_metadata.get('model_artifact_path', 'unknown')}`",
    f"- Corrected train end: `{corrected_metadata.get('train_end', 'unknown')}`",
    "",
    "## Safety",
    "",
    f"- Production registry unchanged: `{corrected_metadata.get('registry_unchanged')}`",
    f"- Live alias unchanged: `{corrected_metadata.get('live_alias_unchanged')}`",
    "",
    "## Metrics",
    "",
    f"- Baseline mean walk-forward accuracy: `{baseline_metrics['accuracy'].mean() if not baseline_metrics.empty else 'N/A'}`",
    f"- Corrected mean walk-forward accuracy: `{corrected_metrics['accuracy'].mean() if not corrected_metrics.empty else 'N/A'}`",
    f"- Mean accuracy diff: `{mean_accuracy_diff}`",
    "",
    "## Signal Deltas",
    "",
    f"- Recent walk-forward differing timestamps: `{walkforward_diff_count}`",
    f"- Corrected-grid repaired-window differing timestamps: `{repair_window_diff_count}`",
    f"- Recent RT differing timestamps: `{recent_rt_diff_count}`",
    f"- Latest RT baseline regime: `{latest_rt_baseline_regime or 'N/A'}`",
    f"- Latest RT corrected regime: `{latest_rt_corrected_regime or 'N/A'}`",
    "",
    "## Recommendation",
    "",
    f"- Recommendation: `{recommendation}`",
    f"- Reason: {recommendation_reason}",
]
summary_path = f"{comparison_dir}/summary.md"
write_text(summary_path, "\n".join(summary_lines) + "\n")

recommendation_payload = {
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "comparison_root": comparison_root,
    "baseline_model_version": baseline_metadata.get("current_model_version"),
    "corrected_model_version": corrected_metadata.get("model_version"),
    "baseline_model_path": baseline_model_path,
    "corrected_model_path": corrected_model_path,
    "mean_accuracy_diff": mean_accuracy_diff,
    "walkforward_diff_count": walkforward_diff_count,
    "repair_window_diff_count": repair_window_diff_count,
    "recent_rt_diff_count": recent_rt_diff_count,
    "latest_rt_baseline_regime": latest_rt_baseline_regime,
    "latest_rt_corrected_regime": latest_rt_corrected_regime,
    "recommendation": recommendation,
    "reason": recommendation_reason,
}
recommendation_path = f"{comparison_dir}/recommendation.json"
write_json(recommendation_path, recommendation_payload)

result = {
    "phase": "comparison",
    "status": "success",
    "comparison_root": comparison_root,
    "comparison_dir": comparison_dir,
    "artifacts": {
        "summary": summary_path,
        "metrics_diff": metrics_diff_path,
        "signal_diff_sample": signal_diff_sample_path,
        "recommendation": recommendation_path,
    },
    "mean_accuracy_diff": mean_accuracy_diff,
    "walkforward_diff_count": walkforward_diff_count,
    "repair_window_diff_count": repair_window_diff_count,
    "recent_rt_diff_count": recent_rt_diff_count,
    "latest_rt_baseline_regime": latest_rt_baseline_regime,
    "latest_rt_corrected_regime": latest_rt_corrected_regime,
    "recommendation": recommendation,
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))
