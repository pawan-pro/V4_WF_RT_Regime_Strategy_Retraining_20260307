# Databricks notebook source
# Decision pack for current production baseline vs full historical ALFRED-consistent rebuild.

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb  # noqa: F401

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
COMPARISON_ROOT = f"{VOLUME_ROOT}/comparison"

COMPARISON_DATE_DEFAULT = datetime.now(timezone.utc).strftime("%Y-%m-%d")
COMPARISON_ROOT_DEFAULT = (
    f"{COMPARISON_ROOT}/{COMPARISON_DATE_DEFAULT}_full_historical_adoption"
)
BASELINE_ROOT_DEFAULT = f"{COMPARISON_ROOT_DEFAULT}/baseline"
FULL_REBUILD_ROOT_DEFAULT = f"{COMPARISON_ROOT_DEFAULT}/full_rebuild_retrain"
PML_ROOT_DEFAULT = f"{COMPARISON_ROOT_DEFAULT}/walkforward_pnl"
OUTPUT_ROOT_DEFAULT = f"{COMPARISON_ROOT_DEFAULT}/comparison"

BASELINE_SILVER_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
FULL_REBUILD_SILVER_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_rebuild"
START_DATE_DEFAULT = "2021-01-01"

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


def normalize_timestamp(pdf, column="timestamp"):
    if column in pdf.columns:
        pdf[column] = pd.to_datetime(pdf[column], errors="coerce")
        try:
            pdf[column] = pdf[column].dt.tz_localize(None)
        except Exception:
            pass
    return pdf


def resolve_existing_path(*candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    raise FileNotFoundError("No model artifact path available for comparison")


def load_model(path_str):
    return joblib.load(path_str)


def load_feature_frame(table_name, start_date):
    sql = f"""
        SELECT timestamp, {", ".join(FEATURES)}
        FROM {table_name}
        WHERE timestamp >= TIMESTAMP('{start_date}')
        ORDER BY timestamp
    """
    feature_pdf = spark.sql(sql).toPandas()
    return normalize_timestamp(feature_pdf)


def score_model(model, feature_pdf, prefix):
    scored = feature_pdf.copy()
    scored = normalize_timestamp(scored)
    X = scored[FEATURES].fillna(0)
    scored[f"{prefix}_predicted_regime"] = model.predict(X).astype(int)
    scored[f"{prefix}_regime_name"] = scored[f"{prefix}_predicted_regime"].map(
        REGIME_NAME_MAP
    )
    probs = model.predict_proba(X)
    probs = np.asarray(probs)
    scored[f"{prefix}_prob_expansion"] = probs[:, 0]
    scored[f"{prefix}_prob_slowdown"] = probs[:, 1]
    scored[f"{prefix}_prob_recession"] = probs[:, 2]
    scored[f"{prefix}_max_probability"] = probs.max(axis=1)
    return scored


def load_signal_frame(path_str, prefix):
    frame = safe_read_csv(path_str)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", f"{prefix}_predicted_regime"])
    frame = normalize_timestamp(frame)
    frame = frame.sort_values("timestamp")
    frame = frame[frame["timestamp"].notna()].copy()
    if "predicted_regime" in frame.columns:
        source_column = "predicted_regime"
    elif "model_predicted_regime" in frame.columns:
        source_column = "model_predicted_regime"
    else:
        raise KeyError(
            f"No predicted regime column found in {path_str}. columns={list(frame.columns)}"
        )
    result = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            f"{prefix}_predicted_regime": pd.to_numeric(
                frame[source_column], errors="coerce"
            ).fillna(2).astype(int),
        }
    )
    if "regime" in frame.columns:
        result[f"{prefix}_actual_regime"] = pd.to_numeric(
            frame["regime"], errors="coerce"
        )
    probability_columns = [
        ("model_prob_expansion", f"{prefix}_prob_expansion"),
        ("model_prob_slowdown", f"{prefix}_prob_slowdown"),
        ("model_prob_recession", f"{prefix}_prob_recession"),
        ("model_max_probability", f"{prefix}_max_probability"),
    ]
    for source_column_name, target_column_name in probability_columns:
        if source_column_name in frame.columns:
            result[target_column_name] = pd.to_numeric(
                frame[source_column_name], errors="coerce"
            )
    return result.drop_duplicates(subset=["timestamp"], keep="last")


def metric_row(metric_name, baseline_value, rebuild_value, category):
    diff_value = np.nan
    if pd.notna(baseline_value) and pd.notna(rebuild_value):
        diff_value = rebuild_value - baseline_value
    return {
        "category": category,
        "metric_name": metric_name,
        "baseline_value": baseline_value,
        "rebuild_value": rebuild_value,
        "diff_rebuild_minus_baseline": diff_value,
    }


comparison_root = COMPARISON_ROOT_DEFAULT
baseline_root = BASELINE_ROOT_DEFAULT
full_rebuild_root = FULL_REBUILD_ROOT_DEFAULT
pnl_root = PML_ROOT_DEFAULT
output_root = OUTPUT_ROOT_DEFAULT
baseline_silver_table = BASELINE_SILVER_TABLE_DEFAULT
full_rebuild_silver_table = FULL_REBUILD_SILVER_TABLE_DEFAULT
start_date = START_DATE_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("comparison_root", COMPARISON_ROOT_DEFAULT)
    dbutils.widgets.text("baseline_root", BASELINE_ROOT_DEFAULT)
    dbutils.widgets.text("full_rebuild_root", FULL_REBUILD_ROOT_DEFAULT)
    dbutils.widgets.text("pnl_root", PML_ROOT_DEFAULT)
    dbutils.widgets.text("output_root", OUTPUT_ROOT_DEFAULT)
    dbutils.widgets.text("baseline_silver_table", BASELINE_SILVER_TABLE_DEFAULT)
    dbutils.widgets.text("full_rebuild_silver_table", FULL_REBUILD_SILVER_TABLE_DEFAULT)
    dbutils.widgets.text("start_date", START_DATE_DEFAULT)

    comparison_root = dbutils.widgets.get("comparison_root").strip() or COMPARISON_ROOT_DEFAULT
    baseline_root = dbutils.widgets.get("baseline_root").strip() or BASELINE_ROOT_DEFAULT
    full_rebuild_root = dbutils.widgets.get("full_rebuild_root").strip() or FULL_REBUILD_ROOT_DEFAULT
    pnl_root = dbutils.widgets.get("pnl_root").strip() or PML_ROOT_DEFAULT
    output_root = dbutils.widgets.get("output_root").strip() or OUTPUT_ROOT_DEFAULT
    baseline_silver_table = (
        dbutils.widgets.get("baseline_silver_table").strip()
        or BASELINE_SILVER_TABLE_DEFAULT
    )
    full_rebuild_silver_table = (
        dbutils.widgets.get("full_rebuild_silver_table").strip()
        or FULL_REBUILD_SILVER_TABLE_DEFAULT
    )
    start_date = dbutils.widgets.get("start_date").strip() or START_DATE_DEFAULT

ensure_dir(output_root)

baseline_metadata = load_json(f"{baseline_root}/current_model_metadata.json")
full_rebuild_metadata = load_json(f"{full_rebuild_root}/model_metadata.json")

baseline_metrics = safe_read_csv(f"{baseline_root}/current_walkforward_metrics.csv")
full_rebuild_metrics = safe_read_csv(
    f"{full_rebuild_root}/monthly_walkforward_validation.csv"
)
pnl_metrics = safe_read_csv(f"{pnl_root}/pnl_metrics.csv")

if not baseline_metrics.empty:
    baseline_metrics = normalize_timestamp(baseline_metrics, "train_end")
    baseline_metrics = normalize_timestamp(baseline_metrics, "test_end")
if not full_rebuild_metrics.empty:
    full_rebuild_metrics = normalize_timestamp(full_rebuild_metrics, "train_end")
    full_rebuild_metrics = normalize_timestamp(full_rebuild_metrics, "test_end")

baseline_wf = load_signal_frame(
    f"{baseline_root}/current_walkforward_signals.csv",
    "baseline",
)
full_rebuild_wf = load_signal_frame(
    f"{full_rebuild_root}/walkforward_signals.csv",
    "full_rebuild",
)

walkforward_signal_compare = baseline_wf.merge(
    full_rebuild_wf,
    on="timestamp",
    how="inner",
)
walkforward_signal_compare = walkforward_signal_compare[
    walkforward_signal_compare["timestamp"] >= pd.Timestamp(start_date)
].copy()
walkforward_signal_compare["baseline_regime_name"] = walkforward_signal_compare[
    "baseline_predicted_regime"
].map(REGIME_NAME_MAP)
walkforward_signal_compare["full_rebuild_regime_name"] = walkforward_signal_compare[
    "full_rebuild_predicted_regime"
].map(REGIME_NAME_MAP)
walkforward_signal_compare["regime_changed"] = (
    walkforward_signal_compare["baseline_predicted_regime"]
    != walkforward_signal_compare["full_rebuild_predicted_regime"]
)
signal_diff = walkforward_signal_compare[
    walkforward_signal_compare["regime_changed"]
].copy()
signal_diff_path = f"{output_root}/signal_diff.csv"
signal_diff.to_csv(signal_diff_path, index=False)

baseline_model_path = resolve_existing_path(
    baseline_metadata.get("current_model_artifact_copy"),
    baseline_metadata.get("current_versioned_model_path"),
    baseline_metadata.get("current_model_path"),
)
full_rebuild_model_path = resolve_existing_path(
    full_rebuild_metadata.get("model_artifact_path")
)

baseline_model = load_model(baseline_model_path)
full_rebuild_model = load_model(full_rebuild_model_path)

baseline_features = load_feature_frame(baseline_silver_table, start_date)
full_rebuild_features = load_feature_frame(full_rebuild_silver_table, start_date)

baseline_scored = score_model(baseline_model, baseline_features, "baseline")
full_rebuild_scored = score_model(
    full_rebuild_model,
    full_rebuild_features,
    "full_rebuild",
)

probability_compare = baseline_scored[
    [
        "timestamp",
        "baseline_predicted_regime",
        "baseline_regime_name",
        "baseline_prob_expansion",
        "baseline_prob_slowdown",
        "baseline_prob_recession",
        "baseline_max_probability",
    ]
].merge(
    full_rebuild_scored[
        [
            "timestamp",
            "full_rebuild_predicted_regime",
            "full_rebuild_regime_name",
            "full_rebuild_prob_expansion",
            "full_rebuild_prob_slowdown",
            "full_rebuild_prob_recession",
            "full_rebuild_max_probability",
        ]
    ],
    on="timestamp",
    how="inner",
)

for class_name in ["expansion", "slowdown", "recession"]:
    probability_compare[f"{class_name}_probability_diff"] = (
        probability_compare[f"full_rebuild_prob_{class_name}"]
        - probability_compare[f"baseline_prob_{class_name}"]
    )
probability_compare["max_probability_diff"] = (
    probability_compare["full_rebuild_max_probability"]
    - probability_compare["baseline_max_probability"]
)
probability_compare["regime_changed"] = (
    probability_compare["baseline_predicted_regime"]
    != probability_compare["full_rebuild_predicted_regime"]
)
probability_compare["material_probability_change"] = (
    probability_compare[
        [
            "expansion_probability_diff",
            "slowdown_probability_diff",
            "recession_probability_diff",
            "max_probability_diff",
        ]
    ]
    .abs()
    .max(axis=1)
    .fillna(0.0)
    >= 0.01
)
probability_diff = probability_compare[
    probability_compare["regime_changed"] | probability_compare["material_probability_change"]
].copy()
probability_diff_path = f"{output_root}/probability_diff.csv"
probability_diff.to_csv(probability_diff_path, index=False)

baseline_mean_accuracy = (
    float(baseline_metrics["accuracy"].mean()) if not baseline_metrics.empty else np.nan
)
baseline_median_accuracy = (
    float(baseline_metrics["accuracy"].median()) if not baseline_metrics.empty else np.nan
)
full_rebuild_mean_accuracy = (
    float(full_rebuild_metrics["accuracy"].mean())
    if not full_rebuild_metrics.empty
    else np.nan
)
full_rebuild_median_accuracy = (
    float(full_rebuild_metrics["accuracy"].median())
    if not full_rebuild_metrics.empty
    else np.nan
)

pnl_common = pnl_metrics[pnl_metrics["scope"] == "common_overlap"].copy()
baseline_pnl = (
    pnl_common.loc[pnl_common["model"] == "baseline"].iloc[0].to_dict()
    if not pnl_common.empty
    else {}
)
full_rebuild_pnl = (
    pnl_common.loc[pnl_common["model"] == "full_rebuild"].iloc[0].to_dict()
    if not pnl_common.empty
    else {}
)

metrics_rows = [
    metric_row("mean_walkforward_accuracy", baseline_mean_accuracy, full_rebuild_mean_accuracy, "walkforward_validation"),
    metric_row("median_walkforward_accuracy", baseline_median_accuracy, full_rebuild_median_accuracy, "walkforward_validation"),
]

for metric_name in [
    "final_equity",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "total_regime_switches",
    "average_holding_bars",
    "expansion_pct_time",
    "slowdown_pct_time",
    "recession_pct_time",
]:
    metrics_rows.append(
        metric_row(
            metric_name,
            baseline_pnl.get(metric_name),
            full_rebuild_pnl.get(metric_name),
            "walkforward_pnl",
        )
    )

metrics_rows.append(
    metric_row(
        "walkforward_signal_change_count",
        0,
        int(signal_diff.shape[0]),
        "signal_differences",
    )
)
metrics_diff = pd.DataFrame(metrics_rows)
metrics_diff_path = f"{output_root}/metrics_diff.csv"
metrics_diff.to_csv(metrics_diff_path, index=False)

latest_walkforward_row = (
    walkforward_signal_compare.sort_values("timestamp").iloc[-1].to_dict()
    if not walkforward_signal_compare.empty
    else {}
)
latest_probability_row = (
    probability_compare.sort_values("timestamp").iloc[-1].to_dict()
    if not probability_compare.empty
    else {}
)

rebuild_safe = bool(full_rebuild_metadata.get("registry_unchanged")) and bool(
    full_rebuild_metadata.get("live_alias_unchanged")
)

rebuild_beats_baseline = (
    pd.notna(full_rebuild_pnl.get("final_equity"))
    and pd.notna(baseline_pnl.get("final_equity"))
    and full_rebuild_pnl.get("final_equity") > baseline_pnl.get("final_equity")
    and full_rebuild_pnl.get("cagr") > baseline_pnl.get("cagr")
    and full_rebuild_pnl.get("sharpe_ratio") >= baseline_pnl.get("sharpe_ratio")
)
rebuild_loses_to_baseline = (
    pd.notna(full_rebuild_pnl.get("final_equity"))
    and pd.notna(baseline_pnl.get("final_equity"))
    and full_rebuild_pnl.get("final_equity") <= baseline_pnl.get("final_equity")
    and full_rebuild_pnl.get("cagr") <= baseline_pnl.get("cagr")
    and full_rebuild_pnl.get("sharpe_ratio") <= baseline_pnl.get("sharpe_ratio")
)

if rebuild_safe and rebuild_beats_baseline and full_rebuild_mean_accuracy >= baseline_mean_accuracy:
    recommendation = "promote rebuilt model after approval"
    recommendation_reason = (
        "The rebuilt model improves walk-forward PnL and does not lose on mean walk-forward validation accuracy."
    )
elif rebuild_loses_to_baseline:
    recommendation = "hold current V4"
    recommendation_reason = (
        "The rebuilt model underperforms the current baseline on the aligned walk-forward invested period."
    )
else:
    recommendation = "further review needed"
    recommendation_reason = (
        "The rebuilt model changes the walk-forward profile, but the combined validation and PnL evidence is mixed."
    )

summary_lines = [
    "# Full Historical ALFRED Adoption Decision Pack",
    "",
    "## Scope",
    "- Walk-forward invested-period evidence only.",
    "- Training warm-up history remains `2016-01-01` to `2020-12-31`.",
    "- Investment evaluation starts `2021-01-01`.",
    "- Full-dataset/full-signal framing is out of scope.",
    "",
    "## Sources",
    f"- Baseline root: `{baseline_root}`",
    f"- Rebuilt retrain root: `{full_rebuild_root}`",
    f"- Walk-forward PnL root: `{pnl_root}`",
    f"- Baseline silver table: `{baseline_silver_table}`",
    f"- Rebuilt silver table: `{full_rebuild_silver_table}`",
    "",
    "## Safety",
    f"- Rebuilt registry unchanged: `{full_rebuild_metadata.get('registry_unchanged')}`",
    f"- Rebuilt live alias unchanged: `{full_rebuild_metadata.get('live_alias_unchanged')}`",
    "",
    "## Walk-Forward Validation",
    f"- Baseline mean accuracy: `{baseline_mean_accuracy}`",
    f"- Rebuilt mean accuracy: `{full_rebuild_mean_accuracy}`",
    f"- Baseline median accuracy: `{baseline_median_accuracy}`",
    f"- Rebuilt median accuracy: `{full_rebuild_median_accuracy}`",
    "",
    "## Walk-Forward PnL",
    f"- Baseline final equity: `{baseline_pnl.get('final_equity')}`",
    f"- Rebuilt final equity: `{full_rebuild_pnl.get('final_equity')}`",
    f"- Baseline CAGR: `{baseline_pnl.get('cagr')}`",
    f"- Rebuilt CAGR: `{full_rebuild_pnl.get('cagr')}`",
    f"- Baseline Sharpe: `{baseline_pnl.get('sharpe_ratio')}`",
    f"- Rebuilt Sharpe: `{full_rebuild_pnl.get('sharpe_ratio')}`",
    f"- Baseline max drawdown: `{baseline_pnl.get('max_drawdown')}`",
    f"- Rebuilt max drawdown: `{full_rebuild_pnl.get('max_drawdown')}`",
    "",
    "## Regime Differences",
    f"- Walk-forward differing timestamps: `{int(signal_diff.shape[0])}`",
    f"- Probability-diff rows: `{int(probability_diff.shape[0])}`",
    "",
    "## Latest Walk-Forward Regime",
    f"- Timestamp: `{latest_walkforward_row.get('timestamp')}`",
    f"- Baseline: `{latest_walkforward_row.get('baseline_regime_name')}`",
    f"- Rebuilt: `{latest_walkforward_row.get('full_rebuild_regime_name')}`",
    f"- Changed: `{latest_walkforward_row.get('regime_changed')}`",
    "",
    "## Latest Probability View",
    f"- Timestamp: `{latest_probability_row.get('timestamp')}`",
    f"- Baseline max probability: `{latest_probability_row.get('baseline_max_probability')}`",
    f"- Rebuilt max probability: `{latest_probability_row.get('full_rebuild_max_probability')}`",
    f"- Max probability diff: `{latest_probability_row.get('max_probability_diff')}`",
    "",
    "## Recommendation",
    f"- Recommendation: `{recommendation}`",
    f"- Reason: {recommendation_reason}",
]

summary_path = f"{output_root}/summary.md"
write_text(summary_path, "\n".join(summary_lines) + "\n")

recommendation_payload = {
    "phase": "full_historical_walkforward_decision_pack",
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "comparison_root": comparison_root,
    "baseline_root": baseline_root,
    "full_rebuild_root": full_rebuild_root,
    "pnl_root": pnl_root,
    "baseline_model_path": baseline_model_path,
    "full_rebuild_model_path": full_rebuild_model_path,
    "baseline_mean_accuracy": baseline_mean_accuracy,
    "full_rebuild_mean_accuracy": full_rebuild_mean_accuracy,
    "baseline_pnl": baseline_pnl,
    "full_rebuild_pnl": full_rebuild_pnl,
    "walkforward_signal_change_count": int(signal_diff.shape[0]),
    "probability_diff_rows": int(probability_diff.shape[0]),
    "latest_walkforward_regime": {
        "timestamp": str(latest_walkforward_row.get("timestamp")),
        "baseline": latest_walkforward_row.get("baseline_regime_name"),
        "full_rebuild": latest_walkforward_row.get("full_rebuild_regime_name"),
        "changed": bool(latest_walkforward_row.get("regime_changed"))
        if latest_walkforward_row
        else None,
    },
    "recommendation": recommendation,
    "reason": recommendation_reason,
    "rebuild_safe": rebuild_safe,
}
recommendation_path = f"{output_root}/recommendation.json"
write_json(recommendation_path, recommendation_payload)

result = {
    "phase": "full_historical_walkforward_decision_pack",
    "status": "success",
    "comparison_root": comparison_root,
    "artifacts": {
        "metrics_diff": metrics_diff_path,
        "signal_diff": signal_diff_path,
        "probability_diff": probability_diff_path,
        "summary": summary_path,
        "recommendation": recommendation_path,
    },
    "baseline_mean_accuracy": baseline_mean_accuracy,
    "full_rebuild_mean_accuracy": full_rebuild_mean_accuracy,
    "recommendation": recommendation,
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))
