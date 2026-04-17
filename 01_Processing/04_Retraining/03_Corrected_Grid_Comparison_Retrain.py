# Databricks notebook source
# Non-promoting corrected-grid retrain for baseline comparison

# MAGIC %pip install xgboost==3.1.1

import json
from datetime import datetime, timezone
from pathlib import Path

from dateutil.relativedelta import relativedelta
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
MODEL_DIR = f"{VOLUME_ROOT}/models"
LIVE_MODEL_PATH = f"{MODEL_DIR}/bt_fred_1d_xgboost_regime_live.joblib"

TRAIN_START = pd.Timestamp("2016-01-01")
LIVE_START = pd.Timestamp("2021-01-01")

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


def serialize(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def write_json(path_str, payload):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(json.dumps(payload, indent=2, default=str))


def write_text(path_str, content):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(content)


def latest_registry_row():
    if not spark.catalog.tableExists(MODEL_REGISTRY_TABLE):
        return {}
    pdf = spark.table(MODEL_REGISTRY_TABLE).toPandas()
    if pdf.empty:
        return {}
    if "promoted_ts" in pdf.columns:
        pdf["promoted_ts"] = pd.to_datetime(pdf["promoted_ts"], errors="coerce")
        pdf = pdf.sort_values("promoted_ts", ascending=False, na_position="last")
    row = pdf.iloc[0].to_dict()
    return {key: serialize(value) for key, value in row.items()}


def file_info(path_str):
    path = Path(path_str)
    if not path.exists():
        return {
            "path": path_str,
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    stat = path.stat()
    return {
        "path": path_str,
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def build_model():
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        eval_metric="merror",
        tree_method="hist",
        random_state=42,
    )


def label_regime(ret):
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


def append_predictions(model, frame, prefix):
    result = frame.copy()
    X = result[FEATURES].fillna(0)
    preds = model.predict(X).astype(int)
    result[f"{prefix}_predicted_regime"] = preds
    result[f"{prefix}_regime_name"] = result[f"{prefix}_predicted_regime"].map(
        REGIME_NAME_MAP
    )
    try:
        probs = model.predict_proba(X)
        if probs is not None and probs.shape[1] == 3:
            result[f"{prefix}_prob_expansion"] = probs[:, 0]
            result[f"{prefix}_prob_slowdown"] = probs[:, 1]
            result[f"{prefix}_prob_recession"] = probs[:, 2]
    except Exception:
        pass
    return result


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

corrected_root = f"{comparison_root}/corrected_retrain"
model_output_dir = f"{corrected_root}/models"
ensure_dir(corrected_root)
ensure_dir(model_output_dir)

pre_registry = latest_registry_row()
pre_live_alias = file_info(LIVE_MODEL_PATH)

run_ts = datetime.now(timezone.utc).replace(tzinfo=None)
run_month = pd.Timestamp(run_ts).replace(
    day=1, hour=0, minute=0, second=0, microsecond=0
)

df = spark.table(SILVER_TABLE).toPandas().sort_values("timestamp")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["fwd_return_1w"] = df["SPY_close"].shift(-168) / df["SPY_close"] - 1
df["regime"] = df["fwd_return_1w"].apply(label_regime)

trainable = df[(df["timestamp"] >= TRAIN_START) & df["regime"].notna()].copy()
trainable["regime"] = trainable["regime"].astype(int)

latest_feature_ts = pd.Timestamp(df["timestamp"].max())
train_end = min(run_month, latest_feature_ts.floor("D"))

retrain_dates = []
retrain_date = LIVE_START
while retrain_date <= train_end:
    retrain_dates.append(retrain_date)
    retrain_date += relativedelta(months=1)

signals = []
metrics_rows = []

for idx in range(len(retrain_dates) - 1):
    fold_train_end = retrain_dates[idx]
    fold_test_end = retrain_dates[idx + 1]
    train_df = trainable[
        (trainable["timestamp"] >= TRAIN_START)
        & (trainable["timestamp"] < fold_train_end)
    ]
    test_df = trainable[
        (trainable["timestamp"] >= fold_train_end)
        & (trainable["timestamp"] < fold_test_end)
    ].copy()
    if train_df.empty or test_df.empty:
        continue

    model = build_model()
    model.fit(train_df[FEATURES], train_df["regime"])
    test_predictions = append_predictions(
        model,
        test_df[["timestamp", "regime"] + FEATURES].copy(),
        "model",
    )
    test_predictions["predicted_regime"] = test_predictions[
        "model_predicted_regime"
    ].astype(int)
    signals.append(test_predictions)
    metrics_rows.append(
        {
            "train_end": fold_train_end,
            "test_end": fold_test_end,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "accuracy": float(
                (
                    test_predictions["regime"]
                    == test_predictions["predicted_regime"]
                ).mean()
            ),
        }
    )

if not signals:
    raise ValueError("Comparison retraining did not produce any monthly folds")

wf_signals = pd.concat(signals).sort_values("timestamp")

final_train_df = trainable[trainable["timestamp"] < train_end].copy()
live_model = build_model()
live_model.fit(final_train_df[FEATURES], final_train_df["regime"])

full_features = append_predictions(live_model, df.copy(), "model")
full_features["predicted_regime"] = full_features["model_predicted_regime"].astype(int)
full_features["regime_name"] = full_features["predicted_regime"].map(REGIME_NAME_MAP)

model_version = (
    f"bt_fred_1d_xgboost_regime_comparison_{run_month.strftime('%Y_%m_%d')}"
)
local_model_path = f"/tmp/{model_version}.joblib"
joblib.dump(live_model, local_model_path)
model_artifact_path = f"{model_output_dir}/{model_version}.joblib"
Path(model_artifact_path).write_bytes(Path(local_model_path).read_bytes())

validation_csv_path = f"{corrected_root}/monthly_walkforward_validation.csv"
wf_signals_csv_path = f"{corrected_root}/walkforward_signals.csv"
full_signal_csv_path = f"{corrected_root}/full_signal_export.csv"
model_metadata_path = f"{corrected_root}/model_metadata.json"
model_artifact_path_txt = f"{corrected_root}/model_artifact_path.txt"
readme_path = f"{corrected_root}/README.md"

pd.DataFrame(metrics_rows).to_csv(validation_csv_path, index=False)
wf_signals.to_csv(wf_signals_csv_path, index=False)
full_features.to_csv(full_signal_csv_path, index=False)
write_text(model_artifact_path_txt, f"{model_artifact_path}\n")

post_registry = latest_registry_row()
post_live_alias = file_info(LIVE_MODEL_PATH)

registry_unchanged = pre_registry == post_registry
live_alias_unchanged = pre_live_alias == post_live_alias

if not registry_unchanged:
    raise RuntimeError("model_registry_current changed during non-promoting retrain")
if not live_alias_unchanged:
    raise RuntimeError("live model alias file changed during non-promoting retrain")

metrics_pdf = pd.DataFrame(metrics_rows)
model_metadata = {
    "phase": "corrected_retrain",
    "status": "success",
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "comparison_root": comparison_root,
    "corrected_root": corrected_root,
    "source_silver_table": SILVER_TABLE,
    "promote_model": False,
    "model_version": model_version,
    "model_artifact_path": model_artifact_path,
    "live_model_alias_path": LIVE_MODEL_PATH,
    "train_start": serialize(TRAIN_START),
    "train_end": serialize(train_end),
    "latest_feature_ts": serialize(latest_feature_ts),
    "run_ts": serialize(run_ts),
    "run_month": serialize(run_month),
    "walkforward_rows": int(len(wf_signals)),
    "full_signal_rows": int(len(full_features)),
    "monthly_fold_count": int(len(metrics_rows)),
    "mean_accuracy": float(metrics_pdf["accuracy"].mean()),
    "median_accuracy": float(metrics_pdf["accuracy"].median()),
    "pre_registry_row": pre_registry,
    "post_registry_row": post_registry,
    "registry_unchanged": registry_unchanged,
    "pre_live_alias_file": pre_live_alias,
    "post_live_alias_file": post_live_alias,
    "live_alias_unchanged": live_alias_unchanged,
    "artifacts": {
        "monthly_walkforward_validation": validation_csv_path,
        "walkforward_signals": wf_signals_csv_path,
        "full_signal_export": full_signal_csv_path,
        "model_artifact_path_txt": model_artifact_path_txt,
        "readme": readme_path,
    },
}
write_json(model_metadata_path, model_metadata)

readme_lines = [
    "# Corrected-Grid Comparison Retrain",
    "",
    f"- Captured at UTC: `{model_metadata['captured_at_utc']}`",
    f"- Source silver table: `{SILVER_TABLE}`",
    f"- Promote model: `{model_metadata['promote_model']}`",
    f"- Model version: `{model_version}`",
    f"- Model artifact: `{model_artifact_path}`",
    f"- Train end: `{serialize(train_end)}`",
    f"- Monthly fold count: `{len(metrics_rows)}`",
    f"- Mean walk-forward accuracy: `{metrics_pdf['accuracy'].mean():.6f}`",
    f"- Registry unchanged: `{registry_unchanged}`",
    f"- Live alias unchanged: `{live_alias_unchanged}`",
]
write_text(readme_path, "\n".join(readme_lines) + "\n")

result = {
    "phase": "corrected_retrain",
    "status": "success",
    "comparison_root": comparison_root,
    "corrected_root": corrected_root,
    "model_version": model_version,
    "model_artifact_path": model_artifact_path,
    "train_end": serialize(train_end),
    "mean_accuracy": float(metrics_pdf["accuracy"].mean()),
    "monthly_fold_count": int(len(metrics_rows)),
    "registry_unchanged": registry_unchanged,
    "live_alias_unchanged": live_alias_unchanged,
    "artifacts": {
        "model_metadata": model_metadata_path,
        "monthly_walkforward_validation": validation_csv_path,
        "walkforward_signals": wf_signals_csv_path,
        "full_signal_export": full_signal_csv_path,
        "model_artifact_path_txt": model_artifact_path_txt,
        "readme": readme_path,
    },
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))
