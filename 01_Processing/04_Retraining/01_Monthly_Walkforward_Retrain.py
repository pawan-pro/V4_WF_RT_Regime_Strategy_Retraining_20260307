# Databricks notebook source
# Monthly walk-forward retraining with optional non-promoting comparison mode

# MAGIC %pip install xgboost

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

DEFAULT_SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
DEFAULT_WF_SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
MODEL_DIR = f"{VOLUME_ROOT}/models"
LIVE_MODEL_PATH = f"{MODEL_DIR}/bt_fred_1d_xgboost_regime_live.joblib"
FULL_SIGNAL_PATH = f"{VOLUME_ROOT}/bt_fred_1d_regime_signals-v4.csv"
WF_SIGNAL_PATH = f"{VOLUME_ROOT}/bt_walkforward_regime_signals.csv"
PERFORMANCE_ROOT = f"{VOLUME_ROOT}/performance"

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


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def append_model_outputs(model, frame, prefix):
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
            result[f"{prefix}_max_probability"] = probs.max(axis=1)
    except Exception:
        pass
    return result


silver_table = DEFAULT_SILVER_TABLE
wf_signal_table = DEFAULT_WF_SIGNAL_TABLE
comparison_root = ""
output_subdir = ""
promote_model = True

comparison_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
comparison_root_default = f"{VOLUME_ROOT}/comparison/{comparison_date}_adhoc_retrain"

if "dbutils" in globals():
    dbutils.widgets.text("silver_table", DEFAULT_SILVER_TABLE)
    dbutils.widgets.text("wf_signal_table", DEFAULT_WF_SIGNAL_TABLE)
    dbutils.widgets.text("comparison_root", "")
    dbutils.widgets.text("output_subdir", "")
    dbutils.widgets.dropdown("promote_model", "true", ["true", "false"])
    silver_table = dbutils.widgets.get("silver_table").strip() or DEFAULT_SILVER_TABLE
    wf_signal_table = (
        dbutils.widgets.get("wf_signal_table").strip() or DEFAULT_WF_SIGNAL_TABLE
    )
    comparison_root = dbutils.widgets.get("comparison_root").strip()
    output_subdir = dbutils.widgets.get("output_subdir").strip()
    promote_model = parse_bool(dbutils.widgets.get("promote_model"), default=True)

custom_inputs = (
    silver_table != DEFAULT_SILVER_TABLE
    or wf_signal_table != DEFAULT_WF_SIGNAL_TABLE
    or bool(comparison_root)
    or bool(output_subdir)
)
if custom_inputs and promote_model:
    print(
        "Custom retrain inputs detected. Forcing non-promoting mode to protect "
        "production outputs."
    )
    promote_model = False

if not comparison_root and not promote_model:
    comparison_root = comparison_root_default
if not output_subdir:
    output_subdir = "full_rebuild_retrain" if not promote_model else "production_retrain"

run_ts = datetime.now(timezone.utc).replace(tzinfo=None)
run_month = pd.Timestamp(run_ts).replace(
    day=1, hour=0, minute=0, second=0, microsecond=0
)

pre_registry = latest_registry_row()
pre_live_alias = file_info(LIVE_MODEL_PATH)

df = spark.table(silver_table).toPandas().sort_values("timestamp")
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
    test_predictions = append_model_outputs(
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
    raise ValueError("Walk-forward retraining did not produce any monthly folds")

wf_signals = pd.concat(signals).sort_values("timestamp")
metrics_pdf = pd.DataFrame(metrics_rows)

final_train_df = trainable[trainable["timestamp"] < train_end].copy()
live_model = build_model()
live_model.fit(final_train_df[FEATURES], final_train_df["regime"])

full_features = append_model_outputs(live_model, df.copy(), "model")
full_features["predicted_regime"] = full_features["model_predicted_regime"].astype(int)
full_features["regime_name"] = full_features["predicted_regime"].map(REGIME_NAME_MAP)

model_version = f"bt_fred_1d_xgboost_regime_{run_month.strftime('%Y_%m_%d')}"
local_model_path = f"/tmp/{model_version}.joblib"
joblib.dump(live_model, local_model_path)

result = {
    "phase": "monthly_walkforward_retrain",
    "status": "success",
    "promote_model": promote_model,
    "source_silver_table": silver_table,
    "train_start": serialize(TRAIN_START),
    "train_end": serialize(train_end),
    "latest_feature_ts": serialize(latest_feature_ts),
    "run_ts": serialize(run_ts),
    "run_month": serialize(run_month),
    "model_version": model_version,
    "monthly_fold_count": int(len(metrics_rows)),
    "mean_accuracy": float(metrics_pdf["accuracy"].mean()),
    "median_accuracy": float(metrics_pdf["accuracy"].median()),
}

if promote_model:
    wf_signal_export = wf_signals[["timestamp", "regime", "predicted_regime"]].copy()
    full_signal_export = full_features.copy()
    full_signal_export = full_signal_export.drop(
        columns=[
            "model_predicted_regime",
            "model_regime_name",
            "model_prob_expansion",
            "model_prob_slowdown",
            "model_prob_recession",
            "model_max_probability",
        ],
        errors="ignore",
    )

    dbutils.fs.mkdirs(MODEL_DIR)
    for folder in ["daily", "weekly", "monthly", "quarterly", "yearly", "summary"]:
        dbutils.fs.mkdirs(f"{PERFORMANCE_ROOT}/{folder}")

    versioned_model_path = f"{MODEL_DIR}/{model_version}.joblib"
    dbutils.fs.cp(f"file:{local_model_path}", versioned_model_path, True)
    dbutils.fs.cp(versioned_model_path, LIVE_MODEL_PATH, True)

    wf_signal_sdf = spark.createDataFrame(wf_signal_export)
    (
        wf_signal_sdf.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .format("delta")
        .saveAsTable(wf_signal_table)
    )

    full_signal_export.to_csv(FULL_SIGNAL_PATH, index=False)
    wf_signal_export.to_csv(WF_SIGNAL_PATH, index=False)
    metrics_pdf.to_csv(
        f"{PERFORMANCE_ROOT}/summary/monthly_walkforward_validation.csv",
        index=False,
    )

    registry_row = spark.createDataFrame(
        [
            {
                "model_version": model_version,
                "model_path": LIVE_MODEL_PATH,
                "versioned_model_path": versioned_model_path,
                "promoted_ts": run_ts,
                "train_start": TRAIN_START.to_pydatetime(),
                "train_end": train_end.to_pydatetime(),
                "signal_path": FULL_SIGNAL_PATH,
                "wf_signal_table": wf_signal_table,
            }
        ]
    )
    (
        registry_row.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(MODEL_REGISTRY_TABLE)
    )

    result.update(
        {
            "mode": "production",
            "wf_signal_table": wf_signal_table,
            "live_model_alias_path": LIVE_MODEL_PATH,
            "versioned_model_path": versioned_model_path,
            "full_signal_path": FULL_SIGNAL_PATH,
            "wf_signal_path": WF_SIGNAL_PATH,
            "performance_root": PERFORMANCE_ROOT,
        }
    )
else:
    retrain_root = f"{comparison_root}/{output_subdir}"
    model_output_dir = f"{retrain_root}/models"
    performance_export_root = f"{retrain_root}/performance_exports"
    summary_export_dir = f"{performance_export_root}/summary"

    ensure_dir(retrain_root)
    ensure_dir(model_output_dir)
    ensure_dir(summary_export_dir)

    model_artifact_path = f"{model_output_dir}/{model_version}.joblib"
    Path(model_artifact_path).write_bytes(Path(local_model_path).read_bytes())

    validation_csv_path = f"{retrain_root}/monthly_walkforward_validation.csv"
    validation_summary_copy_path = (
        f"{summary_export_dir}/monthly_walkforward_validation.csv"
    )
    wf_signals_csv_path = f"{retrain_root}/walkforward_signals.csv"
    full_signal_csv_path = f"{retrain_root}/full_signal_export.csv"
    recent_signal_sample_path = f"{retrain_root}/recent_signal_sample.csv"
    model_metadata_path = f"{retrain_root}/model_metadata.json"
    model_artifact_path_txt = f"{retrain_root}/model_artifact_path.txt"
    readme_path = f"{retrain_root}/README.md"

    metrics_pdf.to_csv(validation_csv_path, index=False)
    metrics_pdf.to_csv(validation_summary_copy_path, index=False)
    wf_signals.to_csv(wf_signals_csv_path, index=False)
    full_features.to_csv(full_signal_csv_path, index=False)
    full_features.sort_values("timestamp").tail(168).to_csv(
        recent_signal_sample_path,
        index=False,
    )
    write_text(model_artifact_path_txt, f"{model_artifact_path}\n")

    post_registry = latest_registry_row()
    post_live_alias = file_info(LIVE_MODEL_PATH)

    registry_unchanged = pre_registry == post_registry
    live_alias_unchanged = pre_live_alias == post_live_alias

    if not registry_unchanged:
        raise RuntimeError("model_registry_current changed during non-promoting retrain")
    if not live_alias_unchanged:
        raise RuntimeError("live model alias changed during non-promoting retrain")

    model_metadata = {
        "phase": "comparison_retrain",
        "status": "success",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_root": comparison_root,
        "retrain_root": retrain_root,
        "output_subdir": output_subdir,
        "source_silver_table": silver_table,
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
            "performance_summary_validation": validation_summary_copy_path,
            "walkforward_signals": wf_signals_csv_path,
            "full_signal_export": full_signal_csv_path,
            "recent_signal_sample": recent_signal_sample_path,
            "model_artifact_path_txt": model_artifact_path_txt,
            "readme": readme_path,
        },
    }
    write_json(model_metadata_path, model_metadata)

    readme_lines = [
        "# Non-Promoting Walk-Forward Retrain",
        "",
        f"- Captured at UTC: `{model_metadata['captured_at_utc']}`",
        f"- Source silver table: `{silver_table}`",
        f"- Model version: `{model_version}`",
        f"- Model artifact: `{model_artifact_path}`",
        f"- Train end: `{serialize(train_end)}`",
        f"- Monthly fold count: `{len(metrics_rows)}`",
        f"- Mean walk-forward accuracy: `{metrics_pdf['accuracy'].mean():.6f}`",
        f"- Recent signal sample: `{recent_signal_sample_path}`",
        f"- Registry unchanged: `{registry_unchanged}`",
        f"- Live alias unchanged: `{live_alias_unchanged}`",
    ]
    write_text(readme_path, "\n".join(readme_lines) + "\n")

    result.update(
        {
            "mode": "comparison",
            "comparison_root": comparison_root,
            "retrain_root": retrain_root,
            "registry_unchanged": registry_unchanged,
            "live_alias_unchanged": live_alias_unchanged,
            "artifacts": {
                "model_metadata": model_metadata_path,
                "monthly_walkforward_validation": validation_csv_path,
                "performance_summary_validation": validation_summary_copy_path,
                "walkforward_signals": wf_signals_csv_path,
                "full_signal_export": full_signal_csv_path,
                "recent_signal_sample": recent_signal_sample_path,
                "model_artifact_path_txt": model_artifact_path_txt,
                "readme": readme_path,
            },
        }
    )

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))
