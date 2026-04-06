# Databricks notebook source
# Monthly walk-forward retraining and live-model promotion

# MAGIC %pip install xgboost

import os
import shutil
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.sql import functions as F

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
WF_SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
MODEL_DIR = f"{VOLUME_ROOT}/models"
LIVE_MODEL_PATH = f"{MODEL_DIR}/bt_fred_1d_xgboost_regime_live.joblib"
FULL_SIGNAL_PATH = f"{VOLUME_ROOT}/bt_fred_1d_regime_signals-v4.csv"
WF_SIGNAL_PATH = f"{VOLUME_ROOT}/bt_walkforward_regime_signals.csv"
PERFORMANCE_ROOT = f"{VOLUME_ROOT}/performance"

TRAIN_START = pd.Timestamp("2016-01-01")
LIVE_START = pd.Timestamp("2021-01-01")
RUN_TS = datetime.now(timezone.utc).replace(tzinfo=None)
RUN_MONTH = pd.Timestamp(RUN_TS).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

FEATURES = [
    "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
    "GDPC1_z", "INDPRO_z", "UMCSENT_z", "VIXCLS_z", "vix3m_z",
    "VIX_Inversion_Z", "Price_Mom_24H", "CPI_Event_Active", "NFP_Event_Active",
]

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

df = spark.table(SILVER_TABLE).toPandas().sort_values("timestamp")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["fwd_return_1w"] = df["SPY_close"].shift(-168) / df["SPY_close"] - 1
df["regime"] = df["fwd_return_1w"].apply(label_regime)

trainable = df[(df["timestamp"] >= TRAIN_START) & df["regime"].notna()].copy()
trainable["regime"] = trainable["regime"].astype(int)

latest_feature_ts = pd.Timestamp(df["timestamp"].max())
train_end = min(RUN_MONTH, latest_feature_ts.floor("D"))
retrain_dates = []
d = LIVE_START
while d <= train_end:
    retrain_dates.append(d)
    d += relativedelta(months=1)

signals = []
metrics_rows = []

for idx in range(len(retrain_dates) - 1):
    fold_train_end = retrain_dates[idx]
    fold_test_end = retrain_dates[idx + 1]
    train_df = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_train_end)]
    test_df = trainable[(trainable["timestamp"] >= fold_train_end) & (trainable["timestamp"] < fold_test_end)].copy()
    if train_df.empty or test_df.empty:
        continue
    model = build_model()
    model.fit(train_df[FEATURES], train_df["regime"])
    preds = model.predict(test_df[FEATURES]).astype(int)
    test_df["predicted_regime"] = preds
    signals.append(test_df[["timestamp", "regime", "predicted_regime"]])
    metrics_rows.append({
        "train_end": fold_train_end,
        "test_end": fold_test_end,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": float((test_df["regime"] == test_df["predicted_regime"]).mean()),
    })

if not signals:
    raise ValueError("Walk-forward retraining did not produce any monthly folds")

wf_signals = pd.concat(signals).sort_values("timestamp")
wf_signal_sdf = spark.createDataFrame(wf_signals)
wf_signal_sdf.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(WF_SIGNAL_TABLE)

final_train_df = trainable[trainable["timestamp"] < train_end].copy()
live_model = build_model()
live_model.fit(final_train_df[FEATURES], final_train_df["regime"])

full_features = df.copy()
full_features["predicted_regime"] = live_model.predict(full_features[FEATURES].fillna(0)).astype(int)
full_features["regime_name"] = full_features["predicted_regime"].map({0: "Expansion", 1: "Slowdown", 2: "Recession"})

dbutils.fs.mkdirs(MODEL_DIR)
for folder in ["daily", "weekly", "monthly", "quarterly", "yearly", "summary"]:
    dbutils.fs.mkdirs(f"{PERFORMANCE_ROOT}/{folder}")

model_version = f"bt_fred_1d_xgboost_regime_{RUN_MONTH.strftime('%Y_%m_%d')}"
local_model_path = f"/tmp/{model_version}.joblib"
joblib.dump(live_model, local_model_path)
versioned_model_path = f"{MODEL_DIR}/{model_version}.joblib"
dbutils.fs.cp(f"file:{local_model_path}", versioned_model_path, True)
dbutils.fs.cp(versioned_model_path, LIVE_MODEL_PATH, True)

full_features.to_csv(FULL_SIGNAL_PATH, index=False)
wf_signals.to_csv(WF_SIGNAL_PATH, index=False)
pd.DataFrame(metrics_rows).to_csv(f"{PERFORMANCE_ROOT}/summary/monthly_walkforward_validation.csv", index=False)

registry_row = spark.createDataFrame([{
    "model_version": model_version,
    "model_path": LIVE_MODEL_PATH,
    "versioned_model_path": versioned_model_path,
    "promoted_ts": RUN_TS,
    "train_start": TRAIN_START.to_pydatetime(),
    "train_end": train_end.to_pydatetime(),
    "signal_path": FULL_SIGNAL_PATH,
    "wf_signal_table": WF_SIGNAL_TABLE,
}])
registry_row.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(MODEL_REGISTRY_TABLE)

print(f"Promoted live model: {model_version}")
print(f"Live alias: {LIVE_MODEL_PATH}")
print(f"Walk-forward signals written to {WF_SIGNAL_TABLE}")
