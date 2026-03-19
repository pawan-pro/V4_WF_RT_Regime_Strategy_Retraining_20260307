# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC Step 1: Load and Label the Data
# MAGIC Step 2: Walk-Forward Training
# MAGIC Step 3: Feature Importance (The "Alpha" Reveal)

# COMMAND ----------

!pip install xgboost

# COMMAND ----------

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt


# 1. Load the Silver Master Grid
df = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h").toPandas()
df = df.sort_values("timestamp").set_index("timestamp")


# 2. Define the Target (Label)
# We predict the return over the next 168 hours (1 week)
df["fwd_return_1w"] = df["SPY_close"].shift(-168) / df["SPY_close"] - 1


def label_regime(ret):
    if pd.isna(ret):
        return None
    if ret > 0.005:
        return 0  # Expansion (Risk-On)
    if ret < -0.005:
        return 2  # Recession (Risk-Off)
    return 1      # Slowdown (Neutral)


df["label"] = df["fwd_return_1w"].apply(label_regime)


# 3. ── JOB 1: TRAINING DATASET (labels required, last 168 rows legitimately dropped) ──
df_ml = df.dropna(subset=["label"])


# 4. Feature Selection
features = [
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

X = df_ml[features]
y = df_ml["label"]

print(f"Dataset Ready for ML. Total Rows: {len(X)}")
print("Class Distribution:")
print(y.value_counts(normalize=True))


# 5. Model Configuration
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.03,
    objective="multi:softmax",
    num_class=3,
    tree_method="hist",
    random_state=42,
)


# 6. Walk-Forward Validation (Time-Series CV)
tscv = TimeSeriesSplit(n_splits=5)

print("\nTime-series CV folds (train → test ranges):")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    train_start, train_end = X.index[train_idx[0]], X.index[train_idx[-1]]
    test_start, test_end = X.index[test_idx[0]], X.index[test_idx[-1]]
    print(f"\nFold {fold}:")
    print(f"  Train: {train_start}  →  {train_end}")
    print(f"  Test:  {test_start}   →  {test_end}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\n--- Fold {fold} Classification Report ---")
    print(classification_report(y_test, preds))


# 7. Final Training on labeled data (in-sample fit for full-history backtest)
model.fit(X, y)

#model_path = "/Volumes/pawan_quant_research_1/default/market_data/bt_v4_xgboost_regime.joblib"

VIX_SOURCE = "fred_1d" #fred_1d #toggle #portara_1h

model_path  = f"/Volumes/pawan_quant_research_1/default/market_data/bt_{VIX_SOURCE}_xgboost_regime.joblib"
joblib.dump(model, model_path)
print(f"\n✅ V4 XGBoost Model saved to {model_path}")


# 8. Feature Importance Visualization
importance = model.feature_importances_
feat_imp = pd.Series(importance, index=features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind="barh", color="teal")
plt.title("V4 Feature Importance: Normalized Economic Drivers")
plt.xlabel("Importance Score")
plt.show()


# ── JOB 2: SIGNAL GENERATION (runs on FULL df — no label needed, just features) ──
# Predict on the complete unfiltered dataset so the last 168 rows are included
print("\nGenerating hourly regime signals on full dataset...")
X_full = df[features].fillna(0)
df["predicted_regime"] = model.predict(X_full)

regime_map = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
df["regime_name"] = df["predicted_regime"].map(regime_map)

#signal_path = "/Volumes/pawan_quant_research_1/default/market_data/bt_regime_signals-v4.csv"

signal_path = f"/Volumes/pawan_quant_research_1/default/market_data/bt_{VIX_SOURCE}_regime_signals-v4.csv"
df.to_csv(signal_path)

print(f"✅ V4 Signal File saved to {signal_path}")
print(f"   Signal coverage: {df.index.min()} → {df.index.max()}")
print(f"   Total rows: {len(df):,}  (including last 168 rows previously missing)")

plt.figure(figsize=(15, 5))
df["predicted_regime"].plot(alpha=0.6, color="navy")
plt.title("V4 Hourly Regime Predictions (2016 - Feb 2026)")
plt.ylabel("Regime (0=Expansion, 1=Slowdown, 2=Recession)")
plt.show()
