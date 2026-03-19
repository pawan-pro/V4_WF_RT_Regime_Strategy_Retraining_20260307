# Databricks notebook source
pip install xgboost

# COMMAND ----------

# ── ONE-TIME MIGRATION: Add all missing columns to bronze_rt_regime_signal ──

TARGET_TABLE = "pawan_quant_research_1.default.bronze_rt_regime_signal"

# Check what currently exists
existing_cols = [
    r["col_name"]
    for r in spark.sql(f"DESCRIBE {TARGET_TABLE}").collect()
    if not r["col_name"].startswith("#")
]
print("Existing columns:", existing_cols)

# All columns that should exist
desired = {
    "PAYEMS_z":        "DOUBLE",
    "GDPC1_z":         "DOUBLE",
    "INDPRO_z":        "DOUBLE",
    "PCEPI_z":         "DOUBLE",
    "PCE_z":           "DOUBLE",
    "UMCSENT_z":       "DOUBLE",
    "VIX_Inversion_Z": "DOUBLE",
}

for col, dtype in desired.items():
    if col not in existing_cols:
        spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {col} {dtype}")
        print(f"✅ Added {col}")
    else:
        print(f"   Skipped {col} (already exists)")

print("\nFinal schema:")
spark.sql(f"DESCRIBE {TARGET_TABLE}").show()

# COMMAND ----------

# Databricks notebook source
# RT_04: Model Inference Engine
import joblib
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from datetime import datetime, timezone

MODEL_PATH   = "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_xgboost_regime.joblib"
SILVER_RT    = "pawan_quant_research_1.default.silver_rt_1h"
TARGET_TABLE = "pawan_quant_research_1.default.bronze_rt_regime_signal"

FEATURES = [
    "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
    "GDPC1_z", "INDPRO_z", "UMCSENT_z", "VIXCLS_z", "vix3m_z",
    "VIX_Inversion_Z", "Price_Mom_24H", "CPI_Event_Active", "NFP_Event_Active",
]

REGIME_MAP   = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
POSITION_MAP = {0: "100% QQQ",  1: "100% SPY",  2: "100% CASH"}

# ── STEP 1: Load model ──
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# ── STEP 2: Get latest silver RT row ──
latest_row = spark.sql(f"""
    SELECT * FROM {SILVER_RT}
    ORDER BY timestamp DESC LIMIT 1
""").toPandas()

if latest_row.empty:
    raise ValueError("❌ silver_rt_1h is empty — run RT_03 first")

inference_ts = latest_row["timestamp"].iloc[0].to_pydatetime().replace(tzinfo=None)
X_rt = latest_row[FEATURES].fillna(0)

# ── STEP 3: Predict ──
regime_int  = int(model.predict(X_rt)[0])
regime_name = REGIME_MAP[regime_int]
position    = POSITION_MAP[regime_int]

print(f"\n{'='*55}")
print(f"  RT REGIME SIGNAL @ {inference_ts}")
print(f"{'='*55}")
print(f"  Regime:     {regime_name} ({regime_int})")
print(f"  Position:   {position}")
print(f"  VIXCLS_z:   {latest_row['VIXCLS_z'].iloc[0]:.3f}")
print(f"  vix3m_z:    {latest_row['vix3m_z'].iloc[0]:.3f}")
print(f"  PAYEMS_z:   {latest_row['PAYEMS_z'].iloc[0]:.3f}  ← labor market signal")
print(f"  UNRATE_z:   {latest_row['UNRATE_z'].iloc[0]:.3f}")
print(f"  CPIAUCSL_z: {latest_row['CPIAUCSL_z'].iloc[0]:.3f}")
print(f"  GDPC1_z:    {latest_row['GDPC1_z'].iloc[0]:.3f}")
print(f"  PriceMom:   {latest_row['Price_Mom_24H'].iloc[0]:.4f}")
print(f"{'='*55}")

# ── STEP 4: Migrate table if PAYEMS_z column is missing ──
existing_cols = [
    r["col_name"]
    for r in spark.sql(f"DESCRIBE {TARGET_TABLE}").collect()
    if not r["col_name"].startswith("#")
]

if "PAYEMS_z" not in existing_cols:
    print("⚠️  Migrating table: adding PAYEMS_z column...")
    spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN PAYEMS_z DOUBLE")
    print("✅ Migration complete")
else:
    print("✅ Schema OK — PAYEMS_z column exists")

# ── STEP 5: Create table if not exists (full schema) ──
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        timestamp     TIMESTAMP,
        regime_int    INT,
        regime_name   STRING,
        position      STRING,
        VIXCLS_z      DOUBLE,
        vix3m_z       DOUBLE,
        Price_Mom_24H DOUBLE,
        CPIAUCSL_z    DOUBLE,
        UNRATE_z      DOUBLE,
        PAYEMS_z      DOUBLE,
        GDPC1_z       DOUBLE,
        INDPRO_z      DOUBLE,
        PCEPI_z       DOUBLE,
        PCE_z         DOUBLE,
        UMCSENT_z     DOUBLE,
        VIX_Inversion_Z DOUBLE,
        source        STRING,
        ingestion_ts  TIMESTAMP
    )
    USING DELTA
""")

# ── STEP 6: Write with explicit schema ──
schema = StructType([
    StructField("timestamp",       TimestampType(), True),
    StructField("regime_int",      IntegerType(),   True),
    StructField("regime_name",     StringType(),    True),
    StructField("position",        StringType(),    True),
    StructField("VIXCLS_z",        DoubleType(),    True),
    StructField("vix3m_z",         DoubleType(),    True),
    StructField("Price_Mom_24H",   DoubleType(),    True),
    StructField("CPIAUCSL_z",      DoubleType(),    True),
    StructField("UNRATE_z",        DoubleType(),    True),
    StructField("PAYEMS_z",        DoubleType(),    True),  # ← FIXED
    StructField("GDPC1_z",         DoubleType(),    True),
    StructField("INDPRO_z",        DoubleType(),    True),
    StructField("PCEPI_z",         DoubleType(),    True),
    StructField("PCE_z",           DoubleType(),    True),
    StructField("UMCSENT_z",       DoubleType(),    True),
    StructField("VIX_Inversion_Z", DoubleType(),    True),
    StructField("source",          StringType(),    True),
])

signal_row = {
    "timestamp":       inference_ts,
    "regime_int":      regime_int,
    "regime_name":     regime_name,
    "position":        position,
    "VIXCLS_z":        float(latest_row["VIXCLS_z"].iloc[0]),
    "vix3m_z":         float(latest_row["vix3m_z"].iloc[0]),
    "Price_Mom_24H":   float(latest_row["Price_Mom_24H"].iloc[0]),
    "CPIAUCSL_z":      float(latest_row["CPIAUCSL_z"].iloc[0]),
    "UNRATE_z":        float(latest_row["UNRATE_z"].iloc[0]),
    "PAYEMS_z":        float(latest_row["PAYEMS_z"].iloc[0]),   # ← FIXED
    "GDPC1_z":         float(latest_row["GDPC1_z"].iloc[0]),
    "INDPRO_z":        float(latest_row["INDPRO_z"].iloc[0]),
    "PCEPI_z":         float(latest_row["PCEPI_z"].iloc[0]),
    "PCE_z":           float(latest_row["PCE_z"].iloc[0]),
    "UMCSENT_z":       float(latest_row["UMCSENT_z"].iloc[0]),
    "VIX_Inversion_Z": float(latest_row["VIX_Inversion_Z"].iloc[0]),
    "source":          "rt_inference_fred_1d",
}

sdf = spark.createDataFrame([signal_row], schema=schema) \
           .withColumn("ingestion_ts", F.current_timestamp())

sdf.createOrReplaceTempView("incoming_signal")
spark.sql(f"""
    MERGE INTO {TARGET_TABLE} t
    USING incoming_signal s
    ON t.timestamp = s.timestamp
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"✅ Signal written to {TARGET_TABLE}")

# ── STEP 7: Audit — cross-check all features match silver_rt_1h ──
print("\n── Audit: bronze_rt_regime_signal (latest) ──")
spark.sql(f"""
    SELECT timestamp, regime_name, position,
           PAYEMS_z, UNRATE_z, CPIAUCSL_z,
           VIXCLS_z, GDPC1_z, Price_Mom_24H
    FROM {TARGET_TABLE}
    ORDER BY timestamp DESC LIMIT 1
""").show(vertical=True)

print("\n── Audit: silver_rt_1h (should match above) ──")
spark.sql(f"""
    SELECT timestamp,
           PAYEMS_z, UNRATE_z, CPIAUCSL_z,
           VIXCLS_z, GDPC1_z, Price_Mom_24H
    FROM {SILVER_RT}
    ORDER BY timestamp DESC LIMIT 1
""").show(vertical=True)

# COMMAND ----------

# ── REGIME TIMELINE: last 7 days ──
spark.sql("""
    SELECT timestamp, regime_name, position,
           VIXCLS_z, vix3m_z, VIX_Inversion_Z,
           PAYEMS_z, CPIAUCSL_z
    FROM pawan_quant_research_1.default.bronze_rt_regime_signal
    ORDER BY timestamp DESC LIMIT 48
""").show(truncate=False)

# COMMAND ----------

# ── What drove yesterday's switch? Check VIX history ──
spark.sql("""
    SELECT timestamp, close as VIX, series
    FROM pawan_quant_research_1.default.bronze_rt_vix_1d
    WHERE series IN ('VIXCLS', 'vix3m')
    AND timestamp >= '2026-02-26'
    ORDER BY timestamp DESC, series
""").show(truncate=False)

# COMMAND ----------

# ── What combination tips the model back to Expansion? ──
import joblib
import pandas as pd
import numpy as np

model = joblib.load(
    "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_xgboost_regime.joblib"
)

# Current state
current = {
    "CPIAUCSL_z": 1.731, "PCEPI_z": 1.355, "PCE_z": 1.481,
    "PAYEMS_z": -3.502,  "UNRATE_z": 0.247, "GDPC1_z": 1.872,
    "INDPRO_z": -1.231,  "UMCSENT_z": -0.701,
    "VIXCLS_z": 0.081,   "vix3m_z": 0.081,
    "VIX_Inversion_Z": 0.0,
    "Price_Mom_24H": 0.015,
    "CPI_Event_Active": 0, "NFP_Event_Active": 0,
}

FEATURES = [
    "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
    "GDPC1_z", "INDPRO_z", "UMCSENT_z", "VIXCLS_z", "vix3m_z",
    "VIX_Inversion_Z", "Price_Mom_24H", "CPI_Event_Active", "NFP_Event_Active",
]
REGIME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}

base = pd.DataFrame([current])[FEATURES]
pred = model.predict(base)[0]
proba = model.predict_proba(base)[0]

print(f"\nCURRENT STATE:")
print(f"  Regime: {REGIME_MAP[pred]}  |  "
      f"Expansion: {proba[0]:.1%}  Slowdown: {proba[1]:.1%}  Recession: {proba[2]:.1%}")

# ── Sensitivity: what if PAYEMS recovers? ──
print(f"\nSENSITIVITY — PAYEMS_z recovery path:")
print(f"  {'PAYEMS_z':>10}  {'Regime':<12}  {'P(Expansion)':>13}  {'P(Recession)':>13}")
print(f"  {'-'*55}")
for payems in [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]:
    row = base.copy()
    row["PAYEMS_z"] = payems
    p = model.predict(row)[0]
    pr = model.predict_proba(row)[0]
    marker = " ← current" if payems == -3.5 else (" ← flip point" if p != pred else "")
    print(f"  {payems:>10.1f}  {REGIME_MAP[p]:<12}  {pr[0]:>13.1%}  {pr[2]:>13.1%}{marker}")

# ── Sensitivity: what if VIX spikes again? ──
print(f"\nSENSITIVITY — vix3m_z stress scenarios (PAYEMS fixed at -3.502):")
print(f"  {'vix3m_z':>10}  {'Regime':<12}  {'P(Expansion)':>13}  {'P(Recession)':>13}")
print(f"  {'-'*55}")
for vix3m in [0.08, 0.5, 1.0, 1.5, 1.811, 2.0, 2.5, 3.0]:
    row = base.copy()
    row["vix3m_z"] = vix3m
    p = model.predict(row)[0]
    pr = model.predict_proba(row)[0]
    marker = " ← yesterday peak" if vix3m == 1.811 else ""
    print(f"  {vix3m:>10.3f}  {REGIME_MAP[p]:<12}  {pr[0]:>13.1%}  {pr[2]:>13.1%}{marker}")

# ── What PAYEMS_z would flip regime back to Expansion? ──
print(f"\nREGIME FLIP THRESHOLD:")
for payems in np.arange(-3.5, 1.0, 0.1):
    row = base.copy()
    row["PAYEMS_z"] = round(payems, 2)
    p = model.predict(row)[0]
    if p != 2:
        print(f"  Regime flips from Recession → {REGIME_MAP[p]} when PAYEMS_z > {payems:.1f}")
        print(f"  That means next NFP needs to show payroll recovery")
        print(f"  Current PAYEMS_z=-3.502 → need ~{payems:.1f}σ = structural improvement")
        break

# COMMAND ----------

# ── DECISION BOUNDARY DEEP DIVE ──────────────────────────────────────

# 1. What SINGLE feature change flips the regime right now?
print("=== SINGLE FEATURE FLIP SCAN ===")
print(f"  {'Feature':<20} {'Value':>8}  {'Nudge+0.1':>12}  {'Nudge-0.1':>12}  {'Flip?'}")
print(f"  {'-'*70}")

for feat in FEATURES:
    if "Event" in feat:
        continue
    row_up   = base.copy(); row_up[feat]   = base[feat].values[0] + 0.1
    row_down = base.copy(); row_down[feat] = base[feat].values[0] - 0.1
    p_up   = REGIME_MAP[model.predict(row_up)[0]]
    p_down = REGIME_MAP[model.predict(row_down)[0]]
    current_regime = REGIME_MAP[pred]
    flip_up   = "🔄 FLIP" if p_up   != current_regime else ""
    flip_down = "🔄 FLIP" if p_down != current_regime else ""
    print(f"  {feat:<20} {base[feat].values[0]:>8.3f}  "
          f"+0.1→{p_up:<10}  -0.1→{p_down:<10}  {flip_up or flip_down}")

# 2. The real issue — probabilities are nearly equal, check the margin
print(f"\n=== CONFIDENCE MARGIN ===")
proba = model.predict_proba(base)[0]
margin = abs(proba[0] - proba[2])
print(f"  Expansion prob:  {proba[0]:.4f}")
print(f"  Slowdown prob:   {proba[1]:.4f}")
print(f"  Recession prob:  {proba[2]:.4f}")
print(f"  Margin:          {margin:.4f}  ({'HIGH confidence' if margin > 0.2 else 'LOW confidence — on the boundary'})")

# 3. What combo of features is creating the non-monotonic vix3m behavior?
print(f"\n=== VIX3M vs VIXCLS JOINT SENSITIVITY ===")
print(f"  {'vix3m_z':>8}  {'VIXCLS_z':>9}  {'Regime':<12}  {'P(Exp)':>8}  {'P(Rec)':>8}")
print(f"  {'-'*55}")
for vix3m in [0.08, 0.5, 1.0, 1.5, 1.811, 2.5]:
    for vixcls in [0.08, 0.5, 1.0, 1.5]:
        row = base.copy()
        row["vix3m_z"]  = vix3m
        row["VIXCLS_z"] = vixcls
        p  = model.predict(row)[0]
        pr = model.predict_proba(row)[0]
        print(f"  {vix3m:>8.3f}  {vixcls:>9.3f}  {REGIME_MAP[p]:<12}  {pr[0]:>8.1%}  {pr[2]:>8.1%}")
    print()

# 4. Most important — what does the BT history look like at similar feature values?
# Find historical rows where the model was also on the boundary
print(f"\n=== HISTORICAL ANALOGUES (similar macro config) ===")
spark.sql(f"""
    SELECT timestamp, 
           ROUND(VIXCLS_z,2) as VIXCLS_z,
           ROUND(vix3m_z,2) as vix3m_z,
           ROUND(PAYEMS_z,2) as PAYEMS_z,
           ROUND(CPIAUCSL_z,2) as CPIAUCSL_z,
           ROUND(GDPC1_z,2) as GDPC1_z,
           ROUND(INDPRO_z,2) as INDPRO_z
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
    WHERE PAYEMS_z < -2.5
    AND VIXCLS_z < 0.5
    AND GDPC1_z > 1.0
    ORDER BY timestamp DESC
    LIMIT 20
""").show(truncate=False)

# COMMAND ----------

# ── OOD DIAGNOSTIC ─────────────────────────────────────────────────────

import numpy as np
from sklearn.preprocessing import StandardScaler

# Get full BT history to measure how far current point is from training data
bt_pdf = spark.sql(f"""
    SELECT {', '.join(FEATURES)}
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
    WHERE {' AND '.join([f'{f} IS NOT NULL' for f in FEATURES])}
""").toPandas()

print(f"Training rows: {len(bt_pdf):,}")

# How far is current point from training distribution (per feature)?
print(f"\n=== CURRENT POINT vs TRAINING DISTRIBUTION ===")
print(f"  {'Feature':<20} {'Train Min':>10} {'Train Max':>10} {'Current':>10} {'OOD?':>8}")
print(f"  {'-'*62}")

ood_flags = []
for feat in FEATURES:
    if "Event" in feat:
        continue
    train_min = bt_pdf[feat].min()
    train_max = bt_pdf[feat].max()
    curr_val  = base[feat].values[0]
    ood = curr_val < train_min or curr_val > train_max
    ood_flags.append((feat, ood))
    flag = "🚨 OOD" if ood else "✅"
    print(f"  {feat:<20} {train_min:>10.3f} {train_max:>10.3f} {curr_val:>10.3f} {flag:>8}")

# Mahalanobis distance — how unusual is this point overall?
from numpy.linalg import inv
scaler = StandardScaler()
bt_scaled = scaler.fit_transform(bt_pdf[FEATURES[:-2]])  # exclude event flags
curr_scaled = scaler.transform(base[FEATURES[:-2]])

cov = np.cov(bt_scaled.T)
try:
    cov_inv = inv(cov)
    diff = curr_scaled[0] - bt_scaled.mean(axis=0)
    mahal = float(np.sqrt(diff @ cov_inv @ diff))
    # Compare to typical distance in training set
    sample_dists = []
    for i in range(0, min(1000, len(bt_scaled)), 10):
        d = bt_scaled[i] - bt_scaled.mean(axis=0)
        sample_dists.append(float(np.sqrt(d @ cov_inv @ d)))
    p95 = np.percentile(sample_dists, 95)
    print(f"\n=== MAHALANOBIS DISTANCE (overall OOD score) ===")
    print(f"  Current point distance: {mahal:.2f}")
    print(f"  Training data p95:      {p95:.2f}")
    print(f"  Ratio:                  {mahal/p95:.2f}x")
    if mahal > p95:
        print(f"  🚨 Current macro state is OUTSIDE normal training distribution")
        print(f"     Model confidence scores are unreliable")
    else:
        print(f"  ✅ Within training distribution")
except:
    print("  Could not compute Mahalanobis — singular covariance matrix")

# What DID the model see during training at Recession calls?
print(f"\n=== TRAINING RECESSION EXAMPLES (what model learned from) ===")
spark.sql(f"""
    SELECT timestamp,
           ROUND(PAYEMS_z,2) as PAYEMS_z,
           ROUND(VIXCLS_z,2) as VIXCLS_z,
           ROUND(vix3m_z,2) as vix3m_z,
           ROUND(GDPC1_z,2) as GDPC1_z,
           ROUND(CPIAUCSL_z,2) as CPIAUCSL_z,
           ROUND(INDPRO_z,2) as INDPRO_z
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
    WHERE PAYEMS_z < -1.5
    ORDER BY PAYEMS_z ASC
    LIMIT 20
""").show(truncate=False)

# COMMAND ----------

# Add this confidence gate to RT_04 signal_row:
proba = model.predict_proba(X_rt)[0]
confidence = float(max(proba))
margin = float(sorted(proba)[-1] - sorted(proba)[-2])

signal_row["confidence"]    = confidence   # add DOUBLE col
signal_row["margin"]        = margin       # add DOUBLE col  
signal_row["low_confidence"] = 1 if margin < 0.10 else 0  # add INT col

# And in your print output:
print(f"  Confidence: {confidence:.1%}  Margin: {margin:.1%}  "
      f"{'⚠️ LOW CONF — review before acting' if margin < 0.10 else '✅'}")

# COMMAND ----------

# When did PAYEMS_z cross -3.5 and what was the actual print?
spark.sql("""
    SELECT timestamp,
           ROUND(PAYEMS_z, 3) as PAYEMS_z,
           ROUND(PAYEMS_z - LAG(PAYEMS_z) OVER (ORDER BY timestamp), 3) as z_change
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
    WHERE timestamp >= '2026-01-01'
    AND PAYEMS_z IS NOT NULL
    ORDER BY timestamp ASC
""").show(truncate=False)

# COMMAND ----------

# ── CURRENT VIX STATE + TERM STRUCTURE ────────────────────────────────
import requests
from datetime import datetime

# Check latest VIX from your bronze feed
spark.sql("""
    SELECT series, timestamp, close,
           ROUND(close - LAG(close) OVER (PARTITION BY series ORDER BY timestamp), 2) as day_change
    FROM pawan_quant_research_1.default.bronze_rt_vix_1d
    WHERE series IN ('VIXCLS', 'vix3m')
    ORDER BY timestamp DESC, series
    LIMIT 10
""").show(truncate=False)

# Term structure: is vix3m > VIXCLS (inversion = fear of sustained stress)?
spark.sql("""
    SELECT 
        v1.timestamp,
        ROUND(v1.close, 2) as VIXCLS,
        ROUND(v2.close, 2) as VIX3M,
        ROUND(v2.close - v1.close, 2) as spread_3m_minus_spot,
        CASE 
            WHEN v2.close > v1.close THEN '⚠️  INVERTED (3M > spot = sustained fear)'
            ELSE '✅ Normal (spot > 3M = short-term spike only)'
        END as structure
    FROM pawan_quant_research_1.default.bronze_rt_vix_1d v1
    JOIN pawan_quant_research_1.default.bronze_rt_vix_1d v2
        ON v1.timestamp = v2.timestamp
        AND v1.series = 'VIXCLS'
        AND v2.series = 'vix3m'
    ORDER BY v1.timestamp DESC
    LIMIT 7
""").show(truncate=False)

# COMMAND ----------

# ── RT_02 DATA INTEGRITY CHECK ─────────────────────────────────────────

print("=== bronze_rt_vix_1d — full recent history ===")
spark.sql("""
    SELECT series, timestamp, close
    FROM pawan_quant_research_1.default.bronze_rt_vix_1d
    WHERE timestamp >= '2026-02-20'
    ORDER BY timestamp DESC, series
""").show(truncate=False)

# The join query that showed REAL values earlier:
print("=== Cross-check: what the JOIN showed (source of truth) ===")
spark.sql("""
    SELECT 
        v1.timestamp,
        v1.close as VIXCLS_actual,
        v2.close as VIX3M_actual,
        ROUND(v2.close - v1.close, 2) as spread
    FROM pawan_quant_research_1.default.bronze_rt_vix_1d v1
    JOIN pawan_quant_research_1.default.bronze_rt_vix_1d v2
        ON v1.timestamp = v2.timestamp
        AND v1.series = 'VIXCLS'
        AND v2.series = 'vix3m'
    WHERE v1.timestamp >= '2026-02-20'
    ORDER BY v1.timestamp DESC
""").show(truncate=False)

# COMMAND ----------

# ── CONFIRM THE Z-SCORE MATH ───────────────────────────────────────────

import pandas as pd
import numpy as np

# Pull the 1-year VIX history that RT_03 uses for z-score calculation
vix_hist = spark.sql("""
    SELECT timestamp, series, close
    FROM pawan_quant_research_1.default.bronze_rt_vix_1d
    WHERE series IN ('VIXCLS', 'vix3m')
    AND timestamp >= add_months(current_date(), -12)
    ORDER BY timestamp ASC
""").toPandas()

for series in ['VIXCLS', 'vix3m']:
    s = vix_hist[vix_hist.series == series]['close']
    mean = s.mean()
    std  = s.std()
    latest = s.iloc[-1]
    z = (latest - mean) / std
    print(f"{series}:")
    print(f"  1Y mean:  {mean:.2f}")
    print(f"  1Y std:   {std:.2f}")
    print(f"  Latest:   {latest:.2f}  (Mar 4 close)")
    print(f"  Z-score:  {z:.3f}")
    print(f"  Min/Max:  {s.min():.2f} / {s.max():.2f}")
    print()

# COMMAND ----------

print("=== When does silver_bt_master_grid_1h end? ===")
spark.sql("""
    SELECT MAX(timestamp) as last_bt_ts,
           ROUND(LAST(VIXCLS_z), 3) as last_VIXCLS_z,
           ROUND(LAST(vix3m_z), 3)  as last_vix3m_z,
           ROUND(LAST(VIX_Inversion_Z), 3) as last_inv_z
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
""").show()

# COMMAND ----------

# ── CONFIRM THE NORMALIZATION WINDOW MISMATCH ─────────────────────────
spark.sql("""
    SELECT 
        MIN(timestamp) as bt_start,
        MAX(timestamp) as bt_end,
        COUNT(DISTINCT DATE(timestamp)) as trading_days,
        ROUND(AVG(VIXCLS_z), 3) as avg_VIXCLS_z,
        ROUND(STDDEV(VIXCLS_z), 3) as std_VIXCLS_z,
        ROUND(MIN(VIXCLS_z), 3) as min_VIXCLS_z,
        ROUND(MAX(VIXCLS_z), 3) as max_VIXCLS_z
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
""").show()