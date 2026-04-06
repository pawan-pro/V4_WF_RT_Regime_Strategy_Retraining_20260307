# Databricks notebook source
# RT_03: Silver RT Enrichment
# Produces one inference-ready row matching silver_bt_master_grid_1h schema

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
import pandas as pd
import numpy as np
from datetime import date

TARGET_TABLE  = "pawan_quant_research_1.v4.silver_rt_1h"
SILVER_HIST   = "pawan_quant_research_1.v4.silver_bt_master_grid_1h"
BRONZE_MARKET = "pawan_quant_research_1.v4.bronze_rt_market_1h"
BRONZE_VIX    = "pawan_quant_research_1.v4.bronze_rt_vix_1d"

# ── CREATE TABLE IF NOT EXISTS ──
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        timestamp         TIMESTAMP,
        SPY_close         DOUBLE,
        CPIAUCSL_z        DOUBLE,
        PCEPI_z           DOUBLE,
        PCE_z             DOUBLE,
        PAYEMS_z          DOUBLE,
        UNRATE_z          DOUBLE,
        GDPC1_z           DOUBLE,
        INDPRO_z          DOUBLE,
        UMCSENT_z         DOUBLE,
        VIXCLS_z          DOUBLE,
        vix3m_z           DOUBLE,
        VIX_Inversion_Z   DOUBLE,
        Price_Mom_24H     DOUBLE,
        CPI_Event_Active  INT,
        NFP_Event_Active  INT,
        source            STRING
    )
    USING DELTA
""")

# ── STEP 1: Get latest RT prices from bronze_rt_market_1h ──
spy_row = spark.sql(f"""
    SELECT timestamp, close FROM {BRONZE_MARKET}
    WHERE symbol = 'SPY'
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]

spy_close = float(spy_row["close"])
rt_ts     = spy_row["timestamp"].replace(minute=0, second=0, microsecond=0)

# VIX from RT_02 FRED feed (bronze_rt_vix_1d — series column)
vix_close   = spark.sql(f"""
    SELECT close FROM {BRONZE_VIX}
    WHERE series = 'VIXCLS'
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]["close"]

vix3m_close = spark.sql(f"""
    SELECT close FROM {BRONZE_VIX}
    WHERE series = 'vix3m'
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]["close"]

# ── BIAS GATE ──
vix_latest_date = spark.sql(f"""
    SELECT MAX(timestamp) as ts FROM {BRONZE_VIX} WHERE series = 'VIXCLS'
""").collect()[0]["ts"]
assert vix_latest_date.date() < date.today(), \
    f"🚨 LOOKAHEAD BIAS: VIX date {vix_latest_date.date()} >= today {date.today()}"

print(f"RT inputs → ts:{rt_ts}  SPY:{spy_close}  VIX:{vix_close}  VIX3M:{vix3m_close}")
print(f"✅ Bias gate passed | VIX date: {vix_latest_date.date()}")

# ── STEP 2: Macro z-scores — carry forward from BT silver history ──
hist_pdf = spark.sql(f"""
    SELECT timestamp, CPIAUCSL_z, PCEPI_z, PCE_z, PAYEMS_z,
           UNRATE_z, GDPC1_z, INDPRO_z, UMCSENT_z,
           CPI_Event_Active, NFP_Event_Active, SPY_close
    FROM {SILVER_HIST}
    ORDER BY timestamp DESC
    LIMIT 8760
""").toPandas().sort_values("timestamp")

last_macro = hist_pdf.iloc[-1]

# ── STEP 3: VIX z-scores — anchor to BT bronze history (correct column names) ──
# bronze_bt_vix_1h (fred_1d): symbol='VX', value in 'close' column
# bronze_bt_vix3m_1h (fred_1d portara path): value in 'vix3m' column
# ── STEP 3: VIX z-scores — use FULL historical silver_bt_master_grid_1h (same as backtest) ──
# ── STEP 3: VIX z-scores — carry forward latest from historical silver grid (same as backtest) ──
latest_hist_vix = spark.sql(f"""
    SELECT timestamp, VIXCLS_z, vix3m_z, VIX_Inversion_Z
    FROM {SILVER_HIST}
    WHERE timestamp <= '{rt_ts}'
    ORDER BY timestamp DESC
    LIMIT 1
""").collect()[0]

vixcls_z_rt   = float(latest_hist_vix["VIXCLS_z"])
vix3m_z_rt    = float(latest_hist_vix["vix3m_z"])
vix_inv_z_rt  = float(latest_hist_vix["VIX_Inversion_Z"])

print(f"  VIXCLS_z (from silver history): {vixcls_z_rt:.4f}")
print(f"  vix3m_z  (from silver history): {vix3m_z_rt:.4f}")
print(f"  Inversion z (from silver history): {vix_inv_z_rt:.4f}")

# ── STEhttps://dbc-f34f42c1-2ab7.cloud.databricks.com/editor/notebooks/4311110377742948?o=7474660273899120$0P 4: Price momentum (lag-safe) ──
spy_lag1  = float(hist_pdf["SPY_close"].iloc[-1])
spy_lag25 = float(hist_pdf["SPY_close"].iloc[-25])
price_mom = (spy_lag1 / spy_lag25) - 1

# ── STEP 5: Assemble inference row ──
def to_py_float(v):
    return float(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None

rt_row = {
    "timestamp":        rt_ts,
    "SPY_close":        to_py_float(spy_close),
    "CPIAUCSL_z":       to_py_float(last_macro["CPIAUCSL_z"]),
    "PCEPI_z":          to_py_float(last_macro["PCEPI_z"]),
    "PCE_z":            to_py_float(last_macro["PCE_z"]),
    "PAYEMS_z":         to_py_float(last_macro["PAYEMS_z"]),
    "UNRATE_z":         to_py_float(last_macro["UNRATE_z"]),
    "GDPC1_z":          to_py_float(last_macro["GDPC1_z"]),
    "INDPRO_z":         to_py_float(last_macro["INDPRO_z"]),
    "UMCSENT_z":        to_py_float(last_macro["UMCSENT_z"]),
    "VIXCLS_z":         to_py_float(vixcls_z_rt),
    "vix3m_z":          to_py_float(vix3m_z_rt),
    "VIX_Inversion_Z":  to_py_float(vix_inv_z_rt),
    "Price_Mom_24H":    to_py_float(price_mom),
    "CPI_Event_Active": int(last_macro["CPI_Event_Active"]),
    "NFP_Event_Active": int(last_macro["NFP_Event_Active"]),
    "source":           "rt_enrichment",
}

# ── STEP 6: Write to silver_rt_1h ──
schema = StructType([
    StructField("timestamp",        TimestampType(), True),
    StructField("SPY_close",        DoubleType(),    True),
    StructField("CPIAUCSL_z",       DoubleType(),    True),
    StructField("PCEPI_z",          DoubleType(),    True),
    StructField("PCE_z",            DoubleType(),    True),
    StructField("PAYEMS_z",         DoubleType(),    True),
    StructField("UNRATE_z",         DoubleType(),    True),
    StructField("GDPC1_z",          DoubleType(),    True),
    StructField("INDPRO_z",         DoubleType(),    True),
    StructField("UMCSENT_z",        DoubleType(),    True),
    StructField("VIXCLS_z",         DoubleType(),    True),
    StructField("vix3m_z",          DoubleType(),    True),
    StructField("VIX_Inversion_Z",  DoubleType(),    True),
    StructField("Price_Mom_24H",    DoubleType(),    True),
    StructField("CPI_Event_Active", IntegerType(),   True),
    StructField("NFP_Event_Active", IntegerType(),   True),
    StructField("source",           StringType(),    True),
])

sdf = spark.createDataFrame([rt_row], schema=schema)
sdf.createOrReplaceTempView("incoming_silver_rt")
spark.sql(f"""
    MERGE INTO {TARGET_TABLE} t
    USING incoming_silver_rt s
    ON t.timestamp = s.timestamp
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"✅ RT Silver row written: {rt_ts}")

# ── AUDIT ──
print("\nAudit — RT Silver row:")
spark.sql(f"SELECT * FROM {TARGET_TABLE} ORDER BY timestamp DESC LIMIT 1").show(vertical=True)
print(f"\n  VIXCLS_z:  {vixcls_z_rt:.3f}  (expected ~0 if normal vol)")
print(f"  vix3m_z:   {vix3m_z_rt:.3f}")
print(f"  Inversion: {vix_inv_z_rt:.3f}")
print(f"  PriceMom:  {price_mom:.4f}")

# COMMAND ----------

import requests

WORKSPACE = "dbc-f34f42c1-2ab7.cloud.databricks.com"  # from your notebook URL
WAREHOUSE_ID = "da771f9b05a5fc84"
TOKEN = "dapid8a209f7fdef384e865b3bd03dca8e41"

resp = requests.post(
    f"https://{WORKSPACE}/api/2.0/sql/statements",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json={
        "warehouse_id": WAREHOUSE_ID,
        "statement": "SELECT * FROM pawan_quant_research_1.v4.silver_rt_1h ORDER BY timestamp DESC LIMIT 1",
        "wait_timeout": "10s"
    }
)
print(resp.status_code)
print(resp.json())



# COMMAND ----------

# ── DASHBOARD CROSS-CHECK ──────────────────────────────────────────────
# Replicates exactly what the dashboard displays — run after RT_03

import pandas as pd

FEATURES = [
    ("VIXCLS_z",        "VIX Spot",        0.127),
    ("vix3m_z",         "VIX 3M",          0.101),
    ("INDPRO_z",        "Ind. Production", 0.091),
    ("UNRATE_z",        "Unemployment",    0.087),
    ("GDPC1_z",         "Real GDP",        0.083),
    ("PAYEMS_z",        "Payrolls",        0.079),
    ("CPIAUCSL_z",      "CPI",             0.078),
    ("PCE_z",           "PCE",             0.075),
    ("UMCSENT_z",       "Consumer Sent.",  0.073),
    ("VIX_Inversion_Z", "VIX Inversion",   0.070),
    ("PCEPI_z",         "PCE Deflator",    0.064),
    ("CPI_Event_Active","CPI Event",       0.031),
    ("NFP_Event_Active","NFP Event",       0.026),
    ("Price_Mom_24H",   "Price Mom 24H",   0.020),
]

# Pull latest row
row = spark.sql("""
    SELECT * FROM pawan_quant_research_1.v4.silver_rt_1h
    ORDER BY timestamp DESC LIMIT 1
""").toPandas().iloc[0]

print(f"\n{'='*65}")
print(f"  DASHBOARD CROSS-CHECK  |  {row['timestamp']}  |  SPY: {row['SPY_close']:.2f}")
print(f"{'='*65}")
print(f"  {'Feature':<20} {'Key':<20} {'Z-Score':>8}  {'Status':<10}  Bar")
print(f"  {'-'*60}")

def status(z):
    a = abs(z)
    if a >= 2.5: return "EXTREME 🔴"
    if a >= 2.0: return "ELEVATED 🟠"
    if a >= 1.5: return "WARNING  🟡"
    if a >= 1.0: return "WATCH    🟢"
    return          "NORMAL   ✅"

def bar(z, width=20):
    center = width // 2
    pos = int(round(max(-3, min(3, z)) / 3 * center))
    b = ['-'] * width
    b[center] = '|'
    idx = center + pos
    if 0 <= idx < width:
        b[idx] = '█'
    return ''.join(b)

stress = 0
extreme_count = 0
for key, label, imp in FEATURES:
    v = row[key]
    if key.endswith("_Active"):
        print(f"  {label:<20} {key:<20} {'ACTIVE' if v else 'INACTIVE':>8}  {'EVENT' if v else 'CLEAR':<10}")
        continue
    z = float(v)
    s = status(z)
    stress += imp * min(abs(z), 3)
    if abs(z) >= 2: extreme_count += 1
    print(f"  {label:<20} {key:<20} {z:>+8.3f}  {s:<14}  {bar(z)}")

max_stress = sum(imp for k, _, imp in FEATURES if not k.endswith("_Active")) * 3
stress_score = int(stress / max_stress * 100)

print(f"\n{'='*65}")
print(f"  REGIME STRESS INDEX : {stress_score}/100")
print(f"  Z > 2σ count        : {extreme_count}/12")
print(f"  Source              : {row['source']}")
print(f"{'='*65}\n")

# COMMAND ----------

# ── DISCREPANCY AUDIT ──────────────────────────────────────────────────

print("=== silver_rt_1h (what RT_03 wrote) ===")
spark.sql("""
    SELECT timestamp, PAYEMS_z, VIXCLS_z, UNRATE_z, source
    FROM pawan_quant_research_1.v4.silver_rt_1h
    ORDER BY timestamp DESC LIMIT 3
""").show(vertical=True)

print("=== gold_rt_dashboard_feed (what RT_04 used) ===")
spark.sql("""
    SELECT signal_ts, PAYEMS_z, VIXCLS_z, regime_name, source
    FROM pawan_quant_research_1.v4.gold_rt_dashboard_feed
    ORDER BY feed_ts DESC LIMIT 3
""").show(vertical=True)

print("=== bronze_rt_regime_signal (RT_04 direct output) ===")
spark.sql("""
    SELECT timestamp, PAYEMS_z, UNRATE_z, regime_name, position
    FROM pawan_quant_research_1.v4.bronze_rt_regime_signal
    ORDER BY timestamp DESC LIMIT 3
""").show(vertical=True)

print("=== silver_bt_master_grid_1h (macro source — last macro row) ===")
spark.sql("""
    SELECT timestamp, PAYEMS_z, CPIAUCSL_z, UNRATE_z
    FROM pawan_quant_research_1.v4.silver_bt_master_grid_1h
    ORDER BY timestamp DESC LIMIT 3
""").show(vertical=True)

# COMMAND ----------

# ── ROOT CAUSE AUDIT ───────────────────────────────────────────────────

print("=== RT_04 actual schema (what columns exist) ===")
spark.sql("""
    DESCRIBE pawan_quant_research_1.v4.bronze_rt_regime_signal
""").show()

print("=== silver_bt_master_grid_1h — last macro row ===")
spark.sql("""
    SELECT timestamp, PAYEMS_z, UNRATE_z, CPIAUCSL_z, GDPC1_z
    FROM pawan_quant_research_1.v4.silver_bt_master_grid_1h
    ORDER BY timestamp DESC LIMIT 1
""").show(vertical=True)

print("=== gold feed PAYEMS_z vs silver PAYEMS_z ===")
spark.sql("""
    SELECT 
        'gold_feed'  as source, CAST(PAYEMS_z as STRING) as PAYEMS_z, signal_ts as ts
    FROM pawan_quant_research_1.v4.gold_rt_dashboard_feed
    ORDER BY feed_ts DESC LIMIT 1
""").show()

print("=== RT_04 FEATURES list vs silver_rt_1h columns ===")
# This shows if RT_04 is reading the RIGHT column for PAYEMS
spark.sql("""
    SELECT timestamp, PAYEMS_z, UNRATE_z
    FROM pawan_quant_research_1.v4.silver_rt_1h
    ORDER BY timestamp DESC LIMIT 1
""").show(vertical=True)