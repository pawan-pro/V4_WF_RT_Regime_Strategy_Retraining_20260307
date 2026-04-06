# Databricks notebook source
# MAGIC %md
# MAGIC Block 1: Readme.md (Context Preservation)

# COMMAND ----------

# MAGIC %md
# MAGIC # 04: VIX Ingestion (Bronze Layer) — BT Namespace
# MAGIC
# MAGIC ### **Objective**
# MAGIC Ingest VIX data for backtesting. Supports two sources via `VIX_SOURCE` toggle:
# MAGIC - `portara_1h` — Portara 1-Minute futures resampled to 1H (original)
# MAGIC - `fred_1d` — FRED VIXCLS/VXVCLS daily spot, PIT-corrected, expanded to hourly grid
# MAGIC
# MAGIC Both paths write to the **BT-namespaced** tables only. Shared tables are never touched.
# MAGIC
# MAGIC ---
# MAGIC ### **Target Tables**
# MAGIC | Source | VIX Table | VIX3M Table |
# MAGIC |---|---|---|
# MAGIC | `portara_1h` | `bronze_bt_vix_1h` | `bronze_bt_vix3m_1h` |
# MAGIC | `fred_1d` | `bronze_bt_vix_1h` | `bronze_bt_vix3m_1h` |
# MAGIC
# MAGIC Same target either way — swap source, rerun, compare downstream.

# COMMAND ----------

# MAGIC %md
# MAGIC Block 2 — Setup & Configuration (full replacement)

# COMMAND ----------

import zipfile
import os
import shutil
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *

# ─────────────────────────────────────────────
# CONFIGURATION — only edit here
# ─────────────────────────────────────────────
CATALOG    = "pawan_quant_research_1"
SCHEMA     = "v4"
BT_PREFIX  = "bt"
VIX_SOURCE = "fred_1d"   # OPTIONS: "portara_1h" | "fred_1d"

TARGET_VIX   = f"{CATALOG}.{SCHEMA}.bronze_{BT_PREFIX}_vix_1h"
TARGET_VIX3M = f"{CATALOG}.{SCHEMA}.bronze_{BT_PREFIX}_vix3m_1h"

# Portara source (read-only — never renamed)
PORTARA_VOLUME = "/Volumes/pawan_quant_research/default/portaradata/vix_data"
ZIP_FILE       = "Avik VX.zip"
SOURCE_ZIP     = f"{PORTARA_VOLUME}/{ZIP_FILE}"
EXTRACT_PATH   = "/tmp/portara_vix_extract"
COL_NAMES      = ["date_raw", "time_raw", "open", "high", "low", "close", "volume", "contract_name"]

# FRED API
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

print(f"✅ Config ready | VIX_SOURCE={VIX_SOURCE} | Targets: {TARGET_VIX}, {TARGET_VIX3M}")

# COMMAND ----------

# MAGIC %md
# MAGIC Block 3 — Branch A: Portara 1H Ingestion

# COMMAND ----------

if VIX_SOURCE == "portara_1h":

    # --- Extract ZIP ---
    if os.path.exists(EXTRACT_PATH):
        shutil.rmtree(EXTRACT_PATH)
    os.makedirs(EXTRACT_PATH)

    print(f"Unzipping from {SOURCE_ZIP}...")
    with zipfile.ZipFile(SOURCE_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    all_extracted_files = []
    for root, dirs, files in os.walk(EXTRACT_PATH):
        for f in files:
            if f.lower().endswith(('.txt', '.csv')):
                all_extracted_files.append(os.path.join(root, f))

    if not all_extracted_files:
        raise ValueError("No data files found in ZIP.")

    data_file_path = max(all_extracted_files, key=os.path.getsize)
    print(f"🎯 Processing: {data_file_path}")

    # --- Parse & TZ convert ---
    pdf = pd.read_csv(data_file_path, header=None, names=COL_NAMES)
    pdf['dt_str']   = pdf['date_raw'].astype(str) + pdf['time_raw'].astype(str).str.zfill(4)
    pdf['ts_local'] = pd.to_datetime(pdf['dt_str'], format='%Y%m%d%H%M')
    pdf['timestamp_utc'] = (pdf['ts_local']
                            .dt.tz_localize('US/Central', ambiguous='infer')
                            .dt.tz_convert('UTC'))
    pdf['timestamp'] = pdf['timestamp_utc'].dt.tz_localize(None)
    pdf_final = pdf[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'contract_name']].copy()
    print(f"✅ Parsed {len(pdf_final):,} 1-minute rows.")

    # --- Resample 1min → 1H ---
    vix_1m_spark = spark.createDataFrame(pdf_final)
    vix_1h_df = (vix_1m_spark
        .withColumn("hour_ts", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("hour_ts")
        .agg(
            F.first("open").alias("open"),
            F.max("high").alias("high"),
            F.min("low").alias("low"),
            F.last("close").alias("close"),
            F.sum("volume").alias("volume"),
            F.last("contract_name").alias("source_contract")
        )
        .withColumnRenamed("hour_ts", "timestamp")
        .withColumn("symbol", F.lit("VX"))
        .withColumn("source", F.lit("portara_1h"))
        .withColumn("year_partition", F.year("timestamp"))
        .withColumn("ingestion_ts", F.current_timestamp())
    )

    vix_1h_df.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .partitionBy("symbol", "year_partition") \
        .saveAsTable(TARGET_VIX)

    # Cleanup
    shutil.rmtree(EXTRACT_PATH)
    del pdf, pdf_final

    print(f"✅ [portara_1h] Written → {TARGET_VIX}")

else:
    print("⏭️ Skipping Portara block (VIX_SOURCE != portara_1h)")

# COMMAND ----------

# MAGIC %md
# MAGIC Block 4 — Branch B: FRED 1D PIT-Corrected Ingestion

# COMMAND ----------

if VIX_SOURCE == "fred_1d":

    %pip install fredapi
    from fredapi import Fred

    fred = Fred(api_key=FRED_API_KEY)

    def fred_pit_to_hourly(series_id: str, symbol_label: str) -> pd.DataFrame:
        """
        PIT correction: value published on day T is assigned to T+1 market open.
        shift(1) ensures no same-day close is ever seen at inference time.
        Expanded to 24 hourly rows per day so the downstream 1H silver join works
        identically to the portara_1h path — no changes needed in 05RT.
        """
        s   = fred.get_series(series_id, observation_start="2016-01-01").dropna()
        pdf = s.reset_index()
        pdf.columns = ["date", "close"]
        pdf["date"]  = pd.to_datetime(pdf["date"]).dt.tz_localize(None)

        # ── BIAS GUARD: shift +1 day (T's value available only on T+1) ──
        pdf["close"] = pdf["close"].shift(1)
        pdf = pdf.dropna()

        records = []
        for _, row in pdf.iterrows():
            day = pd.Timestamp(row["date"]).normalize()
            for hour in range(24):
                records.append({
                    "timestamp":      day + pd.Timedelta(hours=hour),
                    "open":           float(row["close"]),
                    "high":           float(row["close"]),
                    "low":            float(row["close"]),
                    "close":          float(row["close"]),
                    "volume":         0.0,
                    "source_contract": series_id,
                    "symbol":         symbol_label,
                    "source":         f"fred_{series_id.lower()}_pit_1d",
                })

        df_out = pd.DataFrame(records)
        df_out["year_partition"] = df_out["timestamp"].dt.year
        df_out["ingestion_ts"]   = pd.Timestamp.utcnow()
        return df_out

    # --- Fetch both series ---
    print("Fetching VIXCLS (PIT-corrected)...")
    vix_pdf  = fred_pit_to_hourly("VIXCLS", "VX")
    print(f"✅ {len(vix_pdf):,} rows | {vix_pdf['timestamp'].min().date()} → {vix_pdf['timestamp'].max().date()}")

    print("Fetching VXVCLS (PIT-corrected)...")
    vix3m_pdf = fred_pit_to_hourly("VXVCLS", "VIX3M")
    print(f"✅ {len(vix3m_pdf):,} rows | {vix3m_pdf['timestamp'].min().date()} → {vix3m_pdf['timestamp'].max().date()}")

    # --- Write VIX spot ---
    (spark.createDataFrame(vix_pdf)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("symbol", "year_partition")
        .saveAsTable(TARGET_VIX))
    print(f"✅ [fred_1d] Written → {TARGET_VIX}")

    # --- Write VIX3M spot ---
    (spark.createDataFrame(vix3m_pdf)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("symbol", "year_partition")
        .saveAsTable(TARGET_VIX3M))
    print(f"✅ [fred_1d] Written → {TARGET_VIX3M}")

    del vix_pdf, vix3m_pdf

else:
    print("⏭️ Skipping FRED block (VIX_SOURCE != fred_1d)")

# COMMAND ----------

# MAGIC %md
# MAGIC Block 5 — VIX3M Ingest for Portara path (only runs when portara_1h)

# COMMAND ----------

if VIX_SOURCE == "portara_1h":
    # FRED VIX3M still used as the 3M spot term — no Portara equivalent
    %pip install fredapi
    from fredapi import Fred

    fred    = Fred(api_key=FRED_API_KEY)
    s       = fred.get_series("VXVCLS", observation_start="2016-01-01").dropna()
    vix3m_pdf = s.reset_index()
    vix3m_pdf.columns = ["timestamp", "vix3m"]
    vix3m_pdf["timestamp"] = pd.to_datetime(vix3m_pdf["timestamp"]).dt.tz_localize(None)

    (spark.createDataFrame(vix3m_pdf)
        .withColumn("symbol",         F.lit("VIX3M"))
        .withColumn("source",         F.lit("fred_vxvcls"))
        .withColumn("year_partition", F.year("timestamp"))
        .withColumn("ingestion_ts",   F.current_timestamp())
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("symbol", "year_partition")
        .saveAsTable(TARGET_VIX3M))

    print(f"✅ [portara_1h] VIX3M written → {TARGET_VIX3M}")

else:
    print("⏭️ Skipping portara VIX3M block (already handled by fred_1d branch)")

# COMMAND ----------

spark.sql("""
    SELECT symbol, COUNT(*) as rows,
           MIN(timestamp) as first, MAX(timestamp) as last
    FROM pawan_quant_research_1.v4.bronze_bt_vix3m_1h
    GROUP BY symbol
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Block 6 — PIT Audit (runs regardless of source)

# COMMAND ----------

from datetime import date

# ── BIAS GUARD 1: Latest VIX date must be strictly before today ──
vix_latest = spark.sql(f"SELECT MAX(timestamp) as ts FROM {TARGET_VIX}").collect()[0]["ts"]
assert vix_latest.date() < date.today(), \
    f"🚨 LOOKAHEAD BIAS DETECTED: VIX date {vix_latest.date()} >= today {date.today()}"

# ── BIAS GUARD 2: Namespace check — must never point at shared table ──
assert "bt" in TARGET_VIX, \
    f"🚨 NAMESPACE VIOLATION: {TARGET_VIX} is not a BT-namespaced table"

print(f"✅ Bias gate passed | source={VIX_SOURCE} | latest VIX bar: {vix_latest.date()}")

# ── PIT spot-check on COVID stress dates ──
spark.sql(f"""
    SELECT DATE(timestamp) as date, MIN(close) as vix_assigned
    FROM {TARGET_VIX}
    WHERE DATE(timestamp) IN ('2020-03-16','2020-03-17','2020-03-18')
    GROUP BY 1 ORDER BY 1
""").show()
# Expected (fred_1d PIT):
#   2020-03-16 → 57.83  (Friday 13th close, assigned Monday)
#   2020-03-17 → 82.69  (Monday close, assigned Tuesday)
#   2020-03-18 → 75.91  (Tuesday close, assigned Wednesday)
# Portara will show futures prices — different values, not a bug

# COMMAND ----------

# MAGIC %md
# MAGIC Block 7 — Preview audit (replaces old preview block)

# COMMAND ----------

# Replace Block 7 with this schema-aware version:
from pyspark.sql.window import Window

for label, tbl in {"VIX": TARGET_VIX, "VIX3M": TARGET_VIX3M}.items():
    print(f"\n{'='*60}")
    print(f"📊 {label} → {tbl}")
    print(f"{'='*60}")
    df_preview = spark.table(tbl)
    cols = df_preview.columns

    # Determine correct value column based on schema
    val_col = "close" if "close" in cols else "vix3m"

    print("EARLIEST 5:")
    display(df_preview.orderBy("timestamp").limit(5))
    print("LATEST 5:")
    display(df_preview.orderBy(F.col("timestamp").desc()).limit(5))
    print("SUMMARY:")
    display(df_preview.groupBy("symbol").agg(
        F.count("*").alias("rows"),
        F.min("timestamp").alias("first"),
        F.max("timestamp").alias("last"),
        F.round(F.avg(F.col(val_col)), 2).alias("avg_value")
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC To run Portara: VIX_SOURCE = "portara_1h" → blocks 3 + 5 execute, block 4 skips.
# MAGIC To run FRED 1D: VIX_SOURCE = "fred_1d" → block 4 executes, blocks 3 + 5 skip.

# COMMAND ----------

# MAGIC %md
# MAGIC To run Portara: 
# MAGIC
# MAGIC VIX_SOURCE = "portara_1h" → 
# MAGIC blocks 3 + 5 execute, block 4 skips.
# MAGIC
# MAGIC To run FRED 1D: 
# MAGIC VIX_SOURCE = "fred_1d" → block 4 executes, blocks 3 + 5 skip.
# MAGIC
