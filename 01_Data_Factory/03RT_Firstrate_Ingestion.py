# Databricks notebook source
# MAGIC %md
# MAGIC Block 1: Setup & Constants

# COMMAND ----------

import zipfile
import os
import shutil
from pyspark.sql import functions as F
from pyspark.sql.types import *

# --- 1. Path Configuration ---
VOLUME_PATH = "/Volumes/pawan_quant_research/default/firstratedata/futures_data/inception_till_10FEB26"
ZIP_FILE = "futures_full_30min_contin_adj_absolute_11assu1.zip"
SOURCE_ZIP = f"{VOLUME_PATH}/{ZIP_FILE}"
EXTRACT_PATH = "/tmp/firstrate_rates_extract"

# Target Unity Catalog Table
CATALOG = "pawan_quant_research_1"
SCHEMA = "default"
# ADD these two lines after SCHEMA
BT_PREFIX  = "bt"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_{BT_PREFIX}_rates_1h"


# --- 2. Exact Filename Mapping (Based on your Debug) ---
# We map the symbol to the exact filename found in the ZIP
REQUIRED_FILES = {
    "ZN": "ZN_full_30min_continuous_absolute_adjusted.txt", # 10Y Note
    "ZT": "ZT_full_30min_continuous_absolute_adjusted.txt", # 2Y Note
    "ZQ": "ZQ_full_30min_continuous_absolute_adjusted.txt"  # Fed Funds
}

# Define FirstRate Intraday Schema
fr_schema = StructType([
    StructField("timestamp_raw", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Block 2: Selective Extraction & Flattening

# COMMAND ----------

import os
import shutil
import zipfile

EXTRACT_PATH = "/tmp/extracted_files"

if os.path.exists(EXTRACT_PATH):
    shutil.rmtree(EXTRACT_PATH)
os.makedirs(EXTRACT_PATH)

with zipfile.ZipFile(SOURCE_ZIP, 'r') as zip_ref:
    all_zip_files = zip_ref.namelist()
    for symbol, full_filename in REQUIRED_FILES.items():
        if full_filename in all_zip_files:
            with zip_ref.open(full_filename) as source, \
                 open(os.path.join(EXTRACT_PATH, f"{symbol}.csv"), "wb") as target:
                shutil.copyfileobj(source, target)
            print(f"✅ Extracted: {symbol}")
        else:
            print(f"❌ ERROR: {full_filename} not found in ZIP.")

print(f"Extraction complete. Files: {os.listdir(EXTRACT_PATH)}")

# COMMAND ----------

# MAGIC %md
# MAGIC Revised Block 3: Processing using Pandas Bridge

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

all_rates_spark_dfs = []

# Map symbols to the extracted files
local_files = {
    "ZN": f"{EXTRACT_PATH}/ZN.csv",
    "ZT": f"{EXTRACT_PATH}/ZT.csv",
    "ZQ": f"{EXTRACT_PATH}/ZQ.csv"
}

for symbol, path in local_files.items():
    print(f"Converting {symbol} to Spark with TZ Correction (Chicago -> UTC)...")
    
    # 1. Read local file using PANDAS (Driver can see /tmp)
    # FirstRate headers are absent, so we provide names manually
    pdf = pd.read_csv(path, header=None, names=["timestamp_raw", "open", "high", "low", "close", "volume"])
    
    # 2. Timezone Correction (CRITICAL UPDATE)
    # Convert string to datetime object
    pdf['dt_obj'] = pd.to_datetime(pdf['timestamp_raw'], format='%Y-%m-%d %H:%M:%S')
    
    # Localize to America/Chicago (Exchange Time) and convert to UTC
    # 'ambiguous=infer' handles the repeated hour during the Fall DST transition
    pdf['timestamp_utc'] = (pdf['dt_obj']
                            .dt.tz_localize('America/Chicago', ambiguous='infer')
                            .dt.tz_convert('UTC'))
    
    # Remove TZ awareness to make it Spark-compatible (naive UTC by convention)
    pdf['timestamp'] = pdf['timestamp_utc'].dt.tz_localize(None)
    
    # Keep only the cleaned columns for transfer to Spark
    pdf_spark_ready = pdf[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # 3. Convert to Spark DataFrame
    df = spark.createDataFrame(pdf_spark_ready)
    
    # 4. Standardize and Resample 30M -> 1H
    # We use the already-UTC-corrected 'timestamp'
    df_1h = df.withColumn("symbol", F.lit(symbol)) \
              .withColumn("hour_ts", F.date_trunc("hour", F.col("timestamp"))) \
              .groupBy("symbol", "hour_ts") \
              .agg(
                  F.first("open").alias("open"),
                  F.max("high").alias("high"),
                  F.min("low").alias("low"),
                  F.last("close").alias("close"),
                  F.sum("volume").alias("volume")
              ) \
              .withColumnRenamed("hour_ts", "timestamp")
    
    all_rates_spark_dfs.append(df_1h)

# --- Combine Data ---
union_df = all_rates_spark_dfs[0]
for next_df in all_rates_spark_dfs[1:]:
    union_df = union_df.union(next_df)

# --- Calculate T10Y2Y Spread ---
# This spread now calculates using ZN and ZT that are aligned to the exact same UTC hour
spread_calc = union_df.filter(F.col("symbol").isin(["ZN", "ZT"])) \
    .groupBy("timestamp") \
    .pivot("symbol") \
    .agg(F.last("close")) \
    .filter(F.col("ZN").isNotNull() & F.col("ZT").isNotNull())

spread_df = spread_calc.withColumn("symbol", F.lit("T10Y2Y_FUT")) \
    .withColumn("close", F.col("ZN") - F.col("ZT")) \
    .withColumn("open", F.col("close")) \
    .withColumn("high", F.col("close")) \
    .withColumn("low", F.col("close")) \
    .withColumn("volume", F.lit(0.0)) \
    .select("symbol", "timestamp", "open", "high", "low", "close", "volume")

# --- Final Bronze DataFrame ---
final_bronze_rates = union_df.select("symbol", "timestamp", "open", "high", "low", "close", "volume") \
                             .union(spread_df) \
                             .withColumn("year_partition", F.year("timestamp")) \
                             .withColumn("ingestion_ts", F.current_timestamp())

print("✅ Final Bronze Rates DataFrame created with UTC timestamps and yield spread.")

# COMMAND ----------

# MAGIC %md
# MAGIC Block 4: Save to Unity Catalog

# COMMAND ----------

# Save to Managed Table
final_bronze_rates.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("symbol", "year_partition") \
    .saveAsTable(TARGET_TABLE)

# Cleanup
shutil.rmtree(EXTRACT_PATH)

print(f"Successfully saved {TARGET_TABLE}")
display(spark.sql(f"SELECT symbol, count(*), min(timestamp), max(timestamp) FROM {TARGET_TABLE} GROUP BY 1"))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Target Table
TARGET_TABLE = f"pawan_quant_research_1.default.bronze_bt_rates_1h"

# 1. Load the table
rates_df = spark.table(TARGET_TABLE)

# 2. Get the Earliest 5 rows per symbol (Head)
window_head = Window.partitionBy("symbol").orderBy("timestamp")
head_df = rates_df.withColumn("rn", F.row_number().over(window_head)) \
                  .filter(F.col("rn") <= 5) \
                  .drop("rn", "year_partition", "ingestion_ts")

# 3. Get the Latest 5 rows per symbol (Tail)
window_tail = Window.partitionBy("symbol").orderBy(F.col("timestamp").desc())
tail_df = rates_df.withColumn("rn", F.row_number().over(window_tail)) \
                  .filter(F.col("rn") <= 5) \
                  .drop("rn", "year_partition", "ingestion_ts")

print("--- EARLIEST 5 BARS PER SYMBOL (HEAD) ---")
display(head_df.orderBy("symbol", "timestamp"))

print("--- LATEST 5 BARS PER SYMBOL (TAIL) ---")
display(tail_df.orderBy("symbol", "timestamp"))

# COMMAND ----------

# MAGIC %md
# MAGIC # 03: FirstRate Rates Ingestion (Bronze Layer)
# MAGIC
# MAGIC ### **Objective**
# MAGIC Capture intraday (1-Hour) interest rate dynamics and Federal Reserve expectations to support the **V4 Event-Driven Regime Strategy**. By moving from daily macro snapshots (V3) to hourly futures data (V4), we aim to eliminate "Friday Close Bias" and react to market-moving yield shifts in real-time.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **1. Source Metadata**
# MAGIC *   **Provider:** FirstRate Data (Futures Historical Bundle)
# MAGIC *   **Source File:** `futures_full_30min_contin_adj_absolute_11assu1.zip`
# MAGIC *   **UC Volume Path:** `/Volumes/pawan_quant_research/default/firstratedata/futures_data/inception_till_10FEB26/`
# MAGIC *   **Raw Resolution:** 30-Minute Intraday Bars
# MAGIC *   **Original Timezone:** US Eastern Time (EST/EDT)
# MAGIC *   **Adjustment Type:** Absolute Adjusted (Neutralizes price jumps on contract roll dates)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **2. Target Data Definitions**
# MAGIC *   **Database:** `pawan_quant_research_1.default`
# MAGIC *   **Table Name:** `bronze_rates_1h`
# MAGIC *   **Target Resolution:** 1-Hour OHLCV (UTC)
# MAGIC *   **Included Symbols:**
# MAGIC     *   **ZN:** 10-Year Treasury Note Futures (Proxy for long-term rates)
# MAGIC     *   **ZT:** 2-Year Treasury Note Futures (Proxy for short-term rates)
# MAGIC     *   **ZQ:** 30-Day Fed Funds Futures (Proxy for implied Fed path/expectations)
# MAGIC     *   **T10Y2Y_FUT:** Derived Spread (`ZN - ZT`) representing yield curve slope.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **3. Transformation Pipeline (Bronze Layer)**
# MAGIC 1.  **Selective Extraction:** Extracting specific `.txt` files from the FirstRate ZIP directly to driver local storage (`/tmp/`).
# MAGIC 2.  **Pandas-to-Spark Bridge:** Reading local files via Pandas to bypass Databricks Serverless filesystem restrictions.
# MAGIC 3.  **Timezone Normalization:** Converting source timestamps from US Eastern to **UTC** for perfect alignment with Alpaca market data.
# MAGIC 4.  **Resampling:** Aggregating 30-minute ticks into 1-Hour bars using OHLCV logic.
# MAGIC 5.  **Derived Calculation:** Computing the `T10Y2Y_FUT` spread at every hourly interval.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **4. Downstream Usage (Context for Silver Layer)**
# MAGIC *   **Feature Engineering:** These rates will be joined with `bronze_market_1h` using an exact timestamp match.
# MAGIC *   **Regime Detection:** Used as high-frequency inputs for XGBoost to detect intra-week economic "Slowdown" or "Expansion" shifts.
# MAGIC *   **Partitioning:** Table is partitioned by `symbol` and `year_partition` for optimized query performance.
# MAGIC
# MAGIC ---
# MAGIC **Last Updated:** February 15, 2026  
# MAGIC **Status:** ✅ Bronze Table Successfully Populated (100k+ bars per symbol)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Specification Note: FirstRate Futures (Rates Ingestion)
# MAGIC
# MAGIC ### 1. Selection Rationale: Absolute Adjusted
# MAGIC For the V4 Regime Strategy, we utilize the **Absolute Adjusted** continuous series. 
# MAGIC *   **Why:** Unlike "Unadjusted" data, it removes the artificial price gaps (jumps) caused by quarterly contract rollovers.
# MAGIC *   **Spread Integrity:** Unlike "Ratio Adjusted" data (which is optimized for percentage returns), **Absolute Adjustment** maintains the integrity of absolute price differences. This is critical because our primary feature, `T10Y2Y_FUT`, relies on the subtraction of two price series (`ZN - ZT`). Absolute adjustment ensures the yield curve proxy remains mathematically consistent across years of historical rolls.
# MAGIC
# MAGIC ### 2. Timezone & Timestamp Logic
# MAGIC *   **Source Timezone:** US Eastern Time (EST/EDT). 
# MAGIC *   **Normalization Requirement:** To align with Alpaca Market data (which is UTC), all timestamps must be shifted. 
# MAGIC *   **Bar Alignment:** FirstRate timestamps represent the **start of the period**. An hourly bar timestamped `09:00:00` contains data from `09:00:00` to `09:59:59`.
# MAGIC *   **Roll-Over Timing:** Intraday contracts are rolled at **midnight (00:00) EST**.
# MAGIC
# MAGIC ### 3. Data Gaps & Volume
# MAGIC *   **Volume Filter:** Only bars with recorded trading volume are included in the source files. 
# MAGIC *   **Handling Gaps:** Bars with zero volume are omitted. During Silver Layer construction, **Forward Filling** must be used to ensure the Master Time Grid has continuous rate data even during illiquid overnight hours.
# MAGIC
# MAGIC ### 4. Mathematical Conversions (For Feature Engineering)
# MAGIC When moving from V3 (FRED Yields) to V4 (FirstRate Futures Prices), the following relationships apply:
# MAGIC *   **The Sign Flip (Inverse Relationship):** Futures prices move **inversely** to yields. 
# MAGIC     *   *Yield Up (Hawkish) = Futures Price Down.*
# MAGIC     *   *Yield Down (Dovish) = Futures Price Up.*
# MAGIC *   **Fed Path (ZQ Proxy):** 30-Day Fed Funds Futures (`ZQ`) follow a specific pricing convention:
# MAGIC     *   `Implied Fed Rate = 100 - Price`
# MAGIC     *   *Example: A ZQ price of 94.50 implies a market expectation of a 5.50% Fed Funds Rate.*
# MAGIC *   **Yield Curve Slope:** The `T10Y2Y_FUT` spread (`ZN - ZT`) measures the slope of the curve. In price terms, a narrowing spread (flattening curve) indicates shifting economic regimes.
# MAGIC
# MAGIC ### 5. Data Format (Bronze Layer)
# MAGIC *   **Schema:** `{DateTime, Open, High, Low, Close, Volume}`
# MAGIC *   **File Extension:** `.txt` (treated as CSV-formatted).
# MAGIC *   **Storage:** Partitioned by `symbol` and `year_partition` in `bronze_rates_1h` for optimized retrieval during the Silver Join.
# MAGIC
# MAGIC ---
# MAGIC **Application:** These properties allow the V4 model to ingest high-frequency "Interest Rate State" data alongside "Market Price State," providing a synchronized view of macro-financial regimes at an hourly resolution.

# COMMAND ----------

rates_df = spark.table("pawan_quant_research_1.default.bronze_bt_rates_1h").filter(
    (F.col("timestamp") >= "2024-03-01") &
    (F.col("timestamp") <= "2024-04-30")
)

display(rates_df.orderBy("symbol", "timestamp"))