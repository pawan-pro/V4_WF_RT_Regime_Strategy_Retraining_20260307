# Databricks notebook source
# MAGIC %md
# MAGIC Cell 1: Setup & Libraries
# MAGIC

# COMMAND ----------

# MAGIC %pip install alpaca-py

# COMMAND ----------

# MAGIC %pip install alpaca-py
# MAGIC dbutils.library.restartPython()

import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
import time

# --- CONFIGURATION ---
CATALOG     = "pawan_quant_research_1"
SCHEMA      = "default"
VOLUME      = "market_data"
BT_PREFIX   = "bt"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bronze_{BT_PREFIX}_market_1h"
TABLE_NAME  = f"{CATALOG}.{SCHEMA}.bronze_{BT_PREFIX}_market_1h"

ETFS = ["SPY", "QQQ", "MTUM", "USMV", "IQLT", "EFV", "AGG"]
CREDIT_PROXY = ["HYG"]
ALL_TICKERS = ETFS + CREDIT_PROXY
START_YEAR = 2016 
END_YEAR = 2026

print(f"✅ Hourly Ingestion Environment Ready")

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 2: Authentication

# COMMAND ----------

from alpaca.data.historical import StockHistoricalDataClient

try:
    API_KEY = dbutils.secrets.get(scope="alpaca", key="api_key_id")
    API_SECRET = dbutils.secrets.get(scope="alpaca", key="api_secret_key")
    print("✅ Credentials loaded from Secrets")
except:
    API_KEY = "PKPSF55ZI6V53R25GIBFTUGB4B"
    API_SECRET = "5MnZMUeKU8EFaXFV6dHGUM1uS6kG8AgPXqCnaSjHVE2U"
    print("⚠️  Using Fallback Credentials")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 3: Hourly Ingestion Function

# COMMAND ----------

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def fetch_hourly_data(symbol_list, year_val):
    start = datetime(year_val, 1, 1, tzinfo=ZoneInfo("UTC"))
    end = datetime(year_val, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC"))

    # --- FIX: Cap 'end' time to 16 minutes ago to comply with Free Tier limitations ---
    max_allowed_time = datetime.now(ZoneInfo("UTC")) - timedelta(minutes=16)
    if end > max_allowed_time:
        end = max_allowed_time
        
    # Skip requesting if the year is entirely in the future
    if start > end:
        print(f"      ⏩ Skipping: {year_val} is in the future.")
        return None

    req = StockBarsRequest(
        symbol_or_symbols=symbol_list,
        timeframe=TimeFrame.Hour, 
        start=start,
        end=end
    )
    
    try:
        bars = client.get_stock_bars(req)
        if bars.df is None or bars.df.empty: return None
        return bars.df.reset_index()
    except Exception as e:
        print(f"      ⚠️ API Error: {str(e)}")
        return None

# Define Schema
hourly_schema = StructType([
    StructField("symbol", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
    StructField("trade_count", DoubleType(), True),
    StructField("vwap", DoubleType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 4: The Stable Execution Loop
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 9
print(f"🚀 Starting Hourly Ingestion to {VOLUME_PATH}")

for yr in range(START_YEAR, END_YEAR + 1):
    print(f"📅 Processing {yr}...", end=" ")
    
    pdf = fetch_hourly_data(ALL_TICKERS, yr)
    
    if pdf is not None and not pdf.empty:
        sdf = spark.createDataFrame(pdf, schema=hourly_schema)
        sdf = (sdf
            .withColumn("year_partition", F.year("timestamp").cast("integer"))
            .withColumn("ingestion_ts", F.current_timestamp())
        )
        
        # Save directly to Volume Path
        sdf.write.mode("append").format("delta").save(VOLUME_PATH)
        print(f"✅ {len(pdf)} rows saved.")
    else:
        print("⊘ No data.")

# Create the table from the Delta location
#spark.sql(f"""
#    CREATE TABLE IF NOT EXISTS {TABLE_NAME}
#    AS SELECT * FROM delta.`{VOLUME_PATH}`
#""")
#print("🚀 ALL HOURLY DATA IS LEGIT.")

# COMMAND ----------

# 1. Drop the old disconnected managed table
#spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")
spark.sql(f"DROP VIEW IF EXISTS {TABLE_NAME}")

# 2. Create a dynamic VIEW pointing to the Volume
spark.sql(f"""
    CREATE VIEW {TABLE_NAME}
    AS SELECT * FROM delta.`{VOLUME_PATH}`
""")

print("🚀 VIEW CREATED: ALL HOURLY DATA IS LEGIT AND LINKED.")

# COMMAND ----------

# DBTITLE 1,Cell 10
# MAGIC %sql
# MAGIC -- Quick count to verify the table
# MAGIC SELECT symbol, COUNT(*) as total_rows
# MAGIC FROM pawan_quant_research_1.default.bronze_bt_market_1h
# MAGIC GROUP BY symbol;

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 5: Hourly Legitimacy Audit

# COMMAND ----------

# DBTITLE 1,Cell 12
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   symbol, 
# MAGIC   year_partition, 
# MAGIC   COUNT(*) as total_hours,
# MAGIC   COUNT(DISTINCT(TO_DATE(timestamp))) as trading_days,
# MAGIC   ROUND(COUNT(*) / COUNT(DISTINCT(TO_DATE(timestamp))), 1) as avg_bars_per_day
# MAGIC FROM pawan_quant_research_1.default.bronze_bt_market_1h
# MAGIC GROUP BY 1, 2
# MAGIC ORDER BY 1, 2

# COMMAND ----------

# DBTITLE 1,Cell 14
# Display first row for each symbol (head)
print("📊 First row for each symbol (HEAD):")
head_df = spark.sql("""
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp ASC) as rn
        FROM pawan_quant_research_1.default.bronze_bt_market_1h
    )
    SELECT symbol, timestamp, open, high, low, close, volume, trade_count, vwap, year_partition
    FROM ranked
    WHERE rn = 1
    ORDER BY symbol ASC
""")
display(head_df)

# Display last row for each symbol (tail)
print("\n📊 Last row for each symbol (TAIL):")
tail_df = spark.sql("""
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
        FROM pawan_quant_research_1.default.bronze_bt_market_1h
    )
    SELECT symbol, timestamp, open, high, low, close, volume, trade_count, vwap, year_partition
    FROM ranked
    WHERE rn = 1
    ORDER BY symbol ASC
""")
display(tail_df)

# COMMAND ----------

# DBTITLE 1,Dataset Summary by Symbol
# Comprehensive dataset summary for each symbol
print("📊 DATASET SUMMARY BY SYMBOL\n" + "="*80)

summary_df = spark.sql("""
    SELECT 
        symbol,
        MIN(timestamp) as first_date,
        MAX(timestamp) as last_date,
        COUNT(*) as total_bars,
        COUNT(DISTINCT(TO_DATE(timestamp))) as trading_days,
        ROUND(COUNT(*) / COUNT(DISTINCT(TO_DATE(timestamp))), 1) as avg_bars_per_day,
        ROUND(MIN(close), 2) as min_close,
        ROUND(MAX(close), 2) as max_close,
        ROUND(AVG(close), 2) as avg_close,
        ROUND(SUM(volume), 0) as total_volume,
        ROUND(AVG(volume), 0) as avg_volume
    FROM pawan_quant_research_1.default.bronze_bt_market_1h
    GROUP BY symbol
    ORDER BY symbol ASC
""")

display(summary_df)

# Print text summary
print("\n📈 KEY INSIGHTS:")
for row in summary_df.collect():
    print(f"\n{row.symbol}:")
    print(f"  • Period: {row.first_date} to {row.last_date}")
    print(f"  • Total Bars: {row.total_bars:,} | Trading Days: {row.trading_days:,}")
    print(f"  • Avg Bars/Day: {row.avg_bars_per_day}")
    print(f"  • Price Range: ${row.min_close} - ${row.max_close} (Avg: ${row.avg_close})")
    print(f"  • Total Volume: {row.total_volume:,.0f} | Avg Volume: {row.avg_volume:,.0f}")

# COMMAND ----------

etf_df = spark.sql("""
    SELECT *
    FROM pawan_quant_research_1.default.bronze_bt_market_1h
    WHERE timestamp >= '2024-03-01'
      AND timestamp <= '2024-04-30'
""")

etf_df.coalesce(1).write.option("header", True).mode("overwrite").csv(
    "/Volumes/pawan_quant_research_1/default/market_data/raw_etf_prices_2024-03-01_to_2024-04-30.csv"
)

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load(
    "/Volumes/pawan_quant_research_1/default/market_data/raw_etf_prices_2024-03-01_to_2024-04-30.csv"
)
display(df)


