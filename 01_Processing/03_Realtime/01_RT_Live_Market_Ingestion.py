# Databricks notebook source
# MAGIC %md
# MAGIC Cell 1: Setup & Librabies

# COMMAND ----------

# MAGIC %pip install alpaca-py

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 2: Authentication

# COMMAND ----------

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config

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
runtime_config = load_runtime_instrument_config(spark)
print(
    f"Runtime config: {runtime_config['config_name']} | "
    f"symbols={runtime_config['market_symbols']}"
)

# COMMAND ----------

# MAGIC %pip install nest_asyncio
# MAGIC

# COMMAND ----------

# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS pawan_quant_research_1.v4.bronze_rt_market_1h (
# MAGIC   symbol        STRING,
# MAGIC   timestamp     TIMESTAMP,
# MAGIC   open          DOUBLE,
# MAGIC   high          DOUBLE,
# MAGIC   low           DOUBLE,
# MAGIC   close         DOUBLE,
# MAGIC   volume        DOUBLE,
# MAGIC   source        STRING,
# MAGIC   ingestion_ts  TIMESTAMP
# MAGIC )
# MAGIC USING DELTA;
# MAGIC

# COMMAND ----------

# COMMAND ----------
# RT_01: Hybrid Market Ingestion
# Pre/Post Market: REST snapshot → Delta
# Market Hours:    WebSocket stream → Delta

%pip install alpaca-py

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from pyspark.sql import functions as F
import pandas as pd
from datetime import datetime, timezone

API_KEY = "PKPSF55ZI6V53R25GIBFTUGB4B"
API_SECRET = "5MnZMUeKU8EFaXFV6dHGUM1uS6kG8AgPXqCnaSjHVE2U"
#print("⚠️  Using Fallback Credentials")

client = StockHistoricalDataClient(API_KEY, API_SECRET)
TICKERS = runtime_config["market_symbols"]
TARGET_TABLE = "pawan_quant_research_1.v4.bronze_rt_market_1h"

# ── SNAPSHOT: Works pre-market, post-market, weekends ──
def write_latest_snapshot():
    req = StockLatestBarRequest(symbol_or_symbols=TICKERS)
    bars = client.get_stock_latest_bar(req)
    
    rows = []
    for sym, bar in bars.items():
        rows.append({
            "symbol":    sym,
            "timestamp": bar.timestamp.replace(minute=0, second=0, microsecond=0),
            "open":      float(bar.open),
            "high":      float(bar.high),
            "low":       float(bar.low),
            "close":     float(bar.close),
            "volume":    float(bar.volume),
            "source":    "alpaca_rest_latest",
        })
    
    sdf = spark.createDataFrame(rows) \
               .withColumn("ingestion_ts", F.current_timestamp())
    
    # Upsert: don't double-write same bar
    sdf.createOrReplaceTempView("incoming")
    merge_sql = f"""
        MERGE INTO {TARGET_TABLE} t
        USING incoming s
        ON t.symbol = s.symbol AND t.timestamp = s.timestamp
        WHEN NOT MATCHED THEN INSERT *
    """
    execute_merge_sql_with_retry(
        merge_sql,
        operation_name=f"merge latest market bars into {TARGET_TABLE}",
    )
    print(f"✅ Snapshot written: {rows}")

write_latest_snapshot()

# AUDIT CHECK
spark.sql(f"""
    SELECT symbol, MAX(timestamp) as last_bar, MAX(close) as last_close
    FROM {TARGET_TABLE}
    GROUP BY symbol
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Updated the code block to align with ipynb than .py
