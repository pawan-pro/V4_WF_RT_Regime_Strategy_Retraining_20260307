# Databricks notebook source
# SERVERLESS-SAFE SETUP: Path-based storage only (no metastore operations)
%pip install --upgrade alpaca-py
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from datetime import datetime
from pyspark.sql import functions as F
import time

# 1. PATHS (Verified from your Catalog Explorer screenshot)
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
VOLUME = "market_data"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bronze_market_1m"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.bronze_market_1m"

# 2. TICKERS
ETFS = ["SPY", "QQQ", "MTUM", "USMV", "IQLT", "EFV", "AGG"]
CREDIT_PROXY = ["HYG"]
ALL_TICKERS = ETFS + CREDIT_PROXY

# 3. SETTINGS
MAX_RETRIES = 3
START_YEAR = 2016 # 2015 has no data on Alpaca
END_YEAR = 2025

print(f"✅ Command Center Initialized.")
print(f"   Writing to: {VOLUME_PATH}")

# COMMAND ----------

# 1. Identify what we ALREADY have on disk (the 'Legit' data)
try:
    existing_data = spark.sql(f"SELECT DISTINCT symbol, year_partition, month_partition FROM delta.`{VOLUME_PATH}`").toPandas()
    loaded_set = set(zip(existing_data.year_partition, existing_data.month_partition, existing_data.symbol))
    print(f"🔍 Found {len(loaded_set)} existing segments. Checking for gaps...")
except:
    loaded_set = set()
    print("⚠️ No data found. Starting fresh.")

# 2. Targeted Fetch (Only the missing pieces)
for yr in range(START_YEAR, END_YEAR + 1):
    for month_num in range(1, 13):
        # Skip future months
        if yr == 2025 and month_num > 6: continue 
        
        for symbol in ALL_TICKERS:
            if (yr, month_num, symbol) in loaded_set:
                continue # SKIP: This month is already legit
            
            print(f"🩹 Filling Hole: {yr}-{month_num:02d} | {symbol}...", end=" ")
            
            # This calls your existing 'fetch_monthly_data' function
            pdf = fetch_monthly_data([symbol], yr, month_num)
            
            if pdf is not None and not pdf.empty:
                sdf = spark.createDataFrame(pdf, schema=alpaca_schema)
                sdf = (sdf
                    .withColumn("year_partition", F.year("timestamp").cast("integer"))
                    .withColumn("month_partition", F.month("timestamp").cast("integer"))
                    .withColumn("ingestion_ts", F.current_timestamp())
                )
                # Direct Write to Volume Path
                sdf.write.mode("append").format("delta").save(VOLUME_PATH)
                print(f"✅ {len(pdf)} rows.")
                time.sleep(1) # Safety pause for Serverless buffers
            else:
                print("⊘ No data on Alpaca.")

print("\n🚀 MARKET DATA IS NOW COMPLETE AND LEGIT.")

# COMMAND ----------

import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pyspark.sql import functions as F
from pyspark.sql.functions import col, year, month
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, LongType
import time
import traceback
import calendar

# ============================================================================
# SERVERLESS-SAFE CONFIGURATION (FIXED)
# ============================================================================
# ✅ Standardized variable names to prevent NameErrors
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
VOLUME = "market_data"

# Direct Volume Path for high-speed writes (Standard for Unity Catalog Volumes)
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bronze_market_1m"

# Unity Catalog Table Name for SQL Queries (The 3-part name)
TABLE_NAME = f"{CATALOG}.{SCHEMA}.bronze_market_1m"

# --- DATA CONFIGURATION ---
ETFS = ["SPY", "QQQ", "MTUM", "USMV", "IQLT", "EFV", "AGG"]
CREDIT_PROXY = ["HYG"]
ALL_TICKERS = ETFS + CREDIT_PROXY
START_YEAR = 2016
END_YEAR = 2025

# --- PERFORMANCE TUNING ---
MAX_RETRIES = 3
BATCH_SIZE = 1
HEALTH_CHECK_INTERVAL = 6

print(f"✅ Setup Complete")
print(f"   Volume Path:  {VOLUME_PATH}")
print(f"   Table Name:   {TABLE_NAME}")
print(f"   Tickers:      {len(ALL_TICKERS)} symbols")
print(f"   Date Range:   {START_YEAR} to {END_YEAR}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW VOLUMES IN pawan_quant_research_1.v4

# COMMAND ----------

from alpaca.data.historical import StockHistoricalDataClient

try:
    # Use Databricks Secrets (Recommended)
    API_KEY = dbutils.secrets.get(scope="alpaca", key="api_key_id")
    API_SECRET = dbutils.secrets.get(scope="alpaca", key="api_secret_key")
    print("✅ Credentials loaded from Secrets Vault")
except Exception as e:
    print(f"⚠️  Secrets not found ({str(e)[:60]}), using fallback credentials")
    print("⚠️  REMOVE HARDCODED CREDENTIALS BEFORE PRODUCTION!")
    API_KEY = "PKPSF55ZI6V53R25GIBFTUGB4B"
    API_SECRET = "5MnZMUeKU8EFaXFV6dHGUM1uS6kG8AgPXqCnaSjHVE2U"

client = StockHistoricalDataClient(API_KEY, API_SECRET)
print("✅ Alpaca client initialized")

# Verify cluster connection
try:
    cluster_id = spark.conf.get('spark.databricks.clusterUsageTags.clusterId')
    print(f"✅ Connected to Cluster: {cluster_id}")
    print(f"   Spark Version: {spark.version}")
    print(f"   Compute Type: {'Serverless' if 'serverless' in cluster_id.lower() else 'Classic'}")
except Exception as e:
    print(f"⚠️  Could not retrieve cluster info: {str(e)[:80]}")

# COMMAND ----------

# Import the specific Spark Types
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, LongType

# Define the schema strictly to match Alpaca's 1M bar format
alpaca_schema = StructType([
    StructField("symbol", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),      # Note: Alpaca volume is sometimes Double in Spark
    StructField("trade_count", DoubleType(), True),
    StructField("vwap", DoubleType(), True)
])

print("✅ Schema defined successfully")

# COMMAND ----------

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def fetch_monthly_data(symbol_list, year_val, month_val, max_retries=3):
    """Fetch 1-minute bars for a given month using Alpaca API with retry logic"""
    last_day = calendar.monthrange(year_val, month_val)[1]
    
    start = datetime(year_val, month_val, 1, 0, 0, 0, tzinfo=ZoneInfo("UTC"))
    end = datetime(year_val, month_val, last_day, 23, 59, 59, tzinfo=ZoneInfo("UTC"))

    req = StockBarsRequest(
        symbol_or_symbols=symbol_list,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start,
        end=end
    )
    
    for attempt in range(max_retries):
        try:
            bars = client.get_stock_bars(req)
            if bars.df is None or bars.df.empty:
                return None
            return bars.df.reset_index()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"\n      ⚠️ API retry {attempt + 1}/{max_retries} in {wait_time}s: {str(e)[:60]}")
                time.sleep(wait_time)
            else:
                print(f"\n      ❌ API Error after {max_retries} attempts: {str(e)[:80]}")
                return None
    return None


def write_with_retry(sdf, table_path, mode="append", max_retries=MAX_RETRIES):
    """
    Serverless-safe write to Delta path (avoids metastore operations)
    Always uses append mode after initial table creation
    """
    for attempt in range(max_retries):
        try:
            writer = sdf.write.format("delta").mode(mode)
            
            if mode == "overwrite":
                # First run: establish partitioning structure
                (writer
                    .partitionBy("symbol", "year_partition")
                    .option("overwriteSchema", "true")
                    .save(table_path))
            else:
                # Subsequent runs: append only
                writer.save(table_path)
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "PERSIST TABLE" in error_msg or "serverless" in error_msg.lower():
                raise RuntimeError(
                    "Metastore operation blocked in serverless environment. "
                    "Using path-based writes to Unity Catalog volume."
                )
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"\n      ⚠️ Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n      ❌ Write failed: {error_msg[:200]}")
                raise
    return False


def cluster_health_check():
    """Verify cluster is responsive"""
    try:
        spark.sql("SELECT 1 AS health_check").collect()
        return True
    except Exception as e:
        print(f"\n   ⚠️ Cluster health check failed: {str(e)[:100]}")
        return False


def get_loaded_months(table_path):
    """Get set of already-loaded (year, month, symbol) combinations from storage path"""
    try:
        # Read directly from path (serverless-safe)
        df = spark.read.format("delta").load(table_path)
        loaded = df.select("year_partition", "month_partition", "symbol").distinct().collect()
        return {(row.year_partition, row.month_partition, row.symbol) for row in loaded}
    except Exception as e:
        if "Path does not exist" in str(e):
            return set()  # First run
        else:
            print(f"⚠️  Error checking existing data: {str(e)[:100]}")
            return set()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS pawan_quant_research_1.v4.market_data
# MAGIC COMMENT 'Storage volume for ETF and credit spread minute-level data';

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS pawan_quant_research_1.v4.market_data
# MAGIC COMMENT 'Storage volume for ETF and credit spread minute-level data';

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW VOLUMES IN pawan_quant_research_1.v4

# COMMAND ----------

# ============================================================================
# SERVERLESS-SAFE PRE-FLIGHT CHECKS
# ============================================================================

# Check if volume exists
try:
    spark.sql(f"DESCRIBE VOLUME {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}")
    print(f"✅ Unity Catalog Volume exists: {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}")
except Exception as e:
    raise RuntimeError(
        f"Volume {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME} does not exist!\n"
        f"Create it first with:\n"
        f"  CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME};"
    )

# Check existing data
loaded_data = get_loaded_months(TABLE_PATH)
first_run = len(loaded_data) == 0

if first_run:
    print("🆕 First run detected - will create new table structure")
else:
    print(f"📊 Appending to existing data")
    print(f"   Found {len(loaded_data)} symbol-month combinations already loaded")
    print(f"   Example: {list(loaded_data)[:3]}")

# COMMAND ----------

# Track statistics
total_rows_ingested = 0
successful_months = 0
failed_months = 0
skipped_months = 0
month_counter = 0

print("\n" + "="*70)
print("🚀 STARTING SERVERLESS-SAFE DATA INGESTION")
print("="*70)

for yr in range(START_YEAR, END_YEAR + 1):
    print(f"\n{'='*70}")
    print(f"📅 Processing Year: {yr}")
    print(f"{'='*70}")
    
    # FIX: RENAME LOOP VARIABLE to avoid shadowing PySpark functions
    for month_num in range(1, 13):
        month_counter += 1
        
        # Skip future months
        current_date = datetime.now()
        if yr == current_date.year and month_num > current_date.month:
            print(f"   ⏭️  {yr}-{month_num:02d}: Skipping future month")
            continue
        
        # Periodic cluster health check
        if month_counter % HEALTH_CHECK_INTERVAL == 0:
            print(f"\n   🏥 Cluster health check...")
            if cluster_health_check():
                print(f"   ✅ Cluster is healthy")
            else:
                print(f"   ⚠️ Cluster may be experiencing issues")
                time.sleep(5)
        
        # Process each symbol individually
        month_success = True
        month_rows = 0
        
        for symbol in ALL_TICKERS:
            # Check if already loaded
            if (yr, month_num, symbol) in loaded_data:
                if symbol == ALL_TICKERS[0]:  # Only print once per month
                    print(f"   ⏭️  {yr}-{month_num:02d}: Already loaded for all symbols")
                    skipped_months += 1
                    break
                continue
            
            # Print month header for first symbol
            if symbol == ALL_TICKERS[0]:
                print(f"   🔄 {yr}-{month_num:02d} | ", end="", flush=True)
            
            print(f"{symbol}...", end=" ", flush=True)
            
            # Fetch data
            pdf = fetch_monthly_data([symbol], yr, month_num)
            
            if pdf is not None and len(pdf) > 0:
                try:
                    # Convert to Spark DataFrame with schema enforcement
                    sdf = spark.createDataFrame(pdf, schema=alpaca_schema)
                    
                    # FIX: Use F.year() and F.month() explicitly to avoid shadowing
                    sdf = (sdf
                        .withColumn("year_partition", F.year("timestamp").cast("integer"))
                        .withColumn("month_partition", F.month("timestamp").cast("integer"))
                        .withColumn("ingestion_ts", F.current_timestamp())
                    )
                    
                    # ✅ SERVERLESS-SAFE: NO CACHING - Direct count
                    # (Serverless doesn't support .cache()/.persist())
                    row_count = sdf.count()
                    
                    # Determine write mode
                    write_mode = "overwrite" if first_run else "append"
                    
                    # ✅ SERVERLESS-SAFE WRITE TO PATH
                    success = write_with_retry(sdf, TABLE_PATH, write_mode)
                    
                    if success:
                        if first_run:
                            first_run = False
                            print("(table created)", end=" ", flush=True)
                        
                        month_rows += row_count
                        print(f"✅{row_count:,}", end=" ", flush=True)
                    else:
                        month_success = False
                        print(f"❌", end=" ", flush=True)
                    
                    # ✅ NO UNPERSIST NEEDED (never cached)
                    # Clean up Python objects only
                    del sdf, pdf
                    
                except Exception as e:
                    month_success = False
                    print(f"❌ERR", end=" ", flush=True)
                    print(f"\n      Error details: {str(e)[:150]}")
                    traceback.print_exc()
            else:
                print(f"⊘", end=" ", flush=True)  # No data symbol
        
        # End of month summary
        if month_rows > 0:
            print(f"| Total: {month_rows:,} rows", flush=True)
            total_rows_ingested += month_rows
            successful_months += 1
        elif month_success:
            print("", flush=True)  # New line
        
        if not month_success and month_rows == 0:
            failed_months += 1
        
        # Brief pause between months
        time.sleep(0.5)

print("\n" + "="*70)
print("✅ INGESTION PHASE COMPLETE")
print("="*70)
print(f"✅ Successful months: {successful_months}")
print(f"⏭️  Skipped months (already loaded): {skipped_months}")
print(f"❌ Failed months: {failed_months}")
print(f"📊 Total rows ingested this run: {total_rows_ingested:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check what catalogs you have access to
# MAGIC SHOW CATALOGS;
# MAGIC
# MAGIC -- Create the catalog (if you have permissions)
# MAGIC CREATE CATALOG IF NOT EXISTS pawan_quant_research_1;
# MAGIC
# MAGIC -- Create the schema
# MAGIC CREATE SCHEMA IF NOT EXISTS pawan_quant_research_1.v4;
# MAGIC
# MAGIC -- Create the volume (where the raw files live)
# MAGIC CREATE VOLUME IF NOT EXISTS pawan_quant_research_1.v4.market_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- This query identifies months where a symbol has 0 records or significantly fewer than expected
# MAGIC SELECT 
# MAGIC   symbol, 
# MAGIC   year_partition, 
# MAGIC   month_partition, 
# MAGIC   COUNT(*) as record_count,
# MAGIC   CASE 
# MAGIC     WHEN COUNT(*) < 5000 THEN '⚠️  GAP DETECTED' 
# MAGIC     ELSE '✅ OK' 
# MAGIC   END as status
# MAGIC FROM delta.`/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}/bronze_market_1m` -- Use your TABLE_PATH here
# MAGIC GROUP BY 1, 2, 3
# MAGIC ORDER BY year_partition DESC, month_partition DESC, symbol ASC

# COMMAND ----------

print("\n" + "="*70)
print("✅ REGISTERING TABLE IN UNITY CATALOG")
print("="*70)

try:
    # Check if table exists
    if not spark.catalog.tableExists(FULL_TABLE_NAME):
        # ✅ SERVERLESS-SAFE TABLE CREATION (path-based)
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME}
            USING DELTA
            LOCATION '{TABLE_PATH}'
            PARTITIONED BY (symbol, year_partition)
        """)
        print(f"✅ Unity Catalog table created: {FULL_TABLE_NAME}")
    else:
        print(f"✅ Table already exists: {FULL_TABLE_NAME}")
        # Refresh table metadata
        spark.sql(f"REFRESH TABLE {FULL_TABLE_NAME}")
    
    # Verify registration
    print("\n📋 Table details:")
    spark.sql(f"DESCRIBE EXTENDED {FULL_TABLE_NAME}").filter(
        col("col_name").isin("Location", "Provider", "Type", "Partition Columns")
    ).show(truncate=False)
    
except Exception as e:
    print(f"⚠️  Table registration warning (data is SAFE in storage path):")
    print(f"   {str(e)[:200]}")
    print(f"\n💡 QUERY DATA DIRECTLY VIA:")
    print(f"   df = spark.read.format('delta').load('{TABLE_PATH}')")

# COMMAND ----------

print("\n" + "="*70)
print("📊 FINAL TABLE VALIDATION")
print("="*70)

try:
    # Read from Unity Catalog table
    df_summary = spark.table(FULL_TABLE_NAME)
    total_count = df_summary.count()
    
    print(f"\n✅ Total rows in table: {total_count:,}")
    
    print(f"\n📈 Data by Symbol:")
    df_summary.groupBy("symbol").count().orderBy("symbol").show()
    
    print(f"\n📅 Data by Year:")
    df_summary.groupBy("year_partition").count().orderBy("year_partition").show()
    
    print(f"\n🗓️  Date Range:")
    df_summary.select(
        F.min("timestamp").alias("earliest"),
        F.max("timestamp").alias("latest")
    ).show(truncate=False)
    
    # Data quality check
    print(f"\n🔍 Data Quality Check:")
    spark.sql(f"""
        SELECT 
            symbol,
            COUNT(*) as total_bars,
            COUNT(DISTINCT DATE(timestamp)) as trading_days,
            MIN(timestamp) as first_bar,
            MAX(timestamp) as last_bar,
            ROUND(AVG(volume), 0) as avg_volume
        FROM {FULL_TABLE_NAME}
        GROUP BY symbol
        ORDER BY symbol
    """).show(truncate=False)
    
except Exception as e:
    print(f"⚠️  Validation failed: {str(e)[:150]}")
    print(f"\n💡 FALLBACK: Reading directly from storage path...")
    try:
        df_fallback = spark.read.format("delta").load(TABLE_PATH)
        print(f"✅ Fallback read successful: {df_fallback.count():,} rows")
    except Exception as e2:
        print(f"❌ Fallback also failed: {str(e2)[:100]}")

# COMMAND ----------

print("\n" + "="*70)
print("🔧 TABLE OPTIMIZATION (Serverless-Safe)")
print("="*70)

try:
    # ✅ Serverless-safe optimization using path syntax
    print("Running OPTIMIZE with Z-ORDER...")
    spark.sql(f"OPTIMIZE delta.`{TABLE_PATH}` ZORDER BY (timestamp, symbol)")
    print("✅ Optimization complete")
    
    # Show table statistics
    print("\n📋 Table file statistics:")
    spark.sql(f"DESCRIBE DETAIL delta.`{TABLE_PATH}`").select(
        "numFiles", "sizeInBytes", "numRecords", "format"
    ).show(truncate=False)
    
except Exception as e:
    print(f"⚠️ Optimization skipped (non-critical): {str(e)[:100]}")

# COMMAND ----------

print("\n" + "="*70)
print("🔍 SAMPLE QUERIES (Serverless-Safe)")
print("="*70)

# Most recent data for each symbol
print("\n📊 Most Recent Bar for Each Symbol:")
spark.sql(f"""
    WITH ranked AS (
        SELECT 
            symbol,
            timestamp,
            close,
            volume,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
        FROM {FULL_TABLE_NAME}
    )
    SELECT symbol, timestamp, close, volume
    FROM ranked
    WHERE rn = 1
    ORDER BY symbol
""").show(truncate=False)

# SPY last 5 days
print("\n📈 SPY Last 5 Trading Days:")
spark.sql(f"""
    SELECT 
        DATE(timestamp) as trade_date,
        FIRST(open) as open,
        MAX(high) as high,
        MIN(low) as low,
        LAST(close) as close,
        SUM(volume) as volume
    FROM {FULL_TABLE_NAME}
    WHERE symbol = 'SPY'
      AND timestamp >= CURRENT_DATE - INTERVAL 5 DAYS
    GROUP BY DATE(timestamp)
    ORDER BY trade_date DESC
""").show()
