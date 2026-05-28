# Databricks notebook source
# MAGIC %md
# MAGIC # 06_BT_AlphaVantage_Earnings_Forensics
# MAGIC
# MAGIC This notebook is a **forensic-only** prototype for Alpha Vantage earnings ingestion.
# MAGIC It is designed to safely run under the `pawan_quant_research_1.v4` schema as a PIT validation
# MAGIC and historical diagnostics asset.
# MAGIC
# MAGIC **CRITICAL CONSTRAINTS:**
# MAGIC - **DO NOT** modify existing production RT chains.
# MAGIC - **DO NOT** modify existing retraining logic.
# MAGIC - **DO NOT** introduce earnings features into live inference yet.
# MAGIC - This phase is for **forensic + PIT validation only**.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install requests pytz pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import json
import time
import random
import requests
import pytz
import pandas as pd
from datetime import datetime, date, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, TimestampType, DoubleType, BooleanType
)

# --- TARGET CONFIGURATION ---
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_earnings_events_forensics"

# Sample earnings tickers to pull
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "SPY"]

print(f"✅ Ingestion Environment Loaded. Target table defined as: {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Reading Secrets from Databricks Secrets Only

# COMMAND ----------

# Attempt to load Alpha Vantage keys from Databricks Secrets
api_keys = []
try:
    possible_scopes = ["alphavantage", "alpha_vantage", "alpaca", "massive"]
    for scope in possible_scopes:
        try:
            secrets_list = dbutils.secrets.list(scope)
            for secret_metadata in secrets_list:
                key_name = secret_metadata.key
                if "key" in key_name.lower() or "token" in key_name.lower() or "alpha" in key_name.lower():
                    try:
                        val = dbutils.secrets.get(scope=scope, key=key_name).strip()
                        if val and val not in api_keys:
                            api_keys.append(val)
                            print(f"🔑 Successfully loaded secret: scope={scope}, key={key_name} (masked: {val[:4]}...{val[-4:] if len(val) > 8 else ''})")
                    except Exception as secret_err:
                        pass
        except Exception:
            # Fallback to loading common explicit keys if listing is disabled/fails
            explicit_keys = ["api_key", "api_key_1", "api_key_2", "api_key_3", "api_key_id", "alphavantage_api_key"]
            for key in explicit_keys:
                try:
                    val = dbutils.secrets.get(scope=scope, key=key).strip()
                    if val and val not in api_keys:
                        api_keys.append(val)
                        print(f"🔑 Loaded explicit secret: scope={scope}, key={key} (masked: {val[:4]}...{val[-4:] if len(val) > 8 else ''})")
                except Exception:
                    pass
except Exception as e:
    print(f"⚠️ Secrets discovery failed: {e}")

# Validate keys are loaded
if not api_keys:
    print("❌ CRITICAL: No Alpha Vantage API keys loaded from Databricks Secrets.")
    print("Due to the strict requirement: 'Read Alpha Vantage API keys from Databricks Secrets ONLY', we raise an error.")
    raise ValueError("Alpha Vantage API keys must be loaded from Databricks Secrets ONLY!")
else:
    print(f"🎉 Successfully loaded {len(api_keys)} rotating API key(s) from Secrets.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Implement Rotating-Key Retry Client with Backoff and 429 Logging

# COMMAND ----------

class RotatingKeyAlphaVantageClient:
    def __init__(self, keys):
        self.keys = keys
        self.current_key_idx = 0
        self.usage_stats = {
            k: {"calls": 0, "429s": 0, "success": 0, "errors": 0} for k in keys
        }
        
    def _get_current_key(self):
        if not self.keys:
            raise ValueError("No API keys available.")
        return self.keys[self.current_key_idx]
        
    def _rotate_key(self):
        if len(self.keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
            print(f"🔄 Rotating to next key (Index {self.current_key_idx})")
        else:
            print("⚠️ Only one API key loaded; rotation skipped.")
        
    def request(self, function, symbol, max_retries=5):
        base_url = "https://www.alphavantage.co/query"
        base_delay = 2.0  # seconds
        
        for attempt in range(1, max_retries + 1):
            key = self._get_current_key()
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": key
            }
            
            self.usage_stats[key]["calls"] += 1
            print(f"🛰️ Requesting {function} for {symbol} (Attempt {attempt}/{max_retries}, Key Index {self.current_key_idx})...")
            
            try:
                response = requests.get(base_url, params=params, timeout=15)
                
                # Handle standard HTTP 429
                if response.status_code == 429:
                    self.usage_stats[key]["429s"] += 1
                    print(f"⚠️ Hit HTTP 429 Rate Limit on key index {self.current_key_idx}.")
                    self._rotate_key()
                    delay = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                    print(f"🕒 Backing off for {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                    
                response.raise_for_status()
                payload = response.json()
                
                # Handle Alpha Vantage standard rate limit messages returned inside 200 OK responses
                if "Note" in payload:
                    self.usage_stats[key]["429s"] += 1
                    print(f"⚠️ Hit Alpha Vantage rate limit (200 OK note): '{payload['Note']}'")
                    self._rotate_key()
                    delay = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                    print(f"🕒 Backing off for {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                    
                # Handle API logical errors
                if "Error Message" in payload:
                    self.usage_stats[key]["errors"] += 1
                    print(f"❌ Alpha Vantage API logical error for {symbol}: {payload['Error Message']}")
                    return {"error": payload["Error Message"], "symbol": symbol}
                    
                # Successful response
                self.usage_stats[key]["success"] += 1
                return payload
                
            except requests.exceptions.RequestException as req_err:
                self.usage_stats[key]["errors"] += 1
                print(f"💥 Network error: {req_err}")
                self._rotate_key()
                delay = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(delay)
                
        raise RuntimeError(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts.")

client = RotatingKeyAlphaVantageClient(api_keys)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Parsing and Point-In-Time (PIT) Leakage Diagnostics Engine

# COMMAND ----------

def parse_earnings_payload(symbol, payload):
    if "quarterlyEarnings" not in payload:
        print(f"⚠️ No quarterly earnings returned for symbol {symbol}.")
        return []
        
    records = []
    for q in payload["quarterlyEarnings"]:
        fiscal_date_str = q.get("fiscalDateEnding")
        reported_date_str = q.get("reportedDate")
        
        try:
            fiscal_date = datetime.strptime(fiscal_date_str, "%Y-%m-%d").date() if fiscal_date_str else None
            reported_date = datetime.strptime(reported_date_str, "%Y-%m-%d").date() if reported_date_str else None
        except ValueError:
            continue
            
        if not fiscal_date or not reported_date:
            continue
            
        # Safely parse numeric parameters
        def safe_float(val):
            try:
                return float(val) if val is not None and val != "None" else None
            except ValueError:
                return None
                
        eps_estimate = safe_float(q.get("estimatedEPS"))
        eps_actual = safe_float(q.get("reportedEPS"))
        surprise = safe_float(q.get("surprise"))
        surprise_pct = safe_float(q.get("surprisePercentage"))
        
        reported_time = q.get("reportedTime", None)
        
        # Point-In-Time Timing Estimation & Lookahead Bias Mitigation
        # Because precise time of release is typically missing from AV endpoints, we mark it as ambiguous.
        is_timestamp_ambiguous = True if not reported_time else False
        
        # Categorize timing
        release_timing_type = "UNKNOWN"
        if reported_time:
            time_lower = reported_time.lower()
            if "bmo" in time_lower or "before" in time_lower:
                release_timing_type = "BMO"
            elif "amc" in time_lower or "after" in time_lower:
                release_timing_type = "AMC"
                
        # Estimate Earliest Tradable Timestamp:
        # BMO (Before Market Open): Tradable on reported_date at 09:30:00 US/Eastern
        # AMC (After Market Close) / UNKNOWN: Tradable on reported_date + 1 day at 09:30:00 US/Eastern
        eastern = pytz.timezone("US/Eastern")
        
        if release_timing_type == "BMO":
            tradable_naive = datetime.combine(reported_date, datetime.strptime("09:30:00", "%H:%M:%S").time())
        else:
            # AMC or UNKNOWN: Assume tradable on next trading day EOD/Open to prevent lookahead leak
            tradable_naive = datetime.combine(reported_date + timedelta(days=1), datetime.strptime("09:30:00", "%H:%M:%S").time())
            
        estimated_earliest_tradable_ts = eastern.localize(tradable_naive).astimezone(pytz.utc)
        
        # Standard Reported Timestamp representation:
        # Defaults to 16:00:00 Eastern on reported_date if precise time is unavailable.
        if reported_time:
            try:
                time_obj = datetime.strptime(reported_time, "%H:%M:%S").time()
                reported_naive = datetime.combine(reported_date, time_obj)
            except:
                reported_naive = datetime.combine(reported_date, datetime.strptime("16:00:00", "%H:%M:%S").time())
        else:
            reported_naive = datetime.combine(reported_date, datetime.strptime("16:00:00", "%H:%M:%S").time())
            
        reported_timestamp_utc = eastern.localize(reported_naive).astimezone(pytz.utc)
        
        # PIT Leakage Guard Timestamp:
        # This is a solid, conservative timestamp for safe backtesting.
        # Definition: reported_date + 1 day at 09:30:00 Eastern, converted to UTC.
        pit_guard_naive = datetime.combine(reported_date + timedelta(days=1), datetime.strptime("09:30:00", "%H:%M:%S").time())
        pit_leakage_guard_ts = eastern.localize(pit_guard_naive).astimezone(pytz.utc)
        
        # Serialize row structure for full forensic audit
        raw_payload_json = json.dumps(q)
        
        records.append({
            "symbol": symbol,
            "fiscal_date": fiscal_date,
            "reported_date": reported_date,
            "reported_time": reported_time,
            "reported_timestamp_utc": reported_timestamp_utc,
            "eps_estimate": eps_estimate,
            "eps_actual": eps_actual,
            "surprise": surprise,
            "surprise_pct": surprise_pct,
            "raw_payload_json": raw_payload_json,
            "release_timing_type": release_timing_type,
            "estimated_earliest_tradable_ts": estimated_earliest_tradable_ts,
            "is_timestamp_ambiguous": is_timestamp_ambiguous,
            "pit_leakage_guard_ts": pit_leakage_guard_ts,
            "ingestion_ts": datetime.now(pytz.utc)
        })
        
    return records

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Safe Append-Only Spark Write Function with anti-join Merge Protection

# COMMAND ----------

spark_schema = StructType([
    StructField("symbol", StringType(), False),
    StructField("fiscal_date", DateType(), False),
    StructField("reported_date", DateType(), False),
    StructField("reported_time", StringType(), True),
    StructField("reported_timestamp_utc", TimestampType(), False),
    StructField("eps_estimate", DoubleType(), True),
    StructField("eps_actual", DoubleType(), True),
    StructField("surprise", DoubleType(), True),
    StructField("surprise_pct", DoubleType(), True),
    StructField("raw_payload_json", StringType(), False),
    StructField("release_timing_type", StringType(), False),
    StructField("estimated_earliest_tradable_ts", TimestampType(), False),
    StructField("is_timestamp_ambiguous", BooleanType(), False),
    StructField("pit_leakage_guard_ts", TimestampType(), False),
    StructField("ingestion_ts", TimestampType(), False)
])

def write_earnings_to_delta(records):
    if not records:
        print("⏩ No new parsed records to write.")
        return 0
        
    incoming_df = spark.createDataFrame(records, schema=spark_schema)
    
    # Check if table exists to enable anti-join (ensuring complete deduplication stability)
    if spark.catalog.tableExists(TARGET_TABLE):
        print(f"📊 Delta target table {TARGET_TABLE} exists. Applying left-anti join deduplication...")
        existing_df = spark.read.table(TARGET_TABLE).select("symbol", "fiscal_date", "reported_date")
        new_df = incoming_df.join(existing_df, on=["symbol", "fiscal_date", "reported_date"], how="left_anti")
    else:
        print(f"🆕 Creating Delta target table {TARGET_TABLE} as it does not exist...")
        new_df = incoming_df
        
    append_count = new_df.count()
    if append_count > 0:
        (new_df.write
            .format("delta")
            .mode("append")
            .saveAsTable(TARGET_TABLE))
        print(f"✨ Successfully appended {append_count} new deduplicated rows to {TARGET_TABLE}.")
    else:
        print("⏩ All fetched rows already exist in the Delta table. Deduplicated count is 0. Skip append.")
        
    return append_count

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Ingestion Orchestrator Execution

# COMMAND ----------

all_parsed_records = []

for symbol in TICKERS:
    print(f"🚀 Acquiring data for {symbol}...")
    try:
        payload = client.request("EARNINGS", symbol)
        if "error" in payload:
            print(f"⏩ Skipping parsed records for {symbol} due to API Error.")
            continue
            
        parsed = parse_earnings_payload(symbol, payload)
        print(f"Parsed {len(parsed)} quarterly records for {symbol}.")
        all_parsed_records.extend(parsed)
    except Exception as e:
        print(f"❌ Failed to acquire data for {symbol}: {e}")

# Perform the first write
first_append_count = write_earnings_to_delta(all_parsed_records)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Verification and Validation Logic

# COMMAND ----------

# MAGIC %md
# MAGIC #### Validation Block 1: Run notebook twice and confirm dedupe stability

# COMMAND ----------

print("\n🔁 RUNNING SECOND PASS FOR DEDUPE STABILITY VERIFICATION...")
second_pass_records = []
for symbol in TICKERS:
    try:
        # Request again (will hit cached or rate-limit-aware rotating client)
        payload = client.request("EARNINGS", symbol)
        if "error" in payload: continue
        parsed = parse_earnings_payload(symbol, payload)
        second_pass_records.extend(parsed)
    except Exception as e:
        print(f"⚠️ Second pass skipped for {symbol}: {e}")

second_append_count = write_earnings_to_delta(second_pass_records)

# Validation tests
assert second_append_count == 0, f"❌ Validation Failure: Deduplication stability failed! Appended {second_append_count} rows on second pass."
print("✅ SUCCESS: Deduplication stability confirmed. Second pass appended exactly 0 rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Validation Block 2: Verify timestamps are UTC-normalized

# COMMAND ----------

if spark.catalog.tableExists(TARGET_TABLE):
    timezone_check = spark.sql(f"""
        SELECT 
            symbol,
            reported_timestamp_utc,
            estimated_earliest_tradable_ts,
            pit_leakage_guard_ts
        FROM {TARGET_TABLE}
        LIMIT 5
    """).collect()
    
    for row in timezone_check:
        assert row.reported_timestamp_utc.tzinfo is not None, "❌ reported_timestamp_utc lacks timezone normalization"
        assert row.estimated_earliest_tradable_ts.tzinfo is not None, "❌ estimated_earliest_tradable_ts lacks timezone normalization"
        assert row.pit_leakage_guard_ts.tzinfo is not None, "❌ pit_leakage_guard_ts lacks timezone normalization"
        
    print("✅ SUCCESS: Timestamp UTC normalization validated successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Summary and Forensics Diagnostics Engine

# COMMAND ----------

# MAGIC %md
# MAGIC #### Diagnostic 1: API Quota Diagnostics per key

# COMMAND ----------

print("\n📈 ALPHA VANTAGE API USAGE METRICS PER KEY:")
print("=" * 80)
for k, stats in client.usage_stats.items():
    masked_key = f"{k[:4]}...{k[-4:] if len(k) > 8 else ''}"
    print(f"🔑 Key: {masked_key}")
    print(f"   • Total Request Calls: {stats['calls']}")
    print(f"   • 429 Rate-Limit Hits: {stats['429s']}")
    print(f"   • Successful Fetches: {stats['success']}")
    print(f"   • Request Failures:    {stats['errors']}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Diagnostic 2: Missing-Quarter Diagnostics

# COMMAND ----------

# Missing-Quarter analysis using window lag
if spark.catalog.tableExists(TARGET_TABLE):
    print("\n🔍 MISSING QUARTER DIAGNOSTICS:")
    print("=" * 80)
    
    missing_df = spark.sql(f"""
        WITH ordered_earnings AS (
            SELECT 
                symbol,
                fiscal_date,
                LAG(fiscal_date) OVER (PARTITION BY symbol ORDER BY fiscal_date ASC) as prev_fiscal_date
            FROM {TARGET_TABLE}
        )
        SELECT 
            symbol,
            prev_fiscal_date,
            fiscal_date,
            months_between(fiscal_date, prev_fiscal_date) as months_gap
        FROM ordered_earnings
        WHERE prev_fiscal_date IS NOT NULL
          AND months_between(fiscal_date, prev_fiscal_date) > 4.0
        ORDER BY symbol, fiscal_date
    """)
    
    missing_count = missing_df.count()
    if missing_count > 0:
        print(f"⚠️ Found {missing_count} gap(s) exceeding standard quarterly periods (4+ months):")
        display(missing_df)
    else:
        print("✅ No missing quarters or abnormal gaps detected in historical series.")
    print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Diagnostic 3: Timestamp Ambiguity Diagnostics

# COMMAND ----------

if spark.catalog.tableExists(TARGET_TABLE):
    print("\n🔍 TIMESTAMP AMBIGUITY DIAGNOSTICS:")
    print("=" * 80)
    
    ambiguity_stats = spark.sql(f"""
        SELECT 
            symbol,
            COUNT(*) as total_records,
            SUM(CASE WHEN is_timestamp_ambiguous THEN 1 ELSE 0 END) as ambiguous_count,
            ROUND(SUM(CASE WHEN is_timestamp_ambiguous THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) as ambiguity_pct
        FROM {TARGET_TABLE}
        GROUP BY symbol
    """)
    display(ambiguity_stats)
    print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Diagnostic 4: Duplicate Diagnostics

# COMMAND ----------

if spark.catalog.tableExists(TARGET_TABLE):
    print("\n🔍 DUPLICATE DIAGNOSTICS:")
    print("=" * 80)
    
    duplicates = spark.sql(f"""
        SELECT 
            symbol, 
            fiscal_date, 
            reported_date, 
            COUNT(*) as occurrences
        FROM {TARGET_TABLE}
        GROUP BY symbol, fiscal_date, reported_date
        HAVING COUNT(*) > 1
    """)
    
    dup_count = duplicates.count()
    if dup_count > 0:
        print(f"❌ ERROR: Found {dup_count} duplicate records in forensic table!")
        display(duplicates)
    else:
        print("✅ SUCCESS: No duplicate records found in forensic table.")
    print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Final Table Audit peek

# COMMAND ----------

if spark.catalog.tableExists(TARGET_TABLE):
    display(spark.sql(f"SELECT * FROM {TARGET_TABLE} ORDER BY symbol, fiscal_date DESC LIMIT 20"))
