# Databricks notebook source
!pip install finnhub-python

# COMMAND ----------

# MAGIC %md
# MAGIC Block 1: Setup & Finnhub Harvester

# COMMAND ----------

import finnhub
import pandas as pd
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
API_KEY = "d5s3pchr01qoo9r2f3h0d5s3pchr01qoo9r2f3hg"
VOLUME_PATH = "/Volumes/pawan_quant_research_1/v4/market_data"
HAYSTACK_FILE = f"{VOLUME_PATH}/raw_finnhub_haystack.csv"

# CHECK IF FILE EXISTS TO AVOID 30-MIN WAIT
if os.path.exists(HAYSTACK_FILE):
    print(f"♻️ Loading existing Haystack from {HAYSTACK_FILE}. (Skipping 30-min API harvest)")
    haystack_df = pd.read_csv(HAYSTACK_FILE)
else:
    client = finnhub.Client(api_key=API_KEY)
    START_YEAR, END_YEAR = 2015, 2026
    print("🚀 File not found. Harvesting Raw US Macro Haystack from Finnhub...")
    all_raw_events = []

    for yr in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            if yr == 2026 and month > datetime.now().month: continue
            start_date = f"{yr}-{month:02d}-01"
            end_date = f"{yr}-{month:02d}-31" if month == 12 else f"{yr}-{(month+1):02d}-01"
            try:
                res = client.calendar_economic(_from=start_date, to=end_date)
                if 'economicCalendar' in res:
                    month_df = pd.DataFrame(res['economicCalendar'])
                    us_relevant = month_df[(month_df['country'] == 'US') & 
                                           (month_df['impact'].isin(['high', 'medium']))].copy()
                    all_raw_events.append(us_relevant)
            except Exception as e:
                print(f"Error fetching {yr}-{month}: {e}")
            time.sleep(1.2) 

    haystack_df = pd.concat(all_raw_events).drop_duplicates(subset=['time', 'event'])
    haystack_df.to_csv(HAYSTACK_FILE, index=False)
    print(f"✅ Harvest Complete. Haystack saved to {HAYSTACK_FILE}")

# COMMAND ----------

# MAGIC %md
# MAGIC Block 2: The Naming Drift & ALFRED Engine

# COMMAND ----------

# MAGIC %pip install fredapi
# MAGIC

# COMMAND ----------

from fredapi import Fred
import pytz

FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"
fred = Fred(api_key=FRED_API_KEY)

# UPDATED: Includes PCE and handles drift for all 8 core series
INDICATOR_MAP = {
    'PAYEMS':   ['Non-Farm Payrolls', 'United States-Employment-Non-Farm Payrolls', 'Nonfarm Payrolls'],
    'UNRATE':   ['Unemployment Rate', 'United States-Employment-Unemployment Rate'],
    'CPIAUCSL': ['CPI MM, SA', 'United States-CPI-CPI MM, SA', 'Consumer Price Index (CPI)'],
    'PCEPI':    ['Core PCE Price Index', 'United States-Consumption-Core PCE Price Index MM', 'PCE Price Index'],
    'PCE':      ['Personal Consumption Expenditures', 'United States-Consumption-Personal Consumption Expenditures'],
    'INDPRO':   ['Industrial Production'],
    'GDPC1':    ['Real GDP', 'GDP Growth Rate'],
    'UMCSENT':  ['U Mich Sentiment', 'Michigan Consumer Sentiment', 'U Mich Sentiment Final']
}

# Precise times from your report (PCE usually aligns with PCEPI at 8:30)
TYPICAL_TIMES = {
    'PAYEMS': '08:30:00', 'UNRATE': '08:30:00', 'CPIAUCSL': '08:30:00',
    'PCEPI': '08:30:00', 'PCE': '08:30:00', 'INDPRO': '09:15:00', 
    'GDPC1': '08:30:00', 'UMCSENT': '10:00:00'
}

# COMMAND ----------

# MAGIC %md
# MAGIC Block 3: timing Pattern Application (Claude Analysis Logic)

# COMMAND ----------

def get_alfred_first_release(series_id, release_date):
    try:
        vintages = fred.get_series_all_releases(series_id)
        match = vintages[vintages['realtime_start'] == release_date]
        if not match.empty: return match.iloc[0]['value']
    except: return None
    return None

def apply_precise_timing(indicator_id, alfred_date):
    wall_clock_time = TYPICAL_TIMES.get(indicator_id, '08:30:00')
    naive_dt = datetime.strptime(f"{alfred_date} {wall_clock_time}", '%Y-%m-%d %H:%M:%S')
    eastern = pytz.timezone('US/Eastern')
    localized_dt = eastern.localize(naive_dt, ambiguous='infer')
    return localized_dt.astimezone(pytz.utc)

for indicator_id, vendor_names in INDICATOR_MAP.items():
    print(f"📦 Triangulating {indicator_id}...")
    mask = haystack_df['event'].isin(vendor_names)
    indicator_releases = haystack_df[mask].copy()
    indicator_releases['release_date'] = pd.to_datetime(indicator_releases['time']).dt.date
    
    results = []
    for _, row in indicator_releases.iterrows():
        val = get_alfred_first_release(indicator_id, str(row['release_date']))
        if val is not None:
            precise_utc = apply_precise_timing(indicator_id, str(row['release_date']))
            results.append({
                'value': val,
                'release_datetime': precise_utc.strftime('%Y-%m-%d %H:%M:%S'),
                'release_time': TYPICAL_TIMES.get(indicator_id)
            })

    if results:
        pd.DataFrame(results).to_csv(f"{VOLUME_PATH}/first_releases_{indicator_id}.csv", index=False)
        print(f"✅ Created first_releases_{indicator_id}.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Block 4: Final CSV Generation (Ablation-Ready Files)

# COMMAND ----------

from pyspark.sql import functions as F

# 1. Define the indicators to consolidate (using the keys from our INDICATOR_MAP)
FINAL_INDICATORS = list(INDICATOR_MAP.keys())
all_dfs = []

for ind in FINAL_INDICATORS:
    file_path = f"{VOLUME_PATH}/first_releases_{ind}.csv"
    try:
        # Load the CSV
        # We don't use inferSchema so that the "." is read as a string first
        sdf = spark.read.csv(file_path, header=True)
        
        # 2. Use SQL expressions via expr() to handle the try_cast safely
        clean_sdf = sdf.select(
            F.lit(ind).alias("indicator_id"),
            F.to_timestamp(F.col("release_datetime")).alias("timestamp"),
            F.expr("try_cast(value as double)").alias("actual"),
            F.lit(f"First Release: {ind}").alias("event")
        )
        
        # 3. Filter out the malformed rows (the "." entries that became NULL)
        clean_sdf = clean_sdf.filter(F.col("actual").isNotNull())
        
        all_dfs.append(clean_sdf)
        print(f"✅ Prepared {ind} for consolidation.")
        
    except Exception as e:
        print(f"⚠️ Note: {ind} CSV skipped. Reason: {str(e)}")

# 4. Union and add partition columns
if all_dfs:
    final_macro_sdf = all_dfs[0]
    for next_df in all_dfs[1:]:
        final_macro_sdf = final_macro_sdf.union(next_df)
    
    # Add partitions and audit columns
    final_macro_sdf = final_macro_sdf.withColumn("year_partition", F.year("timestamp")) \
                                     .withColumn("ingestion_ts", F.current_timestamp())


    # 5. Save to the Bronze Layer (Managed Table)
    TARGET_TABLE = f"pawan_quant_research_1.v4.bronze_bt_macro_events"

    (final_macro_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("indicator_id")
        .saveAsTable(TARGET_TABLE))

    print(f"\n✨ SUCCESS: Consolidated High-Precision table created at {TARGET_TABLE}")
    
    # Final Audit Display
    summary = spark.sql(f"""
        SELECT indicator_id, count(*) as bar_count, round(avg(actual),2) as avg_val 
        FROM {TARGET_TABLE} 
        GROUP BY 1 
        ORDER BY bar_count DESC
    """)
    display(summary)
else:
    print("❌ Error: No data found to consolidate. Check if CSVs exist in the Volume.")

# COMMAND ----------

# Check the actual modification time of the PAYEMS file
import os
import datetime

file_path = "/Volumes/pawan_quant_research_1/v4/market_data/first_releases_PAYEMS.csv"
stats = os.stat(file_path)
last_mod = datetime.datetime.fromtimestamp(stats.st_mtime)

print(f"File: {file_path}")
print(f"Actual Last Modified (Filesystem): {last_mod}")
print(f"File Size: {stats.st_size / 1024:.2f} KB")

# Peek at the last 5 rows to see if 2025/2026 data is there
display(spark.read.csv(file_path, header=True).tail(5))

# COMMAND ----------

# Check the most recent release date in the actual data
df_check = pd.read_csv("/Volumes/pawan_quant_research_1/v4/market_data/first_releases_PAYEMS.csv")
print(f"Most recent release in file: {df_check['release_date'].max()}")

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load(
    "/Volumes/pawan_quant_research_1/v4/market_data/raw_finnhub_haystack.csv"
)

df_filtered = df.filter(
    (df["time"] >= "2024-03-01") & (df["time"] <= "2024-04-30")
)

display(df_filtered)