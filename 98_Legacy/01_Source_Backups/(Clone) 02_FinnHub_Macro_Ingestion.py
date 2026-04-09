# Databricks notebook source
!pip install finnhub-python

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 1: Setup & Libraries

# COMMAND ----------

# MAGIC %pip install finnhub-python requests

import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType

# --- CONFIGURATION (Unity Catalog) ---
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
VOLUME = "market_data"

# Input/Output Paths
RAW_HAYSTACK_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_finnhub_haystack.csv"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bronze_macro_events"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.bronze_macro_events"

print(f"✅ Triangulation Environment Ready")

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 2: Authentication

# COMMAND ----------

# Use Databricks Secrets or Fallback
try:
    API_KEY = dbutils.secrets.get(scope="alpaca", key="finnhub_api_key")
    print("✅ Finnhub Credentials loaded from Secrets")
except:
    API_KEY = "d5s3pchr01qoo9r2f3h0d5s3pchr01qoo9r2f3hg"
    print("⚠️ Using Fallback Credentials")

# FRED API Key for ALFRED Triangulation
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 3: Logic & Spark Schema

# COMMAND ----------

# This map is tuned to the exact strings found in your 1.csv audit.
# It captures naming drift from 2015 to 2025.
TRIANGULATION_MAP = {
    'CPIAUCSL': {
        'fred_id': 'CPIAUCSL', 
        'keywords': ['CPI MM', 'CPI s.a', 'Inflation Rate', 'CPI YY', 'CPI Index'],
        'hours': [13, 14] 
    },
    'UNRATE': {
        'fred_id': 'UNRATE', 
        'keywords': ['Unemployment Rate'],
        'hours': [13, 14]
    },
    'PAYEMS': {
        'fred_id': 'PAYEMS', 
        'keywords': ['Non Farm Payrolls', 'Non-Farm Payrolls', 'ADP Employment'],
        'hours': [12, 13]
    },
    'GDPC1': {
        'fred_id': 'GDPC1', 
        'keywords': ['GDP Advance', 'GDP Growth Rate', 'GDP Final'],
        'hours': [13, 14, 15]
    },
    'UMCSENT': {
        'fred_id': 'UMCSENT', 
        'keywords': ['U Mich Sentiment', 'Michigan Consumer Sentiment'],
        'hours': [14, 15, 16]
    },
    'INDPRO': {
        'fred_id': 'INDPRO',
        'keywords': ['Industrial Production'],
        'hours': [13, 14, 15]
    }
}

def get_alfred_value(series_id, release_date, api_key):
    """The Golden Source: Queries ALFRED for the real value known on this date."""
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&api_key={api_key}&file_type=json"
           f"&realtime_start={release_date}&realtime_end={release_date}")
    try:
        res = requests.get(url).json()
        if 'observations' in res and res['observations']:
            # The last element is the latest 'vintage' known on that release day
            return float(res['observations'][-1]['value'])
    except:
        return None
    return None

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 4: Execution Loop (Bronze Ingestion)

# COMMAND ----------

print(f"📂 Loading Haystack from {RAW_HAYSTACK_PATH}...")
df_haystack = pd.read_csv(RAW_HAYSTACK_PATH)
df_haystack['dt'] = pd.to_datetime(df_haystack['time'])
df_haystack['release_date'] = df_haystack['dt'].dt.date

final_verified_data = []

print(f"🧪 Triangulating events with ALFRED (2015-2025)...")

for indicator_id, cfg in TRIANGULATION_MAP.items():
    # 1. Broad Search using keywords
    pattern = '|'.join(cfg['keywords'])
    mask = (df_haystack['event'].str.contains(pattern, case=False, na=False)) & \
           (df_haystack['dt'].dt.hour.isin(cfg['hours']))
    
    matches = df_haystack[mask].copy()
    
    # 2. Deduplicate: One release per day per indicator
    unique_releases = matches.drop_duplicates('release_date')
    
    print(f"🔍 {indicator_id}: Found {len(unique_releases)} potential dates. Verifying with ALFRED...", end=" ")
    
    hits = 0
    for _, row in unique_releases.iterrows():
        # 3. Authenticate with ALFRED
        pit_val = get_alfred_value(cfg['fred_id'], row['release_date'], FRED_API_KEY)
        
        if pit_val is not None:
            final_verified_data.append({
                'indicator_id': indicator_id,
                'timestamp': row['dt'],
                'actual': pit_val,
                'event': row['event']
            })
            hits += 1
    print(f"✅ {hits} Verified.")
    time.sleep(0.2) # Avoid hitting FRED rate limits

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 5: Data Audit (SQL)

# COMMAND ----------

if final_verified_data:
    # 1. Convert to Spark
    clean_pd = pd.DataFrame(final_verified_data)
    sdf = spark.createDataFrame(clean_pd)
    
    # 2. Add Partitions and Lineage (Same as Alpaca Notebook)
    sdf = (sdf
        .withColumn("year_partition", F.year("timestamp").cast("integer"))
        .withColumn("ingestion_ts", F.current_timestamp())
    )
    
    # 3. Save to Volume
    print(f"💾 Saving {len(final_verified_data)} authentic events to {VOLUME_PATH}...")
    sdf.write.mode("overwrite") \
       .option("overwriteSchema", "true") \
       .format("delta") \
       .save(VOLUME_PATH)

    # 4. Register Table (Serverless Friendly)
    spark.sql(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM delta.`{VOLUME_PATH}`")
    
    print(f"\n✨ AUTHENTIC BRONZE LAYER COMPLETE: {TABLE_NAME}")
else:
    print("❌ No data verified. Check API keys and raw haystack.")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify data distribution across the 'Big 6'
# MAGIC -- We expect ~120 for monthly and ~40 for quarterly
# MAGIC SELECT 
# MAGIC     indicator_id, 
# MAGIC     COUNT(*) as total_verified_releases,
# MAGIC     MIN(timestamp) as first_release,
# MAGIC     MAX(timestamp) as last_release,
# MAGIC     AVG(actual) as avg_value
# MAGIC FROM pawan_quant_research_1.v4.bronze_macro_events
# MAGIC GROUP BY indicator_id
# MAGIC ORDER BY total_verified_releases DESC;

# COMMAND ----------

# --- CONFIG ---
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db" 

# 1. Read the raw data back from the Volume
print(f"📂 Loading {RAW_CSV_PATH} for analysis...")
df_haystack = pd.read_csv(RAW_CSV_PATH)
df_haystack['dt'] = pd.to_datetime(df_haystack['time'])

# 2. Define our targets (Ultra-Wide keywords)
TRIANGULATION_MAP = {
    'CPIAUCSL': {'fred_id': 'CPIAUCSL', 'keywords': ['CPI', 'Consumer Price', 'Inflation']},
    'UNRATE':   {'fred_id': 'UNRATE',   'keywords': ['Unemployment', 'Employment Situation']},
    'PAYEMS':   {'fred_id': 'PAYEMS',   'keywords': ['Payrolls', 'Nonfarm', 'Employment Change']},
    'GDPC1':    {'fred_id': 'GDPC1',    'keywords': ['GDP', 'Growth Rate']},
    'UMCSENT':  {'fred_id': 'UMCSENT',  'keywords': ['Michigan', 'Sentiment']}
}

final_verified_data = []

print("🧪 Triangulating events with ALFRED First-Prints...")

# 3. Analyze the CSV row by row
for indicator_id, cfg in TRIANGULATION_MAP.items():
    pattern = '|'.join(cfg['keywords'])
    # Filter the CSV data in memory
    matches = df_haystack[df_haystack['event'].str.contains(pattern, case=False, na=False)].copy()
    
    # Deduplicate by date (FinnHub often lists the same release twice)
    matches['release_date'] = matches['dt'].dt.date
    unique_releases = matches.drop_duplicates('release_date')
    
    print(f"🔍 {indicator_id}: Found {len(unique_releases)} potential dates. Verifying...", end=" ")
    
    hits = 0
    for _, row in unique_releases.iterrows():
        # Ask ALFRED: "Was there an official release on this specific date?"
        pit_val = get_alfred_value(cfg['fred_id'], row['release_date'], FRED_API_KEY)
        
        if pit_val is not None:
            final_verified_data.append({
                'indicator_id': indicator_id,
                'timestamp': row['dt'],
                'actual': pit_val,
                'event_name': row['event']
            })
            hits += 1
    print(f"✅ {hits} Verified.")

# 4. Save to final Bronze Table
if final_verified_data:
    clean_df = pd.DataFrame(final_verified_data).rename(columns={'event_name': 'event'})
    sdf = spark.createDataFrame(clean_df)
    
    # Matching the schema of your Alpaca notebook
    sdf = sdf.withColumn("year_partition", F.year("timestamp").cast("integer")) \
             .withColumn("ingestion_ts", F.current_timestamp())
    
    sdf.write.mode("overwrite").option("overwriteSchema", "true").format("delta").save(VOLUME_PATH)
    spark.sql(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM delta.`{VOLUME_PATH}`")
    print(f"\n✨ ANALYSIS COMPLETE. High-Resolution Table ready: {TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC Harvester

# COMMAND ----------

import pandas as pd
import time
from datetime import datetime
import finnhub

# --- CONFIG ---
RAW_CSV_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/raw_finnhub_haystack.csv"
START_YEAR = 2015
END_YEAR = 2025 # Pulling up to current date

# Initialize Client (ensure API_KEY is defined in previous cell)
client = finnhub.Client(api_key=API_KEY)

print("🚀 Harvesting Raw US Macro Haystack (Month-by-Month)...")
all_raw_events = []

for yr in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        # Prevent fetching future dates in 2025
        if yr == 2025 and month > datetime.now().month:
            continue
            
        start_date = f"{yr}-{month:02d}-01"
        
        # Calculate end date logic
        if month == 12:
            end_date = f"{yr}-12-31"
        else:
            end_date = f"{yr}-{(month+1):02d}-01"
            
        print(f"📅 Fetching {yr}-{month:02d}...", end=" ")
        
        try:
            # Monthly fetch ensures we stay under result size limits
            res = client.calendar_economic(_from=start_date, to=end_date)
            
            if 'economicCalendar' in res and res['economicCalendar']:
                month_df = pd.DataFrame(res['economicCalendar'])
                
                # Filter for US and (High/Medium) Impact immediately to keep CSV lean
                us_relevant = month_df[
                    (month_df['country'] == 'US') & 
                    (month_df['impact'].isin(['high', 'medium']))
                ].copy()
                
                all_raw_events.append(us_relevant)
                print(f"✅ {len(us_relevant)} events.")
            else:
                print("⊘ No events.")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            # If we hit a rate limit despite the sleep, wait longer
            if "429" in str(e):
                print("⏳ Rate limit hit. Sleeping 10 seconds...")
                time.sleep(10)
        
        # Crucial: 1.2s delay stays under the 60 calls/minute limit
        time.sleep(1.2)

# Combine and save to CSV
if all_raw_events:
    haystack_full = pd.concat(all_raw_events)
    # Deduplicate in case month boundaries overlapped
    haystack_full = haystack_full.drop_duplicates(subset=['time', 'event', 'actual'])
    
    haystack_full.to_csv(RAW_CSV_PATH, index=False)
    print(f"\n✨ SUCCESS: Raw Haystack saved to {RAW_CSV_PATH}")
    print(f"📊 Total US High/Medium events captured: {len(haystack_full)}")
else:
    print("❌ Critical Failure: No data was captured.")

# COMMAND ----------

import pandas as pd

# Read the CSV
input_path = "/Volumes/pawan_quant_research_1/v4/market_data/raw_finnhub_haystack.csv"
output_path = "/Volumes/pawan_quant_research_1/v4/market_data/filtered_economic_indicators.csv"

df = pd.read_csv(input_path)

print(f"Original dataset: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Define the exact event names to filter for each indicator
# Based on the pattern I saw in the previous CSV and the image
filter_patterns = [
    # CPI - All Urban Consumers
    'CPI',
    'Inflation Rate',
    'Core CPI',
    'Core Inflation Rate',
    
    # PCE Price Index
    'PCE Price Index',
    'Core PCE Price Index',
    'Personal consumption expenditures',
    
    # Real GDP
    'GDP Growth Rate',
    'GDP Price Index',
    
    # Unemployment Rate
    'Unemployment Rate',
    
    # Nonfarm Payrolls
    'Non Farm Payrolls',
    'Non-Farm Payrolls',
    'Nonfarm Payrolls',
    
    # Industrial Production
    'Industrial Production',
    
    # Personal Consumption Expenditures (level)
    'Personal Spending',
    'Personal Income',
    
    # Michigan Consumer Sentiment
    'Michigan Consumer Sentiment',
    'U Mich Sentiment'
]

# Create filter mask
mask = pd.Series([False] * len(df))

# Assuming the column with event names is called 'event' (adjust if different)
event_column = 'event'  # Change this if your column name is different

for pattern in filter_patterns:
    mask |= df[event_column].str.contains(pattern, case=False, na=False)

# Apply filter
filtered_df = df[mask]

print(f"\nFiltered dataset: {len(filtered_df)} rows")
print(f"Reduction: {len(df) - len(filtered_df)} rows removed")

# Show sample of filtered events
print("\nSample of filtered events:")
print(filtered_df[event_column].value_counts().head(20))

# Save filtered data
filtered_df.to_csv(output_path, index=False)

print(f"\nFiltered data saved to: {output_path}")

# Display first few rows
display(filtered_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 1: Setup and FRED API Key
# MAGIC

# COMMAND ----------

!pip install fredapi 

# COMMAND ----------

# MAGIC %pip install fredapi

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import time
import os

# --- CONFIGURATION ---
# Use Databricks secrets or paste key here
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db" 
fred = Fred(api_key=FRED_API_KEY)

INPUT_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/filtered_economic_indicators.csv"
OUTPUT_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/economic_release_timeline_final.csv"

# Comprehensive Map handling all variations found in your CSV audit
EVENT_TO_FRED = {
    # Employment
    'Non-Farm Payrolls': 'PAYEMS', 
    'Non Farm Payrolls': 'PAYEMS',
    'United States-Employment-Non-Farm Payrolls': 'PAYEMS',
    'Unemployment Rate': 'UNRATE',
    'United States-Employment-Unemployment Rate': 'UNRATE',
    # CPI
    'CPI MM, SA': 'CPIAUCSL', 'CPI s.a': 'CPIAUCSL', 'CPI Index  SA': 'CPIAUCSL',
    'United States-CPI-CPI MM, SA': 'CPIAUCSL', 'United States-CPI-CPI Index  SA': 'CPIAUCSL',
    'Core CPI MM, SA': 'CPILFESL', 'United States-CPI-Core CPI MM, SA': 'CPILFESL',
    # GDP
    'GDP Growth Rate QoQ Adv': 'GDPC1', 'GDP Growth Rate QoQ 2nd Est': 'GDPC1', 
    'GDP Growth Rate QoQ Final': 'GDPC1',
    # PCE
    'Core PCE Price Index MM': 'PCEPILFE', 'Core PCE Price Index MoM': 'PCEPILFE',
    'United States-Consumption-Core PCE Price Index MM': 'PCEPILFE',
    # Sentiment
    'U Mich Sentiment Prelim': 'UMCSENT', 'Michigan Consumer Sentiment Prel': 'UMCSENT',
    'U Mich Sentiment Final': 'UMCSENT', 'Michigan Consumer Sentiment Final': 'UMCSENT',
    # Industrial Production
    'Industrial Production MoM': 'INDPRO', 'Industrial Production YoY': 'INDPRO'
}

def get_observation_date(release_date, series_id):
    """Determines which month the data is actually referring to."""
    if series_id == 'GDPC1':
        # Quarterly: Jan release refers to Q4 (Oct 1st)
        month = release_date.month
        if month <= 3: return datetime(release_date.year - 1, 10, 1)
        if month <= 6: return datetime(release_date.year, 1, 1)
        if month <= 9: return datetime(release_date.year, 4, 1)
        return datetime(release_date.year, 7, 1)
    else:
        # Monthly: Feb release refers to Jan data (1st of month)
        first_of_this_month = release_date.replace(day=1)
        return (first_of_this_month - timedelta(days=1)).replace(day=1)

print("✅ Setup complete. Ready to process.")

# COMMAND ----------

# MAGIC %md
# MAGIC Cell 2: The PIT Processing Engine

# COMMAND ----------

# Load and clean names
df = pd.read_csv(INPUT_PATH)
df['event'] = df['event'].str.strip()
df['release_timestamp'] = pd.to_datetime(df['time'])
df['release_date'] = df['release_timestamp'].dt.date

results = []
cache = {}

print(f"🚀 Triangulating {len(df)} rows with ALFRED...")

# Grouping by release date and event automatically handles the short/long form duplicates
for (rel_date, event_name), group in df.groupby(['release_date', 'event']):
    series_id = EVENT_TO_FRED.get(event_name)
    if not series_id: continue
    
    obs_date = get_observation_date(rel_date, series_id)
    obs_str = obs_date.strftime('%Y-%m-%d')
    cache_key = f"{series_id}_{obs_str}"
    
    try:
        if cache_key not in cache:
            # FIX: Use get_series_all_releases instead of get_series_vintages
            all_vintages = fred.get_series_all_releases(series_id)
            if all_vintages.empty: continue
            
            # Convert types to ensure matching works
            all_vintages['date'] = pd.to_datetime(all_vintages['date'])
            all_vintages['realtime_start'] = pd.to_datetime(all_vintages['realtime_start'])
            
            all_vintages = all_vintages.sort_values('realtime_start')
            all_vintages['rel_idx'] = range(len(all_vintages))
            cache[cache_key] = all_vintages
            time.sleep(0.2) # API rate limit protection

        v_data = cache[cache_key]
        
        # AUDIT FIX: Match Finnhub release date to ALFRED with a 1-day tolerance window
        # This ensures we capture releases that ALFRED might date +/- 1 day from the wire
        match = v_data[
            (v_data['date'] == pd.to_datetime(obs_date)) & 
            (v_data['realtime_start'].dt.date >= rel_date - timedelta(days=1)) & 
            (v_data['realtime_start'].dt.date <= rel_date + timedelta(days=1))
        ]
        
        if not match.empty:
            best_v = match.iloc[0]
            rel_num = int(best_v['rel_idx'])
            
            # Determine classification
            if rel_num == 0:
                label = "FIRST_RELEASE"
            elif rel_num == len(v_data) - 1:
                label = "FINAL_REVISION"
            else:
                label = f"REVISION_{rel_num}"

            for _, row in group.iterrows():
                results.append({
                    'timestamp': row['release_timestamp'],
                    'event_name': event_name,
                    'fred_series_id': series_id,
                    'observation_date': obs_date.date(),
                    'release_label': label,
                    'is_revision': rel_num > 0,
                    'finnhub_value': row['actual'],
                    'alfred_verified_level': best_v['value']
                })
                
    except Exception as e:
        print(f"Error on {event_name} for {obs_str}: {e}")
        continue

if results:
    timeline_df = pd.DataFrame(results).sort_values('timestamp')
    timeline_df = timeline_df.drop_duplicates(subset=['timestamp', 'fred_series_id', 'observation_date'])
    timeline_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✨ SUCCESS! Timeline created with {len(timeline_df)} release points.")
    display(timeline_df.head(20))
else:
    print("❌ Critical Match Failure: No data matched. Verify observation_date logic.")

# COMMAND ----------

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import time

FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db" 
fred = Fred(api_key=FRED_API_KEY)

INPUT_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/filtered_economic_indicators.csv"
OUTPUT_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/economic_release_timeline_final.csv"
UNMAPPED_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/unmapped_events.csv"

# Comprehensive event mapping
EVENT_TO_FRED = {
    # Employment
    'Non-Farm Payrolls': 'PAYEMS', 
    'Non Farm Payrolls': 'PAYEMS',
    'United States-Employment-Non-Farm Payrolls': 'PAYEMS',
    'Non Farm Payrolls Annual Revision': 'PAYEMS',
    'Unemployment Rate': 'UNRATE',
    'United States-Employment-Unemployment Rate': 'UNRATE',
    
    # CPI - All map to same series but noted for tracking
    'CPI MM, SA': 'CPIAUCSL',
    'CPI s.a': 'CPIAUCSL', 
    'CPI Index  SA': 'CPIAUCSL',
    'CPI MM NSA': 'CPIAUCSL',
    'CPI': 'CPIAUCSL',
    'CPI YY, NSA': 'CPIAUCSL',
    'United States-CPI-CPI MM, SA': 'CPIAUCSL',
    'United States-CPI-CPI Index  SA': 'CPIAUCSL',
    'United States-CPI-CPI MM NSA': 'CPIAUCSL',
    'United States-CPI-CPI YY, NSA': 'CPIAUCSL',
    'Core CPI MM, SA': 'CPILFESL',
    'Core CPI YY, NSA': 'CPILFESL',
    'United States-CPI-Core CPI MM, SA': 'CPILFESL',
    'United States-CPI-Core CPI YY, NSA': 'CPILFESL',
    'CPI Wage Earner': 'CPILEGSL',
    'United States-CPI-CPI Wage Earner': 'CPILEGSL',
    'Inflation Rate MoM': 'CPIAUCSL',
    'Inflation Rate YoY': 'CPIAUCSL',
    'Inflation Rate MoM Final': 'CPIAUCSL',
    'Core Inflation Rate MoM': 'CPILFESL',
    'Core Inflation Rate YoY': 'CPILFESL',
    'Core Inflation Rate MoM Final': 'CPILFESL',
    
    # PCE
    'Core PCE Price Index MM': 'PCEPILFE',
    'Core PCE Price Index MoM': 'PCEPILFE',
    'Core PCE Price Index YY': 'PCEPILFE',
    'Core PCE Price Index YoY': 'PCEPILFE',
    'United States-Consumption-Core PCE Price Index MM': 'PCEPILFE',
    'United States-Consumption-Core PCE Price Index YY': 'PCEPILFE',
    'PCE Price Index MoM': 'PCEPI',
    'PCE Price Index YoY': 'PCEPI',
    
    # GDP
    'GDP Growth Rate QoQ Adv': 'GDPC1',
    'GDP Growth Rate QoQ 2nd Est': 'GDPC1',
    'GDP Growth Rate QoQ Final': 'GDPC1',
    'GDP Price Index QoQ Adv': 'GDPDEF',
    'GDP Price Index QoQ 2nd Est': 'GDPDEF',
    'GDP Price Index QoQ Final': 'GDPDEF',
    
    # Sentiment
    'U Mich Sentiment Prelim': 'UMCSENT',
    'U Mich Sentiment Final': 'UMCSENT',
    'Michigan Consumer Sentiment Prel': 'UMCSENT',
    'Michigan Consumer Sentiment Final': 'UMCSENT',
    'United States-Michigan subindexes Prelim-U Mich Sentiment Prelim': 'UMCSENT',
    'United States-Michigan subindexes Final-U Mich Sentiment Final': 'UMCSENT',
    
    # Industrial Production
    'Industrial Production MoM': 'INDPRO',
    'Industrial Production YoY': 'INDPRO',
    'United States-Industrial production-Industrial Production YoY': 'INDPRO',
    
    # Personal Income/Spending
    'Personal Income MoM': 'PI',
    'Personal Spending MoM': 'PCE',
    
    # Real Weekly Earnings
    'United States-CPI-Real Weekly Earnings MM': 'CES0500000032',
}

# Series that need MoM calculation
NEEDS_MOM_CALC = ['PAYEMS']

# Series that report current month (not prior month)
CURRENT_MONTH_SERIES = ['UMCSENT']

def get_observation_date(release_date, series_id):
    """Calculate observation date based on series type."""
    if series_id in CURRENT_MONTH_SERIES:
        # Michigan Sentiment: Feb 13 prelim measures February
        return release_date.replace(day=1)
    elif series_id in ['GDPC1', 'GDPDEF']:
        # Quarterly data
        month = release_date.month
        year = release_date.year
        if month <= 3: return datetime(year - 1, 10, 1)
        if month <= 6: return datetime(year, 1, 1)
        if month <= 9: return datetime(year, 4, 1)
        return datetime(year, 7, 1)
    else:
        # Monthly: Feb 6 release measures January
        first_of_month = release_date.replace(day=1)
        prior_month = first_of_month - timedelta(days=1)
        return prior_month.replace(day=1)

def calculate_mom_changes(all_releases, obs_date):
    """Calculate MoM changes for employment data."""
    obs_datetime = pd.to_datetime(obs_date)
    prior_month = (obs_datetime - timedelta(days=1)).replace(day=1)
    
    obs_vintages = all_releases[all_releases['date'] == obs_datetime].copy()
    if len(obs_vintages) == 0:
        return pd.DataFrame()
    
    obs_vintages = obs_vintages.sort_values('realtime_start').reset_index(drop=True)
    
    changes = []
    for idx, row in obs_vintages.iterrows():
        vintage_date = row['realtime_start']
        current_value = row['value']
        
        prior_vintages = all_releases[
            (all_releases['date'] == prior_month) & 
            (all_releases['realtime_start'] <= vintage_date)
        ].sort_values('realtime_start')
        
        if len(prior_vintages) > 0:
            prior_value = prior_vintages.iloc[-1]['value']
            change = current_value - prior_value
            
            changes.append({
                'date': obs_datetime,
                'realtime_start': vintage_date,
                'level': current_value,
                'change': change,
                'vintage_num': idx
            })
    
    return pd.DataFrame(changes)


print("✅ Loading data...")
df = pd.read_csv(INPUT_PATH)
df['event'] = df['event'].str.strip()
df['release_timestamp'] = pd.to_datetime(df['time'])
df['release_date'] = df['release_timestamp'].dt.date

results = []
unmapped = []
cache = {}
api_calls = 0

print(f"🚀 Processing {len(df)} rows...\n")

for (rel_date, event_name), group in df.groupby(['release_date', 'event']):
    series_id = EVENT_TO_FRED.get(event_name)
    
    if not series_id:
        for _, row in group.iterrows():
            unmapped.append({'event': event_name, 'date': rel_date, 'count': len(group)})
        continue
    
    obs_date = get_observation_date(pd.Timestamp(rel_date), series_id)
    obs_str = obs_date.strftime('%Y-%m-%d')
    cache_key = f"{series_id}_{obs_str}"
    
    try:
        if cache_key not in cache:
            # FIX: Use correct method to retrieve all historical release vintages
            all_releases = fred.get_series_all_releases(series_id)
            api_calls += 1
            
            all_releases['date'] = pd.to_datetime(all_releases['date'])
            all_releases['realtime_start'] = pd.to_datetime(all_releases['realtime_start'])
            all_releases['value'] = pd.to_numeric(all_releases['value'], errors='coerce')
            
            if series_id in NEEDS_MOM_CALC:
                vintages = calculate_mom_changes(all_releases, obs_date)
                match_col = 'change'
            else:
                vintages = all_releases[all_releases['date'] == pd.to_datetime(obs_date)].copy()
                vintages = vintages.sort_values('realtime_start').reset_index(drop=True)
                vintages['vintage_num'] = range(len(vintages))
                match_col = 'value'
            
            cache[cache_key] = (vintages, match_col)
            if api_calls % 10 == 0:
                print(f"  Progress: {api_calls} API calls, {len(results)} matched...")
            time.sleep(0.55)
        
        vintages, match_col = cache[cache_key]
        if len(vintages) == 0: continue
        
        for _, row in group.iterrows():
            finnhub_value = row['actual']
            if pd.isna(finnhub_value): continue
            
            # AUDIT FIX: Filter vintages within a 1-day tolerance of the FinnHub release date
            # This captures the Oct 6, 2023 release and others with minor timestamp drift
            time_matched_vintages = vintages[
                (vintages['realtime_start'].dt.date >= rel_date - timedelta(days=1)) & 
                (vintages['realtime_start'].dt.date <= rel_date + timedelta(days=1))
            ].copy()
            
            if time_matched_vintages.empty: continue
            
            time_matched_vintages['value_diff'] = (time_matched_vintages[match_col] - finnhub_value).abs()
            best_match = time_matched_vintages.loc[time_matched_vintages['value_diff'].idxmin()]
            
            # Tolerance Check
            tolerance = 10 if series_id in NEEDS_MOM_CALC else 5
            if best_match['value_diff'] > tolerance: continue
            
            rel_num = int(best_match['vintage_num'])
            total_vintages = len(vintages)
            
            if rel_num == 0: label = "FIRST_RELEASE"
            elif rel_num == total_vintages - 1: label = "FINAL_REVISION"
            else: label = f"REVISION_{rel_num}"
            
            first_vintage_date = vintages.iloc[0]['realtime_start']
            current_vintage_date = best_match['realtime_start']
            days_since = (current_vintage_date - first_vintage_date).days
            
            results.append({
                'timestamp': row['release_timestamp'],
                'event_name': event_name,
                'fred_series_id': series_id,
                'observation_date': obs_date.date(),
                'release_label': label,
                'is_first_release': rel_num == 0,
                'revision_number': rel_num,
                'total_revisions': total_vintages - 1,
                'days_since_first_release': days_since,
                'data_age_years': round(days_since / 365.25, 2),
                'is_stale_capture': (days_since / 365.25) > 1.0,
                'finnhub_value': finnhub_value,
                'alfred_value': best_match[match_col],
                'value_diff': round(best_match['value_diff'], 2),
                'vintage_date': current_vintage_date.date(),
                'first_release_date': first_vintage_date.date()
            })
        
    except Exception as e:
        print(f"⚠️  Error on {event_name} for {obs_str}: {str(e)}")
        continue

print(f"\n✅ Processing complete!\n")

# Save results
if results:
    timeline_df = pd.DataFrame(results).sort_values('timestamp')
    timeline_df = timeline_df.drop_duplicates(
        subset=['timestamp', 'fred_series_id', 'observation_date', 'release_label']
    )
    timeline_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✨ Created timeline with {len(timeline_df)} release points")
    print(f"\n📊 Summary:")
    print(f"   Matched: {len(timeline_df)} / {len(df)} ({len(timeline_df)/len(df)*100:.1f}%)")
    print(f"   First Releases: {timeline_df['is_first_release'].sum()}")
    print(f"   Revisions: {(~timeline_df['is_first_release']).sum()}")
    print(f"   Stale captures (>1yr old): {timeline_df['is_stale_capture'].sum()}")
    print(f"\n📈 Release Types:")
    print(timeline_df['release_label'].value_counts().head(10))
    print(f"\n🎯 Match Quality:")
    print(f"   Mean diff: {timeline_df['value_diff'].mean():.2f}")
    print(f"   Median diff: {timeline_df['value_diff'].median():.2f}")
    
    display(timeline_df.head(20))
    
    print(f"\n💾 Saved to: {OUTPUT_PATH}")

# Save unmapped events
if unmapped:
    unmapped_df = pd.DataFrame(unmapped).drop_duplicates()
    unmapped_summary = unmapped_df.groupby('event')['count'].sum().sort_values(ascending=False)
    unmapped_df.to_csv(UNMAPPED_PATH, index=False)
    print(f"\n⚠️  Unmapped Events ({len(unmapped_summary)} unique):")
    print(unmapped_summary.head(20))
    print(f"\n💾 Saved to: {UNMAPPED_PATH}")


# COMMAND ----------

import pandas as pd
import numpy as np

# 1. Define Paths
input_path = "/Volumes/pawan_quant_research_1/v4/market_data/filtered_economic_indicators.csv"
output_path = "/Volumes/pawan_quant_research_1/v4/market_data/economic_indicators_with_revisions.csv"

# 2. Load the data
df = pd.read_csv(input_path)

# 3. Pre-processing: Convert time and handle duplicates
df['time'] = pd.to_datetime(df['time'])

# Normalize Event Names: 
# This removes prefixes like "United States-Employment-" so that 
# "Non-Farm Payrolls" and "United States-Employment-Non-Farm Payrolls" match.
df['event_clean'] = df['event'].str.replace('United States-', '', case=False)
df['event_clean'] = df['event_clean'].str.split('-').str[-1].str.strip()

# 4. Sort by event and time
# This ensures that the report from February follows the report from January
df = df.sort_values(by=['event_clean', 'time'])

# 5. Identify Revisions using the "Chain-Link" method
# Logic: We look at the 'actual' value from the PREVIOUS report (shift 1) 
# and compare it to the 'prev' value stated in the CURRENT report.
df['last_month_actual'] = df.groupby('event_clean')['actual'].shift(1)

# A revision is detected if 'prev' in current row != 'actual' in previous row
# We use np.isclose to handle small floating point differences
df['was_revised'] = ~np.isclose(df['prev'], df['last_month_actual'], equal_nan=True) & df['last_month_actual'].notna()

# Calculate the magnitude of the revision
df['revision_delta'] = df['prev'] - df['last_month_actual']

# 6. Formatting for the final CSV
# Create a descriptive column for clarity
df['revision_note'] = ""
df.loc[df['was_revised'], 'revision_note'] = (
    "Original: " + df['last_month_actual'].astype(str) + 
    " -> Revised to: " + df['prev'].astype(str)
)

# 7. Save the processed file back to the Volume
# We keep the original columns and add our new analysis columns
df.to_csv(output_path, index=False)

print(f"File saved successfully to: {output_path}")

# Display a sample of the revisions found
display(df[df['was_revised'] == True][['time', 'event_clean', 'last_month_actual', 'prev', 'revision_delta']].head(10))

# COMMAND ----------

import pandas as pd
import os

# Read the CSV file
input_file = "/Volumes/pawan_quant_research_1/v4/market_data/filtered_economic_indicators.csv"
df = pd.read_csv(input_file)

# Define mappings of series IDs to their corresponding event names
# Each series can have multiple event name variations
series_mappings = {
    'CPIAUCSL': [
        'CPI MM',
        'CPI YY',
        'Core CPI MM',
        'Core CPI YY',
        'CPI s.a',
        'Core Inflation Rate',
        'Inflation Rate',
        'United States-CPI',
        'CPI Wage',
    ],
    
    'PCEPI': [
        'PCE Price Index',
        'Core PCE Price Index',
        'United States-Consumption-Core PCE Price Index',
    ],
    
    'GDPC1': [
        'GDP Growth',
        'GDP Price Index',
        'United States-GDP',
    ],
    
    'UNRATE': [
        'Unemployment Rate',
        'United States-Employment-Unemployment Rate',
    ],
    
    'PAYEMS': [
        'Non-Farm Payrolls',
        'United States-Employment-Non-Farm Payrolls',
        'Non Farm Pay',
    ],
    
    'INDPRO': [
        'Industrial Production',
        'Industrial Prod',
    ],
    
    'PCE': [
        'Personal Income',
        'Personal Spending',
        'Personal Spe',
    ],
    
    'UMCSENT': [
        'U Mich Sentiment',
        'Michigan Consumer Sentiment',
        'Michigan Con',
        'United States-Michigan subindexes',
    ]
}

# Create output directory
output_dir = "/Volumes/pawan_quant_research_1/v4/market_data/individual_series"
os.makedirs(output_dir, exist_ok=True)

# Create separate CSV for each series
for series_id, keywords in series_mappings.items():
    # Build a regex pattern that matches any of the keywords
    pattern = '|'.join([f'({keyword})' for keyword in keywords])
    
    # Filter rows that match any of the keywords in the 'event' column
    mask = df['event'].str.contains(pattern, case=False, na=False, regex=True)
    series_df = df[mask].copy()
    
    if not series_df.empty:
        # Sort by time if time column exists
        if 'time' in series_df.columns:
            series_df['time'] = pd.to_datetime(series_df['time'])
            series_df = series_df.sort_values('time')
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{series_id}.csv")
        series_df.to_csv(output_file, index=False)
        print(f"✓ Created: {series_id}.csv ({len(series_df)} rows)")
        
        # Show sample of unique events captured
        unique_events = series_df['event'].unique()
        print(f"  Events included: {', '.join(unique_events[:3])}" + 
              (f"... (+{len(unique_events)-3} more)" if len(unique_events) > 3 else ""))
    else:
        print(f"✗ Warning: No data found for {series_id}")

print(f"\n{'='*60}")
print(f"All series CSV files created in:")
print(f"{output_dir}")
print(f"{'='*60}")

# COMMAND ----------

"""
ALFRED Integration Utility
Combines ALFRED vintage dates with precise release times from historical data
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta, time as dt_time

# FRED/ALFRED API Configuration
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"
ALFRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Load release schedule metadata
with open('/Volumes/pawan_quant_research_1/v4/market_data/release_schedule_report.json', 'r') as f:
    RELEASE_SCHEDULES = json.load(f)


def get_release_time_for_date(series_id, release_date, schedule=None):
    """
    Get the expected release time for a given date based on historical patterns
    
    Args:
        series_id: FRED series ID
        release_date: Date of release (datetime.date or string)
        schedule: Pre-loaded schedule dict (optional)
    
    Returns:
        datetime.time object with expected release time
    """
    if schedule is None:
        schedule = RELEASE_SCHEDULES.get(series_id)
    
    if not schedule:
        return dt_time(12, 30)  # Default to 12:30 PM EST
    
    # Get typical time
    typical_time_str = schedule.get('typical_release_time', '12:30:00')
    hour, minute, second = map(int, typical_time_str.split(':'))
    
    # Check if date falls in DST period (March-November)
    if isinstance(release_date, str):
        release_date = datetime.strptime(release_date, '%Y-%m-%d').date()
    
    month = release_date.month if hasattr(release_date, 'month') else release_date[1]
    
    # Adjust for DST if needed
    dst_info = schedule.get('dst_transitions', {})
    if dst_info.get('has_time_variations'):
        # DST typically runs March-November
        if 3 <= month <= 11:
            # Use EDT time (typically 1 hour later)
            time_dist = schedule.get('time_distribution', {})
            # Find the later time option
            times = sorted([datetime.strptime(t, '%H:%M:%S').time() for t in time_dist.keys()])
            if len(times) > 1:
                return times[-1]  # Use later time during DST
    
    return dt_time(hour, minute, second)


def fetch_alfred_with_times(series_id, start_date='2015-01-01', end_date=None):
    """
    Fetch ALFRED vintage data and add precise release times
    
    Args:
        series_id: FRED series ID
        start_date: Start date for real-time period
        end_date: End date (default: today)
    
    Returns:
        DataFrame with vintage data including precise release times
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch ALFRED data
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'realtime_start': start_date,
        'realtime_end': end_date
    }
    
    print(f"Fetching ALFRED data for {series_id}...")
    
    try:
        response = requests.get(ALFRED_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' not in data:
            print(f"No observations found for {series_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['observations'])
        df['realtime_start'] = pd.to_datetime(df['realtime_start'])
        df['realtime_end'] = pd.to_datetime(df['realtime_end'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Add release times based on schedule
        schedule = RELEASE_SCHEDULES.get(series_id)
        
        df['release_time'] = df['realtime_start'].apply(
            lambda x: get_release_time_for_date(series_id, x.date(), schedule)
        )
        
        # Create full release datetime
        df['release_datetime'] = df.apply(
            lambda row: datetime.combine(row['realtime_start'].date(), row['release_time']),
            axis=1
        )
        
        # Identify first releases (when data first became available)
        df['is_first_release'] = df.groupby('date')['realtime_start'].transform('min') == df['realtime_start']
        
        print(f"✓ Retrieved {len(df)} observations ({df['is_first_release'].sum()} first releases)")
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching ALFRED data: {e}")
        return pd.DataFrame()


def get_first_releases_only(series_id, start_date='2015-01-01', end_date=None):
    """
    Get only the first release values with precise times
    
    This is the most useful function for event studies
    """
    df = fetch_alfred_with_times(series_id, start_date, end_date)
    
    if df.empty:
        return df
    
    # Filter to first releases only
    first_releases = df[df['is_first_release']].copy()
    
    # Sort by date
    first_releases = first_releases.sort_values('date')
    
    # Select relevant columns
    result = first_releases[[
        'date',
        'value',
        'realtime_start',
        'release_datetime',
        'release_time'
    ]].copy()
    
    result.columns = [
        'data_period',
        'value',
        'release_date',
        'release_datetime',
        'release_time'
    ]
    
    return result


def create_event_study_dataset(series_list, start_date='2020-01-01', output_file=None):
    """
    Create a combined dataset for multiple series with first release info
    
    Args:
        series_list: List of FRED series IDs
        start_date: Start date for data
        output_file: Optional CSV file to save results
    
    Returns:
        Dictionary of DataFrames, one per series
    """
    results = {}
    
    print("=" * 80)
    print("CREATING EVENT STUDY DATASET")
    print("=" * 80)
    print()
    
    for series_id in series_list:
        print(f"Processing {series_id}...")
        df = get_first_releases_only(series_id, start_date)
        
        if not df.empty:
            results[series_id] = df
            print(f"  ✓ {len(df)} first releases retrieved")
            
            if output_file:
                csv_file = output_file.replace('.csv', f'_{series_id}.csv')
                df.to_csv(csv_file, index=False)
                print(f"  ✓ Saved to {csv_file}")
        else:
            print(f"  ✗ No data available")
        
        print()
    
    return results


def get_expected_release_schedule(series_id, year=2026, save_to_csv=True):
    """
    Generate expected release schedule for a given year
    
    Useful for planning ahead when you know ALFRED won't have the data yet
    """
    schedule = RELEASE_SCHEDULES.get(series_id)
    
    if not schedule:
        print(f"No schedule found for {series_id}")
        return pd.DataFrame()
    
    freq = schedule['frequency']
    typical_day = schedule['median_day_of_month']
    typical_time = schedule['typical_release_time']
    
    releases = []
    
    if freq == 'monthly':
        for month in range(1, 13):
            try:
                release_date = datetime(year, month, typical_day)
                release_time = get_release_time_for_date(series_id, release_date.date(), schedule)
                release_datetime = datetime.combine(release_date.date(), release_time)
                
                releases.append({
                    'series_id': series_id,
                    'month': month,
                    'expected_date': release_date.date(),
                    'expected_time': release_time,
                    'expected_datetime': release_datetime,
                    'confidence': 'high' if schedule['total_releases'] > 50 else 'medium'
                })
            except ValueError:
                # Handle months without that day (e.g., Feb 30)
                pass
    
    elif freq == 'quarterly':
        for quarter in [1, 4, 7, 10]:  # Jan, Apr, Jul, Oct
            try:
                release_date = datetime(year, quarter + 1, typical_day)  # Released month after quarter
                release_time = get_release_time_for_date(series_id, release_date.date(), schedule)
                release_datetime = datetime.combine(release_date.date(), release_time)
                
                releases.append({
                    'series_id': series_id,
                    'quarter': f'Q{(quarter-1)//3 + 1}',
                    'expected_date': release_date.date(),
                    'expected_time': release_time,
                    'expected_datetime': release_datetime,
                    'confidence': 'medium'
                })
            except ValueError:
                pass
    
    df = pd.DataFrame(releases)
    
    if save_to_csv and not df.empty:
        filename = f'{series_id}_expected_releases_{year}.csv'
        df.to_csv(filename, index=False)
        print(f"Expected releases saved to {filename}")
    
    return df


# Example usage functions
def example_get_payems_first_releases():
    """Example: Get Non-Farm Payrolls first releases with exact times"""
    print("\nExample: Fetching PAYEMS first releases...")
    df = get_first_releases_only('PAYEMS', start_date='2020-01-01')
    
    if not df.empty:
        print("\nFirst 10 releases:")
        print(df.head(10).to_string(index=False))
        
        print(f"\nTotal first releases: {len(df)}")
        print(f"Date range: {df['data_period'].min()} to {df['data_period'].max()}")
    
    return df


def example_create_multi_series_dataset():
    """Example: Create dataset for multiple series"""
    series = ['PAYEMS', 'UNRATE', 'CPIAUCSL']
    results = create_event_study_dataset(series, start_date='2022-01-01', output_file='event_study.csv')
    return results


def example_predict_2026_releases():
    """Example: Generate expected release schedule for 2026"""
    print("\nGenerating expected release schedule for 2026...")
    
    for series_id in ['PAYEMS', 'CPIAUCSL', 'GDPC1']:
        print(f"\n{series_id}:")
        df = get_expected_release_schedule(series_id, year=2026, save_to_csv=False)
        if not df.empty:
            print(df.to_string(index=False))


if __name__ == "__main__":
    print("=" * 80)
    print("ALFRED + RELEASE TIME INTEGRATION UTILITY")
    print("=" * 80)
    
    # Note: The actual ALFRED API calls will only work in an environment with network access
    # In this demo, we show the structure and methodology
    
    print("\n📊 Available Functions:")
    print("  1. get_first_releases_only() - Get first releases with exact times")
    print("  2. create_event_study_dataset() - Multi-series dataset for event studies")
    print("  3. get_expected_release_schedule() - Predict future release dates")
    print("  4. get_release_time_for_date() - Get expected time for specific date")
    
    print("\n" + "=" * 80)
    print("EXAMPLE: Expected Release Schedule for 2026")
    print("=" * 80)
    example_predict_2026_releases()
    
    print("\n" + "=" * 80)
    print("RELEASE SCHEDULE SUMMARY")
    print("=" * 80)
    
    for series_id, schedule in RELEASE_SCHEDULES.items():
        print(f"\n{series_id}:")
        print(f"  Frequency: {schedule['frequency']}")
        print(f"  Typical time: {schedule['typical_release_time']}")
        print(f"  Typical day: {schedule['median_day_of_month']} (median)")
        print(f"  Typical weekday: {schedule['typical_weekday']}")
        if schedule['anomalies']:
            print(f"  ⚠ Historical anomalies: {len(schedule['anomalies'])}")



# This creates separate CSVs for each series with first releases + times
series_list = ['PAYEMS', 'UNRATE', 'CPIAUCSL', 'PCEPI', 'GDPC1', 'INDPRO', 'PCE', 'UMCSENT']

datasets = create_event_study_dataset(
    series_list=series_list,
    start_date='2015-01-01',  # or your preferred start date
    output_file='/Volumes/pawan_quant_research_1/v4/market_data/first_releases.csv'
)

# This creates:
# - first_releases_PAYEMS.csv
# - first_releases_UNRATE.csv
# - first_releases_CPIAUCSL.csv
# ... etc

# COMMAND ----------

# MAGIC %md
# MAGIC # Economic Indicator First Release Data - ALFRED Integration
# MAGIC
# MAGIC ## 📊 Overview
# MAGIC
# MAGIC This dataset contains **first release** values for major US economic indicators from the Federal Reserve's ALFRED (Archival Federal Reserve Economic Data) system, enhanced with precise release times derived from historical analysis.
# MAGIC
# MAGIC ### What is "First Release"?
# MAGIC
# MAGIC When economic agencies publish data, they often release:
# MAGIC 1. **First/Advance estimate** (what we capture here)
# MAGIC 2. **Second/Revised estimate** (weeks later)
# MAGIC 3. **Final estimate** (months later)
# MAGIC
# MAGIC This dataset captures only the **first time each data point became public**, which is critical for:
# MAGIC - Event studies (market reaction analysis)
# MAGIC - Nowcasting models
# MAGIC - Real-time trading strategies
# MAGIC - Economic forecasting validation
# MAGIC
# MAGIC ## 📁 Files Included
# MAGIC
# MAGIC | File | Series ID | Indicator | Records | Date Range |
# MAGIC |------|-----------|-----------|---------|------------|
# MAGIC | `first_releases_PAYEMS.csv` | PAYEMS | Non-Farm Payrolls (Employment) | 1,046 | 1939-01 to 2026-01 |
# MAGIC | `first_releases_UNRATE.csv` | UNRATE | Unemployment Rate | 938 | 1948-01 to 2026-01 |
# MAGIC | `first_releases_CPIAUCSL.csv` | CPIAUCSL | Consumer Price Index (CPI) | 950 | 1947-01 to 2026-01 |
# MAGIC | `first_releases_PCEPI.csv` | PCEPI | PCE Price Index | 804 | 1959-01 to 2025-11 |
# MAGIC | `first_releases_PCE.csv` | PCE | Personal Consumption Expenditures | 804 | 1959-01 to 2025-11 |
# MAGIC | `first_releases_INDPRO.csv` | INDPRO | Industrial Production Index | 1,285 | 1919-01 to 2025-12 |
# MAGIC | `first_releases_GDPC1.csv` | GDPC1 | Real GDP (Quarterly) | 316 | 1947-Q1 to 2025-Q3 |
# MAGIC | `first_releases_UMCSENT.csv` | UMCSENT | Consumer Sentiment Index | 879 | 1952-11 to 2025-12 |
# MAGIC
# MAGIC **Total Records: 7,022 first release observations**
# MAGIC
# MAGIC ## 📋 Data Schema
# MAGIC
# MAGIC Each CSV contains the following columns:
# MAGIC
# MAGIC | Column | Type | Description | Example |
# MAGIC |--------|------|-------------|---------|
# MAGIC | `data_period` | Date | The month/quarter the data covers | 2024-01-01 |
# MAGIC | `value` | Float | The first released value | 156419 |
# MAGIC | `release_date` | Date | Date when data was first released (from ALFRED) | 2024-02-02 |
# MAGIC | `release_datetime` | Datetime | Precise date and time of release | 2024-02-02 12:30:00 |
# MAGIC | `release_time` | Time | Time of day when released | 12:30:00 |
# MAGIC
# MAGIC ### Important Notes
# MAGIC
# MAGIC - **Historical Data**: Early records (pre-2015) show `release_date` as 2015-01-01 because this is when ALFRED first ingested historical data into their system
# MAGIC - **Missing Values**: Some values show "." indicating data was not available or revised away
# MAGIC - **Quarterly Data**: GDPC1 shows monthly releases (advance, 2nd estimate, final) for quarterly data
# MAGIC
# MAGIC ## 🕐 **CRITICAL: Timezone Information**
# MAGIC
# MAGIC ### Release Times Are in US Eastern Time (ET)
# MAGIC
# MAGIC **ALL release times in these CSVs are in US Eastern Time, which observes Daylight Saving Time (DST).**
# MAGIC
# MAGIC #### Understanding EST vs EDT
# MAGIC
# MAGIC The United States Eastern timezone switches between:
# MAGIC
# MAGIC - **EST (Eastern Standard Time)**: UTC-5:00 (November - March)
# MAGIC - **EDT (Eastern Daylight Time)**: UTC-4:00 (March - November)
# MAGIC
# MAGIC **DST Transitions:**
# MAGIC - **Spring Forward**: Second Sunday in March at 2:00 AM → clocks move to 3:00 AM (EST → EDT)
# MAGIC - **Fall Back**: First Sunday in November at 2:00 AM → clocks move back to 1:00 AM (EDT → EST)
# MAGIC
# MAGIC ### Release Time Patterns by Series
# MAGIC
# MAGIC | Series | Standard Time (EST) | Daylight Time (EDT) | Notes |
# MAGIC |--------|---------------------|---------------------|-------|
# MAGIC | **PAYEMS** | 12:30 (7:30 AM PST) | 14:00 (8:30 AM PST) | Employment data - First Friday |
# MAGIC | **UNRATE** | 12:30 (7:30 AM PST) | 13:30 (8:30 AM PST) | Released with PAYEMS |
# MAGIC | **CPIAUCSL** | 12:30 (7:30 AM PST) | 13:30 (8:30 AM PST) | Mid-month release |
# MAGIC | **PCEPI** | 12:30 (7:30 AM PST) | 15:00 (10:30 AM PST) | End of month |
# MAGIC | **PCE** | 12:30 (7:30 AM PST) | 13:30 (8:30 AM PST) | End of month |
# MAGIC | **INDPRO** | 13:15 (8:15 AM PST) | 14:15 (9:15 AM PST) | Mid-month Friday |
# MAGIC | **GDPC1** | 12:30 (7:30 AM PST) | 13:30 (8:30 AM PST) | End of month |
# MAGIC | **UMCSENT** | 14:00 (9:00 AM PST) | 15:00 (10:00 AM PST) | Mid and end of month |
# MAGIC
# MAGIC ### Why Release Times Vary
# MAGIC
# MAGIC Economic releases are scheduled in **Eastern Time**, so:
# MAGIC - **12:30 ET** becomes **12:30 EST** (UTC-5) in winter OR **12:30 EDT** (UTC-4) in summer
# MAGIC - When converting to UTC, you must check whether the release date falls in DST period
# MAGIC
# MAGIC **Example:**
# MAGIC - January 2024 release at 12:30 ET = 12:30 EST = 17:30 UTC
# MAGIC - July 2024 release at 12:30 ET = 12:30 EDT = 16:30 UTC
# MAGIC
# MAGIC The times in this dataset already show the **wall clock time** (what a person in New York would see on their watch), which automatically includes DST adjustments.
# MAGIC
# MAGIC ## 🌍 Converting to Other Timezones
# MAGIC
# MAGIC ### To UTC (Coordinated Universal Time)
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from datetime import datetime
# MAGIC import pytz
# MAGIC
# MAGIC # Read the data
# MAGIC df = pd.read_csv('first_releases_PAYEMS.csv')
# MAGIC
# MAGIC # Parse datetime with Eastern timezone
# MAGIC eastern = pytz.timezone('US/Eastern')
# MAGIC df['release_datetime_et'] = pd.to_datetime(df['release_datetime'])
# MAGIC df['release_datetime_et'] = df['release_datetime_et'].dt.tz_localize(eastern, ambiguous='infer')
# MAGIC
# MAGIC # Convert to UTC
# MAGIC df['release_datetime_utc'] = df['release_datetime_et'].dt.tz_convert('UTC')
# MAGIC
# MAGIC # Examples:
# MAGIC # 2024-01-05 12:30:00 EST → 2024-01-05 17:30:00 UTC
# MAGIC # 2024-07-05 14:00:00 EDT → 2024-07-05 18:00:00 UTC
# MAGIC ```
# MAGIC
# MAGIC ### To Other US Timezones
# MAGIC
# MAGIC ```python
# MAGIC # Convert to Pacific Time
# MAGIC df['release_datetime_pt'] = df['release_datetime_et'].dt.tz_convert('US/Pacific')
# MAGIC
# MAGIC # Convert to Central Time
# MAGIC df['release_datetime_ct'] = df['release_datetime_et'].dt.tz_convert('US/Central')
# MAGIC
# MAGIC # Convert to Mountain Time
# MAGIC df['release_datetime_mt'] = df['release_datetime_et'].dt.tz_convert('US/Mountain')
# MAGIC ```
# MAGIC
# MAGIC ### Quick Reference: US Timezone Offsets
# MAGIC
# MAGIC | Timezone | Standard Time | Daylight Time | UTC Offset (Std) | UTC Offset (DST) |
# MAGIC |----------|---------------|---------------|------------------|------------------|
# MAGIC | Eastern (ET) | EST | EDT | UTC-5 | UTC-4 |
# MAGIC | Central (CT) | CST | CDT | UTC-6 | UTC-5 |
# MAGIC | Mountain (MT) | MST | MDT | UTC-7 | UTC-6 |
# MAGIC | Pacific (PT) | PST | PDT | UTC-8 | UTC-7 |
# MAGIC
# MAGIC ### For International Markets
# MAGIC
# MAGIC ```python
# MAGIC # Convert to London (GMT/BST)
# MAGIC df['release_datetime_london'] = df['release_datetime_et'].dt.tz_convert('Europe/London')
# MAGIC
# MAGIC # Convert to Tokyo (JST - no DST)
# MAGIC df['release_datetime_tokyo'] = df['release_datetime_et'].dt.tz_convert('Asia/Tokyo')
# MAGIC
# MAGIC # Convert to Hong Kong (HKT - no DST)
# MAGIC df['release_datetime_hk'] = df['release_datetime_et'].dt.tz_convert('Asia/Hong_Kong')
# MAGIC ```
# MAGIC
# MAGIC ## 🔄 Combining with Market Data
# MAGIC
# MAGIC ### Example: Merging with Stock Market Data
# MAGIC
# MAGIC When combining with market data, ensure timezone consistency:
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC import pytz
# MAGIC
# MAGIC # Load economic releases
# MAGIC econ_df = pd.read_csv('first_releases_PAYEMS.csv')
# MAGIC eastern = pytz.timezone('US/Eastern')
# MAGIC econ_df['release_datetime'] = pd.to_datetime(econ_df['release_datetime'])
# MAGIC econ_df['release_datetime'] = econ_df['release_datetime'].dt.tz_localize(eastern, ambiguous='infer')
# MAGIC
# MAGIC # Load market data (assuming it's in UTC or has timezone info)
# MAGIC market_df = pd.read_parquet('spy_1min_bars.parquet')
# MAGIC
# MAGIC # If market data is in UTC, convert economic releases to UTC
# MAGIC econ_df['release_datetime_utc'] = econ_df['release_datetime'].dt.tz_convert('UTC')
# MAGIC
# MAGIC # Now you can merge or filter based on matching timezones
# MAGIC # Example: Get market data 30 minutes before and after each release
# MAGIC for idx, release in econ_df.iterrows():
# MAGIC     release_time = release['release_datetime_utc']
# MAGIC     start_time = release_time - pd.Timedelta(minutes=30)
# MAGIC     end_time = release_time + pd.Timedelta(minutes=30)
# MAGIC     
# MAGIC     event_data = market_df[
# MAGIC         (market_df['timestamp'] >= start_time) & 
# MAGIC         (market_df['timestamp'] <= end_time)
# MAGIC     ]
# MAGIC     # Analyze market reaction...
# MAGIC ```
# MAGIC
# MAGIC ### Common Timezone Scenarios
# MAGIC
# MAGIC | Your Data Source | Timezone | Conversion Needed |
# MAGIC |------------------|----------|-------------------|
# MAGIC | Alpaca API | UTC | Convert releases to UTC |
# MAGIC | IEX Cloud | UTC | Convert releases to UTC |
# MAGIC | Polygon.io | US/Eastern or UTC | Check API docs; may need no conversion |
# MAGIC | Yahoo Finance | Market hours (US/Eastern) | Usually no conversion needed |
# MAGIC | Crypto exchanges | Usually UTC | Convert releases to UTC |
# MAGIC | European markets | CET/CEST | Convert releases to Europe/Paris |
# MAGIC | Asian markets | Various | Convert to Asia/Tokyo, Asia/Hong_Kong, etc. |
# MAGIC
# MAGIC ## ⚠️ Important Considerations
# MAGIC
# MAGIC ### 1. Pre-Market vs Market Hours
# MAGIC
# MAGIC Most economic indicators are released **at 8:30 AM ET** (which appears as 12:30 or 13:30 in 24-hour format depending on DST):
# MAGIC - This is **before** regular US stock market hours (9:30 AM - 4:00 PM ET)
# MAGIC - But **during** pre-market trading (4:00 AM - 9:30 AM ET)
# MAGIC
# MAGIC ### 2. Daylight Saving Time Edge Cases
# MAGIC
# MAGIC **Be careful with DST transition dates:**
# MAGIC
# MAGIC ```python
# MAGIC # Example: DST transition in Spring 2024
# MAGIC # March 10, 2024: 2:00 AM EST → 3:00 AM EDT
# MAGIC
# MAGIC # A release at "12:30 ET" on March 10, 2024 is actually:
# MAGIC # 12:30 EDT = UTC-4 = 16:30 UTC
# MAGIC
# MAGIC # To handle this automatically:
# MAGIC df['release_datetime'] = pd.to_datetime(df['release_datetime'])
# MAGIC df['release_datetime'] = df['release_datetime'].dt.tz_localize(
# MAGIC     'US/Eastern', 
# MAGIC     ambiguous='infer',  # Automatically infer DST
# MAGIC     nonexistent='shift_forward'  # Handle the missing hour in spring
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### 3. Holiday Adjustments
# MAGIC
# MAGIC Economic releases are typically **delayed** when they would fall on:
# MAGIC - Federal holidays (New Year's Day, Independence Day, Thanksgiving, Christmas)
# MAGIC - Weekends
# MAGIC
# MAGIC The `release_date` in these CSVs already reflects the actual release date (after any holiday adjustments).
# MAGIC
# MAGIC ### 4. Market Closures
# MAGIC
# MAGIC Even if a release happens, markets may be closed:
# MAGIC - US markets closed on US holidays
# MAGIC - Check if release_date is a trading day before analyzing market reaction
# MAGIC
# MAGIC ## 📈 Use Cases
# MAGIC
# MAGIC ### 1. Event Study Analysis
# MAGIC
# MAGIC ```python
# MAGIC # Measure SPY returns around NFP releases
# MAGIC payems = pd.read_csv('first_releases_PAYEMS.csv')
# MAGIC payems['release_datetime'] = pd.to_datetime(payems['release_datetime'])
# MAGIC payems = payems[payems['data_period'] >= '2020-01-01']  # Recent data only
# MAGIC
# MAGIC # Calculate surprise (actual vs expectations would come from another source)
# MAGIC # Then measure market reaction in 30-min, 1-hour, 1-day windows
# MAGIC ```
# MAGIC
# MAGIC ### 2. Nowcasting Model Validation
# MAGIC
# MAGIC ```python
# MAGIC # Compare your model's forecasts to first releases
# MAGIC gdp = pd.read_csv('first_releases_GDPC1.csv')
# MAGIC
# MAGIC # Your model makes predictions on the last day of the quarter
# MAGIC # Compare to the first release value
# MAGIC ```
# MAGIC
# MAGIC ### 3. Trading Strategy Backtesting
# MAGIC
# MAGIC ```python
# MAGIC # Avoid trading in the 1-hour window around major releases
# MAGIC releases = pd.concat([
# MAGIC     pd.read_csv('first_releases_PAYEMS.csv'),
# MAGIC     pd.read_csv('first_releases_CPIAUCSL.csv')
# MAGIC ])
# MAGIC
# MAGIC # Create a "no-trade" filter for your backtest
# MAGIC ```
# MAGIC
# MAGIC ## 🔍 Data Quality Notes
# MAGIC
# MAGIC ### Historical Releases (Pre-2015)
# MAGIC
# MAGIC - Records with `release_date` of 2015-01-01 are historical data
# MAGIC - These show when ALFRED first ingested the data, not the original release
# MAGIC - For event studies, **filter to `release_date >= 2015-02-01`** for accurate timing
# MAGIC
# MAGIC ### Missing Values
# MAGIC
# MAGIC - Some `value` fields show "." (missing/not available)
# MAGIC - This can happen when:
# MAGIC   - Data was revised away
# MAGIC   - Original estimate was not published
# MAGIC   - Seasonal adjustment issues
# MAGIC
# MAGIC ### Revisions
# MAGIC
# MAGIC - This dataset shows **only first releases**
# MAGIC - Subsequent revisions are not included
# MAGIC - For revision analysis, query ALFRED API directly with different realtime_start dates
# MAGIC
# MAGIC ## 🛠️ Example Databricks Code
# MAGIC
# MAGIC ### Load and Prepare Data
# MAGIC
# MAGIC ```python
# MAGIC # Read all first release files
# MAGIC from pyspark.sql import functions as F
# MAGIC from pyspark.sql.types import TimestampType
# MAGIC
# MAGIC # Read PAYEMS
# MAGIC payems = spark.read.csv(
# MAGIC     '/Volumes/pawan_quant_research_1/v4/market_data/first_releases_PAYEMS.csv',
# MAGIC     header=True,
# MAGIC     inferSchema=True
# MAGIC )
# MAGIC
# MAGIC # Add timezone-aware timestamp
# MAGIC payems = payems.withColumn(
# MAGIC     'release_timestamp_et',
# MAGIC     F.to_timestamp('release_datetime')
# MAGIC )
# MAGIC
# MAGIC # Convert to UTC for joining with market data
# MAGIC payems = payems.withColumn(
# MAGIC     'release_timestamp_utc',
# MAGIC     F.from_utc_timestamp(
# MAGIC         F.to_utc_timestamp('release_timestamp_et', 'America/New_York'),
# MAGIC         'UTC'
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC display(payems.filter(F.col('data_period') >= '2020-01-01'))
# MAGIC ```
# MAGIC
# MAGIC ### Combine Multiple Indicators
# MAGIC
# MAGIC ```python
# MAGIC # Union all indicators into single table
# MAGIC indicators = ['PAYEMS', 'UNRATE', 'CPIAUCSL', 'PCEPI', 'GDPC1']
# MAGIC
# MAGIC dfs = []
# MAGIC for indicator in indicators:
# MAGIC     df = spark.read.csv(
# MAGIC         f'/Volumes/pawan_quant_research_1/v4/market_data/first_releases_{indicator}.csv',
# MAGIC         header=True,
# MAGIC         inferSchema=True
# MAGIC     )
# MAGIC     df = df.withColumn('indicator', F.lit(indicator))
# MAGIC     dfs.append(df)
# MAGIC
# MAGIC combined = dfs[0]
# MAGIC for df in dfs[1:]:
# MAGIC     combined = combined.union(df)
# MAGIC
# MAGIC # Save as Delta table
# MAGIC combined.write.format('delta').mode('overwrite').saveAsTable('economic_releases')
# MAGIC ```
# MAGIC
# MAGIC ## 📚 Additional Resources
# MAGIC
# MAGIC - **FRED/ALFRED API**: https://fred.stlouisfed.org/docs/api/
# MAGIC - **Release Schedule Patterns**: See `release_schedule_report.json`
# MAGIC - **BLS Release Calendar**: https://www.bls.gov/schedule/
# MAGIC - **Census Bureau Calendar**: https://www.census.gov/economic-indicators/
# MAGIC
# MAGIC ## 🔄 Updates
# MAGIC
# MAGIC This dataset was generated on **February 15, 2026** with data through:
# MAGIC - **PAYEMS/UNRATE**: Through January 2026
# MAGIC - **CPIAUCSL**: Through January 2026  
# MAGIC - **PCEPI/PCE**: Through November 2025
# MAGIC - **INDPRO**: Through December 2025
# MAGIC - **GDPC1**: Through Q3 2025
# MAGIC - **UMCSENT**: Through December 2025
# MAGIC
# MAGIC To update with latest releases, re-run the ALFRED integration script.
# MAGIC
# MAGIC ## 📞 Support
# MAGIC
# MAGIC For questions about:
# MAGIC - **Data accuracy**: Cross-reference with FRED/ALFRED at https://alfred.stlouisfed.org/
# MAGIC - **Timezone conversions**: See pytz documentation
# MAGIC - **Market data integration**: Check your data provider's timezone documentation
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Generated by**: ALFRED Integration Utility  
# MAGIC **Source**: Federal Reserve Bank of St. Louis (FRED/ALFRED)  
# MAGIC **Enhancement**: Historical release time analysis (2015-2025)  
# MAGIC **License**: Public Domain (US Government data)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC we got the raw csv from finnhub, filtered medium and high impact events. used claude to create a code block to create separate csvs for each event (based on de-duplicated event names). asked claude to identify pattern and edge cases. asked it to create a script to fetch data series from ALFRED (1st release) and append the time extracted from finnhub. ( report.json has the pattern and edge cases details). Final files are individual csvs 

# COMMAND ----------

# MAGIC %md
# MAGIC /Volumes/pawan_quant_research_1/v4/market_data/first_releases_INDPRO.csv.  

# COMMAND ----------

# DBTITLE 1,Individual CSV Files - Head, Tail & Summary
from pyspark.sql import functions as F

# Define the CSV files to analyze
base_path = "/Volumes/pawan_quant_research_1/v4/market_data"
indicators = ['CPIAUCSL', 'GDPC1', 'INDPRO', 'PAYEMS', 'PCE', 'PCEPI', 'UMCSENT', 'UNRATE']

print("="*100)
print("📊 INDIVIDUAL CSV FILES - FIRST RELEASES FROM ALFRED")
print("="*100)

# Store summary data
summary_data = []

for indicator in indicators:
    file_path = f"{base_path}/first_releases_{indicator}.csv"
    
    print(f"\n\n{'='*100}")
    print(f"📄 {indicator} - {file_path}")
    print("="*100)
    
    # Read CSV with all columns as string first
    df = spark.read.csv(file_path, header=True, inferSchema=False)
    
    # Convert value column, handling "." as NULL
    df = df.withColumn('value_numeric', 
                       F.when(F.col('value') == '.', None)
                        .otherwise(F.col('value').cast('double')))
    
    # Get total count
    total_count = df.count()
    
    # Show first 10 rows
    print(f"\n🔼 FIRST 10 ROWS:")
    head_df = df.orderBy('data_period').limit(5)
    display(head_df.select('data_period', 'value', 'release_date', 'release_datetime', 'release_time'))
    
    # Show last 10 rows
    print(f"\n🔽 LAST 10 ROWS:")
    tail_df = df.orderBy(F.col('data_period').desc()).limit(5)
    display(tail_df.select('data_period', 'value', 'release_date', 'release_datetime', 'release_time'))
    
    # Collect summary statistics (excluding NULL values)
    stats = df.select(
        F.min('data_period').alias('first_period'),
        F.max('data_period').alias('last_period'),
        F.min('release_date').alias('first_release'),
        F.max('release_date').alias('last_release'),
        F.count('*').alias('total_records'),
        F.min('value_numeric').alias('min_value'),
        F.max('value_numeric').alias('max_value'),
        F.avg('value_numeric').alias('avg_value'),
        F.count(F.when(F.col('value') == '.', 1)).alias('missing_count')
    ).collect()[0]
    
    summary_data.append({
        'indicator': indicator,
        'first_period': stats['first_period'],
        'last_period': stats['last_period'],
        'first_release': stats['first_release'],
        'last_release': stats['last_release'],
        'total_records': stats['total_records'],
        'missing_count': stats['missing_count'],
        'min_value': stats['min_value'],
        'max_value': stats['max_value'],
        'avg_value': stats['avg_value']
    })
    
    print(f"\n✅ Total Records: {total_count:,} (Missing: {stats['missing_count']})")

# Create summary DataFrame
print("\n\n" + "="*100)
print("📊 SUMMARY OF ALL CSV FILES")
print("="*100)

summary_df = spark.createDataFrame(summary_data)
display(summary_df)

# Print detailed summary
print("\n📈 DETAILED SUMMARY:\n")
for row in summary_data:
    print(f"{row['indicator']}:")
    print(f"  • Data Period: {row['first_period']} to {row['last_period']}")
    print(f"  • Release Dates: {row['first_release']} to {row['last_release']}")
    print(f"  • Total Records: {row['total_records']:,} (Missing: {row['missing_count']})")
    if row['min_value'] is not None:
        print(f"  • Value Range: {row['min_value']:.2f} - {row['max_value']:.2f} (Avg: {row['avg_value']:.2f})")
    else:
        print(f"  • Value Range: All values missing")
    print()

print("="*100)
print(f"🎯 GRAND TOTAL: {sum(r['total_records'] for r in summary_data):,} records across {len(indicators)} indicators")
print("="*100)
