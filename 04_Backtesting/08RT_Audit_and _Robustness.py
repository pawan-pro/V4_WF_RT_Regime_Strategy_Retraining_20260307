# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 1: PIT (Point-In-Time) Integrity Test

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

# 1. Pick a major known release (e.g., Non-Farm Payrolls on a specific date)
# We look for the moment NFP_Shock was non-zero
#audit_df = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
audit_df = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
    .filter("timestamp BETWEEN '2023-10-06 11:00:00' AND '2023-10-06 15:00:00'") \
    .sort("timestamp").toPandas()

print("--- PIT INTEGRITY AUDIT: NFP RELEASE (Oct 6, 2023) ---")
print("Context: NFP usually drops at 12:30 UTC.")
print(audit_df[['timestamp', 'PAYEMS_z', 'NFP_Event_Active', 'SPY_close']])

# 2. Logic Check: 
# Does the 12:00:00 row have the OLD value? 
# Does the 13:00:00 row have the NEW value? 
# If 12:00:00 already has the new value, you have look-ahead bias.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Traning set vs Test set Leakage

# COMMAND ----------

# Check if the signal file has any timestamps out of order
signal_path = "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_regime_signals-v4.csv"
signals = pd.read_csv(signal_path)
is_sorted = all(signals['timestamp'] == signals['timestamp'].sort_values())

print(f"Audit: Time-Series Chronology Maintained? {is_sorted}")

# Verify that the shift(-168) used for labeling didn't leak back into features
# We check if a feature (like Price_Mom_24H) is used to predict a target 
# that includes those same 24 hours. 
# V4 uses forward 168h, features use past 24h. No overlap = Safe.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Feature Stationarity Audit

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

df_audit = pd.read_csv(signal_path, parse_dates=['timestamp'])

print("--- STATIONARITY AUDIT: FEATURE DISTRIBUTIONS ---")
# Compare CPI Z-score in 2017 vs 2024
df_audit['year'] = df_audit['timestamp'].dt.year
plt.figure(figsize=(12, 5))
sns.kdeplot(data=df_audit[df_audit['year'] == 2017], x='CPIAUCSL_z', label='2017 Inflation Z')
sns.kdeplot(data=df_audit[df_audit['year'] == 2024], x='CPIAUCSL_z', label='2024 Inflation Z')
plt.title("Distribution Stability: 2017 vs 2024")
plt.legend()
plt.show()

# If the distributions overlap perfectly, the model is 'Cycle-Robust.'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: The "Commission Killer: Stress Test

# COMMAND ----------

# --- PRE-FLIGHT: ATTACH PRICES TO AUDIT DATAFRAME ---

# 1. Load QQQ prices from the Bronze Market table
qqq_prices_sdf = spark.table("pawan_quant_research_1.default.bronze_bt_market_1h") \
    .filter("symbol = 'QQQ'") \
    .select(F.col("timestamp"), F.col("close").alias("QQQ_price"))

# 2. Convert to Pandas for the audit loop
qqq_pdf = qqq_prices_sdf.toPandas()
qqq_pdf['timestamp'] = pd.to_datetime(qqq_pdf['timestamp']).dt.tz_localize(None)
qqq_pdf = qqq_pdf.set_index('timestamp')

# 3. Merge into your existing df_audit
# Ensure df_audit index is also datetime
if 'timestamp' in df_audit.columns:
    df_audit = df_audit.set_index('timestamp')
df_audit.index = pd.to_datetime(df_audit.index).tz_localize(None)

# Join the prices
df_audit = df_audit.join(qqq_pdf, how='left')

# 4. Forward fill prices to handle any gaps (weekends/holidays)
df_audit['QQQ_price'] = df_audit['QQQ_price'].ffill()
df_audit['SPY_close'] = df_audit['SPY_close'].ffill()

print(f"✅ Prices attached. Columns available: {df_audit.columns.tolist()}")

# COMMAND ----------

# --- STEP 4: COMMISSION KILLER STRESS TEST ---
print("🚀 Running High-Cost Stress Test (15bps per switch)...")

portfolio_stress = 5 * 10**6
active_asset = "START"
holdings_stress = [portfolio_stress]
cost_rate = 0.0015  # 15 basis points (3x original)

# We use range starting from 1 to have access to i and i-1
for i in range(1, len(df_audit)):
    regime = df_audit['predicted_regime'].iloc[i-1]
    
    if regime == 0: target_asset = "QQQ"
    elif regime == 1: target_asset = "SPY"
    else: target_asset = "CASH"
    
    # Calculate returns
    if target_asset == "QQQ":
        hourly_ret = (df_audit['QQQ_price'].iloc[i] / df_audit['QQQ_price'].iloc[i-1]) - 1
    elif target_asset == "SPY":
        hourly_ret = (df_audit['SPY_close'].iloc[i] / df_audit['SPY_close'].iloc[i-1]) - 1
    else:
        hourly_ret = 0.0
        
    # Apply cost rate ONLY on switches
    current_friction = cost_rate if target_asset != active_asset else 0.0
    
    portfolio_stress *= (1 + hourly_ret - current_friction)
    holdings_stress.append(portfolio_stress)
    active_asset = target_asset

# Results calculation
total_ret = (holdings_stress[-1] / holdings_stress[0]) - 1
# Calculate CAGR (Total period is approx 9.1 years)
stressed_cagr = (1 + total_ret)**(1/9.1) - 1

print(f"--- STRESS TEST RESULTS ---")
print(f"Stressed CAGR (15bps cost): {stressed_cagr:.2%}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM pawan_quant_research_1.default.bronze_bt_macro_events 
# MAGIC WHERE indicator_id = 'PAYEMS' 
# MAGIC AND date(timestamp) = '2023-10-06'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM read_files(
# MAGIC   '/Volumes/pawan_quant_research_1/default/market_data/raw_finnhub_haystack.csv',
# MAGIC   format => 'csv',
# MAGIC   header => true
# MAGIC )
# MAGIC WHERE date(time) = '2023-10-06'
# MAGIC AND event LIKE '%Payroll%'

# COMMAND ----------

# --- EXTENDED AUDIT: Rates vs Price Synchronization ---
# We pick a day with a clear market move (e.g., a post-CPI market open)
# We expect the 'Price_Mom_24H' and 'T10Y2Y' (Yield Curve) to update in the SAME bar.

check_df = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
    .filter("timestamp BETWEEN '2024-03-12 12:00:00' AND '2024-03-12 18:00:00'") \
    .select("timestamp", "SPY_close", "Price_Mom_24H", "vix3m_z", "VIX_Inversion_Z") \
    .sort("timestamp").toPandas()

print("--- CLOCK SYNCHRONIZATION AUDIT ---")
print(check_df)

# LOGIC CHECK: 
# 1. Does the VIX and Price momentum both show activity starting at 13:00/14:00 UTC?
# 2. If VIX is 'stale' (all zeros or unchanging) while Price is moving, your join is broken.

# COMMAND ----------

# --- VIX3M LEAKAGE DIAGNOSTIC ---
# Compare VIXCLS (Hourly) with VIX3M (Daily)
diag_df = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
    .filter("timestamp BETWEEN '2024-03-11 20:00:00' AND '2024-03-12 10:00:00'") \
    .select("timestamp", "VIXCLS_z", "vix3m_z") \
    .sort("timestamp").toPandas()

print(diag_df)

# AUDIT LOGIC:
# If vix3m_z changes at 00:00:00 UTC to a value that reflects the NEXT day's 
# market volatility, you have a join bias. 
# It should ideally only change after the US market close (~21:00 UTC).

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Macro Release "Hour Zero" Test

# COMMAND ----------

# --- MACRO RELEASE HOUR-ZERO AUDIT ---
# Picking a known CPI date: 2024-02-13 (Released at 13:30 UTC due to Winter DST)
macro_diag = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
    .filter("timestamp BETWEEN '2024-02-13 11:00:00' AND '2024-02-13 16:00:00'") \
    .select("timestamp", "CPIAUCSL_z", "CPI_Event_Active", "SPY_close") \
    .sort("timestamp").toPandas()

print("--- MACRO RELEASE HOUR-ZERO AUDIT ---")
print(macro_diag)

# AUDIT LOGIC:
# 1. CPIAUCSL_z MUST be identical for the 11:00, 12:00, and 13:00 rows.
# 2. CPI_Event_Active MUST be 0 until the 14:00 or 15:00 row.
# 3. If the Z-score changes in the 13:00 row, you are peeking 30 mins into the future.

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Interest Rate "Daylight Lag" Test (FirstRate)

# COMMAND ----------

# --- INTEREST RATE SYNC AUDIT ---
# Check the transition at Midnight UTC to see if Yield Curve Z-scores "predict" the open
rate_diag = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
    .filter("timestamp BETWEEN '2024-03-11 20:00:00' AND '2024-03-12 10:00:00'") \
    .select("timestamp", "INDPRO_z", "GDPC1_z") \
    .sort("timestamp").toPandas()

print("--- DAILY MACRO LEVEL AUDIT (INDPRO/GDP) ---")
print(rate_diag)

# AUDIT LOGIC:
# Indicators like INDPRO or GDPC1 only update once a month/quarter.
# They should be PERFECTLY flat across the midnight transition (Row 3 to 4).
# If you see a jump at 00:00:00, you have the same join-bias we found in VIX3M.

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Momentum "Self-Reference" Test

# COMMAND ----------

# --- MOMENTUM PEAKING AUDIT ---
mom_diag = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
    .filter("timestamp BETWEEN '2024-03-12 13:00:00' AND '2024-03-12 15:00:00'") \
    .select("timestamp", "SPY_close", "Price_Mom_24H") \
    .sort("timestamp").toPandas()

# Calculate the percent change of the current price
spy_pct = (mom_diag['SPY_close'].iloc[1] / mom_diag['SPY_close'].iloc[0]) - 1
mom_diff = mom_diag['Price_Mom_24H'].iloc[1] - mom_diag['Price_Mom_24H'].iloc[0]

print("--- MOMENTUM PEAKING AUDIT ---")
print(mom_diag)
print(f"Price Change in bar: {spy_pct:.4%}")
print(f"Momentum Change in bar: {mom_diff:.4%}")

# AUDIT LOGIC:
# If the Price Change and Momentum Change are perfectly correlated in the SAME bar,
# it means the momentum feature is peeking at the current bar's outcome.
# Ideally, the momentum jump should happen 1 hour AFTER the price jump.

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Weekend "Frozen Clock" Test

# COMMAND ----------

# --- WEEKEND FROZEN CLOCK AUDIT ---
weekend_diag = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h") \
    .filter("timestamp BETWEEN '2024-03-09 00:00:00' AND '2024-03-10 23:00:00'") \
    .select("timestamp", "SPY_close", "VIXCLS_z", "UNRATE_z") \
    .sort("timestamp").toPandas()

changes = weekend_diag.nunique()
print("--- WEEKEND UNIQUE VALUES (Should all be 1 except timestamp) ---")
print(changes)

# AUDIT LOGIC:
# Except for the 'timestamp' column, nunique() should be 1 for all columns.
# If SPY_close or UNRATE_z changes on a Sunday morning, you have a 
# data-filling leakage from the following Monday.

# COMMAND ----------

# MAGIC %md
# MAGIC Institutional Reconciliation Audit

# COMMAND ----------

# DBTITLE 1,Cell 23
# MAGIC %pip install finnhub-python
# MAGIC %pip install fredapi
# MAGIC %pip install pytz
# MAGIC %pip install alpaca-py

# COMMAND ----------

# DBTITLE 1,Cell 24
import pandas as pd
import numpy as np
import random
import finnhub
import zipfile
import pytz
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fredapi import Fred

# Official Alpaca SDK imports (matching your environment)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- 1. CONFIGURATION & CREDENTIALS ---
ALPACA_KEY = "PKPSF55ZI6V53R25GIBFTUGB4B"
ALPACA_SECRET = "5MnZMUeKU8EFaXFV6dHGUM1uS6kG8AgPXqCnaSjHVE2U"
FINNHUB_KEY = "d5s3pchr01qoo9r2f3h0d5s3pchr01qoo9r2f3hg"
FRED_KEY = "f52bdf7341f23d0458820ecd806178db"

# Paths to the raw vendor files from your notebooks
FIRST_RATE_ZIP = "/Volumes/pawan_quant_research/default/firstratedata/futures_data/inception_till_10FEB26/futures_full_30min_contin_adj_absolute_11assu1.zip"
PORTARA_ZIP = "/Volumes/pawan_quant_research/default/portaradata/vix_data/Avik VX.zip"

# --- 2. RANDOM TIME SELECTION (2-Month Audit Period) ---
start_year = random.randint(2018, 2024)
start_month = random.randint(1, 10)
audit_start = datetime(start_year, start_month, 1, tzinfo=ZoneInfo("UTC"))
audit_end = audit_start + timedelta(days=60)
lookback_start = audit_start - timedelta(days=365)

print(f"🔍 AUDIT PERIOD: {audit_start.date()} to {audit_end.date()}")
print(f"Lookback initiated from: {lookback_start.date()} to ensure Z-score accuracy...")

# --- 3. RAW DATA FETCHING FUNCTIONS ---

def fetch_raw_alpaca(tickers, start, end):
    """Fetches ETF prices using the new alpaca-py SDK"""
    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end
    )
    bars = client.get_stock_bars(request_params).df
    return bars.reset_index().pivot(index='timestamp', columns='symbol', values='close')

def fetch_raw_macro_triangulation(start, end):
    """Triangulates Finnhub Release Times with ALFRED First-Release values"""
    fh = finnhub.Client(api_key=FINNHUB_KEY)
    fred = Fred(api_key=FRED_KEY)
    
    events = fh.calendar_economic(_from=start.strftime('%Y-%m-%d'), to=end.strftime('%Y-%m-%d'))
    df_events = pd.DataFrame(events['economicCalendar'])
    df_events = df_events[(df_events['country'] == 'US') & (df_events['impact'] == 'high')]
    
    series_map = {
        'Non-Farm Payrolls': 'PAYEMS', 
        'CPI MM, SA': 'CPIAUCSL',
        'Unemployment Rate': 'UNRATE',
        'Real GDP': 'GDPC1'
    }
    
    results = []
    for _, row in df_events.iterrows():
        series_id = series_map.get(row['event'])
        if series_id:
            vintages = fred.get_series_all_releases(series_id)
            release_date = pd.to_datetime(row['time']).strftime('%Y-%m-%d')
            match = vintages[vintages['realtime_start'] == release_date]
            
            if not match.empty:
                results.append({
                    'timestamp': pd.to_datetime(row['time']).replace(tzinfo=ZoneInfo("UTC")),
                    'indicator': f"{series_id}_z",
                    'actual': match.iloc[0]['value']
                })
    return pd.DataFrame(results)

def extract_from_zip_robust(zip_path, search_pattern, symbol_name):
    """Robustly finds and extracts data files from nested ZIP structures (Notebook 04 logic)"""
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find all files matching the pattern (e.g., 'VX.txt' or 'ZN_full')
        namelist = z.namelist()
        matches = [n for n in namelist if search_pattern in n and not n.endswith('/')]
        
        if not matches:
            raise FileNotFoundError(f"Could not find file matching '{search_pattern}' in {zip_path}")
        
        # If multiple matches, pick the one with the largest file size (standard Portara/FirstRate logic)
        target_file = max(matches, key=lambda x: z.getinfo(x).file_size)
        print(f"   Found data file: {target_file}")

        with z.open(target_file) as f:
            df = pd.read_csv(f, header=None, names=['ts_raw', 'open', 'high', 'low', 'close', 'vol'])
            # Timezone standardization (Chicago/Eastern -> UTC)
            df['timestamp'] = pd.to_datetime(df['ts_raw']).dt.tz_localize('America/Chicago', ambiguous='infer').dt.tz_convert('UTC')
            df.set_index('timestamp', inplace=True)
            # Fix: Use '1h' instead of '1H' to avoid FutureWarning
            return df['close'].resample('1h').last().rename(symbol_name)

# --- 4. COMPILING THE MASTER AUDIT GRID ---

print("Fetching ETF Data...")
master_grid = fetch_raw_alpaca(['SPY', 'QQQ', 'HYG'], lookback_start, audit_end)

print("Extracting Rates & VIX from ZIP files...")
# Robust search patterns based on your notebook filenames
zn_rates = extract_from_zip_robust(FIRST_RATE_ZIP, "ZN_full", "ZN")
vix_futures = extract_from_zip_robust(PORTARA_ZIP, "VX.txt", "VIXCLS")

# Join Vendor Files
master_grid = master_grid.join([zn_rates, vix_futures], how='left')

print("Fetching Macro Triangulation data...")
macro_data = fetch_raw_macro_triangulation(audit_start, audit_end)
if not macro_data.empty:
    macro_pivot = macro_data.pivot(index='timestamp', columns='indicator', values='actual')
    master_grid = master_grid.join(macro_pivot, how='left')

# Forward Fill to propagate levels (Notebook 05 logic)
master_grid = master_grid.ffill()

# --- 5. FEATURE ENGINEERING (REPLICATING SILVER LAYER) ---

# Bias-Free Momentum (Notebook 05)
master_grid['Price_Mom_24H'] = (master_grid['SPY'].shift(1) / master_grid['SPY'].shift(25)) - 1

# Rolling Z-Scores (1-year window)
z_cols = ['SPY', 'QQQ', 'ZN', 'VIXCLS', 'PAYEMS_z', 'CPIAUCSL_z']
for col in z_cols:
    if col in master_grid.columns:
        window = 8760 
        role_mean = master_grid[col].rolling(window=window, min_periods=100).mean()
        role_std = master_grid[col].rolling(window=window, min_periods=100).std()
        master_grid[f'{col}_z'] = (master_grid[col] - role_mean) / role_std

# --- 6. EXPORTING AUDIT REPORT ---

# Slice back to the specific 2-month audit window
final_audit_report = master_grid.loc[audit_start:audit_end].copy()

# Clean up index for Excel/CSV compatibility
final_audit_report.index = final_audit_report.index.tz_localize(None)
filename = f"V4_Audit_Report_{start_year}_M{start_month}.csv"
final_audit_report.to_csv(filename)

print(f"\n✅ Audit CSV Successfully Generated: {filename}")
print(f"Records captured: {len(final_audit_report)}")
print("\n--- DATA CROSS-CHECK SAMPLE (Last 10 rows) ---")
print(final_audit_report[['SPY', 'SPY_z', 'ZN_z', 'Price_Mom_24H']].tail(10))

# COMMAND ----------

import os
print(os.listdir())

# COMMAND ----------

import pandas as pd
import zipfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- CONFIGURATION (Sync with your paths) ---
FIRST_RATE_ZIP = "/Volumes/pawan_quant_research/default/firstratedata/futures_data/inception_till_10FEB26/futures_full_30min_contin_adj_absolute_11assu1.zip"
PORTARA_ZIP = "/Volumes/pawan_quant_research/default/portaradata/vix_data/Avik VX.zip"

# Selecting the time period from your screenshot for the manual check
audit_start = "2020-02-03 09:00:00"
audit_end   = "2020-02-04 15:00:00"

def get_raw_vendor_series(zip_path, pattern, col_name, is_portara=False):
    """Extracts raw lines from ZIP without resampling to see true source data"""
    with zipfile.ZipFile(zip_path, 'r') as z:
        matches = [n for n in z.namelist() if pattern in n and not n.endswith('/')]
        target = max(matches, key=lambda x: z.getinfo(x).file_size)
        
        with z.open(target) as f:
            # Load raw data
            df_raw = pd.read_csv(f, header=None, names=['ts_raw', 'open', 'high', 'low', 'close', 'vol'])
            
            # Convert timestamp exactly how the pipeline does
            # FirstRate is Chicago Time, Portara is usually Central/Eastern depending on file
            tz = 'America/Chicago' if not is_portara else 'US/Eastern'
            df_raw['timestamp_utc'] = pd.to_datetime(df_raw['ts_raw']).dt.tz_localize(tz, ambiguous='infer').dt.tz_convert('UTC')
            
            # Keep index as UTC but don't resample yet
            df_raw = df_raw[['timestamp_utc', 'close']].rename(columns={'close': f'RAW_{col_name}'})
            return df_raw

# 1. Fetch RAW values (The "Untouchables")
print("Reading Raw Vendor Files...")
raw_zn = get_raw_vendor_series(FIRST_RATE_ZIP, "ZN_full", "ZN")
raw_vix = get_raw_vendor_series(PORTARA_ZIP, "VX.txt", "VIX", is_portara=True)

# 2. Re-calculate the Silver Grid (The "Processed" values)
# (Assuming 'final_audit_report' exists from the previous code block run)
# If not, use the values from your 'Master Data' table in SQL
silver_data = final_audit_report.loc[audit_start:audit_end].copy()

# 3. Align Raw vs Silver for the Audit
# We join the RAW data onto the SILVER hourly grid to see which raw tick was used
audit_comparison = silver_data[['SPY', 'ZN', 'VIXCLS', 'SPY_z', 'ZN_z']].copy()
audit_comparison.index = audit_comparison.index.tz_localize(None)

# Formatting Raw Data for Join (taking the last available raw tick per hour)
raw_zn_hourly = raw_zn.set_index('timestamp_utc').resample('1h').last().tz_localize(None)
raw_vix_hourly = raw_vix.set_index('timestamp_utc').resample('1h').last().tz_localize(None)

# 4. Join for side-by-side view
audit_table = audit_comparison.join(raw_zn_hourly, how='left')
audit_table = audit_table.join(raw_vix_hourly, how='left')

# Calculate the Delta (difference) to ensure it is 0.0 or negligible (float diffs)
audit_table['ZN_Diff'] = audit_table['ZN'] - audit_table['RAW_ZN']
audit_table['VIX_Diff'] = audit_table['VIXCLS'] - audit_table['RAW_VIX']

# 5. Add Manual Z-Score Check
# Let's show the user the Mean and Std used for that specific SPY_z
window = 8760
spy_series = master_grid['SPY'] # Needs the full history from the previous block
current_mean = spy_series.rolling(window).mean().loc[audit_start:audit_end]
current_std = spy_series.rolling(window).std().loc[audit_start:audit_end]

audit_table['SPY_Mean_Used'] = current_mean.values
audit_table['SPY_Std_Used'] = current_std.values

# Final Output
print("\n" + "="*50)
print("             V4 DATA INTEGRITY AUDIT")
print("="*50)
cols_to_show = [
    'RAW_ZN', 'ZN', 'ZN_Diff',           # Check Rates Ingestion
    'RAW_VIX', 'VIXCLS', 'VIX_Diff',     # Check VIX Ingestion
    'SPY', 'SPY_Mean_Used', 'SPY_z'      # Check Feature Engineering (Z-Score)
]
print(audit_table[cols_to_show].head(10))

# Export to CSV for your manual deep dive
audit_table.to_csv("MANUAL_DATA_COMPARE_REPORT.csv")