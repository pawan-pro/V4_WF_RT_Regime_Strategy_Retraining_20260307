# Databricks notebook source
# BT_02: Incremental macro update for production use
# ALFRED is used for first-release vintages because FRED does not expose them.

# MAGIC %pip install finnhub-python fredapi

import finnhub
import pandas as pd
import pytz
from datetime import datetime, timedelta
from fredapi import Fred
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
LOOKBACK_DAYS = 90

FINNHUB_API_KEY = "d5s3pchr01qoo9r2f3h0d5s3pchr01qoo9r2f3hg"
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

INDICATOR_MAP = {
    "PAYEMS": ["Non-Farm Payrolls", "United States-Employment-Non-Farm Payrolls", "Nonfarm Payrolls"],
    "UNRATE": ["Unemployment Rate", "United States-Employment-Unemployment Rate"],
    "CPIAUCSL": ["CPI MM, SA", "United States-CPI-CPI MM, SA", "Consumer Price Index (CPI)"],
    "PCEPI": ["Core PCE Price Index", "United States-Consumption-Core PCE Price Index MM", "PCE Price Index"],
    "PCE": ["Personal Consumption Expenditures", "United States-Consumption-Personal Consumption Expenditures"],
    "INDPRO": ["Industrial Production"],
    "GDPC1": ["Real GDP", "GDP Growth Rate"],
    "UMCSENT": ["U Mich Sentiment", "Michigan Consumer Sentiment", "U Mich Sentiment Final"],
}

TYPICAL_TIMES = {
    "PAYEMS": "08:30:00",
    "UNRATE": "08:30:00",
    "CPIAUCSL": "08:30:00",
    "PCEPI": "08:30:00",
    "PCE": "08:30:00",
    "INDPRO": "09:15:00",
    "GDPC1": "08:30:00",
    "UMCSENT": "10:00:00",
}

client = finnhub.Client(api_key=FINNHUB_API_KEY)
fred = Fred(api_key=FRED_API_KEY)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        indicator_id    STRING,
        timestamp       TIMESTAMP,
        actual          DOUBLE,
        event           STRING,
        year_partition  INT,
        ingestion_ts    TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (indicator_id)
""")

latest_known = {
    row["indicator_id"]: row["max_ts"]
    for row in spark.sql(f"""
        SELECT indicator_id, MAX(timestamp) AS max_ts
        FROM {TARGET_TABLE}
        GROUP BY indicator_id
    """).collect()
}

today = datetime.utcnow().date()
window_start = today - timedelta(days=LOOKBACK_DAYS)
window_end = today + timedelta(days=7)
print(f"Fetching macro releases from {window_start} to {window_end}")

res = client.calendar_economic(_from=str(window_start), to=str(window_end))
calendar_df = pd.DataFrame(res.get("economicCalendar", []))
if calendar_df.empty:
    print("No macro events returned by Finnhub")
    dbutils.notebook.exit("NO_NEW_MACRO_EVENTS")

calendar_df = calendar_df[
    (calendar_df["country"] == "US") &
    (calendar_df["impact"].isin(["high", "medium"]))
].copy()
calendar_df["release_date"] = pd.to_datetime(calendar_df["time"]).dt.date

def get_alfred_first_release(series_id, release_date):
    try:
        vintages = fred.get_series_all_releases(series_id)
        match = vintages[vintages["realtime_start"] == release_date]
        if not match.empty:
            return float(match.iloc[0]["value"])
    except Exception:
        return None
    return None

def apply_precise_timing(indicator_id, release_date):
    wall_clock_time = TYPICAL_TIMES.get(indicator_id, "08:30:00")
    naive_dt = datetime.strptime(f"{release_date} {wall_clock_time}", "%Y-%m-%d %H:%M:%S")
    eastern = pytz.timezone("US/Eastern")
    localized_dt = eastern.localize(naive_dt, ambiguous="infer")
    return localized_dt.astimezone(pytz.utc).replace(tzinfo=None)

rows = []
for indicator_id, vendor_names in INDICATOR_MAP.items():
    indicator_events = calendar_df[calendar_df["event"].isin(vendor_names)].copy()
    max_ts = latest_known.get(indicator_id)
    seen_dates = set()
    if max_ts is not None:
        existing = spark.sql(f"""
            SELECT CAST(timestamp AS DATE) AS release_date
            FROM {TARGET_TABLE}
            WHERE indicator_id = '{indicator_id}'
              AND timestamp >= TIMESTAMP('{window_start}')
        """).collect()
        seen_dates = {row["release_date"] for row in existing}

    for _, event_row in indicator_events.iterrows():
        release_date = event_row["release_date"]
        if release_date in seen_dates:
            continue
        actual = get_alfred_first_release(indicator_id, str(release_date))
        if actual is None:
            continue
        ts = apply_precise_timing(indicator_id, str(release_date))
        if max_ts is not None and ts <= max_ts:
            continue
        rows.append({
            "indicator_id": indicator_id,
            "timestamp": ts,
            "actual": float(actual),
            "event": f"First Release: {indicator_id}",
        })

if not rows:
    print("No new macro releases to merge")
    dbutils.notebook.exit("NO_NEW_MACRO_EVENTS")

schema = StructType([
    StructField("indicator_id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("actual", DoubleType(), True),
    StructField("event", StringType(), True),
])

incoming = (
    spark.createDataFrame(rows, schema=schema)
    .withColumn("year_partition", F.year("timestamp"))
    .withColumn("ingestion_ts", F.current_timestamp())
)
incoming.createOrReplaceTempView("incoming_macro_updates")

spark.sql(f"""
    MERGE INTO {TARGET_TABLE} t
    USING incoming_macro_updates s
    ON t.indicator_id = s.indicator_id
       AND t.timestamp = s.timestamp
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Merged {len(rows)} new macro releases into {TARGET_TABLE}")
