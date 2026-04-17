# Databricks notebook source
# MAGIC %pip install fredapi

import pandas as pd
import pytz
import time
from datetime import datetime
from fredapi import Fred
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

# --- CONFIG ---
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"
ALFRED_RETRY_ATTEMPTS = 3
ALFRED_RETRY_BASE_SECONDS = 2

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

fred = Fred(api_key=FRED_API_KEY)
alfred_release_cache = {}

# Explicitly target only these stale historical rows.
REPAIR_TARGETS = [
    {
        "indicator_id": "PAYEMS",
        "target_timestamp_utc": "2026-04-03 12:30:00",
    },
    {
        "indicator_id": "INDPRO",
        "target_timestamp_utc": "2026-03-16 13:15:00",
    },
]

def apply_precise_timing(indicator_id, release_date):
    wall_clock_time = TYPICAL_TIMES.get(indicator_id, "08:30:00")
    naive_dt = datetime.strptime(f"{release_date} {wall_clock_time}", "%Y-%m-%d %H:%M:%S")
    eastern = pytz.timezone("US/Eastern")
    localized_dt = eastern.localize(naive_dt)
    return localized_dt.astimezone(pytz.utc).replace(tzinfo=None)

def load_alfred_releases(series_id):
    if series_id in alfred_release_cache:
        return alfred_release_cache[series_id]

    try:
        vintages = fred.get_series_all_releases(series_id).copy()
        vintages["realtime_start"] = pd.to_datetime(vintages["realtime_start"]).dt.date
        vintages["value"] = pd.to_numeric(vintages["value"], errors="coerce")
        cleaned = vintages.dropna(subset=["value"])
        alfred_release_cache[series_id] = cleaned
        return cleaned
    except Exception as exc:
        print(f"ALFRED release fetch failed for {series_id}: {exc}")
        alfred_release_cache[series_id] = pd.DataFrame()
        return pd.DataFrame()

def get_alfred_first_release(series_id, release_date):
    last_error = None
    vintages = pd.DataFrame()
    for attempt in range(ALFRED_RETRY_ATTEMPTS):
        try:
            vintages = load_alfred_releases(series_id)
            if vintages.empty:
                return None

            release_dt = pd.to_datetime(release_date).date()
            match = vintages[pd.to_datetime(vintages["realtime_start"]).dt.date == release_dt]
            if match.empty:
                return None

            observation_col = "observation_date" if "observation_date" in match.columns else "date"
            if observation_col not in match.columns:
                return float(match.iloc[0]["value"])

            match = match.copy()
            match["observation_date"] = pd.to_datetime(match[observation_col], errors="coerce")
            match = match.dropna(subset=["observation_date"])
            if match.empty:
                return None

            latest_index = match["observation_date"].idxmax()
            return float(match.loc[latest_index]["value"])
        except Exception as exc:
            last_error = exc
            if attempt + 1 < ALFRED_RETRY_ATTEMPTS:
                time.sleep(ALFRED_RETRY_BASE_SECONDS * (2 ** attempt))

    print(f"ALFRED first-release fetch failed for {series_id} at {release_date}: {last_error}")
    return None


print("===== TARGETED BRONZE MACRO REPAIR START =====")

repair_rows = []
for target in REPAIR_TARGETS:
    indicator_id = target["indicator_id"]
    target_ts = datetime.strptime(target["target_timestamp_utc"], "%Y-%m-%d %H:%M:%S")
    target_ts_str = target_ts.strftime("%Y-%m-%d %H:%M:%S")

    # Display current bronze value (before correction)
    old_sql = f"""
        SELECT
            indicator_id,
            timestamp,
            actual,
            event,
            year_partition,
            ingestion_ts
        FROM {TARGET_TABLE}
        WHERE indicator_id = '{indicator_id}'
          AND timestamp = TIMESTAMP('{target_ts_str}')
    """
    old_df = spark.sql(old_sql)
    print(f"--- {indicator_id} BEFORE @ {target_ts_str} UTC ---")
    old_df.show(truncate=False)

    if old_df.count() != 1:
        print(f"Warning: expected exactly 1 row for {indicator_id} at {target_ts_str}, found {old_df.count()}")
        continue

    # Use the release date implied by the bronze bucket to recompute latest-observation-at-release value.
    release_date = (
        pd.Timestamp(target_ts, tz="UTC")
        .tz_convert("US/Eastern")
        .date()
        .strftime("%Y-%m-%d")
    )
    inferred_ts = apply_precise_timing(indicator_id, release_date)
    if inferred_ts != target_ts:
        print(
            f"Warning: inferred timestamp {inferred_ts} does not match target "
            f"{target_ts} for {indicator_id}"
        )

    corrected = get_alfred_first_release(indicator_id, release_date)
    if corrected is None:
        print(f"Could not calculate corrected ALFRED value for {indicator_id} on {release_date}")
        continue

    print(f"--- {indicator_id} CORRECTED ({release_date}) = {corrected} ---")
    repair_rows.append({
        "indicator_id": indicator_id,
        "timestamp": target_ts,
        "actual": float(corrected),
        "event": f"First Release: {indicator_id}",
    })

if not repair_rows:
    print("No repair rows resolved; exiting without changes.")
    dbutils.notebook.exit("NO_REPAIR_ROWS")

schema = StructType([
    StructField("indicator_id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("actual", DoubleType(), True),
    StructField("event", StringType(), True),
])

repair_df = (
    spark.createDataFrame(repair_rows, schema=schema)
    .withColumn("year_partition", F.year("timestamp"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

repair_df.createOrReplaceTempView("repair_macro_rows")

# Targeted merge into the exact timestamp/indicator keys.
spark.sql(f"""
    MERGE INTO {TARGET_TABLE} t
    USING repair_macro_rows s
    ON t.indicator_id = s.indicator_id
       AND t.timestamp = s.timestamp
    WHEN MATCHED THEN UPDATE SET
        t.actual = s.actual,
        t.event = s.event,
        t.year_partition = s.year_partition,
        t.ingestion_ts = s.ingestion_ts
""")

print(f"Applied targeted bronze repair to {len(repair_rows)} rows.")

print("===== TARGETED BRONZE MACRO REPAIR AFTER SNAPSHOT =====")
for row in repair_rows:
    indicator_id = row["indicator_id"]
    target_ts = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    post_sql = f"""
        SELECT
            indicator_id,
            timestamp,
            actual,
            event,
            year_partition,
            ingestion_ts
        FROM {TARGET_TABLE}
        WHERE indicator_id = '{indicator_id}'
          AND timestamp = TIMESTAMP('{target_ts}')
    """
    post_df = spark.sql(post_sql)
    print(f"--- {indicator_id} AFTER @ {target_ts} UTC ---")
    post_df.show(truncate=False)

print("===== TARGETED BRONZE MACRO REPAIR COMPLETE =====")
