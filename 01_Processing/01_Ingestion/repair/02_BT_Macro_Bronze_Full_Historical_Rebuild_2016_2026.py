# Databricks notebook source
# Full historical macro bronze rebuild using corrected ALFRED release selection.

# MAGIC %pip install finnhub-python fredapi

import json
import os
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd
import pytz
import requests
from fredapi import Fred
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
PRODUCTION_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
DEFAULT_TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_rebuild"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
DEFAULT_HAYSTACK_FILE = f"{VOLUME_PATH}/raw_finnhub_haystack.csv"
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"

START_YEAR_DEFAULT = 2016
END_YEAR_DEFAULT = datetime.utcnow().year
ALFRED_RETRY_ATTEMPTS = 3
ALFRED_RETRY_BASE_SECONDS = 2

FINNHUB_API_KEY = "d5s3pchr01qoo9r2f3h0d5s3pchr01qoo9r2f3hg"
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

INDICATOR_MAP = {
    "PAYEMS": [
        "Non-Farm Payrolls",
        "United States-Employment-Non-Farm Payrolls",
        "Nonfarm Payrolls",
    ],
    "UNRATE": [
        "Unemployment Rate",
        "United States-Employment-Unemployment Rate",
    ],
    "CPIAUCSL": [
        "CPI MM, SA",
        "United States-CPI-CPI MM, SA",
        "Consumer Price Index (CPI)",
    ],
    "PCEPI": [
        "Core PCE Price Index",
        "United States-Consumption-Core PCE Price Index MM",
        "PCE Price Index",
    ],
    "PCE": [
        "Personal Consumption Expenditures",
        "United States-Consumption-Personal Consumption Expenditures",
    ],
    "INDPRO": ["Industrial Production"],
    "GDPC1": ["Real GDP", "GDP Growth Rate"],
    "UMCSENT": [
        "U Mich Sentiment",
        "Michigan Consumer Sentiment",
        "U Mich Sentiment Final",
    ],
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


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


target_table = DEFAULT_TARGET_TABLE
haystack_file = DEFAULT_HAYSTACK_FILE
start_year = START_YEAR_DEFAULT
end_year = END_YEAR_DEFAULT
allow_production_overwrite = False

if "dbutils" in globals():
    dbutils.widgets.text("target_table", DEFAULT_TARGET_TABLE)
    dbutils.widgets.text("haystack_file", DEFAULT_HAYSTACK_FILE)
    dbutils.widgets.text("start_year", str(START_YEAR_DEFAULT))
    dbutils.widgets.text("end_year", str(END_YEAR_DEFAULT))
    dbutils.widgets.dropdown(
        "allow_production_overwrite",
        "false",
        ["false", "true"],
    )
    target_table = dbutils.widgets.get("target_table").strip() or DEFAULT_TARGET_TABLE
    haystack_file = dbutils.widgets.get("haystack_file").strip() or DEFAULT_HAYSTACK_FILE
    start_year = int(dbutils.widgets.get("start_year").strip() or START_YEAR_DEFAULT)
    end_year = int(dbutils.widgets.get("end_year").strip() or END_YEAR_DEFAULT)
    allow_production_overwrite = parse_bool(
        dbutils.widgets.get("allow_production_overwrite"),
        default=False,
    )

if target_table == PRODUCTION_TABLE and not allow_production_overwrite:
    raise ValueError(
        "Refusing to overwrite production bronze table without "
        "allow_production_overwrite=true"
    )

if start_year > end_year:
    raise ValueError(f"Invalid year range: {start_year} > {end_year}")

rebuild_start_date = datetime(start_year, 1, 1).date()
today_utc = datetime.utcnow().date()
rebuild_end_date = (
    today_utc if end_year >= today_utc.year else datetime(end_year, 12, 31).date()
)

client = finnhub.Client(api_key=FINNHUB_API_KEY)
fred = Fred(api_key=FRED_API_KEY)
alfred_release_cache = {}
vintage_date_cache = {}


def fetch_monthly_haystack(window_start, window_end):
    all_raw_events = []
    cursor = datetime(window_start.year, window_start.month, 1).date()
    last_month = datetime(window_end.year, window_end.month, 1).date()

    while cursor <= last_month:
        next_month = (
            datetime(cursor.year + (1 if cursor.month == 12 else 0), 1 if cursor.month == 12 else cursor.month + 1, 1)
            .date()
        )
        fetch_end = min(window_end + timedelta(days=1), next_month)
        try:
            response = client.calendar_economic(
                _from=cursor.isoformat(),
                to=fetch_end.isoformat(),
            )
            month_df = pd.DataFrame(response.get("economicCalendar", []))
            if not month_df.empty:
                us_relevant = month_df[
                    (month_df["country"] == "US")
                    & (month_df["impact"].isin(["high", "medium"]))
                ].copy()
                if not us_relevant.empty:
                    all_raw_events.append(us_relevant)
        except Exception as exc:
            print(f"Finnhub fetch failed for {cursor:%Y-%m}: {exc}")
        time.sleep(1.2)
        cursor = next_month

    if not all_raw_events:
        return pd.DataFrame()

    haystack_df = pd.concat(all_raw_events, ignore_index=True)
    haystack_df = haystack_df.drop_duplicates(subset=["time", "event"])
    return haystack_df


def load_or_refresh_haystack(path_str, window_start, window_end):
    refresh_required = True
    if os.path.exists(path_str):
        cached = pd.read_csv(path_str)
        if not cached.empty and "time" in cached.columns:
            cached["time"] = pd.to_datetime(cached["time"], errors="coerce")
            cached = cached.dropna(subset=["time"])
            if not cached.empty:
                cached_min = cached["time"].min().date()
                cached_max = cached["time"].max().date()
                refresh_required = not (
                    cached_min <= window_start and cached_max >= window_end
                )
                if not refresh_required:
                    print(
                        f"Using cached Finnhub haystack from {path_str} "
                        f"({cached_min} -> {cached_max})"
                    )
                    return cached

    print(
        f"Refreshing Finnhub haystack for {window_start} -> {window_end} into {path_str}"
    )
    refreshed = fetch_monthly_haystack(window_start, window_end)
    if refreshed.empty:
        raise ValueError("Finnhub haystack refresh returned no rows")
    parent_dir = os.path.dirname(path_str)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    refreshed.to_csv(path_str, index=False)
    return refreshed


def load_alfred_releases(series_id):
    if series_id in alfred_release_cache:
        return alfred_release_cache[series_id]

    last_error = None
    for attempt in range(ALFRED_RETRY_ATTEMPTS):
        try:
            vintages = fred.get_series_all_releases(series_id).copy()
            if "realtime_start" in vintages.columns:
                vintages["realtime_start"] = pd.to_datetime(
                    vintages["realtime_start"], errors="coerce"
                ).dt.date
            if "realtime_end" in vintages.columns:
                vintages["realtime_end"] = pd.to_datetime(
                    vintages["realtime_end"], errors="coerce"
                ).dt.date
            else:
                vintages["realtime_end"] = pd.NaT
            if "date" in vintages.columns and "observation_date" not in vintages.columns:
                vintages = vintages.rename(columns={"date": "observation_date"})
            vintages["observation_date"] = pd.to_datetime(
                vintages["observation_date"], errors="coerce"
            )
            vintages["value"] = pd.to_numeric(vintages["value"], errors="coerce")
            vintages = vintages.dropna(subset=["realtime_start", "observation_date", "value"])
            alfred_release_cache[series_id] = vintages
            return vintages
        except Exception as exc:
            last_error = exc
            if attempt + 1 < ALFRED_RETRY_ATTEMPTS:
                time.sleep(ALFRED_RETRY_BASE_SECONDS * (2 ** attempt))

    print(f"ALFRED release fetch failed for {series_id}: {last_error}")
    alfred_release_cache[series_id] = pd.DataFrame()
    return alfred_release_cache[series_id]


def load_vintage_dates(series_id, window_start, window_end):
    if series_id in vintage_date_cache:
        return vintage_date_cache[series_id]

    last_error = None
    vintage_dates = []
    for attempt in range(ALFRED_RETRY_ATTEMPTS):
        try:
            response = requests.get(
                f"{FRED_API_BASE_URL}/series/vintagedates",
                params={
                    "series_id": series_id,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                },
                timeout=30,
            )
            response.raise_for_status()
            candidate_dates = pd.to_datetime(
                response.json().get("vintage_dates", []),
                errors="coerce",
            )
            vintage_dates = [
                release_date.date()
                for release_date in candidate_dates
                if pd.notna(release_date)
                and window_start <= release_date.date() <= window_end
            ]
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < ALFRED_RETRY_ATTEMPTS:
                time.sleep(ALFRED_RETRY_BASE_SECONDS * (2 ** attempt))

    if last_error and not vintage_dates:
        print(f"ALFRED vintage-date fetch failed for {series_id}: {last_error}")

    vintage_date_cache[series_id] = sorted(set(vintage_dates))
    return vintage_date_cache[series_id]


def corrected_release_value(series_id, release_date):
    vintages = load_alfred_releases(series_id)
    if vintages.empty:
        return None

    release_dt = pd.to_datetime(release_date).date()
    release_rows = vintages[vintages["realtime_start"] == release_dt].copy()
    if release_rows.empty:
        return None

    max_observation_date = release_rows["observation_date"].max()
    if pd.isna(max_observation_date):
        return None

    latest_rows = release_rows[
        release_rows["observation_date"] == max_observation_date
    ].copy()
    sort_columns = [
        column
        for column in ["observation_date", "realtime_start", "realtime_end"]
        if column in latest_rows.columns
    ]
    latest_rows = latest_rows.sort_values(
        sort_columns,
        ascending=[True] * len(sort_columns),
        na_position="last",
    )
    selected = latest_rows.iloc[-1]
    return {
        "actual": float(selected["value"]),
        "selected_observation_date": max_observation_date.date().isoformat(),
    }


def apply_precise_timing(indicator_id, release_date):
    wall_clock_time = TYPICAL_TIMES.get(indicator_id, "08:30:00")
    naive_dt = datetime.strptime(
        f"{release_date} {wall_clock_time}",
        "%Y-%m-%d %H:%M:%S",
    )
    eastern = pytz.timezone("US/Eastern")
    localized_dt = eastern.localize(naive_dt)
    return localized_dt.astimezone(pytz.utc).replace(tzinfo=None)


haystack_df = load_or_refresh_haystack(
    haystack_file,
    rebuild_start_date,
    rebuild_end_date,
)
haystack_df["time"] = pd.to_datetime(haystack_df["time"], errors="coerce")
haystack_df = haystack_df.dropna(subset=["time"])
haystack_df["release_date"] = haystack_df["time"].dt.date
haystack_df = haystack_df[
    haystack_df["release_date"].between(rebuild_start_date, rebuild_end_date)
].copy()

rows = []
audit_rows = []

for indicator_id, vendor_names in INDICATOR_MAP.items():
    indicator_events = haystack_df[haystack_df["event"].isin(vendor_names)].copy()
    release_dates = set(indicator_events["release_date"].dropna().tolist())
    release_dates.update(
        load_vintage_dates(indicator_id, rebuild_start_date, rebuild_end_date)
    )
    release_dates = sorted(release_dates)
    print(f"{indicator_id}: evaluating {len(release_dates)} release dates")

    for release_date in release_dates:
        selection = corrected_release_value(indicator_id, release_date)
        if selection is None:
            continue

        timestamp_utc = apply_precise_timing(indicator_id, str(release_date))
        rows.append(
            {
                "indicator_id": indicator_id,
                "timestamp": timestamp_utc,
                "actual": float(selection["actual"]),
                "event": f"First Release: {indicator_id}",
            }
        )
        audit_rows.append(
            {
                "indicator_id": indicator_id,
                "release_date": str(release_date),
                "timestamp_utc": timestamp_utc.isoformat(sep=" "),
                "selected_observation_date": selection["selected_observation_date"],
                "actual": float(selection["actual"]),
            }
        )

if not rows:
    raise ValueError("Historical bronze rebuild produced no rows")

schema = StructType(
    [
        StructField("indicator_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("actual", DoubleType(), True),
        StructField("event", StringType(), True),
    ]
)

final_macro_sdf = (
    spark.createDataFrame(rows, schema=schema)
    .dropDuplicates(["indicator_id", "timestamp"])
    .withColumn("year_partition", F.year("timestamp"))
    .withColumn("ingestion_ts", F.current_timestamp())
)

(final_macro_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("indicator_id")
    .saveAsTable(target_table))

summary_pdf = (
    final_macro_sdf.groupBy("indicator_id")
    .agg(
        F.count("*").alias("row_count"),
        F.min("timestamp").alias("first_timestamp"),
        F.max("timestamp").alias("last_timestamp"),
    )
    .orderBy("indicator_id")
    .toPandas()
)

result = {
    "phase": "macro_bronze_full_rebuild",
    "status": "success",
    "target_table": target_table,
    "production_table_unchanged": target_table != PRODUCTION_TABLE,
    "start_year": start_year,
    "end_year": end_year,
    "haystack_file": haystack_file,
    "row_count": int(final_macro_sdf.count()),
    "indicator_summary": summary_pdf.to_dict(orient="records"),
    "audit_sample": audit_rows[:10],
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))
