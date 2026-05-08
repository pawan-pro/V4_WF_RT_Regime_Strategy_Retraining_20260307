# Databricks notebook source
# RT_02A: Live macro schedule + polling for core releases (Finnhub-first)

# MAGIC %pip install finnhub-python fredapi

# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import re
import time
import finnhub
from fredapi import Fred
from pyspark.sql import functions as F

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_macro_events_live"

FINNHUB_API_KEY = "d5s3pchr01qoo9r2f3h0d5s3pchr01qoo9r2f3hg"
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

TARGET_INDICATORS = {"CPIAUCSL", "PAYEMS", "UMCSENT", "PCEPI", "PCE", "GDPC1"}
TARGET_EVENT_ALIASES = {
    "CPIAUCSL": [
        "consumer price index (cpi)",
        "cpi",
        "cpi mm, sa",
        "consumer price index",
        "consumer price index (cpi)",
    ],
    "PAYEMS": [
        "non-farm payrolls",
        "nonfarm payrolls",
        "non farm payrolls",
        "united states employment non farm payrolls",
        "united states employment nonfarm payrolls",
    ],
    "UMCSENT": [
        "u mich sentiment",
        "michigan consumer sentiment",
        "consumer sentiment",
        "u mich",
    ],
    "PCEPI": [
        "core pce price index",
        "pce price index",
        "personal consumption expenditures price index",
        "pce price index mm",
        "core pce price index mm",
        "united states consumption core pce price index mm",
    ],
    "PCE": [
        "personal consumption expenditures",
        "personal spending",
        "united states consumption personal consumption expenditures",
    ],
    "GDPC1": [
        "real gdp",
        "gdp growth rate",
        "gross domestic product",
        "real gross domestic product",
    ],
}

INDICATOR_MATCH_ORDER = ["PCEPI", "GDPC1", "CPIAUCSL", "PAYEMS", "UMCSENT", "PCE"]
FINNHUB_CALENDAR_RATE_NOT_LEVEL_INDICATORS = {"PCEPI", "PCE", "GDPC1"}
PAYEMS_MIN_LEVEL = 100000.0
PAYEMS_MAX_LEVEL = 250000.0
ALFRED_RETRY_ATTEMPTS = 3
ALFRED_RETRY_BASE_SECONDS = 2

STATUS_PRIORITY = {
    "observed": 3,
    "scheduled": 1,
}

CONFIDENCE_PRIORITY = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "scheduled": 0,
    None: -1,
}


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower())


def canonical_indicator(event_name):
    normalized = normalize_text(event_name)
    # Keep PCE price-index events out of the broader consumption indicator.
    for indicator_id in INDICATOR_MATCH_ORDER:
        aliases = TARGET_EVENT_ALIASES[indicator_id]
        if any(normalize_text(alias) in normalized for alias in aliases):
            if indicator_id == "PCE" and "price index" in normalized:
                return None
            return indicator_id
    return None


def infer_release_stage(event_name):
    normalized = normalize_text(event_name)
    if "prel" in normalized:
        return "prelim"
    if "final" in normalized:
        return "final"
    return "unknown"


def infer_validation_status(indicator_id, release_stage, status):
    if status != "observed":
        return "not_checked"
    if indicator_id in FINNHUB_CALENDAR_RATE_NOT_LEVEL_INDICATORS:
        return "semantic_blocked_rate_not_level"
    if indicator_id == "CPIAUCSL":
        return "validated"
    if indicator_id == "PAYEMS":
        return "validated"
    if indicator_id == "UMCSENT":
        if release_stage == "prelim":
            return "ignored_prelim"
        if release_stage == "final":
            return "validated"
    return "not_checked"


def estimate_confidence(event_name, status):
    if status != "observed":
        return "scheduled"

    normalized = normalize_text(event_name)
    if "prelim" in normalized:
        return "low"
    if "final" in normalized:
        return "high"
    return "medium"


def parse_release_timestamp(raw_timestamp):
    parsed = pd.to_datetime(raw_timestamp, errors="coerce")
    if pd.isna(parsed):
        return None

    # Finnhub economic calendar timestamps are UTC. Naive payload values must
    # not be localized as New York time or release eligibility is delayed.
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(ZoneInfo("UTC"))
    else:
        parsed = parsed.tz_convert(ZoneInfo("UTC"))

    return parsed.tz_convert(ZoneInfo("UTC")).replace(tzinfo=None).to_pydatetime()


fred = Fred(api_key=FRED_API_KEY)
alfred_release_cache = {}


def load_alfred_releases(series_id):
    if series_id in alfred_release_cache:
        return alfred_release_cache[series_id]

    last_error = None
    for attempt in range(ALFRED_RETRY_ATTEMPTS):
        try:
            vintages = fred.get_series_all_releases(series_id).copy()
            observation_col = "observation_date" if "observation_date" in vintages.columns else "date"
            if observation_col not in vintages.columns or "realtime_start" not in vintages.columns:
                alfred_release_cache[series_id] = pd.DataFrame()
                return alfred_release_cache[series_id]

            vintages["observation_date"] = pd.to_datetime(vintages[observation_col], errors="coerce")
            vintages["realtime_start"] = pd.to_datetime(vintages["realtime_start"], errors="coerce")
            vintages["value"] = pd.to_numeric(vintages["value"], errors="coerce")
            vintages = vintages.dropna(subset=["observation_date", "realtime_start", "value"])
            alfred_release_cache[series_id] = vintages
            return vintages
        except Exception as exc:
            last_error = exc
            if attempt + 1 < ALFRED_RETRY_ATTEMPTS:
                time.sleep(ALFRED_RETRY_BASE_SECONDS * (2 ** attempt))

    print(f"ALFRED release fetch failed for {series_id}: {last_error}")
    alfred_release_cache[series_id] = pd.DataFrame()
    return alfred_release_cache[series_id]


def get_alfred_first_release_level(series_id, scheduled_release):
    vintages = load_alfred_releases(series_id)
    if vintages.empty:
        return None

    release_date = pd.Timestamp(scheduled_release).date()
    release_rows = vintages[pd.to_datetime(vintages["realtime_start"]).dt.date == release_date].copy()
    if release_rows.empty:
        return None

    release_rows = release_rows.sort_values(["observation_date", "realtime_start"])
    latest_obs_date = release_rows["observation_date"].max()
    latest_obs_rows = release_rows[release_rows["observation_date"] == latest_obs_date]
    if latest_obs_rows.empty:
        return None

    observed_value = float(latest_obs_rows.iloc[-1]["value"])
    if not (PAYEMS_MIN_LEVEL <= observed_value <= PAYEMS_MAX_LEVEL):
        print(
            "Ignoring PAYEMS ALFRED value outside plausible raw level range: "
            f"value={observed_value} release_date={release_date}"
        )
        return None

    return observed_value


def payems_observed_level_after_release(scheduled_release, now_utc):
    if scheduled_release > now_utc:
        return None
    return get_alfred_first_release_level("PAYEMS", scheduled_release)


def build_rows_from_calendar(payload):
    if not payload:
        return []

    records = payload.get("economicCalendar", [])
    if not records:
        return []

    events = pd.DataFrame(records)
    if events.empty:
        return []

    events = events[
        (events["country"] == "US")
        & (events["impact"].isin(["high", "medium"]))
    ].copy()
    if events.empty:
        return []

    now_utc = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None)
    rows = []
    for _, row in events.iterrows():
        event_name = str(row.get("event") or "")
        indicator_id = canonical_indicator(event_name)
        if indicator_id not in TARGET_INDICATORS:
            continue

        scheduled_release = parse_release_timestamp(row.get("time"))
        if scheduled_release is None:
            continue

        raw_value = pd.to_numeric(row.get("actual"), errors="coerce")
        actual_value = float(raw_value) if pd.notna(raw_value) else None
        status = "scheduled"
        source_provider = "finnhub"
        reconciled_with_alfred = False

        if indicator_id == "PAYEMS":
            actual_value = payems_observed_level_after_release(scheduled_release, now_utc)
            if actual_value is not None:
                source_provider = "alfred"
                reconciled_with_alfred = True

        if actual_value is not None and scheduled_release <= now_utc:
            status = "observed"
        else:
            actual_value = None
        confidence = estimate_confidence(event_name, status)
        release_stage = infer_release_stage(event_name)
        validation_status = infer_validation_status(indicator_id, release_stage, status)

        rows.append(
            {
                "indicator_id": indicator_id,
                "provider_event_name": event_name,
                "canonical_event_name": indicator_id,
                "scheduled_release_ts": scheduled_release,
                "observed_value": actual_value,
                "source_provider": source_provider,
                "status": status,
                "confidence": confidence,
                "reconciled_with_alfred": reconciled_with_alfred,
                "observed_timestamp": scheduled_release if actual_value is not None else None,
                "release_stage": release_stage,
                "validation_status": validation_status,
            }
        )

    if not rows:
        return []

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(
        subset=["indicator_id", "scheduled_release_ts", "provider_event_name"], keep="last"
    )

    if frame.empty:
        return []

    frame["status_priority"] = frame["status"].map(STATUS_PRIORITY)
    frame["confidence_priority"] = frame["confidence"].map(CONFIDENCE_PRIORITY)
    frame = frame.sort_values(
        ["indicator_id", "scheduled_release_ts", "status_priority", "confidence_priority"]
    )
    frame = frame.groupby(["indicator_id", "scheduled_release_ts"], as_index=False).tail(1)

    return frame.drop(columns=["status_priority", "confidence_priority"]).to_dict(orient="records")


def ensure_live_macro_columns():
    existing_columns = {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {TARGET_TABLE}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }
    missing_columns = []
    if "release_stage" not in existing_columns:
        missing_columns.append("release_stage STRING")
    if "validation_status" not in existing_columns:
        missing_columns.append("validation_status STRING")

    if missing_columns:
        spark.sql(
            f"ALTER TABLE {TARGET_TABLE} ADD COLUMNS ({', '.join(missing_columns)})"
        )


spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        indicator_id STRING,
        provider_event_name STRING,
        canonical_event_name STRING,
        scheduled_release_ts TIMESTAMP,
        observed_value DOUBLE,
        source_provider STRING,
        status STRING,
        confidence STRING,
        reconciled_with_alfred BOOLEAN,
        observed_timestamp TIMESTAMP,
        release_stage STRING,
        validation_status STRING,
        year_partition INT,
        ingestion_ts TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (indicator_id)
    """
)
ensure_live_macro_columns()

client = finnhub.Client(api_key=FINNHUB_API_KEY)

window_start = datetime.now(ZoneInfo("America/New_York")).date()
window_end = window_start + timedelta(days=1)

print(f"Fetching Finnhub US macro events from {window_start} to {window_end}")
res = client.calendar_economic(_from=str(window_start), to=str(window_end))
rows = build_rows_from_calendar(res)

if not rows:
    print("No target macro calendar rows found for live window")
    dbutils.notebook.exit("NO_NEW_CORE_MACRO_EVENTS")

macro_df = pd.DataFrame(rows)
macro_df = macro_df.sort_values(["indicator_id", "scheduled_release_ts", "status"])

sdf = spark.createDataFrame(macro_df)
sdf = (
    sdf.withColumn("year_partition", F.year("scheduled_release_ts"))
    .withColumn("ingestion_ts", F.current_timestamp())
)
sdf.createOrReplaceTempView("incoming_rt_macro_events")

merge_sql = f"""
MERGE INTO {TARGET_TABLE} t
USING incoming_rt_macro_events s
ON t.indicator_id = s.indicator_id
   AND t.scheduled_release_ts = s.scheduled_release_ts
WHEN MATCHED THEN UPDATE SET
  provider_event_name = s.provider_event_name,
  canonical_event_name = s.canonical_event_name,
  observed_value = s.observed_value,
  source_provider = s.source_provider,
  status = s.status,
  confidence = s.confidence,
  reconciled_with_alfred = s.reconciled_with_alfred,
  observed_timestamp = s.observed_timestamp,
  release_stage = s.release_stage,
  validation_status = s.validation_status,
  year_partition = s.year_partition,
  ingestion_ts = s.ingestion_ts
WHEN NOT MATCHED THEN INSERT *
"""

execute_merge_sql_with_retry(
    merge_sql,
    operation_name=f"merge Finnhub live macro events into {TARGET_TABLE}",
)

latest_schedule = spark.sql(
    f"""
    SELECT indicator_id,
           provider_event_name,
           scheduled_release_ts,
           observed_value,
           status,
           confidence,
           reconciled_with_alfred,
           release_stage,
           validation_status
    FROM {TARGET_TABLE}
    WHERE indicator_id IN ('CPIAUCSL', 'PAYEMS', 'UMCSENT', 'PCEPI', 'PCE', 'GDPC1')
    ORDER BY scheduled_release_ts DESC, indicator_id
    """
).show(truncate=False)

print("Live macro events written:", len(rows))
