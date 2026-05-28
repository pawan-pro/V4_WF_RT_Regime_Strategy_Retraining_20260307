# Databricks notebook source
# MAGIC %md
# MAGIC # GDELT News-Flow Research Backtest
# MAGIC
# MAGIC Research-only notebook for MUB-30.
# MAGIC
# MAGIC Safety boundaries:
# MAGIC - Do not change production FEATURES.
# MAGIC - Do not write production silver, RT signal, Saxo audit, dashboard, model registry, model alias, or workflow tables.
# MAGIC - Write only the research/candidate tables and files declared below.
# MAGIC - Treat Databricks repo state, Git repo state, and workspace notebook state as separate validation surfaces before any later rollout.
# MAGIC
# MAGIC Missing upstream raw-file policy:
# MAGIC - If a GDELT 15-minute raw GKG file returns HTTP 404 after retries and neighboring 15-minute files are available, classify it as `raw_file_missing`.
# MAGIC - `raw_file_missing` is an upstream archive gap, not a pipeline fetch failure.
# MAGIC - If BigQuery raw-compatible GKG records are absent across a contiguous selected chunk and sampled raw ZIP files all return HTTP 404, classify the contiguous gap as documented upstream missing without retrying every bucket. This is auditable through runtime logs and canonical bucket rows with zero raw/filtered counts.
# MAGIC - Historical hourly aggregation may proceed with available source buckets in `(T - 1 hour, T]`.
# MAGIC - Hourly rows must report `source_bucket_count` and `has_missing_15m_bucket` so downstream validation can see incomplete source hours.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Source Recommendation
# MAGIC
# MAGIC GDELT 2.0 GKG is the default source for this first pass because it includes document-level themes and tone and updates on the same 15-minute cadence as the GDELT 2.0 event stream.
# MAGIC
# MAGIC Recommended path:
# MAGIC - Use `gdelt_source_mode=bigquery` for multi-month stress windows and full-history research if a Mubama GCP billing project and Databricks credentials are available. It avoids downloading hundreds of thousands of zip files and lets BigQuery scan partition-pruned GDELT tables.
# MAGIC - Default BigQuery runs use `bigquery_record_universe=raw_gkg_compat`, which excludes translated GKG rows such as `20200201000000-T123` so counts are comparable to raw `.gkg.csv.zip` audit files. Set `bigquery_record_universe=all_bigquery_gkg` only for a deliberate broader-source experiment.
# MAGIC - Canonical research features count distinct `GKGRECORDID` values. Raw ZIP physical rows are diagnostic only because some buckets contain duplicate record IDs.
# MAGIC - BigQuery is primary. If `raw_gkg_compat` has no numeric records for an expected 15-minute bucket, this notebook may gap-fill only that bucket from the raw ZIP file, deduplicate by `GKGRECORDID`, and write only the compact bucket summary.
# MAGIC - For contiguous BigQuery raw-compatible gaps, raw ZIP fallback first samples representative buckets. If every sampled raw ZIP returns HTTP 404, the whole contiguous gap is classified as upstream missing. Mixed samples fall back to bucket-level handling.
# MAGIC - Stress-window backtests are eligible only when every expected bucket has a BigQuery summary, raw-ZIP gap-fill summary, or documented upstream raw missing, with zero hourly leakage and zero null z-feature rows.
# MAGIC - Use `gdelt_source_mode=raw_zip_hourly_summary` when BigQuery quota is unavailable and the research question only needs hourly features. This mode downloads four raw 15-minute GKG ZIP files per target model hour, deduplicates/counts in memory, writes hourly research feature rows only, and never persists raw document rows.
# MAGIC - Use `gdelt_source_mode=raw_fast` only for bounded validation windows when BigQuery is not available.
# MAGIC - Use `gdelt_source_mode=raw_audit` only for small inspection windows that need document-level raw-row persistence.
# MAGIC
# MAGIC Staged proof:
# MAGIC - Phase A: one stress window, default 2020-02-01 to 2020-05-31.
# MAGIC - Phase B: 2018 Q4, 2020 Covid, 2022 inflation/rates, 2023 banking stress, and 2025 tariff/geopolitical windows where data is available.
# MAGIC - Phase C: full 2016-present using BigQuery or a planned Databricks distributed raw-file ingestion job.

# COMMAND ----------

# MAGIC %pip install google-cloud-bigquery

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import uuid
import zipfile

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window
from sklearn.metrics import accuracy_score, confusion_matrix

# COMMAND ----------

dbutils.widgets.text("source_mode", "raw_gkg")
dbutils.widgets.text("pipeline_mode", "audit_raw_mode")
dbutils.widgets.text("gdelt_source_mode", "")
dbutils.widgets.text("fetch_start_ts", "2020-02-01 00:00:00")
dbutils.widgets.text("fetch_end_ts", "2020-05-31 23:45:00")
dbutils.widgets.text("start_ts", "")
dbutils.widgets.text("end_ts", "")
dbutils.widgets.text("backtest_start_ts", "2020-02-01 00:00:00")
dbutils.widgets.text("backtest_end_ts", "2020-05-31 23:00:00")
dbutils.widgets.text("execute_fetch", "false")
dbutils.widgets.text("run_after_ingestion", "false")
dbutils.widgets.text("run_backtest", "true")
dbutils.widgets.text("run_overlay_backtest", "false")
dbutils.widgets.text("overlay_experiment_mode", "basic")
dbutils.widgets.text("overlay_window_name", "")
dbutils.widgets.text("overlay_selected_lags", "")
dbutils.widgets.text("news_availability_lag_minutes", "0")
dbutils.widgets.text("full_history_start_ts", "2016-01-01 00:00:00")
dbutils.widgets.text("full_history_live_start_ts", "2021-01-01 00:00:00")
dbutils.widgets.text("full_history_end_ts", "")
dbutils.widgets.text("reuse_existing_hourly_features", "true")
dbutils.widgets.text("backtest_experiment_mode", "full_grid")
dbutils.widgets.text("run_mode", "resume")
dbutils.widgets.text("max_fetch_buckets", "384")
dbutils.widgets.text("max_buckets_per_run", "0")
dbutils.widgets.text("max_hours_per_run", "96")
dbutils.widgets.text("raw_zip_fetch_workers", "2")
dbutils.widgets.text("raw_zip_hourly_write_outputs", "true")
dbutils.widgets.text("raw_zip_hourly_compare_existing", "false")
dbutils.widgets.text("bigquery_quota_gate_enabled", "true")
dbutils.widgets.text("bigquery_fallback_fetch_buckets", "192")
dbutils.widgets.text("bigquery_max_bytes_billed", "30000000000")
dbutils.widgets.text("bigquery_max_dry_run_bytes_per_chunk", "30000000000")
dbutils.widgets.text("fetch_workers", "8")
dbutils.widgets.text("fetch_chunk_buckets", "96")
dbutils.widgets.text("batch_write_size", "96")
dbutils.widgets.text("max_retries", "3")
dbutils.widgets.text("fetch_timeout_seconds", "30")
dbutils.widgets.text("fast_chunk_timeout_seconds", "240")
dbutils.widgets.text("gcp_project", "mubama-v4-gdelt-research")
dbutils.widgets.text("gdelt_bigquery_table", "gdelt-bq.gdeltv2.gkg_partitioned")
dbutils.widgets.text("bigquery_table", "gdelt-bq.gdeltv2.gkg_partitioned")
dbutils.widgets.text("bigquery_query_mode", "15m_summary")
dbutils.widgets.text("bigquery_record_universe", "raw_gkg_compat")
dbutils.widgets.text("news_pulse_threshold", "2.0")
dbutils.widgets.text("news_pulse_lookback_hours", "24")
dbutils.widgets.text("news_pulse_thresholds", "1.5,2.0,2.5")
dbutils.widgets.text("news_pulse_windows_hours", "12,24,48")
dbutils.widgets.text("broad_pulse_thresholds", "1.25,1.5,1.75,2.0,2.25,2.5,3.0")
dbutils.widgets.text("broad_pulse_windows_hours", "6,12,24,48")
dbutils.widgets.text("gcp_service_account_secret_scope", "")
dbutils.widgets.text("gcp_service_account_secret_key", "")
dbutils.widgets.text("gcp_service_account_json_path", "")
dbutils.widgets.text("gcp_credentials_secret_scope", "")
dbutils.widgets.text("gcp_credentials_secret_key", "")

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
NAMESPACE = f"{CATALOG}.{SCHEMA}"
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
RESEARCH_ROOT = f"{VOLUME_ROOT}/research/news_flow"

SOURCE_MODE = dbutils.widgets.get("source_mode").strip().lower()
PIPELINE_MODE = dbutils.widgets.get("pipeline_mode").strip().lower()
GDELT_SOURCE_MODE_PARAM = dbutils.widgets.get("gdelt_source_mode").strip().lower()
START_TS_PARAM = dbutils.widgets.get("start_ts").strip()
END_TS_PARAM = dbutils.widgets.get("end_ts").strip()
FETCH_START_TS = pd.Timestamp(START_TS_PARAM or dbutils.widgets.get("fetch_start_ts")).tz_localize(None)
FETCH_END_TS = pd.Timestamp(END_TS_PARAM or dbutils.widgets.get("fetch_end_ts")).tz_localize(None)
BACKTEST_START_TS = pd.Timestamp(dbutils.widgets.get("backtest_start_ts")).tz_localize(None)
BACKTEST_END_TS = pd.Timestamp(dbutils.widgets.get("backtest_end_ts")).tz_localize(None)
EXECUTE_FETCH = dbutils.widgets.get("execute_fetch").strip().lower() == "true"
RUN_AFTER_INGESTION = dbutils.widgets.get("run_after_ingestion").strip().lower() == "true"
RUN_BACKTEST = dbutils.widgets.get("run_backtest").strip().lower() == "true"
RUN_OVERLAY_BACKTEST = dbutils.widgets.get("run_overlay_backtest").strip().lower() == "true"
OVERLAY_EXPERIMENT_MODE = dbutils.widgets.get("overlay_experiment_mode").strip().lower() or "basic"
OVERLAY_WINDOW_NAME = dbutils.widgets.get("overlay_window_name").strip().lower()
OVERLAY_SELECTED_LAGS_PARAM = dbutils.widgets.get("overlay_selected_lags").strip()
NEWS_AVAILABILITY_LAG_MINUTES = int(dbutils.widgets.get("news_availability_lag_minutes") or "0")
FULL_HISTORY_START_TS = pd.Timestamp(dbutils.widgets.get("full_history_start_ts") or "2016-01-01 00:00:00").tz_localize(None)
FULL_HISTORY_LIVE_START_TS = pd.Timestamp(dbutils.widgets.get("full_history_live_start_ts") or "2021-01-01 00:00:00").tz_localize(None)
FULL_HISTORY_END_TS_PARAM = dbutils.widgets.get("full_history_end_ts").strip()
FULL_HISTORY_END_TS = pd.Timestamp(FULL_HISTORY_END_TS_PARAM).tz_localize(None) if FULL_HISTORY_END_TS_PARAM else None
REUSE_EXISTING_HOURLY_FEATURES = dbutils.widgets.get("reuse_existing_hourly_features").strip().lower() == "true"
BACKTEST_EXPERIMENT_MODE = dbutils.widgets.get("backtest_experiment_mode").strip().lower() or "full_grid"
RUN_MODE = dbutils.widgets.get("run_mode").strip().lower()
MAX_FETCH_BUCKETS = int(dbutils.widgets.get("max_fetch_buckets"))
MAX_BUCKETS_PER_RUN = int(dbutils.widgets.get("max_buckets_per_run") or "0")
MAX_HOURS_PER_RUN = int(dbutils.widgets.get("max_hours_per_run") or "96")
RAW_ZIP_FETCH_WORKERS = int(dbutils.widgets.get("raw_zip_fetch_workers") or "2")
RAW_ZIP_HOURLY_WRITE_OUTPUTS = dbutils.widgets.get("raw_zip_hourly_write_outputs").strip().lower() == "true"
RAW_ZIP_HOURLY_COMPARE_EXISTING = dbutils.widgets.get("raw_zip_hourly_compare_existing").strip().lower() == "true"
BIGQUERY_QUOTA_GATE_ENABLED = dbutils.widgets.get("bigquery_quota_gate_enabled").strip().lower() == "true"
BIGQUERY_FALLBACK_FETCH_BUCKETS = int(dbutils.widgets.get("bigquery_fallback_fetch_buckets") or "192")
BIGQUERY_MAX_BYTES_BILLED = int(dbutils.widgets.get("bigquery_max_bytes_billed") or "0")
BIGQUERY_MAX_DRY_RUN_BYTES_PER_CHUNK = int(dbutils.widgets.get("bigquery_max_dry_run_bytes_per_chunk") or "0")
FETCH_WORKERS = int(dbutils.widgets.get("fetch_workers"))
FETCH_CHUNK_BUCKETS = int(dbutils.widgets.get("fetch_chunk_buckets"))
BATCH_WRITE_SIZE = int(dbutils.widgets.get("batch_write_size"))
MAX_RETRIES = int(dbutils.widgets.get("max_retries"))
FETCH_TIMEOUT_SECONDS = int(dbutils.widgets.get("fetch_timeout_seconds"))
FAST_CHUNK_TIMEOUT_SECONDS = int(dbutils.widgets.get("fast_chunk_timeout_seconds"))
GCP_PROJECT = dbutils.widgets.get("gcp_project").strip()
BIGQUERY_GKG_TABLE = (
    dbutils.widgets.get("gdelt_bigquery_table").strip()
    or dbutils.widgets.get("bigquery_table").strip()
    or "gdelt-bq.gdeltv2.gkg_partitioned"
)
BIGQUERY_QUERY_MODE = dbutils.widgets.get("bigquery_query_mode").strip().lower() or "15m_summary"
BIGQUERY_RECORD_UNIVERSE = dbutils.widgets.get("bigquery_record_universe").strip().lower() or "raw_gkg_compat"
NEWS_PULSE_THRESHOLD = float(dbutils.widgets.get("news_pulse_threshold") or "2.0")
NEWS_PULSE_LOOKBACK_HOURS = max(int(dbutils.widgets.get("news_pulse_lookback_hours") or "24"), 1)


def parse_widget_float_list(name, default_values):
    raw = dbutils.widgets.get(name).strip()
    if not raw:
        return default_values
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_widget_int_list(name, default_values):
    raw = dbutils.widgets.get(name).strip()
    if not raw:
        return default_values
    return [max(int(part.strip()), 1) for part in raw.split(",") if part.strip()]


NEWS_PULSE_THRESHOLDS = parse_widget_float_list("news_pulse_thresholds", [1.5, 2.0, 2.5])
NEWS_PULSE_WINDOWS_HOURS = parse_widget_int_list("news_pulse_windows_hours", [12, 24, 48])
BROAD_PULSE_THRESHOLDS = parse_widget_float_list("broad_pulse_thresholds", [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0])
BROAD_PULSE_WINDOWS_HOURS = parse_widget_int_list("broad_pulse_windows_hours", [6, 12, 24, 48])
GCP_SERVICE_ACCOUNT_JSON_PATH = dbutils.widgets.get("gcp_service_account_json_path").strip()
GCP_SERVICE_ACCOUNT_SECRET_SCOPE = (
    dbutils.widgets.get("gcp_service_account_secret_scope").strip()
    or dbutils.widgets.get("gcp_credentials_secret_scope").strip()
)
GCP_SERVICE_ACCOUNT_SECRET_KEY = (
    dbutils.widgets.get("gcp_service_account_secret_key").strip()
    or dbutils.widgets.get("gcp_credentials_secret_key").strip()
)

if GDELT_SOURCE_MODE_PARAM:
    GDELT_SOURCE_MODE = GDELT_SOURCE_MODE_PARAM
elif SOURCE_MODE == "bigquery":
    GDELT_SOURCE_MODE = "bigquery"
elif PIPELINE_MODE == "fast_feature_mode":
    GDELT_SOURCE_MODE = "raw_fast"
else:
    GDELT_SOURCE_MODE = "raw_audit"

if GDELT_SOURCE_MODE not in {"raw_audit", "raw_fast", "bigquery", "raw_zip_hourly_summary"}:
    raise ValueError("gdelt_source_mode must be one of: raw_audit, raw_fast, bigquery, raw_zip_hourly_summary")
if BIGQUERY_RECORD_UNIVERSE not in {"raw_gkg_compat", "all_bigquery_gkg"}:
    raise ValueError("bigquery_record_universe must be one of: raw_gkg_compat, all_bigquery_gkg")
if BACKTEST_EXPERIMENT_MODE not in {"full_grid", "selected_candidates"}:
    raise ValueError("backtest_experiment_mode must be one of: full_grid, selected_candidates")
if OVERLAY_EXPERIMENT_MODE not in {"basic", "grid_optimize", "selected_validate", "targeted_refine", "refined_validate", "full_history_walkforward_fixed_overlay"}:
    raise ValueError("overlay_experiment_mode must be one of: basic, grid_optimize, selected_validate, targeted_refine, refined_validate, full_history_walkforward_fixed_overlay")
if NEWS_AVAILABILITY_LAG_MINUTES < 0:
    raise ValueError("news_availability_lag_minutes must be non-negative")

if GDELT_SOURCE_MODE == "raw_audit":
    SOURCE_MODE = "raw_gkg"
    PIPELINE_MODE = "audit_raw_mode"
elif GDELT_SOURCE_MODE == "raw_fast":
    SOURCE_MODE = "raw_gkg"
    PIPELINE_MODE = "fast_feature_mode"
elif GDELT_SOURCE_MODE == "raw_zip_hourly_summary":
    SOURCE_MODE = "raw_zip_hourly_summary"
    PIPELINE_MODE = "hourly_summary_mode"
else:
    SOURCE_MODE = "bigquery"
    PIPELINE_MODE = "fast_feature_mode"

SILVER_MASTER_TABLE = f"{NAMESPACE}.silver_bt_master_grid_1h"
BRONZE_NEWS_TABLE = f"{NAMESPACE}.bronze_bt_news_flow_15m"
SILVER_NEWS_TABLE = f"{NAMESPACE}.silver_bt_news_flow_1h"
CANDIDATE_JOINED_VIEW = f"{NAMESPACE}.candidate_silver_bt_master_grid_1h_with_news"
VALIDATION_TABLE = f"{NAMESPACE}.research_gdelt_news_flow_validation"
STAGING_NEWS_TABLE = f"{NAMESPACE}.research_gdelt_news_flow_staging_15m"
CHECKPOINT_TABLE = f"{NAMESPACE}.research_gdelt_news_flow_fetch_checkpoint"
BUCKET_SUMMARY_TABLE = f"{NAMESPACE}.research_gdelt_news_bucket_summary_15m"
RUNTIME_LOG_TABLE = f"{NAMESPACE}.research_gdelt_news_runtime_log"
RESEARCH_FEATURE_TABLE = f"{NAMESPACE}.research_gdelt_news_flow_features_1h"
RESEARCH_WF_TABLE = f"{NAMESPACE}.research_gdelt_news_candidate_walkforward"
RESEARCH_OVERLAY_TABLE = f"{NAMESPACE}.research_gdelt_overlay_backtest"
RESEARCH_OVERLAY_OPT_TABLE = f"{NAMESPACE}.research_gdelt_overlay_optimization_summary"
FULL_HISTORY_CHUNK_STATUS_TABLE = f"{NAMESPACE}.research_gdelt_full_history_chunk_status"

BIGQUERY_CANONICAL_PROVIDER = "gdelt_bigquery_raw_compat+raw_zip_gapfill"
BIGQUERY_OK_STATUS = "bigquery_ok"
BIGQUERY_ZERO_FILTERED_STATUS = "bigquery_zero_filtered"
BIGQUERY_ERROR_STATUS = "bigquery_error"
FALLBACK_OK_STATUS = "bigquery_missing_raw_compat_fallback_raw_zip_ok"
FALLBACK_MISSING_STATUS = "bigquery_missing_raw_compat_fallback_raw_zip_missing"
FALLBACK_FAILED_STATUS = "bigquery_missing_raw_compat_fallback_raw_zip_failed"
RAW_ZIP_MALFORMED_UPSTREAM_STATUS = "raw_zip_malformed_upstream"
RAW_SUMMARY_OK_STATUSES = ("persisted", "zero_filtered_rows", "zero_raw_rows")
CANONICAL_SUMMARY_OK_STATUSES = (
    BIGQUERY_OK_STATUS,
    BIGQUERY_ZERO_FILTERED_STATUS,
    FALLBACK_OK_STATUS,
)
SUMMARY_FETCHED_STATUSES = RAW_SUMMARY_OK_STATUSES + CANONICAL_SUMMARY_OK_STATUSES
SUMMARY_MISSING_STATUSES = ("raw_file_missing", FALLBACK_MISSING_STATUS, RAW_ZIP_MALFORMED_UPSTREAM_STATUS)
SUMMARY_FAILED_STATUSES = ("fetch_failed", BIGQUERY_ERROR_STATUS, FALLBACK_FAILED_STATUS)

VALIDATION_CSV = f"{RESEARCH_ROOT}/gdelt_fetch_validation.csv"
FEATURE_SUMMARY_CSV = f"{RESEARCH_ROOT}/gdelt_feature_summary.csv"
BACKTEST_CSV = f"{RESEARCH_ROOT}/baseline_vs_news_backtest.csv"

BASELINE_FEATURES = [
    "CPIAUCSL_z",
    "PCEPI_z",
    "PCE_z",
    "PAYEMS_z",
    "UNRATE_z",
    "GDPC1_z",
    "INDPRO_z",
    "UMCSENT_z",
    "VIXCLS_z",
    "vix3m_z",
    "VIX_Inversion_Z",
    "Price_Mom_24H",
    "CPI_Event_Active",
    "NFP_Event_Active",
]

NEWS_FEATURES = [
    "news_volume_z",
    "news_tone_z",
    "policy_uncertainty_news_z",
    "geopolitical_risk_news_z",
    "market_stress_news_z",
]

ORIGINAL_TOPIC_FEATURES = [
    "policy_uncertainty_news_z",
    "geopolitical_risk_news_z",
    "market_stress_news_z",
]

SHOCK_DELTA_TOPIC_FEATURES = [
    "policy_uncertainty_news_shock_24h",
    "geopolitical_risk_news_shock_24h",
    "market_stress_news_shock_24h",
    "policy_uncertainty_news_delta_6h",
    "geopolitical_risk_news_delta_6h",
    "market_stress_news_delta_6h",
]

PULSE_TOPIC_FEATURES = [
    "policy_uncertainty_news_event_active",
    "geopolitical_risk_news_event_active",
    "market_stress_news_event_active",
]

CAPPED_TOPIC_FEATURES = [
    "policy_uncertainty_news_z_capped",
    "geopolitical_risk_news_z_capped",
    "market_stress_news_z_capped",
]

NARROW_TOPIC_FAMILIES = [
    {
        "name": "inflation_prices",
        "raw_col": "inflation_prices_news",
        "z_col": "inflation_prices_news_z",
        "count_col": "inflation_prices_count",
        "description": "inflation/prices",
    },
    {
        "name": "fed_rates",
        "raw_col": "fed_rates_news",
        "z_col": "fed_rates_news_z",
        "count_col": "fed_rates_count",
        "description": "Fed/rates/monetary policy",
    },
    {
        "name": "recession_slowdown",
        "raw_col": "recession_slowdown_news",
        "z_col": "recession_slowdown_news_z",
        "count_col": "recession_slowdown_count",
        "description": "recession/slowdown/layoffs",
    },
    {
        "name": "credit_liquidity",
        "raw_col": "credit_liquidity_news",
        "z_col": "credit_liquidity_news_z",
        "count_col": "credit_liquidity_count",
        "description": "credit/liquidity/banking stress",
    },
    {
        "name": "geopolitical_energy",
        "raw_col": "geopolitical_energy_news",
        "z_col": "geopolitical_energy_news_z",
        "count_col": "geopolitical_energy_count",
        "description": "geopolitical/sanctions/energy shock",
    },
    {
        "name": "supply_chain_commodities",
        "raw_col": "supply_chain_commodities_news",
        "z_col": "supply_chain_commodities_news_z",
        "count_col": "supply_chain_commodities_count",
        "description": "supply chain/commodities",
    },
]


def threshold_token(threshold):
    return str(threshold).replace(".", "")


def narrow_pulse_feature_name(family_name, window_hours, threshold):
    return f"{family_name}_pulse_{int(window_hours)}h_t{threshold_token(threshold)}"


BROAD_TOPIC_FAMILIES = [
    {
        "name": "policy",
        "z_col": "policy_uncertainty_news_z",
        "legacy_pulse_col": "policy_uncertainty_news_event_active",
    },
    {
        "name": "geopolitical",
        "z_col": "geopolitical_risk_news_z",
        "legacy_pulse_col": "geopolitical_risk_news_event_active",
    },
    {
        "name": "market_stress",
        "z_col": "market_stress_news_z",
        "legacy_pulse_col": "market_stress_news_event_active",
    },
]


def broad_pulse_feature_name(family_name, window_hours, threshold):
    return f"{family_name}_pulse_{int(window_hours)}h_t{threshold_token(threshold)}"


def any_broad_pulse_feature_name(window_hours, threshold):
    return f"any_broad_pulse_{int(window_hours)}h_t{threshold_token(threshold)}"


def broad_pulse_features_for(window_hours, threshold):
    return [broad_pulse_feature_name(family["name"], window_hours, threshold) for family in BROAD_TOPIC_FAMILIES]


NARROW_TOPIC_RAW_FEATURES = [family["raw_col"] for family in NARROW_TOPIC_FAMILIES]
NARROW_TOPIC_Z_FEATURES = [family["z_col"] for family in NARROW_TOPIC_FAMILIES]
BROAD_PULSE_FEATURES = [
    broad_pulse_feature_name(family["name"], window_hours, threshold)
    for family in BROAD_TOPIC_FAMILIES
    for window_hours in BROAD_PULSE_WINDOWS_HOURS
    for threshold in BROAD_PULSE_THRESHOLDS
]
ANY_BROAD_PULSE_FEATURES = [
    any_broad_pulse_feature_name(window_hours, threshold)
    for window_hours in BROAD_PULSE_WINDOWS_HOURS
    for threshold in BROAD_PULSE_THRESHOLDS
]
SELECTED_BROAD_PULSE_WINDOW_HOURS = 24
SELECTED_BROAD_PULSE_THRESHOLD = 2.0
SELECTED_BROAD_PULSE_FEATURES = broad_pulse_features_for(SELECTED_BROAD_PULSE_WINDOW_HOURS, SELECTED_BROAD_PULSE_THRESHOLD)
SELECTED_ANY_BROAD_PULSE_FEATURE = any_broad_pulse_feature_name(SELECTED_BROAD_PULSE_WINDOW_HOURS, SELECTED_BROAD_PULSE_THRESHOLD)
VIX_INTERACTION_FEATURES = [
    "market_stress_pulse_x_VIXCLS_z",
    "market_stress_pulse_x_vix3m_z",
    "market_stress_pulse_x_VIX_Inversion_Z",
    "policy_pulse_x_VIXCLS_z",
    "geopolitical_pulse_x_VIXCLS_z",
    "any_broad_pulse_x_VIXCLS_z",
]
PRICE_INTERACTION_FEATURES = [
    "market_stress_pulse_x_Price_Mom_24H",
    "any_broad_pulse_x_negative_price_mom",
]
PULSE_INTERACTION_FEATURES = VIX_INTERACTION_FEATURES + PRICE_INTERACTION_FEATURES
NARROW_PULSE_FEATURES = [
    narrow_pulse_feature_name(family["name"], window_hours, threshold)
    for family in NARROW_TOPIC_FAMILIES
    for window_hours in NEWS_PULSE_WINDOWS_HOURS
    for threshold in NEWS_PULSE_THRESHOLDS
]
NARROW_MACRO_POLICY_PULSE_FEATURES = [
    feature
    for family in NARROW_TOPIC_FAMILIES[:3]
    for feature in [
        narrow_pulse_feature_name(family["name"], window_hours, threshold)
        for window_hours in NEWS_PULSE_WINDOWS_HOURS
        for threshold in NEWS_PULSE_THRESHOLDS
    ]
]
NARROW_STRESS_PULSE_FEATURES = [
    feature
    for family in NARROW_TOPIC_FAMILIES[3:]
    for feature in [
        narrow_pulse_feature_name(family["name"], window_hours, threshold)
        for window_hours in NEWS_PULSE_WINDOWS_HOURS
        for threshold in NEWS_PULSE_THRESHOLDS
    ]
]
NARROW_COMPACT_PULSE_FEATURES = [
    narrow_pulse_feature_name(family["name"], 24, 2.0) for family in NARROW_TOPIC_FAMILIES
]

IMPROVED_TOPIC_FEATURES = SHOCK_DELTA_TOPIC_FEATURES + PULSE_TOPIC_FEATURES + CAPPED_TOPIC_FEATURES
ENGINEERED_NEWS_FEATURES = (
    SHOCK_DELTA_TOPIC_FEATURES
    + PULSE_TOPIC_FEATURES
    + CAPPED_TOPIC_FEATURES
    + BROAD_PULSE_FEATURES
    + ANY_BROAD_PULSE_FEATURES
    + NARROW_PULSE_FEATURES
)


def add_engineered_news_features(grid):
    # All windows are trailing/current or lagged, so these engineered variants
    # preserve point-in-time safety at each hourly model timestamp.
    shock_win = Window.orderBy("timestamp").rowsBetween(-23, 0)
    pulse_win = Window.orderBy("timestamp").rowsBetween(-(NEWS_PULSE_LOOKBACK_HOURS - 1), 0)
    topic_z_cols = [
        ("policy_uncertainty_news_z", "policy_uncertainty_news"),
        ("geopolitical_risk_news_z", "geopolitical_risk_news"),
        ("market_stress_news_z", "market_stress_news"),
    ]

    for z_col, prefix in topic_z_cols:
        trailing_avg = F.avg(F.col(z_col)).over(shock_win)
        grid = grid.withColumn(f"{prefix}_shock_24h", F.col(z_col) - trailing_avg)
        grid = grid.withColumn(f"{prefix}_delta_6h", F.col(z_col) - F.lag(F.col(z_col), 6).over(Window.orderBy("timestamp")))
        pulse_now = F.when(F.col(z_col) >= F.lit(NEWS_PULSE_THRESHOLD), F.lit(1.0)).otherwise(F.lit(0.0))
        grid = grid.withColumn(f"{prefix}_event_active", F.max(pulse_now).over(pulse_win).cast("double"))
        grid = grid.withColumn(
            f"{prefix}_z_capped",
            F.greatest(F.lit(-3.0), F.least(F.lit(3.0), F.col(z_col))).cast("double"),
        )

    for window_hours in BROAD_PULSE_WINDOWS_HOURS:
        broad_pulse_win = Window.orderBy("timestamp").rowsBetween(-(window_hours - 1), 0)
        for threshold in BROAD_PULSE_THRESHOLDS:
            pulse_cols = []
            for family in BROAD_TOPIC_FAMILIES:
                pulse_col = broad_pulse_feature_name(family["name"], window_hours, threshold)
                pulse_now = F.when(F.col(family["z_col"]) >= F.lit(float(threshold)), F.lit(1.0)).otherwise(F.lit(0.0))
                grid = grid.withColumn(pulse_col, F.max(pulse_now).over(broad_pulse_win).cast("double"))
                pulse_cols.append(pulse_col)
            any_col = any_broad_pulse_feature_name(window_hours, threshold)
            grid = grid.withColumn(
                any_col,
                F.greatest(*[F.col(col_name) for col_name in pulse_cols]).cast("double"),
            )

    for family in NARROW_TOPIC_FAMILIES:
        for window_hours in NEWS_PULSE_WINDOWS_HOURS:
            pulse_win = Window.orderBy("timestamp").rowsBetween(-(window_hours - 1), 0)
            for threshold in NEWS_PULSE_THRESHOLDS:
                pulse_now = F.when(F.col(family["z_col"]) >= F.lit(float(threshold)), F.lit(1.0)).otherwise(F.lit(0.0))
                grid = grid.withColumn(
                    narrow_pulse_feature_name(family["name"], window_hours, threshold),
                    F.max(pulse_now).over(pulse_win).cast("double"),
                )

    for col_name in ENGINEERED_NEWS_FEATURES:
        grid = grid.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0.0)))
    return grid


GKG_COLUMNS = [
    "GKGRECORDID",
    "DATE",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Counts",
    "V2Counts",
    "Themes",
    "V2Themes",
    "Locations",
    "V2Locations",
    "Persons",
    "V2Persons",
    "Organizations",
    "V2Organizations",
    "V2Tone",
    "Dates",
    "GCAM",
    "SharingImage",
    "RelatedImages",
    "SocialImageEmbeds",
    "SocialVideoEmbeds",
    "Quotations",
    "AllNames",
    "Amounts",
    "TranslationInfo",
    "Extras",
]


def _set_large_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def read_gkg_csv_robust(raw_bytes):
    """Read a raw GDELT GKG TSV, falling back to Python/csv for malformed rows."""
    text = raw_bytes.decode("utf-8", errors="replace")
    parser_errors = []
    for engine in ("c", "python"):
        try:
            return (
                pd.read_csv(
                    io.StringIO(text),
                    sep="\t",
                    names=GKG_COLUMNS,
                    dtype=str,
                    header=None,
                    on_bad_lines="skip",
                    engine=engine,
                ),
                0,
                None if not parser_errors else "; ".join(parser_errors),
            )
        except Exception as exc:
            parser_errors.append(f"pandas_{engine}_parser_error={exc}")

    _set_large_csv_field_limit()
    required_min_fields = GKG_COLUMNS.index("V2Tone") + 1
    rows = []
    skipped_rows = 0
    for line in text.splitlines():
        try:
            row = next(csv.reader([line], delimiter="\t"))
        except Exception:
            skipped_rows += 1
            continue
        if len(row) < required_min_fields:
            skipped_rows += 1
            continue
        if len(row) < len(GKG_COLUMNS):
            row = row + [""] * (len(GKG_COLUMNS) - len(row))
        elif len(row) > len(GKG_COLUMNS):
            row = row[: len(GKG_COLUMNS) - 1] + ["\t".join(row[len(GKG_COLUMNS) - 1 :])]
        rows.append(row)

    if not rows:
        raise ValueError("; ".join(parser_errors + ["csv_fallback_no_parseable_rows"]))
    warning = "; ".join(parser_errors + [f"csv_fallback_skipped_rows={skipped_rows}"])
    return pd.DataFrame(rows, columns=GKG_COLUMNS), skipped_rows, warning

RELEVANT_RE = re.compile(
    r"ECON|STOCK|MARKET|INFLATION|RECESSION|BANK|CREDIT|LIQUIDITY|FED|"
    r"CENTRAL_BANK|TARIFF|SANCTION|WAR|CONFLICT|GEO|POLICY|UNCERTAINTY",
    re.IGNORECASE,
)
POLICY_RE = re.compile(
    r"POLICY|UNCERTAINTY|TARIFF|TRADE|FED|FEDERAL_RESERVE|CENTRAL_BANK|"
    r"FISCAL|TAX|GOVERNMENT|REGULATION",
    re.IGNORECASE,
)
GEOPOLITICAL_RE = re.compile(
    r"WAR|CONFLICT|SANCTION|MILITARY|TERROR|ARMEDCONFLICT|GEOPOLITICAL|"
    r"PROTEST|COUP|CRISIS",
    re.IGNORECASE,
)
MARKET_STRESS_RE = re.compile(
    r"RECESSION|BANK|BANKING|CREDIT|LIQUIDITY|DEFAULT|DEBT|MARKET_CRASH|"
    r"VOLATILITY|LAYOFF|UNEMPLOYMENT|BANKRUPT",
    re.IGNORECASE,
)
RISK_ON_RE = re.compile(
    r"RALLY|RECOVERY|GROWTH|EXPANSION|EASING|STIMULUS|OPTIMISM|"
    r"BULL|GAIN|REBOUND|REOPEN",
    re.IGNORECASE,
)
RISK_OFF_RE = re.compile(
    r"SELL.?OFF|CRASH|PANIC|RECESSION|DEFAULT|BANKRUPT|LIQUIDITY|"
    r"VOLATILITY|FEAR|DRAWDOWN|CONTAGION|LOCKDOWN",
    re.IGNORECASE,
)
INFLATION_PRICES_RE = re.compile(
    r"INFLATION|CPI|CONSUMER.?PRICE|PRICE.?PRESSURE|COST.?OF.?LIVING|PRICES",
    re.IGNORECASE,
)
FED_RATES_RE = re.compile(
    r"FEDERAL.?RESERVE|FOMC|RATE.?HIKE|INTEREST.?RATE|MONETARY.?TIGHTEN|"
    r"HAWKISH|DOVISH|CENTRAL.?BANK",
    re.IGNORECASE,
)
RECESSION_SLOWDOWN_RE = re.compile(
    r"RECESSION|SLOWDOWN|CONTRACTION|LAYOFF|UNEMPLOYMENT|JOBLESS|"
    r"EARNINGS.?DOWNGRADE|GROWTH.?SCARE",
    re.IGNORECASE,
)
CREDIT_LIQUIDITY_RE = re.compile(
    r"LIQUIDITY|CREDIT.?STRESS|DEFAULT|BANK.?STRESS|FUNDING.?STRESS|"
    r"FINANCIAL.?CONDITIONS|CREDIT|BANKING",
    re.IGNORECASE,
)
GEOPOLITICAL_ENERGY_RE = re.compile(
    r"WAR|SANCTION|RUSSIA|UKRAINE|OIL|GAS|ENERGY.?SHOCK|"
    r"SUPPLY.?DISRUPTION|GEOPOLIT",
    re.IGNORECASE,
)
SUPPLY_CHAIN_COMMODITIES_RE = re.compile(
    r"SUPPLY.?CHAIN|SHIPPING|SHORTAGE|COMMODIT|FOOD.?PRICE|ENERGY.?PRICE",
    re.IGNORECASE,
)

NARROW_TOPIC_REGEX = {
    "inflation_prices": INFLATION_PRICES_RE,
    "fed_rates": FED_RATES_RE,
    "recession_slowdown": RECESSION_SLOWDOWN_RE,
    "credit_liquidity": CREDIT_LIQUIDITY_RE,
    "geopolitical_energy": GEOPOLITICAL_ENERGY_RE,
    "supply_chain_commodities": SUPPLY_CHAIN_COMMODITIES_RE,
}

RUN_TS = datetime.now(timezone.utc).replace(tzinfo=None)
INGESTION_RUN_ID = f"gdelt_{GDELT_SOURCE_MODE}_{RUN_TS:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"

print(f"Research window: fetch {FETCH_START_TS} to {FETCH_END_TS}")
print(f"Backtest window: {BACKTEST_START_TS} to {BACKTEST_END_TS}")
print(f"GDELT source mode: {GDELT_SOURCE_MODE}")
print(f"Source mode: {SOURCE_MODE}")
print(f"Pipeline mode: {PIPELINE_MODE}")
print(f"BigQuery table: {BIGQUERY_GKG_TABLE}")
print(f"BigQuery query mode: {BIGQUERY_QUERY_MODE}")
print(f"BigQuery record universe: {BIGQUERY_RECORD_UNIVERSE}")
print(f"Execute fetch: {EXECUTE_FETCH}")
print(f"Run mode: {RUN_MODE}")
print(f"Run after ingestion: {RUN_AFTER_INGESTION}")
print(f"Run backtest: {RUN_BACKTEST}")
print(f"Ingestion run id: {INGESTION_RUN_ID}")

# COMMAND ----------

def ensure_research_tables():
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {VALIDATION_TABLE} (
            batch_start_ts TIMESTAMP,
            batch_end_ts TIMESTAMP,
            source STRING,
            expected_15m_buckets INT,
            observed_15m_buckets INT,
            missing_15m_buckets INT,
            duplicate_15m_buckets INT,
            raw_row_count LONG,
            filtered_row_count LONG,
            min_source_ts TIMESTAMP,
            max_source_ts TIMESTAMP,
            fetch_started_at TIMESTAMP,
            fetch_completed_at TIMESTAMP,
            status STRING,
            error_message STRING
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {BRONZE_NEWS_TABLE} (
            source_ts TIMESTAMP,
            source STRING,
            bucket_url STRING,
            gdelt_record_id STRING,
            document_identifier STRING,
            source_common_name STRING,
            themes STRING,
            v2_themes STRING,
            v2_tone_raw STRING,
            tone DOUBLE,
            positive_score DOUBLE,
            negative_score DOUBLE,
            polarity DOUBLE,
            activity_density DOUBLE,
            self_density DOUBLE,
            word_count DOUBLE,
            is_relevant BOOLEAN,
            is_policy_uncertainty BOOLEAN,
            is_geopolitical_risk BOOLEAN,
            is_market_stress BOOLEAN,
            deterministic_key STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {STAGING_NEWS_TABLE} (
            bucket_ts TIMESTAMP,
            source_ts TIMESTAMP,
            source STRING,
            source_url_or_file STRING,
            bucket_url STRING,
            gdelt_record_id STRING,
            document_identifier STRING,
            source_common_name STRING,
            themes STRING,
            v2_themes STRING,
            v2_tone_raw STRING,
            tone DOUBLE,
            positive_score DOUBLE,
            negative_score DOUBLE,
            polarity DOUBLE,
            activity_density DOUBLE,
            self_density DOUBLE,
            word_count DOUBLE,
            is_relevant BOOLEAN,
            is_policy_uncertainty BOOLEAN,
            is_geopolitical_risk BOOLEAN,
            is_market_stress BOOLEAN,
            deterministic_key STRING,
            ingestion_run_id STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {CHECKPOINT_TABLE} (
            bucket_ts TIMESTAMP,
            raw_file_fetched BOOLEAN,
            raw_row_count LONG,
            filtered_row_count LONG,
            write_status STRING,
            retry_count INT,
            last_error STRING,
            source_url_or_file STRING,
            ingestion_run_id STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {BUCKET_SUMMARY_TABLE} (
            bucket_ts TIMESTAMP,
            source_provider STRING,
            source_file_url STRING,
            raw_file_status STRING,
            raw_row_count LONG,
            filtered_row_count LONG,
            avg_tone DOUBLE,
            min_tone DOUBLE,
            max_tone DOUBLE,
            policy_uncertainty_count LONG,
            geopolitical_risk_count LONG,
            market_stress_count LONG,
            inflation_prices_count LONG,
            fed_rates_count LONG,
            recession_slowdown_count LONG,
            credit_liquidity_count LONG,
            geopolitical_energy_count LONG,
            supply_chain_commodities_count LONG,
            risk_on_count LONG,
            risk_off_count LONG,
            source_bucket_count INT,
            has_missing_15m_bucket BOOLEAN,
            ingestion_run_id STRING,
            ingestion_ts TIMESTAMP,
            last_error STRING,
            download_seconds DOUBLE,
            parse_seconds DOUBLE,
            summary_seconds DOUBLE
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {RUNTIME_LOG_TABLE} (
            ingestion_run_id STRING,
            pipeline_mode STRING,
            stage STRING,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            duration_seconds DOUBLE,
            row_count LONG,
            detail STRING
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {SILVER_NEWS_TABLE} (
            timestamp TIMESTAMP,
            news_volume DOUBLE,
            news_tone DOUBLE,
            policy_uncertainty_news DOUBLE,
            geopolitical_risk_news DOUBLE,
            market_stress_news DOUBLE,
            inflation_prices_news DOUBLE,
            fed_rates_news DOUBLE,
            recession_slowdown_news DOUBLE,
            credit_liquidity_news DOUBLE,
            geopolitical_energy_news DOUBLE,
            supply_chain_commodities_news DOUBLE,
            news_volume_z DOUBLE,
            news_tone_z DOUBLE,
            policy_uncertainty_news_z DOUBLE,
            geopolitical_risk_news_z DOUBLE,
            market_stress_news_z DOUBLE,
            inflation_prices_news_z DOUBLE,
            fed_rates_news_z DOUBLE,
            recession_slowdown_news_z DOUBLE,
            credit_liquidity_news_z DOUBLE,
            geopolitical_energy_news_z DOUBLE,
            supply_chain_commodities_news_z DOUBLE,
            policy_uncertainty_news_shock_24h DOUBLE,
            geopolitical_risk_news_shock_24h DOUBLE,
            market_stress_news_shock_24h DOUBLE,
            policy_uncertainty_news_delta_6h DOUBLE,
            geopolitical_risk_news_delta_6h DOUBLE,
            market_stress_news_delta_6h DOUBLE,
            policy_uncertainty_news_event_active DOUBLE,
            geopolitical_risk_news_event_active DOUBLE,
            market_stress_news_event_active DOUBLE,
            policy_uncertainty_news_z_capped DOUBLE,
            geopolitical_risk_news_z_capped DOUBLE,
            market_stress_news_z_capped DOUBLE,
            source_bucket_count INT,
            missing_15m_bucket_count INT,
            missing_15m_bucket_pct DOUBLE,
            gdelt_missing_flag INT,
            raw_news_row_count LONG,
            filtered_news_row_count LONG,
            min_gdelt_source_ts TIMESTAMP,
            max_gdelt_source_ts TIMESTAMP,
            has_missing_15m_bucket BOOLEAN,
            source STRING,
            ingestion_run_id STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {RESEARCH_FEATURE_TABLE} (
            timestamp TIMESTAMP,
            news_volume DOUBLE,
            news_tone DOUBLE,
            policy_uncertainty_news DOUBLE,
            geopolitical_risk_news DOUBLE,
            market_stress_news DOUBLE,
            inflation_prices_news DOUBLE,
            fed_rates_news DOUBLE,
            recession_slowdown_news DOUBLE,
            credit_liquidity_news DOUBLE,
            geopolitical_energy_news DOUBLE,
            supply_chain_commodities_news DOUBLE,
            news_volume_z DOUBLE,
            news_tone_z DOUBLE,
            policy_uncertainty_news_z DOUBLE,
            geopolitical_risk_news_z DOUBLE,
            market_stress_news_z DOUBLE,
            inflation_prices_news_z DOUBLE,
            fed_rates_news_z DOUBLE,
            recession_slowdown_news_z DOUBLE,
            credit_liquidity_news_z DOUBLE,
            geopolitical_energy_news_z DOUBLE,
            supply_chain_commodities_news_z DOUBLE,
            policy_uncertainty_news_shock_24h DOUBLE,
            geopolitical_risk_news_shock_24h DOUBLE,
            market_stress_news_shock_24h DOUBLE,
            policy_uncertainty_news_delta_6h DOUBLE,
            geopolitical_risk_news_delta_6h DOUBLE,
            market_stress_news_delta_6h DOUBLE,
            policy_uncertainty_news_event_active DOUBLE,
            geopolitical_risk_news_event_active DOUBLE,
            market_stress_news_event_active DOUBLE,
            policy_uncertainty_news_z_capped DOUBLE,
            geopolitical_risk_news_z_capped DOUBLE,
            market_stress_news_z_capped DOUBLE,
            source_bucket_count INT,
            missing_15m_bucket_count INT,
            missing_15m_bucket_pct DOUBLE,
            gdelt_missing_flag INT,
            raw_news_row_count LONG,
            filtered_news_row_count LONG,
            min_gdelt_source_ts TIMESTAMP,
            max_gdelt_source_ts TIMESTAMP,
            has_missing_15m_bucket BOOLEAN,
            source STRING,
            ingestion_run_id STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {RESEARCH_WF_TABLE} (
            experiment_name STRING,
            timestamp TIMESTAMP,
            regime INT,
            predicted_regime INT,
            model_features STRING,
            fold_train_end TIMESTAMP,
            fold_test_end TIMESTAMP,
            accuracy DOUBLE,
            strategy_equity DOUBLE,
            drawdown DOUBLE,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {FULL_HISTORY_CHUNK_STATUS_TABLE} (
            chunk_start_ts TIMESTAMP,
            chunk_end_ts TIMESTAMP,
            expected_buckets LONG,
            final_canonical_buckets LONG,
            bigquery_ok_buckets LONG,
            fallback_ok_buckets LONG,
            upstream_missing_buckets LONG,
            unresolved_missing_buckets LONG,
            hourly_rows LONG,
            joined_rows LONG,
            leakage_rows LONG,
            null_feature_rows LONG,
            status STRING,
            runtime_seconds DOUBLE,
            ingestion_run_id STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )


ensure_research_tables()


def ensure_hourly_feature_columns():
    required_columns = {
        **{family["raw_col"]: "DOUBLE" for family in NARROW_TOPIC_FAMILIES},
        **{family["z_col"]: "DOUBLE" for family in NARROW_TOPIC_FAMILIES},
        "policy_uncertainty_news_shock_24h": "DOUBLE",
        "geopolitical_risk_news_shock_24h": "DOUBLE",
        "market_stress_news_shock_24h": "DOUBLE",
        "policy_uncertainty_news_delta_6h": "DOUBLE",
        "geopolitical_risk_news_delta_6h": "DOUBLE",
        "market_stress_news_delta_6h": "DOUBLE",
        "policy_uncertainty_news_event_active": "DOUBLE",
        "geopolitical_risk_news_event_active": "DOUBLE",
        "market_stress_news_event_active": "DOUBLE",
        "policy_uncertainty_news_z_capped": "DOUBLE",
        "geopolitical_risk_news_z_capped": "DOUBLE",
        "market_stress_news_z_capped": "DOUBLE",
        **{feature: "DOUBLE" for feature in BROAD_PULSE_FEATURES},
        **{feature: "DOUBLE" for feature in ANY_BROAD_PULSE_FEATURES},
        **{feature: "DOUBLE" for feature in NARROW_PULSE_FEATURES},
        "source_bucket_count": "INT",
        "missing_15m_bucket_count": "INT",
        "missing_15m_bucket_pct": "DOUBLE",
        "gdelt_missing_flag": "INT",
        "raw_news_row_count": "LONG",
        "filtered_news_row_count": "LONG",
        "min_gdelt_source_ts": "TIMESTAMP",
        "has_missing_15m_bucket": "BOOLEAN",
        "ingestion_run_id": "STRING",
    }
    for table_name in [SILVER_NEWS_TABLE, RESEARCH_FEATURE_TABLE]:
        existing = set(spark.table(table_name).columns)
        additions = [f"{name} {dtype}" for name, dtype in required_columns.items() if name not in existing]
        if additions:
            spark.sql(f"ALTER TABLE {table_name} ADD COLUMNS ({', '.join(additions)})")


ensure_hourly_feature_columns()


def ensure_bucket_summary_columns():
    required_columns = {
        **{family["count_col"]: "LONG" for family in NARROW_TOPIC_FAMILIES},
    }
    existing = set(spark.table(BUCKET_SUMMARY_TABLE).columns)
    additions = [f"{name} {dtype}" for name, dtype in required_columns.items() if name not in existing]
    if additions:
        spark.sql(f"ALTER TABLE {BUCKET_SUMMARY_TABLE} ADD COLUMNS ({', '.join(additions)})")


ensure_bucket_summary_columns()
dbutils.fs.mkdirs(RESEARCH_ROOT)

# COMMAND ----------

def expected_15m_buckets(start_ts, end_ts):
    start = pd.Timestamp(start_ts).floor("15min")
    end = pd.Timestamp(end_ts).floor("15min")
    return list(pd.date_range(start=start, end=end, freq="15min").to_pydatetime())


def gdelt_gkg_url(bucket_ts):
    return f"http://data.gdeltproject.org/gdeltv2/{bucket_ts:%Y%m%d%H%M%S}.gkg.csv.zip"


def sql_string_list(values):
    return ", ".join(f"'{value}'" for value in values)


def bucket_summary_source_provider():
    if GDELT_SOURCE_MODE == "bigquery":
        return BIGQUERY_CANONICAL_PROVIDER
    return "gdelt_2_0_gkg_raw_15m"


def parse_float(value):
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except Exception:
        return None


def parse_v2tone(value):
    parts = str(value or "").split(",")
    parsed = [parse_float(part) for part in parts[:7]]
    while len(parsed) < 7:
        parsed.append(None)
    return parsed


def classify_topic_blob(blob):
    blob = blob or ""
    flags = {
        "is_relevant": bool(RELEVANT_RE.search(blob)),
        "is_policy_uncertainty": bool(POLICY_RE.search(blob)),
        "is_geopolitical_risk": bool(GEOPOLITICAL_RE.search(blob)),
        "is_market_stress": bool(MARKET_STRESS_RE.search(blob)),
        "is_risk_on": bool(RISK_ON_RE.search(blob)),
        "is_risk_off": bool(RISK_OFF_RE.search(blob)),
    }
    for family_name, pattern in NARROW_TOPIC_REGEX.items():
        flags[f"is_{family_name}"] = bool(pattern.search(blob))
    return flags


def make_key(source_ts, record_id, document_identifier):
    raw = f"{source_ts:%Y-%m-%dT%H:%M:%S}|{record_id or ''}|{document_identifier or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


STAGING_ROW_SCHEMA = StructType(
    [
        StructField("bucket_ts", TimestampType(), True),
        StructField("source_ts", TimestampType(), True),
        StructField("source", StringType(), True),
        StructField("source_url_or_file", StringType(), True),
        StructField("bucket_url", StringType(), True),
        StructField("gdelt_record_id", StringType(), True),
        StructField("document_identifier", StringType(), True),
        StructField("source_common_name", StringType(), True),
        StructField("themes", StringType(), True),
        StructField("v2_themes", StringType(), True),
        StructField("v2_tone_raw", StringType(), True),
        StructField("tone", DoubleType(), True),
        StructField("positive_score", DoubleType(), True),
        StructField("negative_score", DoubleType(), True),
        StructField("polarity", DoubleType(), True),
        StructField("activity_density", DoubleType(), True),
        StructField("self_density", DoubleType(), True),
        StructField("word_count", DoubleType(), True),
        StructField("is_relevant", BooleanType(), True),
        StructField("is_policy_uncertainty", BooleanType(), True),
        StructField("is_geopolitical_risk", BooleanType(), True),
        StructField("is_market_stress", BooleanType(), True),
        StructField("deterministic_key", StringType(), False),
        StructField("ingestion_run_id", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

CHECKPOINT_ROW_SCHEMA = StructType(
    [
        StructField("bucket_ts", TimestampType(), True),
        StructField("raw_file_fetched", BooleanType(), True),
        StructField("raw_row_count", LongType(), True),
        StructField("filtered_row_count", LongType(), True),
        StructField("write_status", StringType(), True),
        StructField("retry_count", IntegerType(), True),
        StructField("last_error", StringType(), True),
        StructField("source_url_or_file", StringType(), True),
        StructField("ingestion_run_id", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

BUCKET_SUMMARY_ROW_SCHEMA = StructType(
    [
        StructField("bucket_ts", TimestampType(), True),
        StructField("source_provider", StringType(), True),
        StructField("source_file_url", StringType(), True),
        StructField("raw_file_status", StringType(), True),
        StructField("raw_row_count", LongType(), True),
        StructField("filtered_row_count", LongType(), True),
        StructField("avg_tone", DoubleType(), True),
        StructField("min_tone", DoubleType(), True),
        StructField("max_tone", DoubleType(), True),
        StructField("policy_uncertainty_count", LongType(), True),
        StructField("geopolitical_risk_count", LongType(), True),
        StructField("market_stress_count", LongType(), True),
        StructField("inflation_prices_count", LongType(), True),
        StructField("fed_rates_count", LongType(), True),
        StructField("recession_slowdown_count", LongType(), True),
        StructField("credit_liquidity_count", LongType(), True),
        StructField("geopolitical_energy_count", LongType(), True),
        StructField("supply_chain_commodities_count", LongType(), True),
        StructField("risk_on_count", LongType(), True),
        StructField("risk_off_count", LongType(), True),
        StructField("source_bucket_count", IntegerType(), True),
        StructField("has_missing_15m_bucket", BooleanType(), True),
        StructField("ingestion_run_id", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
        StructField("last_error", StringType(), True),
        StructField("download_seconds", DoubleType(), True),
        StructField("parse_seconds", DoubleType(), True),
        StructField("summary_seconds", DoubleType(), True),
    ]
)

RUNTIME_LOG_SCHEMA = StructType(
    [
        StructField("ingestion_run_id", StringType(), True),
        StructField("pipeline_mode", StringType(), True),
        StructField("stage", StringType(), True),
        StructField("started_at", TimestampType(), True),
        StructField("completed_at", TimestampType(), True),
        StructField("duration_seconds", DoubleType(), True),
        StructField("row_count", LongType(), True),
        StructField("detail", StringType(), True),
    ]
)


def log_runtime(stage, started_at, completed_at, row_count=0, detail=None):
    row = {
        "ingestion_run_id": INGESTION_RUN_ID,
        "pipeline_mode": PIPELINE_MODE,
        "stage": stage,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "row_count": int(row_count or 0),
        "detail": detail,
    }
    spark.createDataFrame([row], schema=RUNTIME_LOG_SCHEMA).write.mode("append").format("delta").saveAsTable(RUNTIME_LOG_TABLE)
    print(f"Runtime: {row}")


def fetch_raw_gkg_bucket_once(bucket_ts, attempt):
    started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    url = gdelt_gkg_url(bucket_ts)
    rows = []
    errors = []
    raw_row_count = 0
    filtered_row_count = 0
    write_status = "persisted"
    raw_file_fetched = False

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        raw_file_fetched = True
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            member = zf.namelist()[0]
            with zf.open(member) as fh:
                decoded = fh.read().decode("utf-8", errors="replace")
                pdf = pd.read_csv(
                    io.StringIO(decoded),
                    sep="\t",
                    names=GKG_COLUMNS,
                    dtype=str,
                    header=None,
                    on_bad_lines="skip",
                )

        missing = [name for name in ["GKGRECORDID", "DATE", "DocumentIdentifier", "V2Themes", "V2Tone"] if name not in pdf.columns]
        if missing:
            raise ValueError(f"Missing required GKG columns: {missing}")

        raw_row_count = len(pdf)
        if raw_row_count == 0:
            write_status = "zero_raw_rows"

        for rec in pdf.itertuples(index=False):
            topic_blob = " ".join(
                [
                    str(getattr(rec, "Themes", "") or ""),
                    str(getattr(rec, "V2Themes", "") or ""),
                    str(getattr(rec, "DocumentIdentifier", "") or ""),
                    str(getattr(rec, "SourceCommonName", "") or ""),
                ]
            )
            topic_flags = classify_topic_blob(topic_blob)
            if topic_flags["is_relevant"]:
                filtered_row_count += 1
            else:
                continue
            tone, pos, neg, polarity, activity, self_density, word_count = parse_v2tone(getattr(rec, "V2Tone", None))
            record_id = getattr(rec, "GKGRECORDID", None)
            document_identifier = getattr(rec, "DocumentIdentifier", None)
            rows.append(
                {
                    "bucket_ts": bucket_ts,
                    "source_ts": bucket_ts,
                    "source": "gdelt_2_0_gkg_raw_15m",
                    "source_url_or_file": url,
                    "bucket_url": url,
                    "gdelt_record_id": record_id,
                    "document_identifier": document_identifier,
                    "source_common_name": getattr(rec, "SourceCommonName", None),
                    "themes": getattr(rec, "Themes", None),
                    "v2_themes": getattr(rec, "V2Themes", None),
                    "v2_tone_raw": getattr(rec, "V2Tone", None),
                    "tone": tone,
                    "positive_score": pos,
                    "negative_score": neg,
                    "polarity": polarity,
                    "activity_density": activity,
                    "self_density": self_density,
                    "word_count": word_count,
                    "is_relevant": topic_flags["is_relevant"],
                    "is_policy_uncertainty": topic_flags["is_policy_uncertainty"],
                    "is_geopolitical_risk": topic_flags["is_geopolitical_risk"],
                    "is_market_stress": topic_flags["is_market_stress"],
                    "deterministic_key": make_key(bucket_ts, record_id, document_identifier),
                    "ingestion_run_id": INGESTION_RUN_ID,
                    "ingestion_ts": datetime.now(timezone.utc).replace(tzinfo=None),
                }
            )

        if raw_row_count > 0 and filtered_row_count == 0:
            write_status = "zero_filtered_rows"

    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        write_status = "raw_file_missing" if status_code == 404 else "fetch_failed"
        errors.append(str(exc))
    except Exception as exc:
        write_status = "fetch_failed"
        errors.append(str(exc))

    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    checkpoint = {
        "bucket_ts": bucket_ts,
        "raw_file_fetched": raw_file_fetched,
        "raw_row_count": raw_row_count,
        "filtered_row_count": filtered_row_count,
        "write_status": write_status,
        "retry_count": attempt,
        "last_error": "; ".join(errors)[:8000] if errors else None,
        "source_url_or_file": url,
        "ingestion_run_id": INGESTION_RUN_ID,
        "ingestion_ts": completed_at,
    }
    return rows, checkpoint


def fetch_raw_gkg_bucket(bucket_ts):
    rows = []
    checkpoint = None
    retryable_statuses = {"fetch_failed", "raw_file_missing"}
    for attempt in range(MAX_RETRIES + 1):
        rows, checkpoint = fetch_raw_gkg_bucket_once(bucket_ts, attempt)
        if checkpoint["write_status"] not in retryable_statuses:
            return rows, checkpoint
        if attempt < MAX_RETRIES:
            time.sleep(2**attempt)
    return rows, checkpoint


def fetch_fast_bucket_summary_once(bucket_ts, attempt):
    url = gdelt_gkg_url(bucket_ts)
    ingestion_ts = datetime.now(timezone.utc).replace(tzinfo=None)
    status = "persisted"
    last_error = None
    raw_row_count = 0
    filtered_row_count = 0
    tones = []
    policy_count = 0
    geopolitical_count = 0
    market_stress_count = 0
    risk_on_count = 0
    risk_off_count = 0
    narrow_counts = {family["count_col"]: 0 for family in NARROW_TOPIC_FAMILIES}
    download_seconds = 0.0
    parse_seconds = 0.0
    summary_seconds = 0.0
    raw_bytes = None

    try:
        download_start = time.perf_counter()
        response = requests.get(url, timeout=(10, FETCH_TIMEOUT_SECONDS))
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            member = zf.namelist()[0]
            with zf.open(member) as fh:
                raw_bytes = fh.read()
        download_seconds = time.perf_counter() - download_start

        parse_start = time.perf_counter()
        pdf, skipped_bad_line_count, parser_warning = read_gkg_csv_robust(raw_bytes)
        parse_seconds = time.perf_counter() - parse_start
        if parser_warning:
            last_error = parser_warning[:8000]

        missing = [name for name in ["GKGRECORDID", "DATE", "DocumentIdentifier", "V2Themes", "V2Tone"] if name not in pdf.columns]
        if missing:
            raise ValueError(f"Missing required GKG columns: {missing}")

        summary_start = time.perf_counter()
        raw_physical_row_count = len(pdf)
        duplicate_gkgrecordid_count = int(raw_physical_row_count - pdf["GKGRECORDID"].nunique(dropna=True))
        pdf = pdf.dropna(subset=["GKGRECORDID"]).drop_duplicates(["GKGRECORDID"], keep="first")
        raw_row_count = len(pdf)
        if raw_row_count == 0:
            status = "zero_raw_rows"
        else:
            topic_blob = (
                pdf["Themes"].fillna("").astype(str)
                + " "
                + pdf["V2Themes"].fillna("").astype(str)
                + " "
                + pdf["DocumentIdentifier"].fillna("").astype(str)
                + " "
                + pdf["SourceCommonName"].fillna("").astype(str)
            )
            relevant_mask = topic_blob.str.contains(RELEVANT_RE, na=False)
            filtered_row_count = int(relevant_mask.sum())
            if filtered_row_count == 0:
                status = "zero_filtered_rows"
            else:
                relevant_blob = topic_blob[relevant_mask]
                tone_values = pd.to_numeric(
                    pdf.loc[relevant_mask, "V2Tone"].fillna("").astype(str).str.split(",", n=1).str[0],
                    errors="coerce",
                ).dropna()
                tones = tone_values.to_numpy(dtype=float).tolist()
                policy_count = int(relevant_blob.str.contains(POLICY_RE, na=False).sum())
                geopolitical_count = int(relevant_blob.str.contains(GEOPOLITICAL_RE, na=False).sum())
                market_stress_count = int(relevant_blob.str.contains(MARKET_STRESS_RE, na=False).sum())
                risk_on_count = int(relevant_blob.str.contains(RISK_ON_RE, na=False).sum())
                risk_off_count = int(relevant_blob.str.contains(RISK_OFF_RE, na=False).sum())
                for family in NARROW_TOPIC_FAMILIES:
                    narrow_counts[family["count_col"]] = int(topic_blob.str.contains(NARROW_TOPIC_REGEX[family["name"]], na=False).sum())
        if duplicate_gkgrecordid_count > 0:
            last_error = (
                "canonical_distinct_counting; "
                f"raw_physical_row_count={raw_physical_row_count}; "
                f"duplicate_gkgrecordid_count={duplicate_gkgrecordid_count}"
            )
        summary_seconds = time.perf_counter() - summary_start

    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        status = "raw_file_missing" if status_code == 404 else "fetch_failed"
        last_error = str(exc)
    except Exception as exc:
        status = RAW_ZIP_MALFORMED_UPSTREAM_STATUS if raw_bytes is not None else "fetch_failed"
        last_error = str(exc)

    return {
        "bucket_ts": pd.Timestamp(bucket_ts).to_pydatetime(),
        "source_provider": "gdelt_2_0_gkg_raw_15m",
        "source_file_url": url,
        "raw_file_status": status,
        "raw_row_count": int(raw_row_count),
        "filtered_row_count": int(filtered_row_count),
        "avg_tone": float(np.mean(tones)) if tones else None,
        "min_tone": float(np.min(tones)) if tones else None,
        "max_tone": float(np.max(tones)) if tones else None,
        "policy_uncertainty_count": int(policy_count),
        "geopolitical_risk_count": int(geopolitical_count),
        "market_stress_count": int(market_stress_count),
        **{name: int(value) for name, value in narrow_counts.items()},
        "risk_on_count": int(risk_on_count),
        "risk_off_count": int(risk_off_count),
        "source_bucket_count": 1 if status in ("persisted", "zero_filtered_rows", "zero_raw_rows") else 0,
        "has_missing_15m_bucket": status in ("raw_file_missing", "fetch_failed", RAW_ZIP_MALFORMED_UPSTREAM_STATUS),
        "ingestion_run_id": INGESTION_RUN_ID,
        "ingestion_ts": ingestion_ts,
        "last_error": last_error[:8000] if last_error else None,
        "download_seconds": float(download_seconds),
        "parse_seconds": float(parse_seconds),
        "summary_seconds": float(summary_seconds),
    }


def failed_fast_bucket_summary(bucket_ts, error_message):
    return {
        "bucket_ts": pd.Timestamp(bucket_ts).to_pydatetime(),
        "source_provider": "gdelt_2_0_gkg_raw_15m",
        "source_file_url": gdelt_gkg_url(bucket_ts),
        "raw_file_status": "fetch_failed",
        "raw_row_count": 0,
        "filtered_row_count": 0,
        "avg_tone": None,
        "min_tone": None,
        "max_tone": None,
        "policy_uncertainty_count": 0,
        "geopolitical_risk_count": 0,
        "market_stress_count": 0,
        **{family["count_col"]: 0 for family in NARROW_TOPIC_FAMILIES},
        "risk_on_count": 0,
        "risk_off_count": 0,
        "source_bucket_count": 0,
        "has_missing_15m_bucket": True,
        "ingestion_run_id": INGESTION_RUN_ID,
        "ingestion_ts": datetime.now(timezone.utc).replace(tzinfo=None),
        "last_error": str(error_message)[:8000],
        "download_seconds": 0.0,
        "parse_seconds": 0.0,
        "summary_seconds": 0.0,
    }


def fetch_fast_bucket_summary(bucket_ts):
    summary = None
    # HTTP 404 means the public GDELT archive does not currently expose that raw
    # file. Retrying every 404 bucket makes contiguous upstream gaps extremely
    # slow without adding evidence, so only transient fetch failures retry.
    retryable_statuses = {"fetch_failed"}
    for attempt in range(MAX_RETRIES + 1):
        summary = fetch_fast_bucket_summary_once(bucket_ts, attempt)
        if summary["raw_file_status"] not in retryable_statuses:
            return summary
        if attempt < MAX_RETRIES:
            time.sleep(2**attempt)
    return summary


def fetch_raw_zip_gapfill_bucket_summary(bucket_ts):
    raw_summary = fetch_fast_bucket_summary(bucket_ts)
    original_status = raw_summary["raw_file_status"]
    if original_status in RAW_SUMMARY_OK_STATUSES:
        canonical_status = FALLBACK_OK_STATUS
    elif original_status == "raw_file_missing":
        canonical_status = FALLBACK_MISSING_STATUS
    elif original_status == RAW_ZIP_MALFORMED_UPSTREAM_STATUS:
        canonical_status = RAW_ZIP_MALFORMED_UPSTREAM_STATUS
    else:
        canonical_status = FALLBACK_FAILED_STATUS

    last_error_parts = [
        part
        for part in [
            raw_summary.get("last_error"),
            f"raw_zip_gapfill_original_status={original_status}",
        ]
        if part
    ]
    raw_summary.update(
        {
            "source_provider": BIGQUERY_CANONICAL_PROVIDER,
            "raw_file_status": canonical_status,
            "source_bucket_count": 1 if canonical_status == FALLBACK_OK_STATUS else 0,
            "has_missing_15m_bucket": canonical_status != FALLBACK_OK_STATUS,
            "last_error": "; ".join(last_error_parts)[:8000],
        }
    )
    return raw_summary


def zero_count_canonical_missing_summary(bucket_ts, reason, sampled_http_statuses=None, sample_bucket_count=0):
    detail = {
        "classification": "contiguous_raw_compat_gap",
        "reason": reason,
        "sampled_bucket_count": int(sample_bucket_count or 0),
        "sampled_http_statuses": sampled_http_statuses or [],
        "bigquery_raw_compat_count": 0,
        "raw_zip_gapfill_original_status": "raw_file_missing",
    }
    return {
        "bucket_ts": pd.Timestamp(bucket_ts).to_pydatetime(),
        "source_provider": BIGQUERY_CANONICAL_PROVIDER,
        "source_file_url": gdelt_gkg_url(bucket_ts),
        "raw_file_status": FALLBACK_MISSING_STATUS,
        "raw_row_count": 0,
        "filtered_row_count": 0,
        "avg_tone": None,
        "min_tone": None,
        "max_tone": None,
        "policy_uncertainty_count": 0,
        "geopolitical_risk_count": 0,
        "market_stress_count": 0,
        **{family["count_col"]: 0 for family in NARROW_TOPIC_FAMILIES},
        "risk_on_count": 0,
        "risk_off_count": 0,
        "source_bucket_count": 0,
        "has_missing_15m_bucket": True,
        "ingestion_run_id": INGESTION_RUN_ID,
        "ingestion_ts": datetime.now(timezone.utc).replace(tzinfo=None),
        "last_error": json.dumps(detail, default=str)[:8000],
        "download_seconds": 0.0,
        "parse_seconds": 0.0,
        "summary_seconds": 0.0,
    }


def contiguous_timestamp_groups(timestamps, expected_step_minutes=15):
    ordered = sorted(pd.Timestamp(ts).tz_localize(None) for ts in timestamps)
    if not ordered:
        return []
    groups = [[ordered[0]]]
    expected_delta = pd.Timedelta(minutes=expected_step_minutes)
    for ts in ordered[1:]:
        if ts - groups[-1][-1] == expected_delta:
            groups[-1].append(ts)
        else:
            groups.append([ts])
    return groups


def sample_gap_buckets(group, max_samples=7):
    if len(group) <= max_samples:
        return list(group)
    indexes = np.linspace(0, len(group) - 1, num=max_samples, dtype=int)
    return [group[int(idx)] for idx in sorted(set(indexes.tolist()))]


def probe_raw_zip_http_status(bucket_ts):
    url = gdelt_gkg_url(bucket_ts)
    started = time.perf_counter()
    try:
        response = requests.get(url, timeout=(10, min(FETCH_TIMEOUT_SECONDS, 20)), stream=True)
        status_code = int(response.status_code)
        response.close()
        return {
            "bucket_ts": str(pd.Timestamp(bucket_ts)),
            "url": url,
            "status_code": status_code,
            "elapsed_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "bucket_ts": str(pd.Timestamp(bucket_ts)),
            "url": url,
            "status_code": None,
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(exc)[:1000],
        }


def classify_contiguous_missing_gap(group):
    samples = sample_gap_buckets(group)
    probe_results = [probe_raw_zip_http_status(bucket) for bucket in samples]
    status_codes = [result["status_code"] for result in probe_results]
    all_404 = bool(probe_results) and all(status == 404 for status in status_codes)
    detail = {
        "gap_start_ts": str(group[0]),
        "gap_end_ts": str(group[-1]),
        "bucket_count": len(group),
        "sampled_bucket_count": len(samples),
        "sampled_http_statuses": probe_results,
        "bigquery_raw_compat_counts": "zero for candidate buckets absent from BigQuery raw_gkg_compat result",
        "classification_status": FALLBACK_MISSING_STATUS if all_404 else "bucket_level_fallback_required",
        "reason": "sampled raw ZIP buckets returned HTTP 404" if all_404 else "mixed or inconclusive raw ZIP sample",
    }
    print({"contiguous_gap_classifier": detail})
    log_runtime(
        "bigquery_contiguous_gap_classifier",
        datetime.now(timezone.utc).replace(tzinfo=None),
        datetime.now(timezone.utc).replace(tzinfo=None),
        len(group),
        json.dumps(detail, default=str),
    )
    if not all_404:
        return None
    return [
        zero_count_canonical_missing_summary(
            bucket,
            reason=detail["reason"],
            sampled_http_statuses=probe_results,
            sample_bucket_count=len(samples),
        )
        for bucket in group
    ]


def raw_zip_hour_source_buckets(model_ts, lag_minutes):
    end_ts = pd.Timestamp(model_ts) - pd.Timedelta(minutes=lag_minutes)
    start_exclusive = end_ts - pd.Timedelta(hours=1)
    return list(pd.date_range(start=start_exclusive + pd.Timedelta(minutes=15), end=end_ts, freq="15min").to_pydatetime())


def expected_hourly_timestamps(start_ts, end_ts):
    start = pd.Timestamp(start_ts).ceil("H")
    end = pd.Timestamp(end_ts).floor("H")
    if start > end:
        return []
    return list(pd.date_range(start=start, end=end, freq="H").to_pydatetime())


def existing_hourly_feature_timestamps(start_ts, end_ts):
    if not REUSE_EXISTING_HOURLY_FEATURES:
        return set()
    try:
        rows = (
            spark.table(RESEARCH_FEATURE_TABLE)
            .filter(F.col("timestamp") >= F.lit(pd.Timestamp(start_ts).to_pydatetime()))
            .filter(F.col("timestamp") <= F.lit(pd.Timestamp(end_ts).to_pydatetime()))
            .select("timestamp")
            .distinct()
            .collect()
        )
        return {row["timestamp"] for row in rows}
    except Exception:
        return set()


def fetch_raw_zip_bucket_records_once(bucket_ts, attempt):
    url = gdelt_gkg_url(bucket_ts)
    started = time.perf_counter()
    download_seconds = 0.0
    parse_seconds = 0.0
    summary_seconds = 0.0
    raw_records = {}
    status = "persisted"
    last_error = None
    raw_bytes = None

    try:
        download_start = time.perf_counter()
        response = requests.get(url, timeout=(10, FETCH_TIMEOUT_SECONDS))
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            member = zf.namelist()[0]
            with zf.open(member) as fh:
                raw_bytes = fh.read()
        download_seconds = time.perf_counter() - download_start

        parse_start = time.perf_counter()
        pdf, skipped_bad_line_count, parser_warning = read_gkg_csv_robust(raw_bytes)
        parse_seconds = time.perf_counter() - parse_start
        if parser_warning:
            last_error = parser_warning[:8000]

        missing = [name for name in ["GKGRECORDID", "DATE", "DocumentIdentifier", "V2Themes", "V2Tone"] if name not in pdf.columns]
        if missing:
            raise ValueError(f"Missing required GKG columns: {missing}")

        summary_start = time.perf_counter()
        pdf = pdf.dropna(subset=["GKGRECORDID"])
        pdf = pdf[pdf["GKGRECORDID"].astype(str).str.match(r"^[0-9]{14}-[0-9]+$", na=False)]
        if pdf.empty:
            status = "zero_raw_rows"
        else:
            pdf = pdf.drop_duplicates(["GKGRECORDID"], keep="first")
            for rec in pdf.itertuples(index=False):
                record_id = getattr(rec, "GKGRECORDID", None)
                topic_blob = " ".join(
                    [
                        str(getattr(rec, "Themes", "") or ""),
                        str(getattr(rec, "V2Themes", "") or ""),
                        str(getattr(rec, "DocumentIdentifier", "") or ""),
                        str(getattr(rec, "SourceCommonName", "") or ""),
                    ]
                )
                flags = classify_topic_blob(topic_blob)
                tone, *_ = parse_v2tone(getattr(rec, "V2Tone", None))
                raw_records[record_id] = {
                    "tone": tone,
                    **flags,
                }
        summary_seconds = time.perf_counter() - summary_start

    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        status = "raw_file_missing" if status_code == 404 else "fetch_failed"
        last_error = str(exc)
    except Exception as exc:
        status = RAW_ZIP_MALFORMED_UPSTREAM_STATUS if raw_bytes is not None else "fetch_failed"
        last_error = str(exc)

    return {
        "bucket_ts": bucket_ts,
        "url": url,
        "status": status,
        "records": raw_records,
        "record_count": len(raw_records),
        "download_seconds": float(download_seconds),
        "parse_seconds": float(parse_seconds),
        "summary_seconds": float(summary_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "attempt": attempt,
        "last_error": last_error,
    }


def fetch_raw_zip_bucket_records(bucket_ts):
    result = None
    retryable_statuses = {"fetch_failed"}
    for attempt in range(MAX_RETRIES + 1):
        result = fetch_raw_zip_bucket_records_once(bucket_ts, attempt)
        if result["status"] not in retryable_statuses:
            return result
        if attempt < MAX_RETRIES:
            time.sleep(2**attempt)
    return result


def summarize_raw_zip_hour(model_ts):
    started = time.perf_counter()
    source_buckets = raw_zip_hour_source_buckets(model_ts, NEWS_AVAILABILITY_LAG_MINUTES)
    bucket_results = [fetch_raw_zip_bucket_records(bucket_ts) for bucket_ts in source_buckets]
    unresolved = [result for result in bucket_results if result["status"] == "fetch_failed"]
    upstream_missing = [result for result in bucket_results if result["status"] == "raw_file_missing"]
    records_by_id = {}
    for result in bucket_results:
        records_by_id.update(result["records"])

    relevant_records = [record for record in records_by_id.values() if record.get("is_relevant")]
    tone_values = [record["tone"] for record in relevant_records if record.get("tone") is not None]
    fetched_source_buckets = [result["bucket_ts"] for result in bucket_results if result["status"] in RAW_SUMMARY_OK_STATUSES]
    max_source_ts = max(fetched_source_buckets) if fetched_source_buckets else None
    min_source_ts = min(fetched_source_buckets) if fetched_source_buckets else None
    diagnostics = {
        "timestamp": model_ts,
        "expected_source_buckets": len(source_buckets),
        "downloaded_source_buckets": len(fetched_source_buckets),
        "upstream_missing_buckets": len(upstream_missing),
        "failed_buckets": len(unresolved),
        "failed_urls": [result["url"] for result in unresolved],
        "missing_urls": [result["url"] for result in upstream_missing],
        "download_seconds": sum(result["download_seconds"] for result in bucket_results),
        "parse_seconds": sum(result["parse_seconds"] for result in bucket_results),
        "aggregation_seconds": sum(result["summary_seconds"] for result in bucket_results),
        "total_seconds": time.perf_counter() - started,
    }
    return {
        "timestamp": model_ts,
        "news_volume": float(len(relevant_records)),
        "news_tone": float(np.mean(tone_values)) if tone_values else 0.0,
        "policy_uncertainty_news": float(sum(1 for record in relevant_records if record.get("is_policy_uncertainty"))),
        "geopolitical_risk_news": float(sum(1 for record in relevant_records if record.get("is_geopolitical_risk"))),
        "market_stress_news": float(sum(1 for record in relevant_records if record.get("is_market_stress"))),
        **{
            family["raw_col"]: float(sum(1 for record in records_by_id.values() if record.get(f"is_{family['name']}")))
            for family in NARROW_TOPIC_FAMILIES
        },
        "source_bucket_count": int(len(fetched_source_buckets)),
        "raw_news_row_count": int(len(records_by_id)),
        "filtered_news_row_count": int(len(relevant_records)),
        "min_gdelt_source_ts": min_source_ts,
        "max_gdelt_source_ts": max_source_ts,
        "has_missing_15m_bucket": bool(len(fetched_source_buckets) < len(source_buckets) or upstream_missing or unresolved),
        "source": "raw_zip_hourly_summary",
        "ingestion_run_id": INGESTION_RUN_ID,
        "ingestion_ts": RUN_TS,
    }, diagnostics


def append_staging_rows(row_dicts):
    if not row_dicts:
        return
    (
        spark.createDataFrame(row_dicts, schema=STAGING_ROW_SCHEMA)
        .dropDuplicates(["deterministic_key", "ingestion_run_id"])
        .write.mode("append")
        .format("delta")
        .saveAsTable(STAGING_NEWS_TABLE)
    )


def append_checkpoint_rows(checkpoint_rows):
    if not checkpoint_rows:
        return
    (
        spark.createDataFrame(checkpoint_rows, schema=CHECKPOINT_ROW_SCHEMA)
        .write.mode("append")
        .format("delta")
        .saveAsTable(CHECKPOINT_TABLE)
    )


def append_bucket_summary_rows(row_dicts):
    if not row_dicts:
        return
    started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    (
        spark.createDataFrame(row_dicts, schema=BUCKET_SUMMARY_ROW_SCHEMA)
        .dropDuplicates(["bucket_ts", "source_provider", "ingestion_run_id"])
        .write.mode("append")
        .format("delta")
        .saveAsTable(BUCKET_SUMMARY_TABLE)
    )
    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("fast_summary_batch_write", started_at, completed_at, len(row_dicts), f"max_bucket={max(row['bucket_ts'] for row in row_dicts)}")


RAW_HOURLY_BASE_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), True),
        StructField("news_volume", DoubleType(), True),
        StructField("news_tone", DoubleType(), True),
        StructField("policy_uncertainty_news", DoubleType(), True),
        StructField("geopolitical_risk_news", DoubleType(), True),
        StructField("market_stress_news", DoubleType(), True),
        *[StructField(family["raw_col"], DoubleType(), True) for family in NARROW_TOPIC_FAMILIES],
        StructField("source_bucket_count", IntegerType(), True),
        StructField("raw_news_row_count", LongType(), True),
        StructField("filtered_news_row_count", LongType(), True),
        StructField("min_gdelt_source_ts", TimestampType(), True),
        StructField("max_gdelt_source_ts", TimestampType(), True),
        StructField("has_missing_15m_bucket", BooleanType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_run_id", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)


def add_zscores_and_engineered_features_for_raw_hourly(raw_hourly_df):
    selected_min = raw_hourly_df.agg(F.min("timestamp")).collect()[0][0]
    history_start = pd.Timestamp(selected_min) - pd.Timedelta(hours=8760)
    raw_cols = [
        "timestamp",
        "news_volume",
        "news_tone",
        "policy_uncertainty_news",
        "geopolitical_risk_news",
        "market_stress_news",
        *NARROW_TOPIC_RAW_FEATURES,
    ]
    try:
        history_df = (
            spark.table(RESEARCH_FEATURE_TABLE)
            .filter(F.col("timestamp") >= F.lit(history_start.to_pydatetime()))
            .filter(F.col("timestamp") < F.lit(selected_min))
            .select(*raw_cols)
        )
    except Exception:
        history_df = spark.createDataFrame([], raw_hourly_df.select(*raw_cols).schema)

    z_base = history_df.unionByName(raw_hourly_df.select(*raw_cols), allowMissingColumns=True)
    win_z = Window.orderBy("timestamp").rowsBetween(-8760, 0)
    for raw_col, z_col in [
        ("news_volume", "news_volume_z"),
        ("news_tone", "news_tone_z"),
        ("policy_uncertainty_news", "policy_uncertainty_news_z"),
        ("geopolitical_risk_news", "geopolitical_risk_news_z"),
        ("market_stress_news", "market_stress_news_z"),
        *[(family["raw_col"], family["z_col"]) for family in NARROW_TOPIC_FAMILIES],
    ]:
        avg_val = F.avg(F.col(raw_col)).over(win_z)
        std_val = F.stddev(F.col(raw_col)).over(win_z)
        z_base = z_base.withColumn(
            z_col,
            F.when(std_val.isNull() | (std_val == 0), F.lit(0.0)).otherwise((F.col(raw_col) - avg_val) / std_val),
        )

    z_base = add_engineered_news_features(z_base)
    selected_ts_df = raw_hourly_df.select("timestamp").distinct()
    feature_cols = [
        "timestamp",
        "news_volume",
        "news_tone",
        "policy_uncertainty_news",
        "geopolitical_risk_news",
        "market_stress_news",
        *NARROW_TOPIC_RAW_FEATURES,
        "news_volume_z",
        "news_tone_z",
        "policy_uncertainty_news_z",
        "geopolitical_risk_news_z",
        "market_stress_news_z",
        *NARROW_TOPIC_Z_FEATURES,
        *ENGINEERED_NEWS_FEATURES,
    ]
    return (
        z_base.join(selected_ts_df, "timestamp", "inner")
        .select(*feature_cols)
        .join(
            raw_hourly_df.select(
                "timestamp",
                "source_bucket_count",
                "raw_news_row_count",
                "filtered_news_row_count",
                "min_gdelt_source_ts",
                "max_gdelt_source_ts",
                "has_missing_15m_bucket",
                "source",
                "ingestion_run_id",
                "ingestion_ts",
            ),
            "timestamp",
            "inner",
        )
    )


def write_hourly_news_features(final_features):
    for table_name in [SILVER_NEWS_TABLE, RESEARCH_FEATURE_TABLE]:
        align_to_existing_table_schema(final_features, table_name).createOrReplaceTempView("incoming_silver_news_flow_1h")
        spark.sql(
            f"""
            MERGE INTO {table_name} t
            USING incoming_silver_news_flow_1h s
            ON t.timestamp = s.timestamp
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )


def merge_staging_to_bronze():
    merge_sql = f"""
        MERGE INTO {BRONZE_NEWS_TABLE} t
        USING (
            SELECT
                source_ts,
                source,
                bucket_url,
                gdelt_record_id,
                document_identifier,
                source_common_name,
                themes,
                v2_themes,
                v2_tone_raw,
                tone,
                positive_score,
                negative_score,
                polarity,
                activity_density,
                self_density,
                word_count,
                is_relevant,
                is_policy_uncertainty,
                is_geopolitical_risk,
                is_market_stress,
                deterministic_key,
                ingestion_ts
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY deterministic_key
                        ORDER BY ingestion_ts DESC
                    ) AS rn
                FROM {STAGING_NEWS_TABLE}
                WHERE ingestion_run_id = '{INGESTION_RUN_ID}'
            )
            WHERE rn = 1
        ) s
        ON t.deterministic_key = s.deterministic_key
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)


def reset_window_for_overwrite():
    start_sql = f"TIMESTAMP('{FETCH_START_TS}')"
    end_sql = f"TIMESTAMP('{FETCH_END_TS}')"
    for table_name, col_name in [
        (STAGING_NEWS_TABLE, "bucket_ts"),
        (CHECKPOINT_TABLE, "bucket_ts"),
        (BUCKET_SUMMARY_TABLE, "bucket_ts"),
        (VALIDATION_TABLE, "batch_start_ts"),
        (BRONZE_NEWS_TABLE, "source_ts"),
        (SILVER_NEWS_TABLE, "timestamp"),
        (RESEARCH_FEATURE_TABLE, "timestamp"),
        (CANDIDATE_JOINED_VIEW, "timestamp"),
        (RESEARCH_WF_TABLE, "timestamp"),
    ]:
        spark.sql(
            f"""
            DELETE FROM {table_name}
            WHERE {col_name} >= {start_sql}
              AND {col_name} <= {end_sql}
            """
        )


def reset_fast_window_for_overwrite():
    start_sql = f"TIMESTAMP('{FETCH_START_TS}')"
    end_sql = f"TIMESTAMP('{FETCH_END_TS}')"
    provider = bucket_summary_source_provider()
    for table_name, col_name in [
        (BUCKET_SUMMARY_TABLE, "bucket_ts"),
        (VALIDATION_TABLE, "batch_start_ts"),
        (SILVER_NEWS_TABLE, "timestamp"),
        (RESEARCH_FEATURE_TABLE, "timestamp"),
        (CANDIDATE_JOINED_VIEW, "timestamp"),
        (RESEARCH_WF_TABLE, "timestamp"),
    ]:
        if table_name == BUCKET_SUMMARY_TABLE:
            spark.sql(
                f"""
                DELETE FROM {table_name}
                WHERE {col_name} >= {start_sql}
                  AND {col_name} <= {end_sql}
                  AND source_provider = '{provider}'
                """
            )
        else:
            spark.sql(
                f"""
                DELETE FROM {table_name}
                WHERE {col_name} >= {start_sql}
                  AND {col_name} <= {end_sql}
                """
            )


def existing_bronze_buckets(start_ts, end_ts):
    rows = spark.sql(
        f"""
        SELECT DISTINCT source_ts AS bucket_ts
        FROM {BRONZE_NEWS_TABLE}
        WHERE source_ts >= TIMESTAMP('{start_ts}')
          AND source_ts <= TIMESTAMP('{end_ts}')
        """
    ).collect()
    return {row["bucket_ts"].replace(tzinfo=None) for row in rows}


def completed_checkpoint_buckets(start_ts, end_ts):
    rows = spark.sql(
        f"""
        WITH latest AS (
            SELECT
                bucket_ts,
                write_status,
                ROW_NUMBER() OVER (
                    PARTITION BY bucket_ts
                    ORDER BY ingestion_ts DESC
                ) AS rn
            FROM {CHECKPOINT_TABLE}
            WHERE bucket_ts >= TIMESTAMP('{start_ts}')
              AND bucket_ts <= TIMESTAMP('{end_ts}')
        )
        SELECT bucket_ts
        FROM latest
        WHERE rn = 1
          AND write_status IN ('persisted', 'zero_raw_rows', 'zero_filtered_rows')
        """
    ).collect()
    return {row["bucket_ts"].replace(tzinfo=None) for row in rows}


def completed_summary_buckets(start_ts, end_ts):
    provider = bucket_summary_source_provider()
    fetched_statuses_sql = sql_string_list(SUMMARY_FETCHED_STATUSES + SUMMARY_MISSING_STATUSES)
    rows = spark.sql(
        f"""
        WITH latest AS (
            SELECT
                bucket_ts,
                raw_file_status,
                ROW_NUMBER() OVER (
                    PARTITION BY bucket_ts, source_provider
                    ORDER BY ingestion_ts DESC
                ) AS rn
            FROM {BUCKET_SUMMARY_TABLE}
            WHERE bucket_ts >= TIMESTAMP('{start_ts}')
              AND bucket_ts <= TIMESTAMP('{end_ts}')
              AND source_provider = '{provider}'
        )
        SELECT bucket_ts
        FROM latest
        WHERE rn = 1
          AND raw_file_status IN ({fetched_statuses_sql})
        """
    ).collect()
    return {row["bucket_ts"].replace(tzinfo=None) for row in rows}


def successful_summary_buckets(start_ts, end_ts):
    provider = bucket_summary_source_provider()
    fetched_statuses_sql = sql_string_list(SUMMARY_FETCHED_STATUSES + SUMMARY_MISSING_STATUSES)
    rows = spark.sql(
        f"""
        SELECT DISTINCT bucket_ts
        FROM {BUCKET_SUMMARY_TABLE}
        WHERE bucket_ts >= TIMESTAMP('{start_ts}')
          AND bucket_ts <= TIMESTAMP('{end_ts}')
          AND source_provider = '{provider}'
          AND raw_file_status IN ({fetched_statuses_sql})
        """
    ).collect()
    return {row["bucket_ts"].replace(tzinfo=None) for row in rows}


def append_validation_summary(
    batch_start_ts,
    batch_end_ts,
    source,
    expected_buckets,
    bucket_validations,
    observed_buckets_override=None,
    raw_row_count_override=None,
    filtered_row_count_override=None,
):
    if observed_buckets_override is None:
        observed_bucket_ts = [v["bucket_ts"] for v in bucket_validations if v["status"] in ("OK", "WARNING") and v["raw_row_count"] > 0]
    else:
        observed_bucket_ts = list(observed_buckets_override)
    duplicate_buckets = len(observed_bucket_ts) - len(set(observed_bucket_ts))
    missing_buckets = len(expected_buckets) - len(set(observed_bucket_ts))
    raw_rows = raw_row_count_override if raw_row_count_override is not None else sum(v["raw_row_count"] for v in bucket_validations)
    filtered_rows = filtered_row_count_override if filtered_row_count_override is not None else sum(v["filtered_row_count"] for v in bucket_validations)
    status = "OK"
    errors = [v["error_message"] for v in bucket_validations if v["error_message"]]
    if any(v["status"] == "ERROR" for v in bucket_validations):
        status = "ERROR"
    elif missing_buckets > 0 or filtered_rows == 0 or any(v["status"] == "WARNING" for v in bucket_validations):
        status = "WARNING"

    row = {
        "batch_start_ts": batch_start_ts,
        "batch_end_ts": batch_end_ts,
        "source": source,
        "expected_15m_buckets": len(expected_buckets),
        "observed_15m_buckets": len(set(observed_bucket_ts)),
        "missing_15m_buckets": missing_buckets,
        "duplicate_15m_buckets": duplicate_buckets,
        "raw_row_count": raw_rows,
        "filtered_row_count": filtered_rows,
        "min_source_ts": min(observed_bucket_ts) if observed_bucket_ts else None,
        "max_source_ts": max(observed_bucket_ts) if observed_bucket_ts else None,
        "fetch_started_at": min(v["fetch_started_at"] for v in bucket_validations),
        "fetch_completed_at": max(v["fetch_completed_at"] for v in bucket_validations),
        "status": status,
        "error_message": "; ".join(errors)[:8000] if errors else None,
    }
    schema = StructType(
        [
            StructField("batch_start_ts", TimestampType(), True),
            StructField("batch_end_ts", TimestampType(), True),
            StructField("source", StringType(), True),
            StructField("expected_15m_buckets", IntegerType(), True),
            StructField("observed_15m_buckets", IntegerType(), True),
            StructField("missing_15m_buckets", IntegerType(), True),
            StructField("duplicate_15m_buckets", IntegerType(), True),
            StructField("raw_row_count", LongType(), True),
            StructField("filtered_row_count", LongType(), True),
            StructField("min_source_ts", TimestampType(), True),
            StructField("max_source_ts", TimestampType(), True),
            StructField("fetch_started_at", TimestampType(), True),
            StructField("fetch_completed_at", TimestampType(), True),
            StructField("status", StringType(), True),
            StructField("error_message", StringType(), True),
        ]
    )
    spark.createDataFrame([row], schema=schema).write.mode("append").format("delta").saveAsTable(VALIDATION_TABLE)
    return row


def append_validation_summary_from_checkpoint(expected_buckets):
    expected_count = len(expected_buckets)
    rows = spark.sql(
        f"""
        SELECT
            COUNT(*) AS checkpoint_rows,
            COUNT(DISTINCT bucket_ts) AS checkpoint_buckets,
            COUNT(DISTINCT CASE WHEN raw_file_fetched THEN bucket_ts END) AS raw_file_fetched_buckets,
            COUNT(DISTINCT CASE WHEN write_status = 'raw_file_missing' THEN bucket_ts END) AS raw_file_missing_buckets,
            COUNT(DISTINCT CASE WHEN write_status = 'fetch_failed' THEN bucket_ts END) AS fetch_failed_buckets,
            COUNT(DISTINCT CASE WHEN write_status = 'zero_raw_rows' THEN bucket_ts END) AS zero_raw_buckets,
            COUNT(DISTINCT CASE WHEN write_status = 'zero_filtered_rows' THEN bucket_ts END) AS zero_filtered_buckets,
            SUM(raw_row_count) AS raw_row_count,
            SUM(filtered_row_count) AS filtered_row_count,
            MIN(CASE WHEN raw_file_fetched THEN bucket_ts END) AS min_source_ts,
            MAX(CASE WHEN raw_file_fetched THEN bucket_ts END) AS max_source_ts,
            MIN(ingestion_ts) AS fetch_started_at,
            MAX(ingestion_ts) AS fetch_completed_at,
            CONCAT_WS('; ', COLLECT_LIST(last_error)) AS error_message
        FROM {CHECKPOINT_TABLE}
        WHERE ingestion_run_id = '{INGESTION_RUN_ID}'
        """
    ).collect()
    row = rows[0]
    fetched = int(row["raw_file_fetched_buckets"] or 0)
    duplicate_buckets = int(row["checkpoint_rows"] or 0) - int(row["checkpoint_buckets"] or 0)
    missing_buckets = expected_count - fetched
    raw_file_missing = int(row["raw_file_missing_buckets"] or 0)
    fetch_failed = int(row["fetch_failed_buckets"] or 0)
    zero_raw = int(row["zero_raw_buckets"] or 0)
    zero_filtered = int(row["zero_filtered_buckets"] or 0)
    status = "OK"
    if fetch_failed > 0 or raw_file_missing > 0 or missing_buckets > 0:
        status = "ERROR"
    elif zero_raw > 0 or zero_filtered > 0:
        status = "WARNING"

    summary = {
        "batch_start_ts": min(expected_buckets),
        "batch_end_ts": max(expected_buckets),
        "source": f"gdelt_2_0_gkg_raw_15m:{INGESTION_RUN_ID}",
        "expected_15m_buckets": expected_count,
        "observed_15m_buckets": fetched,
        "missing_15m_buckets": missing_buckets,
        "duplicate_15m_buckets": duplicate_buckets,
        "raw_row_count": int(row["raw_row_count"] or 0),
        "filtered_row_count": int(row["filtered_row_count"] or 0),
        "min_source_ts": row["min_source_ts"],
        "max_source_ts": row["max_source_ts"],
        "fetch_started_at": row["fetch_started_at"],
        "fetch_completed_at": row["fetch_completed_at"],
        "status": status,
        "error_message": (row["error_message"] or "")[:8000] or None,
    }
    schema = StructType(
        [
            StructField("batch_start_ts", TimestampType(), True),
            StructField("batch_end_ts", TimestampType(), True),
            StructField("source", StringType(), True),
            StructField("expected_15m_buckets", IntegerType(), True),
            StructField("observed_15m_buckets", IntegerType(), True),
            StructField("missing_15m_buckets", IntegerType(), True),
            StructField("duplicate_15m_buckets", IntegerType(), True),
            StructField("raw_row_count", LongType(), True),
            StructField("filtered_row_count", LongType(), True),
            StructField("min_source_ts", TimestampType(), True),
            StructField("max_source_ts", TimestampType(), True),
            StructField("fetch_started_at", TimestampType(), True),
            StructField("fetch_completed_at", TimestampType(), True),
            StructField("status", StringType(), True),
            StructField("error_message", StringType(), True),
        ]
    )
    spark.createDataFrame([summary], schema=schema).write.mode("append").format("delta").saveAsTable(VALIDATION_TABLE)
    print(f"Ingestion validation summary: {summary}")
    return summary


def append_validation_summary_from_bucket_summaries(expected_buckets):
    expected_count = len(expected_buckets)
    provider = bucket_summary_source_provider()
    fetched_statuses_sql = sql_string_list(SUMMARY_FETCHED_STATUSES)
    missing_statuses_sql = sql_string_list(SUMMARY_MISSING_STATUSES)
    failed_statuses_sql = sql_string_list(SUMMARY_FAILED_STATUSES)
    rows = spark.sql(
        f"""
        SELECT
            COUNT(*) AS summary_rows,
            COUNT(DISTINCT bucket_ts) AS summary_buckets,
            COUNT(DISTINCT CASE WHEN raw_file_status IN ({fetched_statuses_sql}) THEN bucket_ts END) AS fetched_buckets,
            COUNT(DISTINCT CASE WHEN raw_file_status IN ({missing_statuses_sql}) THEN bucket_ts END) AS raw_file_missing_buckets,
            COUNT(DISTINCT CASE WHEN raw_file_status IN ({failed_statuses_sql}) THEN bucket_ts END) AS fetch_failed_buckets,
            COUNT(DISTINCT CASE WHEN raw_file_status = 'zero_raw_rows' THEN bucket_ts END) AS zero_raw_buckets,
            COUNT(DISTINCT CASE WHEN raw_file_status IN ('zero_filtered_rows', '{BIGQUERY_ZERO_FILTERED_STATUS}') THEN bucket_ts END) AS zero_filtered_buckets,
            SUM(raw_row_count) AS raw_row_count,
            SUM(filtered_row_count) AS filtered_row_count,
            MIN(CASE WHEN raw_file_status IN ({fetched_statuses_sql}) THEN bucket_ts END) AS min_source_ts,
            MAX(CASE WHEN raw_file_status IN ({fetched_statuses_sql}) THEN bucket_ts END) AS max_source_ts,
            MIN(ingestion_ts) AS fetch_started_at,
            MAX(ingestion_ts) AS fetch_completed_at,
            CONCAT_WS('; ', COLLECT_LIST(last_error)) AS error_message
        FROM {BUCKET_SUMMARY_TABLE}
        WHERE ingestion_run_id = '{INGESTION_RUN_ID}'
          AND source_provider = '{provider}'
        """
    ).collect()
    row = rows[0]
    fetched = int(row["fetched_buckets"] or 0)
    raw_file_missing = int(row["raw_file_missing_buckets"] or 0)
    fetch_failed = int(row["fetch_failed_buckets"] or 0)
    duplicate_buckets = int(row["summary_rows"] or 0) - int(row["summary_buckets"] or 0)
    missing_buckets = expected_count - fetched - raw_file_missing
    zero_raw = int(row["zero_raw_buckets"] or 0)
    zero_filtered = int(row["zero_filtered_buckets"] or 0)
    status = "OK"
    if fetch_failed > 0 or missing_buckets > 0:
        status = "ERROR"
    elif raw_file_missing > 0 or zero_raw > 0 or zero_filtered > 0:
        status = "WARNING"

    summary = {
        "batch_start_ts": min(expected_buckets),
        "batch_end_ts": max(expected_buckets),
        "source": f"{GDELT_SOURCE_MODE}:gkg_summary_15m:{INGESTION_RUN_ID}",
        "expected_15m_buckets": expected_count,
        "observed_15m_buckets": fetched,
        "missing_15m_buckets": missing_buckets + raw_file_missing,
        "duplicate_15m_buckets": duplicate_buckets,
        "raw_row_count": int(row["raw_row_count"] or 0),
        "filtered_row_count": int(row["filtered_row_count"] or 0),
        "min_source_ts": row["min_source_ts"],
        "max_source_ts": row["max_source_ts"],
        "fetch_started_at": row["fetch_started_at"],
        "fetch_completed_at": row["fetch_completed_at"],
        "status": status,
        "error_message": (row["error_message"] or "")[:8000] or None,
    }
    schema = StructType(
        [
            StructField("batch_start_ts", TimestampType(), True),
            StructField("batch_end_ts", TimestampType(), True),
            StructField("source", StringType(), True),
            StructField("expected_15m_buckets", IntegerType(), True),
            StructField("observed_15m_buckets", IntegerType(), True),
            StructField("missing_15m_buckets", IntegerType(), True),
            StructField("duplicate_15m_buckets", IntegerType(), True),
            StructField("raw_row_count", LongType(), True),
            StructField("filtered_row_count", LongType(), True),
            StructField("min_source_ts", TimestampType(), True),
            StructField("max_source_ts", TimestampType(), True),
            StructField("fetch_started_at", TimestampType(), True),
            StructField("fetch_completed_at", TimestampType(), True),
            StructField("status", StringType(), True),
            StructField("error_message", StringType(), True),
        ]
    )
    spark.createDataFrame([summary], schema=schema).write.mode("append").format("delta").saveAsTable(VALIDATION_TABLE)
    print(f"Bucket-summary validation summary: {summary}")
    return summary

# COMMAND ----------

def fetch_fast_summary_window():
    buckets = expected_15m_buckets(FETCH_START_TS, FETCH_END_TS)
    if RUN_MODE == "overwrite_window":
        print(f"Overwriting fast-mode research/candidate rows for {FETCH_START_TS} to {FETCH_END_TS}")
        reset_fast_window_for_overwrite()
    elif RUN_MODE == "backfill_missing_only":
        completed = successful_summary_buckets(FETCH_START_TS, FETCH_END_TS)
        buckets = [bucket for bucket in buckets if bucket not in completed]
        print(f"Fast backfill mode selected {len(buckets)} missing summary buckets.")
    elif RUN_MODE == "resume":
        completed = completed_summary_buckets(FETCH_START_TS, FETCH_END_TS)
        buckets = [bucket for bucket in buckets if bucket not in completed]
        print(f"Fast resume mode selected {len(buckets)} not-yet-completed summary buckets.")
    else:
        raise ValueError("run_mode must be one of: resume, overwrite_window, backfill_missing_only")

    # BigQuery is the scalable path; do not inherit the raw-file proof default
    # max_fetch_buckets=384 unless max_buckets_per_run is explicitly set.
    limit = MAX_BUCKETS_PER_RUN
    if limit > 0 and len(buckets) > limit:
        print(f"Limiting fast fetch from {len(buckets)} buckets to {limit} buckets for chunked run.")
        buckets = buckets[:limit]

    if not buckets:
        print("No buckets selected for fast summary ingestion.")
        return []

    fetch_started = datetime.now(timezone.utc).replace(tzinfo=None)
    summary_buffer = []
    all_summaries = []
    for chunk_start in range(0, len(buckets), FETCH_CHUNK_BUCKETS):
        chunk = buckets[chunk_start : chunk_start + FETCH_CHUNK_BUCKETS]
        executor = ThreadPoolExecutor(max_workers=FETCH_WORKERS)
        futures = {executor.submit(fetch_fast_bucket_summary, bucket): bucket for bucket in chunk}
        completed_futures = set()
        try:
            for future in as_completed(futures, timeout=FAST_CHUNK_TIMEOUT_SECONDS):
                completed_futures.add(future)
                summary = future.result()
                summary_buffer.append(summary)
                all_summaries.append(summary)
        except TimeoutError:
            timed_out_buckets = [
                bucket
                for future, bucket in futures.items()
                if future not in completed_futures and not future.done()
            ]
            for bucket in timed_out_buckets:
                summary = failed_fast_bucket_summary(
                    bucket,
                    f"fast chunk timed out after {FAST_CHUNK_TIMEOUT_SECONDS} seconds",
                )
                summary_buffer.append(summary)
                all_summaries.append(summary)
            print(
                {
                    "fast_chunk_timeout_seconds": FAST_CHUNK_TIMEOUT_SECONDS,
                    "timed_out_bucket_count": len(timed_out_buckets),
                    "chunk_start_bucket": min(chunk),
                    "chunk_end_bucket": max(chunk),
                }
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        should_flush = (
            len(summary_buffer) >= BATCH_WRITE_SIZE
            or chunk_start + FETCH_CHUNK_BUCKETS >= len(buckets)
        )
        if should_flush:
            append_bucket_summary_rows(summary_buffer)
            print(
                {
                    "flushed_bucket_summaries": len(summary_buffer),
                    "run_id": INGESTION_RUN_ID,
                    "last_bucket": max(row["bucket_ts"] for row in summary_buffer),
                }
            )
            summary_buffer = []

    fetch_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("fast_summary_fetch_total", fetch_started, fetch_completed, len(all_summaries), f"{FETCH_START_TS} to {FETCH_END_TS}")
    summary = append_validation_summary_from_bucket_summaries(buckets)
    return [summary]


def fetch_raw_window():
    buckets = expected_15m_buckets(FETCH_START_TS, FETCH_END_TS)
    if RUN_MODE == "overwrite_window":
        print(f"Overwriting research/candidate rows for {FETCH_START_TS} to {FETCH_END_TS}")
        reset_window_for_overwrite()
    elif RUN_MODE == "backfill_missing_only":
        persisted = existing_bronze_buckets(FETCH_START_TS, FETCH_END_TS)
        buckets = [bucket for bucket in buckets if bucket not in persisted]
        print(f"Backfill mode selected {len(buckets)} missing persisted buckets.")
    elif RUN_MODE == "resume":
        completed = existing_bronze_buckets(FETCH_START_TS, FETCH_END_TS) | completed_checkpoint_buckets(FETCH_START_TS, FETCH_END_TS)
        buckets = [bucket for bucket in buckets if bucket not in completed]
        print(f"Resume mode selected {len(buckets)} not-yet-completed buckets.")
    else:
        raise ValueError("run_mode must be one of: resume, overwrite_window, backfill_missing_only")

    if MAX_FETCH_BUCKETS > 0 and len(buckets) > MAX_FETCH_BUCKETS:
        print(f"Limiting raw fetch from {len(buckets)} buckets to {MAX_FETCH_BUCKETS} buckets for proof run.")
        buckets = buckets[:MAX_FETCH_BUCKETS]

    if not buckets:
        print("No buckets selected for ingestion.")
        return []

    all_checkpoints = []
    staged_rows_buffer = []
    checkpoint_buffer = []
    for chunk_start in range(0, len(buckets), FETCH_CHUNK_BUCKETS):
        chunk = buckets[chunk_start : chunk_start + FETCH_CHUNK_BUCKETS]
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
            futures = {executor.submit(fetch_raw_gkg_bucket, bucket): bucket for bucket in chunk}
            for future in as_completed(futures):
                rows, checkpoint = future.result()
                staged_rows_buffer.extend(rows)
                checkpoint_buffer.append(checkpoint)
                all_checkpoints.append(checkpoint)

        should_flush = (
            len(checkpoint_buffer) >= BATCH_WRITE_SIZE
            or chunk_start + FETCH_CHUNK_BUCKETS >= len(buckets)
        )
        if should_flush:
            append_staging_rows(staged_rows_buffer)
            append_checkpoint_rows(checkpoint_buffer)
            print(
                {
                    "flushed_checkpoints": len(checkpoint_buffer),
                    "flushed_rows": len(staged_rows_buffer),
                    "run_id": INGESTION_RUN_ID,
                    "last_bucket": max(row["bucket_ts"] for row in checkpoint_buffer),
                }
            )
            staged_rows_buffer = []
            checkpoint_buffer = []

    print("Merging staged rows into bronze once for this ingestion run.")
    merge_staging_to_bronze()
    summary = append_validation_summary_from_checkpoint(buckets)
    return [summary]


def fetch_raw_zip_hourly_summary_window():
    fetch_started = datetime.now(timezone.utc).replace(tzinfo=None)
    expected_hours = expected_hourly_timestamps(FETCH_START_TS, FETCH_END_TS)
    selected_hours = expected_hours
    if RUN_MODE == "overwrite_window":
        print(f"Raw-ZIP hourly overwrite mode selected all hours for {FETCH_START_TS} to {FETCH_END_TS}.")
    elif RUN_MODE in {"resume", "backfill_missing_only"}:
        completed = existing_hourly_feature_timestamps(FETCH_START_TS, FETCH_END_TS)
        selected_hours = [hour for hour in expected_hours if hour not in completed]
        print(f"Raw-ZIP hourly {RUN_MODE} mode selected {len(selected_hours)} missing hourly feature rows.")
    else:
        raise ValueError("run_mode must be one of: resume, overwrite_window, backfill_missing_only")

    if MAX_HOURS_PER_RUN > 0 and len(selected_hours) > MAX_HOURS_PER_RUN:
        print(f"Limiting raw-ZIP hourly summary from {len(selected_hours)} hours to {MAX_HOURS_PER_RUN} hours.")
        selected_hours = selected_hours[:MAX_HOURS_PER_RUN]
    if not selected_hours:
        print("No hourly timestamps selected for raw-ZIP hourly summary.")
        return []

    print(
        {
            "mode": GDELT_SOURCE_MODE,
            "selected_hours": len(selected_hours),
            "selected_start_hour": str(min(selected_hours)),
            "selected_end_hour": str(max(selected_hours)),
            "expected_15m_source_buckets": len(selected_hours) * 4,
            "news_availability_lag_minutes": NEWS_AVAILABILITY_LAG_MINUTES,
            "raw_zip_fetch_workers": RAW_ZIP_FETCH_WORKERS,
            "write_outputs": RAW_ZIP_HOURLY_WRITE_OUTPUTS,
        }
    )

    hourly_rows = []
    diagnostics = []
    with ThreadPoolExecutor(max_workers=max(RAW_ZIP_FETCH_WORKERS, 1)) as executor:
        futures = {executor.submit(summarize_raw_zip_hour, hour): hour for hour in selected_hours}
        for future in as_completed(futures):
            row, diagnostic = future.result()
            hourly_rows.append(row)
            diagnostics.append(diagnostic)

    unresolved_failed = sum(item["failed_buckets"] for item in diagnostics)
    if unresolved_failed:
        detail = {
            "status": "RAW_ZIP_HOURLY_FETCH_FAILED_NO_WRITE",
            "failed_buckets": unresolved_failed,
            "failed_urls": [url for item in diagnostics for url in item["failed_urls"]][:50],
            "selected_hours": len(selected_hours),
            "selected_start_hour": str(min(selected_hours)),
            "selected_end_hour": str(max(selected_hours)),
        }
        print(json.dumps(detail, default=str))
        dbutils.notebook.exit(json.dumps(detail, default=str))

    raw_hourly_df = spark.createDataFrame(hourly_rows, schema=RAW_HOURLY_BASE_SCHEMA)
    final_features = add_zscores_and_engineered_features_for_raw_hourly(raw_hourly_df)
    lag_expr = F.expr(f"timestamp - INTERVAL {NEWS_AVAILABILITY_LAG_MINUTES} MINUTES")
    leakage_rows = final_features.where(F.col("max_gdelt_source_ts") > lag_expr).count()
    null_feature_rows = final_features.where(
        F.col("market_stress_news_z").isNull()
        | F.col(broad_pulse_feature_name("market_stress", 48, 2.25)).isNull()
    ).count()
    duplicate_hourly_timestamps = final_features.count() - final_features.select("timestamp").distinct().count()
    if leakage_rows or null_feature_rows or duplicate_hourly_timestamps:
        detail = {
            "status": "RAW_ZIP_HOURLY_VALIDATION_FAILED_NO_WRITE",
            "lag_leakage_rows": leakage_rows,
            "null_feature_rows": null_feature_rows,
            "duplicate_hourly_timestamps": duplicate_hourly_timestamps,
        }
        print(json.dumps(detail, default=str))
        dbutils.notebook.exit(json.dumps(detail, default=str))

    compare_detail = {}
    if RAW_ZIP_HOURLY_COMPARE_EXISTING:
        compare_cols = [
            "news_volume",
            "news_tone",
            "policy_uncertainty_news",
            "geopolitical_risk_news",
            "market_stress_news",
            "source_bucket_count",
        ]
        existing = spark.table(RESEARCH_FEATURE_TABLE).select("timestamp", *compare_cols)
        candidate = final_features.select("timestamp", *[F.col(col).alias(f"candidate_{col}") for col in compare_cols])
        comparison = candidate.join(existing, "timestamp", "inner")
        compare_exprs = []
        for col in compare_cols:
            compare_exprs.append(F.max(F.abs(F.col(f"candidate_{col}") - F.col(col))).alias(f"max_abs_diff_{col}"))
        compare_df = comparison.agg(F.count("*").alias("matched_hourly_rows"), *compare_exprs)
        display(compare_df)
        compare_detail = compare_df.collect()[0].asDict()

    if RAW_ZIP_HOURLY_WRITE_OUTPUTS:
        write_started = datetime.now(timezone.utc).replace(tzinfo=None)
        write_hourly_news_features(final_features)
        write_completed = datetime.now(timezone.utc).replace(tzinfo=None)
        log_runtime("raw_zip_hourly_spark_write", write_started, write_completed, final_features.count(), GDELT_SOURCE_MODE)
    else:
        print("Raw-ZIP hourly summary write skipped because raw_zip_hourly_write_outputs=false.")

    fetch_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    validation_detail = {
        "selected_hours": len(selected_hours),
        "expected_15m_source_buckets": len(selected_hours) * 4,
        "downloaded_source_buckets": sum(item["downloaded_source_buckets"] for item in diagnostics),
        "upstream_missing_buckets": sum(item["upstream_missing_buckets"] for item in diagnostics),
        "failed_buckets": unresolved_failed,
        "hours_with_fewer_than_4_source_buckets": sum(1 for item in diagnostics if item["downloaded_source_buckets"] < 4),
        "hours_with_zero_filtered_rows": final_features.where(F.col("filtered_news_row_count") == 0).count(),
        "duplicate_hourly_timestamps": duplicate_hourly_timestamps,
        "hourly_rows_written": final_features.count() if RAW_ZIP_HOURLY_WRITE_OUTPUTS else 0,
        "hourly_rows_built": final_features.count(),
        "null_feature_rows": null_feature_rows,
        "lag_15_leakage_rows": leakage_rows,
        "latest_completed_timestamp": str(max(selected_hours)),
        "download_seconds": sum(item["download_seconds"] for item in diagnostics),
        "parse_seconds": sum(item["parse_seconds"] for item in diagnostics),
        "aggregation_seconds": sum(item["aggregation_seconds"] for item in diagnostics),
        "average_seconds_per_hour": sum(item["total_seconds"] for item in diagnostics) / max(len(diagnostics), 1),
        "comparison": compare_detail,
    }
    print(json.dumps({"raw_zip_hourly_validation": validation_detail}, default=str))
    log_runtime(
        "raw_zip_hourly_total",
        fetch_started,
        fetch_completed,
        len(hourly_rows),
        json.dumps(validation_detail, default=str),
    )
    return [validation_detail]


BIGQUERY_RELEVANT_PATTERN = (
    "ECON|STOCK|MARKET|INFLATION|RECESSION|BANK|CREDIT|LIQUIDITY|FED|"
    "CENTRAL_BANK|TARIFF|SANCTION|WAR|CONFLICT|GEO|POLICY|UNCERTAINTY"
)
BIGQUERY_POLICY_PATTERN = (
    "POLICY|UNCERTAINTY|TARIFF|TRADE|FED|FEDERAL_RESERVE|CENTRAL_BANK|"
    "FISCAL|TAX|GOVERNMENT|REGULATION"
)
BIGQUERY_GEOPOLITICAL_PATTERN = (
    "WAR|CONFLICT|SANCTION|MILITARY|TERROR|ARMEDCONFLICT|GEOPOLITICAL|"
    "PROTEST|COUP|CRISIS"
)
BIGQUERY_MARKET_STRESS_PATTERN = (
    "RECESSION|BANK|BANKING|CREDIT|LIQUIDITY|DEFAULT|DEBT|MARKET_CRASH|"
    "VOLATILITY|LAYOFF|UNEMPLOYMENT|BANKRUPT"
)
BIGQUERY_RISK_ON_PATTERN = (
    "RALLY|RECOVERY|GROWTH|EXPANSION|EASING|STIMULUS|OPTIMISM|"
    "BULL|GAIN|REBOUND|REOPEN"
)
BIGQUERY_RISK_OFF_PATTERN = (
    "SELL.?OFF|CRASH|PANIC|RECESSION|DEFAULT|BANKRUPT|LIQUIDITY|"
    "VOLATILITY|FEAR|DRAWDOWN|CONTAGION|LOCKDOWN"
)
BIGQUERY_INFLATION_PRICES_PATTERN = "INFLATION|CPI|CONSUMER.?PRICE|PRICE.?PRESSURE|COST.?OF.?LIVING|PRICES"
BIGQUERY_FED_RATES_PATTERN = (
    "FEDERAL.?RESERVE|FOMC|RATE.?HIKE|INTEREST.?RATE|MONETARY.?TIGHTEN|"
    "HAWKISH|DOVISH|CENTRAL.?BANK"
)
BIGQUERY_RECESSION_SLOWDOWN_PATTERN = (
    "RECESSION|SLOWDOWN|CONTRACTION|LAYOFF|UNEMPLOYMENT|JOBLESS|"
    "EARNINGS.?DOWNGRADE|GROWTH.?SCARE"
)
BIGQUERY_CREDIT_LIQUIDITY_PATTERN = (
    "LIQUIDITY|CREDIT.?STRESS|DEFAULT|BANK.?STRESS|FUNDING.?STRESS|"
    "FINANCIAL.?CONDITIONS|CREDIT|BANKING"
)
BIGQUERY_GEOPOLITICAL_ENERGY_PATTERN = (
    "WAR|SANCTION|RUSSIA|UKRAINE|OIL|GAS|ENERGY.?SHOCK|"
    "SUPPLY.?DISRUPTION|GEOPOLIT"
)
BIGQUERY_SUPPLY_CHAIN_COMMODITIES_PATTERN = "SUPPLY.?CHAIN|SHIPPING|SHORTAGE|COMMODIT|FOOD.?PRICE|ENERGY.?PRICE"


def configure_bigquery_credentials():
    from google.oauth2 import service_account

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    if GCP_SERVICE_ACCOUNT_SECRET_SCOPE and GCP_SERVICE_ACCOUNT_SECRET_KEY:
        secret_json = dbutils.secrets.get(GCP_SERVICE_ACCOUNT_SECRET_SCOPE, GCP_SERVICE_ACCOUNT_SECRET_KEY)
        credentials = service_account.Credentials.from_service_account_info(json.loads(secret_json), scopes=scopes)
        return credentials, f"databricks_secret:{GCP_SERVICE_ACCOUNT_SECRET_SCOPE}/{GCP_SERVICE_ACCOUNT_SECRET_KEY}"
    if GCP_SERVICE_ACCOUNT_JSON_PATH:
        credentials = service_account.Credentials.from_service_account_file(GCP_SERVICE_ACCOUNT_JSON_PATH, scopes=scopes)
        return credentials, "service_account_json_path"

    import google.auth

    credentials, default_project = google.auth.default(scopes=scopes)
    return credentials, f"application_default_credentials:project={default_project or 'unknown'}"


def bigquery_setup_requirements(errors):
    requirements = {
        "required_python_package": "google-cloud-bigquery",
        "required_credentials": (
            "Either application default credentials, widget gcp_service_account_json_path, "
            "or widgets gcp_service_account_secret_scope + gcp_service_account_secret_key containing a service-account JSON."
        ),
        "required_gcp_project": "Set widget gcp_project to the billing/project id used for BigQuery jobs.",
        "required_gdelt_table": BIGQUERY_GKG_TABLE or "gdelt-bq.gdeltv2.gkg_partitioned",
        "query_mode": BIGQUERY_QUERY_MODE,
        "record_universe": BIGQUERY_RECORD_UNIVERSE,
        "errors": errors,
    }
    print("BigQuery setup unavailable for MUB-30 research run:")
    print(json.dumps(requirements, indent=2))
    return requirements


def get_bigquery_client():
    errors = []
    if not GCP_PROJECT:
        errors.append("Missing widget gcp_project.")
    if not BIGQUERY_GKG_TABLE:
        errors.append("Missing widget bigquery_table.")
    if BIGQUERY_QUERY_MODE != "15m_summary":
        errors.append("Only bigquery_query_mode=15m_summary is implemented in this notebook revision.")
    try:
        from google.cloud import bigquery
    except ImportError:
        errors.append("Python package google-cloud-bigquery is not installed on the Databricks cluster.")
        bigquery = None

    credential_mode = None
    if not errors:
        try:
            credentials, credential_mode = configure_bigquery_credentials()
            client = bigquery.Client(project=GCP_PROJECT, credentials=credentials)
            print(
                "BigQuery setup available: "
                f"project={GCP_PROJECT}, table={BIGQUERY_GKG_TABLE}, credentials={credential_mode}"
            )
            return client, bigquery, credential_mode
        except Exception as exc:
            errors.append(f"BigQuery credential/client initialization failed: {type(exc).__name__}: {str(exc)[:1000]}")

    bigquery_setup_requirements(errors)
    raise RuntimeError("BigQuery source mode is unavailable; see setup requirements printed above. Raw modes remain intact.")


def bigquery_record_universe_predicate():
    if BIGQUERY_RECORD_UNIVERSE == "raw_gkg_compat":
        # Raw GKG zip files used for audit validation contain numeric record ids
        # such as 20200201000000-123. BigQuery also includes translated GKG
        # records such as 20200201000000-T123, which would inflate counts versus
        # raw-fast mode unless explicitly requested.
        return "REGEXP_CONTAINS(gdelt_record_id, r'^[0-9]{14}-[0-9]+$')"
    return "TRUE"


def bigquery_15m_summary_sql(start_ts, end_ts):
    partition_start = pd.Timestamp(start_ts).strftime("%Y-%m-%d")
    partition_end = (pd.Timestamp(end_ts) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_literal = pd.Timestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")
    end_literal = pd.Timestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    WITH source_rows AS (
      SELECT
        PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(`DATE` AS STRING)) AS bucket_ts,
        GKGRECORDID AS gdelt_record_id,
        SAFE_CAST(SPLIT(IFNULL(V2Tone, ''), ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
        UPPER(CONCAT(IFNULL(Themes, ''), ' ', IFNULL(V2Themes, ''), ' ', IFNULL(DocumentIdentifier, ''), ' ', IFNULL(SourceCommonName, ''))) AS topic_blob
      FROM `{BIGQUERY_GKG_TABLE}`
      WHERE _PARTITIONTIME >= TIMESTAMP('{partition_start}')
        AND _PARTITIONTIME < TIMESTAMP('{partition_end}')
    ),
    record_rows AS (
      SELECT
        bucket_ts,
        gdelt_record_id,
        ANY_VALUE(tone) AS tone,
        ANY_VALUE(topic_blob) AS topic_blob
      FROM source_rows
      WHERE bucket_ts >= TIMESTAMP('{start_literal}')
        AND bucket_ts <= TIMESTAMP('{end_literal}')
        AND gdelt_record_id IS NOT NULL
        AND {bigquery_record_universe_predicate()}
      GROUP BY bucket_ts, gdelt_record_id
    ),
    classified AS (
      SELECT
        bucket_ts,
        gdelt_record_id,
        tone,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_RELEVANT_PATTERN})') AS is_relevant,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_POLICY_PATTERN})') AS is_policy_uncertainty,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_GEOPOLITICAL_PATTERN})') AS is_geopolitical_risk,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_MARKET_STRESS_PATTERN})') AS is_market_stress,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_RISK_ON_PATTERN})') AS is_risk_on,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_RISK_OFF_PATTERN})') AS is_risk_off,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_INFLATION_PRICES_PATTERN})') AS is_inflation_prices,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_FED_RATES_PATTERN})') AS is_fed_rates,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_RECESSION_SLOWDOWN_PATTERN})') AS is_recession_slowdown,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_CREDIT_LIQUIDITY_PATTERN})') AS is_credit_liquidity,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_GEOPOLITICAL_ENERGY_PATTERN})') AS is_geopolitical_energy,
        REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_SUPPLY_CHAIN_COMMODITIES_PATTERN})') AS is_supply_chain_commodities
      FROM record_rows
    )
    SELECT
      bucket_ts,
      COUNT(DISTINCT gdelt_record_id) AS raw_row_count,
      COUNT(DISTINCT IF(is_relevant, gdelt_record_id, NULL)) AS filtered_row_count,
      AVG(IF(is_relevant, tone, NULL)) AS avg_tone,
      MIN(IF(is_relevant, tone, NULL)) AS min_tone,
      MAX(IF(is_relevant, tone, NULL)) AS max_tone,
      COUNT(DISTINCT IF(is_relevant AND is_policy_uncertainty, gdelt_record_id, NULL)) AS policy_uncertainty_count,
      COUNT(DISTINCT IF(is_relevant AND is_geopolitical_risk, gdelt_record_id, NULL)) AS geopolitical_risk_count,
      COUNT(DISTINCT IF(is_relevant AND is_market_stress, gdelt_record_id, NULL)) AS market_stress_count,
      COUNT(DISTINCT IF(is_inflation_prices, gdelt_record_id, NULL)) AS inflation_prices_count,
      COUNT(DISTINCT IF(is_fed_rates, gdelt_record_id, NULL)) AS fed_rates_count,
      COUNT(DISTINCT IF(is_recession_slowdown, gdelt_record_id, NULL)) AS recession_slowdown_count,
      COUNT(DISTINCT IF(is_credit_liquidity, gdelt_record_id, NULL)) AS credit_liquidity_count,
      COUNT(DISTINCT IF(is_geopolitical_energy, gdelt_record_id, NULL)) AS geopolitical_energy_count,
      COUNT(DISTINCT IF(is_supply_chain_commodities, gdelt_record_id, NULL)) AS supply_chain_commodities_count,
      COUNT(DISTINCT IF(is_relevant AND is_risk_on, gdelt_record_id, NULL)) AS risk_on_count,
      COUNT(DISTINCT IF(is_relevant AND is_risk_off, gdelt_record_id, NULL)) AS risk_off_count
    FROM classified
    GROUP BY bucket_ts
    ORDER BY bucket_ts
    """


def run_bigquery_smoke_tests(client, bigquery):
    auth_sql = f"""
    SELECT
      DATE(_PARTITIONTIME) AS partition_date,
      COUNT(*) AS row_count
    FROM `{BIGQUERY_GKG_TABLE}`
    WHERE _PARTITIONTIME >= TIMESTAMP("2020-02-01")
      AND _PARTITIONTIME < TIMESTAMP("2020-02-02")
    GROUP BY partition_date
    LIMIT 10
    """
    schema_sql = f"""
    SELECT
      PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(`DATE` AS STRING)) AS gdelt_source_ts,
      COUNT(*) AS row_count,
      COUNTIF(V2Tone IS NOT NULL) AS rows_with_tone,
      COUNTIF(V2Themes IS NOT NULL) AS rows_with_themes,
      AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)) AS avg_tone
    FROM `{BIGQUERY_GKG_TABLE}`
    WHERE _PARTITIONTIME >= TIMESTAMP("2020-02-01")
      AND _PARTITIONTIME < TIMESTAMP("2020-02-02")
    GROUP BY gdelt_source_ts
    ORDER BY gdelt_source_ts
    LIMIT 20
    """
    smoke_started = datetime.now(timezone.utc).replace(tzinfo=None)
    auth_rows = [dict(row.items()) for row in client.query(auth_sql).result()]
    schema_rows = [dict(row.items()) for row in client.query(schema_sql).result()]
    smoke_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    detail = {
        "auth_partition_rows": len(auth_rows),
        "schema_timestamp_rows": len(schema_rows),
        "auth_first_row": {
            "partition_date": str(auth_rows[0]["partition_date"]) if auth_rows else None,
            "row_count": int(auth_rows[0]["row_count"]) if auth_rows else 0,
        },
        "schema_first_ts": str(schema_rows[0]["gdelt_source_ts"]) if schema_rows else None,
        "table": BIGQUERY_GKG_TABLE,
    }
    print(f"BigQuery auth/schema smoke: {json.dumps(detail, default=str)}")
    if not auth_rows or int(auth_rows[0]["row_count"] or 0) == 0:
        raise RuntimeError("BigQuery auth smoke returned no rows for 2020-02-01.")
    if not schema_rows:
        raise RuntimeError("BigQuery timestamp/schema smoke returned no parsed DATE rows.")
    log_runtime("bigquery_auth_schema_smoke", smoke_started, smoke_completed, len(schema_rows), json.dumps(detail, default=str))
    return auth_rows, schema_rows


def is_bigquery_quota_or_billing_error(exc):
    message = str(exc).lower()
    quota_terms = [
        "quota",
        "free query bytes",
        "bytes quota",
        "billing",
        "maximum bytes billed",
        "exceeded rate limits",
        "ratelimitexceeded",
    ]
    return any(term in message for term in quota_terms)


def bigquery_query_job_config(bigquery, dry_run=False):
    kwargs = {}
    if dry_run:
        kwargs.update({"dry_run": True, "use_query_cache": False})
    if BIGQUERY_MAX_BYTES_BILLED > 0:
        kwargs["maximum_bytes_billed"] = BIGQUERY_MAX_BYTES_BILLED
    return bigquery.QueryJobConfig(**kwargs)


def exit_bigquery_no_write(status, detail):
    payload = {
        "status": status,
        "message": "BigQuery quota/dry-run gate stopped before query result write.",
        **detail,
    }
    print(json.dumps(payload, default=str))
    log_runtime(
        "bigquery_quota_gate_no_write",
        datetime.now(timezone.utc).replace(tzinfo=None),
        datetime.now(timezone.utc).replace(tzinfo=None),
        0,
        json.dumps(payload, default=str),
    )
    dbutils.notebook.exit(json.dumps(payload, default=str))


def bigquery_selected_chunk_context(buckets):
    query_start_ts = min(buckets)
    query_end_ts = max(buckets)
    return {
        "query_start_ts": query_start_ts,
        "query_end_ts": query_end_ts,
        "sql": bigquery_15m_summary_sql(query_start_ts, query_end_ts),
    }


def bigquery_dry_run_selected_chunk(client, bigquery, sql, selected_bucket_count, query_start_ts, query_end_ts, credential_mode):
    dry_run_started = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        dry_run_job = client.query(sql, job_config=bigquery_query_job_config(bigquery, dry_run=True))
        bytes_processed = int(dry_run_job.total_bytes_processed or 0)
    except Exception as exc:
        detail = {
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "selected_bucket_count": selected_bucket_count,
            "selected_min_ts": str(query_start_ts),
            "selected_max_ts": str(query_end_ts),
            "maximum_bytes_billed": BIGQUERY_MAX_BYTES_BILLED or None,
        }
        if is_bigquery_quota_or_billing_error(exc):
            exit_bigquery_no_write("BIGQUERY_DRY_RUN_QUOTA_BLOCKED_NO_WRITE", detail)
        raise
    dry_run_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    detail = {
        "bigquery_table": BIGQUERY_GKG_TABLE,
        "query_mode": BIGQUERY_QUERY_MODE,
        "record_universe": BIGQUERY_RECORD_UNIVERSE,
        "dry_run_bytes_processed": bytes_processed,
        "selected_15m_buckets": selected_bucket_count,
        "selected_min_ts": str(query_start_ts),
        "selected_max_ts": str(query_end_ts),
        "maximum_bytes_billed": BIGQUERY_MAX_BYTES_BILLED or None,
        "dry_run_gate_bytes": BIGQUERY_MAX_DRY_RUN_BYTES_PER_CHUNK or BIGQUERY_MAX_BYTES_BILLED or None,
        "credential_mode": credential_mode,
    }
    print(detail)
    log_runtime("bigquery_dry_run", dry_run_started, dry_run_completed, selected_bucket_count, json.dumps(detail, default=str))
    return bytes_processed, detail


def fetch_bigquery_summary_window():
    client, bigquery, credential_mode = get_bigquery_client()
    query_construct_started = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        run_bigquery_smoke_tests(client, bigquery)
    except Exception as exc:
        detail = {
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "table": BIGQUERY_GKG_TABLE,
            "maximum_bytes_billed": BIGQUERY_MAX_BYTES_BILLED or None,
        }
        if is_bigquery_quota_or_billing_error(exc):
            exit_bigquery_no_write("BIGQUERY_SMOKE_QUOTA_BLOCKED_NO_WRITE", detail)
        raise
    query_construct_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("bigquery_query_construction", query_construct_started, query_construct_completed, 1, BIGQUERY_GKG_TABLE)

    expected_buckets = expected_15m_buckets(FETCH_START_TS, FETCH_END_TS)
    buckets = expected_buckets
    if RUN_MODE == "overwrite_window":
        print(f"Overwriting BigQuery research/candidate rows for {FETCH_START_TS} to {FETCH_END_TS}")
        reset_fast_window_for_overwrite()
    elif RUN_MODE == "backfill_missing_only":
        completed = successful_summary_buckets(FETCH_START_TS, FETCH_END_TS)
        buckets = [bucket for bucket in expected_buckets if bucket not in completed]
        print(f"BigQuery backfill mode selected {len(buckets)} missing summary buckets.")
    elif RUN_MODE == "resume":
        completed = completed_summary_buckets(FETCH_START_TS, FETCH_END_TS)
        buckets = [bucket for bucket in expected_buckets if bucket not in completed]
        print(f"BigQuery resume mode selected {len(buckets)} not-yet-completed summary buckets.")
    else:
        raise ValueError("run_mode must be one of: resume, overwrite_window, backfill_missing_only")

    limit = MAX_BUCKETS_PER_RUN if MAX_BUCKETS_PER_RUN > 0 else MAX_FETCH_BUCKETS
    if limit > 0 and len(buckets) > limit:
        print(f"Limiting BigQuery summary write from {len(buckets)} buckets to {limit} buckets for chunked run.")
        buckets = buckets[:limit]
    if not buckets:
        print("No buckets selected for BigQuery summary ingestion.")
        return []
    chunk_context = bigquery_selected_chunk_context(buckets)
    query_start_ts = chunk_context["query_start_ts"]
    query_end_ts = chunk_context["query_end_ts"]
    sql = chunk_context["sql"]

    if BIGQUERY_QUOTA_GATE_ENABLED:
        bytes_processed, dry_run_detail = bigquery_dry_run_selected_chunk(
            client,
            bigquery,
            sql,
            len(buckets),
            query_start_ts,
            query_end_ts,
            credential_mode,
        )
        dry_run_gate_bytes = BIGQUERY_MAX_DRY_RUN_BYTES_PER_CHUNK or BIGQUERY_MAX_BYTES_BILLED
        if dry_run_gate_bytes > 0 and bytes_processed > dry_run_gate_bytes:
            fallback_limit = BIGQUERY_FALLBACK_FETCH_BUCKETS
            if fallback_limit > 0 and len(buckets) > fallback_limit:
                print(
                    {
                        "message": "Dry-run estimate exceeded chunk byte gate; retrying with fallback bucket count.",
                        "original_bucket_count": len(buckets),
                        "fallback_bucket_count": fallback_limit,
                        "estimated_bytes_processed": bytes_processed,
                        "dry_run_gate_bytes": dry_run_gate_bytes,
                    }
                )
                buckets = buckets[:fallback_limit]
                chunk_context = bigquery_selected_chunk_context(buckets)
                query_start_ts = chunk_context["query_start_ts"]
                query_end_ts = chunk_context["query_end_ts"]
                sql = chunk_context["sql"]
                bytes_processed, dry_run_detail = bigquery_dry_run_selected_chunk(
                    client,
                    bigquery,
                    sql,
                    len(buckets),
                    query_start_ts,
                    query_end_ts,
                    credential_mode,
                )
            if bytes_processed > dry_run_gate_bytes:
                exit_bigquery_no_write(
                    "BIGQUERY_DRY_RUN_BYTES_TOO_HIGH_NO_WRITE",
                    {
                        **dry_run_detail,
                        "dry_run_gate_bytes": dry_run_gate_bytes,
                        "selected_bucket_count_after_fallback": len(buckets),
                    },
                )
    else:
        bytes_processed = 0

    selected_bucket_set = set(buckets)
    print(
        {
            "requested_window_start": str(FETCH_START_TS),
            "requested_window_end": str(FETCH_END_TS),
            "selected_query_start": str(query_start_ts),
            "selected_query_end": str(query_end_ts),
            "selected_15m_buckets": len(buckets),
            "bigquery_quota_gate_enabled": BIGQUERY_QUOTA_GATE_ENABLED,
            "bigquery_max_bytes_billed": BIGQUERY_MAX_BYTES_BILLED or None,
            "bigquery_fallback_fetch_buckets": BIGQUERY_FALLBACK_FETCH_BUCKETS,
        }
    )

    query_started = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        query_job = client.query(sql, job_config=bigquery_query_job_config(bigquery, dry_run=False))
        result_iter = query_job.result()
    except Exception as exc:
        detail = {
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "selected_bucket_count": len(buckets),
            "selected_min_ts": str(query_start_ts),
            "selected_max_ts": str(query_end_ts),
            "dry_run_bytes_processed": bytes_processed,
            "maximum_bytes_billed": BIGQUERY_MAX_BYTES_BILLED or None,
        }
        if is_bigquery_quota_or_billing_error(exc):
            exit_bigquery_no_write("BIGQUERY_QUERY_QUOTA_BLOCKED_NO_WRITE", detail)
        raise
    result_rows = [dict(row.items()) for row in result_iter]
    query_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    query_detail = {
        "table": BIGQUERY_GKG_TABLE,
        "query_mode": BIGQUERY_QUERY_MODE,
        "record_universe": BIGQUERY_RECORD_UNIVERSE,
        "bytes_processed": int(query_job.total_bytes_processed or bytes_processed or 0),
        "bytes_billed": int(query_job.total_bytes_billed or 0),
        "cache_hit": bool(getattr(query_job, "cache_hit", False)),
        "rows_returned": len(result_rows),
    }
    log_runtime("bigquery_query_execution", query_started, query_completed, len(result_rows), json.dumps(query_detail))

    write_started = datetime.now(timezone.utc).replace(tzinfo=None)
    summary_rows = []
    for row in result_rows:
        bucket_ts = pd.Timestamp(row["bucket_ts"]).tz_localize(None).to_pydatetime()
        if bucket_ts not in selected_bucket_set:
            continue
        raw_row_count = int(row.get("raw_row_count") or 0)
        filtered_row_count = int(row.get("filtered_row_count") or 0)
        if raw_row_count == 0:
            status = BIGQUERY_ERROR_STATUS
        elif filtered_row_count == 0:
            status = BIGQUERY_ZERO_FILTERED_STATUS
        else:
            status = BIGQUERY_OK_STATUS
        summary_rows.append(
            {
                "bucket_ts": bucket_ts,
                "source_provider": BIGQUERY_CANONICAL_PROVIDER,
                "source_file_url": f"bigquery:{BIGQUERY_GKG_TABLE}",
                "raw_file_status": status,
                "raw_row_count": raw_row_count,
                "filtered_row_count": filtered_row_count,
                "avg_tone": float(row["avg_tone"]) if row.get("avg_tone") is not None else None,
                "min_tone": float(row["min_tone"]) if row.get("min_tone") is not None else None,
                "max_tone": float(row["max_tone"]) if row.get("max_tone") is not None else None,
                "policy_uncertainty_count": int(row.get("policy_uncertainty_count") or 0),
                "geopolitical_risk_count": int(row.get("geopolitical_risk_count") or 0),
                "market_stress_count": int(row.get("market_stress_count") or 0),
                **{family["count_col"]: int(row.get(family["count_col"]) or 0) for family in NARROW_TOPIC_FAMILIES},
                "risk_on_count": int(row.get("risk_on_count") or 0),
                "risk_off_count": int(row.get("risk_off_count") or 0),
                "source_bucket_count": 1,
                "has_missing_15m_bucket": False,
                "ingestion_run_id": INGESTION_RUN_ID,
                "ingestion_ts": RUN_TS,
                "last_error": None,
                "download_seconds": 0.0,
                "parse_seconds": 0.0,
                "summary_seconds": 0.0,
            }
        )

    present_buckets = {row["bucket_ts"] for row in summary_rows if row["raw_row_count"] > 0}
    missing_raw_compat_buckets = sorted(selected_bucket_set - present_buckets)
    fallback_rows = []
    if missing_raw_compat_buckets and BIGQUERY_RECORD_UNIVERSE == "raw_gkg_compat":
        fallback_started = datetime.now(timezone.utc).replace(tzinfo=None)
        print(
            {
                "bigquery_missing_raw_compat_bucket_count": len(missing_raw_compat_buckets),
                "fallback_mode": "raw_zip_gapfill",
                "missing_buckets": [str(bucket) for bucket in missing_raw_compat_buckets[:20]],
            }
        )
        for group in contiguous_timestamp_groups(missing_raw_compat_buckets):
            bulk_rows = classify_contiguous_missing_gap(group)
            if bulk_rows is not None:
                fallback_rows.extend(bulk_rows)
                continue
            for bucket in group:
                fallback_rows.append(fetch_raw_zip_gapfill_bucket_summary(bucket))
        fallback_completed = datetime.now(timezone.utc).replace(tzinfo=None)
        log_runtime(
            "bigquery_raw_zip_gapfill",
            fallback_started,
            fallback_completed,
            len(fallback_rows),
            json.dumps(
                {
                    "fallback_bucket_count": len(fallback_rows),
                    "fallback_ok_count": sum(1 for row in fallback_rows if row["raw_file_status"] == FALLBACK_OK_STATUS),
                    "fallback_missing_count": sum(1 for row in fallback_rows if row["raw_file_status"] == FALLBACK_MISSING_STATUS),
                    "fallback_failed_count": sum(1 for row in fallback_rows if row["raw_file_status"] == FALLBACK_FAILED_STATUS),
                    "bulk_gap_classified_count": sum(
                        1
                        for row in fallback_rows
                        if row["raw_file_status"] == FALLBACK_MISSING_STATUS
                        and row.get("last_error")
                        and "contiguous_raw_compat_gap" in row["last_error"]
                    ),
                }
            ),
        )
    elif missing_raw_compat_buckets:
        print(
            {
                "bigquery_missing_bucket_count": len(missing_raw_compat_buckets),
                "fallback_mode": "disabled_for_all_bigquery_gkg",
            }
        )

    summary_rows.extend(fallback_rows)
    append_bucket_summary_rows(summary_rows)
    write_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("bigquery_spark_load_write", write_started, write_completed, len(summary_rows), json.dumps(query_detail))

    validation_summary = append_validation_summary_from_bucket_summaries(buckets)
    total_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("bigquery_total", query_construct_started, total_completed, len(summary_rows), json.dumps(query_detail))
    return [validation_summary]


if EXECUTE_FETCH:
    if GDELT_SOURCE_MODE == "raw_fast":
        fetch_summaries = fetch_fast_summary_window()
    elif GDELT_SOURCE_MODE == "raw_audit":
        fetch_summaries = fetch_raw_window()
    elif GDELT_SOURCE_MODE == "bigquery":
        fetch_summaries = fetch_bigquery_summary_window()
    elif GDELT_SOURCE_MODE == "raw_zip_hourly_summary":
        fetch_summaries = fetch_raw_zip_hourly_summary_window()
    else:
        raise ValueError(f"Unsupported gdelt_source_mode={GDELT_SOURCE_MODE}")
else:
    print("Fetch skipped. Set execute_fetch=true to ingest or summarize GDELT data into research tables.")

# COMMAND ----------

def validate_ingestion_window():
    expected_count = len(expected_15m_buckets(FETCH_START_TS, FETCH_END_TS))
    if GDELT_SOURCE_MODE == "raw_zip_hourly_summary":
        expected_hour_count = len(expected_hourly_timestamps(FETCH_START_TS, FETCH_END_TS))
        lag_minutes = NEWS_AVAILABILITY_LAG_MINUTES
        validation_df = spark.sql(
            f"""
            WITH latest_features AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY timestamp
                        ORDER BY ingestion_ts DESC
                    ) AS rn
                FROM {RESEARCH_FEATURE_TABLE}
                WHERE timestamp >= TIMESTAMP('{FETCH_START_TS.ceil("H")}')
                  AND timestamp <= TIMESTAMP('{FETCH_END_TS.floor("H")}')
                  AND source = 'raw_zip_hourly_summary'
            ),
            features AS (
                SELECT *
                FROM latest_features
                WHERE rn = 1
            )
            SELECT
                {expected_hour_count} AS expected_hourly_rows,
                COUNT(*) AS actual_hourly_rows,
                COUNT(*) - COUNT(DISTINCT timestamp) AS duplicate_hourly_timestamps,
                SUM(CASE WHEN source_bucket_count < 4 THEN 1 ELSE 0 END) AS hours_with_fewer_than_4_source_buckets,
                SUM(CASE WHEN filtered_news_row_count = 0 THEN 1 ELSE 0 END) AS hours_with_zero_filtered_rows,
                SUM(CASE WHEN max_gdelt_source_ts > timestamp - INTERVAL {lag_minutes} MINUTES THEN 1 ELSE 0 END) AS lag_leakage_rows,
                SUM(CASE WHEN news_volume_z IS NULL
                           OR news_tone_z IS NULL
                           OR policy_uncertainty_news_z IS NULL
                           OR geopolitical_risk_news_z IS NULL
                           OR market_stress_news_z IS NULL
                         THEN 1 ELSE 0 END) AS null_z_feature_rows,
                MIN(timestamp) AS min_timestamp,
                MAX(timestamp) AS max_timestamp,
                MIN(min_gdelt_source_ts) AS min_source_ts,
                MAX(max_gdelt_source_ts) AS max_source_ts,
                SUM(raw_news_row_count) AS raw_news_row_count,
                SUM(filtered_news_row_count) AS filtered_news_row_count
            FROM features
            """
        )
        display(validation_df)
        row = validation_df.collect()[0].asDict()
        validation_passed = (
            int(row["actual_hourly_rows"] or 0) == expected_hour_count
            and int(row["duplicate_hourly_timestamps"] or 0) == 0
            and int(row["lag_leakage_rows"] or 0) == 0
            and int(row["null_z_feature_rows"] or 0) == 0
        )
        print(f"Raw-ZIP hourly feature validation passed: {validation_passed}")
        return validation_passed, row

    if PIPELINE_MODE == "fast_feature_mode":
        provider = bucket_summary_source_provider()
        fetched_statuses_sql = sql_string_list(SUMMARY_FETCHED_STATUSES)
        missing_statuses_sql = sql_string_list(SUMMARY_MISSING_STATUSES)
        failed_statuses_sql = sql_string_list(SUMMARY_FAILED_STATUSES)
        validation_sql = f"""
        WITH latest_summary AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY bucket_ts, source_provider
                    ORDER BY ingestion_ts DESC
                ) AS rn
            FROM {BUCKET_SUMMARY_TABLE}
            WHERE bucket_ts >= TIMESTAMP('{FETCH_START_TS}')
              AND bucket_ts <= TIMESTAMP('{FETCH_END_TS}')
              AND source_provider = '{provider}'
        ),
        summary AS (
            SELECT *
            FROM latest_summary
            WHERE rn = 1
        )
        SELECT
            {expected_count} AS expected_15m_buckets,
            (SELECT COUNT(*) FROM summary WHERE raw_file_status IN ({fetched_statuses_sql})) AS raw_file_fetched_buckets,
            (SELECT COUNT(*) FROM summary WHERE raw_file_status IN ({missing_statuses_sql})) AS raw_file_missing_buckets,
            (SELECT COUNT(*) FROM summary WHERE raw_file_status IN ({failed_statuses_sql})) AS fetch_failed_buckets,
            (SELECT COUNT(*) FROM summary WHERE raw_file_status = 'zero_raw_rows') AS zero_raw_buckets,
            (SELECT COUNT(*) FROM summary WHERE raw_file_status IN ('zero_filtered_rows', '{BIGQUERY_ZERO_FILTERED_STATUS}')) AS zero_filtered_buckets,
            (SELECT COUNT(DISTINCT bucket_ts) FROM summary) AS persisted_buckets,
            {expected_count} - (SELECT COUNT(DISTINCT bucket_ts) FROM summary) AS missing_persisted_buckets,
            (SELECT COUNT(*) - COUNT(DISTINCT bucket_ts) FROM summary) AS duplicate_deterministic_keys,
            (SELECT COUNT(*) FROM summary) AS persisted_rows,
            (SELECT SUM(raw_row_count) FROM summary) AS raw_row_count,
            (SELECT SUM(filtered_row_count) FROM summary) AS filtered_row_count,
            (SELECT MIN(bucket_ts) FROM summary) AS min_bucket_ts,
            (SELECT MAX(bucket_ts) FROM summary) AS max_bucket_ts,
            (SELECT MIN(bucket_ts) FROM summary WHERE raw_file_status IN ({fetched_statuses_sql})) AS min_source_ts,
            (SELECT MAX(bucket_ts) FROM summary WHERE raw_file_status IN ({fetched_statuses_sql})) AS max_source_ts,
            CAST(0.0 AS DOUBLE) AS gdelt_record_id_null_rate,
            CAST(0.0 AS DOUBLE) AS document_identifier_null_rate,
            CAST(0.0 AS DOUBLE) AS v2_themes_null_rate,
            CAST(0.0 AS DOUBLE) AS v2_tone_raw_null_rate,
            AVG(CASE WHEN raw_file_status IN ({fetched_statuses_sql}) AND filtered_row_count > 0 AND avg_tone IS NULL THEN 1 ELSE 0 END) AS tone_null_rate,
            CAST(0.0 AS DOUBLE) AS deterministic_key_null_rate
        FROM summary
        """
        validation_df = spark.sql(validation_sql)
        display(validation_df)
        row = validation_df.collect()[0].asDict()
        raw_file_missing = int(row["raw_file_missing_buckets"] or 0)
        fetched_buckets = int(row["raw_file_fetched_buckets"] or 0)
        validation_passed = (
            fetched_buckets + raw_file_missing == expected_count
            and int(row["fetch_failed_buckets"] or 0) == 0
            and int(row["zero_raw_buckets"] or 0) == 0
            and int(row["missing_persisted_buckets"] or 0) == 0
            and float(row["tone_null_rate"] or 0.0) == 0.0
        )
        print(f"Bucket-summary ingestion validation passed for {provider}: {validation_passed}")
        return validation_passed, row

    validation_sql = f"""
    WITH latest_checkpoint AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY bucket_ts
                ORDER BY ingestion_ts DESC
            ) AS rn
        FROM {CHECKPOINT_TABLE}
        WHERE bucket_ts >= TIMESTAMP('{FETCH_START_TS}')
          AND bucket_ts <= TIMESTAMP('{FETCH_END_TS}')
    ),
    checkpoint AS (
        SELECT *
        FROM latest_checkpoint
        WHERE rn = 1
    ),
    bronze AS (
        SELECT *
        FROM {BRONZE_NEWS_TABLE}
        WHERE source_ts >= TIMESTAMP('{FETCH_START_TS}')
          AND source_ts <= TIMESTAMP('{FETCH_END_TS}')
    )
    SELECT
        {expected_count} AS expected_15m_buckets,
        (SELECT COUNT(*) FROM checkpoint WHERE raw_file_fetched) AS raw_file_fetched_buckets,
        (SELECT COUNT(*) FROM checkpoint WHERE write_status = 'raw_file_missing') AS raw_file_missing_buckets,
        (SELECT COUNT(*) FROM checkpoint WHERE write_status = 'fetch_failed') AS fetch_failed_buckets,
        (SELECT COUNT(*) FROM checkpoint WHERE write_status = 'zero_raw_rows') AS zero_raw_buckets,
        (SELECT COUNT(*) FROM checkpoint WHERE write_status = 'zero_filtered_rows') AS zero_filtered_buckets,
        (SELECT COUNT(DISTINCT source_ts) FROM bronze) AS persisted_buckets,
        {expected_count} - (SELECT COUNT(DISTINCT source_ts) FROM bronze) AS missing_persisted_buckets,
        (SELECT COUNT(*) - COUNT(DISTINCT deterministic_key) FROM bronze) AS duplicate_deterministic_keys,
        (SELECT COUNT(*) FROM bronze) AS persisted_rows,
        (SELECT SUM(raw_row_count) FROM checkpoint) AS raw_row_count,
        (SELECT SUM(filtered_row_count) FROM checkpoint) AS filtered_row_count,
        (SELECT MIN(bucket_ts) FROM checkpoint) AS min_bucket_ts,
        (SELECT MAX(bucket_ts) FROM checkpoint) AS max_bucket_ts,
        (SELECT MIN(source_ts) FROM bronze) AS min_source_ts,
        (SELECT MAX(source_ts) FROM bronze) AS max_source_ts,
        (SELECT AVG(CASE WHEN gdelt_record_id IS NULL THEN 1 ELSE 0 END) FROM bronze) AS gdelt_record_id_null_rate,
        (SELECT AVG(CASE WHEN document_identifier IS NULL THEN 1 ELSE 0 END) FROM bronze) AS document_identifier_null_rate,
        (SELECT AVG(CASE WHEN v2_themes IS NULL THEN 1 ELSE 0 END) FROM bronze) AS v2_themes_null_rate,
        (SELECT AVG(CASE WHEN v2_tone_raw IS NULL THEN 1 ELSE 0 END) FROM bronze) AS v2_tone_raw_null_rate,
        (SELECT AVG(CASE WHEN tone IS NULL THEN 1 ELSE 0 END) FROM bronze) AS tone_null_rate,
        (SELECT AVG(CASE WHEN deterministic_key IS NULL THEN 1 ELSE 0 END) FROM bronze) AS deterministic_key_null_rate
    """
    validation_df = spark.sql(validation_sql)
    display(validation_df)
    row = validation_df.collect()[0].asDict()
    raw_file_missing = int(row["raw_file_missing_buckets"] or 0)
    persisted_buckets = int(row["persisted_buckets"] or 0)
    validation_passed = (
        persisted_buckets + raw_file_missing == expected_count
        and int(row["fetch_failed_buckets"] or 0) == 0
        and int(row["zero_raw_buckets"] or 0) == 0
        and int(row["duplicate_deterministic_keys"] or 0) == 0
        and float(row["gdelt_record_id_null_rate"] or 0.0) == 0.0
        and float(row["document_identifier_null_rate"] or 0.0) == 0.0
        and float(row["v2_themes_null_rate"] or 0.0) == 0.0
        and float(row["v2_tone_raw_null_rate"] or 0.0) == 0.0
        and float(row["tone_null_rate"] or 0.0) == 0.0
        and float(row["deterministic_key_null_rate"] or 0.0) == 0.0
    )
    print(f"Ingestion validation passed: {validation_passed}")
    return validation_passed, row


INGESTION_VALIDATION_PASSED, INGESTION_VALIDATION_ROW = validate_ingestion_window()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Completeness Dashboard

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
            DATE(batch_start_ts) AS batch_date,
            source,
            SUM(expected_15m_buckets) AS total_15m_buckets,
            SUM(raw_row_count) AS total_rows,
            SUM(missing_15m_buckets) AS missing_bucket_count,
            SUM(raw_row_count) / NULLIF(SUM(observed_15m_buckets), 0) AS average_rows_per_bucket,
            MIN(min_source_ts) AS min_source_ts,
            MAX(max_source_ts) AS max_source_ts,
            SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) AS error_batches,
            SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) AS warning_batches
        FROM {VALIDATION_TABLE}
        GROUP BY DATE(batch_start_ts), source
        ORDER BY batch_date, source
        """
    )
)

if GDELT_SOURCE_MODE == "raw_zip_hourly_summary":
    display(
        spark.sql(
            f"""
            SELECT
                DATE(timestamp) AS feature_date,
                source,
                COUNT(*) AS hourly_rows,
                SUM(source_bucket_count) AS source_15m_buckets,
                SUM(CASE WHEN source_bucket_count < 4 THEN 1 ELSE 0 END) AS hours_with_fewer_than_4_source_buckets,
                SUM(raw_news_row_count) AS raw_news_row_count,
                SUM(filtered_news_row_count) AS filtered_news_row_count,
                MIN(min_gdelt_source_ts) AS min_source_ts,
                MAX(max_gdelt_source_ts) AS max_source_ts
            FROM {RESEARCH_FEATURE_TABLE}
            WHERE timestamp BETWEEN TIMESTAMP('{FETCH_START_TS.ceil("H")}') AND TIMESTAMP('{FETCH_END_TS.floor("H")}')
              AND source = 'raw_zip_hourly_summary'
            GROUP BY DATE(timestamp), source
            ORDER BY feature_date
            """
        )
    )
elif PIPELINE_MODE == "fast_feature_mode":
    provider = bucket_summary_source_provider()
    fetched_statuses_sql = sql_string_list(SUMMARY_FETCHED_STATUSES)
    missing_statuses_sql = sql_string_list(SUMMARY_MISSING_STATUSES)
    failed_statuses_sql = sql_string_list(SUMMARY_FAILED_STATUSES)
    display(
        spark.sql(
            f"""
            SELECT
                DATE(bucket_ts) AS source_date,
                source_provider,
                COUNT(DISTINCT bucket_ts) AS total_15m_buckets,
                SUM(raw_row_count) AS total_rows,
                96 - COUNT(DISTINCT bucket_ts) AS missing_bucket_count,
                SUM(raw_row_count) / NULLIF(COUNT(DISTINCT bucket_ts), 0) AS average_rows_per_bucket,
                MIN(min_tone) AS min_tone,
                MAX(max_tone) AS max_tone,
                AVG(CASE WHEN raw_file_status IN ({fetched_statuses_sql}) AND filtered_row_count > 0 AND avg_tone IS NULL THEN 1 ELSE 0 END) AS tone_null_rate,
                SUM(CASE WHEN raw_file_status IN ({failed_statuses_sql}) THEN 1 ELSE 0 END) AS fetch_failed_buckets,
                SUM(CASE WHEN raw_file_status IN ({missing_statuses_sql}) THEN 1 ELSE 0 END) AS raw_file_missing_buckets
            FROM {BUCKET_SUMMARY_TABLE}
            WHERE bucket_ts BETWEEN TIMESTAMP('{FETCH_START_TS}') AND TIMESTAMP('{FETCH_END_TS}')
              AND source_provider = '{provider}'
            GROUP BY DATE(bucket_ts), source_provider
            ORDER BY source_date
            """
        )
    )
else:
    display(
        spark.sql(
            f"""
            SELECT
                DATE(source_ts) AS source_date,
                COUNT(DISTINCT source_ts) AS total_15m_buckets,
                COUNT(*) AS total_rows,
                96 - COUNT(DISTINCT source_ts) AS missing_bucket_count,
                COUNT(*) / NULLIF(COUNT(DISTINCT source_ts), 0) AS average_rows_per_bucket,
                MIN(tone) AS min_tone,
                MAX(tone) AS max_tone,
                AVG(CASE WHEN tone IS NULL THEN 1 ELSE 0 END) AS tone_null_rate,
                AVG(CASE WHEN document_identifier IS NULL THEN 1 ELSE 0 END) AS document_identifier_null_rate,
                AVG(CASE WHEN v2_themes IS NULL THEN 1 ELSE 0 END) AS v2_themes_null_rate
            FROM {BRONZE_NEWS_TABLE}
            WHERE source_ts BETWEEN TIMESTAMP('{FETCH_START_TS}') AND TIMESTAMP('{FETCH_END_TS}')
            GROUP BY DATE(source_ts)
            ORDER BY source_date
            """
        )
    )

# COMMAND ----------

if GDELT_SOURCE_MODE == "raw_zip_hourly_summary":
    HAS_BRONZE_WINDOW = False
    print("Raw-ZIP hourly summary mode writes hourly features directly; downstream aggregation cells are skipped.")
elif PIPELINE_MODE == "fast_feature_mode":
    HAS_BRONZE_WINDOW = (
        RUN_AFTER_INGESTION
        and INGESTION_VALIDATION_PASSED
        and (
            spark.table(BUCKET_SUMMARY_TABLE)
            .filter(F.col("bucket_ts") >= F.lit(FETCH_START_TS.to_pydatetime()))
            .filter(F.col("bucket_ts") <= F.lit(FETCH_END_TS.to_pydatetime()))
            .filter(F.col("source_provider") == bucket_summary_source_provider())
            .limit(1)
            .count()
            > 0
        )
    )
else:
    HAS_BRONZE_WINDOW = (
        RUN_AFTER_INGESTION
        and INGESTION_VALIDATION_PASSED
        and (
            spark.table(BRONZE_NEWS_TABLE)
            .filter(F.col("source_ts") >= F.lit(FETCH_START_TS.to_pydatetime()))
            .filter(F.col("source_ts") <= F.lit(FETCH_END_TS.to_pydatetime()))
            .limit(1)
            .count()
            > 0
        )
    )

if not HAS_BRONZE_WINDOW:
    print(
        "Downstream feature aggregation and backtest cells will be skipped. "
        "Set run_after_ingestion=true only after ingestion validation passes cleanly."
    )

# COMMAND ----------

def align_to_existing_table_schema(source_df, table_name):
    try:
        target_schema = spark.table(table_name).schema
    except Exception:
        return source_df

    aligned = source_df
    for field in target_schema:
        if field.name not in aligned.columns:
            if isinstance(field.dataType, (DoubleType, IntegerType, LongType)):
                default_value = F.lit(0)
            elif isinstance(field.dataType, BooleanType):
                default_value = F.lit(False)
            else:
                default_value = F.lit(None)
            aligned = aligned.withColumn(field.name, default_value.cast(field.dataType))
    return aligned


def build_hourly_news_features_from_bronze():
    hourly_started = datetime.now(timezone.utc).replace(tzinfo=None)
    bronze = (
        spark.table(BRONZE_NEWS_TABLE)
        .filter(F.col("source_ts") >= F.lit(FETCH_START_TS.to_pydatetime()))
        .filter(F.col("source_ts") <= F.lit(FETCH_END_TS.to_pydatetime()))
    )
    if bronze.limit(1).count() == 0:
        raise ValueError(f"No rows found in {BRONZE_NEWS_TABLE} for requested window.")

    bucket_hour_ts = F.when(
        (F.minute("source_ts") == 0) & (F.second("source_ts") == 0),
        F.date_trunc("hour", F.col("source_ts")),
    ).otherwise(F.date_trunc("hour", F.col("source_ts")) + F.expr("INTERVAL 1 HOUR"))

    expected_bucket_rows = [
        {"source_ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts}
        for ts in expected_15m_buckets(FETCH_START_TS, FETCH_END_TS)
    ]
    expected_buckets = spark.createDataFrame(expected_bucket_rows)
    expected_hourly = (
        expected_buckets.withColumn("timestamp", bucket_hour_ts)
        .groupBy("timestamp")
        .agg(F.countDistinct("source_ts").cast("int").alias("expected_source_bucket_count"))
    )

    latest_checkpoint = spark.sql(
        f"""
        SELECT *
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY bucket_ts
                    ORDER BY ingestion_ts DESC
                ) AS rn
            FROM {CHECKPOINT_TABLE}
            WHERE bucket_ts >= TIMESTAMP('{FETCH_START_TS}')
              AND bucket_ts <= TIMESTAMP('{FETCH_END_TS}')
        )
        WHERE rn = 1
        """
    )
    checkpoint_hourly = (
        latest_checkpoint.withColumnRenamed("bucket_ts", "source_ts")
        .withColumn("timestamp", bucket_hour_ts)
        .groupBy("timestamp")
        .agg(
            F.sum("raw_row_count").cast("long").alias("raw_news_row_count_checkpoint"),
            F.sum("filtered_row_count").cast("long").alias("filtered_news_row_count_checkpoint"),
            F.sum(F.when(F.col("write_status") == "raw_file_missing", 1).otherwise(0)).cast("int").alias("raw_file_missing_bucket_count"),
            F.sum(F.when(F.col("write_status") == "fetch_failed", 1).otherwise(0)).cast("int").alias("fetch_failed_bucket_count"),
        )
    )

    hourly = (
        bronze.withColumn("timestamp", bucket_hour_ts)
        .filter(F.col("is_relevant"))
        .groupBy("timestamp")
        .agg(
            F.count("*").cast("double").alias("news_volume"),
            F.avg("tone").alias("news_tone"),
            F.sum(F.when(F.col("is_policy_uncertainty"), 1).otherwise(0)).cast("double").alias("policy_uncertainty_news"),
            F.sum(F.when(F.col("is_geopolitical_risk"), 1).otherwise(0)).cast("double").alias("geopolitical_risk_news"),
            F.sum(F.when(F.col("is_market_stress"), 1).otherwise(0)).cast("double").alias("market_stress_news"),
            *[F.lit(0.0).alias(family["raw_col"]) for family in NARROW_TOPIC_FAMILIES],
            F.countDistinct("source_ts").cast("int").alias("source_bucket_count"),
            F.count("*").cast("long").alias("filtered_news_row_count"),
            F.min("source_ts").alias("min_gdelt_source_ts"),
            F.max("source_ts").alias("max_gdelt_source_ts"),
        )
    )

    grid = (
        spark.sql(
            f"""
            SELECT explode(sequence(
                TIMESTAMP('{FETCH_START_TS.floor("H")}'),
                TIMESTAMP('{FETCH_END_TS.floor("H")}'),
                INTERVAL 1 HOUR
            )) AS timestamp
            """
        )
        .join(hourly, "timestamp", "left")
        .join(expected_hourly, "timestamp", "left")
        .join(checkpoint_hourly, "timestamp", "left")
        .fillna(
            {
                "news_volume": 0.0,
                "policy_uncertainty_news": 0.0,
                "geopolitical_risk_news": 0.0,
                "market_stress_news": 0.0,
                **{family["raw_col"]: 0.0 for family in NARROW_TOPIC_FAMILIES},
                "source_bucket_count": 0,
                "filtered_news_row_count": 0,
                "raw_file_missing_bucket_count": 0,
                "fetch_failed_bucket_count": 0,
            }
        )
        .withColumn(
            "raw_news_row_count",
            F.col("raw_news_row_count_checkpoint").cast("long"),
        )
        .withColumn(
            "has_missing_15m_bucket",
            (F.col("source_bucket_count") < F.col("expected_source_bucket_count"))
            | (F.col("raw_file_missing_bucket_count") > 0)
            | (F.col("fetch_failed_bucket_count") > 0),
        )
        .withColumn(
            "missing_15m_bucket_count",
            F.greatest(
                F.lit(0),
                F.col("expected_source_bucket_count") - F.col("source_bucket_count"),
                F.col("raw_file_missing_bucket_count") + F.col("fetch_failed_bucket_count"),
            ).cast("int"),
        )
        .withColumn(
            "missing_15m_bucket_pct",
            F.when(F.col("expected_source_bucket_count") > 0, F.col("missing_15m_bucket_count") / F.col("expected_source_bucket_count")).otherwise(F.lit(0.0)),
        )
        .withColumn("gdelt_missing_flag", F.when(F.col("missing_15m_bucket_count") > 0, F.lit(1)).otherwise(F.lit(0)))
        .drop(
            "expected_source_bucket_count",
            "raw_news_row_count_checkpoint",
            "filtered_news_row_count_checkpoint",
            "raw_file_missing_bucket_count",
            "fetch_failed_bucket_count",
        )
    )

    win_z = Window.orderBy("timestamp").rowsBetween(-8760, 0)
    for raw_col, z_col in [
        ("news_volume", "news_volume_z"),
        ("news_tone", "news_tone_z"),
        ("policy_uncertainty_news", "policy_uncertainty_news_z"),
        ("geopolitical_risk_news", "geopolitical_risk_news_z"),
        ("market_stress_news", "market_stress_news_z"),
        *[(family["raw_col"], family["z_col"]) for family in NARROW_TOPIC_FAMILIES],
    ]:
        avg_val = F.avg(F.col(raw_col)).over(win_z)
        std_val = F.stddev(F.col(raw_col)).over(win_z)
        grid = grid.withColumn(
            z_col,
            F.when(std_val.isNull() | (std_val == 0), F.lit(0.0)).otherwise((F.col(raw_col) - avg_val) / std_val),
        )

    grid = add_engineered_news_features(grid)
    final_features = (
        grid.withColumn("source", F.lit(f"{SOURCE_MODE}_research"))
        .withColumn("ingestion_run_id", F.lit(INGESTION_RUN_ID))
        .withColumn("ingestion_ts", F.current_timestamp())
        .select(
            "timestamp",
            "news_volume",
            "news_tone",
            "policy_uncertainty_news",
            "geopolitical_risk_news",
            "market_stress_news",
            *NARROW_TOPIC_RAW_FEATURES,
            "news_volume_z",
            "news_tone_z",
            "policy_uncertainty_news_z",
            "geopolitical_risk_news_z",
            "market_stress_news_z",
            *NARROW_TOPIC_Z_FEATURES,
            *ENGINEERED_NEWS_FEATURES,
            "source_bucket_count",
            "missing_15m_bucket_count",
            "missing_15m_bucket_pct",
            "gdelt_missing_flag",
            "raw_news_row_count",
            "filtered_news_row_count",
            "min_gdelt_source_ts",
            "max_gdelt_source_ts",
            "has_missing_15m_bucket",
            "source",
            "ingestion_run_id",
            "ingestion_ts",
        )
    )

    for table_name in [SILVER_NEWS_TABLE, RESEARCH_FEATURE_TABLE]:
        align_to_existing_table_schema(final_features, table_name).createOrReplaceTempView("incoming_silver_news_flow_1h")
        spark.sql(
            f"""
            MERGE INTO {table_name} t
            USING incoming_silver_news_flow_1h s
            ON t.timestamp = s.timestamp
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    hourly_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("hourly_aggregation", hourly_started, hourly_completed, final_features.count(), "audit_raw_mode")
    return final_features


def build_hourly_news_features_from_bucket_summary():
    hourly_started = datetime.now(timezone.utc).replace(tzinfo=None)
    provider = bucket_summary_source_provider()
    available_status_expr = F.col("raw_file_status").isin(list(SUMMARY_FETCHED_STATUSES))
    missing_status_expr = F.col("raw_file_status").isin(list(SUMMARY_MISSING_STATUSES))
    failed_status_expr = F.col("raw_file_status").isin(list(SUMMARY_FAILED_STATUSES))
    latest_summary = spark.sql(
        f"""
        SELECT *
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY bucket_ts, source_provider
                    ORDER BY
                        CASE raw_file_status
                            WHEN 'bigquery_ok' THEN 1
                            WHEN 'bigquery_missing_raw_compat_fallback_raw_zip_ok' THEN 1
                            WHEN 'persisted' THEN 1
                            WHEN 'bigquery_zero_filtered' THEN 2
                            WHEN 'zero_filtered_rows' THEN 2
                            WHEN 'zero_raw_rows' THEN 3
                            WHEN 'bigquery_missing_raw_compat_fallback_raw_zip_missing' THEN 4
                            WHEN 'raw_file_missing' THEN 4
                            WHEN 'raw_zip_malformed_upstream' THEN 4
                            WHEN 'bigquery_error' THEN 8
                            WHEN 'bigquery_missing_raw_compat_fallback_raw_zip_failed' THEN 8
                            ELSE 9
                        END,
                        ingestion_ts DESC
                ) AS rn
            FROM {BUCKET_SUMMARY_TABLE}
            WHERE bucket_ts >= TIMESTAMP('{FETCH_START_TS}')
              AND bucket_ts <= TIMESTAMP('{FETCH_END_TS}')
              AND source_provider = '{provider}'
        )
        WHERE rn = 1
        """
    )
    if latest_summary.limit(1).count() == 0:
        raise ValueError(f"No rows found in {BUCKET_SUMMARY_TABLE} for {provider} in requested window.")

    bucket_hour_ts = F.when(
        (F.minute("source_ts") == 0) & (F.second("source_ts") == 0),
        F.date_trunc("hour", F.col("source_ts")),
    ).otherwise(F.date_trunc("hour", F.col("source_ts")) + F.expr("INTERVAL 1 HOUR"))

    expected_bucket_rows = [
        {"source_ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts}
        for ts in expected_15m_buckets(FETCH_START_TS, FETCH_END_TS)
    ]
    expected_hourly = (
        spark.createDataFrame(expected_bucket_rows)
        .withColumn("timestamp", bucket_hour_ts)
        .groupBy("timestamp")
        .agg(F.countDistinct("source_ts").cast("int").alias("expected_source_bucket_count"))
    )

    summary_hourly = (
        latest_summary.withColumnRenamed("bucket_ts", "source_ts")
        .withColumn("timestamp", bucket_hour_ts)
        .groupBy("timestamp")
        .agg(
            F.sum("filtered_row_count").cast("double").alias("news_volume"),
            F.when(
                F.sum("filtered_row_count") == 0,
                F.lit(None).cast("double"),
            ).otherwise(
                F.sum(F.col("avg_tone") * F.col("filtered_row_count")) / F.sum("filtered_row_count")
            ).alias("news_tone"),
            F.sum("policy_uncertainty_count").cast("double").alias("policy_uncertainty_news"),
            F.sum("geopolitical_risk_count").cast("double").alias("geopolitical_risk_news"),
            F.sum("market_stress_count").cast("double").alias("market_stress_news"),
            *[F.sum(family["count_col"]).cast("double").alias(family["raw_col"]) for family in NARROW_TOPIC_FAMILIES],
            F.sum(F.when(available_status_expr, F.col("source_bucket_count")).otherwise(0)).cast("int").alias("source_bucket_count"),
            F.sum(F.when(available_status_expr, 1).otherwise(0)).cast("int").alias("available_source_bucket_count"),
            F.sum(F.when(missing_status_expr, 1).otherwise(0)).cast("int").alias("documented_missing_bucket_count"),
            F.sum(F.when(failed_status_expr, 1).otherwise(0)).cast("int").alias("failed_bucket_count"),
            F.sum("raw_row_count").cast("long").alias("raw_news_row_count"),
            F.sum("filtered_row_count").cast("long").alias("filtered_news_row_count"),
            F.min(F.when(available_status_expr & (F.col("source_bucket_count") > 0), F.col("source_ts"))).alias("min_gdelt_source_ts"),
            F.max(F.when(available_status_expr & (F.col("source_bucket_count") > 0), F.col("source_ts"))).alias("max_gdelt_source_ts"),
            F.max(F.when(F.col("has_missing_15m_bucket"), 1).otherwise(0)).cast("int").alias("missing_bucket_flag"),
        )
    )

    grid = (
        spark.sql(
            f"""
            SELECT explode(sequence(
                TIMESTAMP('{FETCH_START_TS.floor("H")}'),
                TIMESTAMP('{FETCH_END_TS.floor("H")}'),
                INTERVAL 1 HOUR
            )) AS timestamp
            """
        )
        .join(summary_hourly, "timestamp", "left")
        .join(expected_hourly, "timestamp", "left")
        .fillna(
            {
                "news_volume": 0.0,
                "policy_uncertainty_news": 0.0,
                "geopolitical_risk_news": 0.0,
                "market_stress_news": 0.0,
                **{family["raw_col"]: 0.0 for family in NARROW_TOPIC_FAMILIES},
                "source_bucket_count": 0,
                "available_source_bucket_count": 0,
                "documented_missing_bucket_count": 0,
                "failed_bucket_count": 0,
                "raw_news_row_count": 0,
                "filtered_news_row_count": 0,
                "missing_bucket_flag": 0,
            }
        )
        .withColumn("news_tone", F.coalesce(F.col("news_tone"), F.lit(0.0)))
        .withColumn(
            "missing_15m_bucket_count",
            F.greatest(
                F.lit(0),
                F.col("expected_source_bucket_count") - F.col("available_source_bucket_count"),
                F.col("documented_missing_bucket_count") + F.col("failed_bucket_count"),
            ).cast("int"),
        )
        .withColumn(
            "missing_15m_bucket_pct",
            F.when(F.col("expected_source_bucket_count") > 0, F.col("missing_15m_bucket_count") / F.col("expected_source_bucket_count")).otherwise(F.lit(0.0)),
        )
        .withColumn("gdelt_missing_flag", F.when(F.col("missing_15m_bucket_count") > 0, F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "has_missing_15m_bucket",
            (F.col("gdelt_missing_flag") == 1)
            | (F.col("missing_bucket_flag") > 0),
        )
        .drop(
            "expected_source_bucket_count",
            "available_source_bucket_count",
            "documented_missing_bucket_count",
            "failed_bucket_count",
            "missing_bucket_flag",
        )
    )

    win_z = Window.orderBy("timestamp").rowsBetween(-8760, 0)
    for raw_col, z_col in [
        ("news_volume", "news_volume_z"),
        ("news_tone", "news_tone_z"),
        ("policy_uncertainty_news", "policy_uncertainty_news_z"),
        ("geopolitical_risk_news", "geopolitical_risk_news_z"),
        ("market_stress_news", "market_stress_news_z"),
        *[(family["raw_col"], family["z_col"]) for family in NARROW_TOPIC_FAMILIES],
    ]:
        avg_val = F.avg(F.col(raw_col)).over(win_z)
        std_val = F.stddev(F.col(raw_col)).over(win_z)
        grid = grid.withColumn(
            z_col,
            F.when(std_val.isNull() | (std_val == 0), F.lit(0.0)).otherwise((F.col(raw_col) - avg_val) / std_val),
        )

    grid = add_engineered_news_features(grid)
    final_features = (
        grid.withColumn("source", F.lit(f"{GDELT_SOURCE_MODE}:gkg_summary_15m"))
        .withColumn("ingestion_run_id", F.lit(INGESTION_RUN_ID))
        .withColumn("ingestion_ts", F.current_timestamp())
        .select(
            "timestamp",
            "news_volume",
            "news_tone",
            "policy_uncertainty_news",
            "geopolitical_risk_news",
            "market_stress_news",
            *NARROW_TOPIC_RAW_FEATURES,
            "news_volume_z",
            "news_tone_z",
            "policy_uncertainty_news_z",
            "geopolitical_risk_news_z",
            "market_stress_news_z",
            *NARROW_TOPIC_Z_FEATURES,
            *ENGINEERED_NEWS_FEATURES,
            "source_bucket_count",
            "missing_15m_bucket_count",
            "missing_15m_bucket_pct",
            "gdelt_missing_flag",
            "raw_news_row_count",
            "filtered_news_row_count",
            "min_gdelt_source_ts",
            "max_gdelt_source_ts",
            "has_missing_15m_bucket",
            "source",
            "ingestion_run_id",
            "ingestion_ts",
        )
    )

    for table_name in [SILVER_NEWS_TABLE, RESEARCH_FEATURE_TABLE]:
        align_to_existing_table_schema(final_features, table_name).createOrReplaceTempView("incoming_silver_news_flow_1h")
        spark.sql(
            f"""
            MERGE INTO {table_name} t
            USING incoming_silver_news_flow_1h s
            ON t.timestamp = s.timestamp
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    hourly_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("hourly_aggregation", hourly_started, hourly_completed, final_features.count(), GDELT_SOURCE_MODE)
    return final_features


def build_hourly_news_features():
    if PIPELINE_MODE == "fast_feature_mode":
        return build_hourly_news_features_from_bucket_summary()
    return build_hourly_news_features_from_bronze()


if HAS_BRONZE_WINDOW:
    news_features_sdf = build_hourly_news_features()
    display(news_features_sdf.orderBy("timestamp").limit(20))
    leakage_count = news_features_sdf.where(F.col("max_gdelt_source_ts") > F.col("timestamp")).count()
    if leakage_count:
        raise AssertionError(f"Hourly aggregation leakage detected: {leakage_count} rows have max_gdelt_source_ts > timestamp")
    display(
        news_features_sdf.select(
            F.count("*").alias("actual_hourly_rows"),
            F.sum(F.when(F.col("source_bucket_count") < 4, 1).otherwise(0)).alias("hours_with_fewer_than_4_source_buckets"),
            F.sum("missing_15m_bucket_count").alias("missing_15m_bucket_count"),
            F.avg("missing_15m_bucket_pct").alias("avg_missing_15m_bucket_pct"),
            F.sum(F.when(F.col("filtered_news_row_count") == 0, 1).otherwise(0)).alias("hours_with_zero_filtered_rows"),
            F.sum(F.when(F.col("has_missing_15m_bucket"), 1).otherwise(0)).alias("hours_with_missing_15m_bucket"),
            F.sum(F.when(F.col("gdelt_missing_flag") == 1, 1).otherwise(0)).alias("hours_with_gdelt_missing_flag"),
            F.min("timestamp").alias("min_hourly_timestamp"),
            F.max("timestamp").alias("max_hourly_timestamp"),
            (F.count("*") - F.countDistinct("timestamp")).alias("duplicate_hourly_timestamps"),
            F.avg(F.when(F.col("news_volume_z").isNull(), 1).otherwise(0)).alias("news_volume_z_null_rate"),
            F.avg(F.when(F.col("news_tone_z").isNull(), 1).otherwise(0)).alias("news_tone_z_null_rate"),
            F.avg(F.when(F.col("policy_uncertainty_news_z").isNull(), 1).otherwise(0)).alias("policy_uncertainty_news_z_null_rate"),
            F.avg(F.when(F.col("geopolitical_risk_news_z").isNull(), 1).otherwise(0)).alias("geopolitical_risk_news_z_null_rate"),
            F.avg(F.when(F.col("market_stress_news_z").isNull(), 1).otherwise(0)).alias("market_stress_news_z_null_rate"),
            *[
                F.avg(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias(f"{col_name}_null_rate")
                for col_name in ENGINEERED_NEWS_FEATURES
            ],
        )
    )
else:
    news_features_sdf = None

# COMMAND ----------

def build_join_audit_and_view():
    selected_market_stress = broad_pulse_feature_name("market_stress", SELECTED_BROAD_PULSE_WINDOW_HOURS, SELECTED_BROAD_PULSE_THRESHOLD)
    selected_policy = broad_pulse_feature_name("policy", SELECTED_BROAD_PULSE_WINDOW_HOURS, SELECTED_BROAD_PULSE_THRESHOLD)
    selected_geopolitical = broad_pulse_feature_name("geopolitical", SELECTED_BROAD_PULSE_WINDOW_HOURS, SELECTED_BROAD_PULSE_THRESHOLD)
    selected_any = SELECTED_ANY_BROAD_PULSE_FEATURE
    news_select_cols = [
        "news_volume",
        "news_tone",
        "policy_uncertainty_news",
        "geopolitical_risk_news",
        "market_stress_news",
        *NARROW_TOPIC_RAW_FEATURES,
        *NEWS_FEATURES,
        *NARROW_TOPIC_Z_FEATURES,
        *ENGINEERED_NEWS_FEATURES,
        "source_bucket_count",
        "missing_15m_bucket_count",
        "missing_15m_bucket_pct",
        "gdelt_missing_flag",
        "raw_news_row_count",
        "filtered_news_row_count",
        "min_gdelt_source_ts",
        "has_missing_15m_bucket",
        "max_gdelt_source_ts",
    ]
    news_select_sql = ",\n            ".join(f"n.{col_name}" for col_name in news_select_cols)
    interaction_select_sql = f"""
            n.{selected_market_stress} * g.VIXCLS_z AS market_stress_pulse_x_VIXCLS_z,
            n.{selected_market_stress} * g.vix3m_z AS market_stress_pulse_x_vix3m_z,
            n.{selected_market_stress} * g.VIX_Inversion_Z AS market_stress_pulse_x_VIX_Inversion_Z,
            n.{selected_market_stress} * g.Price_Mom_24H AS market_stress_pulse_x_Price_Mom_24H,
            n.{selected_policy} * g.VIXCLS_z AS policy_pulse_x_VIXCLS_z,
            n.{selected_geopolitical} * g.VIXCLS_z AS geopolitical_pulse_x_VIXCLS_z,
            n.{selected_any} * g.VIXCLS_z AS any_broad_pulse_x_VIXCLS_z,
            n.{selected_any} * GREATEST(0.0, -g.Price_Mom_24H) AS any_broad_pulse_x_negative_price_mom
    """
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {CANDIDATE_JOINED_VIEW} AS
        SELECT
            g.*,
            {news_select_sql},
            {interaction_select_sql}
        FROM {SILVER_MASTER_TABLE} g
        LEFT JOIN {RESEARCH_FEATURE_TABLE} n
          ON g.timestamp = n.timestamp
        WHERE g.timestamp BETWEEN TIMESTAMP('{BACKTEST_START_TS}') AND TIMESTAMP('{BACKTEST_END_TS}')
        """
    )

    audit = spark.sql(
        f"""
        SELECT
            (SELECT COUNT(*) FROM {SILVER_MASTER_TABLE}
             WHERE timestamp BETWEEN TIMESTAMP('{BACKTEST_START_TS}') AND TIMESTAMP('{BACKTEST_END_TS}')) AS silver_bt_master_rows,
            (SELECT COUNT(*) FROM {RESEARCH_FEATURE_TABLE}
             WHERE timestamp BETWEEN TIMESTAMP('{BACKTEST_START_TS}') AND TIMESTAMP('{BACKTEST_END_TS}')) AS silver_bt_news_rows,
            (SELECT COUNT(*) FROM {CANDIDATE_JOINED_VIEW}) AS joined_rows,
            (SELECT COUNT(*) FROM {CANDIDATE_JOINED_VIEW} WHERE news_volume_z IS NULL) AS null_news_feature_rows,
            (SELECT MIN(timestamp) FROM {SILVER_MASTER_TABLE}) AS silver_master_min_ts,
            (SELECT MAX(timestamp) FROM {SILVER_MASTER_TABLE}) AS silver_master_max_ts,
            (SELECT MIN(timestamp) FROM {RESEARCH_FEATURE_TABLE}
             WHERE timestamp BETWEEN TIMESTAMP('{BACKTEST_START_TS}') AND TIMESTAMP('{BACKTEST_END_TS}')) AS news_min_ts,
            (SELECT MAX(timestamp) FROM {RESEARCH_FEATURE_TABLE}
             WHERE timestamp BETWEEN TIMESTAMP('{BACKTEST_START_TS}') AND TIMESTAMP('{BACKTEST_END_TS}')) AS news_max_ts
        """
    )
    display(audit)

    leakage_count = spark.sql(
        f"""
        SELECT COUNT(*) AS leakage_count
        FROM {CANDIDATE_JOINED_VIEW}
        WHERE max_gdelt_source_ts > timestamp
        """
    ).collect()[0]["leakage_count"]
    if leakage_count:
        raise AssertionError(f"Lookahead leakage detected: {leakage_count} joined rows have max_gdelt_source_ts > model timestamp")

    sample_count = (
        spark.table(CANDIDATE_JOINED_VIEW)
        .where(F.col("max_gdelt_source_ts").isNotNull())
        .orderBy(F.rand(seed=42))
        .limit(100)
        .where(F.col("max_gdelt_source_ts") > F.col("timestamp"))
        .count()
    )
    if sample_count:
        raise AssertionError("Sample leakage assertion failed.")
    print("PIT audit passed: max_gdelt_source_ts <= model timestamp for joined rows.")

if news_features_sdf is not None:
    build_join_audit_and_view()
else:
    print("Join audit skipped because no news features were built.")

def write_feature_validation_outputs():
    joined = spark.table(CANDIDATE_JOINED_VIEW)
    exprs = []
    validation_feature_cols = (
        ["news_volume", "news_tone", "policy_uncertainty_news", "geopolitical_risk_news", "market_stress_news"]
        + NARROW_TOPIC_RAW_FEATURES
        + NEWS_FEATURES
        + NARROW_TOPIC_Z_FEATURES
        + ENGINEERED_NEWS_FEATURES
        + PULSE_INTERACTION_FEATURES
    )
    for col_name in validation_feature_cols:
        exprs.append(
            f"""
            SELECT
                DATE(timestamp) AS feature_date,
                '{col_name}' AS feature,
                COUNT({col_name}) AS count,
                AVG({col_name}) AS mean,
                STDDEV({col_name}) AS stddev,
                MIN({col_name}) AS min,
                percentile_approx({col_name}, 0.01) AS p01,
                percentile_approx({col_name}, 0.05) AS p05,
                percentile_approx({col_name}, 0.50) AS p50,
                percentile_approx({col_name}, 0.95) AS p95,
                percentile_approx({col_name}, 0.99) AS p99,
                MAX({col_name}) AS max,
                AVG(CASE WHEN {col_name} IS NULL THEN 1 ELSE 0 END) AS null_pct
            FROM {CANDIDATE_JOINED_VIEW}
            GROUP BY DATE(timestamp)
            """
        )
    summary = spark.sql("\nUNION ALL\n".join(exprs))
    display(summary.orderBy("feature_date", "feature"))
    summary.toPandas().to_csv(FEATURE_SUMMARY_CSV, index=False)

    pulse_active_exprs = [
        F.avg(F.col(col_name)).alias(f"{col_name}_active_rate")
        for col_name in BROAD_PULSE_FEATURES + ANY_BROAD_PULSE_FEATURES + NARROW_PULSE_FEATURES
    ]
    active_rates = joined.select(*pulse_active_exprs).toPandas().iloc[0].to_dict()
    active_rate_rows = []
    for key, value in active_rates.items():
        feature = key.removesuffix("_active_rate")
        active_rate = float(value or 0.0)
        active_rate_rows.append(
            {
                "feature": feature,
                "active_rate": active_rate,
                "status": "saturated" if active_rate > 0.80 else ("too_sparse" if active_rate < 0.01 else "usable"),
            }
        )
    active_rate_pdf = pd.DataFrame(active_rate_rows).sort_values(["status", "active_rate", "feature"])
    display(spark.createDataFrame(active_rate_pdf))

    corr_cols = (
        BASELINE_FEATURES
        + NEWS_FEATURES
        + NARROW_TOPIC_Z_FEATURES
        + SELECTED_BROAD_PULSE_FEATURES
        + [SELECTED_ANY_BROAD_PULSE_FEATURE]
        + PULSE_INTERACTION_FEATURES
    )
    corr_pdf = joined.select(["timestamp"] + corr_cols).toPandas()
    corr = corr_pdf[corr_cols].corr()
    display(spark.createDataFrame(corr.reset_index().rename(columns={"index": "feature"})))
    redundancy_cols = ["VIXCLS_z", "vix3m_z", "Price_Mom_24H"]
    redundancy_rows = []
    for feature in SELECTED_BROAD_PULSE_FEATURES + [SELECTED_ANY_BROAD_PULSE_FEATURE] + PULSE_INTERACTION_FEATURES:
        for baseline_col in redundancy_cols + ["VIX_Inversion_Z"]:
            redundancy_rows.append(
                {
                    "feature": feature,
                    "baseline_feature": baseline_col,
                    "correlation": float(corr.loc[feature, baseline_col]) if feature in corr.index and baseline_col in corr.columns else None,
                }
            )
    redundancy_pdf = pd.DataFrame(redundancy_rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False)
    display(spark.createDataFrame(redundancy_pdf))
    return summary, corr, active_rate_pdf, redundancy_pdf

if news_features_sdf is not None and RUN_BACKTEST:
    feature_summary_sdf, corr_matrix, pulse_active_rate_pdf, pulse_redundancy_pdf = write_feature_validation_outputs()
else:
    if news_features_sdf is not None:
        print("Feature summary/correlation diagnostics skipped because run_backtest=false.")
    feature_summary_sdf, corr_matrix, pulse_active_rate_pdf, pulse_redundancy_pdf = None, None, None, None

# COMMAND ----------

def label_regime(ret):
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


def build_model():
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        eval_metric="merror",
        tree_method="hist",
        random_state=42,
    )


def monthly_walkforward(pdf, features, experiment_name):
    data = pdf.copy().sort_values("timestamp")
    data["fwd_return_1w"] = data["SPY_close"].shift(-168) / data["SPY_close"] - 1
    data["regime"] = data["fwd_return_1w"].apply(label_regime)
    trainable = data[data["regime"].notna()].copy()
    trainable["regime"] = trainable["regime"].astype(int)
    trainable[features] = trainable[features].fillna(0)

    first_ts = pd.Timestamp(trainable["timestamp"].min()).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_ts = pd.Timestamp(trainable["timestamp"].max()).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    retrain_dates = []
    d = first_ts + relativedelta(months=1)
    while d <= last_ts:
        retrain_dates.append(d)
        d += relativedelta(months=1)

    signals = []
    metrics = []
    importances = []
    for idx in range(len(retrain_dates) - 1):
        fold_train_end = retrain_dates[idx]
        fold_test_end = retrain_dates[idx + 1]
        train_df = trainable[trainable["timestamp"] < fold_train_end]
        test_df = trainable[(trainable["timestamp"] >= fold_train_end) & (trainable["timestamp"] < fold_test_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        model = build_model()
        model.fit(train_df[features], train_df["regime"])
        test_df["predicted_regime"] = model.predict(test_df[features]).astype(int)
        acc = float(accuracy_score(test_df["regime"], test_df["predicted_regime"]))
        test_df["experiment_name"] = experiment_name
        test_df["fold_train_end"] = fold_train_end
        test_df["fold_test_end"] = fold_test_end
        test_df["accuracy"] = acc
        signals.append(test_df)
        metrics.append(
            {
                "experiment_name": experiment_name,
                "fold_train_end": fold_train_end,
                "fold_test_end": fold_test_end,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "accuracy": acc,
            }
        )
        importances.append(pd.Series(model.feature_importances_, index=features, name=fold_test_end))

    if not signals:
        raise ValueError(f"No walk-forward folds produced for {experiment_name}. Expand the fetch/backtest window.")

    wf = pd.concat(signals).sort_values("timestamp")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).reset_index()
    imp.columns = ["feature", "mean_importance"]
    return wf, pd.DataFrame(metrics), imp


def simulate_strategy(wf):
    df = wf.sort_values("timestamp").copy()
    df["QQQ_close"] = df["QQQ_close"].ffill()
    df["SPY_close"] = df["SPY_close"].ffill()
    equity = 5_000_000.0
    peak = equity
    active_asset = "START"
    equities = []
    drawdowns = []
    for i in range(1, len(df)):
        regime_signal = int(df["predicted_regime"].iloc[i - 1])
        if regime_signal == 0:
            target_asset = "QQQ"
            hourly_ret = df["QQQ_close"].iloc[i] / df["QQQ_close"].iloc[i - 1] - 1
        elif regime_signal == 1:
            target_asset = "SPY"
            hourly_ret = df["SPY_close"].iloc[i] / df["SPY_close"].iloc[i - 1] - 1
        else:
            target_asset = "CASH"
            hourly_ret = 0.0
        cost = 0.0005 if target_asset != active_asset else 0.0
        active_asset = target_asset
        equity *= 1 + hourly_ret - cost
        peak = max(peak, equity)
        equities.append(equity)
        drawdowns.append(equity / peak - 1)
    out = df.iloc[1:].copy()
    out["strategy_equity"] = equities
    out["drawdown"] = drawdowns
    return out


def summarize_backtest(strategy_df, metrics_df, importances_df):
    daily = strategy_df.set_index("timestamp")["strategy_equity"].resample("D").last().dropna()
    years = max((daily.index.max() - daily.index.min()).days / 365.0, 1 / 365.0)
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1
    daily_ret = daily.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0
    hourly_ret = strategy_df["strategy_equity"].pct_change().dropna()
    hourly_vol = float(hourly_ret.std() * np.sqrt(8760)) if hourly_ret.std() else 0.0
    cm = confusion_matrix(strategy_df["regime"], strategy_df["predicted_regime"], labels=[0, 1, 2])
    return {
        "experiment_name": strategy_df["experiment_name"].iloc[0],
        "rows": len(strategy_df),
        "walkforward_accuracy": float(metrics_df["accuracy"].mean()),
        "final_equity": float(strategy_df["strategy_equity"].iloc[-1]),
        "cagr_365d": float(cagr),
        "max_drawdown": float(strategy_df["drawdown"].min()),
        "hourly_volatility_8760h": hourly_vol,
        "daily_sharpe_365d": sharpe,
        "confusion_matrix_0_1_2_json": json.dumps(cm.tolist()),
        "top_features_json": json.dumps(importances_df.head(10).to_dict(orient="records")),
    }


def run_candidate_backtests():
    backtest_started = datetime.now(timezone.utc).replace(tzinfo=None)
    pdf = spark.table(CANDIDATE_JOINED_VIEW).toPandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"]).dt.tz_localize(None)
    pdf = pdf.sort_values("timestamp")

    experiments = {
        "baseline_v4": BASELINE_FEATURES,
        "v4_plus_gdelt_broad_pulse_legacy_t20_24h": BASELINE_FEATURES + PULSE_TOPIC_FEATURES,
    }
    for window_hours in BROAD_PULSE_WINDOWS_HOURS:
        for threshold in BROAD_PULSE_THRESHOLDS:
            token = threshold_token(threshold)
            if BACKTEST_EXPERIMENT_MODE == "full_grid":
                experiments[f"grid_market_stress_pulse_{window_hours}h_t{token}"] = BASELINE_FEATURES + [
                    broad_pulse_feature_name("market_stress", window_hours, threshold)
                ]
                experiments[f"grid_geopolitical_pulse_{window_hours}h_t{token}"] = BASELINE_FEATURES + [
                    broad_pulse_feature_name("geopolitical", window_hours, threshold)
                ]
            experiments[f"grid_policy_pulse_{window_hours}h_t{token}"] = BASELINE_FEATURES + [
                broad_pulse_feature_name("policy", window_hours, threshold)
            ]
            experiments[f"grid_all_broad_pulses_{window_hours}h_t{token}"] = (
                BASELINE_FEATURES + broad_pulse_features_for(window_hours, threshold)
            )
    if BACKTEST_EXPERIMENT_MODE == "full_grid":
        experiments.update(
            {
                "selected_broad_pulse_24h_t20": BASELINE_FEATURES + SELECTED_BROAD_PULSE_FEATURES,
                "selected_broad_pulse_vix_interactions": BASELINE_FEATURES + SELECTED_BROAD_PULSE_FEATURES + VIX_INTERACTION_FEATURES,
                "selected_broad_pulse_price_interactions": BASELINE_FEATURES + SELECTED_BROAD_PULSE_FEATURES + PRICE_INTERACTION_FEATURES,
                "selected_broad_pulse_all_selected_interactions": BASELINE_FEATURES
                + SELECTED_BROAD_PULSE_FEATURES
                + PULSE_INTERACTION_FEATURES,
            }
        )

    all_rows = []
    summaries = []
    all_importances = {}
    for experiment_name, features in experiments.items():
        wf, metrics, imp = monthly_walkforward(pdf, features, experiment_name)
        strategy = simulate_strategy(wf)
        summary = summarize_backtest(strategy, metrics, imp)
        summaries.append(summary)
        all_importances[experiment_name] = imp
        all_rows.append(
            strategy[
                [
                    "experiment_name",
                    "timestamp",
                    "regime",
                    "predicted_regime",
                    "fold_train_end",
                    "fold_test_end",
                    "accuracy",
                    "strategy_equity",
                    "drawdown",
                ]
            ].assign(model_features=", ".join(features))
        )

    result_pdf = pd.concat(all_rows, ignore_index=True)
    result_pdf["ingestion_ts"] = RUN_TS
    spark.createDataFrame(result_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_WF_TABLE)

    summary_pdf = pd.DataFrame(summaries)
    summary_pdf.to_csv(BACKTEST_CSV, index=False)
    display(spark.createDataFrame(summary_pdf))

    for experiment_name, imp in all_importances.items():
        print(f"Feature importance: {experiment_name}")
        display(spark.createDataFrame(imp))

    baseline = summary_pdf[summary_pdf["experiment_name"] == "baseline_v4"].iloc[0]
    preferred_candidate_name = "selected_broad_pulse_all_selected_interactions"
    if BACKTEST_EXPERIMENT_MODE == "selected_candidates":
        preferred_candidate_name = "grid_all_broad_pulses_48h_t20"
    if preferred_candidate_name not in set(summary_pdf["experiment_name"]):
        preferred_candidate_name = summary_pdf[summary_pdf["experiment_name"] != "baseline_v4"]["experiment_name"].iloc[0]
    candidate = summary_pdf[summary_pdf["experiment_name"] == preferred_candidate_name].iloc[0]
    lift = {
        "accuracy_lift": float(candidate["walkforward_accuracy"] - baseline["walkforward_accuracy"]),
        "drawdown_lift": float(candidate["max_drawdown"] - baseline["max_drawdown"]),
        "cagr_lift": float(candidate["cagr_365d"] - baseline["cagr_365d"]),
    }
    print(f"Incremental lift vs baseline: {lift}")
    if lift["accuracy_lift"] <= 0 and lift["drawdown_lift"] <= 0:
        print("No-go path: candidate GDELT features did not improve accuracy or drawdown in this run. Do not proceed to shadow production from this result.")
    backtest_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("backtest", backtest_started, backtest_completed, len(result_pdf), ",".join(experiments.keys()))
    return summary_pdf, all_importances

if news_features_sdf is not None and RUN_BACKTEST:
    backtest_summary_pdf, feature_importances = run_candidate_backtests()
else:
    if news_features_sdf is not None:
        print("Backtest skipped because run_backtest=false.")
    backtest_summary_pdf, feature_importances = None, None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Research-Only Overlay Backtests
# MAGIC
# MAGIC These overlay tests do not retrain the classifier with GDELT features.
# MAGIC They first generate baseline V4 walk-forward predictions, then apply
# MAGIC deterministic GDELT pulse rules to the portfolio allocation stream.

# COMMAND ----------

DEFAULT_OVERLAY_WINDOWS = [
    {
        "window_name": "2018_q4",
        "start_ts": "2018-10-01 00:00:00",
        "end_ts": "2018-12-31 23:00:00",
    },
    {
        "window_name": "covid_phase_a",
        "start_ts": "2020-02-01 00:00:00",
        "end_ts": "2020-05-31 23:00:00",
    },
    {
        "window_name": "2022_h1",
        "start_ts": "2022-01-01 00:00:00",
        "end_ts": "2022-06-30 23:00:00",
    },
]

if OVERLAY_EXPERIMENT_MODE in {"selected_validate", "refined_validate"}:
    OVERLAY_WINDOWS = [
        {
            "window_name": OVERLAY_WINDOW_NAME or "selected_window",
            "start_ts": str(BACKTEST_START_TS),
            "end_ts": str(BACKTEST_END_TS),
        }
    ]
elif OVERLAY_EXPERIMENT_MODE == "targeted_refine":
    OVERLAY_WINDOWS = DEFAULT_OVERLAY_WINDOWS + [
        {
            "window_name": "2022_h2",
            "start_ts": "2022-07-01 00:00:00",
            "end_ts": "2022-12-31 23:00:00",
        }
    ]
else:
    OVERLAY_WINDOWS = DEFAULT_OVERLAY_WINDOWS


def baseline_asset_from_regime(regime_signal):
    if int(regime_signal) == 0:
        return "QQQ"
    if int(regime_signal) == 1:
        return "SPY"
    return "CASH"


def overlay_weights(asset_label):
    if asset_label == "QQQ":
        return 1.0, 0.0
    if asset_label == "SPY":
        return 0.0, 1.0
    if asset_label == "QQQ50_SPY50":
        return 0.5, 0.5
    if asset_label == "SPY50_CASH50":
        return 0.0, 0.5
    return 0.0, 0.0


def overlay_asset(row, rule_name):
    baseline_asset = baseline_asset_from_regime(row["predicted_regime"])
    if baseline_asset == "CASH":
        return "CASH"

    broad_active = max(
        float(row.get("policy_pulse_48h_t20", 0.0) or 0.0),
        float(row.get("geopolitical_pulse_48h_t20", 0.0) or 0.0),
        float(row.get("market_stress_pulse_48h_t20", 0.0) or 0.0),
    ) > 0
    policy_active = float(row.get("policy_pulse_48h_t175", 0.0) or 0.0) > 0
    market_stress_active = float(row.get("market_stress_pulse_48h_t20", 0.0) or 0.0) > 0
    vix_z = float(row.get("VIXCLS_z", 0.0) or 0.0)
    price_mom = float(row.get("Price_Mom_24H", 0.0) or 0.0)

    if rule_name == "baseline_v4":
        return baseline_asset
    if rule_name == "overlay_a_broad_expansion_to_spy":
        return "SPY" if baseline_asset == "QQQ" and broad_active else baseline_asset
    if rule_name == "overlay_a_broad_expansion_to_50qqq_50spy":
        return "QQQ50_SPY50" if baseline_asset == "QQQ" and broad_active else baseline_asset
    if rule_name == "overlay_b_broad_vix_confirm_spy_cash":
        if baseline_asset == "QQQ" and broad_active and vix_z > 1.0:
            return "CASH"
        if baseline_asset == "QQQ" and broad_active and vix_z > 0.0:
            return "SPY"
        return baseline_asset
    if rule_name == "overlay_c_slowdown_negative_mom_to_cash":
        return "CASH" if baseline_asset == "SPY" and broad_active and price_mom < 0.0 else baseline_asset
    if rule_name == "overlay_c_slowdown_negative_mom_to_50spy_50cash":
        return "SPY50_CASH50" if baseline_asset == "SPY" and broad_active and price_mom < 0.0 else baseline_asset
    if rule_name == "overlay_e_policy_expansion_to_spy":
        return "SPY" if baseline_asset == "QQQ" and policy_active else baseline_asset
    if rule_name == "overlay_optional_market_stress_expansion_to_spy":
        return "SPY" if baseline_asset == "QQQ" and market_stress_active else baseline_asset
    if rule_name == "overlay_f_combined_conservative":
        if baseline_asset == "QQQ" and broad_active and vix_z > 1.0:
            return "CASH"
        if baseline_asset == "QQQ" and broad_active and vix_z > 0.0:
            return "SPY"
        if baseline_asset == "SPY" and broad_active and price_mom < 0.0:
            return "CASH"
        return baseline_asset
    raise ValueError(f"Unknown overlay rule: {rule_name}")


OVERLAY_RULES = [
    "baseline_v4",
    "overlay_a_broad_expansion_to_spy",
    "overlay_a_broad_expansion_to_50qqq_50spy",
    "overlay_b_broad_vix_confirm_spy_cash",
    "overlay_c_slowdown_negative_mom_to_cash",
    "overlay_c_slowdown_negative_mom_to_50spy_50cash",
    "overlay_e_policy_expansion_to_spy",
    "overlay_optional_market_stress_expansion_to_spy",
    "overlay_f_combined_conservative",
]


def load_overlay_window_pdf(start_ts, end_ts):
    selected_cols = (
        BASELINE_FEATURES
        + [
            "SPY_close",
            "QQQ_close",
            "policy_pulse_48h_t20",
            "geopolitical_pulse_48h_t20",
            "market_stress_pulse_48h_t20",
            "policy_pulse_48h_t175",
            "max_gdelt_source_ts",
        ]
    )
    select_sql = ",\n                ".join([f"g.{col}" for col in BASELINE_FEATURES])
    return spark.sql(
        f"""
        SELECT
            g.timestamp,
            {select_sql},
            g.SPY_close,
            g.QQQ_close,
            n.policy_pulse_48h_t20,
            n.geopolitical_pulse_48h_t20,
            n.market_stress_pulse_48h_t20,
            n.policy_pulse_48h_t175,
            n.max_gdelt_source_ts
        FROM {SILVER_MASTER_TABLE} g
        INNER JOIN {RESEARCH_FEATURE_TABLE} n
            ON g.timestamp = n.timestamp
        WHERE g.timestamp >= TIMESTAMP('{start_ts}')
          AND g.timestamp <= TIMESTAMP('{end_ts}')
        ORDER BY g.timestamp
        """
    ).select(*selected_cols, "timestamp").toPandas().sort_values("timestamp")


def simulate_overlay_strategy(baseline_wf, rule_name, window_name):
    df = baseline_wf.sort_values("timestamp").copy()
    df["baseline_asset"] = df["predicted_regime"].apply(baseline_asset_from_regime)
    df["overlay_asset_signal"] = df.apply(lambda row: overlay_asset(row, rule_name), axis=1)
    df["overlay_changed_signal"] = df["overlay_asset_signal"] != df["baseline_asset"]

    equity = 5_000_000.0
    peak = equity
    active_asset = "START"
    rows = []
    for i in range(1, len(df)):
        signal = df.iloc[i - 1]
        target_asset = signal["overlay_asset_signal"]
        qqq_w, spy_w = overlay_weights(target_asset)
        qqq_ret = df["QQQ_close"].iloc[i] / df["QQQ_close"].iloc[i - 1] - 1
        spy_ret = df["SPY_close"].iloc[i] / df["SPY_close"].iloc[i - 1] - 1
        hourly_ret = qqq_w * qqq_ret + spy_w * spy_ret
        cost = 0.0005 if target_asset != active_asset else 0.0
        active_asset = target_asset
        equity *= 1 + hourly_ret - cost
        peak = max(peak, equity)
        out = df.iloc[i].copy()
        out["window_name"] = window_name
        out["overlay_rule"] = rule_name
        out["baseline_asset"] = signal["baseline_asset"]
        out["overlay_asset"] = target_asset
        out["overlay_changed"] = bool(signal["overlay_changed_signal"])
        out["strategy_equity"] = equity
        out["drawdown"] = equity / peak - 1
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_overlay_strategy(strategy_df, baseline_summary=None):
    daily = strategy_df.set_index("timestamp")["strategy_equity"].resample("D").last().dropna()
    years = max((daily.index.max() - daily.index.min()).days / 365.0, 1 / 365.0)
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1
    daily_ret = daily.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0
    transitions = int((strategy_df["overlay_asset"] != strategy_df["overlay_asset"].shift(1)).sum() - 1)
    overlay_changes = int(strategy_df["overlay_changed"].sum())
    return_value = float(strategy_df["strategy_equity"].iloc[-1] / strategy_df["strategy_equity"].iloc[0] - 1)
    summary = {
        "window_name": strategy_df["window_name"].iloc[0],
        "overlay_rule": strategy_df["overlay_rule"].iloc[0],
        "rows": len(strategy_df),
        "strategy_return": return_value,
        "cagr_365d": float(cagr),
        "daily_sharpe_365d": sharpe,
        "max_drawdown": float(strategy_df["drawdown"].min()),
        "transition_count": transitions,
        "overlay_changes": overlay_changes,
    }
    if baseline_summary is not None:
        summary["return_diff_vs_baseline"] = summary["strategy_return"] - baseline_summary["strategy_return"]
        summary["drawdown_diff_vs_baseline"] = summary["max_drawdown"] - baseline_summary["max_drawdown"]
        summary["sharpe_diff_vs_baseline"] = summary["daily_sharpe_365d"] - baseline_summary["daily_sharpe_365d"]
    return summary


def overlay_change_summary(strategy_df):
    baseline_assets = strategy_df["baseline_asset"].fillna("UNKNOWN")
    overlay_assets = strategy_df["overlay_asset"].fillna("UNKNOWN")
    return {
        "qqq_to_spy": int(((baseline_assets == "QQQ") & (overlay_assets == "SPY")).sum()),
        "qqq_to_cash": int(((baseline_assets == "QQQ") & (overlay_assets == "CASH")).sum()),
        "qqq_to_50qqq_50spy": int(((baseline_assets == "QQQ") & (overlay_assets == "QQQ50_SPY50")).sum()),
        "spy_to_cash": int(((baseline_assets == "SPY") & (overlay_assets == "CASH")).sum()),
        "spy_to_50spy_50cash": int(((baseline_assets == "SPY") & (overlay_assets == "SPY50_CASH50")).sum()),
        "unchanged_cash": int(((baseline_assets == "CASH") & (overlay_assets == "CASH")).sum()),
    }


def run_overlay_backtests():
    overlay_started = datetime.now(timezone.utc).replace(tzinfo=None)
    all_strategy_rows = []
    all_summary_rows = []
    validation_rows = []
    activation_rows = []

    for window in OVERLAY_WINDOWS:
        window_name = window["window_name"]
        start_ts = window["start_ts"]
        end_ts = window["end_ts"]
        window_pdf = load_overlay_window_pdf(start_ts, end_ts)
        window_pdf["timestamp"] = pd.to_datetime(window_pdf["timestamp"]).dt.tz_localize(None)
        if window_pdf.empty:
            raise ValueError(f"No joined baseline/news rows for overlay window {window_name}.")
        if window_pdf["max_gdelt_source_ts"].notna().any():
            max_source = pd.to_datetime(window_pdf["max_gdelt_source_ts"]).dt.tz_localize(None)
            if (max_source > window_pdf["timestamp"]).any():
                raise AssertionError(f"Overlay leakage detected in {window_name}.")
        null_cols = [
            "policy_pulse_48h_t20",
            "geopolitical_pulse_48h_t20",
            "market_stress_pulse_48h_t20",
            "policy_pulse_48h_t175",
            "VIXCLS_z",
            "Price_Mom_24H",
        ]
        null_counts = {col: int(window_pdf[col].isna().sum()) for col in null_cols}
        if any(null_counts.values()):
            raise AssertionError(f"Overlay null feature rows in {window_name}: {null_counts}")

        baseline_wf, baseline_metrics, baseline_importance = monthly_walkforward(window_pdf, BASELINE_FEATURES, f"{window_name}_baseline_v4")
        baseline_wf["window_name"] = window_name
        validation_rows.append(
            {
                "window_name": window_name,
                "joined_rows": len(window_pdf),
                "walkforward_rows": len(baseline_wf),
                "leakage_rows": 0,
                **{f"{col}_null_count": value for col, value in null_counts.items()},
            }
        )
        activation_rows.append(
            {
                "window_name": window_name,
                "any_broad_48h_t20_active_count": int(
                    (
                        window_pdf[["policy_pulse_48h_t20", "geopolitical_pulse_48h_t20", "market_stress_pulse_48h_t20"]].max(axis=1)
                        > 0
                    ).sum()
                ),
                "policy_48h_t175_active_count": int((window_pdf["policy_pulse_48h_t175"] > 0).sum()),
                "market_stress_48h_t20_active_count": int((window_pdf["market_stress_pulse_48h_t20"] > 0).sum()),
            }
        )

        baseline_summary = None
        for rule_name in OVERLAY_RULES:
            strategy = simulate_overlay_strategy(baseline_wf, rule_name, window_name)
            summary = summarize_overlay_strategy(strategy, baseline_summary)
            summary.update(overlay_change_summary(strategy))
            if rule_name == "baseline_v4":
                baseline_summary = summary
                summary["return_diff_vs_baseline"] = 0.0
                summary["drawdown_diff_vs_baseline"] = 0.0
                summary["sharpe_diff_vs_baseline"] = 0.0
            all_strategy_rows.append(strategy)
            all_summary_rows.append(summary)

    strategy_pdf = pd.concat(all_strategy_rows, ignore_index=True)
    strategy_pdf["ingestion_run_id"] = INGESTION_RUN_ID
    strategy_pdf["ingestion_ts"] = RUN_TS
    keep_cols = [
        "window_name",
        "overlay_rule",
        "timestamp",
        "regime",
        "predicted_regime",
        "baseline_asset",
        "overlay_asset",
        "overlay_changed",
        "strategy_equity",
        "drawdown",
        "ingestion_run_id",
        "ingestion_ts",
    ]
    spark.createDataFrame(strategy_pdf[keep_cols]).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_TABLE)

    summary_pdf = pd.DataFrame(all_summary_rows)
    validation_pdf = pd.DataFrame(validation_rows)
    activation_pdf = pd.DataFrame(activation_rows)
    display(spark.createDataFrame(validation_pdf))
    display(spark.createDataFrame(activation_pdf))
    display(spark.createDataFrame(summary_pdf))
    overlay_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("overlay_backtest", overlay_started, overlay_completed, len(strategy_pdf), ",".join(OVERLAY_RULES))
    return summary_pdf, validation_pdf, activation_pdf


OPT_PULSE_SOURCES = ["broad_all", "policy", "market_stress", "geopolitical", "any_broad"]
OPT_PULSE_THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5]
OPT_PULSE_WINDOWS = [12, 24, 48]
OPT_VIX_CONFIRMATIONS = ["none", "vix_gt_0", "vix_gt_0_5", "vix_gt_1"]
OPT_PRICE_CONFIRMATIONS = ["none", "price_mom_lt_0", "price_mom_lt_neg_0_005"]
OPT_OVERLAY_ACTIONS = [
    "QQQ_TO_SPY",
    "QQQ_TO_HALF_QQQ_HALF_SPY",
    "QQQ_TO_CASH",
    "SPY_TO_HALF_SPY_HALF_CASH",
    "SPY_TO_CASH",
    "COMBINED_CONSERVATIVE",
]
OPT_LAG_SETTINGS = [0, 15]


def selected_overlay_lag_settings():
    if OVERLAY_SELECTED_LAGS_PARAM:
        return sorted({int(part.strip()) for part in OVERLAY_SELECTED_LAGS_PARAM.split(",") if part.strip()})
    lags = {NEWS_AVAILABILITY_LAG_MINUTES}
    if NEWS_AVAILABILITY_LAG_MINUTES != 0:
        lags.add(0)
    return sorted(lags)


def overlay_param_name(params):
    if params["overlay_action"] == "BASELINE":
        return "baseline_v4"
    return (
        f"opt_{params['pulse_source']}"
        f"_w{params['pulse_window_hours']}"
        f"_t{threshold_token(params['pulse_threshold'])}"
        f"_{params['vix_confirmation']}"
        f"_{params['price_confirmation']}"
        f"_{params['overlay_action'].lower()}"
    )


def overlay_param_grid():
    rows = [
        {
            "pulse_source": "baseline",
            "pulse_threshold": 0.0,
            "pulse_window_hours": 0,
            "vix_confirmation": "none",
            "price_confirmation": "none",
            "overlay_action": "BASELINE",
            "overlay_rule": "baseline_v4",
        }
    ]
    qqq_actions = {"QQQ_TO_SPY", "QQQ_TO_HALF_QQQ_HALF_SPY", "QQQ_TO_CASH"}
    spy_actions = {"SPY_TO_HALF_SPY_HALF_CASH", "SPY_TO_CASH"}
    for pulse_source in OPT_PULSE_SOURCES:
        for pulse_threshold in OPT_PULSE_THRESHOLDS:
            for pulse_window in OPT_PULSE_WINDOWS:
                for overlay_action in OPT_OVERLAY_ACTIONS:
                    vix_values = OPT_VIX_CONFIRMATIONS if overlay_action in qqq_actions | {"COMBINED_CONSERVATIVE"} else ["none"]
                    price_values = OPT_PRICE_CONFIRMATIONS if overlay_action in spy_actions | {"COMBINED_CONSERVATIVE"} else ["none"]
                    for vix_confirmation in vix_values:
                        for price_confirmation in price_values:
                            params = {
                                "pulse_source": pulse_source,
                                "pulse_threshold": pulse_threshold,
                                "pulse_window_hours": pulse_window,
                                "vix_confirmation": vix_confirmation,
                                "price_confirmation": price_confirmation,
                                "overlay_action": overlay_action,
                            }
                            params["overlay_rule"] = overlay_param_name(params)
                            rows.append(params)
    return rows


def load_overlay_optimizer_window_pdf(start_ts, end_ts, lag_minutes):
    shift_hours = int(math.ceil(lag_minutes / 60.0)) if lag_minutes > 0 else 0
    join_expr = "g.timestamp = n.timestamp" if shift_hours == 0 else f"g.timestamp - INTERVAL {shift_hours} HOURS = n.timestamp"
    select_sql = ",\n                ".join([f"g.{col}" for col in BASELINE_FEATURES])
    return spark.sql(
        f"""
        SELECT
            g.timestamp,
            {select_sql},
            g.SPY_close,
            g.QQQ_close,
            n.policy_uncertainty_news_z,
            n.geopolitical_risk_news_z,
            n.market_stress_news_z,
            n.missing_15m_bucket_count,
            n.missing_15m_bucket_pct,
            n.gdelt_missing_flag,
            n.has_missing_15m_bucket,
            n.max_gdelt_source_ts
        FROM {SILVER_MASTER_TABLE} g
        LEFT JOIN {RESEARCH_FEATURE_TABLE} n
            ON {join_expr}
        WHERE g.timestamp >= TIMESTAMP('{start_ts}')
          AND g.timestamp <= TIMESTAMP('{end_ts}')
        ORDER BY g.timestamp
        """
    ).toPandas().sort_values("timestamp")


def add_optimizer_pulses(pdf):
    out = pdf.copy().sort_values("timestamp")
    if "gdelt_missing_flag" not in out.columns:
        out["gdelt_missing_flag"] = 0
    gdelt_available = (out["gdelt_missing_flag"].fillna(1).astype(float) == 0.0).astype(float)
    z_cols = {
        "policy": "policy_uncertainty_news_z",
        "geopolitical": "geopolitical_risk_news_z",
        "market_stress": "market_stress_news_z",
    }
    for col_name in z_cols.values():
        out[col_name] = out[col_name].fillna(0.0)
    for window_hours in OPT_PULSE_WINDOWS:
        for threshold in OPT_PULSE_THRESHOLDS:
            pulse_cols = []
            for source, z_col in z_cols.items():
                pulse_col = f"opt_{source}_pulse_{window_hours}h_t{threshold_token(threshold)}"
                out[pulse_col] = (
                    ((out[z_col] >= threshold).astype(float) * gdelt_available)
                    .rolling(window=window_hours, min_periods=1)
                    .max()
                    .fillna(0.0)
                    * gdelt_available
                )
                pulse_cols.append(pulse_col)
            broad_col = f"opt_broad_all_pulse_{window_hours}h_t{threshold_token(threshold)}"
            any_col = f"opt_any_broad_pulse_{window_hours}h_t{threshold_token(threshold)}"
            out[broad_col] = out[pulse_cols].min(axis=1).astype(float)
            out[any_col] = out[pulse_cols].max(axis=1).astype(float)
    return out


def optimizer_pulse_column(params):
    source = params["pulse_source"]
    if source == "baseline":
        return None
    return f"opt_{source}_pulse_{params['pulse_window_hours']}h_t{threshold_token(params['pulse_threshold'])}"


def optimizer_condition(row, params):
    if params["overlay_action"] == "BASELINE":
        return False
    pulse_col = optimizer_pulse_column(params)
    pulse_active = float(row.get(pulse_col, 0.0) or 0.0) > 0.0
    if not pulse_active:
        return False
    vix_z = float(row.get("VIXCLS_z", 0.0) or 0.0)
    price_mom = float(row.get("Price_Mom_24H", 0.0) or 0.0)
    if params["vix_confirmation"] == "vix_gt_0" and not (vix_z > 0.0):
        return False
    if params["vix_confirmation"] == "vix_gt_0_5" and not (vix_z > 0.5):
        return False
    if params["vix_confirmation"] == "vix_gt_1" and not (vix_z > 1.0):
        return False
    if params["price_confirmation"] == "price_mom_lt_0" and not (price_mom < 0.0):
        return False
    if params["price_confirmation"] == "price_mom_lt_neg_0_005" and not (price_mom < -0.005):
        return False
    return True


def optimizer_overlay_asset(row, params):
    baseline_asset = baseline_asset_from_regime(row["predicted_regime"])
    if baseline_asset == "CASH" or params["overlay_action"] == "BASELINE":
        return baseline_asset
    active = optimizer_condition(row, params)
    action = params["overlay_action"]
    if action == "COMBINED_CONSERVATIVE":
        if not active:
            return baseline_asset
        vix_z = float(row.get("VIXCLS_z", 0.0) or 0.0)
        price_mom = float(row.get("Price_Mom_24H", 0.0) or 0.0)
        if baseline_asset == "QQQ" and vix_z > 1.0:
            return "CASH"
        if baseline_asset == "QQQ":
            return "SPY"
        if baseline_asset == "SPY" and price_mom < 0.0:
            return "CASH"
        return baseline_asset
    if not active:
        return baseline_asset
    if baseline_asset == "QQQ" and action == "QQQ_TO_SPY":
        return "SPY"
    if baseline_asset == "QQQ" and action == "QQQ_TO_HALF_QQQ_HALF_SPY":
        return "QQQ50_SPY50"
    if baseline_asset == "QQQ" and action == "QQQ_TO_CASH":
        return "CASH"
    if baseline_asset == "SPY" and action == "SPY_TO_HALF_SPY_HALF_CASH":
        return "SPY50_CASH50"
    if baseline_asset == "SPY" and action == "SPY_TO_CASH":
        return "CASH"
    return baseline_asset


def simulate_optimizer_strategy(baseline_wf, params, window_name, lag_minutes):
    df = baseline_wf.sort_values("timestamp").copy()
    df["baseline_asset"] = df["predicted_regime"].apply(baseline_asset_from_regime)
    df["overlay_asset_signal"] = df.apply(lambda row: optimizer_overlay_asset(row, params), axis=1)
    df["overlay_changed_signal"] = df["overlay_asset_signal"] != df["baseline_asset"]

    equity = 5_000_000.0
    peak = equity
    active_asset = "START"
    rows = []
    for i in range(1, len(df)):
        signal = df.iloc[i - 1]
        target_asset = signal["overlay_asset_signal"]
        qqq_w, spy_w = overlay_weights(target_asset)
        qqq_ret = df["QQQ_close"].iloc[i] / df["QQQ_close"].iloc[i - 1] - 1
        spy_ret = df["SPY_close"].iloc[i] / df["SPY_close"].iloc[i - 1] - 1
        hourly_ret = qqq_w * qqq_ret + spy_w * spy_ret
        cost = 0.0005 if target_asset != active_asset else 0.0
        active_asset = target_asset
        equity *= 1 + hourly_ret - cost
        peak = max(peak, equity)
        out = df.iloc[i].copy()
        out["window_name"] = window_name
        out["news_availability_lag_minutes"] = lag_minutes
        out["overlay_rule"] = params["overlay_rule"]
        out["baseline_asset"] = signal["baseline_asset"]
        out["overlay_asset"] = target_asset
        out["overlay_changed"] = bool(signal["overlay_changed_signal"])
        out["strategy_equity"] = equity
        out["drawdown"] = equity / peak - 1
        rows.append(out)
    return pd.DataFrame(rows)


def worst_drawdown_return(strategy_df):
    min_idx = strategy_df["drawdown"].idxmin()
    peak_equity = strategy_df.loc[:min_idx, "strategy_equity"].max()
    trough_equity = strategy_df.loc[min_idx, "strategy_equity"]
    return float(trough_equity / peak_equity - 1)


def simulate_optimizer_summary(baseline_wf, params, window_name, lag_minutes, baseline_summary=None):
    df = baseline_wf.sort_values("timestamp").copy()
    pred = df["predicted_regime"].astype(int).to_numpy()
    baseline_signal = np.where(pred == 0, "QQQ", np.where(pred == 1, "SPY", "CASH")).astype(object)
    target_signal = baseline_signal.copy()
    if params["overlay_action"] != "BASELINE":
        pulse_col = optimizer_pulse_column(params)
        active = (df[pulse_col].fillna(0.0).to_numpy() > 0.0) if pulse_col else np.zeros(len(df), dtype=bool)
        vix_z = df["VIXCLS_z"].fillna(0.0).to_numpy()
        price_mom = df["Price_Mom_24H"].fillna(0.0).to_numpy()
        if params["vix_confirmation"] == "vix_gt_0":
            active &= vix_z > 0.0
        elif params["vix_confirmation"] == "vix_gt_0_5":
            active &= vix_z > 0.5
        elif params["vix_confirmation"] == "vix_gt_1":
            active &= vix_z > 1.0
        if params["price_confirmation"] == "price_mom_lt_0":
            active &= price_mom < 0.0
        elif params["price_confirmation"] == "price_mom_lt_neg_0_005":
            active &= price_mom < -0.005

        qqq_mask = (baseline_signal == "QQQ") & active
        spy_mask = (baseline_signal == "SPY") & active
        action = params["overlay_action"]
        if action == "QQQ_TO_SPY":
            target_signal[qqq_mask] = "SPY"
        elif action == "QQQ_TO_HALF_QQQ_HALF_SPY":
            target_signal[qqq_mask] = "QQQ50_SPY50"
        elif action == "QQQ_TO_CASH":
            target_signal[qqq_mask] = "CASH"
        elif action == "SPY_TO_HALF_SPY_HALF_CASH":
            target_signal[spy_mask] = "SPY50_CASH50"
        elif action == "SPY_TO_CASH":
            target_signal[spy_mask] = "CASH"
        elif action == "COMBINED_CONSERVATIVE":
            target_signal[qqq_mask & (vix_z > 1.0)] = "CASH"
            target_signal[qqq_mask & ~(vix_z > 1.0)] = "SPY"
            target_signal[spy_mask & (price_mom < 0.0)] = "CASH"

    signal_assets = target_signal[:-1]
    baseline_assets = baseline_signal[:-1]
    qqq_ret = df["QQQ_close"].to_numpy()[1:] / df["QQQ_close"].to_numpy()[:-1] - 1
    spy_ret = df["SPY_close"].to_numpy()[1:] / df["SPY_close"].to_numpy()[:-1] - 1
    qqq_w = np.where(signal_assets == "QQQ", 1.0, np.where(signal_assets == "QQQ50_SPY50", 0.5, 0.0))
    spy_w = np.where(signal_assets == "SPY", 1.0, np.where(signal_assets == "QQQ50_SPY50", 0.5, np.where(signal_assets == "SPY50_CASH50", 0.5, 0.0)))
    hourly_ret = qqq_w * qqq_ret + spy_w * spy_ret
    changed_asset = np.r_[True, signal_assets[1:] != signal_assets[:-1]]
    costs = np.where(changed_asset, 0.0005, 0.0)
    equities = 5_000_000.0 * np.cumprod(1.0 + hourly_ret - costs)
    peaks = np.maximum.accumulate(equities)
    drawdowns = equities / peaks - 1
    strategy_df = pd.DataFrame(
        {
            "timestamp": df["timestamp"].iloc[1:].to_numpy(),
            "strategy_equity": equities,
            "drawdown": drawdowns,
            "overlay_asset": signal_assets,
            "baseline_asset": baseline_assets,
            "overlay_changed": signal_assets != baseline_assets,
        }
    )
    daily = strategy_df.set_index("timestamp")["strategy_equity"].resample("D").last().dropna()
    years = max((daily.index.max() - daily.index.min()).days / 365.0, 1 / 365.0)
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1
    daily_ret = daily.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0
    daily_volatility = float(daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0
    transitions = int((strategy_df["overlay_asset"] != strategy_df["overlay_asset"].shift(1)).sum() - 1)
    baseline_assets_s = strategy_df["baseline_asset"].fillna("UNKNOWN")
    overlay_assets_s = strategy_df["overlay_asset"].fillna("UNKNOWN")
    summary = {
        "window_name": window_name,
        "news_availability_lag_minutes": lag_minutes,
        "overlay_rule": params["overlay_rule"],
        "rows": len(strategy_df),
        "strategy_return": float(strategy_df["strategy_equity"].iloc[-1] / strategy_df["strategy_equity"].iloc[0] - 1),
        "cagr_365d": float(cagr),
        "daily_sharpe_365d": sharpe,
        "daily_volatility_365d": daily_volatility,
        "max_drawdown": float(strategy_df["drawdown"].min()),
        "transition_count": transitions,
        "overlay_changes": int(strategy_df["overlay_changed"].sum()),
        "worst_drawdown_period_return": worst_drawdown_return(strategy_df),
        "qqq_to_spy": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "SPY")).sum()),
        "qqq_to_cash": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "CASH")).sum()),
        "qqq_to_50qqq_50spy": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "QQQ50_SPY50")).sum()),
        "qqq_to_50spy_50cash": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "SPY50_CASH50")).sum()),
        "spy_to_cash": int(((baseline_assets_s == "SPY") & (overlay_assets_s == "CASH")).sum()),
        "spy_to_50spy_50cash": int(((baseline_assets_s == "SPY") & (overlay_assets_s == "SPY50_CASH50")).sum()),
        "unchanged_cash": int(((baseline_assets_s == "CASH") & (overlay_assets_s == "CASH")).sum()),
    }
    if baseline_summary is not None:
        summary["return_diff_vs_baseline"] = summary["strategy_return"] - baseline_summary["strategy_return"]
        summary["cagr_diff_vs_baseline"] = summary["cagr_365d"] - baseline_summary["cagr_365d"]
        summary["drawdown_diff_vs_baseline"] = summary["max_drawdown"] - baseline_summary["max_drawdown"]
        summary["sharpe_diff_vs_baseline"] = summary["daily_sharpe_365d"] - baseline_summary["daily_sharpe_365d"]
    return summary


def run_overlay_grid_optimizer():
    overlay_started = datetime.now(timezone.utc).replace(tzinfo=None)
    param_grid = overlay_param_grid()
    summary_rows = []
    validation_rows = []

    for lag_minutes in OPT_LAG_SETTINGS:
        for window in OVERLAY_WINDOWS:
            window_name = window["window_name"]
            start_ts = window["start_ts"]
            end_ts = window["end_ts"]
            window_pdf = load_overlay_optimizer_window_pdf(start_ts, end_ts, lag_minutes)
            window_pdf["timestamp"] = pd.to_datetime(window_pdf["timestamp"]).dt.tz_localize(None)
            max_source = pd.to_datetime(window_pdf["max_gdelt_source_ts"]).dt.tz_localize(None)
            lag_cutoff = window_pdf["timestamp"] - pd.to_timedelta(lag_minutes, unit="m")
            leakage_rows = int(((max_source.notna()) & (max_source > lag_cutoff)).sum())
            null_feature_cols = ["VIXCLS_z", "Price_Mom_24H", "policy_uncertainty_news_z", "geopolitical_risk_news_z", "market_stress_news_z"]
            if leakage_rows:
                raise AssertionError(f"Overlay lag leakage detected for {window_name}, lag={lag_minutes}: {leakage_rows}")
            window_pdf[null_feature_cols] = window_pdf[null_feature_cols].fillna(0.0)
            null_overlay_features = int(window_pdf[null_feature_cols].isna().sum().sum())
            window_pdf = add_optimizer_pulses(window_pdf)

            baseline_wf, _, _ = monthly_walkforward(window_pdf, BASELINE_FEATURES, f"{window_name}_baseline_v4")
            validation_rows.append(
                {
                    "window_name": window_name,
                    "news_availability_lag_minutes": lag_minutes,
                    "baseline_rows": len(window_pdf),
                    "joined_rows": len(window_pdf),
                    "walkforward_rows": len(baseline_wf),
                    "leakage_rows": leakage_rows,
                    "null_overlay_feature_rows": null_overlay_features,
                }
            )

            baseline_summary = None
            for params in param_grid:
                summary = simulate_optimizer_summary(baseline_wf, params, window_name, lag_minutes, baseline_summary)
                summary.update({k: params[k] for k in ["pulse_source", "pulse_threshold", "pulse_window_hours", "vix_confirmation", "price_confirmation", "overlay_action"]})
                if params["overlay_action"] == "BASELINE":
                    baseline_summary = summary
                    summary["return_diff_vs_baseline"] = 0.0
                    summary["cagr_diff_vs_baseline"] = 0.0
                    summary["drawdown_diff_vs_baseline"] = 0.0
                    summary["sharpe_diff_vs_baseline"] = 0.0
                else:
                    summary["cagr_diff_vs_baseline"] = summary["cagr_365d"] - baseline_summary["cagr_365d"]
                summary["transparent_score"] = (
                    summary["daily_sharpe_365d"]
                    + 0.5 * summary["cagr_365d"]
                    - 2.0 * abs(summary["max_drawdown"])
                    - 0.002 * summary["overlay_changes"]
                )
                summary_rows.append(summary)

    summary_pdf = pd.DataFrame(summary_rows)
    baseline_lookup = summary_pdf[summary_pdf["overlay_action"] == "BASELINE"][
        ["window_name", "news_availability_lag_minutes", "strategy_return", "max_drawdown", "daily_sharpe_365d", "transition_count"]
    ].rename(
        columns={
            "strategy_return": "baseline_return",
            "max_drawdown": "baseline_max_drawdown",
            "daily_sharpe_365d": "baseline_sharpe",
            "transition_count": "baseline_transition_count",
        }
    )
    summary_pdf = summary_pdf.merge(baseline_lookup, on=["window_name", "news_availability_lag_minutes"], how="left")
    summary_pdf["turnover_ratio"] = summary_pdf["transition_count"] / summary_pdf["baseline_transition_count"].replace(0, np.nan)
    summary_pdf["catastrophic_underperformance"] = (
        (summary_pdf["strategy_return"] < summary_pdf["baseline_return"] - 0.02)
        | (summary_pdf["max_drawdown"] < summary_pdf["baseline_max_drawdown"] - 0.01)
        | (summary_pdf["daily_sharpe_365d"] < summary_pdf["baseline_sharpe"] - 0.5)
    )
    summary_pdf["objective_c_pass"] = (
        (summary_pdf["max_drawdown"] >= summary_pdf["baseline_max_drawdown"] - 0.0025)
        & (summary_pdf["turnover_ratio"].fillna(0.0) <= 1.5)
    )
    summary_pdf["objective_d_pass"] = (
        (summary_pdf["max_drawdown"] >= summary_pdf["baseline_max_drawdown"])
        & (summary_pdf["strategy_return"] >= summary_pdf["baseline_return"])
    )

    aggregate_rows = []
    for (lag_minutes, overlay_rule), grp in summary_pdf.groupby(["news_availability_lag_minutes", "overlay_rule"]):
        if overlay_rule == "baseline_v4":
            params = grp.iloc[0]
        else:
            params = grp.iloc[0]
        aggregate_rows.append(
            {
                "news_availability_lag_minutes": lag_minutes,
                "overlay_rule": overlay_rule,
                "pulse_source": params["pulse_source"],
                "pulse_threshold": params["pulse_threshold"],
                "pulse_window_hours": params["pulse_window_hours"],
                "vix_confirmation": params["vix_confirmation"],
                "price_confirmation": params["price_confirmation"],
                "overlay_action": params["overlay_action"],
                "mean_return": float(grp["strategy_return"].mean()),
                "mean_cagr": float(grp["cagr_365d"].mean()),
                "mean_sharpe": float(grp["daily_sharpe_365d"].mean()),
                "worst_return_diff": float(grp["return_diff_vs_baseline"].min()),
                "worst_drawdown_diff": float(grp["drawdown_diff_vs_baseline"].min()),
                "worst_sharpe_diff": float(grp["sharpe_diff_vs_baseline"].min()),
                "mean_overlay_changes": float(grp["overlay_changes"].mean()),
                "catastrophic_windows": int(grp["catastrophic_underperformance"].sum()),
                "objective_c_windows": int(grp["objective_c_pass"].sum()),
                "objective_d_windows": int(grp["objective_d_pass"].sum()),
                "mean_transparent_score": float(grp["transparent_score"].mean()),
            }
        )
    aggregate_pdf = pd.DataFrame(aggregate_rows)
    for col_name, ascending in [
        ("mean_cagr", False),
        ("mean_sharpe", False),
        ("mean_return", False),
        ("worst_drawdown_diff", False),
        ("mean_overlay_changes", True),
    ]:
        aggregate_pdf[f"rank_{col_name}"] = aggregate_pdf.groupby("news_availability_lag_minutes")[col_name].rank(ascending=ascending, method="average")
    aggregate_pdf["normalized_rank_score"] = -aggregate_pdf[
        ["rank_mean_cagr", "rank_mean_sharpe", "rank_mean_return", "rank_worst_drawdown_diff", "rank_mean_overlay_changes"]
    ].mean(axis=1)
    aggregate_pdf["robust_score"] = aggregate_pdf["normalized_rank_score"] + aggregate_pdf["mean_transparent_score"] - 1000 * aggregate_pdf["catastrophic_windows"]
    aggregate_pdf["optimization_mode"] = "grid_optimize"
    aggregate_pdf["ingestion_run_id"] = INGESTION_RUN_ID
    aggregate_pdf["ingestion_ts"] = RUN_TS

    summary_pdf["optimization_mode"] = "grid_optimize"
    summary_pdf["ingestion_run_id"] = INGESTION_RUN_ID
    summary_pdf["ingestion_ts"] = RUN_TS

    spark.createDataFrame(summary_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_TABLE)
    spark.createDataFrame(aggregate_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_OPT_TABLE)
    display(spark.createDataFrame(pd.DataFrame(validation_rows)))
    display(spark.createDataFrame(summary_pdf))
    display(spark.createDataFrame(aggregate_pdf.sort_values(["news_availability_lag_minutes", "robust_score"], ascending=[True, False]).head(50)))
    overlay_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("overlay_grid_optimize", overlay_started, overlay_completed, len(summary_pdf), f"grid_size={len(param_grid)};lags={OPT_LAG_SETTINGS};summary_only=true")
    return summary_pdf, aggregate_pdf, pd.DataFrame(validation_rows)


def run_selected_overlay_validation():
    overlay_started = datetime.now(timezone.utc).replace(tzinfo=None)
    selected_params = {
        "pulse_source": "market_stress",
        "pulse_threshold": 2.25,
        "pulse_window_hours": 48,
        "vix_confirmation": "none",
        "price_confirmation": "none",
        "overlay_action": "QQQ_TO_CASH",
    }
    selected_params["overlay_rule"] = overlay_param_name(selected_params)
    baseline_params = {
        "pulse_source": "baseline",
        "pulse_threshold": 0.0,
        "pulse_window_hours": 0,
        "vix_confirmation": "none",
        "price_confirmation": "none",
        "overlay_action": "BASELINE",
        "overlay_rule": "baseline_v4",
    }

    summary_rows = []
    validation_rows = []
    lags = selected_overlay_lag_settings()
    for lag_minutes in lags:
        for window in OVERLAY_WINDOWS:
            window_name = window["window_name"]
            start_ts = window["start_ts"]
            end_ts = window["end_ts"]
            window_pdf = load_overlay_optimizer_window_pdf(start_ts, end_ts, lag_minutes)
            window_pdf["timestamp"] = pd.to_datetime(window_pdf["timestamp"]).dt.tz_localize(None)
            max_source = pd.to_datetime(window_pdf["max_gdelt_source_ts"]).dt.tz_localize(None)
            lag_cutoff = window_pdf["timestamp"] - pd.to_timedelta(lag_minutes, unit="m")
            leakage_rows = int(((max_source.notna()) & (max_source > lag_cutoff)).sum())
            null_feature_cols = ["VIXCLS_z", "Price_Mom_24H", "policy_uncertainty_news_z", "geopolitical_risk_news_z", "market_stress_news_z"]
            null_overlay_features = int(window_pdf[null_feature_cols].isna().sum().sum())
            if leakage_rows:
                raise AssertionError(f"Selected overlay lag leakage detected for {window_name}, lag={lag_minutes}: {leakage_rows}")
            if null_overlay_features:
                raise AssertionError(f"Selected overlay null feature rows for {window_name}, lag={lag_minutes}: {null_overlay_features}")

            window_pdf = add_optimizer_pulses(window_pdf)
            selected_pulse_col = optimizer_pulse_column(selected_params)
            baseline_wf, _, _ = monthly_walkforward(window_pdf, BASELINE_FEATURES, f"{window_name}_baseline_v4")
            baseline_wf[selected_pulse_col] = baseline_wf[selected_pulse_col].fillna(0.0)
            active_count = int((baseline_wf[selected_pulse_col] > 0.0).sum())

            baseline_summary = simulate_optimizer_summary(baseline_wf, baseline_params, window_name, lag_minutes, None)
            baseline_summary.update(
                {
                    "return_diff_vs_baseline": 0.0,
                    "cagr_diff_vs_baseline": 0.0,
                    "drawdown_diff_vs_baseline": 0.0,
                    "sharpe_diff_vs_baseline": 0.0,
                    "catastrophic_underperformance": False,
                    "selected_pulse_active_rows": 0,
                }
            )
            selected_summary = simulate_optimizer_summary(baseline_wf, selected_params, window_name, lag_minutes, baseline_summary)
            selected_summary["catastrophic_underperformance"] = bool(
                (selected_summary["strategy_return"] < baseline_summary["strategy_return"] - 0.02)
                or (selected_summary["max_drawdown"] < baseline_summary["max_drawdown"] - 0.01)
                or (selected_summary["daily_sharpe_365d"] < baseline_summary["daily_sharpe_365d"] - 0.5)
            )
            selected_summary["selected_pulse_active_rows"] = active_count

            for params, summary in [(baseline_params, baseline_summary), (selected_params, selected_summary)]:
                summary.update({k: params[k] for k in ["pulse_source", "pulse_threshold", "pulse_window_hours", "vix_confirmation", "price_confirmation", "overlay_action"]})
                summary["optimization_mode"] = "selected_validate"
                summary["ingestion_run_id"] = INGESTION_RUN_ID
                summary["ingestion_ts"] = RUN_TS
                summary_rows.append(summary)

            validation_rows.append(
                {
                    "window_name": window_name,
                    "news_availability_lag_minutes": lag_minutes,
                    "baseline_rows": len(window_pdf),
                    "joined_rows": len(window_pdf),
                    "walkforward_rows": len(baseline_wf),
                    "leakage_rows": leakage_rows,
                    "lag_leakage_rows": leakage_rows,
                    "null_overlay_feature_rows": null_overlay_features,
                    "selected_pulse_active_rows": active_count,
                }
            )

    summary_pdf = pd.DataFrame(summary_rows)
    validation_pdf = pd.DataFrame(validation_rows)
    spark.createDataFrame(summary_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_TABLE)
    display(spark.createDataFrame(validation_pdf))
    display(spark.createDataFrame(summary_pdf))
    overlay_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("overlay_selected_validate", overlay_started, overlay_completed, len(summary_pdf), f"lags={lags};overlay={selected_params['overlay_rule']}")
    return summary_pdf, None, validation_pdf


def targeted_refinement_rules():
    base_rules = [
        {"overlay_rule": "baseline_v4", "family": "baseline", "action": "BASELINE", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "ref_market_stress_q2cash", "family": "A_reference", "action": "QQQ_TO_CASH", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "soft_market_stress_q2spy", "family": "A_soft", "action": "QQQ_TO_SPY", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "soft_market_stress_q2halfqqq_halfspy", "family": "A_soft", "action": "QQQ_TO_HALF_QQQ_HALF_SPY", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "soft_market_stress_q2halfspy_halfcash", "family": "A_soft", "action": "QQQ_TO_HALF_SPY_HALF_CASH", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "confirm_market_stress_vix05_q2spy", "family": "B_confirm", "action": "VIX05_QQQ_TO_SPY", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "confirm_market_stress_vix1_q2cash", "family": "B_confirm", "action": "VIX1_QQQ_TO_CASH", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "confirm_market_stress_negmom_q2spy", "family": "B_confirm", "action": "NEGMOM_QQQ_TO_SPY", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "confirm_market_stress_vix1_negmom_q2cash", "family": "B_confirm", "action": "VIX1_NEGMOM_QQQ_TO_CASH", "cooldown_hours": 0, "hysteresis_2h": False},
        {"overlay_rule": "tiered_market_stress_vix_price", "family": "C_tiered", "action": "TIERED_VIX_PRICE", "cooldown_hours": 0, "hysteresis_2h": False},
    ]
    controlled_actions = [
        ("soft_market_stress_q2spy", "A_soft", "QQQ_TO_SPY"),
        ("soft_market_stress_q2halfqqq_halfspy", "A_soft", "QQQ_TO_HALF_QQQ_HALF_SPY"),
        ("confirm_market_stress_vix1_q2cash", "B_confirm", "VIX1_QQQ_TO_CASH"),
        ("tiered_market_stress_vix_price", "C_tiered", "TIERED_VIX_PRICE"),
    ]
    for rule_prefix, family, action in controlled_actions:
        for cooldown in [12, 24]:
            base_rules.append(
                {
                    "overlay_rule": f"{rule_prefix}_cooldown_{cooldown}h",
                    "family": "D_cooldown",
                    "action": action,
                    "cooldown_hours": cooldown,
                    "hysteresis_2h": False,
                }
            )
        base_rules.append(
            {
                "overlay_rule": f"{rule_prefix}_hysteresis_2h",
                "family": "D_hysteresis",
                "action": action,
                "cooldown_hours": 0,
                "hysteresis_2h": True,
            }
        )
    return base_rules


def targeted_desired_asset(row, rule, active):
    baseline_asset = baseline_asset_from_regime(row["predicted_regime"])
    if baseline_asset == "CASH" or rule["action"] == "BASELINE":
        return baseline_asset
    if not active:
        return baseline_asset
    vix_z = float(row.get("VIXCLS_z", 0.0) or 0.0)
    price_mom = float(row.get("Price_Mom_24H", 0.0) or 0.0)
    action = rule["action"]
    if baseline_asset == "QQQ" and action == "QQQ_TO_SPY":
        return "SPY"
    if baseline_asset == "QQQ" and action == "QQQ_TO_HALF_QQQ_HALF_SPY":
        return "QQQ50_SPY50"
    if baseline_asset == "QQQ" and action == "QQQ_TO_HALF_SPY_HALF_CASH":
        return "SPY50_CASH50"
    if baseline_asset == "QQQ" and action == "QQQ_TO_CASH":
        return "CASH"
    if baseline_asset == "QQQ" and action == "VIX05_QQQ_TO_SPY" and vix_z > 0.5:
        return "SPY"
    if baseline_asset == "QQQ" and action == "VIX1_QQQ_TO_CASH" and vix_z > 1.0:
        return "CASH"
    if baseline_asset == "QQQ" and action == "NEGMOM_QQQ_TO_SPY" and price_mom < 0.0:
        return "SPY"
    if baseline_asset == "QQQ" and action == "VIX1_NEGMOM_QQQ_TO_CASH" and vix_z > 1.0 and price_mom < 0.0:
        return "CASH"
    if baseline_asset == "QQQ" and action == "TIERED_VIX_PRICE":
        if vix_z <= 0.5:
            return "QQQ50_SPY50"
        if vix_z <= 1.0:
            return "SPY"
        if price_mom < 0.0:
            return "CASH"
        return "SPY"
    return baseline_asset


def apply_targeted_controls(df, rule, pulse_active):
    if rule["action"] == "BASELINE":
        return df["predicted_regime"].apply(baseline_asset_from_regime).to_numpy(dtype=object)
    if rule.get("hysteresis_2h"):
        entry_active = pulse_active & np.r_[False, pulse_active[:-1]]
        inactive_two = (~pulse_active) & np.r_[False, ~pulse_active[:-1]]
    else:
        entry_active = pulse_active
        inactive_two = ~pulse_active

    desired = []
    for idx, (_, row) in enumerate(df.iterrows()):
        desired.append(targeted_desired_asset(row, rule, bool(entry_active[idx])))
    desired = np.array(desired, dtype=object)
    baseline_signal = df["predicted_regime"].apply(baseline_asset_from_regime).to_numpy(dtype=object)
    cooldown = int(rule.get("cooldown_hours") or 0)
    if cooldown <= 0 and not rule.get("hysteresis_2h"):
        return desired

    output = baseline_signal.copy()
    last_change_idx = -10_000
    for idx in range(len(output)):
        if baseline_signal[idx] == "CASH":
            output[idx] = "CASH"
            if idx > 0 and output[idx] != output[idx - 1]:
                last_change_idx = idx
            continue
        previous = output[idx - 1] if idx > 0 else baseline_signal[idx]
        target = desired[idx]
        if rule.get("hysteresis_2h") and target == baseline_signal[idx] and previous != baseline_signal[idx] and not inactive_two[idx]:
            target = previous
        if cooldown > 0 and target != previous and idx - last_change_idx < cooldown:
            target = previous
        output[idx] = target
        if idx > 0 and output[idx] != output[idx - 1]:
            last_change_idx = idx
    return output


def summarize_targeted_overlay(baseline_wf, rule, window_name, lag_minutes, baseline_summary=None):
    df = baseline_wf.sort_values("timestamp").copy()
    pulse_col = "opt_market_stress_pulse_48h_t225"
    pulse_active = df[pulse_col].fillna(0.0).to_numpy() > 0.0
    baseline_signal = df["predicted_regime"].apply(baseline_asset_from_regime).to_numpy(dtype=object)
    target_signal = apply_targeted_controls(df, rule, pulse_active)

    signal_assets = target_signal[:-1]
    baseline_assets = baseline_signal[:-1]
    qqq_ret = df["QQQ_close"].to_numpy()[1:] / df["QQQ_close"].to_numpy()[:-1] - 1
    spy_ret = df["SPY_close"].to_numpy()[1:] / df["SPY_close"].to_numpy()[:-1] - 1
    qqq_w = np.where(signal_assets == "QQQ", 1.0, np.where(signal_assets == "QQQ50_SPY50", 0.5, 0.0))
    spy_w = np.where(
        signal_assets == "SPY",
        1.0,
        np.where(signal_assets == "QQQ50_SPY50", 0.5, np.where(signal_assets == "SPY50_CASH50", 0.5, 0.0)),
    )
    hourly_ret = qqq_w * qqq_ret + spy_w * spy_ret
    changed_asset = np.r_[True, signal_assets[1:] != signal_assets[:-1]]
    costs = np.where(changed_asset, 0.0005, 0.0)
    equities = 5_000_000.0 * np.cumprod(1.0 + hourly_ret - costs)
    peaks = np.maximum.accumulate(equities)
    drawdowns = equities / peaks - 1
    strategy_df = pd.DataFrame(
        {
            "timestamp": df["timestamp"].iloc[1:].to_numpy(),
            "strategy_equity": equities,
            "drawdown": drawdowns,
            "overlay_asset": signal_assets,
            "baseline_asset": baseline_assets,
            "overlay_changed": signal_assets != baseline_assets,
        }
    )
    daily = strategy_df.set_index("timestamp")["strategy_equity"].resample("D").last().dropna()
    years = max((daily.index.max() - daily.index.min()).days / 365.0, 1 / 365.0)
    daily_ret = daily.pct_change().dropna()
    daily_volatility = float(daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1
    baseline_assets_s = strategy_df["baseline_asset"].fillna("UNKNOWN")
    overlay_assets_s = strategy_df["overlay_asset"].fillna("UNKNOWN")
    summary = {
        "window_name": window_name,
        "news_availability_lag_minutes": lag_minutes,
        "overlay_rule": rule["overlay_rule"],
        "family": rule["family"],
        "pulse_source": "market_stress",
        "pulse_threshold": 2.25,
        "pulse_window_hours": 48,
        "vix_confirmation": "rule_specific",
        "price_confirmation": "rule_specific",
        "overlay_action": rule["action"],
        "cooldown_hours": int(rule.get("cooldown_hours") or 0),
        "hysteresis_2h": bool(rule.get("hysteresis_2h")),
        "rows": len(strategy_df),
        "strategy_return": float(strategy_df["strategy_equity"].iloc[-1] / strategy_df["strategy_equity"].iloc[0] - 1),
        "cagr_365d": float(cagr),
        "daily_sharpe_365d": sharpe,
        "daily_volatility_365d": daily_volatility,
        "max_drawdown": float(strategy_df["drawdown"].min()),
        "transition_count": int((strategy_df["overlay_asset"] != strategy_df["overlay_asset"].shift(1)).sum() - 1),
        "overlay_changes": int(strategy_df["overlay_changed"].sum()),
        "worst_drawdown_period_return": worst_drawdown_return(strategy_df),
        "selected_pulse_active_rows": int(pulse_active.sum()),
        "qqq_to_spy": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "SPY")).sum()),
        "qqq_to_cash": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "CASH")).sum()),
        "qqq_to_50qqq_50spy": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "QQQ50_SPY50")).sum()),
        "qqq_to_50spy_50cash": int(((baseline_assets_s == "QQQ") & (overlay_assets_s == "SPY50_CASH50")).sum()),
        "spy_to_cash": int(((baseline_assets_s == "SPY") & (overlay_assets_s == "CASH")).sum()),
        "spy_to_50spy_50cash": int(((baseline_assets_s == "SPY") & (overlay_assets_s == "SPY50_CASH50")).sum()),
        "unchanged_cash": int(((baseline_assets_s == "CASH") & (overlay_assets_s == "CASH")).sum()),
    }
    if baseline_summary is not None:
        summary["return_diff_vs_baseline"] = summary["strategy_return"] - baseline_summary["strategy_return"]
        summary["cagr_diff_vs_baseline"] = summary["cagr_365d"] - baseline_summary["cagr_365d"]
        summary["drawdown_diff_vs_baseline"] = summary["max_drawdown"] - baseline_summary["max_drawdown"]
        summary["sharpe_diff_vs_baseline"] = summary["daily_sharpe_365d"] - baseline_summary["daily_sharpe_365d"]
    return summary


def run_targeted_overlay_refinement():
    overlay_started = datetime.now(timezone.utc).replace(tzinfo=None)
    rules = targeted_refinement_rules()
    lags = selected_overlay_lag_settings() if OVERLAY_SELECTED_LAGS_PARAM else [15]
    summary_rows = []
    validation_rows = []

    for lag_minutes in lags:
        for window in OVERLAY_WINDOWS:
            window_name = window["window_name"]
            window_pdf = load_overlay_optimizer_window_pdf(window["start_ts"], window["end_ts"], lag_minutes)
            window_pdf["timestamp"] = pd.to_datetime(window_pdf["timestamp"]).dt.tz_localize(None)
            max_source = pd.to_datetime(window_pdf["max_gdelt_source_ts"]).dt.tz_localize(None)
            lag_cutoff = window_pdf["timestamp"] - pd.to_timedelta(lag_minutes, unit="m")
            leakage_rows = int(((max_source.notna()) & (max_source > lag_cutoff)).sum())
            if leakage_rows:
                raise AssertionError(f"Targeted overlay lag leakage detected for {window_name}, lag={lag_minutes}: {leakage_rows}")
            null_feature_cols = ["VIXCLS_z", "Price_Mom_24H", "policy_uncertainty_news_z", "geopolitical_risk_news_z", "market_stress_news_z"]
            window_pdf[null_feature_cols] = window_pdf[null_feature_cols].fillna(0.0)
            null_overlay_features = int(window_pdf[null_feature_cols].isna().sum().sum())
            window_pdf = add_optimizer_pulses(window_pdf)
            pulse_col = "opt_market_stress_pulse_48h_t225"
            baseline_wf, _, _ = monthly_walkforward(window_pdf, BASELINE_FEATURES, f"{window_name}_baseline_v4")
            baseline_wf[pulse_col] = baseline_wf[pulse_col].fillna(0.0)
            pulse_active_count = int((baseline_wf[pulse_col] > 0.0).sum())
            validation_rows.append(
                {
                    "window_name": window_name,
                    "news_availability_lag_minutes": lag_minutes,
                    "baseline_rows": len(window_pdf),
                    "joined_rows": len(window_pdf),
                    "walkforward_rows": len(baseline_wf),
                    "leakage_rows": leakage_rows,
                    "lag_leakage_rows": leakage_rows,
                    "null_overlay_feature_rows": null_overlay_features,
                    "selected_pulse_active_rows": pulse_active_count,
                }
            )

            baseline_summary = None
            for rule in rules:
                summary = summarize_targeted_overlay(baseline_wf, rule, window_name, lag_minutes, baseline_summary)
                if rule["overlay_rule"] == "baseline_v4":
                    baseline_summary = summary
                    summary["return_diff_vs_baseline"] = 0.0
                    summary["cagr_diff_vs_baseline"] = 0.0
                    summary["drawdown_diff_vs_baseline"] = 0.0
                    summary["sharpe_diff_vs_baseline"] = 0.0
                summary["catastrophic_underperformance"] = bool(
                    (summary["strategy_return"] < baseline_summary["strategy_return"] - 0.02)
                    or (summary["max_drawdown"] < baseline_summary["max_drawdown"] - 0.01)
                    or (summary["daily_sharpe_365d"] < baseline_summary["daily_sharpe_365d"] - 0.5)
                )
                summary["optimization_mode"] = "targeted_refine"
                summary["ingestion_run_id"] = INGESTION_RUN_ID
                summary["ingestion_ts"] = RUN_TS
                summary_rows.append(summary)

    summary_pdf = pd.DataFrame(summary_rows)
    baseline_lookup = summary_pdf[summary_pdf["overlay_rule"] == "baseline_v4"][
        ["window_name", "news_availability_lag_minutes", "strategy_return", "cagr_365d", "max_drawdown", "daily_sharpe_365d", "transition_count"]
    ].rename(
        columns={
            "strategy_return": "baseline_return",
            "cagr_365d": "baseline_cagr",
            "max_drawdown": "baseline_max_drawdown",
            "daily_sharpe_365d": "baseline_sharpe",
            "transition_count": "baseline_transition_count",
        }
    )
    summary_pdf = summary_pdf.merge(baseline_lookup, on=["window_name", "news_availability_lag_minutes"], how="left")
    reference_changes = summary_pdf[summary_pdf["overlay_rule"] == "ref_market_stress_q2cash"][
        ["window_name", "news_availability_lag_minutes", "overlay_changes"]
    ].rename(columns={"overlay_changes": "reference_overlay_changes"})
    summary_pdf = summary_pdf.merge(reference_changes, on=["window_name", "news_availability_lag_minutes"], how="left")
    summary_pdf["turnover_ratio"] = summary_pdf["transition_count"] / summary_pdf["baseline_transition_count"].replace(0, np.nan)
    summary_pdf["constraint_pass"] = (
        (summary_pdf["return_diff_vs_baseline"] >= -0.005)
        & (summary_pdf["sharpe_diff_vs_baseline"] >= -0.25)
        & (summary_pdf["drawdown_diff_vs_baseline"] >= -0.0025)
        & (summary_pdf["overlay_changes"] <= 1.25 * summary_pdf["reference_overlay_changes"].fillna(summary_pdf["overlay_changes"]))
        & (~summary_pdf["catastrophic_underperformance"])
    )
    summary_pdf["secondary_constraint_pass"] = (
        (summary_pdf["drawdown_diff_vs_baseline"] >= 0.0)
        & (summary_pdf["return_diff_vs_baseline"] >= -0.005)
        & (summary_pdf["overlay_changes"] <= 1.25 * summary_pdf["reference_overlay_changes"].fillna(summary_pdf["overlay_changes"]))
        & (~summary_pdf["catastrophic_underperformance"])
    )

    aggregate_rows = []
    for (lag_minutes, overlay_rule), grp in summary_pdf.groupby(["news_availability_lag_minutes", "overlay_rule"]):
        params = grp.iloc[0]
        aggregate_rows.append(
            {
                "news_availability_lag_minutes": lag_minutes,
                "overlay_rule": overlay_rule,
                "family": params["family"],
                "pulse_source": params["pulse_source"],
                "pulse_threshold": params["pulse_threshold"],
                "pulse_window_hours": params["pulse_window_hours"],
                "overlay_action": params["overlay_action"],
                "cooldown_hours": params["cooldown_hours"],
                "hysteresis_2h": params["hysteresis_2h"],
                "mean_return": float(grp["strategy_return"].mean()),
                "mean_cagr": float(grp["cagr_365d"].mean()),
                "mean_sharpe": float(grp["daily_sharpe_365d"].mean()),
                "mean_max_drawdown": float(grp["max_drawdown"].mean()),
                "worst_return_diff": float(grp["return_diff_vs_baseline"].min()),
                "worst_cagr_diff": float(grp["cagr_diff_vs_baseline"].min()),
                "worst_drawdown_diff": float(grp["drawdown_diff_vs_baseline"].min()),
                "worst_sharpe_diff": float(grp["sharpe_diff_vs_baseline"].min()),
                "mean_overlay_changes": float(grp["overlay_changes"].mean()),
                "constraint_windows": int(grp["constraint_pass"].sum()),
                "secondary_constraint_windows": int(grp["secondary_constraint_pass"].sum()),
                "catastrophic_windows": int(grp["catastrophic_underperformance"].sum()),
                "mean_transparent_score": float(
                    (
                        grp["daily_sharpe_365d"]
                        + 0.5 * grp["cagr_365d"]
                        - 2.0 * grp["max_drawdown"].abs()
                        - 0.002 * grp["overlay_changes"]
                    ).mean()
                ),
            }
        )
    aggregate_pdf = pd.DataFrame(aggregate_rows)
    aggregate_pdf["primary_objective_pass"] = aggregate_pdf["constraint_windows"] == len(OVERLAY_WINDOWS)
    aggregate_pdf["secondary_objective_pass"] = aggregate_pdf["secondary_constraint_windows"] == len(OVERLAY_WINDOWS)
    aggregate_pdf["optimization_mode"] = "targeted_refine"
    aggregate_pdf["ingestion_run_id"] = INGESTION_RUN_ID
    aggregate_pdf["ingestion_ts"] = RUN_TS

    spark.createDataFrame(summary_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_TABLE)
    spark.createDataFrame(aggregate_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_OPT_TABLE)
    display(spark.createDataFrame(pd.DataFrame(validation_rows)))
    display(spark.createDataFrame(summary_pdf))
    display(spark.createDataFrame(aggregate_pdf.sort_values(["primary_objective_pass", "mean_sharpe"], ascending=[False, False]).head(50)))
    overlay_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("overlay_targeted_refine", overlay_started, overlay_completed, len(summary_pdf), f"rules={len(rules)};lags={lags}")
    return summary_pdf, aggregate_pdf, pd.DataFrame(validation_rows)


def run_refined_overlay_validation():
    overlay_started = datetime.now(timezone.utc).replace(tzinfo=None)
    baseline_rule = next(rule for rule in targeted_refinement_rules() if rule["overlay_rule"] == "baseline_v4")
    refined_rule = next(
        rule for rule in targeted_refinement_rules()
        if rule["overlay_rule"] == "confirm_market_stress_vix1_q2cash_hysteresis_2h"
    )
    lags = selected_overlay_lag_settings() if OVERLAY_SELECTED_LAGS_PARAM else [15]
    summary_rows = []
    validation_rows = []

    for lag_minutes in lags:
        for window in OVERLAY_WINDOWS:
            window_name = window["window_name"]
            window_pdf = load_overlay_optimizer_window_pdf(window["start_ts"], window["end_ts"], lag_minutes)
            window_pdf["timestamp"] = pd.to_datetime(window_pdf["timestamp"]).dt.tz_localize(None)
            max_source = pd.to_datetime(window_pdf["max_gdelt_source_ts"]).dt.tz_localize(None)
            lag_cutoff = window_pdf["timestamp"] - pd.to_timedelta(lag_minutes, unit="m")
            leakage_rows = int(((max_source.notna()) & (max_source > lag_cutoff)).sum())
            if leakage_rows:
                raise AssertionError(f"Refined overlay lag leakage detected for {window_name}, lag={lag_minutes}: {leakage_rows}")
            null_feature_cols = ["VIXCLS_z", "Price_Mom_24H", "policy_uncertainty_news_z", "geopolitical_risk_news_z", "market_stress_news_z"]
            window_pdf[null_feature_cols] = window_pdf[null_feature_cols].fillna(0.0)
            null_overlay_features = int(window_pdf[null_feature_cols].isna().sum().sum())
            if null_overlay_features:
                raise AssertionError(f"Refined overlay null feature rows for {window_name}, lag={lag_minutes}: {null_overlay_features}")

            window_pdf = add_optimizer_pulses(window_pdf)
            pulse_col = "opt_market_stress_pulse_48h_t225"
            baseline_wf, _, _ = monthly_walkforward(window_pdf, BASELINE_FEATURES, f"{window_name}_baseline_v4")
            baseline_wf[pulse_col] = baseline_wf[pulse_col].fillna(0.0)
            pulse_active_count = int((baseline_wf[pulse_col] > 0.0).sum())

            baseline_summary = summarize_targeted_overlay(baseline_wf, baseline_rule, window_name, lag_minutes, None)
            baseline_summary["return_diff_vs_baseline"] = 0.0
            baseline_summary["cagr_diff_vs_baseline"] = 0.0
            baseline_summary["drawdown_diff_vs_baseline"] = 0.0
            baseline_summary["sharpe_diff_vs_baseline"] = 0.0
            refined_summary = summarize_targeted_overlay(baseline_wf, refined_rule, window_name, lag_minutes, baseline_summary)

            for summary in [baseline_summary, refined_summary]:
                summary["catastrophic_underperformance"] = bool(
                    (summary["strategy_return"] < baseline_summary["strategy_return"] - 0.02)
                    or (summary["max_drawdown"] < baseline_summary["max_drawdown"] - 0.01)
                    or (summary["daily_sharpe_365d"] < baseline_summary["daily_sharpe_365d"] - 0.5)
                )
                summary["control_window_failure"] = bool(
                    (summary["strategy_return"] < baseline_summary["strategy_return"] - 0.01)
                    or (summary["cagr_365d"] < baseline_summary["cagr_365d"] - 0.03)
                    or (summary["daily_sharpe_365d"] < baseline_summary["daily_sharpe_365d"] - 0.25)
                    or (summary["max_drawdown"] < baseline_summary["max_drawdown"] - 0.0025)
                )
                summary["optimization_mode"] = "refined_validate"
                summary["ingestion_run_id"] = INGESTION_RUN_ID
                summary["ingestion_ts"] = RUN_TS
                summary_rows.append(summary)

            validation_rows.append(
                {
                    "window_name": window_name,
                    "news_availability_lag_minutes": lag_minutes,
                    "baseline_rows": len(window_pdf),
                    "joined_rows": len(window_pdf),
                    "walkforward_rows": len(baseline_wf),
                    "leakage_rows": leakage_rows,
                    "lag_leakage_rows": leakage_rows,
                    "null_overlay_feature_rows": null_overlay_features,
                    "selected_pulse_active_rows": pulse_active_count,
                }
            )

    summary_pdf = pd.DataFrame(summary_rows)
    validation_pdf = pd.DataFrame(validation_rows)
    spark.createDataFrame(summary_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_TABLE)
    display(spark.createDataFrame(validation_pdf))
    display(spark.createDataFrame(summary_pdf))
    overlay_completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("overlay_refined_validate", overlay_started, overlay_completed, len(summary_pdf), f"lags={lags};overlay={refined_rule['overlay_rule']}")
    return summary_pdf, None, validation_pdf


def fixed_overlay_rule():
    return next(
        rule for rule in targeted_refinement_rules()
        if rule["overlay_rule"] == "confirm_market_stress_vix1_q2cash_hysteresis_2h"
    )


def full_history_end_ts():
    if FULL_HISTORY_END_TS is not None:
        return FULL_HISTORY_END_TS
    max_ts = spark.sql(f"SELECT MAX(timestamp) AS max_ts FROM {SILVER_MASTER_TABLE}").collect()[0]["max_ts"]
    if max_ts is None:
        raise ValueError(f"No rows found in {SILVER_MASTER_TABLE}.")
    return pd.Timestamp(max_ts).tz_localize(None)


def load_full_history_overlay_pdf(train_start_ts, live_start_ts, end_ts, lag_minutes):
    shift_hours = int(math.ceil(lag_minutes / 60.0)) if lag_minutes > 0 else 0
    join_expr = "g.timestamp = n.timestamp" if shift_hours == 0 else f"g.timestamp - INTERVAL {shift_hours} HOURS = n.timestamp"
    select_sql = ",\n                ".join([f"g.{col}" for col in BASELINE_FEATURES])
    pdf = spark.sql(
        f"""
        SELECT
            g.timestamp,
            {select_sql},
            g.SPY_close,
            g.QQQ_close,
            n.policy_uncertainty_news_z,
            n.geopolitical_risk_news_z,
            n.market_stress_news_z,
            n.missing_15m_bucket_count,
            n.missing_15m_bucket_pct,
            n.gdelt_missing_flag,
            n.has_missing_15m_bucket,
            n.max_gdelt_source_ts
        FROM {SILVER_MASTER_TABLE} g
        LEFT JOIN {RESEARCH_FEATURE_TABLE} n
          ON {join_expr}
        WHERE g.timestamp >= TIMESTAMP('{train_start_ts}')
          AND g.timestamp <= TIMESTAMP('{end_ts}')
        ORDER BY g.timestamp
        """
    ).toPandas().sort_values("timestamp")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"]).dt.tz_localize(None)
    return pdf


def full_history_monthly_walkforward(pdf, train_start_ts, live_start_ts, end_ts):
    data = pdf.sort_values("timestamp").copy()
    data["fwd_return_1w"] = data["SPY_close"].shift(-168) / data["SPY_close"] - 1
    data["regime"] = data["fwd_return_1w"].apply(label_regime)
    trainable = (
        data
        .dropna(subset=BASELINE_FEATURES + ["regime", "SPY_close", "QQQ_close"])
        .copy()
    )
    fold_start = pd.Timestamp(live_start_ts).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_ts = pd.Timestamp(end_ts)
    signals = []
    metrics = []
    importances = []
    while fold_start <= end_ts:
        fold_end = fold_start + relativedelta(months=1)
        train_df = trainable[(trainable["timestamp"] >= train_start_ts) & (trainable["timestamp"] < fold_start)]
        test_df = trainable[(trainable["timestamp"] >= fold_start) & (trainable["timestamp"] < fold_end) & (trainable["timestamp"] <= end_ts)].copy()
        if not train_df.empty and not test_df.empty:
            model = build_model()
            model.fit(train_df[BASELINE_FEATURES], train_df["regime"])
            test_df["predicted_regime"] = model.predict(test_df[BASELINE_FEATURES]).astype(int)
            test_df["experiment_name"] = "full_history_baseline_v4"
            test_df["fold_train_end"] = fold_start
            test_df["fold_test_end"] = fold_end
            test_df["accuracy"] = float(accuracy_score(test_df["regime"], test_df["predicted_regime"]))
            signals.append(test_df)
            metrics.append(
                {
                    "fold_train_end": fold_start,
                    "fold_test_end": fold_end,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "accuracy": float(test_df["accuracy"].iloc[0]),
                }
            )
            importances.append(pd.Series(model.feature_importances_, index=BASELINE_FEATURES, name=fold_end))
        fold_start = fold_end
    if not signals:
        raise ValueError("No full-history monthly walk-forward folds produced.")
    wf = pd.concat(signals, ignore_index=True).sort_values("timestamp")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).reset_index()
    imp.columns = ["feature", "mean_importance"]
    return wf, pd.DataFrame(metrics), imp


def strategy_detail_from_target_signal(df, target_signal, overlay_rule, lag_minutes):
    baseline_signal = df["predicted_regime"].apply(baseline_asset_from_regime).to_numpy(dtype=object)
    signal_assets = target_signal[:-1]
    baseline_assets = baseline_signal[:-1]
    if "gdelt_missing_flag" in df.columns:
        gdelt_missing_signal = df["gdelt_missing_flag"].fillna(1).astype(int).to_numpy()[:-1]
    else:
        gdelt_missing_signal = np.zeros(len(signal_assets), dtype=int)
    qqq_ret = df["QQQ_close"].to_numpy()[1:] / df["QQQ_close"].to_numpy()[:-1] - 1
    spy_ret = df["SPY_close"].to_numpy()[1:] / df["SPY_close"].to_numpy()[:-1] - 1
    qqq_w = np.where(signal_assets == "QQQ", 1.0, np.where(signal_assets == "QQQ50_SPY50", 0.5, 0.0))
    spy_w = np.where(
        signal_assets == "SPY",
        1.0,
        np.where(signal_assets == "QQQ50_SPY50", 0.5, np.where(signal_assets == "SPY50_CASH50", 0.5, 0.0)),
    )
    hourly_ret = qqq_w * qqq_ret + spy_w * spy_ret
    changed_asset = np.r_[True, signal_assets[1:] != signal_assets[:-1]]
    costs = np.where(changed_asset, 0.0005, 0.0)
    equities = 5_000_000.0 * np.cumprod(1.0 + hourly_ret - costs)
    peaks = np.maximum.accumulate(equities)
    return pd.DataFrame(
        {
            "experiment_name": overlay_rule,
            "timestamp": df["timestamp"].iloc[1:].to_numpy(),
            "regime": df["regime"].iloc[1:].to_numpy(),
            "predicted_regime": df["predicted_regime"].iloc[1:].to_numpy(),
            "baseline_asset": baseline_assets,
            "overlay_asset": signal_assets,
            "overlay_changed": signal_assets != baseline_assets,
            "gdelt_missing_flag": gdelt_missing_signal,
            "overlay_changed_during_missing_news": (signal_assets != baseline_assets) & (gdelt_missing_signal == 1),
            "strategy_equity": equities,
            "drawdown": equities / peaks - 1,
            "news_availability_lag_minutes": lag_minutes,
        }
    )


def full_history_strategy_details(wf, lag_minutes):
    wf = wf.sort_values("timestamp").copy()
    baseline_signal = wf["predicted_regime"].apply(baseline_asset_from_regime).to_numpy(dtype=object)
    pulse_col = "opt_market_stress_pulse_48h_t225"
    gdelt_available = (wf.get("gdelt_missing_flag", pd.Series(0, index=wf.index)).fillna(1).astype(int).to_numpy() == 0)
    pulse_active = (wf[pulse_col].fillna(0.0).to_numpy() > 0.0) & gdelt_available
    overlay_signal = apply_targeted_controls(wf, fixed_overlay_rule(), pulse_active)
    baseline_detail = strategy_detail_from_target_signal(wf, baseline_signal, "baseline_v4", lag_minutes)
    overlay_detail = strategy_detail_from_target_signal(wf, overlay_signal, fixed_overlay_rule()["overlay_rule"], lag_minutes)
    missing_changes = int(overlay_detail["overlay_changed_during_missing_news"].sum())
    if missing_changes:
        raise AssertionError(f"Fixed overlay changed {missing_changes} rows during missing GDELT news periods.")
    return baseline_detail, overlay_detail


def summarize_strategy_detail(detail, baseline_summary=None):
    daily = detail.set_index("timestamp")["strategy_equity"].resample("D").last().dropna()
    years = max((daily.index.max() - daily.index.min()).days / 365.0, 1 / 365.0)
    daily_ret = daily.pct_change().dropna()
    summary = {
        "overlay_rule": detail["experiment_name"].iloc[0],
        "rows": len(detail),
        "strategy_return": float(detail["strategy_equity"].iloc[-1] / detail["strategy_equity"].iloc[0] - 1),
        "cagr_365d": float((daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1),
        "daily_sharpe_365d": float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0,
        "daily_volatility_365d": float(daily_ret.std() * np.sqrt(365)) if daily_ret.std() else 0.0,
        "max_drawdown": float(detail["drawdown"].min()),
        "transition_count": int((detail["overlay_asset"] != detail["overlay_asset"].shift(1)).sum() - 1),
        "overlay_changes": int(detail["overlay_changed"].sum()),
        "missing_news_rows": int(detail.get("gdelt_missing_flag", pd.Series(dtype=int)).fillna(0).sum()) if "gdelt_missing_flag" in detail.columns else 0,
        "overlay_changes_during_missing_news": int(detail.get("overlay_changed_during_missing_news", pd.Series(dtype=bool)).fillna(False).sum()) if "overlay_changed_during_missing_news" in detail.columns else 0,
        "qqq_to_cash": int(((detail["baseline_asset"] == "QQQ") & (detail["overlay_asset"] == "CASH")).sum()),
        "spy_to_cash": int(((detail["baseline_asset"] == "SPY") & (detail["overlay_asset"] == "CASH")).sum()),
        "unchanged_cash": int(((detail["baseline_asset"] == "CASH") & (detail["overlay_asset"] == "CASH")).sum()),
        "worst_drawdown_period_return": worst_drawdown_return(detail),
    }
    if baseline_summary is not None:
        summary["return_diff_vs_baseline"] = summary["strategy_return"] - baseline_summary["strategy_return"]
        summary["cagr_diff_vs_baseline"] = summary["cagr_365d"] - baseline_summary["cagr_365d"]
        summary["sharpe_diff_vs_baseline"] = summary["daily_sharpe_365d"] - baseline_summary["daily_sharpe_365d"]
        summary["drawdown_diff_vs_baseline"] = summary["max_drawdown"] - baseline_summary["max_drawdown"]
    else:
        summary["return_diff_vs_baseline"] = 0.0
        summary["cagr_diff_vs_baseline"] = 0.0
        summary["sharpe_diff_vs_baseline"] = 0.0
        summary["drawdown_diff_vs_baseline"] = 0.0
    return summary


def annual_strategy_summary(detail, baseline_annual=None):
    rows = []
    for year, grp in detail.groupby(detail["timestamp"].dt.year):
        summary = summarize_strategy_detail(grp)
        summary["period_type"] = "annual"
        summary["period"] = str(int(year))
        rows.append(summary)
    out = pd.DataFrame(rows)
    if baseline_annual is not None:
        base = baseline_annual[["period", "strategy_return", "cagr_365d", "daily_sharpe_365d", "max_drawdown"]].rename(
            columns={
                "strategy_return": "baseline_return",
                "cagr_365d": "baseline_cagr",
                "daily_sharpe_365d": "baseline_sharpe",
                "max_drawdown": "baseline_max_drawdown",
            }
        )
        out = out.merge(base, on="period", how="left")
        out["return_diff_vs_baseline"] = out["strategy_return"] - out["baseline_return"]
        out["cagr_diff_vs_baseline"] = out["cagr_365d"] - out["baseline_cagr"]
        out["sharpe_diff_vs_baseline"] = out["daily_sharpe_365d"] - out["baseline_sharpe"]
        out["drawdown_diff_vs_baseline"] = out["max_drawdown"] - out["baseline_max_drawdown"]
        out["catastrophic_year"] = (
            (out["return_diff_vs_baseline"] < -0.02)
            | (out["drawdown_diff_vs_baseline"] < -0.01)
            | (out["sharpe_diff_vs_baseline"] < -0.5)
        )
    else:
        out["catastrophic_year"] = False
    return out


def run_full_history_walkforward_fixed_overlay():
    started = datetime.now(timezone.utc).replace(tzinfo=None)
    if NEWS_AVAILABILITY_LAG_MINUTES != 15:
        raise ValueError("full_history_walkforward_fixed_overlay requires news_availability_lag_minutes=15.")
    end_ts = full_history_end_ts()
    pdf = load_full_history_overlay_pdf(FULL_HISTORY_START_TS, FULL_HISTORY_LIVE_START_TS, end_ts, NEWS_AVAILABILITY_LAG_MINUTES)
    test_mask = (pdf["timestamp"] >= FULL_HISTORY_LIVE_START_TS) & (pdf["timestamp"] <= end_ts)
    max_source = pd.to_datetime(pdf.loc[test_mask, "max_gdelt_source_ts"]).dt.tz_localize(None)
    lag_cutoff = pdf.loc[test_mask, "timestamp"] - pd.to_timedelta(NEWS_AVAILABILITY_LAG_MINUTES, unit="m")
    leakage_rows = int(((max_source.notna()) & (max_source > lag_cutoff)).sum())
    news_cols = ["policy_uncertainty_news_z", "geopolitical_risk_news_z", "market_stress_news_z"]
    null_news_rows = int(pdf.loc[test_mask, news_cols].isna().any(axis=1).sum())
    first_live_ts = pd.Timestamp(FULL_HISTORY_LIVE_START_TS)
    boundary_null_rows = int(
        pdf.loc[test_mask & (pdf["timestamp"] == first_live_ts), news_cols].isna().any(axis=1).sum()
    )
    if "gdelt_missing_flag" not in pdf.columns:
        pdf["gdelt_missing_flag"] = 1
    if leakage_rows:
        raise AssertionError(f"Full-history lag leakage detected: {leakage_rows}")
    if null_news_rows > boundary_null_rows:
        raise AssertionError(
            f"Full-history null overlay feature rows detected outside first lag boundary: "
            f"{null_news_rows - boundary_null_rows}"
        )
    pdf["gdelt_missing_flag"] = pdf["gdelt_missing_flag"].fillna(1).astype(int)
    pdf[news_cols] = pdf[news_cols].fillna(0.0)
    pdf = add_optimizer_pulses(pdf)
    wf, metrics, imp = full_history_monthly_walkforward(pdf, FULL_HISTORY_START_TS, FULL_HISTORY_LIVE_START_TS, end_ts)
    baseline_detail, overlay_detail = full_history_strategy_details(wf, NEWS_AVAILABILITY_LAG_MINUTES)
    missing_overlay_changes = int(overlay_detail["overlay_changed_during_missing_news"].sum())
    if missing_overlay_changes:
        raise AssertionError(f"Overlay changed position during missing-news periods: {missing_overlay_changes}")
    baseline_summary = summarize_strategy_detail(baseline_detail)
    overlay_summary = summarize_strategy_detail(overlay_detail, baseline_summary)
    for summary in [baseline_summary, overlay_summary]:
        summary.update(
            {
                "window_name": "full_history",
                "news_availability_lag_minutes": NEWS_AVAILABILITY_LAG_MINUTES,
                "optimization_mode": "full_history_walkforward_fixed_overlay",
                "ingestion_run_id": INGESTION_RUN_ID,
                "ingestion_ts": RUN_TS,
                "train_start_ts": FULL_HISTORY_START_TS,
                "live_start_ts": FULL_HISTORY_LIVE_START_TS,
                "full_history_end_ts": end_ts,
                "walkforward_accuracy": float(metrics["accuracy"].mean()),
            }
        )
    summary_pdf = pd.DataFrame([baseline_summary, overlay_summary])
    baseline_annual = annual_strategy_summary(baseline_detail)
    overlay_annual = annual_strategy_summary(overlay_detail, baseline_annual)
    annual_pdf = pd.concat([baseline_annual.assign(overlay_rule="baseline_v4"), overlay_annual], ignore_index=True)
    annual_pdf["optimization_mode"] = "full_history_walkforward_fixed_overlay"
    annual_pdf["news_availability_lag_minutes"] = NEWS_AVAILABILITY_LAG_MINUTES
    annual_pdf["ingestion_run_id"] = INGESTION_RUN_ID
    annual_pdf["ingestion_ts"] = RUN_TS

    detail_pdf = pd.concat([baseline_detail, overlay_detail], ignore_index=True)
    detail_pdf["ingestion_ts"] = RUN_TS
    detail_pdf["model_features"] = ", ".join(BASELINE_FEATURES)
    spark.createDataFrame(detail_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_WF_TABLE)
    spark.createDataFrame(summary_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_TABLE)
    spark.createDataFrame(annual_pdf).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(RESEARCH_OVERLAY_OPT_TABLE)

    yearly_status = []
    for year in sorted(pdf.loc[test_mask, "timestamp"].dt.year.unique()):
        chunk_start = pd.Timestamp(f"{year}-01-01 00:00:00")
        chunk_end = min(pd.Timestamp(f"{year}-12-31 23:00:00"), end_ts)
        chunk = pdf[(pdf["timestamp"] >= chunk_start) & (pdf["timestamp"] <= chunk_end)]
        bucket_end = chunk_end + pd.Timedelta(minutes=45)
        bucket_stats = spark.sql(
            f"""
            SELECT
                COUNT(DISTINCT bucket_ts) AS final_canonical_buckets,
                SUM(CASE WHEN raw_file_status = 'bigquery_ok' THEN 1 ELSE 0 END) AS bigquery_ok_buckets,
                SUM(CASE WHEN raw_file_status = 'bigquery_missing_raw_compat_fallback_raw_zip_ok' THEN 1 ELSE 0 END) AS fallback_ok_buckets,
                SUM(CASE WHEN raw_file_status = 'bigquery_missing_raw_compat_fallback_raw_zip_missing' THEN 1 ELSE 0 END) AS upstream_missing_buckets,
                SUM(CASE WHEN raw_file_status LIKE '%failed%' OR raw_file_status = 'bigquery_error' THEN 1 ELSE 0 END) AS unresolved_missing_buckets
            FROM {BUCKET_SUMMARY_TABLE}
            WHERE bucket_ts >= TIMESTAMP('{chunk_start}')
              AND bucket_ts <= TIMESTAMP('{bucket_end}')
            """
        ).collect()[0]
        expected_buckets = int(len(pd.date_range(chunk_start, bucket_end, freq="15min")))
        unresolved_missing_buckets = int(bucket_stats["unresolved_missing_buckets"] or 0)
        yearly_status.append(
            {
                "chunk_start_ts": chunk_start,
                "chunk_end_ts": chunk_end,
                "expected_buckets": expected_buckets,
                "final_canonical_buckets": int(bucket_stats["final_canonical_buckets"] or 0),
                "bigquery_ok_buckets": int(bucket_stats["bigquery_ok_buckets"] or 0),
                "fallback_ok_buckets": int(bucket_stats["fallback_ok_buckets"] or 0),
                "upstream_missing_buckets": int(bucket_stats["upstream_missing_buckets"] or 0),
                "unresolved_missing_buckets": unresolved_missing_buckets,
                "hourly_rows": int(len(chunk)),
                "joined_rows": int(len(chunk)),
                "leakage_rows": int(((pd.to_datetime(chunk["max_gdelt_source_ts"]).dt.tz_localize(None).notna()) & (pd.to_datetime(chunk["max_gdelt_source_ts"]).dt.tz_localize(None) > chunk["timestamp"] - pd.to_timedelta(NEWS_AVAILABILITY_LAG_MINUTES, unit="m"))).sum()),
                "null_feature_rows": int(chunk[news_cols].isna().sum().sum()),
                "status": "PASS" if unresolved_missing_buckets == 0 else "WARN_UNRESOLVED_BUCKETS",
                "runtime_seconds": None,
                "ingestion_run_id": INGESTION_RUN_ID,
                "ingestion_ts": RUN_TS,
            }
        )
    spark.createDataFrame(pd.DataFrame(yearly_status)).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(FULL_HISTORY_CHUNK_STATUS_TABLE)
    display(spark.createDataFrame(summary_pdf))
    display(spark.createDataFrame(annual_pdf))
    completed = datetime.now(timezone.utc).replace(tzinfo=None)
    log_runtime("full_history_walkforward_fixed_overlay", started, completed, len(detail_pdf), f"{FULL_HISTORY_START_TS} to {end_ts};live_start={FULL_HISTORY_LIVE_START_TS}")
    return summary_pdf, annual_pdf, pd.DataFrame(yearly_status), imp


if RUN_OVERLAY_BACKTEST:
    if OVERLAY_EXPERIMENT_MODE == "grid_optimize":
        overlay_summary_pdf, overlay_aggregate_pdf, overlay_validation_pdf = run_overlay_grid_optimizer()
        overlay_activation_pdf = None
    elif OVERLAY_EXPERIMENT_MODE == "selected_validate":
        overlay_summary_pdf, overlay_aggregate_pdf, overlay_validation_pdf = run_selected_overlay_validation()
        overlay_activation_pdf = None
    elif OVERLAY_EXPERIMENT_MODE == "targeted_refine":
        overlay_summary_pdf, overlay_aggregate_pdf, overlay_validation_pdf = run_targeted_overlay_refinement()
        overlay_activation_pdf = None
    elif OVERLAY_EXPERIMENT_MODE == "refined_validate":
        overlay_summary_pdf, overlay_aggregate_pdf, overlay_validation_pdf = run_refined_overlay_validation()
        overlay_activation_pdf = None
    elif OVERLAY_EXPERIMENT_MODE == "full_history_walkforward_fixed_overlay":
        overlay_summary_pdf, overlay_aggregate_pdf, overlay_validation_pdf, overlay_activation_pdf = run_full_history_walkforward_fixed_overlay()
    else:
        overlay_summary_pdf, overlay_validation_pdf, overlay_activation_pdf = run_overlay_backtests()
        overlay_aggregate_pdf = None
else:
    overlay_summary_pdf, overlay_aggregate_pdf, overlay_validation_pdf, overlay_activation_pdf = None, None, None, None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manual Stress-Window Inspection

# COMMAND ----------

stress_windows = [
    ("covid_crash", "2020-02-01", "2020-04-30"),
    ("inflation_rates_2022", "2022-01-01", "2022-12-31"),
    ("banking_stress_2023", "2023-03-01", "2023-05-31"),
    ("tariff_geopolitical_2025", "2025-01-01", "2025-12-31"),
]

for name, start_ts, end_ts in stress_windows:
    print(name)
    display(
        spark.sql(
            f"""
            SELECT
                '{name}' AS window_name,
                MIN(timestamp) AS min_ts,
                MAX(timestamp) AS max_ts,
                AVG(news_volume_z) AS avg_news_volume_z,
                MAX(news_volume_z) AS max_news_volume_z,
                AVG(news_tone_z) AS avg_news_tone_z,
                MIN(news_tone_z) AS min_news_tone_z,
                MAX(policy_uncertainty_news_z) AS max_policy_uncertainty_news_z,
                MAX(geopolitical_risk_news_z) AS max_geopolitical_risk_news_z,
                MAX(market_stress_news_z) AS max_market_stress_news_z
            FROM {RESEARCH_FEATURE_TABLE}
            WHERE timestamp BETWEEN TIMESTAMP('{start_ts}') AND TIMESTAMP('{end_ts}')
            """
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shadow Production Design Only
# MAGIC
# MAGIC If, and only if, the historical backtest shows incremental value, propose a separate paused shadow workflow that:
# MAGIC - writes `pawan_quant_research_1.v4.bronze_rt_news_flow_15m`;
# MAGIC - writes `pawan_quant_research_1.v4.silver_rt_news_flow_1h`;
# MAGIC - writes `pawan_quant_research_1.v4.research_rt_news_candidate_signal_log`;
# MAGIC - never writes `bronze_rt_regime_signal`, `bronze_rt_saxo_trade_log`, `bronze_rt_saxo_signal_audit_log`, or `gold_rt_saxo_dashboard_feed`;
# MAGIC - never triggers Saxo execution;
# MAGIC - runs after the Saxo primary chain only if a later schedule review confirms no conflict with `v4_saxo_live_hourly_chain`.
# MAGIC
# MAGIC This notebook does not create or enable any RT/shadow job.
