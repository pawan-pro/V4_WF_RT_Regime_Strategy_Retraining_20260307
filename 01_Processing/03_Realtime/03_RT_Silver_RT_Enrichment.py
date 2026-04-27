# Databricks notebook source
# RT_03: Silver RT Enrichment
# Produces one inference-ready row matching silver_bt_master_grid_1h schema.

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config
# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

from datetime import date, timedelta
import math

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

TARGET_TABLE = f"{CATALOG}.{SCHEMA}.silver_rt_1h"
SILVER_HIST = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
BRONZE_BT_MARKET = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"
BRONZE_RT_MARKET = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
BRONZE_BT_VIX = f"{CATALOG}.{SCHEMA}.bronze_bt_vix_1h"
BRONZE_BT_VIX3M = f"{CATALOG}.{SCHEMA}.bronze_bt_vix3m_1h"
BRONZE_RT_VIX = f"{CATALOG}.{SCHEMA}.bronze_rt_vix_1d"
LIVE_MACRO_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_macro_events_live"
BRONZE_BT_MACRO_EVENTS = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"

VIX_Z_WINDOW_HOURS = 8760
PRICE_CONTEXT_HOURS = 72
LIVE_MACRO_INDICATORS = {
    "CPIAUCSL": "CPIAUCSL",
    "UMCSENT": "UMCSENT",
    "PCEPI": "PCEPI",
    "PCE": "PCE",
    "GDPC1": "GDPC1",
}

runtime_config = load_runtime_instrument_config(spark)
PRIMARY_SYMBOL = runtime_config["primary_symbol"]


def ensure_target_table():
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            timestamp TIMESTAMP,
            SPY_close DOUBLE,
            CPIAUCSL_z DOUBLE,
            PCEPI_z DOUBLE,
            PCE_z DOUBLE,
            PAYEMS_z DOUBLE,
            UNRATE_z DOUBLE,
            GDPC1_z DOUBLE,
            INDPRO_z DOUBLE,
            UMCSENT_z DOUBLE,
            VIXCLS_z DOUBLE,
            vix3m_z DOUBLE,
            VIX_Inversion_Z DOUBLE,
            Price_Mom_24H DOUBLE,
            CPI_Event_Active INT,
            NFP_Event_Active INT,
            source STRING
        )
        USING DELTA
        """
    )


def existing_columns(table_name):
    if not spark.catalog.tableExists(table_name):
        return set()
    return {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {table_name}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }


def load_latest_rt_inputs():
    spy_row = spark.sql(
        f"""
        SELECT timestamp, close
        FROM {BRONZE_RT_MARKET}
        WHERE symbol = '{PRIMARY_SYMBOL}'
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).collect()
    if not spy_row:
        raise ValueError(f"No realtime {PRIMARY_SYMBOL} rows found in {BRONZE_RT_MARKET}")

    vix_row = spark.sql(
        f"""
        SELECT timestamp, close
        FROM {BRONZE_RT_VIX}
        WHERE series = 'VIXCLS'
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).collect()
    if not vix_row:
        raise ValueError("No VIXCLS rows found in bronze_rt_vix_1d")

    vix3m_row = spark.sql(
        f"""
        SELECT timestamp, close
        FROM {BRONZE_RT_VIX}
        WHERE series = 'vix3m'
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).collect()
    if not vix3m_row:
        raise ValueError("No vix3m rows found in bronze_rt_vix_1d")

    spy_row = spy_row[0]
    vix_row = vix_row[0]
    vix3m_row = vix3m_row[0]

    rt_ts = spy_row["timestamp"].replace(minute=0, second=0, microsecond=0)
    vix_latest_date = vix_row["timestamp"]
    if vix_latest_date.date() >= date.today():
        raise AssertionError(
            f"LOOKAHEAD BIAS: VIX date {vix_latest_date.date()} >= today {date.today()}"
        )

    return rt_ts, float(spy_row["close"]), float(vix_row["close"]), float(vix3m_row["close"]), vix_latest_date


def load_macro_state(rt_ts):
    rows = spark.sql(
        f"""
        SELECT
            timestamp,
            CPIAUCSL_z,
            PCEPI_z,
            PCE_z,
            PAYEMS_z,
            UNRATE_z,
            GDPC1_z,
            INDPRO_z,
            UMCSENT_z,
            CPI_Event_Active,
            NFP_Event_Active
        FROM {SILVER_HIST}
        WHERE timestamp <= TIMESTAMP('{rt_ts}')
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).collect()
    if not rows:
        raise ValueError(f"No macro carry-forward row found in {SILVER_HIST}")
    return rows[0]


def load_live_macro_level(indicator_id, rt_ts):
    if not spark.catalog.tableExists(LIVE_MACRO_TABLE):
        return None

    gating_filter = ""
    if indicator_id == "UMCSENT":
        gating_filter = (
            "AND release_stage = 'final' "
            "AND validation_status = 'validated' "
        )

    rows = spark.sql(
        f"""
        SELECT
            provider_event_name,
            observed_value,
            status,
            source_provider,
            release_stage,
            validation_status,
            scheduled_release_ts
        FROM {LIVE_MACRO_TABLE}
        WHERE indicator_id = '{indicator_id}'
          AND observed_value IS NOT NULL
          AND status IN ('observed', 'reconciled_observed')
          AND validation_status = 'validated'
          {gating_filter}
          AND scheduled_release_ts <= TIMESTAMP('{rt_ts}')
        ORDER BY scheduled_release_ts DESC, ingestion_ts DESC
        LIMIT 1
        """
    ).collect()
    if not rows:
        return None
    return rows[0]


def compute_live_macro_zscore_from_silver(table_name, value_column, current_value, rt_ts):
    history_pdf = spark.sql(
        f"""
        SELECT timestamp, {value_column} AS value
        FROM {table_name}
        WHERE timestamp < TIMESTAMP('{rt_ts}')
          AND {value_column} IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT {VIX_Z_WINDOW_HOURS - 1}
        """
    ).toPandas()

    if history_pdf.empty:
        raise ValueError(f"Insufficient history in {table_name} to compute {value_column} z-score")

    values = history_pdf["value"].astype(float).tolist()
    values.reverse()
    values.append(float(current_value))

    series = pd.Series(values, dtype="float64")
    mean_val = float(series.mean())
    std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    if math.isnan(std_val) or std_val == 0.0:
        return 0.0
    return (float(current_value) - mean_val) / std_val


def bronze_macro_value_column():
    columns = existing_columns(BRONZE_BT_MACRO_EVENTS)
    if "actual" in columns:
        return "actual"
    if "observed_value" in columns:
        return "observed_value"
    raise ValueError(f"Unsupported schema for {BRONZE_BT_MACRO_EVENTS}: expected actual or observed_value")


def compute_live_macro_zscore_from_bronze(indicator_id, current_value, rt_ts):
    value_column = bronze_macro_value_column()
    history_start = pd.Timestamp(rt_ts) - pd.Timedelta(hours=VIX_Z_WINDOW_HOURS - 1)
    history_pdf = spark.sql(
        f"""
        WITH hourly_grid AS (
            SELECT explode(sequence(
                TIMESTAMP('{history_start}'),
                TIMESTAMP('{rt_ts}') - INTERVAL 1 HOUR,
                INTERVAL 1 HOUR
            )) AS grid_ts
        ),
        macro_releases AS (
            SELECT
                timestamp + INTERVAL 2 HOURS AS available_ts,
                {value_column} AS value
            FROM {BRONZE_BT_MACRO_EVENTS}
            WHERE indicator_id = '{indicator_id}'
              AND {value_column} IS NOT NULL
              AND timestamp + INTERVAL 2 HOURS <= TIMESTAMP('{rt_ts}')
        ),
        asof_values AS (
            SELECT
                g.grid_ts AS timestamp,
                m.value,
                ROW_NUMBER() OVER (
                    PARTITION BY g.grid_ts
                    ORDER BY m.available_ts DESC
                ) AS rn
            FROM hourly_grid g
            LEFT JOIN macro_releases m
              ON m.available_ts <= g.grid_ts
        )
        SELECT timestamp, value
        FROM asof_values
        WHERE rn = 1
          AND value IS NOT NULL
        ORDER BY timestamp ASC
        """
    ).toPandas()

    if history_pdf.empty:
        raise ValueError(f"Insufficient PIT history in {BRONZE_BT_MACRO_EVENTS} to compute {indicator_id} z-score")

    values = history_pdf["value"].astype(float).tolist()
    values.append(float(current_value))

    series = pd.Series(values, dtype="float64")
    mean_val = float(series.mean())
    std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    if math.isnan(std_val) or std_val == 0.0:
        return 0.0
    return (float(current_value) - mean_val) / std_val


def compute_live_macro_zscore(indicator_id, value_column, current_value, rt_ts):
    silver_columns = existing_columns(SILVER_HIST)
    if value_column in silver_columns:
        print(f"Live macro z-score history for {indicator_id}: using {SILVER_HIST}.{value_column}")
        return compute_live_macro_zscore_from_silver(SILVER_HIST, value_column, current_value, rt_ts)

    print(
        f"Live macro z-score history for {indicator_id}: {value_column} not found in {SILVER_HIST}; "
        f"using PIT reconstruction from {BRONZE_BT_MACRO_EVENTS}"
    )
    return compute_live_macro_zscore_from_bronze(indicator_id, current_value, rt_ts)


def load_market_history(rt_ts, lookback_hours):
    history_start = rt_ts - timedelta(hours=lookback_hours)
    union_pdf = spark.sql(
        f"""
        SELECT timestamp, close, source_rank
        FROM (
            SELECT timestamp, close, 0 AS source_rank
            FROM {BRONZE_BT_MARKET}
            WHERE symbol = '{PRIMARY_SYMBOL}'
              AND timestamp >= TIMESTAMP('{history_start}')
              AND timestamp <= TIMESTAMP('{rt_ts}')

            UNION ALL

            SELECT timestamp, close, 1 AS source_rank
            FROM {BRONZE_RT_MARKET}
            WHERE symbol = '{PRIMARY_SYMBOL}'
              AND timestamp >= TIMESTAMP('{history_start}')
              AND timestamp <= TIMESTAMP('{rt_ts}')
        )
        ORDER BY timestamp ASC, source_rank ASC
        """
    ).toPandas()

    anchor_rows = spark.sql(
        f"""
        SELECT timestamp, close, source_rank
        FROM (
            SELECT timestamp, close, 0 AS source_rank
            FROM {BRONZE_BT_MARKET}
            WHERE symbol = '{PRIMARY_SYMBOL}'
              AND timestamp < TIMESTAMP('{history_start}')

            UNION ALL

            SELECT timestamp, close, 1 AS source_rank
            FROM {BRONZE_RT_MARKET}
            WHERE symbol = '{PRIMARY_SYMBOL}'
              AND timestamp < TIMESTAMP('{history_start}')
        )
        ORDER BY timestamp DESC, source_rank DESC
        LIMIT 1
        """
    ).collect()

    if union_pdf.empty and not anchor_rows:
        raise ValueError(f"No market history found for {PRIMARY_SYMBOL}")

    if anchor_rows:
        anchor_pdf = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(history_start),
                    "close": float(anchor_rows[0]["close"]),
                    "source_rank": int(anchor_rows[0]["source_rank"]),
                }
            ]
        )
        union_pdf = pd.concat([anchor_pdf, union_pdf], ignore_index=True)

    union_pdf["timestamp"] = pd.to_datetime(union_pdf["timestamp"])
    union_pdf = union_pdf.sort_values(["timestamp", "source_rank"])
    union_pdf = union_pdf.drop_duplicates(subset=["timestamp"], keep="last")

    hourly_index = pd.date_range(start=history_start, end=rt_ts, freq="h")
    market_series = union_pdf.set_index("timestamp")["close"].astype(float).reindex(hourly_index).ffill()
    if market_series.isna().any():
        market_series = market_series.bfill()
    return market_series


def compute_live_price_momentum(rt_ts, current_spy_close):
    market_series = load_market_history(rt_ts, PRICE_CONTEXT_HOURS)
    lag_ts = pd.Timestamp(rt_ts) - pd.Timedelta(hours=24)
    lag_close = market_series.loc[lag_ts]
    current_close = float(current_spy_close)
    if lag_close is None or pd.isna(lag_close):
        raise ValueError(f"Missing 24H lag close for {PRIMARY_SYMBOL} at {lag_ts}")
    return current_close / float(lag_close) - 1.0


def resolve_value_column(table_name):
    columns = existing_columns(table_name)
    if "close" in columns:
        return "close"
    if "vix3m" in columns:
        return "vix3m"
    raise ValueError(f"Unsupported schema for {table_name}: expected close or vix3m column")


def compute_live_zscore(table_name, value_column, current_value, rt_ts):
    history_pdf = spark.sql(
        f"""
        SELECT timestamp, {value_column} AS value
        FROM {table_name}
        WHERE timestamp < TIMESTAMP('{rt_ts}')
        ORDER BY timestamp DESC
        LIMIT {VIX_Z_WINDOW_HOURS - 1}
        """
    ).toPandas()

    if history_pdf.empty:
        raise ValueError(f"Insufficient history in {table_name} to compute z-score")

    values = history_pdf["value"].astype(float).tolist()
    values.reverse()
    values.append(float(current_value))

    series = pd.Series(values, dtype="float64")
    mean_val = float(series.mean())
    std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    if math.isnan(std_val) or std_val == 0.0:
        return 0.0
    return (float(current_value) - mean_val) / std_val


def to_py_float(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return float(value)


ensure_target_table()

rt_ts, spy_close, vix_close, vix3m_close, vix_latest_date = load_latest_rt_inputs()
macro_state = load_macro_state(rt_ts)
live_macro_source = {indicator_id: "bt_silver" for indicator_id in LIVE_MACRO_INDICATORS}
macro_features = {
    "CPIAUCSL_z": to_py_float(macro_state["CPIAUCSL_z"]),
    "PCEPI_z": to_py_float(macro_state["PCEPI_z"]),
    "PCE_z": to_py_float(macro_state["PCE_z"]),
    "GDPC1_z": to_py_float(macro_state["GDPC1_z"]),
    "UMCSENT_z": to_py_float(macro_state["UMCSENT_z"]),
    "CPI_Event_Active": int(macro_state["CPI_Event_Active"]),
    "NFP_Event_Active": int(macro_state["NFP_Event_Active"]),
}

for indicator_id, raw_col in LIVE_MACRO_INDICATORS.items():
    live_row = load_live_macro_level(indicator_id, rt_ts)
    if not live_row:
        print(
            f"Live macro override skipped | indicator_id={indicator_id} "
            "provider_event_name=N/A source_provider=N/A status=N/A "
            "release_stage=N/A validation_status=N/A scheduled_release_ts=N/A "
            "observed_value=N/A z_score=N/A reason=no_valid_observed_live_row"
        )
        continue

    observed_value = to_py_float(live_row["observed_value"])
    if observed_value is None:
        print(
            f"Live macro override skipped | indicator_id={indicator_id} "
            f"provider_event_name={live_row['provider_event_name']} "
            f"source_provider={live_row['source_provider']} "
            f"status={live_row['status']} "
            f"release_stage={live_row['release_stage']} "
            f"validation_status={live_row['validation_status']} "
            f"scheduled_release_ts={live_row['scheduled_release_ts']} "
            "observed_value=N/A z_score=N/A reason=missing_observed_value"
        )
        continue

    try:
        macro_state_key = f"{indicator_id}_z"
        macro_raw_z = compute_live_macro_zscore(indicator_id, raw_col, observed_value, rt_ts)
        macro_features[macro_state_key] = to_py_float(macro_raw_z)
        live_macro_source[indicator_id] = f"live_{live_row['source_provider']}"
        print(
            f"Live macro override applied | indicator_id={indicator_id} "
            f"provider_event_name={live_row['provider_event_name']} "
            f"source_provider={live_row['source_provider']} "
            f"status={live_row['status']} "
            f"release_stage={live_row['release_stage']} "
            f"validation_status={live_row['validation_status']} "
            f"scheduled_release_ts={live_row['scheduled_release_ts']} "
            f"observed_value={observed_value} z_score={macro_raw_z:.6f}"
        )
    except Exception as exc:
        print(
            f"Live macro override skipped | indicator_id={indicator_id} "
            f"provider_event_name={live_row['provider_event_name']} "
            f"source_provider={live_row['source_provider']} "
            f"status={live_row['status']} "
            f"release_stage={live_row['release_stage']} "
            f"validation_status={live_row['validation_status']} "
            f"scheduled_release_ts={live_row['scheduled_release_ts']} "
            f"observed_value={observed_value} z_score=N/A reason={exc}"
        )

macro_source = (
    "rt_enrichment_live_recompute+"
    f"{','.join([k for k, v in live_macro_source.items() if v.startswith('live_')])}"
)
if macro_source == "rt_enrichment_live_recompute+":
    macro_source = "rt_enrichment_live_recompute"
price_mom_24h = compute_live_price_momentum(rt_ts, spy_close)
vixcls_z_rt = compute_live_zscore(BRONZE_BT_VIX, "close", vix_close, rt_ts)
vix3m_z_rt = compute_live_zscore(
    BRONZE_BT_VIX3M,
    resolve_value_column(BRONZE_BT_VIX3M),
    vix3m_close,
    rt_ts,
)
vix_inversion_z_rt = vixcls_z_rt - vix3m_z_rt

print(
    f"RT inputs -> ts:{rt_ts} {PRIMARY_SYMBOL}:{spy_close:.4f} "
    f"VIX:{vix_close:.4f} VIX3M:{vix3m_close:.4f}"
)
print(
    "Live recompute -> "
    f"Price_Mom_24H:{price_mom_24h:.6f} "
    f"VIXCLS_z:{vixcls_z_rt:.6f} "
    f"vix3m_z:{vix3m_z_rt:.6f} "
    f"VIX_Inversion_Z:{vix_inversion_z_rt:.6f}"
)
print(f"Bias gate passed | VIX date: {vix_latest_date.date()}")

rt_row = {
    "timestamp": rt_ts,
    "SPY_close": to_py_float(spy_close),
    "CPIAUCSL_z": macro_features["CPIAUCSL_z"],
    "PCEPI_z": macro_features["PCEPI_z"],
    "PCE_z": macro_features["PCE_z"],
    "PAYEMS_z": to_py_float(macro_state["PAYEMS_z"]),
    "UNRATE_z": to_py_float(macro_state["UNRATE_z"]),
    "GDPC1_z": macro_features["GDPC1_z"],
    "INDPRO_z": to_py_float(macro_state["INDPRO_z"]),
    "UMCSENT_z": macro_features["UMCSENT_z"],
    # Phase 1 live macro overrides can recompute selected macro z-scores from validated Finnhub-live values.
    "VIXCLS_z": to_py_float(vixcls_z_rt),
    "vix3m_z": to_py_float(vix3m_z_rt),
    "VIX_Inversion_Z": to_py_float(vix_inversion_z_rt),
    "Price_Mom_24H": to_py_float(price_mom_24h),
    "CPI_Event_Active": macro_features["CPI_Event_Active"],
    "NFP_Event_Active": macro_features["NFP_Event_Active"],
    "source": macro_source,
}

schema = StructType(
    [
        StructField("timestamp", TimestampType(), True),
        StructField("SPY_close", DoubleType(), True),
        StructField("CPIAUCSL_z", DoubleType(), True),
        StructField("PCEPI_z", DoubleType(), True),
        StructField("PCE_z", DoubleType(), True),
        StructField("PAYEMS_z", DoubleType(), True),
        StructField("UNRATE_z", DoubleType(), True),
        StructField("GDPC1_z", DoubleType(), True),
        StructField("INDPRO_z", DoubleType(), True),
        StructField("UMCSENT_z", DoubleType(), True),
        StructField("VIXCLS_z", DoubleType(), True),
        StructField("vix3m_z", DoubleType(), True),
        StructField("VIX_Inversion_Z", DoubleType(), True),
        StructField("Price_Mom_24H", DoubleType(), True),
        StructField("CPI_Event_Active", IntegerType(), True),
        StructField("NFP_Event_Active", IntegerType(), True),
        StructField("source", StringType(), True),
    ]
)

sdf = spark.createDataFrame([rt_row], schema=schema)
sdf.createOrReplaceTempView("incoming_silver_rt")

merge_sql = f"""
    MERGE INTO {TARGET_TABLE} t
    USING incoming_silver_rt s
    ON t.timestamp = s.timestamp
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
"""
execute_merge_sql_with_retry(
    merge_sql,
    operation_name=f"merge enriched realtime row into {TARGET_TABLE}",
)

print(f"RT Silver row written: {rt_ts}")
spark.sql(f"SELECT * FROM {TARGET_TABLE} ORDER BY timestamp DESC LIMIT 1").show(vertical=True)
