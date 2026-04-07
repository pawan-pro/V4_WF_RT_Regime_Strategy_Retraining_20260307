# Databricks notebook source
# RT_07: Dashboard Feed

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config

from datetime import datetime, timezone

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

SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
SILVER_RT = f"{CATALOG}.{SCHEMA}.silver_rt_1h"
TRADE_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_paper_trade_log"
AUDIT_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_signal_audit_log"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
MACRO_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.gold_rt_dashboard_feed"
runtime_config = load_runtime_instrument_config(spark, CATALOG, SCHEMA)
PRIMARY_SYMBOL = runtime_config["primary_symbol"]
SECONDARY_SYMBOL = runtime_config["secondary_symbol"]
PRICE_SYMBOLS_SQL = ", ".join([f"'{symbol}'" for symbol in {PRIMARY_SYMBOL, SECONDARY_SYMBOL} if symbol])


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def maybe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def get_model_version(default_version):
    if spark.catalog.tableExists(MODEL_REGISTRY_TABLE):
        rows = spark.sql(
            f"""
            SELECT model_version
            FROM {MODEL_REGISTRY_TABLE}
            ORDER BY promoted_ts DESC
            LIMIT 1
            """
        ).collect()
        if rows:
            return rows[0]["model_version"]
    return default_version


signal = spark.sql(
    f"""
    SELECT *
    FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC
    LIMIT 1
    """
).collect()[0]
signal_data = signal.asDict(recursive=True)

silver = spark.sql(
    f"""
    SELECT *
    FROM {SILVER_RT}
    ORDER BY timestamp DESC
    LIMIT 1
    """
).collect()[0]

trade = spark.sql(
    f"""
    SELECT *
    FROM {TRADE_LOG}
    ORDER BY trade_ts DESC
    LIMIT 1
    """
).collect()[0]

price_ts = spark.sql(
    f"""
    SELECT MAX(timestamp) AS max_ts
    FROM {MARKET_TABLE}
    """
).collect()[0]["max_ts"]

prices = spark.sql(
    f"""
    SELECT symbol, close
    FROM {MARKET_TABLE}
    WHERE symbol IN ({PRICE_SYMBOLS_SQL})
      AND timestamp = '{price_ts}'
    """
).collect()
price_map = {row["symbol"]: maybe_float(row["close"]) for row in prices}

portfolio_stats = spark.sql(
    f"""
    SELECT
        COUNT(*) AS total_cycles,
        SUM(CASE WHEN switched THEN 1 ELSE 0 END) AS total_switches,
        MAX(account_equity) AS peak_equity
    FROM {TRADE_LOG}
    """
).collect()[0]

starting_equity_row = spark.sql(
    f"""
    SELECT account_equity
    FROM {TRADE_LOG}
    WHERE account_equity IS NOT NULL
    ORDER BY trade_ts ASC
    LIMIT 1
    """
).collect()
starting_equity = maybe_float(starting_equity_row[0]["account_equity"]) if starting_equity_row else None
latest_equity = maybe_float(trade["account_equity"])
peak_equity = maybe_float(portfolio_stats["peak_equity"])

total_return_pct = None
if starting_equity not in (None, 0.0) and latest_equity is not None:
    total_return_pct = latest_equity / starting_equity - 1.0

drawdown_pct = None
if peak_equity not in (None, 0.0) and latest_equity is not None:
    drawdown_pct = latest_equity / peak_equity - 1.0

regime_dist = spark.sql(
    f"""
    SELECT
        ROUND(100.0 * SUM(CASE WHEN regime_int = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_expansion,
        ROUND(100.0 * SUM(CASE WHEN regime_int = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_slowdown,
        ROUND(100.0 * SUM(CASE WHEN regime_int = 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_recession
    FROM {AUDIT_LOG}
    """
).collect()[0]

macro_last_release_ts = spark.sql(
    f"""
    SELECT MAX(timestamp) AS macro_last_release_ts
    FROM {MACRO_TABLE}
    """
).collect()[0]["macro_last_release_ts"]

vix_source_date = spark.sql(
    f"""
    SELECT MAX(CAST(timestamp AS DATE)) AS vix_source_date
    FROM {CATALOG}.{SCHEMA}.bronze_rt_vix_1d
    WHERE series = 'VIXCLS'
    """
).collect()[0]["vix_source_date"]

dashboard_ts = datetime.now(timezone.utc).replace(tzinfo=None)
model_version = get_model_version(signal_data.get("model_version") or "bootstrap_live_model")

schema = StructType(
    [
        StructField("feed_ts", TimestampType(), True),
        StructField("dashboard_ts", TimestampType(), True),
        StructField("signal_ts", TimestampType(), True),
        StructField("trade_ts", TimestampType(), True),
        StructField("price_ts", TimestampType(), True),
        StructField("macro_last_release_ts", TimestampType(), True),
        StructField("regime_int", IntegerType(), True),
        StructField("regime_name", StringType(), True),
        StructField("position", StringType(), True),
        StructField("target_ticker", StringType(), True),
        StructField("VIXCLS_z", DoubleType(), True),
        StructField("vix3m_z", DoubleType(), True),
        StructField("VIX_Inversion_Z", DoubleType(), True),
        StructField("Price_Mom_24H", DoubleType(), True),
        StructField("CPIAUCSL_z", DoubleType(), True),
        StructField("PCEPI_z", DoubleType(), True),
        StructField("PCE_z", DoubleType(), True),
        StructField("PAYEMS_z", DoubleType(), True),
        StructField("UNRATE_z", DoubleType(), True),
        StructField("GDPC1_z", DoubleType(), True),
        StructField("INDPRO_z", DoubleType(), True),
        StructField("UMCSENT_z", DoubleType(), True),
        StructField("CPI_Event_Active", IntegerType(), True),
        StructField("NFP_Event_Active", IntegerType(), True),
        StructField("SPY_close", DoubleType(), True),
        StructField("QQQ_close", DoubleType(), True),
        StructField("account_equity", DoubleType(), True),
        StructField("account_cash", DoubleType(), True),
        StructField("positions_json", StringType(), True),
        StructField("positions_count", IntegerType(), True),
        StructField("positions_market_value", DoubleType(), True),
        StructField("positions_unrealized_pl", DoubleType(), True),
        StructField("positions_unrealized_plpc", DoubleType(), True),
        StructField("day_pnl", DoubleType(), True),
        StructField("day_pnl_pct", DoubleType(), True),
        StructField("total_return_pct", DoubleType(), True),
        StructField("drawdown_pct", DoubleType(), True),
        StructField("total_cycles", IntegerType(), True),
        StructField("total_switches", IntegerType(), True),
        StructField("peak_equity", DoubleType(), True),
        StructField("pct_expansion", DoubleType(), True),
        StructField("pct_slowdown", DoubleType(), True),
        StructField("pct_recession", DoubleType(), True),
        StructField("vix_source_date", StringType(), True),
        StructField("model_version", StringType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

dashboard_row = {
    "feed_ts": dashboard_ts,
    "dashboard_ts": dashboard_ts,
    "signal_ts": signal["timestamp"],
    "trade_ts": trade["trade_ts"],
    "price_ts": price_ts,
    "macro_last_release_ts": macro_last_release_ts,
    "regime_int": signal["regime_int"],
    "regime_name": signal["regime_name"],
    "position": signal["position"],
    "target_ticker": trade["target_ticker"],
    "VIXCLS_z": maybe_float(silver["VIXCLS_z"]),
    "vix3m_z": maybe_float(silver["vix3m_z"]),
    "VIX_Inversion_Z": maybe_float(silver["VIX_Inversion_Z"]),
    "Price_Mom_24H": maybe_float(silver["Price_Mom_24H"]),
    "CPIAUCSL_z": maybe_float(silver["CPIAUCSL_z"]),
    "PCEPI_z": maybe_float(silver["PCEPI_z"]),
    "PCE_z": maybe_float(silver["PCE_z"]),
    "PAYEMS_z": maybe_float(silver["PAYEMS_z"]),
    "UNRATE_z": maybe_float(silver["UNRATE_z"]),
    "GDPC1_z": maybe_float(silver["GDPC1_z"]),
    "INDPRO_z": maybe_float(silver["INDPRO_z"]),
    "UMCSENT_z": maybe_float(silver["UMCSENT_z"]),
    "CPI_Event_Active": maybe_int(silver["CPI_Event_Active"]),
    "NFP_Event_Active": maybe_int(silver["NFP_Event_Active"]),
    "SPY_close": price_map.get(PRIMARY_SYMBOL),
    "QQQ_close": price_map.get(SECONDARY_SYMBOL),
    "account_equity": latest_equity,
    "account_cash": maybe_float(trade["account_cash"]),
    "positions_json": trade["positions_json"],
    "positions_count": maybe_int(trade["positions_count"]),
    "positions_market_value": maybe_float(trade["positions_market_value"]),
    "positions_unrealized_pl": maybe_float(trade["positions_unrealized_pl"]),
    "positions_unrealized_plpc": maybe_float(trade["positions_unrealized_plpc"]),
    "day_pnl": maybe_float(trade["day_pnl"]),
    "day_pnl_pct": maybe_float(trade["day_pnl_pct"]),
    "total_return_pct": total_return_pct,
    "drawdown_pct": drawdown_pct,
    "total_cycles": maybe_int(portfolio_stats["total_cycles"]),
    "total_switches": maybe_int(portfolio_stats["total_switches"]),
    "peak_equity": peak_equity,
    "pct_expansion": maybe_float(regime_dist["pct_expansion"]),
    "pct_slowdown": maybe_float(regime_dist["pct_slowdown"]),
    "pct_recession": maybe_float(regime_dist["pct_recession"]),
    "vix_source_date": str(vix_source_date) if vix_source_date is not None else None,
    "model_version": model_version,
    "source": "rt_dashboard_feed_v4",
    "ingestion_ts": dashboard_ts,
}

sdf = spark.createDataFrame([dashboard_row], schema=schema)
sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TARGET_TABLE)

print(f"Dashboard feed written to {TARGET_TABLE}")
print(f"Dashboard timestamp: {dashboard_ts}")
print(f"Regime: {dashboard_row['regime_name']} ({dashboard_row['regime_int']})")
print(f"Target allocation: {dashboard_row['position']}")
print(f"Account equity: {dashboard_row['account_equity']}")
print(f"Day PnL: {dashboard_row['day_pnl']}")
print(f"Total return pct: {dashboard_row['total_return_pct']}")
print(f"Drawdown pct: {dashboard_row['drawdown_pct']}")
print(f"Model version: {model_version}")
