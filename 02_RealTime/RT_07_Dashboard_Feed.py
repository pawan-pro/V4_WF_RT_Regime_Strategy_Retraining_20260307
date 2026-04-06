# Databricks notebook source
# RT_07: Dashboard Feed
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timezone

SIGNAL_TABLE = "pawan_quant_research_1.v4.bronze_rt_regime_signal"
SILVER_RT    = "pawan_quant_research_1.v4.silver_rt_1h"
TRADE_LOG    = "pawan_quant_research_1.v4.bronze_rt_paper_trade_log"
AUDIT_LOG    = "pawan_quant_research_1.v4.bronze_rt_signal_audit_log"
MARKET_TABLE = "pawan_quant_research_1.v4.bronze_rt_market_1h"
TARGET_TABLE = "pawan_quant_research_1.v4.gold_rt_dashboard_feed"

# STEP 1: Latest signal
signal = spark.sql(f"""
    SELECT * FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]

# STEP 2: Latest silver row (for VIX_Inversion_Z)
silver = spark.sql(f"""
    SELECT VIX_Inversion_Z, Price_Mom_24H FROM {SILVER_RT}
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]

# STEP 3: Latest trade
trade = spark.sql(f"""
    SELECT * FROM {TRADE_LOG}
    ORDER BY trade_ts DESC LIMIT 1
""").collect()[0]

# STEP 4: Latest prices
price_ts_row = spark.sql(f"""
    SELECT MAX(timestamp) as max_ts FROM {MARKET_TABLE}
""").collect()[0]["max_ts"]

prices = spark.sql(f"""
    SELECT symbol, close FROM {MARKET_TABLE}
    WHERE symbol IN ('SPY','QQQ')
    AND timestamp = '{price_ts_row}'
""").collect()
price_map = {r["symbol"]: float(r["close"]) for r in prices}

# STEP 5: Portfolio stats
portfolio_stats = spark.sql(f"""
    SELECT
        COUNT(*)                                   as total_cycles,
        SUM(CASE WHEN switched THEN 1 ELSE 0 END) as total_switches,
        MAX(account_equity)                        as peak_equity
    FROM {TRADE_LOG}
""").collect()[0]

latest_equity = spark.sql(f"""
    SELECT account_equity FROM {TRADE_LOG}
    ORDER BY trade_ts DESC LIMIT 1
""").collect()[0]["account_equity"]

# STEP 6: Regime distribution from audit log
regime_dist = spark.sql(f"""
    SELECT
        ROUND(100.0 * SUM(CASE WHEN regime_int=0 THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_expansion,
        ROUND(100.0 * SUM(CASE WHEN regime_int=1 THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_slowdown,
        ROUND(100.0 * SUM(CASE WHEN regime_int=2 THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_recession
    FROM {AUDIT_LOG}
""").collect()[0]

# STEP 7: Assemble
feed_ts = datetime.now(timezone.utc).replace(tzinfo=None)

schema = StructType([
    StructField("feed_ts",          TimestampType(), True),
    StructField("signal_ts",        TimestampType(), True),
    StructField("regime_int",       IntegerType(),   True),
    StructField("regime_name",      StringType(),    True),
    StructField("position",         StringType(),    True),
    StructField("VIXCLS_z",         DoubleType(),    True),
    StructField("vix3m_z",          DoubleType(),    True),
    StructField("VIX_Inversion_Z",  DoubleType(),    True),
    StructField("CPIAUCSL_z",       DoubleType(),    True),
    StructField("PAYEMS_z",         DoubleType(),    True),
    StructField("Price_Mom_24H",    DoubleType(),    True),
    StructField("SPY_close",        DoubleType(),    True),
    StructField("QQQ_close",        DoubleType(),    True),
    StructField("price_ts",         TimestampType(), True),
    StructField("account_equity",   DoubleType(),    True),
    StructField("account_cash",     DoubleType(),    True),
    StructField("total_cycles",     IntegerType(),   True),
    StructField("total_switches",   IntegerType(),   True),
    StructField("peak_equity",      DoubleType(),    True),
    StructField("pct_expansion",    DoubleType(),    True),
    StructField("pct_slowdown",     DoubleType(),    True),
    StructField("pct_recession",    DoubleType(),    True),
    StructField("model_version",    StringType(),    True),
    StructField("source",           StringType(),    True),
    StructField("ingestion_ts",     TimestampType(), True),
])

dashboard_row = {
    "feed_ts":         feed_ts,
    "signal_ts":       signal["timestamp"],
    "regime_int":      signal["regime_int"],
    "regime_name":     signal["regime_name"],
    "position":        signal["position"],
    "VIXCLS_z":        float(signal["VIXCLS_z"]),
    "vix3m_z":         float(signal["vix3m_z"]),
    "VIX_Inversion_Z": float(silver["VIX_Inversion_Z"] or 0),
    "CPIAUCSL_z":      float(signal["CPIAUCSL_z"]),
    # Carry the payroll signal through to the dashboard feed.
    "PAYEMS_z":        float(signal["PAYEMS_z"]),
    "Price_Mom_24H":   float(silver["Price_Mom_24H"] or 0),
    "SPY_close":       price_map.get("SPY"),
    "QQQ_close":       price_map.get("QQQ"),
    "price_ts":        price_ts_row,
    "account_equity":  float(latest_equity),
    "account_cash":    float(trade["account_cash"]),
    "total_cycles":    int(portfolio_stats["total_cycles"]),
    "total_switches":  int(portfolio_stats["total_switches"]),
    "peak_equity":     float(portfolio_stats["peak_equity"]),
    "pct_expansion":   float(regime_dist["pct_expansion"] or 0),
    "pct_slowdown":    float(regime_dist["pct_slowdown"] or 0),
    "pct_recession":   float(regime_dist["pct_recession"] or 0),
    "model_version":   "bt_fred_1d_xgboost_v4",
    "source":          "rt_dashboard_feed",
    "ingestion_ts":    feed_ts,
}

def fmt_num(x, fmt="{:.3f}", default="n/a"):
    if x is None:
        return default
    try:
        return fmt.format(x)
    except Exception:
        return default

def fmt_money(x, default="n/a"):
    if x is None:
        return default
    try:
        return f"${x:,.2f}"
    except Exception:
        return default

sdf = spark.createDataFrame([dashboard_row], schema=schema)
sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TARGET_TABLE)

print(f"Dashboard feed written to {TARGET_TABLE}")
print(f"\n{'='*55}")
print(f"  V4 LIVE DASHBOARD @ {feed_ts.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"{'='*55}")
print(f"  Regime:       {dashboard_row['regime_name']} ({dashboard_row['regime_int']})")
print(f"  Position:     {dashboard_row['position']}")
print(f"  SPY:          {fmt_money(dashboard_row['SPY_close'])}")
print(f"  QQQ:          {fmt_money(dashboard_row['QQQ_close'])}")
print(f"  VIXCLS_z:     {fmt_num(dashboard_row['VIXCLS_z'])}")
print(f"  vix3m_z:      {fmt_num(dashboard_row['vix3m_z'])}")
print(f"  VIX_Inv_Z:    {fmt_num(dashboard_row['VIX_Inversion_Z'])}")
print(f"  CPIAUCSL_z:   {fmt_num(dashboard_row['CPIAUCSL_z'])}")
print(f"  Equity:       {fmt_money(dashboard_row['account_equity'])}")
print(f"  Cycles:       {dashboard_row['total_cycles']}")
print(f"  Switches:     {dashboard_row['total_switches']}")
print(f"  Expansion:    {dashboard_row['pct_expansion'] if dashboard_row['pct_expansion'] is not None else 'n/a'}%")
print(f"  Slowdown:     {dashboard_row['pct_slowdown'] if dashboard_row['pct_slowdown'] is not None else 'n/a'}%")
print(f"  Recession:    {dashboard_row['pct_recession'] if dashboard_row['pct_recession'] is not None else 'n/a'}%")
print(f"  Model:        {dashboard_row['model_version']}")
print(f"{'='*55}")

spark.sql(f"SELECT * FROM {TARGET_TABLE}").show(vertical=True)

# COMMAND ----------
