# Databricks notebook source
# BT_05: bounded silver refresh for production use

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
VIX_SOURCE = "fred_1d"
REFRESH_HOURS = 8760
SAFETY_BUFFER_HOURS = 72

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"
MACRO_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
VIX_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_vix_1h"
VIX3M_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_vix3m_1h"

max_ts_row = spark.sql(f"SELECT MAX(timestamp) AS max_ts FROM {MARKET_TABLE}").collect()[0]
end_date = max_ts_row["max_ts"]
if end_date is None:
    raise ValueError("bronze_bt_market_1h is empty")

refresh_start = spark.sql(f"""
    SELECT COALESCE(
        MAX(timestamp) - INTERVAL {REFRESH_HOURS + SAFETY_BUFFER_HOURS} HOURS,
        TIMESTAMP('2016-01-01 00:00:00')
    ) AS refresh_start
    FROM {SILVER_TABLE}
""").collect()[0]["refresh_start"]

print(f"Refreshing silver window from {refresh_start} to {end_date}")

backbone = (
    spark.range(0, 24 * 400)
    .withColumn("timestamp", F.to_timestamp(F.lit(refresh_start)) + (F.col("id") * F.expr("INTERVAL 1 HOUR")))
    .filter(F.col("timestamp") <= F.lit(end_date))
    .select("timestamp")
)

macro_df = spark.table(MACRO_TABLE).filter(F.col("timestamp") >= F.lit(refresh_start) - F.expr("INTERVAL 7 DAY"))
market_df = spark.table(MARKET_TABLE).filter(F.col("timestamp") >= F.lit(refresh_start) - F.expr("INTERVAL 7 DAY"))
qqq_df = market_df.filter(F.col("symbol") == "QQQ").dropDuplicates(["timestamp"])
spy_df = market_df.filter(F.col("symbol") == "SPY").dropDuplicates(["timestamp"])
vix_df = spark.table(VIX_TABLE).filter(F.col("timestamp") >= F.lit(refresh_start) - F.expr("INTERVAL 7 DAY")).dropDuplicates(["timestamp"])
vix3m_df = spark.table(VIX3M_TABLE).filter(F.col("timestamp") >= F.lit(refresh_start) - F.expr("INTERVAL 7 DAY")).dropDuplicates(["timestamp"])

if VIX_SOURCE == "portara_1h":
    vix3m_lagged = vix3m_df.withColumn("v3m_join_ts", F.col("timestamp") + F.expr("INTERVAL 1 DAY"))
else:
    vix3m_lagged = vix3m_df.withColumnRenamed("close", "vix3m").withColumn("v3m_join_ts", F.col("timestamp"))

macro_prepared = (
    macro_df
    .withColumn("join_ts", F.date_trunc("hour", F.col("timestamp") + F.expr("INTERVAL 2 HOUR")))
    .select("join_ts", "indicator_id", "actual")
    .groupBy("join_ts")
    .pivot("indicator_id")
    .avg("actual")
)

silver_grid = (
    backbone
    .join(macro_prepared, backbone.timestamp == macro_prepared.join_ts, "left")
    .join(spy_df.select(F.col("timestamp").alias("spy_ts"), F.col("close").alias("SPY_close")), F.col("timestamp") == F.col("spy_ts"), "left")
    .join(qqq_df.select(F.col("timestamp").alias("qqq_ts"), F.col("close").alias("QQQ_close")), F.col("timestamp") == F.col("qqq_ts"), "left")
    .join(vix3m_lagged.select(F.col("v3m_join_ts").alias("v3m_ts"), F.col("vix3m")), F.col("timestamp") == F.col("v3m_ts"), "left")
    .join(vix_df.select(F.col("timestamp").alias("vix_ts"), F.col("close").alias("VIXCLS")), F.col("timestamp") == F.col("vix_ts"), "left")
)

win_ffill = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)
for col_name in ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT", "VIXCLS", "vix3m", "SPY_close", "QQQ_close"]:
    silver_grid = silver_grid.withColumn(col_name, F.last(F.col(col_name), True).over(win_ffill))

win_lag = Window.orderBy("timestamp")
win_z = Window.orderBy("timestamp").rowsBetween(-8760, 0)
for col_name in ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT", "VIXCLS", "vix3m"]:
    avg = F.avg(F.col(col_name)).over(win_z)
    std = F.stddev(F.col(col_name)).over(win_z)
    silver_grid = silver_grid.withColumn(f"{col_name}_z", (F.col(col_name) - avg) / F.when(std == 0, 1).otherwise(std))

silver_grid = (
    silver_grid
    .withColumn("CPI_Shock", F.abs(F.col("CPIAUCSL") - F.lag("CPIAUCSL", 1).over(win_lag)))
    .withColumn("NFP_Shock", F.abs(F.col("PAYEMS") - F.lag("PAYEMS", 1).over(win_lag)))
)

def pulse_event(col_name, hours=48):
    return F.max(F.when(F.col(col_name) > 0, 1).otherwise(0)).over(Window.orderBy("timestamp").rowsBetween(-hours, 0))

silver_grid = (
    silver_grid
    .withColumn("CPI_Event_Active", pulse_event("CPI_Shock"))
    .withColumn("NFP_Event_Active", pulse_event("NFP_Shock"))
    .withColumn("VIX_Inversion_Z", F.col("VIXCLS_z") - F.col("vix3m_z"))
    .withColumn("Price_Mom_24H", (F.lag("SPY_close", 1).over(win_lag) / F.lag("SPY_close", 25).over(win_lag)) - 1)
)

refresh_df = (
    silver_grid
    .select(
        "timestamp", "SPY_close", "QQQ_close",
        "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
        "GDPC1_z", "INDPRO_z", "UMCSENT_z",
        "VIXCLS_z", "vix3m_z", "VIX_Inversion_Z", "Price_Mom_24H",
        "CPI_Event_Active", "NFP_Event_Active",
    )
    .filter(F.col("timestamp") >= F.lit(refresh_start))
    .filter(F.col("SPY_close").isNotNull())
)

refresh_df.createOrReplaceTempView("silver_refresh_window")
spark.sql(f"""
    MERGE INTO {SILVER_TABLE} t
    USING silver_refresh_window s
    ON t.timestamp = s.timestamp
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Refreshed trailing silver window into {SILVER_TABLE}")
