# Databricks notebook source
# MAGIC %md
# MAGIC master_time_grid_silver

# COMMAND ----------

# DBTITLE 1,Cell 2
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ── VIX SOURCE FLAG — must match what was used in 04RT ──
VIX_SOURCE = "fred_1d"   # "portara_1h" | "fred_1d"

# 1. Dynamically get the latest timestamp from the Market Bronze Table
max_ts_row = spark.sql('SELECT MAX(timestamp) as max_ts FROM pawan_quant_research_1.default.bronze_bt_market_1h').collect()[0]
end_date = max_ts_row["max_ts"]
start_date = "2016-01-01 00:00:00"

print(f"🔄 Building grid from {start_date} to {end_date}... | VIX_SOURCE={VIX_SOURCE}")

# 2. Generate the Hourly Backbone
backbone = spark.range(0, 15 * 365 * 24) \
    .withColumn("timestamp", F.to_timestamp(F.lit(start_date)) + (F.col("id") * F.expr("INTERVAL 1 HOUR"))) \
    .filter(F.col("timestamp") <= F.lit(end_date)) \
    .select("timestamp")

# 3. Load and Deduplicate Bronze Data
macro_df    = spark.table("pawan_quant_research_1.default.bronze_bt_macro_events").dropDuplicates(["timestamp", "indicator_id"])
market_df   = spark.table("pawan_quant_research_1.default.bronze_bt_market_1h").filter(F.col("symbol") == "SPY").dropDuplicates(["timestamp"])
qqq_df      = spark.table("pawan_quant_research_1.default.bronze_bt_market_1h").filter(F.col("symbol") == "QQQ").dropDuplicates(["timestamp"])
vix3m_df    = spark.table("pawan_quant_research_1.default.bronze_bt_vix3m_1h").dropDuplicates(["timestamp"])
vix_spot_df = spark.table("pawan_quant_research_1.default.bronze_bt_vix_1h").dropDuplicates(["timestamp"])

# 4. VIX3M Lag — conditional on source
# portara_1h: column is "vix3m", no PIT at ingestion → apply 1-day shift here
# fred_1d:    column is "close", PIT already applied at ingestion → join on actual timestamp, rename close → vix3m
if VIX_SOURCE == "portara_1h":
    # Portara: column is already named "vix3m", apply 1-day PIT shift at join time
    vix3m_lagged = vix3m_df.withColumn(
        "v3m_join_ts",
        F.to_timestamp(F.col("timestamp")) + F.expr("INTERVAL 1 DAY")
    )
else:
    # fred_1d: column is named "close", PIT already applied at ingestion
    # Rename "close" → "vix3m" so downstream join is schema-consistent
    vix3m_lagged = (vix3m_df
        .withColumnRenamed("close", "vix3m")
        .withColumn("v3m_join_ts", F.col("timestamp"))
    )

# 5. Prepare Macro Data (Point-In-Time: +2H after release)
macro_prepared = (macro_df
    .withColumn("join_ts", F.date_trunc("hour", F.col("timestamp") + F.expr("INTERVAL 2 HOUR")))
    .select("join_ts", "indicator_id", "actual")
    .groupBy("join_ts")
    .pivot("indicator_id")
    .avg("actual")
)

# 6. Master Join
silver_grid = (backbone
    .join(macro_prepared, backbone.timestamp == macro_prepared.join_ts, "left")
    .join(market_df.select(F.col("timestamp").alias("m_ts"), F.col("close").alias("SPY_close")),
          F.col("timestamp") == F.col("m_ts"), "left")
    .join(qqq_df.select(F.col("timestamp").alias("q_ts"), F.col("close").alias("QQQ_close")),
          F.col("timestamp") == F.col("q_ts"), "left")
    .join(vix3m_lagged.select(F.col("v3m_join_ts").alias("v3m_ts"), F.col("vix3m")),
          F.col("timestamp") == F.col("v3m_ts"), "left")
    .join(vix_spot_df.select(F.col("timestamp").alias("vs_ts"), F.col("close").alias("VIXCLS")),
          F.col("timestamp") == F.col("vs_ts"), "left")
)

# 7. Forward Fill 24/7
win_ffill = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)
fill_cols = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT", "VIXCLS", "vix3m", "SPY_close", "QQQ_close"]

for col_name in fill_cols:
    silver_grid = silver_grid.withColumn(col_name, F.last(F.col(col_name), True).over(win_ffill))

# 8. Calculate V4 Alpha Features
win_lag = Window.orderBy("timestamp")
win_z   = Window.orderBy("timestamp").rowsBetween(-8760, 0)

# A. Rolling Z-Scores (1-year lookback window)
z_targets = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT", "VIXCLS", "vix3m"]
for col in z_targets:
    avg = F.avg(F.col(col)).over(win_z)
    std = F.stddev(F.col(col)).over(win_z)
    silver_grid = silver_grid.withColumn(f"{col}_z", (F.col(col) - avg) / F.when(std == 0, 1).otherwise(std))

# B. Pulsed Event Flags
silver_grid = silver_grid \
    .withColumn("CPI_Shock", F.abs(F.col("CPIAUCSL") - F.lag("CPIAUCSL", 1).over(win_lag))) \
    .withColumn("NFP_Shock", F.abs(F.col("PAYEMS")   - F.lag("PAYEMS",   1).over(win_lag)))

def pulse_event(col_name, hours=48):
    is_event = F.when(F.col(col_name) > 0, 1).otherwise(0)
    return F.max(is_event).over(Window.orderBy("timestamp").rowsBetween(-hours, 0))

silver_grid = silver_grid \
    .withColumn("CPI_Event_Active", pulse_event("CPI_Shock")) \
    .withColumn("NFP_Event_Active", pulse_event("NFP_Shock"))

# C. Derived Features & Lagged Momentum
silver_grid = silver_grid \
    .withColumn("VIX_Inversion_Z", F.col("VIXCLS_z") - F.col("vix3m_z")) \
    .withColumn("Price_Mom_24H",
                (F.lag(F.col("SPY_close"), 1).over(win_lag) /
                 F.lag(F.col("SPY_close"), 25).over(win_lag)) - 1)

# 9. Final Selection & Save
final_silver = silver_grid.select(
    "timestamp", "SPY_close", "QQQ_close",
    "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z", "GDPC1_z", "INDPRO_z", "UMCSENT_z",
    "VIXCLS_z", "vix3m_z", "VIX_Inversion_Z", "Price_Mom_24H",
    "CPI_Event_Active", "NFP_Event_Active"
).filter(F.col("SPY_close").isNotNull())

(final_silver.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .format("delta")
    .saveAsTable("pawan_quant_research_1.default.silver_bt_master_grid_1h"))

print(f"✅ SUCCESS: silver_bt_master_grid_1h created | VIX_SOURCE={VIX_SOURCE}")

# COMMAND ----------

# DBTITLE 1,Preview: Head and Tail of Silver Master Grid
# Display first 10 rows (head)
print("📊 First 10 rows (HEAD):")
head_df = spark.sql("""
    SELECT * 
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
    ORDER BY timestamp ASC
    LIMIT 10
""")
display(head_df)

# Display last 10 rows (tail)
print("\n📊 Last 10 rows (TAIL):")
tail_df = spark.sql("""
    SELECT * 
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
    ORDER BY timestamp DESC
    LIMIT 10
""")
display(tail_df)

# Summary statistics
print("\n📈 SUMMARY:")
summary_df = spark.sql("""
    SELECT 
        COUNT(*) as total_records,
        MIN(timestamp) as first_timestamp,
        MAX(timestamp) as last_timestamp,
        COUNT(DISTINCT DATE(timestamp)) as unique_days,
        ROUND(AVG(SPY_close), 2) as avg_spy_close,
        ROUND(AVG(VIXCLS_z), 2) as avg_vix_spot_z,
        ROUND(AVG(vix3m_z), 2) as avg_vix3m_z,
        ROUND(AVG(VIX_Inversion_Z), 4) as avg_vix_inversion_z,
        COUNT(CASE WHEN CPIAUCSL_z IS NOT NULL THEN 1 END) as cpi_records,
        COUNT(CASE WHEN PAYEMS_z IS NOT NULL THEN 1 END) as payems_records
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
""")
display(summary_df)

# COMMAND ----------

df = spark.table("pawan_quant_research_1.default.silver_bt_master_grid_1h").filter(
    (F.col("timestamp") >= "2024-03-01") &
    (F.col("timestamp") <= "2024-04-30")
)

display(df)

# COMMAND ----------

spark.sql("""
    SELECT 
        MAX(timestamp) as last_ts,
        COUNT(*) as total_rows,
        COUNT(CASE WHEN CPIAUCSL_z IS NOT NULL THEN 1 END) as rows_with_cpi,
        COUNT(CASE WHEN SPY_close IS NOT NULL THEN 1 END) as rows_with_spy
    FROM pawan_quant_research_1.default.silver_bt_master_grid_1h
""").show()

# COMMAND ----------

spark.sql("""
    SELECT DISTINCT symbol, COUNT(*) as rows 
    FROM pawan_quant_research_1.default.bronze_bt_vix3m_1h 
    GROUP BY symbol
""").show()
