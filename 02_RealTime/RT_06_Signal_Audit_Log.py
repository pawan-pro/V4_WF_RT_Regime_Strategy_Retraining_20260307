# Databricks notebook source
# RT_06: Signal Audit Log
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timezone, date

SIGNAL_TABLE = "pawan_quant_research_1.v4.bronze_rt_regime_signal"
SILVER_RT    = "pawan_quant_research_1.v4.silver_rt_1h"
TRADE_LOG    = "pawan_quant_research_1.v4.bronze_rt_paper_trade_log"
TARGET_TABLE = "pawan_quant_research_1.v4.bronze_rt_signal_audit_log"

# Add ingestion_ts column if missing (ACL clusters block auto-merge)
try:
    spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN ingestion_ts TIMESTAMP")
    print("Added ingestion_ts column")
except Exception as e:
    print(f"Column already exists or skipped: {e}")

# STEP 1: Latest signal
signal = spark.sql(f"""
    SELECT * FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]

# STEP 2: Full feature vector from silver_rt
silver = spark.sql(f"""
    SELECT * FROM {SILVER_RT}
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]

# STEP 3: Latest trade log
trade = spark.sql(f"""
    SELECT switched, orders_placed, account_equity
    FROM {TRADE_LOG}
    ORDER BY trade_ts DESC LIMIT 1
""").collect()[0]

# STEP 4: PIT lag computation
vix_source_date = spark.sql("""
    SELECT CAST(MAX(timestamp) AS DATE) as vix_date
    FROM pawan_quant_research_1.v4.bronze_rt_vix_1d
    WHERE series = 'VIXCLS'
""").collect()[0]["vix_date"]

pit_lag_days = (date.today() - vix_source_date).days
bias_gate_ok = vix_source_date < date.today()
audit_ts     = datetime.now(timezone.utc).replace(tzinfo=None)

print(f"  Signal ts:      {signal['timestamp']}")
print(f"  Regime:         {signal['regime_name']} ({signal['regime_int']})")
print(f"  VIX source dt:  {vix_source_date}  (lag: {pit_lag_days} days)")
print(f"  Bias gate:      {'PASSED' if bias_gate_ok else 'FAILED'}")
print(f"  Switched:       {trade['switched']}")

# STEP 5: Assemble audit row
schema = StructType([
    StructField("audit_ts",         TimestampType(), True),
    StructField("signal_ts",        TimestampType(), True),
    StructField("regime_int",       IntegerType(),   True),
    StructField("regime_name",      StringType(),    True),
    StructField("position",         StringType(),    True),
    StructField("VIXCLS_z",         DoubleType(),    True),
    StructField("vix3m_z",          DoubleType(),    True),
    StructField("VIX_Inversion_Z",  DoubleType(),    True),
    StructField("Price_Mom_24H",    DoubleType(),    True),
    StructField("CPIAUCSL_z",       DoubleType(),    True),
    StructField("PCEPI_z",          DoubleType(),    True),
    StructField("PCE_z",            DoubleType(),    True),
    StructField("PAYEMS_z",         DoubleType(),    True),
    StructField("UNRATE_z",         DoubleType(),    True),
    StructField("GDPC1_z",          DoubleType(),    True),
    StructField("INDPRO_z",         DoubleType(),    True),
    StructField("UMCSENT_z",        DoubleType(),    True),
    StructField("CPI_Event_Active", IntegerType(),   True),
    StructField("NFP_Event_Active", IntegerType(),   True),
    StructField("SPY_close",        DoubleType(),    True),
    StructField("switched",         BooleanType(),   True),
    StructField("orders_placed",    StringType(),    True),
    StructField("account_equity",   DoubleType(),    True),
    StructField("vix_source_date",  DateType(),      True),
    StructField("pit_lag_days",     IntegerType(),   True),
    StructField("bias_gate_passed", BooleanType(),   True),
    StructField("model_version",    StringType(),    True),
    StructField("source",           StringType(),    True),
    StructField("ingestion_ts",     TimestampType(), True),
])

audit_row = {
    "audit_ts":         audit_ts,
    "signal_ts":        signal["timestamp"],
    "regime_int":       signal["regime_int"],
    "regime_name":      signal["regime_name"],
    "position":         signal["position"],
    "VIXCLS_z":         float(silver["VIXCLS_z"]),
    "vix3m_z":          float(silver["vix3m_z"]),
    "VIX_Inversion_Z":  float(silver["VIX_Inversion_Z"]),
    "Price_Mom_24H":    float(silver["Price_Mom_24H"]),
    "CPIAUCSL_z":       float(silver["CPIAUCSL_z"]),
    "PCEPI_z":          float(silver["PCEPI_z"]),
    "PCE_z":            float(silver["PCE_z"]),
    "PAYEMS_z":         float(silver["PAYEMS_z"]),
    "UNRATE_z":         float(silver["UNRATE_z"]),
    "GDPC1_z":          float(silver["GDPC1_z"]),
    "INDPRO_z":         float(silver["INDPRO_z"]),
    "UMCSENT_z":        float(silver["UMCSENT_z"]),
    "CPI_Event_Active": int(silver["CPI_Event_Active"]),
    "NFP_Event_Active": int(silver["NFP_Event_Active"]),
    "SPY_close":        float(silver["SPY_close"]),
    "switched":         bool(trade["switched"]),
    "orders_placed":    str(trade["orders_placed"]),
    "account_equity":   float(trade["account_equity"]),
    "vix_source_date":  vix_source_date,
    "pit_lag_days":     pit_lag_days,
    "bias_gate_passed": bias_gate_ok,
    "model_version":    "bt_fred_1d_xgboost_v4",
    "source":           "rt_signal_audit",
    "ingestion_ts":     audit_ts,
}

sdf = spark.createDataFrame([audit_row], schema=schema)
sdf.write.format("delta").mode("append").saveAsTable(TARGET_TABLE)

print(f"\nAudit log written to {TARGET_TABLE}")
spark.sql(f"""
    SELECT audit_ts, regime_name, position, VIXCLS_z, vix3m_z,
           pit_lag_days, bias_gate_passed, switched, account_equity
    FROM {TARGET_TABLE}
    ORDER BY audit_ts DESC LIMIT 5
""").show(vertical=True)