# Databricks notebook source
# RT_04: Model Inference Engine

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config

# MAGIC %pip install xgboost

import joblib
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

SILVER_RT = f"{CATALOG}.{SCHEMA}.silver_rt_1h"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"
LIVE_MODEL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/models/bt_fred_1d_xgboost_regime_live.joblib"

FEATURES = [
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

REGIME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
runtime_config = load_runtime_instrument_config(spark, CATALOG, SCHEMA)
POSITION_MAP = runtime_config["position_map"]

SIGNAL_COLUMNS = {
    "timestamp": "TIMESTAMP",
    "regime_int": "INT",
    "regime_name": "STRING",
    "position": "STRING",
    "VIXCLS_z": "DOUBLE",
    "vix3m_z": "DOUBLE",
    "Price_Mom_24H": "DOUBLE",
    "CPIAUCSL_z": "DOUBLE",
    "PCEPI_z": "DOUBLE",
    "PCE_z": "DOUBLE",
    "PAYEMS_z": "DOUBLE",
    "UNRATE_z": "DOUBLE",
    "GDPC1_z": "DOUBLE",
    "INDPRO_z": "DOUBLE",
    "UMCSENT_z": "DOUBLE",
    "VIX_Inversion_Z": "DOUBLE",
    "CPI_Event_Active": "INT",
    "NFP_Event_Active": "INT",
    "model_version": "STRING",
    "model_path": "STRING",
    "source": "STRING",
    "ingestion_ts": "TIMESTAMP",
}

MERGE_COLUMNS = list(SIGNAL_COLUMNS.keys())


def existing_columns(table_name):
    if not spark.catalog.tableExists(table_name):
        return set()
    return {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {table_name}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }


def ensure_signal_table():
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            timestamp TIMESTAMP,
            regime_int INT,
            regime_name STRING,
            position STRING,
            VIXCLS_z DOUBLE,
            vix3m_z DOUBLE,
            Price_Mom_24H DOUBLE,
            CPIAUCSL_z DOUBLE,
            PCEPI_z DOUBLE,
            PCE_z DOUBLE,
            PAYEMS_z DOUBLE,
            UNRATE_z DOUBLE,
            GDPC1_z DOUBLE,
            INDPRO_z DOUBLE,
            UMCSENT_z DOUBLE,
            VIX_Inversion_Z DOUBLE,
            CPI_Event_Active INT,
            NFP_Event_Active INT,
            model_version STRING,
            model_path STRING,
            source STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    current = existing_columns(TARGET_TABLE)
    for name, dtype in SIGNAL_COLUMNS.items():
        if name not in current:
            spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {name} {dtype}")


def get_active_model_metadata():
    if spark.catalog.tableExists(MODEL_REGISTRY_TABLE):
        registry_rows = spark.sql(
            f"""
            SELECT model_version, model_path
            FROM {MODEL_REGISTRY_TABLE}
            ORDER BY promoted_ts DESC
            LIMIT 1
            """
        ).collect()
        if registry_rows:
            return registry_rows[0]["model_version"], registry_rows[0]["model_path"]
    return "bootstrap_live_model", LIVE_MODEL_PATH


ensure_signal_table()
model_version, model_path = get_active_model_metadata()
model = joblib.load(model_path)

latest_row = spark.sql(
    f"""
    SELECT * FROM {SILVER_RT}
    ORDER BY timestamp DESC
    LIMIT 1
    """
).toPandas()

if latest_row.empty:
    raise ValueError("silver_rt_1h is empty. Run RT_03 first.")

inference_ts = latest_row["timestamp"].iloc[0].to_pydatetime().replace(tzinfo=None)
X_rt = latest_row[FEATURES].fillna(0)
regime_int = int(model.predict(X_rt)[0])
regime_name = REGIME_MAP[regime_int]
position = POSITION_MAP[regime_int]

schema = StructType(
    [
        StructField("timestamp", TimestampType(), True),
        StructField("regime_int", IntegerType(), True),
        StructField("regime_name", StringType(), True),
        StructField("position", StringType(), True),
        StructField("VIXCLS_z", DoubleType(), True),
        StructField("vix3m_z", DoubleType(), True),
        StructField("Price_Mom_24H", DoubleType(), True),
        StructField("CPIAUCSL_z", DoubleType(), True),
        StructField("PCEPI_z", DoubleType(), True),
        StructField("PCE_z", DoubleType(), True),
        StructField("PAYEMS_z", DoubleType(), True),
        StructField("UNRATE_z", DoubleType(), True),
        StructField("GDPC1_z", DoubleType(), True),
        StructField("INDPRO_z", DoubleType(), True),
        StructField("UMCSENT_z", DoubleType(), True),
        StructField("VIX_Inversion_Z", DoubleType(), True),
        StructField("CPI_Event_Active", IntegerType(), True),
        StructField("NFP_Event_Active", IntegerType(), True),
        StructField("model_version", StringType(), True),
        StructField("model_path", StringType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

signal_row = {
    "timestamp": inference_ts,
    "regime_int": regime_int,
    "regime_name": regime_name,
    "position": position,
    "VIXCLS_z": float(latest_row["VIXCLS_z"].iloc[0]),
    "vix3m_z": float(latest_row["vix3m_z"].iloc[0]),
    "Price_Mom_24H": float(latest_row["Price_Mom_24H"].iloc[0]),
    "CPIAUCSL_z": float(latest_row["CPIAUCSL_z"].iloc[0]),
    "PCEPI_z": float(latest_row["PCEPI_z"].iloc[0]),
    "PCE_z": float(latest_row["PCE_z"].iloc[0]),
    "PAYEMS_z": float(latest_row["PAYEMS_z"].iloc[0]),
    "UNRATE_z": float(latest_row["UNRATE_z"].iloc[0]),
    "GDPC1_z": float(latest_row["GDPC1_z"].iloc[0]),
    "INDPRO_z": float(latest_row["INDPRO_z"].iloc[0]),
    "UMCSENT_z": float(latest_row["UMCSENT_z"].iloc[0]),
    "VIX_Inversion_Z": float(latest_row["VIX_Inversion_Z"].iloc[0]),
    "CPI_Event_Active": int(latest_row["CPI_Event_Active"].iloc[0]),
    "NFP_Event_Active": int(latest_row["NFP_Event_Active"].iloc[0]),
    "model_version": model_version,
    "model_path": model_path,
    "source": "rt_inference_v4_live_alias",
    "ingestion_ts": None,
}

sdf = spark.createDataFrame([signal_row], schema=schema).withColumn(
    "ingestion_ts", F.current_timestamp()
)
sdf.createOrReplaceTempView("incoming_rt_regime_signal")

update_clause = ",\n        ".join([f"t.{col} = s.{col}" for col in MERGE_COLUMNS])
insert_columns = ", ".join(MERGE_COLUMNS)
insert_values = ", ".join([f"s.{col}" for col in MERGE_COLUMNS])

spark.sql(
    f"""
    MERGE INTO {TARGET_TABLE} t
    USING incoming_rt_regime_signal s
    ON t.timestamp = s.timestamp
    WHEN MATCHED THEN UPDATE SET
        {update_clause}
    WHEN NOT MATCHED THEN INSERT ({insert_columns})
    VALUES ({insert_values})
    """
)

print(f"Signal written to {TARGET_TABLE}")
print(f"Model version: {model_version}")
print(f"Model path: {model_path}")
print(f"Signal timestamp: {inference_ts}")
print(f"Predicted regime: {regime_name} ({regime_int})")
print(f"Target allocation: {position}")
