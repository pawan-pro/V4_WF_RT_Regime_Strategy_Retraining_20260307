# Databricks notebook source
# RT_09: Saxo signal audit log

# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

from datetime import date, datetime, timezone

from pyspark.sql.types import (
    BooleanType,
    DateType,
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
TRADE_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_trade_log"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_signal_audit_log"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"


def existing_columns(table_name):
    if not spark.catalog.tableExists(table_name):
        return set()
    return {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {table_name}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }


def ensure_audit_table():
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            audit_ts TIMESTAMP,
            trade_ts TIMESTAMP,
            signal_ts TIMESTAMP,
            regime_int INT,
            regime_name STRING,
            position STRING,
            target_ticker STRING,
            VIXCLS_z DOUBLE,
            vix3m_z DOUBLE,
            VIX_Inversion_Z DOUBLE,
            Price_Mom_24H DOUBLE,
            CPIAUCSL_z DOUBLE,
            PCEPI_z DOUBLE,
            PCE_z DOUBLE,
            PAYEMS_z DOUBLE,
            UNRATE_z DOUBLE,
            GDPC1_z DOUBLE,
            INDPRO_z DOUBLE,
            UMCSENT_z DOUBLE,
            CPI_Event_Active INT,
            NFP_Event_Active INT,
            SPY_close DOUBLE,
            QQQ_close DOUBLE,
            switched BOOLEAN,
            orders_placed STRING,
            account_balance DOUBLE,
            account_equity DOUBLE,
            account_cash DOUBLE,
            positions_count INT,
            positions_market_value DOUBLE,
            positions_unrealized_pl DOUBLE,
            hourly_pnl DOUBLE,
            hourly_pnl_pct DOUBLE,
            vix_source_date DATE,
            pit_lag_days INT,
            bias_gate_passed BOOLEAN,
            model_version STRING,
            execution_mode STRING,
            source STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    desired = {
        "trade_ts": "TIMESTAMP",
        "account_balance": "DOUBLE",
        "positions_market_value": "DOUBLE",
        "positions_unrealized_pl": "DOUBLE",
        "hourly_pnl": "DOUBLE",
        "hourly_pnl_pct": "DOUBLE",
        "execution_mode": "STRING",
        "model_version": "STRING",
        "ingestion_ts": "TIMESTAMP",
    }
    current = existing_columns(TARGET_TABLE)
    for name, dtype in desired.items():
        if name not in current:
            spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {name} {dtype}")


def get_model_version():
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
    return "bootstrap_live_model"


ensure_audit_table()

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

vix_source_date = spark.sql(
    f"""
    SELECT CAST(MAX(timestamp) AS DATE) AS vix_date
    FROM {CATALOG}.{SCHEMA}.bronze_rt_vix_1d
    WHERE series = 'VIXCLS'
    """
).collect()[0]["vix_date"]

pit_lag_days = (date.today() - vix_source_date).days if vix_source_date else None
bias_gate_ok = vix_source_date < date.today() if vix_source_date else None
audit_ts = datetime.now(timezone.utc).replace(tzinfo=None)
model_version = signal_data.get("model_version") or get_model_version()

schema = StructType(
    [
        StructField("audit_ts", TimestampType(), True),
        StructField("trade_ts", TimestampType(), True),
        StructField("signal_ts", TimestampType(), True),
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
        StructField("switched", BooleanType(), True),
        StructField("orders_placed", StringType(), True),
        StructField("account_balance", DoubleType(), True),
        StructField("account_equity", DoubleType(), True),
        StructField("account_cash", DoubleType(), True),
        StructField("positions_count", IntegerType(), True),
        StructField("positions_market_value", DoubleType(), True),
        StructField("positions_unrealized_pl", DoubleType(), True),
        StructField("hourly_pnl", DoubleType(), True),
        StructField("hourly_pnl_pct", DoubleType(), True),
        StructField("vix_source_date", DateType(), True),
        StructField("pit_lag_days", IntegerType(), True),
        StructField("bias_gate_passed", BooleanType(), True),
        StructField("model_version", StringType(), True),
        StructField("execution_mode", StringType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

audit_row = {
    "audit_ts": audit_ts,
    "trade_ts": trade["trade_ts"],
    "signal_ts": signal["timestamp"],
    "regime_int": signal["regime_int"],
    "regime_name": signal["regime_name"],
    "position": signal["position"],
    "target_ticker": trade["target_ticker"],
    "VIXCLS_z": float(silver["VIXCLS_z"]),
    "vix3m_z": float(silver["vix3m_z"]),
    "VIX_Inversion_Z": float(silver["VIX_Inversion_Z"]),
    "Price_Mom_24H": float(silver["Price_Mom_24H"]),
    "CPIAUCSL_z": float(silver["CPIAUCSL_z"]),
    "PCEPI_z": float(silver["PCEPI_z"]),
    "PCE_z": float(silver["PCE_z"]),
    "PAYEMS_z": float(silver["PAYEMS_z"]),
    "UNRATE_z": float(silver["UNRATE_z"]),
    "GDPC1_z": float(silver["GDPC1_z"]),
    "INDPRO_z": float(silver["INDPRO_z"]),
    "UMCSENT_z": float(silver["UMCSENT_z"]),
    "CPI_Event_Active": int(silver["CPI_Event_Active"]),
    "NFP_Event_Active": int(silver["NFP_Event_Active"]),
    "SPY_close": float(trade["spy_close"]) if trade["spy_close"] is not None else None,
    "QQQ_close": float(trade["qqq_close"]) if trade["qqq_close"] is not None else None,
    "switched": bool(trade["switched"]),
    "orders_placed": str(trade["orders_placed"]),
    "account_balance": float(trade["account_balance"]) if trade["account_balance"] is not None else None,
    "account_equity": float(trade["account_equity"]) if trade["account_equity"] is not None else None,
    "account_cash": float(trade["account_cash"]) if trade["account_cash"] is not None else None,
    "positions_count": int(trade["positions_count"]) if trade["positions_count"] is not None else 0,
    "positions_market_value": float(trade["positions_market_value"]) if trade["positions_market_value"] is not None else None,
    "positions_unrealized_pl": float(trade["positions_unrealized_pl"]) if trade["positions_unrealized_pl"] is not None else None,
    "hourly_pnl": float(trade["hourly_pnl"]) if trade["hourly_pnl"] is not None else None,
    "hourly_pnl_pct": float(trade["hourly_pnl_pct"]) if trade["hourly_pnl_pct"] is not None else None,
    "vix_source_date": vix_source_date,
    "pit_lag_days": pit_lag_days,
    "bias_gate_passed": bias_gate_ok,
    "model_version": model_version,
    "execution_mode": trade["execution_mode"],
    "source": "rt_saxo_signal_audit_v4",
    "ingestion_ts": audit_ts,
}

sdf = spark.createDataFrame([audit_row], schema=schema)
append_dataframe_with_retry(
    sdf,
    TARGET_TABLE,
    operation_name=f"append Saxo audit row into {TARGET_TABLE}",
)

print(f"Saxo audit log written to {TARGET_TABLE}")
print(f"Signal timestamp: {signal['timestamp']}")
print(f"Trade timestamp: {trade['trade_ts']}")
print(f"Execution mode: {trade['execution_mode']}")
print(f"Regime: {signal['regime_name']} ({signal['regime_int']})")
