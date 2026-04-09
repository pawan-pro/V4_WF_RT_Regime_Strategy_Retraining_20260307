# Databricks notebook source
# RT_10: Saxo dashboard feed

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config
# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
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
TRADE_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_trade_log"
AUDIT_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_signal_audit_log"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
MACRO_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.gold_rt_saxo_dashboard_feed"

runtime_config = load_runtime_instrument_config(spark, CATALOG, SCHEMA)
PRIMARY_SYMBOL = runtime_config["primary_symbol"]
SECONDARY_SYMBOL = runtime_config["secondary_symbol"]
PRICE_SYMBOLS_SQL = ", ".join([f"'{symbol}'" for symbol in {PRIMARY_SYMBOL, SECONDARY_SYMBOL} if symbol])

try:
    dbutils.widgets.dropdown("execution_mode", "auto", ["auto", "shadow", "live"])
    EXECUTION_MODE = (dbutils.widgets.get("execution_mode") or "auto").strip().lower()
except Exception:
    EXECUTION_MODE = "auto"

if EXECUTION_MODE not in {"auto", "shadow", "live"}:
    raise ValueError(f"Unsupported execution_mode: {EXECUTION_MODE}")


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


def to_naive_datetime(value):
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = pd.to_datetime(value).to_pydatetime()
        except Exception:
            return None

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


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


def resolve_execution_mode():
    if EXECUTION_MODE != "auto":
        return EXECUTION_MODE

    rows = spark.sql(
        f"""
        SELECT execution_mode
        FROM {TRADE_LOG}
        ORDER BY trade_ts DESC
        LIMIT 1
        """
    ).collect()
    if rows and rows[0]["execution_mode"] in {"shadow", "live"}:
        return rows[0]["execution_mode"]
    return "shadow"


def get_current_regime_start_ts(execution_mode, current_regime_int, current_signal_ts):
    if current_regime_int is None:
        return to_naive_datetime(current_signal_ts)

    def first_contiguous_regime_start(rows, regime_col):
        if rows.empty:
            return None
        if regime_col not in rows.columns:
            return None

        contiguous_start_ts = None
        for _, row in rows.iterrows():
            if pd.isna(row[regime_col]):
                continue
            if int(row[regime_col]) != int(current_regime_int):
                break
            contiguous_start_ts = row["event_ts"]
        return contiguous_start_ts

    if spark.catalog.tableExists(AUDIT_LOG):
        audit_rows = (
            spark.sql(
                f"""
                SELECT
                    audit_ts AS event_ts,
                    regime_int
                FROM {AUDIT_LOG}
                WHERE execution_mode = '{execution_mode}'
                  AND audit_ts IS NOT NULL
                  AND regime_int IS NOT NULL
                ORDER BY audit_ts DESC
                """
            )
            .toPandas()
        )
        if not audit_rows.empty:
            audit_rows["event_ts"] = pd.to_datetime(audit_rows["event_ts"])
            start_ts = first_contiguous_regime_start(audit_rows, "regime_int")
            if start_ts is not None:
                return to_naive_datetime(start_ts)

    if spark.catalog.tableExists(TRADE_LOG):
        trade_rows = (
            spark.sql(
                f"""
                SELECT
                    trade_ts AS event_ts,
                    regime_int
                FROM {TRADE_LOG}
                WHERE execution_mode = '{execution_mode}'
                  AND trade_ts IS NOT NULL
                  AND regime_int IS NOT NULL
                ORDER BY trade_ts DESC
                """
            )
            .toPandas()
        )
        if not trade_rows.empty:
            trade_rows["event_ts"] = pd.to_datetime(trade_rows["event_ts"])
            start_ts = first_contiguous_regime_start(trade_rows, "regime_int")
            if start_ts is not None:
                return to_naive_datetime(start_ts)

    return to_naive_datetime(current_signal_ts)


def load_performance_history(current_regime_start_ts):
    history = spark.sql(
        f"""
        SELECT
            trade_ts,
            signal_ts,
            regime_int,
            regime_name,
            account_equity,
            spy_close
        FROM {TRADE_LOG}
        WHERE account_equity IS NOT NULL
          AND spy_close IS NOT NULL
          AND execution_mode = '{EXECUTION_MODE}'
        ORDER BY trade_ts
        """
    ).toPandas()

    if history.empty:
        return history

    history["trade_ts"] = pd.to_datetime(history["trade_ts"])
    history["signal_ts"] = pd.to_datetime(history["signal_ts"])
    history["account_equity"] = history["account_equity"].astype(float)
    history["spy_close"] = history["spy_close"].astype(float)
    if current_regime_start_ts is not None:
        history = history[history["trade_ts"] >= pd.to_datetime(current_regime_start_ts)].copy()
        if history.empty:
            return history

    initial_equity = float(history["account_equity"].iloc[0])
    initial_spy_close = float(history["spy_close"].iloc[0])

    history["strategy_value"] = history["account_equity"]
    history["spy_benchmark"] = initial_equity * history["spy_close"] / initial_spy_close
    history["strategy_return_pct"] = history["strategy_value"] / initial_equity - 1.0
    history["benchmark_return_pct"] = history["spy_benchmark"] / initial_equity - 1.0
    history["alpha_pct"] = history["strategy_return_pct"] - history["benchmark_return_pct"]

    return history


def render_performance_chart(performance_history, regime_start_ts=None):
    if performance_history.empty or performance_history.shape[0] < 2:
        print("Not enough Saxo history to render performance chart.")
        return

    chart_df = performance_history.sort_values("trade_ts").reset_index(drop=True).copy()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    ax1.plot(chart_df["trade_ts"], chart_df["strategy_value"], label="Saxo Strategy", linewidth=2)
    ax1.plot(chart_df["trade_ts"], chart_df["spy_benchmark"], label="SPY Benchmark", alpha=0.75)
    regime_title = "Saxo Live Portfolio Performance vs SPY - Since Current Regime Began"

    if regime_start_ts is not None:
        regime_title = f"{regime_title} ({pd.to_datetime(regime_start_ts)})"
    ax1.set_title(regime_title)
    ax1.set_ylabel("Portfolio Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    regimes = chart_df["regime_int"]
    ax2.fill_between(chart_df["trade_ts"], 0, 1, where=(regimes == 0), color="green", alpha=0.35, label="Expansion")
    ax2.fill_between(chart_df["trade_ts"], 0, 1, where=(regimes == 1), color="gold", alpha=0.35, label="Slowdown")
    ax2.fill_between(chart_df["trade_ts"], 0, 1, where=(regimes == 2), color="red", alpha=0.35, label="Recession")
    ax2.set_yticks([])
    ax2.set_ylabel("Regime")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


signal = spark.sql(
    f"""
    SELECT *
    FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC
    LIMIT 1
    """
).collect()[0]
signal_data = signal.asDict(recursive=True)
EXECUTION_MODE = resolve_execution_mode()

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
    WHERE execution_mode = '{EXECUTION_MODE}'
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
    WHERE execution_mode = '{EXECUTION_MODE}'
    """
).collect()[0]

starting_equity_row = spark.sql(
    f"""
    SELECT account_equity
    FROM {TRADE_LOG}
    WHERE account_equity IS NOT NULL
      AND execution_mode = '{EXECUTION_MODE}'
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

current_regime_start_ts = get_current_regime_start_ts(
    EXECUTION_MODE,
    signal["regime_int"],
    signal["timestamp"],
)

regime_start_equity = None
regime_return_pct = None
regime_drawdown_pct = None
if current_regime_start_ts is not None and signal["regime_int"] is not None:
    regime_start_row = spark.sql(
        f"""
        SELECT account_equity
        FROM {TRADE_LOG}
        WHERE execution_mode = '{EXECUTION_MODE}'
          AND trade_ts >= '{current_regime_start_ts}'
          AND regime_int = {int(signal["regime_int"])}
          AND account_equity IS NOT NULL
        ORDER BY trade_ts ASC
        LIMIT 1
        """
    ).collect()
    if regime_start_row:
        regime_start_equity = maybe_float(regime_start_row[0]["account_equity"])
        if regime_start_equity not in (None, 0.0) and latest_equity is not None:
            regime_return_pct = latest_equity / regime_start_equity - 1.0

    regime_peak_row = spark.sql(
        f"""
        SELECT MAX(account_equity) AS regime_peak_equity
        FROM {TRADE_LOG}
        WHERE execution_mode = '{EXECUTION_MODE}'
          AND trade_ts >= '{current_regime_start_ts}'
          AND account_equity IS NOT NULL
        """
    ).collect()[0]
    regime_peak_equity = maybe_float(regime_peak_row["regime_peak_equity"]) if regime_peak_row else None
    if regime_peak_equity not in (None, 0.0) and latest_equity is not None:
        regime_drawdown_pct = latest_equity / regime_peak_equity - 1.0

regime_dist = spark.sql(
    f"""
    SELECT
        ROUND(100.0 * SUM(CASE WHEN regime_int = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_expansion,
        ROUND(100.0 * SUM(CASE WHEN regime_int = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_slowdown,
        ROUND(100.0 * SUM(CASE WHEN regime_int = 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_recession
    FROM {AUDIT_LOG}
    WHERE execution_mode = '{EXECUTION_MODE}'
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
performance_history = load_performance_history(current_regime_start_ts)
latest_alpha_pct = (
    float(performance_history["alpha_pct"].iloc[-1])
    if not performance_history.empty
    else None
)

schema = StructType(
    [
        StructField("feed_ts", TimestampType(), True),
        StructField("dashboard_ts", TimestampType(), True),
        StructField("signal_ts", TimestampType(), True),
        StructField("trade_ts", TimestampType(), True),
        StructField("current_regime_start_ts", TimestampType(), True),
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
        StructField("account_balance", DoubleType(), True),
        StructField("account_equity", DoubleType(), True),
        StructField("account_cash", DoubleType(), True),
        StructField("positions_json", StringType(), True),
        StructField("positions_count", IntegerType(), True),
        StructField("positions_market_value", DoubleType(), True),
        StructField("positions_unrealized_pl", DoubleType(), True),
        StructField("positions_unrealized_plpc", DoubleType(), True),
        StructField("hourly_pnl", DoubleType(), True),
        StructField("hourly_pnl_pct", DoubleType(), True),
        StructField("day_pnl", DoubleType(), True),
        StructField("day_pnl_pct", DoubleType(), True),
        StructField("total_return_pct", DoubleType(), True),
        StructField("regime_return_pct", DoubleType(), True),
        StructField("alpha_pct", DoubleType(), True),
        StructField("drawdown_pct", DoubleType(), True),
        StructField("regime_drawdown_pct", DoubleType(), True),
        StructField("total_cycles", IntegerType(), True),
        StructField("total_switches", IntegerType(), True),
        StructField("peak_equity", DoubleType(), True),
        StructField("pct_expansion", DoubleType(), True),
        StructField("pct_slowdown", DoubleType(), True),
        StructField("pct_recession", DoubleType(), True),
        StructField("execution_mode", StringType(), True),
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
    "current_regime_start_ts": current_regime_start_ts,
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
    "account_balance": maybe_float(trade["account_balance"]),
    "account_equity": latest_equity,
    "account_cash": maybe_float(trade["account_cash"]),
    "positions_json": trade["positions_json"],
    "positions_count": maybe_int(trade["positions_count"]),
    "positions_market_value": maybe_float(trade["positions_market_value"]),
    "positions_unrealized_pl": maybe_float(trade["positions_unrealized_pl"]),
    "positions_unrealized_plpc": maybe_float(trade["positions_unrealized_plpc"]),
    "hourly_pnl": maybe_float(trade["hourly_pnl"]),
    "hourly_pnl_pct": maybe_float(trade["hourly_pnl_pct"]),
    "day_pnl": maybe_float(trade["day_pnl"]),
    "day_pnl_pct": maybe_float(trade["day_pnl_pct"]),
    "total_return_pct": total_return_pct,
    "regime_return_pct": regime_return_pct,
    "alpha_pct": latest_alpha_pct,
    "drawdown_pct": drawdown_pct,
    "regime_drawdown_pct": regime_drawdown_pct,
    "total_cycles": maybe_int(portfolio_stats["total_cycles"]),
    "total_switches": maybe_int(portfolio_stats["total_switches"]),
    "peak_equity": peak_equity,
    "pct_expansion": maybe_float(regime_dist["pct_expansion"]),
    "pct_slowdown": maybe_float(regime_dist["pct_slowdown"]),
    "pct_recession": maybe_float(regime_dist["pct_recession"]),
    "execution_mode": trade["execution_mode"],
    "vix_source_date": str(vix_source_date) if vix_source_date is not None else None,
    "model_version": model_version,
    "source": "rt_saxo_dashboard_feed_v4",
    "ingestion_ts": dashboard_ts,
}

sdf = spark.createDataFrame([dashboard_row], schema=schema)
overwrite_dataframe_with_retry(
    sdf,
    TARGET_TABLE,
    options={"overwriteSchema": "true"},
    operation_name=f"overwrite Saxo dashboard feed table {TARGET_TABLE}",
)

render_performance_chart(performance_history, current_regime_start_ts)

print(f"Saxo dashboard feed written to {TARGET_TABLE}")
print(f"Dashboard timestamp: {dashboard_ts}")
print(f"Execution mode: {trade['execution_mode']}")
print(f"Regime: {dashboard_row['regime_name']} ({dashboard_row['regime_int']})")
if current_regime_start_ts is not None:
    print(f"Current regime start: {current_regime_start_ts}")
print(f"Target allocation: {dashboard_row['position']}")
print(f"Account balance: {dashboard_row['account_balance']}")
print(f"Account equity: {dashboard_row['account_equity']}")
if regime_start_equity is not None:
    print(f"Regime start equity: {regime_start_equity}")
if regime_return_pct is not None:
    print(f"Regime return pct: {regime_return_pct:+.2%}")
if regime_drawdown_pct is not None:
    print(f"Regime drawdown pct: {regime_drawdown_pct:+.2%}")
print(f"Hourly PnL: {dashboard_row['hourly_pnl']}")
print(
    "Alpha pct: "
    + (f"{dashboard_row['alpha_pct']:+.2%}" if dashboard_row["alpha_pct"] is not None else "N/A")
)
