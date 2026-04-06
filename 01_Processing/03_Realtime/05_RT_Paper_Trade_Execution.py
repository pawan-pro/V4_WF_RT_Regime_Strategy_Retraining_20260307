# Databricks notebook source
# RT_05: Paper Trade Execution

# MAGIC %pip install alpaca-py

import json
import time
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

ALPACA_KEY = "PKBPXGRUOI6AQB43ENROLTREJX"
ALPACA_SECRET = "Dtw5J51BMZrWpgHg1auvxnpazjPcwi6sXg34V4FprNcQ"
PAPER = True

SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_paper_trade_log"

REGIME_TICKER = {0: "QQQ", 1: "SPY", 2: None}
TRADE_VALUE = 100_000

TRADE_COLUMNS = {
    "trade_ts": "TIMESTAMP",
    "signal_ts": "TIMESTAMP",
    "regime_int": "INT",
    "regime_name": "STRING",
    "target_position": "STRING",
    "target_ticker": "STRING",
    "switched": "BOOLEAN",
    "transaction_cost": "DOUBLE",
    "orders_placed": "STRING",
    "spy_close": "DOUBLE",
    "qqq_close": "DOUBLE",
    "account_equity": "DOUBLE",
    "account_cash": "DOUBLE",
    "last_equity": "DOUBLE",
    "day_pnl": "DOUBLE",
    "day_pnl_pct": "DOUBLE",
    "positions_json": "STRING",
    "positions_count": "INT",
    "positions_market_value": "DOUBLE",
    "positions_unrealized_pl": "DOUBLE",
    "positions_unrealized_plpc": "DOUBLE",
    "source": "STRING",
    "ingestion_ts": "TIMESTAMP",
}


def existing_columns(table_name):
    if not spark.catalog.tableExists(table_name):
        return set()
    return {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {table_name}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }


def ensure_trade_table():
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            trade_ts TIMESTAMP,
            signal_ts TIMESTAMP,
            regime_int INT,
            regime_name STRING,
            target_position STRING,
            target_ticker STRING,
            switched BOOLEAN,
            transaction_cost DOUBLE,
            orders_placed STRING,
            spy_close DOUBLE,
            qqq_close DOUBLE,
            account_equity DOUBLE,
            account_cash DOUBLE,
            last_equity DOUBLE,
            day_pnl DOUBLE,
            day_pnl_pct DOUBLE,
            positions_json STRING,
            positions_count INT,
            positions_market_value DOUBLE,
            positions_unrealized_pl DOUBLE,
            positions_unrealized_plpc DOUBLE,
            source STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    current = existing_columns(TARGET_TABLE)
    for name, dtype in TRADE_COLUMNS.items():
        if name not in current:
            spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {name} {dtype}")


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def serialize_positions(position_rows):
    payload = []
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    weighted_plpc_numerator = 0.0

    for position in position_rows:
        market_value = safe_float(position.market_value) or 0.0
        unrealized_pl = safe_float(position.unrealized_pl) or 0.0
        unrealized_plpc = safe_float(position.unrealized_plpc)
        qty = safe_float(position.qty)
        avg_entry_price = safe_float(position.avg_entry_price)
        current_price = safe_float(position.current_price)

        payload.append(
            {
                "symbol": position.symbol,
                "qty": qty,
                "side": getattr(position, "side", None),
                "avg_entry_price": avg_entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
            }
        )

        total_market_value += market_value
        total_unrealized_pl += unrealized_pl
        if unrealized_plpc is not None:
            weighted_plpc_numerator += abs(market_value) * unrealized_plpc

    weighted_plpc = None
    if total_market_value:
        weighted_plpc = weighted_plpc_numerator / abs(total_market_value)

    return {
        "positions_json": json.dumps(payload),
        "positions_count": len(payload),
        "positions_market_value": total_market_value,
        "positions_unrealized_pl": total_unrealized_pl,
        "positions_unrealized_plpc": weighted_plpc,
    }


ensure_trade_table()

trading_client = TradingClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=PAPER,
)

signal = spark.sql(
    f"""
    SELECT * FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC
    LIMIT 1
    """
).collect()[0]

regime_int = signal["regime_int"]
regime_name = signal["regime_name"]
position_str = signal["position"]
signal_ts = signal["timestamp"]
target_ticker = REGIME_TICKER[regime_int]

pre_positions = trading_client.get_all_positions()
held_tickers = {position.symbol: safe_float(position.qty) for position in pre_positions}
held_symbols = list(held_tickers.keys())

already_in_cash = target_ticker is None and not held_symbols
already_in_target = (
    target_ticker is not None
    and target_ticker in held_tickers
    and len(held_symbols) == 1
)

orders_placed = []
switched = False
txn_cost = 0.0

if already_in_cash or already_in_target:
    print("Portfolio already aligned with latest regime signal.")
else:
    switched = True
    txn_cost = 0.0005

    if held_symbols:
        print(f"Closing positions: {held_symbols}")
        for symbol in held_symbols:
            try:
                trading_client.close_position(symbol)
                orders_placed.append(f"CLOSE:{symbol}")
            except Exception as exc:
                orders_placed.append(f"CLOSE_FAILED:{symbol}:{exc}")

    if target_ticker is not None:
        price_row = spark.sql(
            f"""
            SELECT close
            FROM {MARKET_TABLE}
            WHERE symbol = '{target_ticker}'
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).collect()
        if not price_row:
            raise ValueError(f"No price data available for {target_ticker}")

        latest_price = float(price_row[0]["close"])
        qty = max(1, int(TRADE_VALUE / latest_price))

        try:
            order = trading_client.submit_order(
                MarketOrderRequest(
                    symbol=target_ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            orders_placed.append(f"BUY:{target_ticker}:{qty}:{order.id}")
        except Exception as exc:
            orders_placed.append(f"BUY_FAILED:{target_ticker}:{exc}")
    else:
        orders_placed.append("TARGET:CASH")

time.sleep(2)

post_account = trading_client.get_account()
post_positions = trading_client.get_all_positions()
position_snapshot = serialize_positions(post_positions)

equity = safe_float(post_account.equity)
cash = safe_float(post_account.cash)
last_equity = safe_float(getattr(post_account, "last_equity", None))
day_pnl = None if equity is None or last_equity in (None, 0.0) else equity - last_equity
day_pnl_pct = None if day_pnl is None or last_equity in (None, 0.0) else day_pnl / last_equity

price_map = {}
for symbol in ["SPY", "QQQ"]:
    rows = spark.sql(
        f"""
        SELECT close
        FROM {MARKET_TABLE}
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).collect()
    price_map[symbol] = float(rows[0]["close"]) if rows else None

trade_ts = datetime.now(timezone.utc).replace(tzinfo=None)

schema = StructType(
    [
        StructField("trade_ts", TimestampType(), True),
        StructField("signal_ts", TimestampType(), True),
        StructField("regime_int", IntegerType(), True),
        StructField("regime_name", StringType(), True),
        StructField("target_position", StringType(), True),
        StructField("target_ticker", StringType(), True),
        StructField("switched", BooleanType(), True),
        StructField("transaction_cost", DoubleType(), True),
        StructField("orders_placed", StringType(), True),
        StructField("spy_close", DoubleType(), True),
        StructField("qqq_close", DoubleType(), True),
        StructField("account_equity", DoubleType(), True),
        StructField("account_cash", DoubleType(), True),
        StructField("last_equity", DoubleType(), True),
        StructField("day_pnl", DoubleType(), True),
        StructField("day_pnl_pct", DoubleType(), True),
        StructField("positions_json", StringType(), True),
        StructField("positions_count", IntegerType(), True),
        StructField("positions_market_value", DoubleType(), True),
        StructField("positions_unrealized_pl", DoubleType(), True),
        StructField("positions_unrealized_plpc", DoubleType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

trade_row = {
    "trade_ts": trade_ts,
    "signal_ts": signal_ts,
    "regime_int": regime_int,
    "regime_name": regime_name,
    "target_position": position_str,
    "target_ticker": target_ticker if target_ticker else "CASH",
    "switched": switched,
    "transaction_cost": txn_cost,
    "orders_placed": "|".join(orders_placed) if orders_placed else "NONE",
    "spy_close": price_map.get("SPY"),
    "qqq_close": price_map.get("QQQ"),
    "account_equity": equity,
    "account_cash": cash,
    "last_equity": last_equity,
    "day_pnl": day_pnl,
    "day_pnl_pct": day_pnl_pct,
    "positions_json": position_snapshot["positions_json"],
    "positions_count": position_snapshot["positions_count"],
    "positions_market_value": position_snapshot["positions_market_value"],
    "positions_unrealized_pl": position_snapshot["positions_unrealized_pl"],
    "positions_unrealized_plpc": position_snapshot["positions_unrealized_plpc"],
    "source": "rt_paper_trade_alpaca",
    "ingestion_ts": None,
}

sdf = spark.createDataFrame([trade_row], schema=schema).withColumn(
    "ingestion_ts", F.current_timestamp()
)
sdf.write.format("delta").mode("append").saveAsTable(TARGET_TABLE)

print(f"Trade log written to {TARGET_TABLE}")
print(f"Signal timestamp: {signal_ts}")
print(f"Regime: {regime_name} ({regime_int})")
print(f"Target allocation: {position_str}")
print(f"Orders: {trade_row['orders_placed']}")
print(f"Equity: {equity}")
print(f"Cash: {cash}")
print(f"Day PnL: {day_pnl}")
print(f"Open positions: {position_snapshot['positions_count']}")
