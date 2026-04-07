# Databricks notebook source
# RT_08: Saxo trade execution / shadow account snapshot

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config

import json
import time
from datetime import datetime, timezone

import requests
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

AUTH_BASE_URL = "https://live.logonvalidation.net"
BASE_URL = "https://gateway.saxobank.com/openapi"
REDIRECT_URI = "https://localhost/callback"

CLIENT_ID = "5b0974db36f54f3b96ec07b8f5fc657d"
CLIENT_SECRET = "8a7f591c89ca4f0299256b041ba891e2"
CLIENT_KEY = "hLMMvwa3Mzmiem2AMhpKwA=="
ACCOUNT_KEY = "I|QvYu7A1ynLbi6kR32qfQ=="
TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"

SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_trade_log"

NOTIONAL_FRACTION = 0.95

runtime_config = load_runtime_instrument_config(spark, CATALOG, SCHEMA)
PRIMARY_SYMBOL = runtime_config["primary_symbol"]
SECONDARY_SYMBOL = runtime_config["secondary_symbol"]
PRICE_SYMBOLS = runtime_config["market_symbols"] or [PRIMARY_SYMBOL, SECONDARY_SYMBOL]

SAXO_ASSET_MAP = {
    runtime_config["position_map"][0]: {"symbol": runtime_config["regime_ticker"][0], "Uic": 4328771, "AssetType": "Etf"},
    runtime_config["position_map"][1]: {"symbol": runtime_config["regime_ticker"][1], "Uic": 36590, "AssetType": "Etf"},
    runtime_config["position_map"][2]: None,
}

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
    "account_balance": "DOUBLE",
    "account_equity": "DOUBLE",
    "account_cash": "DOUBLE",
    "last_equity": "DOUBLE",
    "hourly_pnl": "DOUBLE",
    "hourly_pnl_pct": "DOUBLE",
    "day_pnl": "DOUBLE",
    "day_pnl_pct": "DOUBLE",
    "positions_json": "STRING",
    "positions_count": "INT",
    "positions_market_value": "DOUBLE",
    "positions_unrealized_pl": "DOUBLE",
    "positions_unrealized_plpc": "DOUBLE",
    "execution_mode": "STRING",
    "source": "STRING",
    "ingestion_ts": "TIMESTAMP",
}


try:
    dbutils.widgets.dropdown("execution_mode", "shadow", ["shadow", "live"])
    EXECUTION_MODE = (dbutils.widgets.get("execution_mode") or "shadow").strip().lower()
except Exception:
    EXECUTION_MODE = "shadow"

if EXECUTION_MODE not in {"shadow", "live"}:
    raise ValueError(f"Unsupported execution_mode: {EXECUTION_MODE}")


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


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
            account_balance DOUBLE,
            account_equity DOUBLE,
            account_cash DOUBLE,
            last_equity DOUBLE,
            hourly_pnl DOUBLE,
            hourly_pnl_pct DOUBLE,
            day_pnl DOUBLE,
            day_pnl_pct DOUBLE,
            positions_json STRING,
            positions_count INT,
            positions_market_value DOUBLE,
            positions_unrealized_pl DOUBLE,
            positions_unrealized_plpc DOUBLE,
            execution_mode STRING,
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


def load_token():
    try:
        return json.loads(dbutils.fs.head(TOKEN_STATE_PATH))
    except Exception:
        return None


def get_access_token_readonly():
    state = load_token()
    if not state:
        raise RuntimeError("Token state missing — run Saxo OAuth bootstrap.")

    now = time.time()
    if state["expires_at"] < now:
        raise RuntimeError("Token expired — refresher job failed or not running.")

    return state["access_token"]


def request_json(method, url, headers, payload=None):
    resp = requests.request(method, url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    if not resp.text:
        return {}
    return resp.json()


def fetch_positions(headers):
    return request_json(
        "GET",
        f"{BASE_URL}/port/v1/positions/me?ClientKey={CLIENT_KEY}",
        headers,
    )


def fetch_balance(headers):
    return request_json(
        "GET",
        f"{BASE_URL}/port/v1/balances/me?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}",
        headers,
    )


def get_price(headers, uic, asset_type):
    quote_payload = request_json(
        "GET",
        f"{BASE_URL}/trade/v1/infoprices?Uic={uic}&AssetType={asset_type}",
        headers,
    )
    quote = quote_payload.get("Quote", {})
    return safe_float(
        first_present(
            quote.get("Mid"),
            quote.get("Ask"),
            quote.get("Last"),
            quote.get("Bid"),
        )
    )


def active_positions(positions_payload):
    rows = []
    for position in positions_payload.get("Data", []):
        base = position.get("PositionBase", {})
        amount = safe_float(base.get("Amount")) or 0.0
        if amount == 0.0:
            continue
        rows.append(
            {
                "raw": position,
                "amount": amount,
                "uic": base.get("Uic"),
                "asset_type": base.get("AssetType"),
                "symbol": first_present(base.get("Symbol"), position.get("DisplayAndFormat", {}).get("Symbol"), str(base.get("Uic"))),
            }
        )
    return rows


def serialize_positions(positions_payload):
    payload = []
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    weighted_plpc_numerator = 0.0

    for row in active_positions(positions_payload):
        raw = row["raw"]
        base = raw.get("PositionBase", {})
        net_view = raw.get("NetPositionView", {})
        position_view = raw.get("PositionView", {})

        market_value = safe_float(
            first_present(
                raw.get("MarketValue"),
                net_view.get("MarketValue"),
                position_view.get("MarketValue"),
                net_view.get("Exposure"),
            )
        ) or 0.0
        unrealized_pl = safe_float(
            first_present(
                raw.get("UnrealizedProfitLoss"),
                net_view.get("ProfitLossOnTrade"),
                net_view.get("ProfitLossOnTradeInBaseCurrency"),
                position_view.get("ProfitLossOnTrade"),
            )
        ) or 0.0
        current_price = safe_float(first_present(raw.get("CurrentPrice"), base.get("CurrentPrice")))
        avg_entry_price = safe_float(first_present(base.get("AverageOpenPrice"), base.get("OpenPrice")))
        unrealized_plpc = (
            unrealized_pl / abs(market_value)
            if market_value not in (0.0, None)
            else None
        )

        payload.append(
            {
                "symbol": row["symbol"],
                "uic": row["uic"],
                "asset_type": row["asset_type"],
                "qty": row["amount"],
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


def extract_balance_metrics(balance_payload):
    account_equity = safe_float(
        first_present(
            balance_payload.get("TotalValue"),
            balance_payload.get("NetEquityForMargin"),
            balance_payload.get("PortfolioValue"),
        )
    )
    account_balance = safe_float(
        first_present(
            balance_payload.get("CashBalance"),
            balance_payload.get("AvailableFunds"),
            balance_payload.get("CashAvailableForTrading"),
            balance_payload.get("Balance"),
        )
    )
    day_pnl = safe_float(
        first_present(
            balance_payload.get("ProfitLoss"),
            balance_payload.get("ProfitLossOnCurrentPositions"),
            balance_payload.get("UnrealizedProfitLoss"),
        )
    )
    return {
        "account_equity": account_equity,
        "account_balance": account_balance,
        "account_cash": account_balance,
        "day_pnl": day_pnl,
    }


def close_position(headers, position_row):
    side = "Sell" if position_row["amount"] > 0 else "Buy"
    order = {
        "AccountKey": ACCOUNT_KEY,
        "Uic": position_row["uic"],
        "AssetType": position_row["asset_type"],
        "BuySell": side,
        "OrderType": "Market",
        "Amount": abs(position_row["amount"]),
        "ManualOrder": True,
        "OrderDuration": {"DurationType": "DayOrder"},
    }
    resp = requests.post(
        f"{BASE_URL}/trade/v2/orders",
        headers=headers,
        json=order,
        timeout=30,
    )
    return {
        "action": "close",
        "uic": position_row["uic"],
        "asset_type": position_row["asset_type"],
        "side": side,
        "amount": abs(position_row["amount"]),
        "status_code": resp.status_code,
        "response": resp.text,
    }


def place_market_buy(headers, target_asset, qty):
    order = {
        "AccountKey": ACCOUNT_KEY,
        "Uic": target_asset["Uic"],
        "AssetType": target_asset["AssetType"],
        "BuySell": "Buy",
        "OrderType": "Market",
        "Amount": qty,
        "ManualOrder": False,
        "OrderDuration": {"DurationType": "DayOrder"},
    }
    resp = requests.post(
        f"{BASE_URL}/trade/v2/orders",
        headers=headers,
        json=order,
        timeout=30,
    )
    return {
        "action": "buy",
        "uic": target_asset["Uic"],
        "asset_type": target_asset["AssetType"],
        "symbol": target_asset["symbol"],
        "amount": qty,
        "status_code": resp.status_code,
        "response": resp.text,
    }


def latest_close(symbol):
    rows = spark.sql(
        f"""
        SELECT close
        FROM {MARKET_TABLE}
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).collect()
    return safe_float(rows[0]["close"]) if rows else None


ensure_trade_table()

access_token = get_access_token_readonly()
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

auth_test = requests.get(f"{BASE_URL}/port/v1/users/me", headers=headers, timeout=30)
if auth_test.status_code != 200:
    raise RuntimeError(f"Saxo auth failed: {auth_test.text}")

signal = spark.sql(
    f"""
    SELECT *
    FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC
    LIMIT 1
    """
).collect()[0]

signal_ts = signal["timestamp"]
regime_int = int(signal["regime_int"])
regime_name = signal["regime_name"]
target_position = signal["position"]
target_asset = SAXO_ASSET_MAP.get(target_position)
target_ticker = target_asset["symbol"] if target_asset else "CASH"

pre_positions_payload = fetch_positions(headers)
pre_active_positions = active_positions(pre_positions_payload)
pre_balance_payload = fetch_balance(headers)
pre_balance = extract_balance_metrics(pre_balance_payload)

last_mode_equity_row = spark.sql(
    f"""
    SELECT account_equity
    FROM {TARGET_TABLE}
    WHERE execution_mode = '{EXECUTION_MODE}'
      AND account_equity IS NOT NULL
    ORDER BY trade_ts DESC
    LIMIT 1
    """
).collect()
last_equity = safe_float(last_mode_equity_row[0]["account_equity"]) if last_mode_equity_row else None

switched = False
orders_placed = []
transaction_cost = 0.0

already_in_cash = target_asset is None and not pre_active_positions
already_in_target = (
    target_asset is not None
    and len(pre_active_positions) == 1
    and pre_active_positions[0]["uic"] == target_asset["Uic"]
    and pre_active_positions[0]["asset_type"] == target_asset["AssetType"]
)

if EXECUTION_MODE == "shadow":
    orders_placed.append({"action": "shadow", "message": "No execution performed"})
else:
    last_live_trade = spark.sql(
        f"""
        SELECT target_position, trade_ts
        FROM {TARGET_TABLE}
        WHERE execution_mode = 'live'
        ORDER BY trade_ts DESC
        LIMIT 1
        """
    ).collect()

    if last_live_trade:
        last_position = last_live_trade[0]["target_position"]
        last_trade_ts = last_live_trade[0]["trade_ts"].replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_since = (now - last_trade_ts).total_seconds() / 3600
        if last_position == target_position and hours_since < 1:
            orders_placed.append({"action": "skip", "message": "Duplicate signal within 1 hour"})
        elif already_in_cash or already_in_target:
            orders_placed.append({"action": "skip", "message": "Portfolio already aligned"})
        else:
            switched = True
    elif already_in_cash or already_in_target:
        orders_placed.append({"action": "skip", "message": "Portfolio already aligned"})
    else:
        switched = True

    if switched:
        if target_asset is None:
            for row in pre_active_positions:
                orders_placed.append(close_position(headers, row))
                time.sleep(1)
        else:
            for row in pre_active_positions:
                if row["uic"] != target_asset["Uic"] or row["asset_type"] != target_asset["AssetType"]:
                    orders_placed.append(close_position(headers, row))
                    time.sleep(1)

            refreshed_balance_payload = fetch_balance(headers)
            refreshed_balance = extract_balance_metrics(refreshed_balance_payload)
            target_value = safe_float(refreshed_balance["account_equity"]) or safe_float(pre_balance["account_equity"])
            target_value = target_value * NOTIONAL_FRACTION if target_value is not None else None
            latest_price = get_price(headers, target_asset["Uic"], target_asset["AssetType"])
            qty = max(1, int(target_value / latest_price)) if target_value and latest_price else 0

            if qty > 0:
                orders_placed.append(place_market_buy(headers, target_asset, qty))
            else:
                orders_placed.append({"action": "skip", "message": "Unable to size order"})

time.sleep(2)

post_positions_payload = fetch_positions(headers)
post_balance_payload = fetch_balance(headers)
post_balance = extract_balance_metrics(post_balance_payload)
position_snapshot = serialize_positions(post_positions_payload)

account_equity = post_balance["account_equity"]
account_balance = post_balance["account_balance"]
account_cash = post_balance["account_cash"]

hourly_pnl = (
    account_equity - last_equity
    if account_equity is not None and last_equity not in (None, 0.0)
    else None
)
hourly_pnl_pct = (
    hourly_pnl / last_equity
    if hourly_pnl is not None and last_equity not in (None, 0.0)
    else None
)
day_pnl = post_balance["day_pnl"]
day_pnl_pct = (
    day_pnl / last_equity
    if day_pnl is not None and last_equity not in (None, 0.0)
    else None
)

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
        StructField("account_balance", DoubleType(), True),
        StructField("account_equity", DoubleType(), True),
        StructField("account_cash", DoubleType(), True),
        StructField("last_equity", DoubleType(), True),
        StructField("hourly_pnl", DoubleType(), True),
        StructField("hourly_pnl_pct", DoubleType(), True),
        StructField("day_pnl", DoubleType(), True),
        StructField("day_pnl_pct", DoubleType(), True),
        StructField("positions_json", StringType(), True),
        StructField("positions_count", IntegerType(), True),
        StructField("positions_market_value", DoubleType(), True),
        StructField("positions_unrealized_pl", DoubleType(), True),
        StructField("positions_unrealized_plpc", DoubleType(), True),
        StructField("execution_mode", StringType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

trade_row = {
    "trade_ts": trade_ts,
    "signal_ts": signal_ts,
    "regime_int": regime_int,
    "regime_name": regime_name,
    "target_position": target_position,
    "target_ticker": target_ticker,
    "switched": switched,
    "transaction_cost": transaction_cost,
    "orders_placed": json.dumps(orders_placed),
    "spy_close": latest_close(PRIMARY_SYMBOL),
    "qqq_close": latest_close(SECONDARY_SYMBOL),
    "account_balance": account_balance,
    "account_equity": account_equity,
    "account_cash": account_cash,
    "last_equity": last_equity,
    "hourly_pnl": hourly_pnl,
    "hourly_pnl_pct": hourly_pnl_pct,
    "day_pnl": day_pnl,
    "day_pnl_pct": day_pnl_pct,
    "positions_json": position_snapshot["positions_json"],
    "positions_count": position_snapshot["positions_count"],
    "positions_market_value": position_snapshot["positions_market_value"],
    "positions_unrealized_pl": position_snapshot["positions_unrealized_pl"],
    "positions_unrealized_plpc": position_snapshot["positions_unrealized_plpc"],
    "execution_mode": EXECUTION_MODE,
    "source": "rt_saxo_shadow_snapshot" if EXECUTION_MODE == "shadow" else "rt_saxo_live_execution",
    "ingestion_ts": None,
}

sdf = spark.createDataFrame([trade_row], schema=schema).withColumn(
    "ingestion_ts", F.current_timestamp()
)
sdf.write.format("delta").mode("append").saveAsTable(TARGET_TABLE)

print(f"Saxo trade log written to {TARGET_TABLE}")
print(f"Execution mode: {EXECUTION_MODE}")
print(f"Signal timestamp: {signal_ts}")
print(f"Target allocation: {target_position}")
print(f"Orders: {trade_row['orders_placed']}")
print(f"Account balance: {account_balance}")
print(f"Account equity: {account_equity}")
print(f"Hourly PnL: {hourly_pnl}")
print(f"Open positions: {position_snapshot['positions_count']}")
