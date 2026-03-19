# Databricks notebook source
# RT_05: Paper Trade Execution
# Reads regime signal → computes target position → places orders via Alpaca Paper Trading

%pip install alpaca-py

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType, TimestampType
from datetime import datetime, timezone

# ── CONFIGURATION ──


ALPACA_KEY    = "PKBPXGRUOI6AQB43ENROLTREJX"   # Paper trading account
ALPACA_SECRET = "Dtw5J51BMZrWpgHg1auvxnpazjPcwi6sXg34V4FprNcQ"  # ← confirm this is correct for paper account
PAPER         = True

SIGNAL_TABLE  = "pawan_quant_research_1.default.bronze_rt_regime_signal"
MARKET_TABLE  = "pawan_quant_research_1.default.bronze_rt_market_1h"
TARGET_TABLE  = "pawan_quant_research_1.default.bronze_rt_paper_trade_log"

# Regime → ETF mapping
REGIME_TICKER = {0: "QQQ", 1: "SPY", 2: None}  # None = CASH (close all)
TRADE_VALUE   = 100_000  # Notional per execution cycle ($100K paper)

# ── STEP 1: Init Alpaca trading client ──
trading_client = TradingClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=PAPER
)
account = trading_client.get_account()
equity  = float(account.equity)
cash    = float(account.cash)
print(f"✅ Alpaca Paper Account connected")
print(f"   Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}")

# ── STEP 2: Get latest regime signal ──
signal = spark.sql(f"""
    SELECT * FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC LIMIT 1
""").collect()[0]

regime_int   = signal["regime_int"]
regime_name  = signal["regime_name"]
position_str = signal["position"]
signal_ts    = signal["timestamp"]
target_ticker = REGIME_TICKER[regime_int]

print(f"\n📡 Signal: {regime_name} ({regime_int}) → {position_str} @ {signal_ts}")

# ── STEP 3: Check current Alpaca positions ──
positions     = trading_client.get_all_positions()
held_tickers  = {p.symbol: float(p.qty) for p in positions}
held_symbols  = list(held_tickers.keys())
print(f"   Current holdings: {held_symbols if held_symbols else 'CASH (no positions)'}")

# ── STEP 4: Determine if switch needed ──
currently_in_target = (
    target_ticker is not None and
    target_ticker in held_tickers and
    len(held_symbols) == 1
)

orders_placed = []

if currently_in_target:
    print(f"✅ Already holding {target_ticker} — no trade needed.")
    switched = False
    txn_cost = 0.0

else:
    switched = True
    txn_cost = 0.0005  # 5bps

    # ── STEP 4a: Close all existing positions ──
    if held_symbols:
        print(f"🔄 Closing existing positions: {held_symbols}")
        for sym in held_symbols:
            try:
                trading_client.close_position(sym)
                print(f"   ✅ Closed {sym}")
                orders_placed.append(f"CLOSE:{sym}")
            except Exception as e:
                print(f"   ❌ Failed to close {sym}: {e}")

    # ── STEP 4b: Open new position ──
    if target_ticker is not None:
        # Get latest price from bronze_rt_market_1h
        price_row = spark.sql(f"""
            SELECT close FROM {MARKET_TABLE}
            WHERE symbol = '{target_ticker}'
            ORDER BY timestamp DESC LIMIT 1
        """).collect()

        if price_row:
            latest_price = float(price_row[0]["close"])
            qty = max(1, int(TRADE_VALUE / latest_price))
            print(f"📈 Buying {qty} shares of {target_ticker} @ ~${latest_price:.2f}")

            try:
                order = trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=target_ticker,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                )
                print(f"   ✅ Order submitted: {order.id} | status: {order.status}")
                orders_placed.append(f"BUY:{target_ticker}:{qty}")
            except Exception as e:
                print(f"   ❌ Order failed: {e}")
        else:
            print(f"⚠️  No price data for {target_ticker} — skipping buy")

    else:
        print("💵 Regime = CASH — all positions closed, holding cash.")

# ── STEP 5: Get latest prices for log ──
price_map = {}
for sym in ["SPY", "QQQ"]:
    rows = spark.sql(f"""
        SELECT close FROM {MARKET_TABLE}
        WHERE symbol = '{sym}'
        ORDER BY timestamp DESC LIMIT 1
    """).collect()
    price_map[sym] = float(rows[0]["close"]) if rows else None

# ── STEP 6: Write trade log ──
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        trade_ts         TIMESTAMP,
        signal_ts        TIMESTAMP,
        regime_int       INT,
        regime_name      STRING,
        target_position  STRING,
        target_ticker    STRING,
        switched         BOOLEAN,
        transaction_cost DOUBLE,
        orders_placed    STRING,
        spy_close        DOUBLE,
        qqq_close        DOUBLE,
        account_equity   DOUBLE,
        account_cash     DOUBLE,
        source           STRING,
        ingestion_ts     TIMESTAMP
    )
    USING DELTA
""")

trade_ts = datetime.now(timezone.utc).replace(tzinfo=None)

schema = StructType([
    StructField("trade_ts",         TimestampType(), True),
    StructField("signal_ts",        TimestampType(), True),
    StructField("regime_int",       IntegerType(),   True),
    StructField("regime_name",      StringType(),    True),
    StructField("target_position",  StringType(),    True),
    StructField("target_ticker",    StringType(),    True),
    StructField("switched",         BooleanType(),   True),
    StructField("transaction_cost", DoubleType(),    True),
    StructField("orders_placed",    StringType(),    True),
    StructField("spy_close",        DoubleType(),    True),
    StructField("qqq_close",        DoubleType(),    True),
    StructField("account_equity",   DoubleType(),    True),
    StructField("account_cash",     DoubleType(),    True),
    StructField("source",           StringType(),    True),
])

trade_row = {
    "trade_ts":         trade_ts,
    "signal_ts":        signal_ts,
    "regime_int":       regime_int,
    "regime_name":      regime_name,
    "target_position":  position_str,
    "target_ticker":    target_ticker if target_ticker else "CASH",
    "switched":         switched,
    "transaction_cost": txn_cost,
    "orders_placed":    "|".join(orders_placed) if orders_placed else "NONE",
    "spy_close":        price_map.get("SPY"),
    "qqq_close":        price_map.get("QQQ"),
    "account_equity":   equity,
    "account_cash":     cash,
    "source":           "rt_paper_trade_alpaca",
}

sdf = spark.createDataFrame([trade_row], schema=schema) \
           .withColumn("ingestion_ts", F.current_timestamp())
sdf.write.format("delta").mode("append").saveAsTable(TARGET_TABLE)

print(f"\n✅ Trade log written → {TARGET_TABLE}")
print(f"\n{'='*50}")
print(f"  EXECUTION SUMMARY @ {trade_ts}")
print(f"{'='*50}")
print(f"  Regime:   {regime_name} ({regime_int})")
print(f"  Target:   {position_str}")
print(f"  Switched: {switched}")
print(f"  Orders:   {orders_placed if orders_placed else 'None'}")
print(f"  Equity:   ${equity:,.2f}")
print(f"  Cash:     ${cash:,.2f}")
print(f"{'='*50}")

# ── STEP 7: Confirm positions post-execution ──
import time
time.sleep(2)  # brief wait for order processing
post_positions = trading_client.get_all_positions()
print(f"\nPost-execution positions:")
if post_positions:
    for p in post_positions:
        print(f"  {p.symbol}: {p.qty} shares @ avg ${float(p.avg_entry_price):.2f} | MV: ${float(p.market_value):.2f}")
else:
    print("  No open positions (CASH)")