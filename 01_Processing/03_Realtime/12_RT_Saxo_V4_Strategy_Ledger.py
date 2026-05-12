# Databricks notebook source
# RT_12: Saxo V4 strategy-only ledger, equity curve, and transaction-cost analysis
#
# Production-safety scope:
# - Read-only against existing Saxo execution/audit/market tables.
# - Writes only the new gold analysis tables declared below.
# - Does not call Saxo APIs, token refreshers, order endpoints, or Databricks Jobs APIs.

# COMMAND ----------

# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

# COMMAND ----------

import json
from datetime import datetime, timezone
from typing import Any, Optional

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Constants and table names
# ---------------------------------------------------------------------------
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

BROKER_HISTORY_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_broker_trade_history"
INTERNAL_TRADE_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_trade_log"
AUDIT_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_signal_audit_log"
DASHBOARD_FEED = f"{CATALOG}.{SCHEMA}.gold_rt_saxo_dashboard_feed"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"

LEDGER_TABLE = f"{CATALOG}.{SCHEMA}.gold_saxo_v4_strategy_trade_ledger"
EQUITY_TABLE = f"{CATALOG}.{SCHEMA}.gold_saxo_v4_strategy_equity_curve"
TCA_TABLE = f"{CATALOG}.{SCHEMA}.gold_saxo_v4_transaction_cost_analysis"

V4_SYMBOLS = ["SPY", "QQQ"]
V4_SYMBOLS_SQL = "'SPY', 'QQQ'"
SAXO_UIC_TO_SYMBOL = {
    "36590": "SPY",
    "4328771": "QQQ",
}

SOURCE_BROKER = "broker_native_saxo_history"
SOURCE_INTERNAL = "internal_saxo_trade_log"
NOTEBOOK_SOURCE = "rt_saxo_v4_strategy_ledger"
COST_NOTES = "Saxo payload populated ConversionCost/ConversionRate; commission/fee fields not populated."

try:
    dbutils.widgets.text("initial_strategy_capital", "5000")
    dbutils.widgets.text("ledger_start_ts", "")
    dbutils.widgets.text("ledger_end_ts", "")
    INITIAL_STRATEGY_CAPITAL = float(dbutils.widgets.get("initial_strategy_capital") or "5000")
    LEDGER_START_TS = (dbutils.widgets.get("ledger_start_ts") or "").strip()
    LEDGER_END_TS = (dbutils.widgets.get("ledger_end_ts") or "").strip()
except Exception:
    INITIAL_STRATEGY_CAPITAL = 5000.0
    LEDGER_START_TS = ""
    LEDGER_END_TS = ""

if INITIAL_STRATEGY_CAPITAL <= 0:
    raise ValueError("initial_strategy_capital must be positive.")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def table_exists(table_name: str) -> bool:
    try:
        return spark.catalog.tableExists(table_name)
    except Exception:
        return False


def table_row_count(table_name: str) -> int:
    if not table_exists(table_name):
        return 0
    try:
        return int(spark.sql(f"SELECT COUNT(*) AS n FROM {table_name}").collect()[0]["n"])
    except Exception:
        return 0


def existing_columns(table_name: str) -> set[str]:
    if not table_exists(table_name):
        return set()
    return {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {table_name}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }


def first_existing(columns: set[str], candidates: list[str]) -> Optional[str]:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def col_or_null(columns: set[str], candidates: list[str], dtype: str = "string"):
    column = first_existing(columns, candidates)
    if column:
        return F.col(column).cast(dtype)
    return F.lit(None).cast(dtype)


def sql_ts_filter(column_name: str) -> str:
    clauses = []
    if LEDGER_START_TS:
        clauses.append(f"{column_name} >= TIMESTAMP('{LEDGER_START_TS}')")
    if LEDGER_END_TS:
        clauses.append(f"{column_name} <= TIMESTAMP('{LEDGER_END_TS}')")
    return " AND ".join(clauses)


def write_gold_table(dataframe, table_name: str, operation_name: str) -> None:
    overwrite_dataframe_with_retry(
        dataframe,
        table_name,
        options={"overwriteSchema": "true"},
        operation_name=operation_name,
    )


def normalize_symbol_expr(raw_symbol):
    upper_symbol = F.upper(F.trim(raw_symbol.cast("string")))
    return (
        F.when(upper_symbol.rlike(r"(^|[^A-Z])SPY([^A-Z]|$)|SPY"), F.lit("SPY"))
        .when(upper_symbol.rlike(r"(^|[^A-Z])QQQ([^A-Z]|$)|QQQ"), F.lit("QQQ"))
        .when((upper_symbol == "") | upper_symbol.isNull(), F.lit(None).cast("string"))
        .otherwise(upper_symbol)
    )


def normalize_side_expr(raw_side):
    side = F.upper(F.trim(raw_side.cast("string")))
    return (
        F.when(side.isin("BUY", "BOUGHT", "B", "BOT"), F.lit("BUY"))
        .when(side.isin("SELL", "SOLD", "S", "SLD"), F.lit("SELL"))
        .otherwise(side)
    )


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_float = float(value)
    except Exception:
        return None
    if value_float != value_float:
        return None
    return value_float


def maybe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_json_payload(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Output schemas / table creation
# ---------------------------------------------------------------------------
def ledger_schema() -> StructType:
    return StructType(
        [
            StructField("event_ts", TimestampType(), True),
            StructField("signal_ts", TimestampType(), True),
            StructField("symbol", StringType(), True),
            StructField("side", StringType(), True),
            StructField("quantity", DoubleType(), True),
            StructField("price", DoubleType(), True),
            StructField("gross_notional", DoubleType(), True),
            StructField("commission", DoubleType(), True),
            StructField("fees", DoubleType(), True),
            StructField("fx_cost", DoubleType(), True),
            StructField("total_transaction_cost", DoubleType(), True),
            StructField("strategy_stance_before", StringType(), True),
            StructField("strategy_stance_after", StringType(), True),
            StructField("is_v4_strategy_trade", BooleanType(), True),
            StructField("ignore_reason", StringType(), True),
            StructField("source", StringType(), True),
            StructField("raw_json", StringType(), True),
            StructField("ingestion_ts", TimestampType(), True),
            StructField("event_action", StringType(), True),
            StructField("event_status_code", IntegerType(), True),
            StructField("cost_is_estimated", BooleanType(), True),
            StructField("cost_estimation_method", StringType(), True),
            StructField("cost_source", StringType(), True),
            StructField("trade_id", StringType(), True),
            StructField("position_id", StringType(), True),
            StructField("cost_observation_quality", StringType(), True),
            StructField("all_in_cost_confirmed", BooleanType(), True),
            StructField("cost_notes", StringType(), True),
            StructField("source_record_id", StringType(), True),
            StructField("event_sequence", IntegerType(), True),
        ]
    )


def equity_schema() -> StructType:
    return StructType(
        [
            StructField("timestamp", TimestampType(), True),
            StructField("strategy_stance", StringType(), True),
            StructField("strategy_cash", DoubleType(), True),
            StructField("strategy_position_symbol", StringType(), True),
            StructField("strategy_position_qty", DoubleType(), True),
            StructField("strategy_position_market_value", DoubleType(), True),
            StructField("strategy_equity_gross", DoubleType(), True),
            StructField("strategy_equity_net", DoubleType(), True),
            StructField("cumulative_transaction_cost", DoubleType(), True),
            StructField("strategy_return_gross", DoubleType(), True),
            StructField("strategy_return_net", DoubleType(), True),
            StructField("source", StringType(), True),
            StructField("ingestion_ts", TimestampType(), True),
        ]
    )


def tca_schema() -> StructType:
    return StructType(
        [
            StructField("account_size", DoubleType(), True),
            StructField("cost_model", StringType(), True),
            StructField("trade_count", IntegerType(), True),
            StructField("total_turnover", DoubleType(), True),
            StructField("total_cost", DoubleType(), True),
            StructField("total_cost_bps", DoubleType(), True),
            StructField("cost_as_pct_of_account", DoubleType(), True),
            StructField("breakeven_return_required", DoubleType(), True),
            StructField("notes", StringType(), True),
            StructField("source", StringType(), True),
            StructField("ingestion_ts", TimestampType(), True),
        ]
    )


def ensure_gold_tables() -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
            event_ts TIMESTAMP,
            signal_ts TIMESTAMP,
            symbol STRING,
            side STRING,
            quantity DOUBLE,
            price DOUBLE,
            gross_notional DOUBLE,
            commission DOUBLE,
            fees DOUBLE,
            fx_cost DOUBLE,
            total_transaction_cost DOUBLE,
            strategy_stance_before STRING,
            strategy_stance_after STRING,
            is_v4_strategy_trade BOOLEAN,
            ignore_reason STRING,
            source STRING,
            raw_json STRING,
            ingestion_ts TIMESTAMP,
            event_action STRING,
            event_status_code INT,
            cost_is_estimated BOOLEAN,
            cost_estimation_method STRING,
            cost_source STRING,
            trade_id STRING,
            position_id STRING,
            cost_observation_quality STRING,
            all_in_cost_confirmed BOOLEAN,
            cost_notes STRING,
            source_record_id STRING,
            event_sequence INT
        )
        USING DELTA
        """
    )
    ledger_current = existing_columns(LEDGER_TABLE)
    ledger_desired = {
        "cost_source": "STRING",
        "trade_id": "STRING",
        "position_id": "STRING",
        "cost_observation_quality": "STRING",
        "all_in_cost_confirmed": "BOOLEAN",
        "cost_notes": "STRING",
    }
    for name, dtype in ledger_desired.items():
        if name not in ledger_current:
            spark.sql(f"ALTER TABLE {LEDGER_TABLE} ADD COLUMN {name} {dtype}")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {EQUITY_TABLE} (
            timestamp TIMESTAMP,
            strategy_stance STRING,
            strategy_cash DOUBLE,
            strategy_position_symbol STRING,
            strategy_position_qty DOUBLE,
            strategy_position_market_value DOUBLE,
            strategy_equity_gross DOUBLE,
            strategy_equity_net DOUBLE,
            cumulative_transaction_cost DOUBLE,
            strategy_return_gross DOUBLE,
            strategy_return_net DOUBLE,
            source STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TCA_TABLE} (
            account_size DOUBLE,
            cost_model STRING,
            trade_count INT,
            total_turnover DOUBLE,
            total_cost DOUBLE,
            total_cost_bps DOUBLE,
            cost_as_pct_of_account DOUBLE,
            breakeven_return_required DOUBLE,
            notes STRING,
            source STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )


# ---------------------------------------------------------------------------
# Source normalization: broker-native table first, internal execution log second
# ---------------------------------------------------------------------------
def resolve_source_table() -> tuple[str, str]:
    broker_rows = table_row_count(BROKER_HISTORY_TABLE)
    if broker_rows > 0:
        return BROKER_HISTORY_TABLE, SOURCE_BROKER
    if table_row_count(INTERNAL_TRADE_LOG) > 0:
        return INTERNAL_TRADE_LOG, SOURCE_INTERNAL
    raise RuntimeError(
        "No Saxo trade source has rows. Expected broker-native history or internal Saxo trade log."
    )


def normalize_broker_history(table_name: str):
    columns = existing_columns(table_name)
    source_df = spark.table(table_name)
    raw_json = F.to_json(F.struct(*[F.col(column).alias(column) for column in source_df.columns]))

    source_name = col_or_null(columns, ["source"], "string")
    status = col_or_null(columns, ["status"], "string")
    event_ts = F.coalesce(
        col_or_null(
            columns,
            [
                "broker_event_ts",
                "event_ts",
                "trade_ts",
                "execution_ts",
                "executed_at",
                "order_ts",
                "transaction_ts",
                "timestamp",
            ],
            "timestamp",
        ),
        col_or_null(columns, ["created_at", "created_ts", "ingestion_ts"], "timestamp"),
    )
    signal_ts = col_or_null(columns, ["signal_ts"], "timestamp")
    symbol_raw = F.coalesce(
        col_or_null(columns, ["symbol", "ticker", "instrument_symbol", "instrument", "display_symbol", "asset_symbol"], "string"),
        col_or_null(columns, ["uic", "Uic"], "string"),
    )
    symbol_from_uic = F.create_map(
        *[item for pair in SAXO_UIC_TO_SYMBOL.items() for item in (F.lit(pair[0]), F.lit(pair[1]))]
    ).getItem(col_or_null(columns, ["uic", "Uic"], "string"))
    symbol = F.coalesce(symbol_from_uic, normalize_symbol_expr(symbol_raw))
    side = normalize_side_expr(
        F.coalesce(
            col_or_null(columns, ["side", "buy_sell", "BuySell", "buysell", "action", "transaction_type"], "string"),
            F.when(col_or_null(columns, ["quantity", "amount", "qty"], "double") < 0, F.lit("SELL")).otherwise(F.lit("BUY")),
        )
    )
    quantity = F.abs(
        F.coalesce(
            col_or_null(columns, ["quantity", "qty", "amount", "filled_quantity", "executed_quantity"], "double"),
            F.lit(None).cast("double"),
        )
    )
    price = F.coalesce(
        col_or_null(columns, ["price", "execution_price", "executed_price", "fill_price", "average_price", "avg_price"], "double"),
        F.lit(None).cast("double"),
    )
    gross_notional = F.abs(
        F.coalesce(
            col_or_null(columns, ["gross_notional", "notional", "trade_value", "amount_value", "value"], "double"),
            quantity * price,
        )
    )
    commission = F.abs(F.coalesce(col_or_null(columns, ["commission", "commissions"], "double"), F.lit(0.0)))
    fees = F.abs(F.coalesce(col_or_null(columns, ["fees", "fee", "exchange_fee", "sec_fee", "clearing_fee"], "double"), F.lit(0.0)))
    fx_cost = F.abs(F.coalesce(col_or_null(columns, ["fx_cost", "fx_fee", "currency_conversion_cost"], "double"), F.lit(0.0)))
    explicit_total_cost = F.abs(
        F.coalesce(
            col_or_null(columns, ["total_transaction_cost", "transaction_cost", "total_cost", "cost"], "double"),
            F.lit(None).cast("double"),
        )
    )
    exact_component_cost = commission + fees + fx_cost
    cash_effect = col_or_null(
        columns,
        ["cash_amount", "cash_delta", "net_cash_amount", "settlement_amount", "net_amount", "cash_movement"],
        "double",
    )
    estimated_cost_from_cash = F.when(
        cash_effect.isNotNull() & gross_notional.isNotNull(),
        F.abs(F.abs(cash_effect) - gross_notional),
    )
    row_total_transaction_cost = F.coalesce(
        explicit_total_cost,
        F.when(exact_component_cost > 0, exact_component_cost),
        estimated_cost_from_cash,
        F.lit(0.0),
    )
    cost_is_estimated = explicit_total_cost.isNull() & (exact_component_cost == 0) & (estimated_cost_from_cash > 0)
    cost_estimation_method = F.when(
        cost_is_estimated,
        F.lit("cash_or_settlement_movement_minus_modeled_notional"),
    )

    cost_candidates = (
        source_df.select(
            F.coalesce(col_or_null(columns, ["trade_id", "TradeId", "TradeID", "ExecutionId", "ExecutionID"], "string"), F.lit("")).alias("cost_trade_id"),
            F.coalesce(
                col_or_null(columns, ["position_id", "PositionId", "ClosedPositionId", "PositionBase.PositionId"], "string"),
                F.lit(""),
            ).alias("cost_position_id"),
            row_total_transaction_cost.alias("matched_transaction_cost"),
            cost_is_estimated.alias("matched_cost_is_estimated"),
            cost_estimation_method.alias("matched_cost_estimation_method"),
        )
        .where(F.col("matched_transaction_cost") > 0)
    )
    cost_by_trade = (
        cost_candidates.where(F.col("cost_trade_id") != "")
        .groupBy("cost_trade_id")
        .agg(
            F.sum("matched_transaction_cost").alias("trade_matched_transaction_cost"),
            F.max(F.col("matched_cost_is_estimated").cast("int")).cast("boolean").alias("trade_matched_cost_is_estimated"),
            F.first("matched_cost_estimation_method", ignorenulls=True).alias("trade_matched_cost_estimation_method"),
        )
    )
    cost_by_position = (
        cost_candidates.where(F.col("cost_position_id") != "")
        .groupBy("cost_position_id")
        .agg(
            F.sum("matched_transaction_cost").alias("position_matched_transaction_cost"),
            F.max(F.col("matched_cost_is_estimated").cast("int")).cast("boolean").alias("position_matched_cost_is_estimated"),
            F.first("matched_cost_estimation_method", ignorenulls=True).alias("position_matched_cost_estimation_method"),
        )
    )

    base = source_df.select(
            event_ts.alias("event_ts"),
            signal_ts.alias("signal_ts"),
            symbol.alias("symbol"),
            side.alias("side"),
            quantity.alias("quantity"),
            price.alias("price"),
            gross_notional.alias("gross_notional"),
            commission.alias("commission"),
            fees.alias("fees"),
            fx_cost.alias("fx_cost"),
            row_total_transaction_cost.alias("row_total_transaction_cost"),
            cost_is_estimated.alias("parsed_cost_is_estimated"),
            cost_estimation_method.alias("parsed_cost_estimation_method"),
            F.lit(SOURCE_BROKER).alias("source"),
            raw_json.alias("raw_json"),
            source_name.alias("event_action"),
            status.alias("broker_status"),
            F.lit(None).cast("int").alias("event_status_code"),
            F.sha2(raw_json, 256).alias("source_record_id"),
            col_or_null(columns, ["trade_id", "TradeId", "TradeID", "ExecutionId", "ExecutionID"], "string").alias("trade_id"),
            col_or_null(columns, ["position_id", "PositionId", "ClosedPositionId", "PositionBase.PositionId"], "string").alias("position_id"),
        )

    normalized = (
        base.join(
            cost_by_trade,
            F.coalesce(F.col("trade_id"), F.lit("")) == F.col("cost_trade_id"),
            "left",
        )
        .join(
            cost_by_position,
            F.coalesce(F.col("position_id"), F.lit("")) == F.col("cost_position_id"),
            "left",
        )
        .withColumn(
            "parsed_total_transaction_cost",
            F.when(F.col("row_total_transaction_cost") > 0, F.col("row_total_transaction_cost")).otherwise(
                F.coalesce(F.col("trade_matched_transaction_cost"), F.col("position_matched_transaction_cost"), F.lit(0.0))
            ),
        )
        .withColumn(
            "parsed_cost_is_estimated",
            F.coalesce(
                F.col("parsed_cost_is_estimated"),
                F.col("trade_matched_cost_is_estimated"),
                F.col("position_matched_cost_is_estimated"),
                F.lit(False),
            ),
        )
        .withColumn(
            "parsed_cost_estimation_method",
            F.coalesce(
                F.col("parsed_cost_estimation_method"),
                F.col("trade_matched_cost_estimation_method"),
                F.col("position_matched_cost_estimation_method"),
            ),
        )
        .withColumn(
            "parsed_cost_source",
            F.when(
                (F.col("row_total_transaction_cost") > 0) & (F.col("event_action") == "hist_transactions"),
                F.lit("hist_transactions_conversion_cost_row"),
            )
            .when(F.col("trade_matched_transaction_cost") > 0, F.lit("hist_transactions_by_trade_id"))
            .when(F.col("position_matched_transaction_cost") > 0, F.lit("hist_transactions_by_position_id"))
            .otherwise(F.lit(None).cast("string")),
        )
        .withColumn("event_sequence", F.row_number().over(Window.orderBy("event_ts", "source_record_id")))
        .drop(
            "row_total_transaction_cost",
            "broker_status",
            "cost_trade_id",
            "cost_position_id",
            "trade_matched_transaction_cost",
            "trade_matched_cost_is_estimated",
            "trade_matched_cost_estimation_method",
            "position_matched_transaction_cost",
            "position_matched_cost_is_estimated",
            "position_matched_cost_estimation_method",
        )
    )
    if LEDGER_START_TS or LEDGER_END_TS:
        date_filter = sql_ts_filter("event_ts")
        normalized = normalized.where(date_filter)
    return normalized


internal_event_schema = ArrayType(
    StructType(
        [
            StructField("event_ts", TimestampType(), True),
            StructField("signal_ts", TimestampType(), True),
            StructField("symbol", StringType(), True),
            StructField("side", StringType(), True),
            StructField("quantity", DoubleType(), True),
            StructField("price", DoubleType(), True),
            StructField("gross_notional", DoubleType(), True),
            StructField("commission", DoubleType(), True),
            StructField("fees", DoubleType(), True),
            StructField("fx_cost", DoubleType(), True),
            StructField("parsed_total_transaction_cost", DoubleType(), True),
            StructField("parsed_cost_is_estimated", BooleanType(), True),
            StructField("parsed_cost_estimation_method", StringType(), True),
            StructField("parsed_cost_source", StringType(), True),
            StructField("source", StringType(), True),
            StructField("raw_json", StringType(), True),
            StructField("event_action", StringType(), True),
            StructField("event_status_code", IntegerType(), True),
            StructField("trade_id", StringType(), True),
            StructField("position_id", StringType(), True),
            StructField("source_record_id", StringType(), True),
            StructField("event_sequence", IntegerType(), True),
        ]
    )
)


def _extract_internal_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row is None:
        row = {}
    elif not isinstance(row, dict):
        try:
            row = row.asDict(recursive=True)
        except Exception:
            row = dict(row)

    trade_ts = row.get("trade_ts")
    signal_ts = row.get("signal_ts")
    source_record_id = row.get("source_record_id")
    target_ticker = maybe_str(row.get("target_ticker"))
    transaction_cost = maybe_float(row.get("transaction_cost"))
    spy_close = maybe_float(row.get("spy_close"))
    qqq_close = maybe_float(row.get("qqq_close"))

    positions = parse_json_payload(row.get("positions_json"))
    v4_position_symbol = None
    v4_position_qty = None
    v4_position_market_value = None
    if isinstance(positions, list):
        for position in positions:
            if not isinstance(position, dict):
                continue
            raw_symbol = maybe_str(position.get("symbol"))
            symbol = _normalize_symbol_py(raw_symbol, position.get("uic"))
            if symbol not in V4_SYMBOLS:
                continue
            qty = maybe_float(position.get("qty"))
            market_value = maybe_float(position.get("market_value"))
            if qty is None or qty == 0:
                continue
            if v4_position_market_value is None or abs(market_value or 0.0) > abs(v4_position_market_value or 0.0):
                v4_position_symbol = symbol
                v4_position_qty = qty
                v4_position_market_value = market_value

    orders = parse_json_payload(row.get("orders_placed"))
    if isinstance(orders, dict):
        if isinstance(orders.get("orders_placed"), list):
            order_events = [event for event in orders["orders_placed"] if isinstance(event, dict)]
        else:
            order_events = [orders]
    elif isinstance(orders, list):
        order_events = [event for event in orders if isinstance(event, dict)]
    else:
        order_events = []

    if not order_events:
        order_events = [{"action": "no_order_snapshot", "message": "No parsed orders_placed events"}]

    output = []
    for event_sequence, event in enumerate(order_events, start=1):
        action = (maybe_str(event.get("action")) or "unknown").lower()
        symbol = _normalize_symbol_py(
            event.get("symbol")
            or event.get("InstrumentSymbol")
            or event.get("DisplaySymbol")
            or event.get("asset_symbol"),
            event.get("uic") or event.get("Uic"),
        )
        side = maybe_str(event.get("side") or event.get("BuySell") or event.get("buy_sell"))
        if side is None and action == "buy":
            side = "BUY"
        elif side is None and action in {"sell", "close"}:
            side = "SELL"
        side = side.upper() if side else None

        quantity = maybe_float(
            event.get("quantity")
            or event.get("qty")
            or event.get("amount")
            or event.get("Amount")
            or event.get("filled_quantity")
        )
        if quantity is not None:
            quantity = abs(quantity)

        price = maybe_float(
            event.get("price")
            or event.get("execution_price")
            or event.get("fill_price")
            or event.get("avg_price")
        )
        if price is None and symbol == "SPY":
            price = spy_close
        elif price is None and symbol == "QQQ":
            price = qqq_close

        gross_notional = maybe_float(event.get("gross_notional") or event.get("notional"))
        if gross_notional is None and quantity is not None and price is not None:
            gross_notional = abs(quantity * price)

        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        estimated_cash_required = maybe_float(
            event.get("estimated_cash_required") or payload.get("EstimatedCashRequired")
        )
        parsed_cost = None
        cost_is_estimated = False
        cost_method = None
        if transaction_cost is not None and transaction_cost > 0:
            parsed_cost = transaction_cost
            cost_method = "internal_trade_log_transaction_cost"
        elif (
            estimated_cash_required is not None
            and gross_notional is not None
            and side == "BUY"
            and estimated_cash_required > gross_notional
        ):
            parsed_cost = estimated_cash_required - gross_notional
            cost_is_estimated = True
            cost_method = "estimated_cash_required_minus_modeled_notional"

        status_code = None
        try:
            if event.get("status_code") is not None:
                status_code = int(event.get("status_code"))
        except Exception:
            status_code = None

        raw_event = {
            "trade_log": {
                "trade_ts": str(trade_ts) if trade_ts is not None else None,
                "signal_ts": str(signal_ts) if signal_ts is not None else None,
                "target_position": row.get("target_position"),
                "target_ticker": target_ticker,
                "switched": row.get("switched"),
                "execution_mode": row.get("execution_mode"),
                "source": row.get("row_source"),
            },
            "order_event": event,
            "v4_position_after": {
                "symbol": v4_position_symbol,
                "qty": v4_position_qty,
                "market_value": v4_position_market_value,
            },
        }

        output.append(
            {
                "event_ts": trade_ts,
                "signal_ts": signal_ts,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "gross_notional": gross_notional,
                "commission": None,
                "fees": None,
                "fx_cost": None,
                "parsed_total_transaction_cost": parsed_cost,
                "parsed_cost_is_estimated": cost_is_estimated,
                "parsed_cost_estimation_method": cost_method,
                "parsed_cost_source": "internal_trade_log_transaction_cost" if parsed_cost is not None else None,
                "source": SOURCE_INTERNAL,
                "raw_json": json.dumps(raw_event, default=str),
                "event_action": action,
                "event_status_code": status_code,
                "trade_id": maybe_str(event.get("trade_id") or event.get("TradeId")),
                "position_id": maybe_str(event.get("position_id") or event.get("PositionId")),
                "source_record_id": source_record_id,
                "event_sequence": event_sequence,
            }
        )
    return output


def _normalize_symbol_py(raw_symbol: Any, raw_uic: Any = None) -> Optional[str]:
    uic = maybe_str(raw_uic)
    if uic and uic in SAXO_UIC_TO_SYMBOL:
        return SAXO_UIC_TO_SYMBOL[uic]
    text = maybe_str(raw_symbol)
    if text is None:
        return None
    upper_text = text.upper()
    if "SPY" in upper_text:
        return "SPY"
    if "QQQ" in upper_text:
        return "QQQ"
    return upper_text


extract_internal_events_udf = F.udf(_extract_internal_events, internal_event_schema)


def normalize_internal_trade_log(table_name: str):
    columns = existing_columns(table_name)
    source_df = spark.table(table_name)
    wanted = [
        "trade_ts",
        "signal_ts",
        "target_position",
        "target_ticker",
        "switched",
        "transaction_cost",
        "orders_placed",
        "positions_json",
        "spy_close",
        "qqq_close",
        "execution_mode",
        "source",
        "ingestion_ts",
    ]
    selected = []
    for column in wanted:
        if column in columns:
            selected.append(F.col(column).alias(column))
        else:
            selected.append(F.lit(None).alias(column))

    base = source_df.select(*selected)
    if LEDGER_START_TS or LEDGER_END_TS:
        date_filter = sql_ts_filter("trade_ts")
        base = base.where(date_filter)

    base = base.withColumn(
        "source_record_id",
        F.sha2(
            F.to_json(
                F.struct(
                    F.col("trade_ts").cast("string"),
                    F.col("signal_ts").cast("string"),
                    F.col("target_ticker").cast("string"),
                    F.col("orders_placed").cast("string"),
                    F.col("positions_json").cast("string"),
                    F.col("ingestion_ts").cast("string"),
                )
            ),
            256,
        ),
    ).withColumnRenamed("source", "row_source")

    parsed = (
        base.withColumn("events", extract_internal_events_udf(F.struct(*[F.col(column) for column in base.columns])))
        .select(F.explode_outer("events").alias("event"))
        .select("event.*")
    )
    return parsed


def build_base_normalized_events(source_table: str, source_type: str):
    if source_type == SOURCE_BROKER:
        return normalize_broker_history(source_table)
    return normalize_internal_trade_log(source_table)


# ---------------------------------------------------------------------------
# Ledger assembly
# ---------------------------------------------------------------------------
def build_strategy_ledger(base_events):
    if table_exists(MARKET_TABLE):
        market_prices = (
            spark.table(MARKET_TABLE)
            .where(F.col("symbol").isin(V4_SYMBOLS))
            .select(
                F.col("symbol").alias("market_symbol"),
                F.date_trunc("hour", F.col("timestamp")).alias("market_hour"),
                F.col("close").cast("double").alias("market_close"),
            )
            .groupBy("market_symbol", "market_hour")
            .agg(F.max("market_close").alias("market_close"))
        )
    else:
        market_prices = spark.createDataFrame(
            [],
            StructType(
                [
                    StructField("market_symbol", StringType(), True),
                    StructField("market_hour", TimestampType(), True),
                    StructField("market_close", DoubleType(), True),
                ]
            ),
        )

    events = (
        base_events.withColumn("symbol", normalize_symbol_expr(F.col("symbol")))
        .withColumn("side", normalize_side_expr(F.col("side")))
        .join(
            market_prices,
            (F.col("symbol") == F.col("market_symbol"))
            & (F.date_trunc("hour", F.col("event_ts")) == F.col("market_hour")),
            "left",
        )
        .drop("market_symbol", "market_hour")
        .withColumn("price", F.coalesce(F.col("price"), F.col("market_close")))
        .withColumn(
            "gross_notional",
            F.abs(F.coalesce(F.col("gross_notional"), F.col("quantity") * F.col("price"))),
        )
        .withColumn("commission", F.abs(F.coalesce(F.col("commission"), F.lit(0.0))))
        .withColumn("fees", F.abs(F.coalesce(F.col("fees"), F.lit(0.0))))
        .withColumn("fx_cost", F.abs(F.coalesce(F.col("fx_cost"), F.lit(0.0))))
        .withColumn(
            "candidate_total_transaction_cost",
            F.coalesce(
                F.col("parsed_total_transaction_cost"),
                F.col("commission") + F.col("fees") + F.col("fx_cost"),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "order_success",
            F.coalesce(
                (F.col("source") == SOURCE_BROKER)
                | F.col("event_status_code").between(200, 299),
                F.lit(False),
            ),
        )
        .withColumn(
            "candidate_v4_trade",
            F.col("symbol").isin(V4_SYMBOLS)
            & F.col("side").isin("BUY", "SELL")
            & (F.col("quantity") > 0)
            & F.when(
                F.col("source") == SOURCE_BROKER,
                (F.col("event_action") == "cs_reports_trades") & F.col("gross_notional").isNotNull(),
            ).otherwise(F.lit(True))
            & F.col("order_success"),
        )
    )

    source_window = Window.partitionBy("source_record_id")
    events = (
        events.withColumn(
            "v4_trade_events_in_source_record",
            F.sum(F.when(F.col("candidate_v4_trade"), F.lit(1)).otherwise(F.lit(0))).over(source_window),
        )
        .withColumn(
            "total_transaction_cost",
            F.when(
                (F.col("source") == SOURCE_INTERNAL)
                & F.col("candidate_v4_trade")
                & (F.col("parsed_total_transaction_cost") > 0)
                & (F.col("v4_trade_events_in_source_record") > 1),
                F.col("parsed_total_transaction_cost") / F.col("v4_trade_events_in_source_record"),
            ).otherwise(F.col("candidate_total_transaction_cost")),
        )
        .withColumn(
            "cost_is_estimated",
            F.coalesce(F.col("parsed_cost_is_estimated"), F.lit(False)),
        )
        .withColumn(
            "cost_estimation_method",
            F.when(
                F.col("candidate_v4_trade")
                & (F.col("source") == SOURCE_INTERNAL)
                & (F.col("total_transaction_cost") == 0),
                F.lit("exact_broker_cost_unavailable_not_inferred_from_full_account"),
            ).otherwise(F.col("parsed_cost_estimation_method")),
        )
        .withColumn(
            "cost_source",
            F.when(
                (F.col("source") == SOURCE_BROKER) & (F.col("total_transaction_cost") > 0),
                F.coalesce(F.col("parsed_cost_source"), F.lit("broker_reported_conversion_cost")),
            )
            .when(F.col("source") == SOURCE_INTERNAL, F.col("parsed_cost_source"))
            .otherwise(F.lit(None).cast("string")),
        )
        .withColumn(
            "cost_observation_quality",
            F.when(
                (F.col("source") == SOURCE_BROKER) & (F.col("total_transaction_cost") > 0),
                F.lit("broker_confirmed_conversion_cost"),
            )
            .when(F.col("source") == SOURCE_BROKER, F.lit("broker_fill_no_conversion_cost_observed"))
            .when(F.col("cost_is_estimated"), F.lit("estimated"))
            .otherwise(F.lit("internal_or_unconfirmed_cost")),
        )
        .withColumn("all_in_cost_confirmed", F.lit(False))
        .withColumn("cost_notes", F.lit(COST_NOTES))
        .withColumn(
            "is_v4_strategy_trade",
            F.col("candidate_v4_trade"),
        )
        .withColumn(
            "ignore_reason",
            F.when(F.col("is_v4_strategy_trade"), F.lit(None).cast("string"))
            .when(F.col("event_action").isin("skip", "shadow", "defer", "snapshot_defer", "no_order_snapshot"), F.lit("non_trade_action"))
            .when(~F.col("order_success"), F.lit("order_not_successful_or_unconfirmed"))
            .when((F.col("source") == SOURCE_BROKER) & (F.col("event_action") != "cs_reports_trades"), F.lit("broker_context_not_trade_report_fill"))
            .when(F.col("symbol").isNull(), F.lit("missing_symbol"))
            .when(~F.col("symbol").isin(V4_SYMBOLS), F.lit("non_v4_instrument"))
            .when(F.col("side").isNull() | ~F.col("side").isin("BUY", "SELL"), F.lit("missing_or_unsupported_side"))
            .when(F.col("quantity").isNull() | (F.col("quantity") <= 0), F.lit("missing_or_zero_quantity"))
            .otherwise(F.lit("not_v4_strategy_trade")),
        )
        .withColumn(
            "stance_after_event",
            F.when(F.col("is_v4_strategy_trade") & (F.col("side") == "BUY"), F.col("symbol"))
            .when(F.col("is_v4_strategy_trade") & (F.col("side") == "SELL"), F.lit("CASH"))
            .otherwise(F.lit(None).cast("string")),
        )
    )

    order_window = Window.orderBy(
        F.col("event_ts").asc_nulls_last(),
        F.col("source_record_id").asc_nulls_last(),
        F.col("event_sequence").asc_nulls_last(),
    )
    before_window = order_window.rowsBetween(Window.unboundedPreceding, -1)

    ledger = (
        events.withColumn(
            "strategy_stance_before",
            F.coalesce(F.last("stance_after_event", ignorenulls=True).over(before_window), F.lit("CASH")),
        )
        .withColumn(
            "strategy_stance_after",
            F.when(F.col("is_v4_strategy_trade"), F.col("stance_after_event")).otherwise(F.col("strategy_stance_before")),
        )
        .withColumn("ingestion_ts", F.current_timestamp())
        .select(
            "event_ts",
            "signal_ts",
            "symbol",
            "side",
            "quantity",
            "price",
            "gross_notional",
            "commission",
            "fees",
            "fx_cost",
            "total_transaction_cost",
            "strategy_stance_before",
            "strategy_stance_after",
            "is_v4_strategy_trade",
            "ignore_reason",
            "source",
            "raw_json",
            "ingestion_ts",
            "event_action",
            "event_status_code",
            "cost_is_estimated",
            "cost_estimation_method",
            "cost_source",
            "trade_id",
            "position_id",
            "cost_observation_quality",
            "all_in_cost_confirmed",
            "cost_notes",
            "source_record_id",
            "event_sequence",
        )
    )
    return spark.createDataFrame(ledger.rdd, schema=ledger_schema())


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------
def build_empty_equity_curve():
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    row = {
        "timestamp": now_utc,
        "strategy_stance": "CASH",
        "strategy_cash": INITIAL_STRATEGY_CAPITAL,
        "strategy_position_symbol": None,
        "strategy_position_qty": 0.0,
        "strategy_position_market_value": 0.0,
        "strategy_equity_gross": INITIAL_STRATEGY_CAPITAL,
        "strategy_equity_net": INITIAL_STRATEGY_CAPITAL,
        "cumulative_transaction_cost": 0.0,
        "strategy_return_gross": 0.0,
        "strategy_return_net": 0.0,
        "source": NOTEBOOK_SOURCE,
        "ingestion_ts": now_utc,
    }
    return spark.createDataFrame([row], schema=equity_schema())


def build_equity_curve(ledger_df):
    v4_events = (
        ledger_df.where(F.col("is_v4_strategy_trade"))
        .where(F.col("event_ts").isNotNull())
        .withColumn("quantity", F.coalesce(F.col("quantity"), F.lit(0.0)))
        .withColumn("price", F.coalesce(F.col("price"), F.lit(0.0)))
        .withColumn("gross_notional", F.coalesce(F.col("gross_notional"), F.abs(F.col("quantity") * F.col("price"))))
        .withColumn("total_transaction_cost", F.coalesce(F.col("total_transaction_cost"), F.lit(0.0)))
        .withColumn(
            "event_order",
            F.row_number().over(
                Window.orderBy("event_ts", "source_record_id", "event_sequence")
            ),
        )
    )

    if v4_events.limit(1).count() == 0:
        return build_empty_equity_curve()

    impacts = (
        v4_events.withColumn("buy_sell_sign", F.when(F.col("side") == "BUY", F.lit(1.0)).otherwise(F.lit(-1.0)))
        .withColumn("spy_qty_delta", F.when(F.col("symbol") == "SPY", F.col("buy_sell_sign") * F.col("quantity")).otherwise(F.lit(0.0)))
        .withColumn("qqq_qty_delta", F.when(F.col("symbol") == "QQQ", F.col("buy_sell_sign") * F.col("quantity")).otherwise(F.lit(0.0)))
        .withColumn(
            "cash_delta_gross",
            F.when(F.col("side") == "BUY", -F.col("gross_notional")).otherwise(F.col("gross_notional")),
        )
        .withColumn(
            "cash_delta_net",
            F.when(F.col("side") == "BUY", -(F.col("gross_notional") + F.col("total_transaction_cost")))
            .otherwise(F.col("gross_notional") - F.col("total_transaction_cost")),
        )
    )
    event_window = Window.orderBy("event_ts", "event_order").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    state = (
        impacts.withColumn("strategy_cash_gross", F.lit(INITIAL_STRATEGY_CAPITAL) + F.sum("cash_delta_gross").over(event_window))
        .withColumn("strategy_cash_net", F.lit(INITIAL_STRATEGY_CAPITAL) + F.sum("cash_delta_net").over(event_window))
        .withColumn("spy_qty", F.sum("spy_qty_delta").over(event_window))
        .withColumn("qqq_qty", F.sum("qqq_qty_delta").over(event_window))
        .withColumn("cumulative_transaction_cost", F.sum("total_transaction_cost").over(event_window))
        .withColumn("last_spy_trade_price", F.last(F.when(F.col("symbol") == "SPY", F.col("price")), ignorenulls=True).over(event_window))
        .withColumn("last_qqq_trade_price", F.last(F.when(F.col("symbol") == "QQQ", F.col("price")), ignorenulls=True).over(event_window))
        .select(
            F.col("event_ts").alias("state_ts"),
            "event_order",
            "strategy_cash_gross",
            "strategy_cash_net",
            "spy_qty",
            "qqq_qty",
            "cumulative_transaction_cost",
            "strategy_stance_after",
            "last_spy_trade_price",
            "last_qqq_trade_price",
        )
    )

    min_event_ts = v4_events.agg(F.min("event_ts").alias("min_ts")).collect()[0]["min_ts"]
    state.createOrReplaceTempView("_saxo_v4_event_state")

    if table_exists(MARKET_TABLE):
        market_pivot = (
            spark.table(MARKET_TABLE)
            .where(F.col("symbol").isin(V4_SYMBOLS))
            .where(F.col("timestamp") >= F.lit(min_event_ts))
            .groupBy("timestamp")
            .agg(
                F.max(F.when(F.col("symbol") == "SPY", F.col("close").cast("double"))).alias("spy_price"),
                F.max(F.when(F.col("symbol") == "QQQ", F.col("close").cast("double"))).alias("qqq_price"),
            )
        )
    else:
        market_pivot = spark.createDataFrame(
            [],
            StructType(
                [
                    StructField("timestamp", TimestampType(), True),
                    StructField("spy_price", DoubleType(), True),
                    StructField("qqq_price", DoubleType(), True),
                ]
            ),
        )
    market_pivot.createOrReplaceTempView("_saxo_v4_market_pivot")

    timeline = (
        market_pivot.select("timestamp")
        .union(state.select(F.col("state_ts").alias("timestamp")))
        .distinct()
    )
    timeline.createOrReplaceTempView("_saxo_v4_timeline")

    asof_state = spark.sql(
        """
        SELECT *
        FROM (
            SELECT
                t.timestamp,
                s.*,
                ROW_NUMBER() OVER (
                    PARTITION BY t.timestamp
                    ORDER BY s.state_ts DESC NULLS LAST, s.event_order DESC NULLS LAST
                ) AS rn
            FROM _saxo_v4_timeline t
            LEFT JOIN _saxo_v4_event_state s
              ON s.state_ts <= t.timestamp
        )
        WHERE rn = 1
        """
    )

    asof = (
        asof_state.join(market_pivot, "timestamp", "left")
        .withColumn(
            "spy_price_filled",
            F.coalesce(
                F.last("spy_price", ignorenulls=True).over(Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)),
                F.col("last_spy_trade_price"),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "qqq_price_filled",
            F.coalesce(
                F.last("qqq_price", ignorenulls=True).over(Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)),
                F.col("last_qqq_trade_price"),
                F.lit(0.0),
            ),
        )
        .withColumn("spy_market_value", F.coalesce(F.col("spy_qty"), F.lit(0.0)) * F.col("spy_price_filled"))
        .withColumn("qqq_market_value", F.coalesce(F.col("qqq_qty"), F.lit(0.0)) * F.col("qqq_price_filled"))
        .withColumn(
            "strategy_position_symbol",
            F.when(F.abs(F.col("qqq_market_value")) > F.abs(F.col("spy_market_value")), F.lit("QQQ"))
            .when(F.abs(F.col("spy_market_value")) > 0, F.lit("SPY"))
            .otherwise(F.lit(None).cast("string")),
        )
        .withColumn(
            "strategy_position_qty",
            F.when(F.col("strategy_position_symbol") == "QQQ", F.col("qqq_qty"))
            .when(F.col("strategy_position_symbol") == "SPY", F.col("spy_qty"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "strategy_position_market_value",
            F.when(F.col("strategy_position_symbol") == "QQQ", F.col("qqq_market_value"))
            .when(F.col("strategy_position_symbol") == "SPY", F.col("spy_market_value"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "strategy_stance",
            F.coalesce(F.col("strategy_position_symbol"), F.lit("CASH")),
        )
        .withColumn("strategy_equity_gross", F.col("strategy_cash_gross") + F.col("spy_market_value") + F.col("qqq_market_value"))
        .withColumn("strategy_equity_net", F.col("strategy_cash_net") + F.col("spy_market_value") + F.col("qqq_market_value"))
        .withColumn("strategy_return_gross", F.col("strategy_equity_gross") / F.lit(INITIAL_STRATEGY_CAPITAL) - F.lit(1.0))
        .withColumn("strategy_return_net", F.col("strategy_equity_net") / F.lit(INITIAL_STRATEGY_CAPITAL) - F.lit(1.0))
        .withColumn("source", F.lit(NOTEBOOK_SOURCE))
        .withColumn("ingestion_ts", F.current_timestamp())
        .select(
            "timestamp",
            "strategy_stance",
            F.col("strategy_cash_net").alias("strategy_cash"),
            "strategy_position_symbol",
            "strategy_position_qty",
            "strategy_position_market_value",
            "strategy_equity_gross",
            "strategy_equity_net",
            "cumulative_transaction_cost",
            "strategy_return_gross",
            "strategy_return_net",
            "source",
            "ingestion_ts",
        )
        .orderBy("timestamp")
    )
    return spark.createDataFrame(asof.rdd, schema=equity_schema())


# ---------------------------------------------------------------------------
# Transaction-cost analysis
# ---------------------------------------------------------------------------
def build_transaction_cost_analysis(ledger_df):
    v4_costs = (
        ledger_df.where(F.col("is_v4_strategy_trade"))
        .select(
            F.coalesce(F.col("gross_notional"), F.abs(F.col("quantity") * F.col("price")), F.lit(0.0)).alias("gross_notional"),
            F.coalesce(F.col("total_transaction_cost"), F.lit(0.0)).alias("total_transaction_cost"),
            F.coalesce(F.col("cost_is_estimated"), F.lit(False)).alias("cost_is_estimated"),
            F.col("cost_estimation_method"),
        )
    )
    summary = v4_costs.agg(
        F.count("*").alias("trade_count"),
        F.sum("gross_notional").alias("total_turnover"),
        F.sum("total_transaction_cost").alias("observed_cost"),
        F.sum(F.when(F.col("cost_is_estimated"), F.lit(1)).otherwise(F.lit(0))).alias("estimated_cost_rows"),
        F.collect_set("cost_estimation_method").alias("cost_methods"),
    ).collect()[0]

    trade_count = int(summary["trade_count"] or 0)
    base_turnover = float(summary["total_turnover"] or 0.0)
    observed_cost = float(summary["observed_cost"] or 0.0)
    estimated_cost_rows = int(summary["estimated_cost_rows"] or 0)
    cost_methods = [method for method in (summary["cost_methods"] or []) if method]
    observed_cost_rate = observed_cost / base_turnover if base_turnover > 0 else 0.0

    rows = []
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    for account_size in [5000.0, 25000.0, 100000.0, 500000.0]:
        scale = account_size / INITIAL_STRATEGY_CAPITAL
        scaled_turnover = base_turnover * scale
        models = {
            "actual_observed_cost": scaled_turnover * observed_cost_rate,
            "fixed_25_usd": 25.0 * trade_count,
            "5_bps": scaled_turnover * 0.0005,
            "10_bps": scaled_turnover * 0.0010,
        }
        for cost_model, total_cost in models.items():
            total_cost_bps = (total_cost / scaled_turnover * 10000.0) if scaled_turnover > 0 else 0.0
            cost_as_pct = (total_cost / account_size) if account_size > 0 else None
            if cost_model == "actual_observed_cost":
                if observed_cost == 0:
                    note = "No captured broker-reported conversion cost observed; full Saxo account cash/equity was not used because unrelated account trades may exist."
                elif estimated_cost_rows > 0:
                    note = "Includes estimated cost rows: " + ", ".join(cost_methods)
                else:
                    note = "Uses captured broker-reported conversion cost only; all-in commission/fee/slippage cost is not confirmed."
            elif cost_model == "fixed_25_usd":
                note = "Sensitivity replay: $25 per V4 strategy trade event."
            else:
                note = f"Sensitivity replay: {cost_model.replace('_', ' ')} of V4 gross turnover."
            rows.append(
                {
                    "account_size": account_size,
                    "cost_model": cost_model,
                    "trade_count": trade_count,
                    "total_turnover": scaled_turnover,
                    "total_cost": total_cost,
                    "total_cost_bps": total_cost_bps,
                    "cost_as_pct_of_account": cost_as_pct,
                    "breakeven_return_required": cost_as_pct,
                    "notes": note,
                    "source": NOTEBOOK_SOURCE,
                    "ingestion_ts": now_utc,
                }
            )
    return spark.createDataFrame(rows, schema=tca_schema())


# ---------------------------------------------------------------------------
# Validation displays
# ---------------------------------------------------------------------------
def display_validation_outputs(source_table: str, source_type: str) -> None:
    print("Validation summary")
    print(f"Input source selected: {source_table} ({source_type})")
    print(f"Initial V4 strategy capital: {INITIAL_STRATEGY_CAPITAL:,.2f}")
    print("Production safety: no Saxo API/order/token code is called by this notebook.")
    print("Production safety: no Databricks workflow schedule APIs are called by this notebook.")
    print("Production safety: existing execution notebook 01_Processing/03_Realtime/08_RT_Saxo_Trade_Execution.py is not imported or modified.")
    print(
        "Cost framing: total_transaction_cost is captured broker-reported conversion cost "
        "where Saxo populated ConversionCost/ConversionRate. It is not confirmed all-in "
        "transaction cost because commission/fee/slippage fields were not populated."
    )

    print("0A. V4 fill-level ledger")
    display(
        spark.sql(
            f"""
            SELECT
                event_ts AS broker_event_ts,
                symbol,
                side,
                quantity,
                price,
                gross_notional,
                total_transaction_cost,
                cost_source,
                trade_id,
                position_id,
                cost_observation_quality,
                all_in_cost_confirmed,
                cost_notes
            FROM {LEDGER_TABLE}
            WHERE is_v4_strategy_trade = true
            ORDER BY event_ts DESC NULLS LAST, event_sequence DESC NULLS LAST
            """
        )
    )

    print("0B. Captured broker conversion cost by symbol and side")
    display(
        spark.sql(
            f"""
            SELECT
                symbol,
                side,
                COUNT(*) AS fills,
                SUM(gross_notional) AS gross_notional,
                SUM(total_transaction_cost) AS captured_cost,
                CASE
                    WHEN SUM(gross_notional) > 0
                    THEN SUM(total_transaction_cost) / SUM(gross_notional) * 10000.0
                END AS captured_cost_bps
            FROM {LEDGER_TABLE}
            WHERE is_v4_strategy_trade = true
            GROUP BY symbol, side
            ORDER BY symbol, side
            """
        )
    )

    print("0C. Daily / switch-level captured broker conversion cost")
    display(
        spark.sql(
            f"""
            SELECT
                CAST(event_ts AS DATE) AS trade_date,
                COUNT(*) AS fills,
                SUM(gross_notional) AS gross_notional,
                SUM(total_transaction_cost) AS captured_cost,
                CASE
                    WHEN SUM(gross_notional) > 0
                    THEN SUM(total_transaction_cost) / SUM(gross_notional) * 10000.0
                END AS captured_cost_bps
            FROM {LEDGER_TABLE}
            WHERE is_v4_strategy_trade = true
            GROUP BY CAST(event_ts AS DATE)
            ORDER BY trade_date DESC
            """
        )
    )

    print("0D. Small-account impact table")
    display(
        spark.sql(
            f"""
            WITH base AS (
                SELECT
                    COUNT(*) AS fills,
                    COALESCE(SUM(gross_notional), 0.0) AS gross_notional,
                    COALESCE(SUM(total_transaction_cost), 0.0) AS captured_cost
                FROM {LEDGER_TABLE}
                WHERE is_v4_strategy_trade = true
            ),
            account_sizes AS (
                SELECT 5000.0 AS account_size UNION ALL
                SELECT 25000.0 UNION ALL
                SELECT 100000.0 UNION ALL
                SELECT 500000.0
            )
            SELECT
                account_size,
                captured_cost / account_size AS captured_cost_as_pct_of_account,
                (25.0 * fills) / account_size AS fixed_cost_as_pct_of_account,
                (gross_notional * 0.0005) / account_size AS `5bps_cost_as_pct_of_account`,
                (gross_notional * 0.0010) / account_size AS `10bps_cost_as_pct_of_account`
            FROM account_sizes
            CROSS JOIN base
            ORDER BY account_size
            """
        )
    )

    print("0E. Cost validation checks")
    display(
        spark.sql(
            f"""
            WITH latest_reconstructed AS (
                SELECT strategy_stance_after AS reconstructed_stance, event_ts AS reconstructed_event_ts
                FROM {LEDGER_TABLE}
                WHERE is_v4_strategy_trade = true
                ORDER BY event_ts DESC NULLS LAST, event_sequence DESC NULLS LAST
                LIMIT 1
            ),
            latest_target AS (
                SELECT target_ticker, target_position, execution_mode, trade_ts
                FROM {INTERNAL_TRADE_LOG}
                ORDER BY trade_ts DESC
                LIMIT 1
            ),
            checks AS (
                SELECT
                    (SELECT reconstructed_stance FROM latest_reconstructed) AS reconstructed_stance,
                    (SELECT target_ticker FROM latest_target) AS latest_target_ticker,
                    (SELECT target_position FROM latest_target) AS latest_target_position,
                    (SELECT execution_mode FROM latest_target) AS latest_execution_mode,
                    (SELECT COUNT(*) FROM {LEDGER_TABLE} WHERE is_v4_strategy_trade = true AND cost_is_estimated = true) AS estimated_cost_rows,
                    (SELECT COUNT(*) FROM {LEDGER_TABLE} WHERE is_v4_strategy_trade = true AND event_action != 'cs_reports_trades') AS non_trade_report_strategy_rows,
                    (SELECT COUNT(*) FROM {LEDGER_TABLE} WHERE is_v4_strategy_trade = true AND cost_source NOT IN ('hist_transactions_by_trade_id', 'hist_transactions_by_position_id')) AS fills_without_hist_transaction_cost_link,
                    (SELECT COUNT(*) FROM {LEDGER_TABLE} WHERE is_v4_strategy_trade = true) AS strategy_trade_rows
            )
            SELECT
                *,
                reconstructed_stance = latest_target_ticker AS latest_stance_matches_target,
                estimated_cost_rows = 0 AS no_estimated_cost_rows,
                non_trade_report_strategy_rows = 0 AS all_strategy_trades_from_cs_reports_trades,
                fills_without_hist_transaction_cost_link = 0 AS all_costs_attach_from_hist_transactions
            FROM checks
            """
        )
    )

    print("1. Non-SPY/QQQ or non-trade Saxo events ignored with ignore_reason")
    display(
        spark.table(LEDGER_TABLE)
        .where(~F.col("is_v4_strategy_trade"))
        .select(
            "event_ts",
            "symbol",
            "side",
            "quantity",
            "event_action",
            "event_status_code",
            "ignore_reason",
            "source",
        )
        .orderBy(F.col("event_ts").desc_nulls_last(), F.col("event_sequence").desc_nulls_last())
        .limit(100)
    )

    print("2. Filtered V4 ledger symbols and stances")
    display(
        spark.table(LEDGER_TABLE)
        .where(F.col("is_v4_strategy_trade"))
        .groupBy("symbol", "side", "strategy_stance_before", "strategy_stance_after")
        .agg(F.count("*").alias("rows"))
        .orderBy("symbol", "side", "strategy_stance_after")
    )

    print("3. Latest 20 V4 stance changes")
    display(
        spark.table(LEDGER_TABLE)
        .where(F.col("is_v4_strategy_trade"))
        .where(F.col("strategy_stance_before") != F.col("strategy_stance_after"))
        .select(
            "event_ts",
            "symbol",
            "side",
            "quantity",
            "price",
            "gross_notional",
            "total_transaction_cost",
            "cost_source",
            "strategy_stance_before",
            "strategy_stance_after",
            "cost_is_estimated",
            "cost_estimation_method",
        )
        .orderBy(F.col("event_ts").desc(), F.col("event_sequence").desc())
        .limit(20)
    )

    print("4. Latest reconstructed V4 stance vs latest internal Saxo target")
    if table_exists(INTERNAL_TRADE_LOG):
        latest_target = spark.sql(
            f"""
            SELECT trade_ts, target_ticker, target_position, execution_mode
            FROM {INTERNAL_TRADE_LOG}
            ORDER BY trade_ts DESC
            LIMIT 1
            """
        )
    else:
        latest_target = spark.createDataFrame(
            [],
            StructType(
                [
                    StructField("trade_ts", TimestampType(), True),
                    StructField("target_ticker", StringType(), True),
                    StructField("target_position", StringType(), True),
                    StructField("execution_mode", StringType(), True),
                ]
            ),
        )
    latest_equity_stance = spark.sql(
        f"""
        SELECT timestamp, strategy_stance, strategy_position_symbol, strategy_position_qty, strategy_equity_net
        FROM {EQUITY_TABLE}
        ORDER BY timestamp DESC
        LIMIT 1
        """
    )
    display(latest_target.crossJoin(latest_equity_stance))

    print("5. Net vs gross strategy equity chart")
    chart_pdf = (
        spark.table(EQUITY_TABLE)
        .select("timestamp", "strategy_equity_gross", "strategy_equity_net", "cumulative_transaction_cost")
        .orderBy("timestamp")
        .limit(5000)
        .toPandas()
    )
    if not chart_pdf.empty:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(chart_pdf["timestamp"], chart_pdf["strategy_equity_gross"], label="V4 strategy equity gross", linewidth=2)
        ax.plot(chart_pdf["timestamp"], chart_pdf["strategy_equity_net"], label="V4 strategy equity net after captured conversion cost", linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(
            chart_pdf["timestamp"],
            chart_pdf["cumulative_transaction_cost"],
            label="Cumulative captured conversion cost",
            linewidth=1.75,
            linestyle="--",
            color="tab:red",
        )
        ax.set_title("V4 Saxo Strategy-Only Equity: Gross vs Net Captured Conversion Cost")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Strategy equity")
        ax2.set_ylabel("Cumulative captured conversion cost")
        ax.grid(True, alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("No equity rows available to chart.")

    print("6. Cost drag table by account size and cost model")
    display(
        spark.table(TCA_TABLE)
        .where(F.col("account_size").isin(5000.0, 25000.0, 100000.0, 500000.0))
        .orderBy("account_size", "cost_model")
    )

    print("7. Existing production execution notebook modified: NO (this run writes only new gold analysis tables).")
    print("8. Workflow schedule changed: NO (this notebook does not call Databricks Jobs APIs).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_gold_tables()
    source_table, source_type = resolve_source_table()
    base_events = build_base_normalized_events(source_table, source_type)
    ledger_df = build_strategy_ledger(base_events)
    write_gold_table(
        ledger_df,
        LEDGER_TABLE,
        f"overwrite V4 Saxo strategy trade ledger {LEDGER_TABLE}",
    )

    persisted_ledger = spark.table(LEDGER_TABLE)
    equity_df = build_equity_curve(persisted_ledger)
    write_gold_table(
        equity_df,
        EQUITY_TABLE,
        f"overwrite V4 Saxo strategy equity curve {EQUITY_TABLE}",
    )

    tca_df = build_transaction_cost_analysis(persisted_ledger)
    write_gold_table(
        tca_df,
        TCA_TABLE,
        f"overwrite V4 Saxo transaction cost analysis {TCA_TABLE}",
    )

    print(f"Wrote {LEDGER_TABLE}: {table_row_count(LEDGER_TABLE)} rows")
    print(f"Wrote {EQUITY_TABLE}: {table_row_count(EQUITY_TABLE)} rows")
    print(f"Wrote {TCA_TABLE}: {table_row_count(TCA_TABLE)} rows")
    display_validation_outputs(source_table, source_type)


main()
