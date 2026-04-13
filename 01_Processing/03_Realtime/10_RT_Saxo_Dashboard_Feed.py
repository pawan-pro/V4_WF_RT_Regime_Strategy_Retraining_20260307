# Databricks notebook source
# RT_10: Saxo dashboard feed (canonical live producer)

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config
# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

from datetime import datetime, timezone
from typing import Any, Optional

from pyspark.sql import Row
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Constants and table locations
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Section: Input normalization
# ---------------------------------------------------------------------------
def maybe_float(value: Any) -> Optional[float]:
    """Convert a nullable input into a Python float when possible."""
    if value is None:
        return None
    try:
        value_float = float(value)
    except Exception:
        return None
    if value_float != value_float:
        return None
    return value_float


def maybe_int(value: Any) -> Optional[int]:
    """Convert a nullable input into an int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def maybe_str(value: Any) -> Optional[str]:
    """Convert a value into a non-empty string or return None."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def safe_ts(row: Any, key: str) -> Optional[datetime]:
    """Extract a datetime field from a Spark Row-like object."""
    if row is None:
        return None
    value = row[key]
    if value is None:
        return None
    return value


def table_exists(table_name: str) -> bool:
    return spark.catalog.tableExists(table_name)


# ---------------------------------------------------------------------------
# Section: SQL read helpers
# ---------------------------------------------------------------------------
def latest_row(table_name: str, order_by: str, limit: int = 1, where_clause: str = "") -> Optional[Row]:
    """Fetch a single latest row from a table with safe handling when empty."""
    query = f"""
        SELECT *
        FROM {table_name}
        {where_clause}
        ORDER BY {order_by} DESC
        LIMIT {limit}
    """
    rows = spark.sql(query).collect()
    if not rows:
        return None
    if limit == 1:
        return rows[0]
    return rows


def scalar_row(sql: str) -> Optional[dict]:
    """Execute a scalar SQL query and return a single-row dict or None."""
    rows = spark.sql(sql).collect()
    if not rows:
        return None
    return rows[0].asDict(recursive=True)


def resolve_execution_mode() -> str:
    """Resolve execution mode using widget input with auto-fallback from latest live trade log."""
    try:
        dbutils.widgets.dropdown("execution_mode", "auto", ["auto", "shadow", "live"])
        requested = (dbutils.widgets.get("execution_mode") or "auto").strip().lower()
    except Exception:
        requested = "auto"

    if requested not in {"auto", "shadow", "live"}:
        raise ValueError(f"Unsupported execution_mode: {requested}")

    if requested != "auto":
        return requested

    latest_trade = latest_row(TRADE_LOG, "trade_ts")
    if latest_trade and latest_trade["execution_mode"] in {"shadow", "live"}:
        return latest_trade["execution_mode"]
    return "live"


def get_model_version(default_version: str = "bootstrap_live_model") -> str:
    """Read latest model version from registry if available."""
    if table_exists(MODEL_REGISTRY_TABLE):
        row = scalar_row(
            f"""
            SELECT model_version
            FROM {MODEL_REGISTRY_TABLE}
            ORDER BY promoted_ts DESC
            LIMIT 1
            """
        )
        if row and row.get("model_version"):
            return str(row["model_version"])
    return default_version


def existing_columns(table_name: str) -> set[str]:
    if not table_exists(table_name):
        return set()
    return {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {table_name}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }


def ensure_target_table(schema: StructType) -> None:
    """Create dashboard table if missing and add new columns when needed."""
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            feed_ts TIMESTAMP,
            dashboard_ts TIMESTAMP,
            signal_ts TIMESTAMP,
            trade_ts TIMESTAMP,
            audit_ts TIMESTAMP,
            current_regime_start_ts TIMESTAMP,
            price_ts TIMESTAMP,
            macro_last_release_ts TIMESTAMP,
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
            account_balance DOUBLE,
            account_equity DOUBLE,
            account_cash DOUBLE,
            positions_json STRING,
            positions_count INT,
            positions_market_value DOUBLE,
            positions_unrealized_pl DOUBLE,
            positions_unrealized_plpc DOUBLE,
            hourly_pnl DOUBLE,
            hourly_pnl_pct DOUBLE,
            day_pnl DOUBLE,
            day_pnl_pct DOUBLE,
            total_return_pct DOUBLE,
            regime_return_pct DOUBLE,
            alpha_pct DOUBLE,
            drawdown_pct DOUBLE,
            regime_drawdown_pct DOUBLE,
            total_cycles INT,
            total_switches INT,
            peak_equity DOUBLE,
            regime_peak_equity DOUBLE,
            pct_expansion DOUBLE,
            pct_slowdown DOUBLE,
            pct_recession DOUBLE,
            audit_rows_total INT,
            audit_rows_last_24h INT,
            latest_audit_ts TIMESTAMP,
            audit_lag_minutes DOUBLE,
            execution_mode STRING,
            vix_source_date STRING,
            model_version STRING,
            source STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )

    current = existing_columns(TARGET_TABLE)
    for field in schema.fields:
        if field.name not in current:
            spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {field.name} {field.dataType.simpleString()}")


def load_latest_signal() -> Row:
    row = latest_row(SIGNAL_TABLE, "timestamp")
    if not row:
        raise ValueError(f"No rows found in {SIGNAL_TABLE}. Run inference before dashboard.")
    return row


def load_latest_silver() -> Row:
    row = latest_row(SILVER_RT, "timestamp")
    if not row:
        raise ValueError(f"No rows found in {SILVER_RT}. Run RT enrichment before dashboard.")
    return row


def load_latest_trade_snapshot(execution_mode: str) -> Optional[Row]:
    return latest_row(
        TRADE_LOG,
        "trade_ts",
        where_clause=f"WHERE execution_mode = '{execution_mode}'",
    )


def load_price_snapshot() -> dict:
    """Read current close for configured symbols from live market table."""
    if not table_exists(MARKET_TABLE):
        return {}

    row = scalar_row("SELECT MAX(timestamp) AS max_ts FROM " + MARKET_TABLE)
    if not row or not row["max_ts"]:
        return {}

    prices = spark.sql(
        f"""
        SELECT symbol, close
        FROM {MARKET_TABLE}
        WHERE symbol IN ({PRICE_SYMBOLS_SQL})
          AND timestamp = '{row['max_ts']}'
        """
    ).collect()
    return {row["symbol"]: maybe_float(row["close"]) for row in prices}


def infer_regime_start_ts(execution_mode: str, current_regime_int: Optional[int]) -> Optional[datetime]:
    """
    Find the first timestamp in the current contiguous regime segment.
    Checks audit trail first for regime continuity, then trade log.
    """
    if current_regime_int is None:
        return None

    def _first_contiguous_ts(rows, timestamp_col: str) -> Optional[datetime]:
        if not rows:
            return None
        if isinstance(rows, Row):
            rows = [rows]

        current_ts = None
        for row in rows:
            row_regime = maybe_int(row["regime_int"])
            if row_regime is None or row_regime != current_regime_int:
                break
            current_ts = safe_ts(row, timestamp_col)
        return current_ts

    if table_exists(AUDIT_LOG):
        audit_rows = latest_row(
            AUDIT_LOG,
            "audit_ts",
            where_clause=f"WHERE execution_mode = '{execution_mode}' AND audit_ts IS NOT NULL",
            limit=500,
        )
        start_ts = _first_contiguous_ts(audit_rows, "audit_ts")
        if start_ts is not None:
            return start_ts

    trade_rows = latest_row(
        TRADE_LOG,
        "trade_ts",
        where_clause=f"WHERE execution_mode = '{execution_mode}' AND trade_ts IS NOT NULL",
        limit=500,
    )
    return _first_contiguous_ts(trade_rows, "trade_ts")


def load_regime_return_metrics(execution_mode: str, latest_equity: Optional[float], regime_start_ts: Optional[datetime], current_regime_int: Optional[int]) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute total regime return and max-drawdown for the current regime.
    Returns regime_return_pct, regime_drawdown_pct, regime_peak_equity, regime_start_equity.
    """
    if latest_equity is None or regime_start_ts is None or current_regime_int is None:
        return None, None, None, None

    regime_start_row = scalar_row(
        f"""
        SELECT account_equity
        FROM {TRADE_LOG}
        WHERE execution_mode = '{execution_mode}'
          AND trade_ts >= TIMESTAMP('{regime_start_ts}')
          AND regime_int = {int(current_regime_int)}
          AND account_equity IS NOT NULL
        ORDER BY trade_ts ASC
        LIMIT 1
        """
    )
    if not regime_start_row or regime_start_row["account_equity"] is None:
        return None, None, None, None

    regime_start_equity = maybe_float(regime_start_row["account_equity"])
    if regime_start_equity in (None, 0.0):
        return None, None, regime_start_equity, None

    regime_peak_row = scalar_row(
        f"""
        SELECT MAX(account_equity) AS regime_peak_equity
        FROM {TRADE_LOG}
        WHERE execution_mode = '{execution_mode}'
          AND trade_ts >= TIMESTAMP('{regime_start_ts}')
          AND account_equity IS NOT NULL
        """
    )
    regime_peak_equity = maybe_float(regime_peak_row["regime_peak_equity"]) if regime_peak_row else None

    regime_return = latest_equity / regime_start_equity - 1.0
    regime_drawdown = None
    if regime_peak_equity not in (None, 0.0):
        regime_drawdown = latest_equity / regime_peak_equity - 1.0

    return regime_return, regime_drawdown, regime_peak_equity, regime_start_equity


def load_portfolio_metrics(execution_mode: str) -> dict[str, Optional[float]]:
    """Load durable counters and equity drawdown inputs for the selected execution mode."""
    metrics = scalar_row(
        f"""
        SELECT
            COUNT(*) AS total_cycles,
            SUM(CASE WHEN switched THEN 1 ELSE 0 END) AS total_switches,
            MAX(account_equity) AS peak_equity,
            MIN(CASE WHEN account_equity IS NOT NULL THEN account_equity END) AS min_equity,
            MIN(trade_ts) AS first_trade_ts
        FROM {TRADE_LOG}
        WHERE execution_mode = '{execution_mode}'
        """
    )
    if not metrics:
        return {
            "total_cycles": None,
            "total_switches": None,
            "peak_equity": None,
            "min_equity": None,
            "first_trade_ts": None,
        }

    return {
        "total_cycles": maybe_int(metrics["total_cycles"]),
        "total_switches": maybe_int(metrics["total_switches"]),
        "peak_equity": maybe_float(metrics["peak_equity"]),
        "min_equity": maybe_float(metrics["min_equity"]),
        "first_trade_ts": safe_ts(metrics, "first_trade_ts"),
    }


def load_total_return(latest_equity: Optional[float], first_trade_ts: Optional[datetime], execution_mode: str) -> Optional[float]:
    if latest_equity is None or first_trade_ts is None:
        return None

    anchor_row = scalar_row(
        f"""
        SELECT account_equity AS anchor_equity
        FROM {TRADE_LOG}
        WHERE execution_mode = '{execution_mode}'
          AND trade_ts >= TIMESTAMP('{first_trade_ts}')
          AND account_equity IS NOT NULL
        ORDER BY trade_ts ASC
        LIMIT 1
        """
    )
    if not anchor_row or anchor_row["anchor_equity"] is None:
        return None

    anchor_equity = maybe_float(anchor_row["anchor_equity"])
    if anchor_equity in (None, 0.0):
        return None
    return latest_equity / anchor_equity - 1.0


def load_alpha_pct(execution_mode: str, latest_equity: Optional[float], latest_trade_ts: Optional[datetime], regime_start_ts: Optional[datetime]) -> Optional[float]:
    """
    Compute benchmark-adjusted alpha from strategy and SPY path.
    Uses sparse-safe fallback to latest signal row if history is thin.
    """
    if latest_equity is None or latest_trade_ts is None:
        return None

    anchor_ts = regime_start_ts or latest_trade_ts
    anchor_row = scalar_row(
        f"""
        SELECT account_equity AS strategy_equity_anchor, spy_close AS spy_close_anchor
        FROM {TRADE_LOG}
        WHERE execution_mode = '{execution_mode}'
          AND trade_ts >= TIMESTAMP('{anchor_ts}')
          AND account_equity IS NOT NULL
          AND spy_close IS NOT NULL
        ORDER BY trade_ts ASC
        LIMIT 1
        """
    )
    if not anchor_row:
        return None

    strategy_anchor = maybe_float(anchor_row["strategy_equity_anchor"])
    spy_anchor = maybe_float(anchor_row["spy_close_anchor"])
    if strategy_anchor in (None, 0.0) or spy_anchor in (None, 0.0):
        return None

    latest = scalar_row(
        f"""
        SELECT spy_close
        FROM {TRADE_LOG}
        WHERE execution_mode = '{execution_mode}'
          AND trade_ts = TIMESTAMP('{latest_trade_ts}')
          AND spy_close IS NOT NULL
        LIMIT 1
        """
    )
    latest_spy = maybe_float(latest["spy_close"]) if latest else None
    if latest_spy is None or latest_spy in (None, 0.0):
        return None

    strategy_return = latest_equity / strategy_anchor - 1.0
    spy_return = latest_spy / spy_anchor - 1.0
    return strategy_return - spy_return


def load_regime_distribution(execution_mode: str) -> dict[str, Optional[float]]:
    if not table_exists(AUDIT_LOG):
        return {"pct_expansion": 0.0, "pct_slowdown": 0.0, "pct_recession": 0.0, "rows": 0}

    row = scalar_row(
        f"""
        SELECT
            COUNT(*) AS rows,
            ROUND(
                100.0 * SUM(CASE WHEN regime_int = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
                1
            ) AS pct_expansion,
            ROUND(
                100.0 * SUM(CASE WHEN regime_int = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
                1
            ) AS pct_slowdown,
            ROUND(
                100.0 * SUM(CASE WHEN regime_int = 2 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
                1
            ) AS pct_recession
        FROM {AUDIT_LOG}
        WHERE execution_mode = '{execution_mode}'
        """
    )
    if not row:
        return {"pct_expansion": 0.0, "pct_slowdown": 0.0, "pct_recession": 0.0, "rows": 0}

    return {
        "pct_expansion": maybe_float(row["pct_expansion"]),
        "pct_slowdown": maybe_float(row["pct_slowdown"]),
        "pct_recession": maybe_float(row["pct_recession"]),
        "rows": maybe_int(row["rows"]),
    }


def load_audit_health(execution_mode: str) -> dict[str, Optional[float] | Optional[int]]:
    if not table_exists(AUDIT_LOG):
        return {
            "audit_rows_total": 0,
            "audit_rows_last_24h": 0,
            "latest_audit_ts": None,
            "audit_lag_minutes": None,
        }

    total_row = scalar_row(
        f"""
        SELECT COUNT(*) AS total_rows
        FROM {AUDIT_LOG}
        WHERE execution_mode = '{execution_mode}'
        """
    )
    recent_row = scalar_row(
        f"""
        SELECT COUNT(*) AS rows_last_24h,
               MAX(audit_ts) AS latest_audit_ts
        FROM {AUDIT_LOG}
        WHERE execution_mode = '{execution_mode}'
          AND audit_ts >= TIMESTAMPADD(HOUR, -24, CURRENT_TIMESTAMP())
        """
    )
    latest_ts = recent_row["latest_audit_ts"] if recent_row else None
    if latest_ts is None:
        latest_row = scalar_row(
            f"""
            SELECT MAX(audit_ts) AS latest_audit_ts
            FROM {AUDIT_LOG}
            WHERE execution_mode = '{execution_mode}'
            """
        )
        latest_ts = latest_row["latest_audit_ts"] if latest_row else None

    audit_lag_minutes = None
    if latest_ts is not None:
        try:
            audit_lag_minutes = (datetime.now(timezone.utc).replace(tzinfo=None) - latest_ts).total_seconds() / 60.0
        except Exception:
            audit_lag_minutes = None

    return {
        "audit_rows_total": maybe_int(total_row["total_rows"]) if total_row else 0,
        "audit_rows_last_24h": maybe_int(recent_row["rows_last_24h"]) if recent_row else 0,
        "latest_audit_ts": latest_ts,
        "audit_lag_minutes": maybe_float(audit_lag_minutes),
    }


def load_vix_source_date() -> Optional[Any]:
    row = scalar_row(
        f"""
        SELECT MAX(CAST(timestamp AS DATE)) AS vix_source_date
        FROM {CATALOG}.{SCHEMA}.bronze_rt_vix_1d
        WHERE series = 'VIXCLS'
        """
    )
    if not row:
        return None
    value = row["vix_source_date"]
    if value is None:
        return None
    return str(value) if value is not None else None


# ---------------------------------------------------------------------------
# Section: Output assembly and write
# ---------------------------------------------------------------------------


def build_output_schema() -> StructType:
    return StructType(
        [
            StructField("feed_ts", TimestampType(), True),
            StructField("dashboard_ts", TimestampType(), True),
            StructField("signal_ts", TimestampType(), True),
            StructField("trade_ts", TimestampType(), True),
            StructField("audit_ts", TimestampType(), True),
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
            StructField("regime_peak_equity", DoubleType(), True),
            StructField("pct_expansion", DoubleType(), True),
            StructField("pct_slowdown", DoubleType(), True),
            StructField("pct_recession", DoubleType(), True),
            StructField("audit_rows_total", IntegerType(), True),
            StructField("audit_rows_last_24h", IntegerType(), True),
            StructField("latest_audit_ts", TimestampType(), True),
            StructField("audit_lag_minutes", DoubleType(), True),
            StructField("execution_mode", StringType(), True),
            StructField("vix_source_date", StringType(), True),
            StructField("model_version", StringType(), True),
            StructField("source", StringType(), True),
            StructField("ingestion_ts", TimestampType(), True),
        ]
    )


def main() -> None:
    execution_mode = resolve_execution_mode()

    signal_row = load_latest_signal()
    silver_row = load_latest_silver()
    trade_row = load_latest_trade_snapshot(execution_mode)

    prices = load_price_snapshot()
    portfolio = load_portfolio_metrics(execution_mode)
    portfolio_rows = load_regime_distribution(execution_mode)
    audit_health = load_audit_health(execution_mode)

    signal_ts = safe_ts(signal_row, "timestamp")
    current_regime_int = maybe_int(signal_row["regime_int"])
    current_regime_start_ts = infer_regime_start_ts(execution_mode, current_regime_int)
    if current_regime_start_ts is None:
        current_regime_start_ts = safe_ts(trade_row, "trade_ts") if trade_row else None

    latest_equity = maybe_float(trade_row["account_equity"]) if trade_row else None
    latest_audit_ts = safe_ts(audit_health, "latest_audit_ts")

    regime_return_pct, regime_drawdown_pct, regime_peak_equity, _regime_start_equity = load_regime_return_metrics(
        execution_mode,
        latest_equity,
        current_regime_start_ts,
        current_regime_int,
    )

    total_return_pct = load_total_return(
        latest_equity,
        portfolio.get("first_trade_ts"),
        execution_mode,
    )
    drawdown_pct = None
    if portfolio.get("peak_equity") not in (None, 0.0) and latest_equity is not None:
        drawdown_pct = latest_equity / portfolio["peak_equity"] - 1.0

    alpha_pct = load_alpha_pct(
        execution_mode=execution_mode,
        latest_equity=latest_equity,
        latest_trade_ts=safe_ts(trade_row, "trade_ts"),
        regime_start_ts=current_regime_start_ts,
    )

    macro_last_release = None
    if table_exists(MACRO_TABLE):
        macro_last = scalar_row(f"SELECT MAX(timestamp) AS macro_last_release_ts FROM {MACRO_TABLE}")
        if macro_last:
            macro_last_release = macro_last["macro_last_release_ts"]

    dashboard_ts = datetime.now(timezone.utc).replace(tzinfo=None)
    price_ts = None
    if table_exists(MARKET_TABLE):
        price_ts_row = scalar_row(f"SELECT MAX(timestamp) AS max_ts FROM {MARKET_TABLE}")
        if price_ts_row:
            price_ts = price_ts_row.get("max_ts")

    dashboard_row = {
        "feed_ts": dashboard_ts,
        "dashboard_ts": dashboard_ts,
        "signal_ts": signal_ts,
        "trade_ts": safe_ts(trade_row, "trade_ts") if trade_row else None,
        "audit_ts": latest_audit_ts,
        "current_regime_start_ts": current_regime_start_ts,
        "price_ts": price_ts,
        "macro_last_release_ts": macro_last_release,
        "regime_int": current_regime_int,
        "regime_name": maybe_str(signal_row["regime_name"]),
        "position": maybe_str(signal_row["position"]),
        "target_ticker": maybe_str(trade_row["target_ticker"]) if trade_row else None,
        "VIXCLS_z": maybe_float(silver_row["VIXCLS_z"]),
        "vix3m_z": maybe_float(silver_row["vix3m_z"]),
        "VIX_Inversion_Z": maybe_float(silver_row["VIX_Inversion_Z"]),
        "Price_Mom_24H": maybe_float(silver_row["Price_Mom_24H"]),
        "CPIAUCSL_z": maybe_float(silver_row["CPIAUCSL_z"]),
        "PCEPI_z": maybe_float(silver_row["PCEPI_z"]),
        "PCE_z": maybe_float(silver_row["PCE_z"]),
        "PAYEMS_z": maybe_float(silver_row["PAYEMS_z"]),
        "UNRATE_z": maybe_float(silver_row["UNRATE_z"]),
        "GDPC1_z": maybe_float(silver_row["GDPC1_z"]),
        "INDPRO_z": maybe_float(silver_row["INDPRO_z"]),
        "UMCSENT_z": maybe_float(silver_row["UMCSENT_z"]),
        "CPI_Event_Active": maybe_int(silver_row["CPI_Event_Active"]),
        "NFP_Event_Active": maybe_int(silver_row["NFP_Event_Active"]),
        "SPY_close": prices.get(PRIMARY_SYMBOL) if prices else None,
        "QQQ_close": prices.get(SECONDARY_SYMBOL) if prices else None,
        "account_balance": maybe_float(trade_row["account_balance"]) if trade_row else None,
        "account_equity": latest_equity,
        "account_cash": maybe_float(trade_row["account_cash"]) if trade_row else None,
        "positions_json": maybe_str(trade_row["positions_json"]) if trade_row else None,
        "positions_count": maybe_int(trade_row["positions_count"]) if trade_row else None,
        "positions_market_value": maybe_float(trade_row["positions_market_value"]) if trade_row else None,
        "positions_unrealized_pl": maybe_float(trade_row["positions_unrealized_pl"]) if trade_row else None,
        "positions_unrealized_plpc": maybe_float(trade_row["positions_unrealized_plpc"]) if trade_row else None,
        "hourly_pnl": maybe_float(trade_row["hourly_pnl"]) if trade_row else None,
        "hourly_pnl_pct": maybe_float(trade_row["hourly_pnl_pct"]) if trade_row else None,
        "day_pnl": maybe_float(trade_row["day_pnl"]) if trade_row else None,
        "day_pnl_pct": maybe_float(trade_row["day_pnl_pct"]) if trade_row else None,
        "total_return_pct": total_return_pct,
        "regime_return_pct": regime_return_pct,
        "alpha_pct": alpha_pct,
        "drawdown_pct": drawdown_pct,
        "regime_drawdown_pct": regime_drawdown_pct,
        "total_cycles": portfolio.get("total_cycles"),
        "total_switches": portfolio.get("total_switches"),
        "peak_equity": portfolio.get("peak_equity"),
        "regime_peak_equity": regime_peak_equity,
        "pct_expansion": portfolio_rows["pct_expansion"],
        "pct_slowdown": portfolio_rows["pct_slowdown"],
        "pct_recession": portfolio_rows["pct_recession"],
        "audit_rows_total": audit_health["audit_rows_total"],
        "audit_rows_last_24h": audit_health["audit_rows_last_24h"],
        "latest_audit_ts": audit_health["latest_audit_ts"],
        "audit_lag_minutes": audit_health["audit_lag_minutes"],
        "execution_mode": execution_mode,
        "vix_source_date": load_vix_source_date(),
        "model_version": get_model_version(maybe_str(signal_row["model_version"]) or "bootstrap_live_model"),
        "source": "rt_saxo_dashboard_feed_v4",
        "ingestion_ts": dashboard_ts,
    }

    output_schema = build_output_schema()
    ensure_target_table(output_schema)

    output_df = spark.createDataFrame([dashboard_row], schema=output_schema)
    overwrite_dataframe_with_retry(
        output_df,
        TARGET_TABLE,
        options={"overwriteSchema": "true"},
        operation_name=f"overwrite Saxo dashboard feed table {TARGET_TABLE}",
    )

    print(f"Saxo dashboard feed written to {TARGET_TABLE}")
    print(f"Execution mode: {execution_mode}")
    print(f"Execution timestamp: {dashboard_ts}")
    print(f"Signal timestamp: {signal_ts}")
    if trade_row is not None:
        print(f"Trade timestamp: {safe_ts(trade_row, 'trade_ts')}")
        print(f"Account equity: {dashboard_row['account_equity']}")
    print(f"Regime: {dashboard_row['regime_name']} ({dashboard_row['regime_int']})")
    if dashboard_row["regime_return_pct"] is not None:
        print(f"Regime return: {dashboard_row['regime_return_pct']:+.2%}")
    if dashboard_row["regime_drawdown_pct"] is not None:
        print(f"Regime drawdown: {dashboard_row['regime_drawdown_pct']:+.2%}")
    if dashboard_row["alpha_pct"] is not None:
        print(f"Alpha (vs SPY): {dashboard_row['alpha_pct']:+.2%}")


main()
