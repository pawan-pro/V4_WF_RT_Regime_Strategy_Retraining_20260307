# Databricks notebook source
# Shared runtime instrument configuration for live Databricks jobs.

DEFAULT_CATALOG = "pawan_quant_research_1"
DEFAULT_SCHEMA = "v4"


def runtime_config_table(catalog=DEFAULT_CATALOG, schema=DEFAULT_SCHEMA):
    return f"{catalog}.{schema}.rt_runtime_instrument_config"


def target_position_label(ticker):
    return f"100% {ticker}" if ticker else "100% CASH"


def ensure_runtime_instrument_config_table(
    spark, catalog=DEFAULT_CATALOG, schema=DEFAULT_SCHEMA
):
    table_name = runtime_config_table(catalog, schema)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            config_name STRING,
            market_symbols_csv STRING,
            primary_symbol STRING,
            secondary_symbol STRING,
            regime_0_ticker STRING,
            regime_1_ticker STRING,
            regime_2_ticker STRING,
            data_source STRING,
            trading_source STRING,
            notes STRING,
            updated_ts TIMESTAMP,
            updated_by STRING
        )
        USING DELTA
        """
    )

    row_count = spark.sql(f"SELECT COUNT(*) AS n FROM {table_name}").collect()[0]["n"]
    if row_count == 0:
        spark.sql(
            f"""
            INSERT INTO {table_name}
            VALUES (
                'default_live',
                'SPY,QQQ',
                'SPY',
                'QQQ',
                'QQQ',
                'SPY',
                NULL,
                'alpaca_stocks',
                'alpaca_paper',
                'Default live runtime mapping for the V4 hourly chain',
                current_timestamp(),
                current_user()
            )
            """
        )

    return table_name


def load_runtime_instrument_config(
    spark, catalog=DEFAULT_CATALOG, schema=DEFAULT_SCHEMA
):
    table_name = ensure_runtime_instrument_config_table(spark, catalog, schema)
    row = spark.sql(
        f"""
        SELECT *
        FROM {table_name}
        ORDER BY updated_ts DESC
        LIMIT 1
        """
    ).collect()[0]

    market_symbols = [
        symbol.strip()
        for symbol in (row["market_symbols_csv"] or "").split(",")
        if symbol.strip()
    ]
    primary_symbol = row["primary_symbol"] or (market_symbols[0] if market_symbols else None)
    secondary_symbol = row["secondary_symbol"] or (
        market_symbols[1] if len(market_symbols) > 1 else primary_symbol
    )
    regime_ticker = {
        0: row["regime_0_ticker"],
        1: row["regime_1_ticker"],
        2: row["regime_2_ticker"] or None,
    }

    return {
        "table_name": table_name,
        "config_name": row["config_name"],
        "market_symbols": market_symbols,
        "primary_symbol": primary_symbol,
        "secondary_symbol": secondary_symbol,
        "regime_ticker": regime_ticker,
        "position_map": {
            regime: target_position_label(ticker)
            for regime, ticker in regime_ticker.items()
        },
        "data_source": row["data_source"],
        "trading_source": row["trading_source"],
        "notes": row["notes"],
        "updated_ts": row["updated_ts"],
        "updated_by": row["updated_by"],
    }
