# Databricks notebook source
# Live portfolio performance since first V4 signal

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config

import matplotlib.pyplot as plt
import pandas as pd

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
LEGACY_SCHEMA = "default"
runtime_config = load_runtime_instrument_config(spark, CATALOG, SCHEMA)
PRIMARY_SYMBOL = runtime_config["primary_symbol"]
SECONDARY_SYMBOL = runtime_config["secondary_symbol"]
SYMBOLS_SQL = ", ".join([f"'{symbol}'" for symbol in {PRIMARY_SYMBOL, SECONDARY_SYMBOL} if symbol])

INITIAL_CAPITAL = 100_000.0
TRADING_COST = 0.0005

SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
TRADE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_paper_trade_log"
LEGACY_SIGNAL_TABLE = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_rt_regime_signal"
LEGACY_MARKET_TABLE = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_rt_market_1h"
TARGET_TIMESERIES = f"{CATALOG}.{SCHEMA}.gold_rt_portfolio_timeseries"
TARGET_SUMMARY = f"{CATALOG}.{SCHEMA}.gold_rt_portfolio_summary"
EXPORT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/performance/live"
CHART_PATH = f"{EXPORT_ROOT}/live_portfolio_performance.png"

REGIME_NAME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
POSITION_MAP = runtime_config["position_map"]


def load_union_frame(query_blocks):
    sql = "\nUNION ALL\n".join(query_blocks)
    if not sql.strip():
        return pd.DataFrame()
    return spark.sql(sql).toPandas()


def table_exists(table_name):
    return spark.catalog.tableExists(table_name)


def load_inputs():
    signal_queries = []
    if table_exists(LEGACY_SIGNAL_TABLE):
        signal_queries.append(
            f"""
            SELECT
                timestamp,
                regime_int,
                CAST(NULL AS STRING) AS regime_name,
                CAST(NULL AS STRING) AS position,
                'legacy_default_rt' AS model_version,
                0 AS source_rank
            FROM {LEGACY_SIGNAL_TABLE}
            """
        )
    if table_exists(SIGNAL_TABLE):
        signal_queries.append(
            f"""
            SELECT
                timestamp,
                regime_int,
                regime_name,
                position,
                COALESCE(model_version, 'v4_rt') AS model_version,
                1 AS source_rank
            FROM {SIGNAL_TABLE}
            """
        )

    signals = load_union_frame(signal_queries)
    if signals.empty:
        raise ValueError("No RT regime signals found. Run RT_04 first.")

    price_queries = []
    if table_exists(LEGACY_MARKET_TABLE):
        price_queries.append(
            f"""
            SELECT timestamp, symbol, close, 0 AS source_rank
            FROM {LEGACY_MARKET_TABLE}
            WHERE symbol IN ({SYMBOLS_SQL})
            """
        )
    if table_exists(MARKET_TABLE):
        price_queries.append(
            f"""
            SELECT timestamp, symbol, close, 1 AS source_rank
            FROM {MARKET_TABLE}
            WHERE symbol IN ({SYMBOLS_SQL})
            """
        )

    prices = load_union_frame(price_queries)
    if prices.empty:
        raise ValueError("No RT market prices found. Run RT_01 first.")

    trades = spark.sql(
        f"""
        SELECT
            signal_ts,
            trade_ts,
            account_equity,
            account_cash,
            positions_count,
            positions_market_value,
            positions_unrealized_pl,
            switched,
            orders_placed
        FROM {TRADE_TABLE}
        """
    ).toPandas()

    signals["timestamp"] = pd.to_datetime(signals["timestamp"])
    signals = (
        signals.sort_values(["timestamp", "source_rank"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .drop(columns=["source_rank"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    signals["regime_name"] = signals["regime_name"].fillna(signals["regime_int"].map(REGIME_NAME_MAP))
    signals["position"] = signals["position"].fillna(signals["regime_int"].map(POSITION_MAP))

    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    if not trades.empty:
        trades["signal_ts"] = pd.to_datetime(trades["signal_ts"])
        trades["trade_ts"] = pd.to_datetime(trades["trade_ts"])

    prices = (
        prices.sort_values(["timestamp", "symbol", "source_rank"])
        .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        .drop(columns=["source_rank"])
        .pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )

    merged = (
        signals.set_index("timestamp")
        .join(prices, how="inner")
        .dropna(subset=[PRIMARY_SYMBOL, SECONDARY_SYMBOL])
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )

    if not trades.empty:
        merged = merged.merge(
            trades,
            left_on="timestamp",
            right_on="signal_ts",
            how="left",
        ).drop(columns=["signal_ts"])

    return merged
def run_live_nav(data):
    data = data.sort_values("timestamp").reset_index(drop=True).copy()
    portfolio = [INITIAL_CAPITAL]
    current_val = INITIAL_CAPITAL
    active_asset = "START"
    active_assets = ["START"]
    trade_costs = [0.0]
    step_returns = [0.0]

    for i in range(1, len(data)):
        regime = int(data["regime_int"].iloc[i - 1])
        target = runtime_config["regime_ticker"].get(regime) or "CASH"

        if target == SECONDARY_SYMBOL:
            ret = (data[SECONDARY_SYMBOL].iloc[i] / data[SECONDARY_SYMBOL].iloc[i - 1]) - 1.0
        elif target == PRIMARY_SYMBOL:
            ret = (data[PRIMARY_SYMBOL].iloc[i] / data[PRIMARY_SYMBOL].iloc[i - 1]) - 1.0
        else:
            ret = 0.0

        friction = TRADING_COST if target != active_asset else 0.0
        current_val *= (1.0 + ret - friction)

        portfolio.append(current_val)
        active_assets.append(target)
        trade_costs.append(friction)
        step_returns.append(ret)
        active_asset = target

    data["strategy_value"] = portfolio
    data["spy_benchmark"] = (
        data[PRIMARY_SYMBOL] / data[PRIMARY_SYMBOL].iloc[0]
    ) * INITIAL_CAPITAL
    data["active_asset"] = active_assets
    data["step_return"] = step_returns
    data["transaction_cost"] = trade_costs
    data["strategy_return_pct"] = data["strategy_value"].pct_change().fillna(0.0)
    data["spy_return_pct"] = data["spy_benchmark"].pct_change().fillna(0.0)
    data["strategy_drawdown_pct"] = (
        data["strategy_value"] / data["strategy_value"].cummax() - 1.0
    ).fillna(0.0)
    data["benchmark_drawdown_pct"] = (
        data["spy_benchmark"] / data["spy_benchmark"].cummax() - 1.0
    ).fillna(0.0)
    data["alpha_pct"] = (
        data["strategy_value"] / data["strategy_value"].iloc[0] - 1.0
    ) - (
        data["spy_benchmark"] / data["spy_benchmark"].iloc[0] - 1.0
    )
    return data


def save_outputs(data):
    dbutils.fs.mkdirs(EXPORT_ROOT)

    spark.createDataFrame(data).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(TARGET_TIMESERIES)
    data.to_csv(f"{EXPORT_ROOT}/live_portfolio_timeseries.csv", index=False)

    latest = data.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "as_of_ts": latest["timestamp"],
                "start_ts": data["timestamp"].iloc[0],
                "bars": int(len(data)),
                "start_capital": float(data["strategy_value"].iloc[0]),
                "strategy_value": float(latest["strategy_value"]),
                "spy_benchmark": float(latest["spy_benchmark"]),
                "strategy_total_return_pct": float(
                    latest["strategy_value"] / data["strategy_value"].iloc[0] - 1.0
                ),
                "spy_total_return_pct": float(
                    latest["spy_benchmark"] / data["spy_benchmark"].iloc[0] - 1.0
                ),
                "alpha_pct": float(latest["alpha_pct"]),
                "strategy_drawdown_pct": float(latest["strategy_drawdown_pct"]),
                "benchmark_drawdown_pct": float(latest["benchmark_drawdown_pct"]),
                "latest_regime_int": int(latest["regime_int"]),
                "latest_regime_name": latest["regime_name"],
                "latest_position": latest["position"],
                "model_version": latest["model_version"],
                "account_equity": (
                    float(latest["account_equity"])
                    if pd.notna(latest.get("account_equity"))
                    else None
                ),
                "account_cash": (
                    float(latest["account_cash"])
                    if pd.notna(latest.get("account_cash"))
                    else None
                ),
            }
        ]
    )

    spark.createDataFrame(summary).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(TARGET_SUMMARY)
    summary.to_csv(f"{EXPORT_ROOT}/live_portfolio_summary.csv", index=False)


def render_chart(data):
    chart_df = data.sort_values("timestamp").reset_index(drop=True).copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    ax1.plot(chart_df["timestamp"], chart_df["strategy_value"], label="Strategy", linewidth=2)
    ax1.plot(chart_df["timestamp"], chart_df["spy_benchmark"], label="SPY Benchmark", alpha=0.7)
    ax1.set_title("Live Portfolio Performance vs SPY")
    ax1.set_ylabel("Portfolio Value")
    ax1.legend()
    ax1.grid(True)

    regimes = chart_df["regime_int"]
    ax2.fill_between(chart_df["timestamp"], 0, 1, where=(regimes == 0), color="green", alpha=0.35, label="Expansion (QQQ)")
    ax2.fill_between(chart_df["timestamp"], 0, 1, where=(regimes == 1), color="gold", alpha=0.35, label="Slowdown (SPY)")
    ax2.fill_between(chart_df["timestamp"], 0, 1, where=(regimes == 2), color="red", alpha=0.35, label="Recession (Cash)")
    ax2.set_yticks([])
    ax2.set_ylabel("Regime")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


portfolio_df = run_live_nav(load_inputs())
save_outputs(portfolio_df)
render_chart(portfolio_df)

latest = portfolio_df.iloc[-1]
print(f"Live portfolio time series written to {TARGET_TIMESERIES}")
print(f"Live portfolio summary written to {TARGET_SUMMARY}")
print(f"Live portfolio chart written to {CHART_PATH}")
print(f"Start timestamp: {portfolio_df['timestamp'].iloc[0]}")
print(f"As of timestamp: {latest['timestamp']}")
print(f"Strategy value: {latest['strategy_value']}")
print(f"SPY benchmark: {latest['spy_benchmark']}")
print(f"Strategy total return pct: {latest['strategy_value'] / portfolio_df['strategy_value'].iloc[0] - 1.0}")
print(f"SPY total return pct: {latest['spy_benchmark'] / portfolio_df['spy_benchmark'].iloc[0] - 1.0}")
print(f"Alpha pct: {latest['alpha_pct']}")
