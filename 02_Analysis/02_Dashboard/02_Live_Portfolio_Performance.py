# Databricks notebook source
# Live portfolio performance since first V4 signal

import pandas as pd

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

INITIAL_CAPITAL = 10_000.0
TRADING_COST = 0.0005

SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
TRADE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_paper_trade_log"
TARGET_TIMESERIES = f"{CATALOG}.{SCHEMA}.gold_rt_portfolio_timeseries"
TARGET_SUMMARY = f"{CATALOG}.{SCHEMA}.gold_rt_portfolio_summary"
EXPORT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/performance/live"


def load_inputs():
    signals = spark.sql(
        f"""
        SELECT timestamp, regime_int, regime_name, position, model_version
        FROM {SIGNAL_TABLE}
        ORDER BY timestamp
        """
    ).toPandas()
    if signals.empty:
        raise ValueError("No RT regime signals found. Run RT_04 first.")

    prices = spark.sql(
        f"""
        SELECT timestamp, symbol, close
        FROM {MARKET_TABLE}
        WHERE symbol IN ('SPY', 'QQQ')
        """
    ).toPandas()
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
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    if not trades.empty:
        trades["signal_ts"] = pd.to_datetime(trades["signal_ts"])
        trades["trade_ts"] = pd.to_datetime(trades["trade_ts"])

    prices = (
        prices.drop_duplicates(subset=["timestamp", "symbol"])
        .pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )

    merged = (
        signals.set_index("timestamp")
        .join(prices, how="inner")
        .dropna(subset=["SPY", "QQQ"])
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


def starting_capital(data):
    account_equity = data["account_equity"].dropna() if "account_equity" in data.columns else pd.Series(dtype=float)
    if not account_equity.empty:
        return float(account_equity.iloc[0])
    return INITIAL_CAPITAL


def run_live_nav(data):
    data = data.sort_values("timestamp").reset_index(drop=True).copy()
    start_capital = starting_capital(data)
    portfolio = [start_capital]
    current_val = start_capital
    active_asset = "START"
    active_assets = ["START"]
    trade_costs = [0.0]
    step_returns = [0.0]

    for i in range(1, len(data)):
        regime = int(data["regime_int"].iloc[i - 1])
        target = "QQQ" if regime == 0 else "SPY" if regime == 1 else "CASH"

        if target == "QQQ":
            ret = (data["QQQ"].iloc[i] / data["QQQ"].iloc[i - 1]) - 1.0
        elif target == "SPY":
            ret = (data["SPY"].iloc[i] / data["SPY"].iloc[i - 1]) - 1.0
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
    data["spy_benchmark"] = (data["SPY"] / data["SPY"].iloc[0]) * start_capital
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


portfolio_df = run_live_nav(load_inputs())
save_outputs(portfolio_df)

latest = portfolio_df.iloc[-1]
print(f"Live portfolio time series written to {TARGET_TIMESERIES}")
print(f"Live portfolio summary written to {TARGET_SUMMARY}")
print(f"Start timestamp: {portfolio_df['timestamp'].iloc[0]}")
print(f"As of timestamp: {latest['timestamp']}")
print(f"Strategy value: {latest['strategy_value']}")
print(f"SPY benchmark: {latest['spy_benchmark']}")
print(f"Alpha pct: {latest['alpha_pct']}")
