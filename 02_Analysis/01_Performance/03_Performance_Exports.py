# Databricks notebook source
# Canonical V4 performance exports

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"
EXPORT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/performance"

def ensure_export_dirs():
    for suffix in ["daily", "weekly", "monthly", "quarterly", "yearly", "summary"]:
        dbutils.fs.mkdirs(f"{EXPORT_ROOT}/{suffix}")

def load_signal_and_prices():
    signals = spark.table(SIGNAL_TABLE).toPandas()
    prices = (
        spark.table(MARKET_TABLE)
        .filter("symbol IN ('SPY','QQQ')")
        .toPandas()
        .drop_duplicates(subset=["timestamp", "symbol"])
    )
    signals["timestamp"] = pd.to_datetime(signals["timestamp"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices_pivot = prices.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
    merged = signals.merge(prices_pivot.reset_index(), on="timestamp", how="left").sort_values("timestamp")
    merged["SPY"] = merged["SPY"].ffill()
    merged["QQQ"] = merged["QQQ"].ffill()
    return merged

def run_nav(data, initial_capital=5_000_000):
    nav = [initial_capital]
    current = initial_capital
    active_asset = "START"
    for i in range(1, len(data)):
        regime = int(data["predicted_regime"].iloc[i - 1])
        target = "QQQ" if regime == 0 else "SPY" if regime == 1 else "CASH"
        if target == "QQQ":
            ret = (data["QQQ"].iloc[i] / data["QQQ"].iloc[i - 1]) - 1
        elif target == "SPY":
            ret = (data["SPY"].iloc[i] / data["SPY"].iloc[i - 1]) - 1
        else:
            ret = 0.0
        friction = 0.0005 if target != active_asset else 0.0
        current *= (1 + ret - friction)
        nav.append(current)
        active_asset = target
    data = data.copy()
    data["WalkForward_V4"] = nav
    data["SPY_Benchmark"] = (data["SPY"] / data["SPY"].iloc[0]) * initial_capital
    return data

def save_frequency_table(pdf, table_name, export_path):
    spark.createDataFrame(pdf.reset_index().rename(columns={"index": "timestamp"})).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    pdf.to_csv(export_path)

ensure_export_dirs()
merged = run_nav(load_signal_and_prices())
daily_returns = merged.set_index("timestamp")[["WalkForward_V4", "SPY_Benchmark"]].resample("D").last().pct_change().dropna()
weekly_returns = daily_returns.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)
monthly_returns = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
quarterly_returns = daily_returns.resample("QE").apply(lambda x: (1 + x).prod() - 1)
yearly_returns = daily_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)

summary = pd.DataFrame({
    "final_value": merged[["WalkForward_V4", "SPY_Benchmark"]].iloc[-1],
    "total_return": (merged[["WalkForward_V4", "SPY_Benchmark"]].iloc[-1] / merged[["WalkForward_V4", "SPY_Benchmark"]].iloc[0]) - 1,
})

save_frequency_table(daily_returns, f"{CATALOG}.{SCHEMA}.gold_bt_returns_daily", f"{EXPORT_ROOT}/daily/walkforward_daily_returns.csv")
save_frequency_table(weekly_returns, f"{CATALOG}.{SCHEMA}.gold_bt_returns_weekly", f"{EXPORT_ROOT}/weekly/walkforward_weekly_returns.csv")
save_frequency_table(monthly_returns, f"{CATALOG}.{SCHEMA}.gold_bt_returns_monthly", f"{EXPORT_ROOT}/monthly/walkforward_monthly_returns.csv")
save_frequency_table(quarterly_returns, f"{CATALOG}.{SCHEMA}.gold_bt_returns_quarterly", f"{EXPORT_ROOT}/quarterly/walkforward_quarterly_returns.csv")
save_frequency_table(yearly_returns, f"{CATALOG}.{SCHEMA}.gold_bt_returns_yearly", f"{EXPORT_ROOT}/yearly/walkforward_yearly_returns.csv")
spark.createDataFrame(summary.reset_index().rename(columns={"index": "strategy"})).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.gold_bt_performance_summary")
summary.to_csv(f"{EXPORT_ROOT}/summary/walkforward_performance_summary.csv", index=True)

print("Canonical performance tables and CSV exports written.")
