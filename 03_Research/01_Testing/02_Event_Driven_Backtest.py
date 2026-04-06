# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 1: Setup and Data Loading

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the V4 Signal File
VIX_SOURCE  = "fred_1d" #toggle #portara_1h #fred_1d
signal_path = f"/Volumes/pawan_quant_research_1/v4/market_data/bt_{VIX_SOURCE}_regime_signals-v4.csv"
df = pd.read_csv(signal_path, index_col='timestamp', parse_dates=True)
df.index = df.index.tz_localize(None) # Harmonize to Naive UTC

# 2. Load Hourly ETF Prices
market_sdf = spark.table("pawan_quant_research_1.v4.bronze_bt_market_1h")
# FIX: Filter and Deduplicate before converting to Pandas
qqq_prices_pdf = market_sdf.filter("symbol = 'QQQ'").dropDuplicates(["timestamp"]).toPandas()
qqq_prices_pdf['timestamp'] = pd.to_datetime(qqq_prices_pdf['timestamp']).dt.tz_localize(None)
qqq_prices = qqq_prices_pdf.sort_values("timestamp").set_index("timestamp")['close'].rename("QQQ_price")

# 3. Merge QQQ into master signal dataframe
df = df.join(qqq_prices, how='left')
df['QQQ_price'] = df['QQQ_price'].ffill()

# ── FIX: Anchor all NAVs to a common start date ──
BACKTEST_START = "2016-02-01"
df = df.loc[BACKTEST_START:]
print(f"✅ Backtest anchored from {df.index.min()} → {df.index.max()}")

print(f"Backtest Data Loaded. Shape: {df.shape}")

print(f"Backtest Data Loaded. Shape: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: The V4 Event-Driven Engine

# COMMAND ----------

# MAGIC %md
# MAGIC - #### **Regime 0 (Expansion)**: 100% QQQ (Aggressive Growth)
# MAGIC - #### **Regime 1 (Slowdown)**: 100% SPY (Core Exposure)
# MAGIC - #### **Regime 2 (Rececssion)**: 100% Cash (Protecting Capital)

# COMMAND ----------

# 1. Initialize Portfolio
portfolio_value = 5_000_000
holdings_values = []
active_asset = "START"  # Track the name of the asset held

# 2. Iterate through every hour
for i in range(1, len(df)):
    # Use previous hour's prediction to determine current hour's position (no look-ahead)
    regime_signal = df['predicted_regime'].iloc[i-1]

    # Determine target asset for this hour
    if regime_signal == 0:
        target_asset = "QQQ"
        hourly_ret = (df['QQQ_price'].iloc[i] / df['QQQ_price'].iloc[i-1]) - 1
    elif regime_signal == 1:
        target_asset = "SPY"
        hourly_ret = (df['SPY_close'].iloc[i] / df['SPY_close'].iloc[i-1]) - 1
    else:
        target_asset = "CASH"
        hourly_ret = 0.0

    # Apply Transaction Costs (5 bps) ONLY if we switch assets
    if target_asset != active_asset:
        cost = 0.0005
        active_asset = target_asset  # Update our tracker
    else:
        cost = 0.0

    # Update portfolio value
    portfolio_value *= (1 + hourly_ret - cost)

    # Store for analysis
    holdings_values.append(portfolio_value)

# 3. Update the results dataframe
df_backtest = df.iloc[1:].copy()
df_backtest['Strategy_Value'] = holdings_values
df_backtest['Benchmark_Value'] = (df_backtest['SPY_close'] / df_backtest['SPY_close'].iloc[0]) * 5_000_000

print("✅ Backtest Simulation Fixed and Complete.")
print(f"   Period: {df_backtest.index.min().date()} → {df_backtest.index.max().date()}")
print(f"   Final Strategy Value: ${df_backtest['Strategy_Value'].iloc[-1]:,.0f}")
print(f"   Final Benchmark Value: ${df_backtest['Benchmark_Value'].iloc[-1]:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Performance Analysis & Alpha Quantification

# COMMAND ----------

# Step 3: Performance Analysis & Alpha Quantification

# ── FIX: Use same CAGR method as Risk Report (calendar days, initial capital anchor) ──
def calculate_metrics(equity_series):
    daily = equity_series.resample('D').last().dropna()
    n_days = len(daily)
    years = n_days / 252                          # trading days — matches Risk Report
    cagr = (daily.iloc[-1] / 5_000_000) ** (1 / years) - 1
    return cagr

strat_cagr = calculate_metrics(df_backtest['Strategy_Value'])
bench_cagr = calculate_metrics(df_backtest['Benchmark_Value'])

plt.figure(figsize=(15, 7))
plt.plot(df_backtest['Strategy_Value'], label=f"V4 Strategy (CAGR: {strat_cagr:.2%})", color='navy', linewidth=2)
plt.plot(df_backtest['Benchmark_Value'], label=f"SPY Benchmark (CAGR: {bench_cagr:.2%})", color='grey', alpha=0.6)
plt.title("V4 Event-Driven Alpha: Reacting at the Hour of Impact")
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()

print(f"V4 CAGR:           {strat_cagr:.2%}")
print(f"SPY CAGR:          {bench_cagr:.2%}")
print(f"V4 Outperformance: {strat_cagr - bench_cagr:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The V4 Risk Report

# COMMAND ----------

# 1. Calculate Daily Returns from our Hourly Value
# Use daily close prices and compute simple daily returns
daily_strat = df_backtest['Strategy_Value'].resample('D').last().pct_change().dropna()
daily_bench = df_backtest['Benchmark_Value'].resample('D').last().pct_change().dropna()

TRADING_DAYS = 252

# 2. Performance Metrics (industry-style: daily returns -> annual Sharpe)
def get_risk_metrics(daily_returns):
    # Total period info
    n_days = daily_returns.shape[0]
    years = n_days / TRADING_DAYS

    # CAGR from cumulative equity, not from mean
    cum_ret = (1 + daily_returns).cumprod()
    cagr = cum_ret.iloc[-1] ** (1 / years) - 1

    # Annualized volatility from daily returns
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS)

    # Sharpe using arithmetic mean of daily returns
    mean_ret = daily_returns.mean()
    sharpe = (mean_ret / daily_returns.std()) * np.sqrt(TRADING_DAYS) if daily_returns.std() != 0 else 0.0

    # Drawdown on cumulative equity
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    mdd = drawdown.min()

    return {
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{mdd:.2%}"
    }

strat_metrics = get_risk_metrics(daily_strat)
bench_metrics = get_risk_metrics(daily_bench)

print("--- V4 STRATEGY RISK REPORT ---")
for k, v in strat_metrics.items():
    print(f"{k}: {v}")

print("\n--- BENCHMARK RISK REPORT ---")
for k, v in bench_metrics.items():
    print(f"{k}: {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## The "Current State" Dashboard

# COMMAND ----------

# Get the most recent row from your backtest
latest_bar = df_backtest.iloc[-1]

print("==================================================")
print(f"       V4 STRATEGY STATUS: {latest_bar.name}")
print("==================================================")
print(f"Current Regime:      {latest_bar['regime_name']} ({int(latest_bar['predicted_regime'])})")
print(f"Holdings Suggestion: {'100% QQQ' if latest_bar['predicted_regime']==0 else '100% SPY' if latest_bar['predicted_regime']==1 else '100% CASH'}")
print("--------------------------------------------------")
# Using the updated Z-Score feature name
print(f"VIX Inversion (Z):   {latest_bar['VIX_Inversion_Z']:.4f}")
print(f"Price Momentum (24H): {latest_bar['Price_Mom_24H']:.2%}")
print(f"Inflation Z-Score:   {latest_bar['CPIAUCSL_z']:.2f}")
print(f"Unemployment Z-Score: {latest_bar['UNRATE_z']:.2f}")
print("--------------------------------------------------")
print(f"Macro Event Status:  {'CPI Active' if latest_bar['CPI_Event_Active']==1 else 'NFP Active' if latest_bar['NFP_Event_Active']==1 else 'No Active News'}")
print("==================================================")