# Databricks notebook source
# MAGIC %md
# MAGIC V4 Strategy Performance Deep Dive

# COMMAND ----------

# 09RT: Performance Deep Dive
# Updated: BT namespace (bronze_bt_market_1h, bt_fred_1d signals)
# New: Monthly / Quarterly / Yearly returns export for V3, V4, SPY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# ── SETTINGS ──
INITIAL_CAPITAL = 5_000_000
SIGNAL_PATH     = "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_regime_signals-v4.csv"
CSV_OUT_BASE    = "/Volumes/pawan_quant_research_1/default/market_data"

# ── LOAD SIGNALS ──
df = pd.read_csv(SIGNAL_PATH, index_col='timestamp', parse_dates=True)
df.index = df.index.tz_localize(None)

# ── LOAD PRICES FROM BT NAMESPACE ──
market_sdf = spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
prices_pdf = market_sdf.filter("symbol IN ('SPY', 'QQQ')").toPandas()
prices_pdf = prices_pdf.drop_duplicates(subset=['timestamp', 'symbol'])
prices_pdf['timestamp'] = pd.to_datetime(prices_pdf['timestamp']).dt.tz_localize(None)

prices_pivoted = prices_pdf.pivot(index='timestamp', columns='symbol', values='close')

df = df.join(prices_pivoted, how='left')
df['QQQ'] = df['QQQ'].ffill()
df['SPY'] = df['SPY'].ffill()

last_real_qqq = prices_pivoted['QQQ'].dropna().index.max()
last_real_spy = prices_pivoted['SPY'].dropna().index.max()
print(f"Last real QQQ price: {last_real_qqq}")
print(f"Last real SPY price: {last_real_spy}")
print(f"Signal file ends:    {df.index.max()}")

PRICE_END = min(last_real_qqq, last_real_spy)
df = df.loc[:PRICE_END]
print(f"df trimmed to: {df.index.min()} -> {df.index.max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC Part 1: The Core Comparison (Overall Alpha)

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRADING_DAYS = 252

# --- 2. GENERATE STRATEGY VS BENCHMARK EQUITY ---
def run_simple_backtest(data, start_val):
    portfolio = [start_val]
    active_asset = "START"
    cost = 0.0005  # 5bps

    qqq_arr = data['QQQ'].values
    spy_arr = data['SPY'].values
    regime_arr = data['predicted_regime'].values

    current_val = start_val

    for i in range(1, len(data)):
        regime = regime_arr[i - 1]

        if regime == 0:
            target = "QQQ"
        elif regime == 1:
            target = "SPY"
        else:
            target = "CASH"

        if target == "QQQ":
            ret = (qqq_arr[i] / qqq_arr[i - 1]) - 1
        elif target == "SPY":
            ret = (spy_arr[i] / spy_arr[i - 1]) - 1
        else:
            ret = 0.0

        friction = cost if target != active_asset else 0.0
        current_val *= (1 + ret - friction)
        portfolio.append(current_val)
        active_asset = target

    return portfolio


# 1. Run the Backtest
df['Strategy_Value'] = run_simple_backtest(df, INITIAL_CAPITAL)
df['Benchmark_Value'] = (df['SPY'] / df['SPY'].iloc[0]) * INITIAL_CAPITAL


# 2. METRICS CALCULATION ENGINE (using DAILY returns)
def calculate_comprehensive_metrics_from_daily(daily_equity):
    # Time metrics
    n_days = daily_equity.shape[0]
    years = n_days / TRADING_DAYS

    # Returns series
    daily_ret = daily_equity.pct_change().dropna()

    # Initial / final
    initial_val = daily_equity.iloc[0]
    final_val = daily_equity.iloc[-1]
    total_return = (final_val / initial_val) - 1

    # CAGR from equity path
    cagr = (final_val / initial_val) ** (1 / years) - 1 if years > 0 else np.nan

    # Volatility and Sharpe (mean of daily returns)
    sigma = daily_ret.std()
    vol = sigma * np.sqrt(TRADING_DAYS)
    mean_ret = daily_ret.mean()
    sharpe = (mean_ret / sigma) * np.sqrt(TRADING_DAYS) if sigma != 0 else 0.0

    # Downside risk (Sortino)
    downside = daily_ret[daily_ret < 0]
    downside_sigma = downside.std()
    sortino = (mean_ret / downside_sigma) * np.sqrt(TRADING_DAYS) if downside_sigma not in [0, np.nan] else 0.0

    # Drawdown on daily equity
    rolling_max = daily_equity.cummax()
    drawdown = (daily_equity / rolling_max) - 1
    max_dd = drawdown.min()

    return {
        "Initial Value": initial_val,
        "Final Value": final_val,
        "Total Return": total_return,
        "CAGR": cagr,
        "Ann. Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd
    }


# Build DAILY equity curves from intraday path
daily_strat_eq = df['Strategy_Value'].resample('D').last().dropna()
daily_bench_eq = df['Benchmark_Value'].resample('D').last().dropna()

# Calculate for both (using daily equity)
strat_m = calculate_comprehensive_metrics_from_daily(daily_strat_eq)
bench_m = calculate_comprehensive_metrics_from_daily(daily_bench_eq)

# 3. DISPLAY PERFORMANCE TABLE
metrics_df = pd.DataFrame([strat_m, bench_m], index=['V4 Strategy', 'S&P 500 Benchmark']).T
metrics_df['Alpha / Improvement'] = metrics_df['V4 Strategy'] - metrics_df['S&P 500 Benchmark']

print("\n" + "=" * 50)
print("📊 V4 STRATEGY PERFORMANCE SUMMARY")
print("=" * 50)
pd.options.display.float_format = '{:,.2f}'.format

# Format percentages for readability (work on a copy to avoid dtype warnings)
display_df = metrics_df.copy()
for pct_metric in ['Total Return', 'CAGR', 'Ann. Volatility', 'Max Drawdown']:
    display_df.loc[pct_metric] = metrics_df.loc[pct_metric].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "NA"
    )

print(display_df)

# 4. PLOT RESULTS (full-resolution equity, e.g., hourly)
plt.figure(figsize=(16, 7))
plt.plot(df['Strategy_Value'], label=f"V4 Strategy (Sharpe: {strat_m['Sharpe Ratio']:.2f})", color='navy', linewidth=2)
plt.plot(df['Benchmark_Value'], label=f"S&P 500 (Sharpe: {bench_m['Sharpe Ratio']:.2f})", color='grey', alpha=0.5)
plt.yscale('log')
plt.title("V4 Event-Driven Engine: Equity Growth vs S&P 500", fontsize=16)
plt.ylabel("Portfolio Value (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Part 2: The "Crisis Zoom" (COVID-19 and 2022 Bear Market)

# COMMAND ----------

def plot_crisis_period(data, start_date, end_date, title):
    zoom = data.loc[start_date:end_date].copy()
    zoom['Strat_Norm'] = (zoom['Strategy_Value'] / zoom['Strategy_Value'].iloc[0]) * 100
    zoom['Bench_Norm'] = (zoom['Benchmark_Value'] / zoom['Benchmark_Value'].iloc[0]) * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Performance
    ax1.plot(zoom['Strat_Norm'], label='Strategy', color='navy', linewidth=2.5)
    ax1.plot(zoom['Bench_Norm'], label='S&P 500', color='firebrick', linestyle='--', alpha=0.7)
    ax1.set_title(title, fontsize=15)
    ax1.legend()
    
    # Bottom: Regime Allocation (What was the model thinking?)
    ax2.fill_between(zoom.index, 0, 1, where=(zoom['predicted_regime'] == 0), color='green', alpha=0.3, label='Expansion (QQQ)')
    ax2.fill_between(zoom.index, 0, 1, where=(zoom['predicted_regime'] == 1), color='yellow', alpha=0.3, label='Slowdown (SPY)')
    ax2.fill_between(zoom.index, 0, 1, where=(zoom['predicted_regime'] == 2), color='red', alpha=0.3, label='Recession (CASH)')
    ax2.set_yticks([])
    ax2.set_ylabel("Regime")
    
    plt.tight_layout()
    plt.show()

# Stress Test 1: COVID-19 Crash (The Speed Test)
plot_crisis_period(df, '2020-01-01', '2020-07-01', "Crisis Audit 1: COVID-19 Liquidity Shock")

# Stress Test 2: 2022 Inflationary Bear Market (The Persistence Test)
plot_crisis_period(df, '2022-01-01', '2022-12-31', "Crisis Audit 2: 2022 Rate Hike Cycle")

# COMMAND ----------

# MAGIC %md
# MAGIC Part 3: Drawdown and Risk Metrics

# COMMAND ----------

# Calculate Drawdowns
def get_drawdown(series):
    rolling_max = series.cummax()
    return (series / rolling_max) - 1

df['Strat_DD'] = get_drawdown(df['Strategy_Value'])
df['Bench_DD'] = get_drawdown(df['Benchmark_Value'])

plt.figure(figsize=(16, 5))
plt.fill_between(df.index, df['Strat_DD'], 0, color='navy', alpha=0.4, label='Strategy DD')
plt.fill_between(df.index, df['Bench_DD'], 0, color='red', alpha=0.2, label='S&P 500 DD')
plt.title("Risk Audit: Drawdown Comparison (V4 vs Benchmark)")
plt.ylabel("Percentage Decline")
plt.legend()
plt.show()

print(f"Max Strategy Drawdown: {df['Strat_DD'].min():.2%}")
print(f"Max S&P 500 Drawdown: {df['Bench_DD'].min():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC V3 Inclusion

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from delta.tables import DeltaTable

TRADING_DAYS = 252
INITIAL_CAPITAL = 5_000_000

# 1. LOAD V3 DAILY RETURNS
V3_PATH = (
    "/Workspace/Users/pawan.jatale@mubama.group/"
    "Global-Dynamic-Factor-Rotation/Global Factors/notebooks/"
    "v3_daily_returns.csv"
)
v3 = (
    pd.read_csv(V3_PATH, parse_dates=["Date"])
    .set_index("Date")
    .sort_index()
)
v3["v3_daily_return"] = v3["v3_daily_return"].fillna(0.0)

# 2. V4 HOURLY TO DAILY (compound intraday to daily)
if "Strategy_Value" not in df.columns:
    df["Strategy_Value"] = run_simple_backtest(df, INITIAL_CAPITAL)

v4_hourly = df[["Strategy_Value"]].copy()
v4_hourly["v4_hourly_return"] = v4_hourly["Strategy_Value"].pct_change().fillna(0.0)
v4_daily = (
    v4_hourly["v4_hourly_return"]
    .resample("1D")
    .apply(lambda x: (1 + x).prod() - 1)
    .dropna()
    .to_frame(name="v4_daily_return")
)

# 3. ── FIX: V4 drives the window — V3 is extended via forward-fill beyond Jun 2025 ──
start_date = v4_daily.index.min()
end_date   = v4_daily.index.max()          # was: min(v3.index.max(), v4_daily.index.max())

v4_aligned = v4_daily.loc[start_date:end_date].copy()

# Reindex V3 to the full V4 window; forward-fill where V3 has no new data (post Jun 2025)
# NOTE: V3 NAV will flatline after Jun 2025 — this is correct, V3 has no new signals
v3_aligned = (
    v3
    .reindex(v4_aligned.index)
    .ffill()
    .fillna(0.0)
)

# 4. REBUILD NAVS FROM COMMON START, SAME CAPITAL
v3_aligned["V3_Value"] = INITIAL_CAPITAL * (1 + v3_aligned["v3_daily_return"]).cumprod()
v4_aligned["V4_Value"] = INITIAL_CAPITAL * (1 + v4_aligned["v4_daily_return"]).cumprod()

# 5. SPY NAV FROM SAME START
# ── FIX: pull spy_returns from v3_aligned (now covers full window) ──
spy_returns = v3_aligned["benchmark_daily_return"].fillna(0.0)  # was: v3.loc[start_date:end_date, ...]
spy_nav     = INITIAL_CAPITAL * (1 + spy_returns).cumprod()

# 6. BUILD COMPARISON DF — reindex to v4 trading days (V4 is now the spine)
# ── FIX: spine is v4_aligned.index, not v3_aligned.index ──
comparison_df = (
    pd.DataFrame({
        "V3_Return": v3_aligned["v3_daily_return"],
        "V4_Return": v4_aligned["v4_daily_return"],
        "SPY_Return": spy_returns,
        "V3_Value":  v3_aligned["V3_Value"],
        "V4_Value":  v4_aligned["V4_Value"],
        "SPY_Value": spy_nav,
    })
    .reindex(v4_aligned.index)          # was: .reindex(v3_aligned.index)
    .ffill()
    .dropna()
)

# Deduplicate on date before any write
comparison_df = comparison_df[~comparison_df.index.duplicated(keep="last")]

print(f"✅ Comparison DF coverage: {comparison_df.index.min().date()} → {comparison_df.index.max().date()}")
print(f"   V3 data ends: {v3.index.max().date()} (flatlined beyond this point)")

# 7. METRICS
def compute_metrics(values: pd.Series, returns: pd.Series = None):
    if returns is None:
        returns = values.pct_change().dropna()
    returns = returns[returns != 0]

    n_days = len(returns)
    years  = n_days / TRADING_DAYS if n_days > 0 else np.nan

    initial_val  = values.iloc[0]
    final_val    = values.iloc[-1]
    total_return = (final_val / initial_val) - 1 if initial_val != 0 else np.nan
    cagr         = (final_val / initial_val) ** (1 / years) - 1 if (initial_val != 0 and years > 0) else np.nan

    sigma    = returns.std()
    mean_ret = returns.mean()
    vol      = sigma * np.sqrt(TRADING_DAYS)
    sharpe   = (mean_ret / sigma) * np.sqrt(TRADING_DAYS) if sigma not in [0, np.nan] else 0.0

    rolling_max = values.cummax()
    drawdown    = values / rolling_max - 1
    max_dd      = drawdown.min()

    return {
        "final_value":    final_val,
        "total_return":   total_return,
        "cagr":           cagr,
        "ann_volatility": vol,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
    }

v3_metrics  = compute_metrics(comparison_df["V3_Value"], comparison_df["V3_Return"])
v4_metrics  = compute_metrics(comparison_df["V4_Value"], comparison_df["V4_Return"])
spy_metrics = compute_metrics(comparison_df["SPY_Value"], comparison_df["SPY_Return"])

metrics_df = pd.DataFrame(
    [v3_metrics, v4_metrics, spy_metrics],
    index=["V3 (Daily Engine)", "V4 (Hourly Engine)", "SPY (Benchmark)"]
)

# 8. DISPLAY METRICS TABLE
print("\n📊 V3 vs V4 vs SPY PERFORMANCE METRICS")

def _fmt_value(x, col):
    if pd.isna(x):                                                              return "NA"
    if col == "final_value":                                                    return f"${x:,.0f}"
    if col in ["total_return", "cagr", "ann_volatility", "max_drawdown"]:      return f"{x:.2%}"
    if col == "sharpe":                                                         return f"{x:.2f}"
    return f"{x}"

display(
    metrics_df.apply(
        lambda col: col.map(lambda v: _fmt_value(v, col.name))
    )
)

# 9. PLOT
plt.figure(figsize=(16, 7))
plt.plot(
    comparison_df.index, comparison_df["V3_Value"],
    label=f"V3 (Sharpe {v3_metrics['sharpe']:.2f})",
    linewidth=2, color="#08519c"
)
plt.plot(
    comparison_df.index, comparison_df["V4_Value"],
    label=f"V4 (Sharpe {v4_metrics['sharpe']:.2f})",
    linewidth=2, color="#e6550d"
)
plt.plot(
    comparison_df.index, comparison_df["SPY_Value"],
    label=f"SPY (Sharpe {spy_metrics['sharpe']:.2f})",
    linewidth=2, color="grey", alpha=0.7
)
plt.yscale("log")
plt.title("V3 vs V4 vs SPY Strategy Comparison (Full Window through Feb 2026)", fontsize=16)
plt.ylabel("Portfolio Value (Log Scale)")
plt.legend()
plt.grid(True, which="both", alpha=0.25)
plt.tight_layout()
plt.show()

# 10. EXPORT TO DELTA TABLES
catalog = "pawan_quant_research"
schema  = "default"
comparison_table = f"{catalog}.{schema}.gdfr_v3_v4_spy_timeseries"
metrics_table    = f"{catalog}.{schema}.gdfr_v3_v4_spy_metrics"

comparison_df_reset = (
    comparison_df
    .reset_index()
    .rename(columns={"index": "date", "Date": "date"})
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

metrics_df_reset = (
    metrics_df
    .reset_index()
    .rename(columns={"index": "strategy"})
)

print(f"\n✅ Rows to write — timeseries: {len(comparison_df_reset)}, metrics: {len(metrics_df_reset)}")
print(f"   Date range: {comparison_df_reset['date'].min()} → {comparison_df_reset['date'].max()}")

comparison_sdf = spark.createDataFrame(comparison_df_reset)
metrics_sdf    = spark.createDataFrame(metrics_df_reset)

(comparison_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(comparison_table))

(metrics_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(metrics_table))

try:
    spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    spark.sql(f"VACUUM {comparison_table} RETAIN 0 HOURS")
    spark.sql(f"VACUUM {metrics_table} RETAIN 0 HOURS")
    spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "true")
except Exception:
    spark.sql(f"VACUUM {comparison_table} RETAIN 168 HOURS")
    spark.sql(f"VACUUM {metrics_table} RETAIN 168 HOURS")
    print("ℹ️  Vacuumed with default 168h retention (cluster policy restricts 0h override)")

print(f"\n✅ Tables written successfully:")
print(f"   {comparison_table}  →  {spark.table(comparison_table).count()} rows")
print(f"   {metrics_table}     →  {spark.table(metrics_table).count()} rows")

dup_count = (
    spark.table(comparison_table)
    .groupBy("date").count()
    .filter("count > 1")
    .count()
)
if dup_count > 0:
    print(f"\n⚠️  WARNING: {dup_count} duplicate date(s) still found in {comparison_table}")
else:
    print("\n✅ No duplicate dates — table is clean for dashboard use")

# COMMAND ----------

# At the top of your performance analysis notebook, add:
#%run "/Workspace/Users/pawan.jatale@mubama.group/V4_Regime_Strategy/04_Backtesting/07_Event_Driven_Backtest"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reload V3 daily returns
V3_PATH = "/Workspace/Users/pawan.jatale@mubama.group/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_daily_returns.csv"
v3 = pd.read_csv(V3_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
v3["v3_daily_return"] = v3["v3_daily_return"].fillna(0.0)

# Assume df exists and has 'Strategy_Value' — if not, you must run backtest first!
if "Strategy_Value" not in df.columns:
    raise ValueError("df['Strategy_Value'] missing → run the backtest / main comparison cell first")

# V4 → daily returns
v4_hourly = df[["Strategy_Value"]].copy()
v4_hourly["v4_hourly_return"] = v4_hourly["Strategy_Value"].pct_change().fillna(0.0)
v4_daily = (
    v4_hourly["v4_hourly_return"]
    .resample("1D")
    .apply(lambda x: (1 + x).prod() - 1)
    .to_frame("v4_daily_return")
)

# ── FIX: V4 drives the window — V3 is extended via forward-fill beyond Jun 2025 ──
start_date = v4_daily.index.min()
end_date   = v4_daily.index.max()          # was: min(v3.index.max(), v4_daily.index.max())

v4_aligned = v4_daily.loc[start_date:end_date].copy()

# Reindex V3 to the full V4 window; forward-fill where V3 has no new data (post Jun 2025)
# NOTE: V3 NAV will flatline after Jun 2025 — this is correct, V3 has no new signals
v3_aligned = (
    v3
    .reindex(v4_aligned.index)
    .ffill()
    .fillna(0.0)
)

# NAVs
INITIAL_CAPITAL = 5_000_000
v3_nav  = INITIAL_CAPITAL * (1 + v3_aligned["v3_daily_return"]).cumprod()
v4_nav  = INITIAL_CAPITAL * (1 + v4_aligned["v4_daily_return"]).cumprod()
spy_nav = INITIAL_CAPITAL * (1 + v3_aligned["benchmark_daily_return"].fillna(0.0)).cumprod()

# Comparison DF (NAV only)
nav_df = pd.DataFrame({
    "V3_Value":  v3_nav,
    "V4_Value":  v4_nav,
    "SPY_Value": spy_nav
}).dropna()

print(f"✅ NAV coverage: {nav_df.index.min().date()} → {nav_df.index.max().date()}")
print(f"   V3 data ends: {v3.index.max().date()} (flatlined beyond this point)")

# ── Monthly returns ───────────────────────────────────────
monthly = nav_df.resample('ME').last()
monthly_returns = monthly.pct_change().dropna() * 100

monthly_returns['Year']  = monthly_returns.index.year
monthly_returns['Month'] = monthly_returns.index.month

def pivot_returns(col):
    piv = monthly_returns.pivot(index='Year', columns='Month', values=col).round(2)
    piv.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    return piv

v3_m  = pivot_returns('V3_Value')
v4_m  = pivot_returns('V4_Value')
spy_m = pivot_returns('SPY_Value')

# Heatmap function
def plot_heatmap(data, title, vmin=-20, vmax=20, cmap="RdYlGn"):
    plt.figure(figsize=(14, max(6, len(data) * 0.45)))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap=cmap, center=0,
        vmin=vmin, vmax=vmax, linewidths=0.6, linecolor='white',
        cbar_kws={'label': 'Monthly Return (%)'}, annot_kws={"size": 10}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("Year")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.show()

plot_heatmap(v3_m,  "V3 Monthly Returns (%)", vmin=-20, vmax=20)
plot_heatmap(v4_m,  "V4 Monthly Returns (%)", vmin=-20, vmax=20)
plot_heatmap(spy_m, "SPY Monthly Returns (%)", vmin=-15, vmax=15)

# Optional difference — only compute where both grids share the same shape
if v3_m.shape == v4_m.shape:
    diff = v4_m - v3_m
    plot_heatmap(diff, "V4 − V3 Monthly Difference (%)", cmap="coolwarm", vmin=-10, vmax=10)
else:
    # After the fix, V4 will have Jan/Feb 2026 rows that V3 does not — shapes will differ
    # Compute difference only over the common date range
    common_years = v3_m.index.intersection(v4_m.index)
    diff = v4_m.loc[common_years] - v3_m.loc[common_years]
    plot_heatmap(diff, "V4 − V3 Monthly Difference (%) — Common Period Only", cmap="coolwarm", vmin=-10, vmax=10)

# COMMAND ----------

# ── Weekly returns ────────────────────────────────────────
weekly = nav_df.resample('W-FRI').last()          # End-of-week (Friday close)
weekly_returns = weekly.pct_change().dropna() * 100

weekly_returns['Year']  = weekly_returns.index.year
weekly_returns['Week']  = weekly_returns.index.isocalendar().week

# Pivot: Years as rows, ISO weeks (1–52/53) as columns
def pivot_weekly_returns(col):
    piv = (
        weekly_returns
        .pivot(index='Year', columns='Week', values=col)
        .round(2)
    )
    return piv

v3_w  = pivot_weekly_returns('V3_Value')
v4_w  = pivot_weekly_returns('V4_Value')
spy_w = pivot_weekly_returns('SPY_Value')

# ── 1. Nice printed weekly performance tables ───────────────────────────────
def print_performance_table(df, strategy_name):
    print(f"\n{strategy_name} – Weekly Returns (%)")
    print("-" * 80)
    print(df.to_string(float_format="{:,.2f}".format))
    print("\n")

print_performance_table(v3_w,  "V3 Strategy")
print_performance_table(v4_w,  "V4 Strategy (Hourly Engine)")
print_performance_table(spy_w, "SPY Benchmark")

# ── 2. Optional: Weekly heatmap (similar style to monthly) ───────────────────
def plot_weekly_heatmap(data, title, vmin=-10, vmax=10, cmap="RdYlGn"):
    plt.figure(figsize=(16, max(7, len(data) * 0.5)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.4,
        linecolor='white',
        cbar_kws={'label': 'Weekly Return (%)'},
        annot_kws={"size": 8}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("Year")
    plt.xlabel("ISO Week of Year")
    plt.tight_layout()
    plt.show()

# Uncomment if you want visuals
plot_weekly_heatmap(v3_w,  "V3 Strategy – Weekly Returns (%)", vmin=-12, vmax=12)
plot_weekly_heatmap(v4_w,  "V4 Strategy – Weekly Returns (%)", vmin=-12, vmax=12)
plot_weekly_heatmap(spy_w, "SPY Benchmark – Weekly Returns (%)", vmin=-10, vmax=10)

# Optional: V4 vs V3 difference heatmap
if v3_w.shape == v4_w.shape:
    diff_w = v4_w - v3_w
    plot_weekly_heatmap(diff_w, "Weekly Return Difference: V4 − V3 (%)",
                        cmap="coolwarm", vmin=-8, vmax=8)

# COMMAND ----------

# MAGIC %md
# MAGIC Save Weekly Returns for Dashboard
# MAGIC

# COMMAND ----------

# ── Save Weekly Returns for Dashboard ─────────────────────────────────────────

# 1. Build tidy long-form DataFrame directly from weekly_returns
#    (it already carries Year, Week, and the Friday date as index)

def build_weekly_long(weekly_ret_df):
    """
    Melt V3_Value / V4_Value / SPY_Value into a dashboard-ready long table:

        week_end_date | year | iso_week | Strategy | ReturnPct
    """
    df = weekly_ret_df[['V3_Value', 'V4_Value', 'SPY_Value', 'Year', 'Week']].copy()
    df['week_end_date'] = df.index.normalize()   # keep date only (no time)

    # ── FIX: Arrow/Spark doesn't support pandas UInt32 from isocalendar().week ──
    # Year is usually fine, Week is often UInt32 → cast both to signed ints.
    df['Year'] = df['Year'].astype('int64')
    df['Week'] = df['Week'].astype('int64')
    # ──────────────────────────────────────────────────────────────────────────

    long = df.melt(
        id_vars=['week_end_date', 'Year', 'Week'],
        value_vars=['V3_Value', 'V4_Value', 'SPY_Value'],
        var_name='Strategy',
        value_name='ReturnPct'
    )

    long['Strategy'] = long['Strategy'].str.replace('_Value', '', regex=False)
    long = (
        long
        .rename(columns={'Year': 'year', 'Week': 'iso_week'})
        .dropna(subset=['ReturnPct'])
        .sort_values(['week_end_date', 'Strategy'])
        .reset_index(drop=True)
    )
    return long


weekly_long = build_weekly_long(weekly_returns)

print(f"✅ weekly_long shape : {weekly_long.shape}")
print(f"   Strategies        : {weekly_long['Strategy'].unique().tolist()}")
print(f"   Date range        : {weekly_long['week_end_date'].min()} → {weekly_long['week_end_date'].max()}")
display(weekly_long.head(12))


# 2. Save to Delta table (mirrors v4_monthly_returns pattern already in notebook)
CATALOG = "pawan_quant_research"
SCHEMA  = "mubama_v4"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

weekly_sdf = spark.createDataFrame(weekly_long)
weekly_sdf.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.v4_weekly_returns"
)
print(f"✅ Delta table saved → {CATALOG}.{SCHEMA}.v4_weekly_returns")


# 3. CSV backup to Volume (handy for external tools / quick inspection)
CSV_OUT = "/Volumes/pawan_quant_research_1/default/market_data/weekly_returns_v3_v4_spy.csv"
weekly_long.to_csv(CSV_OUT, index=False)
print(f"✅ CSV written       → {CSV_OUT}")


# COMMAND ----------

# Replace your existing pivot with this transposed version

def pivot_and_transpose(col):
    piv = (
        weekly_returns                     # or monthly_returns
        .pivot(index='Year', columns='Week', values=col)   # or 'Month'
        .round(2)
        .T                                 # ← transpose here
    )
    return piv

v3_w_t  = pivot_and_transpose('V3_Value')
v4_w_t  = pivot_and_transpose('V4_Value')
spy_w_t = pivot_and_transpose('SPY_Value')

# Then use the transposed versions in heatmaps
plot_weekly_heatmap(v4_w_t, "V4 Strategy – Weekly Returns (%) (Transposed)", vmin=-12, vmax=12)

# COMMAND ----------

def plot_heatmap_clean(data, title, vmin=-12, vmax=12, cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=(max(10, data.shape[1]*0.6), max(8, data.shape[0]*0.45)))
    
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",                  # ← fewer decimals = less visual noise
        cmap=cmap,
        center=0,
        vmin=vmin, vmax=vmax,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'Return (%)', 'shrink': 0.7},
        annot_kws={"size": 8},      # smaller font
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    
    # Rotate labels to save space
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # Add title and tight layout
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel(data.columns.name or "Period", fontsize=11)
    plt.ylabel(data.index.name or "Year", fontsize=11)
    plt.tight_layout()
    plt.show()

plot_heatmap_clean(v4_w_t, "V4 Strategy – Weekly Returns (%) (Transposed)", vmin=-12, vmax=12)

# COMMAND ----------

def plot_heatmap_clean(data, title, vmin=-12, vmax=12, cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=(max(10, data.shape[1]*0.6), max(8, data.shape[0]*0.45)))
    
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",                  # ← fewer decimals = less visual noise
        cmap=cmap,
        center=0,
        vmin=vmin, vmax=vmax,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'Return (%)', 'shrink': 0.7},
        annot_kws={"size": 8},      # smaller font
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    
    # Rotate labels to save space
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # Add title and tight layout
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel(data.columns.name or "Period", fontsize=11)
    plt.ylabel(data.index.name or "Year", fontsize=11)
    plt.tight_layout()
    plt.show()

plot_heatmap_clean(v4_w_t, "V4 Strategy – Weekly Returns (%) (Transposed)", vmin=-12, vmax=12)


# COMMAND ----------

weekly_returns_melt = weekly_returns.reset_index().melt(
    id_vars=['Year'], value_vars=['V3_Value','V4_Value','SPY_Value'],
    var_name='Strategy', value_name='Weekly Return (%)'
)

plt.figure(figsize=(14,7))
sns.boxplot(data=weekly_returns_melt, x='Year', y='Weekly Return (%)', hue='Strategy')
plt.title("Distribution of Weekly Returns by Year")
plt.axhline(0, color='black', lw=0.8, alpha=0.5)
plt.grid(axis='y', alpha=0.3)
plt.show()


# COMMAND ----------

# ── RETURNS EXPORT: Monthly / Quarterly / Yearly for V3, V4, SPY ──
# Requires comparison_df to be built (run V3 inclusion cell first)

print("Building periodic returns export...")

# ── STEP 1: Build daily returns from comparison_df ──
daily_returns = pd.DataFrame({
    "V3":  comparison_df["V3_Value"].pct_change(),
    "V4":  comparison_df["V4_Value"].pct_change(),
    "SPY": comparison_df["SPY_Value"].pct_change(),
}).dropna()

# ── STEP 2: Compounding function ──
def compound_returns(ret_series, freq):
    """Resample daily returns to compounded periodic returns."""
    return ret_series.resample(freq).apply(lambda x: (1 + x).prod() - 1)

# ── STEP 3: Monthly returns ──
monthly = pd.DataFrame({
    "V3_Return":  compound_returns(daily_returns["V3"],  "ME"),
    "V4_Return":  compound_returns(daily_returns["V4"],  "ME"),
    "SPY_Return": compound_returns(daily_returns["SPY"], "ME"),
})
monthly.index.name = "period_end"
monthly["period"]  = monthly.index.to_period("M").astype(str)
monthly["year"]    = monthly.index.year
monthly["month"]   = monthly.index.month
monthly = monthly[["period", "year", "month", "V3_Return", "V4_Return", "SPY_Return"]]
monthly["V4_alpha_vs_SPY"] = monthly["V4_Return"] - monthly["SPY_Return"]
monthly["V4_alpha_vs_V3"]  = monthly["V4_Return"] - monthly["V3_Return"]

# ── STEP 4: Quarterly returns ──
quarterly = pd.DataFrame({
    "V3_Return":  compound_returns(daily_returns["V3"],  "QE"),
    "V4_Return":  compound_returns(daily_returns["V4"],  "QE"),
    "SPY_Return": compound_returns(daily_returns["SPY"], "QE"),
})
quarterly.index.name = "period_end"
quarterly["period"]  = quarterly.index.to_period("Q").astype(str)
quarterly["year"]    = quarterly.index.year
quarterly["quarter"] = quarterly.index.quarter
quarterly = quarterly[["period", "year", "quarter", "V3_Return", "V4_Return", "SPY_Return"]]
quarterly["V4_alpha_vs_SPY"] = quarterly["V4_Return"] - quarterly["SPY_Return"]
quarterly["V4_alpha_vs_V3"]  = quarterly["V4_Return"] - quarterly["V3_Return"]

# ── STEP 5: Yearly returns ──
yearly = pd.DataFrame({
    "V3_Return":  compound_returns(daily_returns["V3"],  "YE"),
    "V4_Return":  compound_returns(daily_returns["V4"],  "YE"),
    "SPY_Return": compound_returns(daily_returns["SPY"], "YE"),
})
yearly.index.name = "period_end"
yearly["period"] = yearly.index.year.astype(str)
yearly["year"]   = yearly.index.year
yearly = yearly[["period", "year", "V3_Return", "V4_Return", "SPY_Return"]]
yearly["V4_alpha_vs_SPY"] = yearly["V4_Return"] - yearly["SPY_Return"]
yearly["V4_alpha_vs_V3"]  = yearly["V4_Return"] - yearly["V3_Return"]

# ── STEP 6: Save CSVs ──
monthly_path   = f"{CSV_OUT_BASE}/returns_monthly_v3_v4_spy.csv"
quarterly_path = f"{CSV_OUT_BASE}/returns_quarterly_v3_v4_spy.csv"
yearly_path    = f"{CSV_OUT_BASE}/returns_yearly_v3_v4_spy.csv"

monthly.to_csv(monthly_path,   index=True)
quarterly.to_csv(quarterly_path, index=True)
yearly.to_csv(yearly_path,     index=True)

print(f"Monthly   CSV: {monthly_path}   ({len(monthly)} rows)")
print(f"Quarterly CSV: {quarterly_path} ({len(quarterly)} rows)")
print(f"Yearly    CSV: {yearly_path}    ({len(yearly)} rows)")

# ── STEP 7: Print preview tables ──
def pct_fmt(df, cols):
    out = df.copy().astype(object)
    for c in cols:
        out[c] = out[c].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "NA")
    return out

ret_cols = ["V3_Return", "V4_Return", "SPY_Return", "V4_alpha_vs_SPY", "V4_alpha_vs_V3"]

print("\n--- YEARLY RETURNS ---")
print(pct_fmt(yearly, ret_cols).to_string(index=False))

print("\n--- QUARTERLY RETURNS (last 8 quarters) ---")
print(pct_fmt(quarterly.tail(8), ret_cols).to_string(index=False))

print("\n--- MONTHLY RETURNS (last 12 months) ---")
print(pct_fmt(monthly.tail(12), ret_cols).to_string(index=False))

# ── STEP 8: Save to Delta for dashboard consumption ──
for label, pdf, freq in [
    ("monthly",   monthly,   "monthly"),
    ("quarterly", quarterly, "quarterly"),
    ("yearly",    yearly,    "yearly"),
]:
    sdf = spark.createDataFrame(pdf.reset_index())
    sdf.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"pawan_quant_research_1.default.gold_bt_returns_{freq}")
    print(f"Delta saved: gold_bt_returns_{freq}")

print("\nAll returns exported successfully.")