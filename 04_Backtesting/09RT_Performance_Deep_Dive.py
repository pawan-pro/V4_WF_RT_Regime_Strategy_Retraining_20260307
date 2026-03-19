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
def calculate_comprehensive_metrics_from_daily(daily_equity: pd.Series):
    # Calendar-year based horizon
    start_date = daily_equity.index[0]
    end_date   = daily_equity.index[-1]
    years = (end_date - start_date).days / 365.25

    daily_ret = daily_equity.pct_change().dropna()

    initial_val = INITIAL_CAPITAL
    final_val = daily_equity.iloc[-1]
    total_return = (final_val / INITIAL_CAPITAL) - 1  # anchored to INITIAL_CAPITAL

    # CAGR with calendar-year horizon
    cagr = (final_val / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan

    sigma = daily_ret.std()
    vol = sigma * np.sqrt(TRADING_DAYS)
    mean_ret = daily_ret.mean()
    sharpe = (mean_ret / sigma) * np.sqrt(TRADING_DAYS) if sigma != 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    downside_sigma = downside.std()
    sortino = (mean_ret / downside_sigma) * np.sqrt(TRADING_DAYS) if downside_sigma not in [0, np.nan] else 0.0

    rolling_max = daily_equity.cummax()
    drawdown = (daily_equity / rolling_max) - 1
    max_dd = drawdown.min()

    return {
        "Initial Value": INITIAL_CAPITAL,
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
metrics_df = pd.DataFrame(
    [strat_m, bench_m],
    index=['V4 Strategy', 'S&P 500 Benchmark']
).T
metrics_df['Alpha / Improvement'] = metrics_df['V4 Strategy'] - metrics_df['S&P 500 Benchmark']

print("\n" + "=" * 50)
print("📊 V4 STRATEGY PERFORMANCE SUMMARY")
print("=" * 50)
pd.options.display.float_format = '{:,.2f}'.format

# Format percentages for readability (work on a copy to avoid dtype warnings)
display_df = metrics_df.copy().astype(object)
for pct_metric in ['Total Return', 'CAGR', 'Ann. Volatility', 'Max Drawdown']:
    if pct_metric in display_df.index:
        display_df.loc[pct_metric] = display_df.loc[pct_metric].apply(
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

# ── V3 vs V4 vs SPY COMPARISON ──
# Trimmed to V3 data end (Q2 2025) for fair comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRADING_DAYS    = 252
INITIAL_CAPITAL = 5_000_000

# ── 1. Load V3 daily returns ──
V3_PATH = (
    "/Workspace/Users/pawan.jatale@mubama.group/"
    "Global-Dynamic-Factor-Rotation/Global Factors/notebooks/"
    "v3_daily_returns.csv"
)
v3 = pd.read_csv(V3_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
v3["v3_daily_return"] = v3["v3_daily_return"].fillna(0.0)
V3_END = v3.index.max()
print(f"V3 data ends: {V3_END.date()}")

# ── 2. V4 hourly → daily ──
if "Strategy_Value" not in df.columns:
    df["Strategy_Value"] = run_simple_backtest(df, INITIAL_CAPITAL)

v4_hourly = df[["Strategy_Value"]].copy()
v4_hourly["v4_hourly_return"] = v4_hourly["Strategy_Value"].pct_change().fillna(0.0)
v4_daily_ret = (
    v4_hourly["v4_hourly_return"]
    .resample("1D")
    .apply(lambda x: (1 + x).prod() - 1)
    .dropna()
    .to_frame(name="v4_daily_return")
)
v4_daily_eq = INITIAL_CAPITAL * (1 + v4_daily_ret["v4_daily_return"]).cumprod()

# ── 3. Alignment window — cap at V3_END for fair comparison ──
start_date = v4_daily_ret.index.min()
end_date   = V3_END   # hard cap — no flatline distortion

v4_aligned_ret = v4_daily_ret.loc[start_date:end_date].copy()
v4_aligned_eq  = v4_daily_eq.loc[start_date:end_date].copy()

v3_aligned = (
    v3.reindex(v4_aligned_ret.index).ffill().fillna(0.0)
)
v3_daily_eq = INITIAL_CAPITAL * (1 + v3_aligned["v3_daily_return"]).cumprod()

spy_daily_prices  = df['SPY'].resample('D').last().dropna()
spy_aligned       = spy_daily_prices.reindex(v4_aligned_ret.index).ffill()
spy_daily_eq      = (spy_aligned / spy_aligned.iloc[0]) * INITIAL_CAPITAL

# ── 4. Build comparison df ──
comparison_df = pd.DataFrame({
    "V3_Value":  v3_daily_eq.reindex(v4_aligned_ret.index).ffill(),
    "V4_Value":  v4_aligned_eq,
    "SPY_Value": spy_daily_eq,
}).dropna()

print(f"Comparison window: {comparison_df.index.min().date()} -> {comparison_df.index.max().date()}")
print(f"Total days: {len(comparison_df)}")

# ── 5. Metrics engine — CAGR uses actual calendar days / 365.25 ──
def calc_metrics(daily_equity: pd.Series, label: str) -> dict:
    # Remove weekends/holidays gaps — use actual calendar span
    start      = daily_equity.index[0]
    end        = daily_equity.index[-1]
    cal_days   = (end - start).days
    years      = cal_days / 365.25            # always calendar-based

    daily_ret  = daily_equity.pct_change().dropna()
    final_val  = daily_equity.iloc[-1]
    total_ret  = (final_val / INITIAL_CAPITAL) - 1
    cagr       = (final_val / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan

    sigma      = daily_ret.std()
    vol        = sigma * np.sqrt(TRADING_DAYS)
    mean_ret   = daily_ret.mean()
    sharpe     = (mean_ret / sigma) * np.sqrt(TRADING_DAYS) if sigma != 0 else 0.0

    downside   = daily_ret[daily_ret < 0]
    ds_sigma   = downside.std()
    sortino    = (mean_ret / ds_sigma) * np.sqrt(TRADING_DAYS) if ds_sigma > 0 else 0.0

    roll_max   = daily_equity.cummax()
    max_dd     = ((daily_equity / roll_max) - 1).min()

    return {
        "Strategy":       label,
        "Final Value":    final_val,
        "Total Return":   total_ret,
        "CAGR":           cagr,
        "Ann. Volatility":vol,
        "Sharpe":         sharpe,
        "Sortino":        sortino,
        "Max Drawdown":   max_dd,
        "Years":          round(years, 2),
    }

v3_m   = calc_metrics(comparison_df["V3_Value"],  "V3")
v4_m   = calc_metrics(comparison_df["V4_Value"],  "V4")
spy_m  = calc_metrics(comparison_df["SPY_Value"], "SPY")

metrics_df = pd.DataFrame([v3_m, v4_m, spy_m]).set_index("Strategy")

# ── 6. Print clean table ──
print("\n--- V3 vs V4 vs SPY (through Q2 2025) ---")
fmt = metrics_df.copy().astype(object)
for col in ["Total Return", "CAGR", "Ann. Volatility", "Max Drawdown"]:
    fmt[col] = fmt[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "NA")
for col in ["Sharpe", "Sortino", "Years"]:
    fmt[col] = fmt[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "NA")
fmt["Final Value"] = fmt["Final Value"].apply(lambda x: f"${float(x):,.0f}")
print(fmt.to_string())

# ── 7. Plot (log scale, all three aligned through V3_END) ──
plt.figure(figsize=(16, 7))
plt.plot(comparison_df.index, comparison_df["V3_Value"],
         label=f"V3 (Sharpe {v3_m['Sharpe']:.2f})", linewidth=2, color="#08519c")
plt.plot(comparison_df.index, comparison_df["V4_Value"],
         label=f"V4 (Sharpe {v4_m['Sharpe']:.2f})", linewidth=2, color="darkorange")
plt.plot(comparison_df.index, comparison_df["SPY_Value"],
         label=f"SPY (Sharpe {spy_m['Sharpe']:.2f})", linewidth=2, color="grey", alpha=0.7)
plt.yscale("log")
plt.title(f"V3 vs V4 vs SPY Strategy Comparison\n(Common Window: {comparison_df.index.min().date()} to {comparison_df.index.max().date()})", fontsize=15)
plt.ylabel("Portfolio Value (Log Scale)")
plt.legend()
plt.grid(True, which="both", alpha=0.25)
plt.tight_layout()
plt.show()

# COMMAND ----------

print("SPY equity start date:", spy_daily_eq.index[0])
print("SPY equity end date:  ", spy_daily_eq.index[-1])
print("n_days:", spy_daily_eq.shape[0], "years:", spy_daily_eq.shape[0] / 252)
print("Total return:", (spy_daily_eq.iloc[-1] / INITIAL_CAPITAL) - 1)


# COMMAND ----------

# MAGIC %md
# MAGIC $10,000 Live Portfolio from Jan 1, 2026:

# COMMAND ----------

# In comparison notebook — verify what "SPY" column actually contains
print(df[['SPY', 'QQQ']].head(3))
print(f"Comparison SPY start: {df.index.min()}, end: {df.index.max()}")
print(f"Comparison initial SPY price: {df['SPY'].iloc[0]:.2f}")
# If SPY ~$200+ in 2016 → correct; if >$400 → you're likely loading QQQ

# COMMAND ----------



# COMMAND ----------

# ── NEW BLOCK: Last 6 Months Live Simulation (No Look-Ahead) ──
from datetime import timedelta

SIXM_CAPITAL = 10_000  # or reuse LIVE_CAPITAL if you prefer

# Define rolling 6M window ending at last available timestamp in df
live_end = df.index.max()
sixm_start = live_end - pd.DateOffset(months=6)

sixm_df = df.loc[sixm_start:live_end].copy()

if sixm_df.empty or len(sixm_df) < 2:
    print("=" * 60)
    print("⚠️  6M BLOCK PENDING — Not enough data for last 6 months")
    print("=" * 60)
    print(f"Data range in df: {df.index.min().date()} → {df.index.max().date()}")
else:
    print(f"✅ 6M window: {sixm_df.index.min().date()} → {sixm_df.index.max().date()}")
    print(f"   Total hours: {len(sixm_df):,}")

    # Portfolio simulation (same logic as main live block)
    sixm_portfolio = [SIXM_CAPITAL]
    sixm_spy_bench = [SIXM_CAPITAL]
    active_asset   = "START"
    cost           = 0.0005

    qqq_arr    = sixm_df['QQQ'].values
    spy_arr    = sixm_df['SPY'].values
    regime_arr = sixm_df['predicted_regime'].values
    spy_base   = spy_arr[0]
    current_val = SIXM_CAPITAL

    for i in range(1, len(sixm_df)):
        # NO LOOK-AHEAD: use previous bar's regime
        regime = regime_arr[i - 1]

        if regime == 0:
            target = "QQQ"
            ret    = (qqq_arr[i] / qqq_arr[i - 1]) - 1
        elif regime == 1:
            target = "SPY"
            ret    = (spy_arr[i] / spy_arr[i - 1]) - 1
        else:
            target = "CASH"
            ret    = 0.0

        friction     = cost if target != active_asset else 0.0
        current_val *= (1 + ret - friction)
        active_asset = target

        sixm_portfolio.append(current_val)
        sixm_spy_bench.append((spy_arr[i] / spy_base) * SIXM_CAPITAL)

    sixm_results = sixm_df.iloc[:len(sixm_portfolio)].copy()
    sixm_results['Portfolio_Value'] = sixm_portfolio
    sixm_results['SPY_Value']       = sixm_spy_bench

    # Metrics with calendar-year CAGR for the 6M window
    def sixm_metrics(series, label):
        daily   = series.resample('D').last().dropna()
        returns = daily.pct_change().dropna()

        if daily.empty or returns.empty:
            print(f"\n--- {label} ---")
            print("  Not enough data for metrics.")
            return np.nan, np.nan, np.nan

        start_date = daily.index[0]
        end_date   = daily.index[-1]
        years      = (end_date - start_date).days / 365.25

        total_r = (daily.iloc[-1] / SIXM_CAPITAL) - 1
        cagr    = (daily.iloc[-1] / SIXM_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
        vol     = returns.std() * np.sqrt(252)
        sharpe  = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        mdd     = ((daily / daily.cummax()) - 1).min()

        print(f"\n--- {label} (Last 6 Months) ---")
        print(f"  Start:         {start_date.date()}  →  End: {end_date.date()}")
        print(f"  Start Capital: ${SIXM_CAPITAL:,.0f}")
        print(f"  Current Value: ${daily.iloc[-1]:,.2f}")
        print(f"  Total Return:  {total_r:.2%}")
        print(f"  Ann. CAGR:     {cagr:.2%}")
        print(f"  Volatility:    {vol:.2%}")
        print(f"  Sharpe Ratio:  {sharpe:.2f}")
        print(f"  Max Drawdown:  {mdd:.2%}")

        return sharpe, total_r, cagr

    print("\n" + "=" * 60)
    print("  LAST 6 MONTHS PORTFOLIO REPORT")
    print("  (No look-ahead: regime[t-1] drives returns at t)")
    print("=" * 60)

    sixm_strat_sharpe, sixm_strat_ret, sixm_strat_cagr = sixm_metrics(
        sixm_results['Portfolio_Value'], "V4 Strategy (6M)"
    )
    sixm_bench_sharpe, sixm_bench_ret, sixm_bench_cagr = sixm_metrics(
        sixm_results['SPY_Value'], "SPY Benchmark (6M)"
    )

    # Plot with regime color bands at bottom (same style as main live block)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Equity curves
    ax1.plot(
        sixm_results['Portfolio_Value'],
        label=f'V4 Strategy (6M) [Sharpe: {sixm_strat_sharpe:.2f}]',
        color='navy', linewidth=2
    )
    ax1.plot(
        sixm_results['SPY_Value'],
        label=f'SPY Benchmark (6M) [Sharpe: {sixm_bench_sharpe:.2f}]',
        color='grey', linewidth=1.5, alpha=0.7
    )

    ax1.axhline(
        SIXM_CAPITAL, color='black', linestyle='--', alpha=0.4,
        label=f'Start (${SIXM_CAPITAL:,})'
    )

    ax1.set_title(
        f"Last 6 Months: ${SIXM_CAPITAL:,} Live Portfolio vs SPY",
        fontsize=14, fontweight='bold'
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.2)



    # Regime background colors
    ax2.fill_between(
        sixm_results.index, 0, 1,
        where=(sixm_results['predicted_regime'] == 0),
        color='green', alpha=0.4, label='Expansion (QQQ)'
    )
    ax2.fill_between(
        sixm_results.index, 0, 1,
        where=(sixm_results['predicted_regime'] == 1),
        color='gold', alpha=0.4, label='Slowdown (SPY)'
    )
    ax2.fill_between(
        sixm_results.index, 0, 1,
        where=(sixm_results['predicted_regime'] == 2),
        color='red', alpha=0.4, label='Recession (CASH)'
    )

    ax2.set_yticks([])
    ax2.set_ylabel("Regime Signal")
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Last 12 Months Live Simulation (No Look-Ahead)

# COMMAND ----------

# ── NEW BLOCK: Last 12 Months Live Simulation (No Look-Ahead) ──
from datetime import timedelta

ONEYRM_CAPITAL = 10_000  # or reuse LIVE_CAPITAL if you prefer

# Define rolling 12M window ending at last available timestamp in df
live_end = df.index.max()
oneyrm_start = live_end - pd.DateOffset(months=12)

oneyrm_df = df.loc[oneyrm_start:live_end].copy()

if oneyrm_df.empty or len(oneyrm_df) < 2:
    print("=" * 60)
    print("⚠️  12M BLOCK PENDING — Not enough data for last 12 months")
    print("=" * 60)
    print(f"Data range in df: {df.index.min().date()} → {df.index.max().date()}")
else:
    print(f"✅ 12M window: {oneyrm_df.index.min().date()} → {oneyrm_df.index.max().date()}")
    print(f"   Total hours: {len(oneyrm_df):,}")

    # Portfolio simulation (same logic as main live block)
    oneyrm_portfolio = [ONEYRM_CAPITAL]
    oneyrm_spy_bench = [ONEYRM_CAPITAL]
    active_asset     = "START"
    cost             = 0.0005

    qqq_arr    = oneyrm_df['QQQ'].values
    spy_arr    = oneyrm_df['SPY'].values
    regime_arr = oneyrm_df['predicted_regime'].values
    spy_base   = spy_arr[0]
    current_val = ONEYRM_CAPITAL

    for i in range(1, len(oneyrm_df)):
        # NO LOOK-AHEAD: use previous bar's regime
        regime = regime_arr[i - 1]

        if regime == 0:
            target = "QQQ"
            ret    = (qqq_arr[i] / qqq_arr[i - 1]) - 1
        elif regime == 1:
            target = "SPY"
            ret    = (spy_arr[i] / spy_arr[i - 1]) - 1
        else:
            target = "CASH"
            ret    = 0.0

        friction     = cost if target != active_asset else 0.0
        current_val *= (1 + ret - friction)
        active_asset = target

        oneyrm_portfolio.append(current_val)
        oneyrm_spy_bench.append((spy_arr[i] / spy_base) * ONEYRM_CAPITAL)

    oneyrm_results = oneyrm_df.iloc[:len(oneyrm_portfolio)].copy()
    oneyrm_results['Portfolio_Value'] = oneyrm_portfolio
    oneyrm_results['SPY_Value']       = oneyrm_spy_bench

    # Metrics with calendar-year CAGR for the 12M window
    def oneyrm_metrics(series, label):
        daily   = series.resample('D').last().dropna()
        returns = daily.pct_change().dropna()

        if daily.empty or returns.empty:
            print(f"\n--- {label} ---")
            print("  Not enough data for metrics.")
            return np.nan, np.nan, np.nan

        start_date = daily.index[0]
        end_date   = daily.index[-1]
        years      = (end_date - start_date).days / 365.25

        total_r = (daily.iloc[-1] / ONEYRM_CAPITAL) - 1
        cagr    = (daily.iloc[-1] / ONEYRM_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
        vol     = returns.std() * np.sqrt(252)
        sharpe  = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        mdd     = ((daily / daily.cummax()) - 1).min()

        print(f"\n--- {label} (Last 12 Months) ---")
        print(f"  Start:         {start_date.date()}  →  End: {end_date.date()}")
        print(f"  Start Capital: ${ONEYRM_CAPITAL:,.0f}")
        print(f"  Current Value: ${daily.iloc[-1]:,.2f}")
        print(f"  Total Return:  {total_r:.2%}")
        print(f"  Ann. CAGR:     {cagr:.2%}")
        print(f"  Volatility:    {vol:.2%}")
        print(f"  Sharpe Ratio:  {sharpe:.2f}")
        print(f"  Max Drawdown:  {mdd:.2%}")

        return sharpe, total_r, cagr

    print("\n" + "=" * 60)
    print("  LAST 12 MONTHS PORTFOLIO REPORT")
    print("  (No look-ahead: regime[t-1] drives returns at t)")
    print("=" * 60)

    strat_sharpe, strat_ret, strat_cagr = oneyrm_metrics(
        oneyrm_results['Portfolio_Value'], "V4 Strategy (12M)"
    )
    bench_sharpe, bench_ret, bench_cagr = oneyrm_metrics(
        oneyrm_results['SPY_Value'], "SPY Benchmark (12M)"
    )

    # Plot with regime color bands at bottom
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Equity curves
    ax1.plot(
        oneyrm_results['Portfolio_Value'],
        label=f'V4 Strategy (12M) [Sharpe: {strat_sharpe:.2f}]',
        color='navy', linewidth=2
    )
    ax1.plot(
        oneyrm_results['SPY_Value'],
        label=f'SPY Benchmark (12M) [Sharpe: {bench_sharpe:.2f}]',
        color='grey', linewidth=1.5, alpha=0.7
    )

    ax1.axhline(
        ONEYRM_CAPITAL, color='black', linestyle='--', alpha=0.4,
        label=f'Start (${ONEYRM_CAPITAL:,})'
    )

    ax1.set_title(
        f"Last 12 Months: ${ONEYRM_CAPITAL:,} Live Portfolio vs SPY",
        fontsize=14, fontweight='bold'
    )
    ax1.set_ylabel("Portfolio Value ($)")

    # Single legend box, moved to top-left
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)

    # Regime background colors
    ax2.fill_between(
        oneyrm_results.index, 0, 1,
        where=(oneyrm_results['predicted_regime'] == 0),
        color='green', alpha=0.4, label='Expansion (QQQ)'
    )
    ax2.fill_between(
        oneyrm_results.index, 0, 1,
        where=(oneyrm_results['predicted_regime'] == 1),
        color='gold', alpha=0.4, label='Slowdown (SPY)'
    )
    ax2.fill_between(
        oneyrm_results.index, 0, 1,
        where=(oneyrm_results['predicted_regime'] == 2),
        color='red', alpha=0.4, label='Recession (CASH)'
    )

    ax2.set_yticks([])
    ax2.set_ylabel("Regime Signal")
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Last 1 Month Live Simulation (No Look-Ahead)

# COMMAND ----------

# ── NEW BLOCK: Live Portfolio Simulation — Jan 1 2026 onwards ──
LIVE_START      = "2026-01-01"
LIVE_CAPITAL    = 10_000
TRAINING_CUTOFF = "2025-12-31"

live_df = df.loc[LIVE_START:].copy()

if live_df.empty:
    print("=" * 55)
    print("⚠️  LIVE BLOCK PENDING — No signals after 2026-01-01")
    print("=" * 55)
    print("Root cause: regime_signals-v4.csv currently ends at:")
    print(f"   {df.index.max().date()}")
    print("\nTo activate this block, complete the following steps:")
    print("  1. Update 05_Master_Time_Grid_Silver:")
    print('     end_date = "2026-02-24 23:00:00"')
    print("  2. Re-run notebook 05 to extend the silver grid")
    print("  3. Re-run notebook 06 to regenerate signals through Feb 2026")
    print("  4. Re-run this notebook — live block will populate automatically")
    print(f"\nExpected live period: {LIVE_START} → 2026-02-24")
    print(f"Expected hours of live data: ~{55*24:,} bars")
else:
    print(f"✅ Live window: {live_df.index.min().date()} → {live_df.index.max().date()}")
    print(f"   Total hours: {len(live_df):,}")

    # Portfolio simulation
    live_portfolio = [LIVE_CAPITAL]
    live_spy_bench =[LIVE_CAPITAL]
    active_asset   = "START"
    cost           = 0.0005

    qqq_arr    = live_df['QQQ'].values
    spy_arr    = live_df['SPY'].values
    regime_arr = live_df['predicted_regime'].values
    spy_base   = spy_arr[0]
    current_val = LIVE_CAPITAL

    for i in range(1, len(live_df)):
        regime = regime_arr[i - 1]

        if regime == 0:
            target = "QQQ"
            ret    = (qqq_arr[i] / qqq_arr[i - 1]) - 1
        elif regime == 1:
            target = "SPY"
            ret    = (spy_arr[i] / spy_arr[i - 1]) - 1
        else:
            target = "CASH"
            ret    = 0.0

        friction     = cost if target != active_asset else 0.0
        current_val *= (1 + ret - friction)
        active_asset = target

        live_portfolio.append(current_val)
        live_spy_bench.append((spy_arr[i] / spy_base) * LIVE_CAPITAL)

    live_results = live_df.iloc[:len(live_portfolio)].copy()
    live_results['Portfolio_Value'] = live_portfolio
    live_results['SPY_Value']       = live_spy_bench

    # Metrics function now returns metrics for chart inclusion
    def live_metrics(series, label):
        daily   = series.resample('D').last().dropna()
        returns = daily.pct_change().dropna()
        years   = len(returns) / 252
        cagr    = (daily.iloc[-1] / LIVE_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
        vol     = returns.std() * np.sqrt(252)
        sharpe  = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        mdd     = ((daily / daily.cummax()) - 1).min()
        total_r = (daily.iloc[-1] / LIVE_CAPITAL) - 1
        
        # -- TABLE OUTPUT --
        print(f"\n--- {label} ---")
        print(f"  Start Capital:  ${LIVE_CAPITAL:,.0f}")
        print(f"  Current Value:  ${daily.iloc[-1]:,.2f}")
        print(f"  Total Return:   {total_r:.2%}")
        print(f"  Ann. CAGR:      {cagr:.2%}")
        print(f"  Volatility:     {vol:.2%}")
        print(f"  Sharpe Ratio:   {sharpe:.2f}")
        print(f"  Max Drawdown:   {mdd:.2%}")
        
        return sharpe, total_r, cagr

    print("\n" + "=" * 55)
    print(f"  LIVE PORTFOLIO REPORT — from {LIVE_START}")
    print(f"  Model trained through {TRAINING_CUTOFF}")
    print("=" * 55)
    
    # Capture metrics for the charts
    strat_sharpe, strat_ret, strat_cagr = live_metrics(live_results['Portfolio_Value'], "V4 Strategy (Live)")
    bench_sharpe, bench_ret, bench_cagr = live_metrics(live_results['SPY_Value'],       "SPY Benchmark (Live)")

    # Current regime
    latest = live_results.iloc[-1]
    print(f"\n{'='*55}")
    print(f"  CURRENT SIGNAL as of {live_results.index[-1].date()}")
    print(f"{'='*55}")
    print(f"  Regime:    {latest['regime_name']} ({int(latest['predicted_regime'])})")
    print(f"  Position:  {'100% QQQ' if latest['predicted_regime']==0 else '100% SPY' if latest['predicted_regime']==1 else '100% CASH'}")
    print(f"  Value:     ${latest['Portfolio_Value']:,.2f}  (started ${LIVE_CAPITAL:,})")

    # Plot
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Line Plots with Sharpe in the Legend
    ax1.plot(live_results['Portfolio_Value'], label=f'V4 Strategy (Live) [Sharpe: {strat_sharpe:.2f}]', color='navy',  linewidth=2)
    ax1.plot(live_results['SPY_Value'],       label=f'SPY Benchmark [Sharpe: {bench_sharpe:.2f}]',      color='grey',  linewidth=1.5, alpha=0.7)
    
    ax1.axhline(LIVE_CAPITAL, color='black', linestyle='--', alpha=0.4, label=f'Start (${LIVE_CAPITAL:,})')
    ax1.set_title(f"Live Portfolio: ${LIVE_CAPITAL:,} invested on {LIVE_START}\n(Model trained through {TRAINING_CUTOFF})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)

    # Regime Background Colors
    ax2.fill_between(live_results.index, 0, 1,
                     where=(live_results['predicted_regime'] == 0),
                     color='green', alpha=0.4, label='Expansion (QQQ)')
    ax2.fill_between(live_results.index, 0, 1,
                     where=(live_results['predicted_regime'] == 1),
                     color='gold',  alpha=0.4, label='Slowdown (SPY)')
    ax2.fill_between(live_results.index, 0, 1,
                     where=(live_results['predicted_regime'] == 2),
                     color='red',   alpha=0.4, label='Recession (CASH)')
    
    ax2.set_yticks([])
    ax2.set_ylabel("Regime Signal")
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# ── RETURNS EXPORT: Monthly / Quarterly / Yearly for V3, V4, SPY ──
# Window: start_date through V3_END (Q2 2025) — consistent with comparison chart

print(f"Exporting returns: {comparison_df.index.min().date()} -> {comparison_df.index.max().date()}")

CSV_OUT_BASE = "/Volumes/pawan_quant_research_1/default/market_data"

# ── Daily returns from equity curves ──
daily_rets = pd.DataFrame({
    "V3":  comparison_df["V3_Value"].pct_change(),
    "V4":  comparison_df["V4_Value"].pct_change(),
    "SPY": comparison_df["SPY_Value"].pct_change(),
}).dropna()

# ── Compounding function ──
def compound(ret_series, freq):
    return ret_series.resample(freq).apply(lambda x: (1 + x).prod() - 1)

# ── Monthly ──
monthly = pd.DataFrame({
    "V3_Return":  compound(daily_rets["V3"],  "ME"),
    "V4_Return":  compound(daily_rets["V4"],  "ME"),
    "SPY_Return": compound(daily_rets["SPY"], "ME"),
})
monthly["period"]              = monthly.index.to_period("M").astype(str)
monthly["year"]                = monthly.index.year
monthly["month"]               = monthly.index.month
monthly["V4_alpha_vs_SPY"]     = monthly["V4_Return"] - monthly["SPY_Return"]
monthly["V4_alpha_vs_V3"]      = monthly["V4_Return"] - monthly["V3_Return"]
monthly.index.name             = "period_end"
monthly = monthly.reset_index()[
    ["period_end","period","year","month",
     "V3_Return","V4_Return","SPY_Return",
     "V4_alpha_vs_SPY","V4_alpha_vs_V3"]
]

# ── Quarterly ──
quarterly = pd.DataFrame({
    "V3_Return":  compound(daily_rets["V3"],  "QE"),
    "V4_Return":  compound(daily_rets["V4"],  "QE"),
    "SPY_Return": compound(daily_rets["SPY"], "QE"),
})
quarterly["period"]          = quarterly.index.to_period("Q").astype(str)
quarterly["year"]            = quarterly.index.year
quarterly["quarter"]         = quarterly.index.quarter
quarterly["V4_alpha_vs_SPY"] = quarterly["V4_Return"] - quarterly["SPY_Return"]
quarterly["V4_alpha_vs_V3"]  = quarterly["V4_Return"] - quarterly["V3_Return"]
quarterly.index.name         = "period_end"
quarterly = quarterly.reset_index()[
    ["period_end","period","year","quarter",
     "V3_Return","V4_Return","SPY_Return",
     "V4_alpha_vs_SPY","V4_alpha_vs_V3"]
]

# ── Yearly ──
yearly = pd.DataFrame({
    "V3_Return":  compound(daily_rets["V3"],  "YE"),
    "V4_Return":  compound(daily_rets["V4"],  "YE"),
    "SPY_Return": compound(daily_rets["SPY"], "YE"),
})
yearly["period"]          = yearly.index.year.astype(str)
yearly["year"]            = yearly.index.year
yearly["V4_alpha_vs_SPY"] = yearly["V4_Return"] - yearly["SPY_Return"]
yearly["V4_alpha_vs_V3"]  = yearly["V4_Return"] - yearly["V3_Return"]
yearly.index.name         = "period_end"
yearly = yearly.reset_index()[
    ["period_end","period","year",
     "V3_Return","V4_Return","SPY_Return",
     "V4_alpha_vs_SPY","V4_alpha_vs_V3"]
]

# ── Save CSVs ──
monthly.to_csv(   f"{CSV_OUT_BASE}/returns_monthly_v3_v4_spy.csv",   index=False)
quarterly.to_csv( f"{CSV_OUT_BASE}/returns_quarterly_v3_v4_spy.csv", index=False)
yearly.to_csv(    f"{CSV_OUT_BASE}/returns_yearly_v3_v4_spy.csv",    index=False)

print(f"Monthly   CSV: {len(monthly)} rows")
print(f"Quarterly CSV: {len(quarterly)} rows")
print(f"Yearly    CSV: {len(yearly)} rows")

# ── Print previews ──
ret_cols = ["V3_Return","V4_Return","SPY_Return","V4_alpha_vs_SPY","V4_alpha_vs_V3"]

def pct_fmt(df, cols):
    out = df.copy().astype(object)
    for c in cols:
        out[c] = out[c].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "NA")
    return out

print("\n--- YEARLY RETURNS ---")
print(pct_fmt(yearly, ret_cols).to_string(index=False))

print("\n--- QUARTERLY RETURNS (last 8) ---")
print(pct_fmt(quarterly.tail(8), ret_cols).to_string(index=False))

print("\n--- MONTHLY RETURNS (last 12) ---")
print(pct_fmt(monthly.tail(12), ret_cols).to_string(index=False))

# ── Save to Delta ──
for label, pdf in [("monthly", monthly), ("quarterly", quarterly), ("yearly", yearly)]:
    sdf = spark.createDataFrame(pdf.astype(str))   # cast all to string avoids Arrow date issues
    sdf.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"pawan_quant_research_1.default.gold_bt_returns_{label}")
    print(f"Delta saved: gold_bt_returns_{label}")

print("\nAll returns exported successfully.")
