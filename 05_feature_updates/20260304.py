# Databricks notebook source
# MAGIC %md
# MAGIC # V4 Regime Model Analysis Notebook
# MAGIC March 2026
# MAGIC
# MAGIC Goals:
# MAGIC - Extract and visualize feature importance
# MAGIC - Document input features (frequency, source, meaning)
# MAGIC - Identify regime changes / implied trades Jan–Mar 2026
# MAGIC - Show current regime state as of latest data
# MAGIC
# MAGIC Assumes:
# MAGIC - Signal file exists: `/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv`
# MAGIC   (or adjust path if using portara_1h version)
# MAGIC - Model file exists (optional for importance): same volume

# COMMAND ----------

#Cell 2 – Installs & Imports

%pip install xgboost joblib seaborn calplot

# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

pd.set_option('display.max_columns', 40)
pd.set_option('display.float_format', '{:,.4f}'.format)

# COMMAND ----------

# Cell 2 - Installs & Imports

# Adjust path if you're using the portara version
SIGNAL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv"
# SIGNAL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_portara_1h_regime_signals-v4.csv"

df = pd.read_csv(SIGNAL_PATH, index_col="timestamp", parse_dates=True)
df.index = df.index.tz_localize(None)  # make naive UTC

print("Shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())
print("\nLast 5 rows:")
display(df.tail())

print("\nRegime distribution (full history):")
print(df["regime_name"].value_counts(normalize=True).round(3))

# COMMAND ----------

# Cell 3 - Load the Regime Signal File

# Adjust path if you're using the portara version
SIGNAL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv"
# SIGNAL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_portara_1h_regime_signals-v4.csv"

df = pd.read_csv(SIGNAL_PATH, index_col="timestamp", parse_dates=True)
df.index = df.index.tz_localize(None)  # make naive UTC

print("Shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())
print("\nLast 5 rows:")
display(df.tail())

print("\nRegime distribution (full history):")
print(df["regime_name"].value_counts(normalize=True).round(3))

# COMMAND ----------

# Cell 4 - Document Input Features & Their Frequencies

features_info = pd.DataFrame([
    {"Feature": "CPIAUCSL_z",      "Original Freq": "Monthly",   "Model Freq": "1h (ffill)", "Description": "CPI inflation z-score"},
    {"Feature": "PCEPI_z",         "Original Freq": "Monthly",   "Model Freq": "1h (ffill)", "Description": "Core PCE inflation z-score"},
    {"Feature": "PCE_z",           "Original Freq": "Monthly",   "Model Freq": "1h (ffill)", "Description": "PCE spending z-score"},
    {"Feature": "PAYEMS_z",        "Original Freq": "Monthly",   "Model Freq": "1h (ffill)", "Description": "Nonfarm Payrolls z-score"},
    {"Feature": "UNRATE_z",        "Original Freq": "Monthly",   "Model Freq": "1h (ffill)", "Description": "Unemployment rate z-score"},
    {"Feature": "GDPC1_z",         "Original Freq": "Quarterly", "Model Freq": "1h (ffill)", "Description": "Real GDP z-score"},
    {"Feature": "INDPRO_z",        "Original Freq": "Monthly",   "Model Freq": "1h (ffill)", "Description": "Industrial production z-score"},
    {"Feature": "UMCSENT_z",       "Original Freq": "Monthly",   "Model Freq": "1h (ffill)", "Description": "Consumer sentiment z-score"},
    {"Feature": "VIXCLS_z",        "Original Freq": "Daily / 1h","Model Freq": "1h",         "Description": "VIX spot volatility z-score"},
    {"Feature": "vix3m_z",         "Original Freq": "Daily",     "Model Freq": "1h (lagged+ffill)", "Description": "VIX 3-month z-score"},
    {"Feature": "VIX_Inversion_Z", "Original Freq": "Computed",  "Model Freq": "1h",         "Description": "VIX term structure inversion z-score"},
    {"Feature": "Price_Mom_24H",   "Original Freq": "1h",        "Model Freq": "1h",         "Description": "SPY 24-hour price momentum"},
    {"Feature": "CPI_Event_Active","Original Freq": "Event",     "Model Freq": "1h",         "Description": "CPI release window flag (24h)"},
    {"Feature": "NFP_Event_Active", "Original Freq": "Event",    "Model Freq": "1h",         "Description": "NFP release window flag (24h)"},
])

display(features_info.style.set_caption("V4 Model Input Features"))

# COMMAND ----------

# Cell 5 - Feature Importance

MODEL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_xgboost_regime.joblib"
# MODEL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_portara_1h_xgboost_regime.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully\n")
    
    features = [
        "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
        "GDPC1_z", "INDPRO_z", "UMCSENT_z", "VIXCLS_z", "vix3m_z",
        "VIX_Inversion_Z", "Price_Mom_24H", "CPI_Event_Active", "NFP_Event_Active"
    ]
    
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    fi_df["importance_pct"] = (fi_df["importance"] / fi_df["importance"].sum() * 100).round(2)
    
    display(fi_df.style.format({"importance": "{:.4f}", "importance_pct": "{:.2f}%"})
                 .set_caption("XGBoost Feature Importance (gain)"))
    
    # Plot
    plt.figure(figsize=(10, 7))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
    plt.title("V4 Regime Model – Feature Importance")
    plt.xlabel("Importance (normalized)")
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print(f"Model file not found: {MODEL_PATH}")
    print("→ You can still analyze signals without the model file.")

# COMMAND ----------



# COMMAND ----------

# Cell 6 - Regime Changes & Implied Trades (Jan-Mar 2026)

period_start = "2026-01-01"
period_end   = "2026-03-03"   # or "2026-03-04" if data exists

df_period = df.loc[period_start:period_end].copy()

# Detect regime change
df_period["prev_regime"] = df_period["predicted_regime"].shift(1)
df_period["trade_triggered"] = df_period["predicted_regime"] != df_period["prev_regime"]

trades = df_period[df_period["trade_triggered"]].copy()

# Format output
trades_display = trades[[
    "predicted_regime", "regime_name",
    "VIX_Inversion_Z", "Price_Mom_24H",
    "CPI_Event_Active", "NFP_Event_Active"
]].copy()

trades_display.index = trades_display.index.strftime("%Y-%m-%d %H:%M")
trades_display["Action"] = trades_display["regime_name"].map({
    "Expansion": "→ 100% QQQ",
    "Slowdown":  "→ 100% SPY",
    "Recession": "→ 100% Cash"
})

print(f"Regime changes / implied trades between {period_start} – {period_end}")
if len(trades_display) == 0:
    print("→ No regime changes detected in this period.")
else:
    display(trades_display)

# Daily summary of last regime per day
daily_last = df_period.resample("D")["regime_name"].last().to_frame()
daily_last["Hold Suggestion"] = daily_last["regime_name"].map({
    "Expansion": "100% QQQ",
    "Slowdown":  "100% SPY",
    "Recession": "100% Cash"
})

print("\nDaily ending regime (Jan–Mar 2026):")
display(daily_last)

# COMMAND ----------

# Cell 7 - Current Regime Dashboard (Latest Bar)

latest = df.iloc[-1]

print("┌──────────────────────────────────────────────┐")
print(f"  Latest data point: {latest.name.strftime('%Y-%m-%d %H:%M UTC')}")
print("────────────────────────────────────────────────")
print(f"  Current Regime : {latest['regime_name']} ({int(latest['predicted_regime'])})")
print(f"  Position Suggestion : {'100% QQQ' if latest['predicted_regime']==0 else '100% SPY' if latest['predicted_regime']==1 else '100% CASH'}")
print("────────────────────────────────────────────────")
print(f"  VIX Inversion (Z)   : {latest.get('VIX_Inversion_Z', np.nan):.4f}")
print(f"  Price Mom 24H       : {latest.get('Price_Mom_24H', np.nan):+.2%}")
print(f"  CPI Event Active    : {bool(latest.get('CPI_Event_Active', 0))}")
print(f"  NFP Event Active    : {bool(latest.get('NFP_Event_Active', 0))}")
print("└──────────────────────────────────────────────┘")

# COMMAND ----------

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

# Dates
period_start = "2026-01-01"
period_end = datetime.now().strftime("%Y-%m-%d %H:%M:00")  # Today: ~2026-03-04 18:07 UTC

INITIAL_CAPITAL = 10000  # $10k starting account
SIGNAL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv"

# Load signals
df = pd.read_csv(SIGNAL_PATH, index_col='timestamp', parse_dates=True)
df.index = df.index.tz_localize(None)

# Filter to period
df_period = df.loc[period_start:period_end].copy()
print(f"Signals loaded: {len(df_period):,} rows from {df_period.index.min()} to {df_period.index.max()}")
print("\nRegime distribution in period:")
print(df_period["regime_name"].value_counts(normalize=True).round(3))

# COMMAND ----------

# Pull hourly prices from bronze table
market_sdf = spark.table("pawan_quant_research_1.v4.bronze_bt_market_1h")  # Use BT table for full history
prices_pdf = market_sdf.filter("symbol IN ('SPY', 'QQQ') AND timestamp BETWEEN '2026-01-01' AND '2026-03-05'") \
    .dropDuplicates(["timestamp", "symbol"]).toPandas()

prices_pdf['timestamp'] = pd.to_datetime(prices_pdf['timestamp']).dt.tz_localize(None)
prices_pdf = prices_pdf.pivot(index='timestamp', columns='symbol', values='close').rename(columns={'SPY': 'SPY_price', 'QQQ': 'QQQ_price'})

# Join to signals
df_period = df_period.join(prices_pdf, how='left')
df_period[['SPY_price', 'QQQ_price']] = df_period[['SPY_price', 'QQQ_price']].ffill()  # Forward-fill missing prices

# Compute hourly returns
df_period['SPY_return'] = df_period['SPY_price'].pct_change()
df_period['QQQ_return'] = df_period['QQQ_price'].pct_change()

print("Prices joined. Sample:")
display(df_period[['predicted_regime', 'regime_name', 'SPY_price', 'QQQ_price', 'SPY_return', 'QQQ_return']].head(10))

# COMMAND ----------

# Detect regime changes (trades trigger on flip)
df_period['prev_regime'] = df_period['predicted_regime'].shift(1)
df_period['trade_triggered'] = (df_period['predicted_regime'] != df_period['prev_regime']) & df_period['prev_regime'].notna()

trades = df_period[df_period['trade_triggered']][[
    'predicted_regime', 'regime_name', 'SPY_price', 'QQQ_price',
    'VIX_Inversion_Z', 'Price_Mom_24H', 'CPI_Event_Active', 'NFP_Event_Active'
]].copy()

# Map to actions
trades['Action'] = trades['regime_name'].map({
    'Expansion': '→ 100% QQQ',
    'Slowdown': '→ 100% SPY',
    'Recession': '→ 100% CASH'
})

trades.index = trades.index.strftime("%Y-%m-%d %H:%M UTC")
print(f"\nTrades detected: {len(trades)} regime changes from Jan 1 – {period_end}")
if len(trades) > 0:
    display(trades)
else:
    print("No regime changes in period — held initial position throughout.")

# COMMAND ----------

# Detect regime changes (trades trigger on flip)
df_period['prev_regime'] = df_period['predicted_regime'].shift(1)
df_period['trade_triggered'] = (df_period['predicted_regime'] != df_period['prev_regime']) & df_period['prev_regime'].notna()

trades = df_period[df_period['trade_triggered']][[
    'predicted_regime', 'regime_name', 'SPY_price', 'QQQ_price',
    'VIX_Inversion_Z', 'Price_Mom_24H', 'CPI_Event_Active', 'NFP_Event_Active'
]].copy()

# Map to actions
trades['Action'] = trades['regime_name'].map({
    'Expansion': '→ 100% QQQ',
    'Slowdown': '→ 100% SPY',
    'Recession': '→ 100% CASH'
})

trades.index = trades.index.strftime("%Y-%m-%d %H:%M UTC")
print(f"\nTrades detected: {len(trades)} regime changes from Jan 1 – {period_end}")
if len(trades) > 0:
    display(trades)
else:
    print("No regime changes in period — held initial position throughout.")

# COMMAND ----------

# Simulate positions and P&L
df_sim = df_period.copy()
df_sim['target_asset'] = df_sim['predicted_regime'].map({0: 'QQQ_price', 1: 'SPY_price', 2: None})
df_sim['asset_return'] = np.where(
    df_sim['target_asset'] == 'QQQ_price', df_sim['QQQ_return'],
    np.where(df_sim['target_asset'] == 'SPY_price', df_sim['SPY_return'], 0.0)
)

df_sim['target_asset'] = df_sim['target_asset'].ffill()
df_sim['asset_return'] = df_sim['asset_return'].ffill()

df_sim['portfolio_return'] = df_sim['asset_return']

# Strategy equity curve
df_sim['equity'] = INITIAL_CAPITAL * (1 + df_sim['portfolio_return']).cumprod()
df_sim['cumulative_return'] = df_sim['equity'] / INITIAL_CAPITAL - 1
df_sim.loc[df_sim.index[0], 'equity'] = INITIAL_CAPITAL

# ── SPY Buy & Hold benchmark (NEW) ──
df_sim['SPY_equity'] = INITIAL_CAPITAL * (1 + df_sim['SPY_return'].fillna(0)).cumprod()
df_sim['SPY_cum_return'] = df_sim['SPY_equity'] / INITIAL_CAPITAL - 1

print("P&L Simulation Summary:")
print(f"Starting Equity: ${INITIAL_CAPITAL:,.0f}")
print(f"Ending Equity: ${df_sim['equity'].iloc[-1]:,.0f}")
print(f"Total Return: {df_sim['cumulative_return'].iloc[-1]:.2%}")
print(f"Max Drawdown: {((df_sim['equity'].cummax() - df_sim['equity']) / df_sim['equity'].cummax()).max():.2%}")
print(f"Sharpe Ratio (assuming 0% risk-free): {df_sim['portfolio_return'].mean() / df_sim['portfolio_return'].std() * np.sqrt(8760):.2f} (annualized)")

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# ── Risk-free rate assumption (annualized) ──
RISK_FREE_RATE = 0.00  # 0% for simplicity; change to e.g. 0.04 for 4% if you have a proxy

# ── Sharpe Ratio Calculation ──
# Annualization factor: sqrt(8760) for hourly data
annual_factor = np.sqrt(8760)

# Strategy Sharpe
strat_mean_ret = df_sim['portfolio_return'].mean()
strat_std_ret  = df_sim['portfolio_return'].std()
sharpe_strategy = (strat_mean_ret - RISK_FREE_RATE / annual_factor) / strat_std_ret * annual_factor if strat_std_ret > 0 else 0

# SPY Sharpe
spy_mean_ret = df_sim['SPY_return'].mean()
spy_std_ret  = df_sim['SPY_return'].std()
sharpe_spy = (spy_mean_ret - RISK_FREE_RATE / annual_factor) / spy_std_ret * annual_factor if spy_std_ret > 0 else 0

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1], sharex=True)

# Equity curves
ax1.plot(df_sim.index, df_sim['equity'], label='V4 Strategy Equity', color='blue', linewidth=2)
ax1.plot(df_sim.index, df_sim['SPY_equity'], label='100% SPY Buy & Hold', color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5, label='Starting Capital $10,000')
ax1.set_ylabel('Equity ($)')
ax1.set_title(f'V4 Strategy vs SPY Benchmark\nJan 1, 2026 – {period_end} | Initial: ${INITIAL_CAPITAL:,}')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Regime background
ax2.fill_between(df_sim.index, 0, 1,
                 where=(df_sim['predicted_regime'] == 0), color='green', alpha=0.4, label='Expansion (QQQ)')
ax2.fill_between(df_sim.index, 0, 1,
                 where=(df_sim['predicted_regime'] == 1), color='gold', alpha=0.4, label='Slowdown (SPY)')
ax2.fill_between(df_sim.index, 0, 1,
                 where=(df_sim['predicted_regime'] == 2), color='red', alpha=0.4, label='Recession (CASH)')
ax2.set_ylabel('Regime')
ax2.set_yticks([])
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── Monthly summary with SPY benchmark ──
monthly = df_sim.resample('M').agg({
    'equity': ['first', 'last'],
    'cumulative_return': 'last',
    'SPY_equity': ['first', 'last'],
    'SPY_cum_return': 'last',
    'predicted_regime': lambda x: x.mode().iloc[0] if not x.empty else np.nan
}).round(4)

monthly.columns = [
    'Strategy Start', 'Strategy End', 'Strategy Return',
    'SPY Start', 'SPY End', 'SPY Return',
    'Dominant Regime'
]

monthly['Strat vs SPY'] = monthly['Strategy Return'] - monthly['SPY Return']

display(monthly)

# ── Final Performance Summary with Sharpe ──
print("\nFinal Performance (as of last bar):")
print(f"  V4 Strategy Ending Equity:    ${df_sim['equity'].iloc[-1]:,.2f}")
print(f"  V4 Strategy Total Return:      {df_sim['cumulative_return'].iloc[-1]:+.2%}")
print(f"  V4 Strategy Sharpe Ratio (ann.): {sharpe_strategy:.2f}")
print("")
print(f"  100% SPY Ending Equity:        ${df_sim['SPY_equity'].iloc[-1]:,.2f}")
print(f"  100% SPY Total Return:         {df_sim['SPY_cum_return'].iloc[-1]:+.2%}")
print(f"  100% SPY Sharpe Ratio (ann.):  {sharpe_spy:.2f}")
print("")
print(f"  Strategy Outperformance:       {(df_sim['cumulative_return'].iloc[-1] - df_sim['SPY_cum_return'].iloc[-1]):+.2%}")
print(f"  Sharpe Improvement vs SPY:     {sharpe_strategy - sharpe_spy:+.2f}")

# COMMAND ----------

# Databricks notebook source
# MAGIC %md
# MAGIC ## Export Trade CSVs
# MAGIC - **CSV 1:** Historical trades 2016-01-01 → 2025-12-31
# MAGIC - **CSV 2:** YTD trades 2026-01-01 → latest
# MAGIC Each row = one regime switch (day trade trigger), with reason columns.

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime

SIGNAL_PATH  = "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv"
OUTPUT_VOL   = "/Volumes/pawan_quant_research_1/v4/market_data"

REGIME_MAP   = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
POSITION_MAP = {0: "100% QQQ",  1: "100% SPY",  2: "100% CASH"}

# ── Load full signal history ──
df = pd.read_csv(SIGNAL_PATH, index_col="timestamp", parse_dates=True)
df.index = df.index.tz_localize(None)
df = df.sort_index()

print(f"Full signal range: {df.index.min()} → {df.index.max()}")
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# COMMAND ----------

def build_trade_log(df_in, label=""):
    """
    Detect every hourly regime switch and produce one row per trade.
    Includes the leading/trailing macro context so the 'reason' is readable.
    """
    df_w = df_in.copy()

    # ── Previous regime (shift by 1 hour) ──
    df_w["prev_regime"]      = df_w["predicted_regime"].shift(1)
    df_w["prev_regime_name"] = df_w["prev_regime"].map(REGIME_MAP)
    df_w["prev_position"]    = df_w["prev_regime"].map(POSITION_MAP)

    # ── Detect switches (ignore the very first row which has no predecessor) ──
    switches = df_w[
        (df_w["predicted_regime"] != df_w["prev_regime"]) &
        df_w["prev_regime"].notna()
    ].copy()

    if switches.empty:
        print(f"[{label}] No regime switches found.")
        return pd.DataFrame()

    # ── New position ──
    switches["new_position"] = switches["predicted_regime"].map(POSITION_MAP)

    # ── Human-readable switch direction ──
    switches["switch"] = switches["prev_regime_name"] + " → " + switches["regime_name"]

    # ── Regime switch reason: which signals drove the change ──
    # We flag the top contributing signals at the moment of the switch
    def regime_reason(row):
        reasons = []

        # VIX signals
        vix_z   = row.get("VIXCLS_z",        np.nan)
        vix3m_z = row.get("vix3m_z",          np.nan)
        inv_z   = row.get("VIX_Inversion_Z",  np.nan)
        mom     = row.get("Price_Mom_24H",     np.nan)
        cpi_z   = row.get("CPIAUCSL_z",        np.nan)
        pay_z   = row.get("PAYEMS_z",          np.nan)
        unr_z   = row.get("UNRATE_z",          np.nan)
        gdp_z   = row.get("GDPC1_z",           np.nan)
        ind_z   = row.get("INDPRO_z",          np.nan)
        ums_z   = row.get("UMCSENT_z",         np.nan)
        cpi_ev  = row.get("CPI_Event_Active",  0)
        nfp_ev  = row.get("NFP_Event_Active",  0)

        # Macro stress signals
        if not np.isnan(pay_z):
            if pay_z < -1.5:
                reasons.append(f"Labor market stress (PAYEMS_z={pay_z:.2f})")
            elif pay_z > 1.5:
                reasons.append(f"Strong payrolls (PAYEMS_z={pay_z:.2f})")

        if not np.isnan(unr_z):
            if unr_z > 1.5:
                reasons.append(f"Rising unemployment (UNRATE_z={unr_z:.2f})")
            elif unr_z < -1.5:
                reasons.append(f"Low unemployment (UNRATE_z={unr_z:.2f})")

        if not np.isnan(cpi_z):
            if cpi_z > 1.5:
                reasons.append(f"Elevated inflation (CPIAUCSL_z={cpi_z:.2f})")
            elif cpi_z < -1.0:
                reasons.append(f"Low inflation (CPIAUCSL_z={cpi_z:.2f})")

        if not np.isnan(gdp_z):
            if gdp_z < -1.0:
                reasons.append(f"Weak GDP (GDPC1_z={gdp_z:.2f})")
            elif gdp_z > 1.5:
                reasons.append(f"Strong GDP (GDPC1_z={gdp_z:.2f})")

        if not np.isnan(ind_z):
            if ind_z < -1.0:
                reasons.append(f"Industrial contraction (INDPRO_z={ind_z:.2f})")

        if not np.isnan(ums_z):
            if ums_z < -1.0:
                reasons.append(f"Weak consumer sentiment (UMCSENT_z={ums_z:.2f})")

        # VIX / vol signals
        if not np.isnan(vix_z):
            if vix_z > 1.5:
                reasons.append(f"VIX spike (VIXCLS_z={vix_z:.2f})")
            elif vix_z < -1.0:
                reasons.append(f"VIX compression (VIXCLS_z={vix_z:.2f})")

        if not np.isnan(vix3m_z):
            if vix3m_z > 1.5:
                reasons.append(f"Elevated 3M vol (vix3m_z={vix3m_z:.2f})")

        if not np.isnan(inv_z):
            if inv_z < -0.5:
                reasons.append(f"VIX term inversion (VIX_Inversion_Z={inv_z:.2f})")
            elif inv_z > 0.5:
                reasons.append(f"VIX term normalisation (VIX_Inversion_Z={inv_z:.2f})")

        # Price momentum
        if not np.isnan(mom):
            if mom < -0.01:
                reasons.append(f"Negative price momentum (Price_Mom_24H={mom:.2%})")
            elif mom > 0.01:
                reasons.append(f"Positive price momentum (Price_Mom_24H={mom:.2%})")

        # Event flags
        if cpi_ev:
            reasons.append("CPI release window active")
        if nfp_ev:
            reasons.append("NFP release window active")

        return "; ".join(reasons) if reasons else "Model boundary crossing (low-margin decision)"

    switches["regime_switch_reason"] = switches.apply(regime_reason, axis=1)

    # ── Select and order final columns ──
    cols_available = [c for c in [
        "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
        "GDPC1_z", "INDPRO_z", "UMCSENT_z",
        "VIXCLS_z", "vix3m_z", "VIX_Inversion_Z",
        "Price_Mom_24H", "CPI_Event_Active", "NFP_Event_Active",
        "SPY_price", "QQQ_price"
    ] if c in switches.columns]

    out = switches[[
        "prev_regime_name", "prev_position",
        "regime_name", "new_position",
        "switch", "regime_switch_reason",
    ] + cols_available].copy()

    out.index.name = "trade_timestamp"
    out = out.rename(columns={
        "prev_regime_name": "from_regime",
        "regime_name":      "to_regime",
        "prev_position":    "exiting_position",
        "new_position":     "entering_position",
    })

    print(f"[{label}] {len(out)} regime switches found.")
    return out

# COMMAND ----------

# ── CSV 1: Historical  2016-01-01 → 2025-12-31 ──
df_hist = df.loc["2016-01-01":"2025-12-31"]
trades_hist = build_trade_log(df_hist, label="Historical 2016–2025")

out_path_hist = f"{OUTPUT_VOL}/trades_historical_2016_2025.csv"
trades_hist.to_csv(out_path_hist)
print(f"✅  Saved: {out_path_hist}")
display(trades_hist.head(10))

# COMMAND ----------

# ── CSV 2: YTD  2026-01-01 → latest ──
df_ytd = df.loc["2026-01-01":]
trades_ytd = build_trade_log(df_ytd, label="YTD 2026")

out_path_ytd = f"{OUTPUT_VOL}/trades_ytd_2026.csv"
trades_ytd.to_csv(out_path_ytd)
print(f"✅  Saved: {out_path_ytd}")
display(trades_ytd)

# COMMAND ----------

# ── Sanity check: regime distribution per CSV ──
for label, t in [("Historical 2016–2025", trades_hist), ("YTD 2026", trades_ytd)]:
    print(f"\n── {label} ──")
    print(f"  Total switches : {len(t)}")
    if not t.empty:
        print(f"  By destination regime:")
        print(t["to_regime"].value_counts().to_string())
        print(f"  Date range    : {t.index.min()} → {t.index.max()}")