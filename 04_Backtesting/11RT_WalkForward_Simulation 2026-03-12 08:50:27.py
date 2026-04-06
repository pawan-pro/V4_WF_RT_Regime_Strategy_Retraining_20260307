# Databricks notebook source
!pip install xgboost 

# COMMAND ----------

# Databricks notebook source
# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F

# COMMAND ----------

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"

LABEL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv"

TRAIN_START = "2016-01-01"
LIVE_START = "2021-01-01"
LIVE_END = "2026-01-01"

FEATURES = [
"CPIAUCSL_z","PCEPI_z","PCE_z","PAYEMS_z","UNRATE_z",
"GDPC1_z","INDPRO_z","UMCSENT_z","VIXCLS_z","vix3m_z",
"VIX_Inversion_Z","Price_Mom_24H","CPI_Event_Active","NFP_Event_Active"
]

TARGET = "regime"

print("Configuration loaded")

# COMMAND ----------

# ------------------------------------------------------------
# LOAD FEATURE DATA (SPARK FILTER FIRST)
# ------------------------------------------------------------

spark_df = (
    spark.table(SILVER_TABLE)
    .filter(f"timestamp >= '{TRAIN_START}'")
    .select(["timestamp"] + FEATURES)
)

silver_df = spark_df.toPandas()

silver_df["timestamp"] = pd.to_datetime(silver_df["timestamp"])

silver_df = silver_df.sort_values("timestamp")

print("Rows loaded:", len(silver_df))
print("Range:", silver_df.timestamp.min(), "→", silver_df.timestamp.max())

# COMMAND ----------

# ------------------------------------------------------------
# LOAD LABELS
# ------------------------------------------------------------

labels_df = pd.read_csv(LABEL_PATH, parse_dates=["timestamp"])

labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"])

labels_df = labels_df[["timestamp","predicted_regime"]]

labels_df = labels_df.rename(columns={"predicted_regime":"regime"})

# COMMAND ----------

# ------------------------------------------------------------
# MERGE DATA
# ------------------------------------------------------------

silver_df = silver_df.merge(labels_df, on="timestamp", how="left")

silver_df = silver_df.dropna(subset=["regime"])

silver_df["regime"] = silver_df["regime"].astype(int)

print("Dataset size:", len(silver_df))

print(silver_df.regime.value_counts())

# COMMAND ----------

# ------------------------------------------------------------
# WALK FORWARD SCHEDULE
# ------------------------------------------------------------

train_start = pd.Timestamp(TRAIN_START)
live_start = pd.Timestamp(LIVE_START)
live_end = pd.Timestamp(LIVE_END)

retrain_dates = []

d = live_start

while d <= live_end:

    retrain_dates.append(d)

    d += relativedelta(months=1)

print("Retrain cycles:", len(retrain_dates))

# COMMAND ----------

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

def train_model(train_df):

    X = train_df[FEATURES]

    y = train_df[TARGET]

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        tree_method="hist"
    )

    model.fit(X,y)

    return model

# COMMAND ----------

# ------------------------------------------------------------
# STORE FEATURE IMPORTANCE MONTHLY
# ------------------------------------------------------------
feature_importance_history = []

# ------------------------------------------------------------
# WALK-FORWARD TRAINING LOOP
# ------------------------------------------------------------
signals = []

for i in range(len(retrain_dates)-1):

    train_end = retrain_dates[i]
    test_end = retrain_dates[i+1]

    print("------------------------------------------------")
    print("TRAIN UNTIL:", train_end)
    print("TEST UNTIL:", test_end)

    train_df = silver_df[
        (silver_df.timestamp >= train_start) &
        (silver_df.timestamp < train_end)
    ]

    test_df = silver_df[
        (silver_df.timestamp >= train_end) &
        (silver_df.timestamp < test_end)
    ]

    if len(test_df) == 0:
        continue

    # TRAIN MODEL
    model = train_model(train_df)

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE EXTRACTION (IMPORTANT: INSIDE LOOP)
    # ------------------------------------------------------------
    booster = model.get_booster()

    importance_dict = booster.get_score(importance_type="gain")

    imp_df = pd.DataFrame({
        "feature": list(importance_dict.keys()),
        "importance": list(importance_dict.values())
    })

    imp_df["date"] = train_end

    feature_importance_history.append(imp_df)

    # ------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------
    test_df = test_df.copy()

    #preds = model.predict(test_df[FEATURES])

    probs = model.predict_proba(test_df[FEATURES])
    preds = np.argmax(probs, axis=1)

    test_df["prob_regime0"] = probs[:,0]
    test_df["prob_regime1"] = probs[:,1]
    test_df["prob_regime2"] = probs[:,2]

    test_df["predicted_regime"] = preds

    signals.append(
        test_df[["timestamp","predicted_regime"]]
    )

# COMMAND ----------

fi_df = pd.concat(feature_importance_history)

# normalize each month
fi_df["importance"] = fi_df.groupby("date")["importance"].transform(
    lambda x: x / x.sum()
)

fi_matrix = fi_df.pivot(
    index="date",
    columns="feature",
    values="importance"
).fillna(0)

fi_matrix = fi_matrix.sort_index()

# COMMAND ----------

plt.figure(figsize=(16,8))

fi_matrix.plot.area(
    figsize=(16,8),
    colormap="tab20",
    alpha=0.85
)

plt.title("Feature Importance Evolution (2021–2025)")
plt.ylabel("Normalized Importance")
plt.xlabel("Date")
plt.legend(bbox_to_anchor=(1.05,1))

plt.show()

# COMMAND ----------

top_features = (
    fi_matrix.mean()
    .sort_values(ascending=False)
    .head(5)
    .index
)

plt.figure(figsize=(14,6))

for f in top_features:
    plt.plot(fi_matrix.index, fi_matrix[f], label=f)

plt.legend()
plt.title("Top Feature Importance Through Time")
plt.ylabel("Importance")
plt.xlabel("Date")

plt.show()

# COMMAND ----------

fi_matrix.to_csv("feature_importance_monthly.csv")

# COMMAND ----------

# ------------------------------------------------------------
# CONCAT RESULTS
# ------------------------------------------------------------

signals_df = pd.concat(signals)

print("Signals generated:", len(signals_df))

signals_df.head()

# COMMAND ----------

# ------------------------------------------------------------
# SAVE TO DELTA (SAFE OVERWRITE)
# ------------------------------------------------------------

spark_df = spark.createDataFrame(signals_df)

(
spark_df.write
.mode("overwrite")
.option("overwriteSchema","true")
.format("delta")
.saveAsTable(SIGNAL_TABLE)
)

print("Signals saved:", SIGNAL_TABLE)

# COMMAND ----------

# ------------------------------------------------------------
# LOAD MARKET DATA (DEDUP SAFE)
# ------------------------------------------------------------

prices = spark.table(
    "pawan_quant_research_1.v4.bronze_bt_market_1h"
).filter(
    "symbol IN ('SPY','QQQ')"
).toPandas()

prices["timestamp"] = pd.to_datetime(prices["timestamp"])

# pivot safely even if duplicates exist
prices = prices.pivot_table(
    index="timestamp",
    columns="symbol",
    values="close",
    aggfunc="last"
)

prices = prices.reset_index()

print("Price rows:", len(prices))
print("Range:", prices.timestamp.min(), "→", prices.timestamp.max())

# COMMAND ----------

signals_df = signals_df.merge(prices,on="timestamp",how="left")

signals_df["SPY"] = signals_df["SPY"].ffill()
signals_df["QQQ"] = signals_df["QQQ"].ffill()

# COMMAND ----------

capital = 5_000_000

portfolio = [capital]

active_asset = "START"

for i in range(1,len(signals_df)):

    regime = signals_df.predicted_regime.iloc[i-1]

    if regime == 0:
        target = "QQQ"

    elif regime == 1:
        target = "SPY"

    else:
        target = "CASH"

    if target == "QQQ":
        ret = signals_df.QQQ.iloc[i] / signals_df.QQQ.iloc[i-1] - 1

    elif target == "SPY":
        ret = signals_df.SPY.iloc[i] / signals_df.SPY.iloc[i-1] - 1

    else:
        ret = 0

    cost = 0.0005 if target != active_asset else 0

    capital *= (1 + ret - cost)

    portfolio.append(capital)

    active_asset = target

signals_df["portfolio"] = portfolio

# COMMAND ----------

plt.figure(figsize=(14,6))

plt.plot(signals_df.timestamp, signals_df.portfolio)

plt.title("Walk Forward Strategy Equity Curve")

plt.ylabel("Portfolio Value")

plt.show()

# COMMAND ----------

# ------------------------------------------------------------
# COMPARE WALK-FORWARD vs V4 vs SPY
# ------------------------------------------------------------

INITIAL_CAPITAL = 5_000_000
TRADING_DAYS = 252

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# WALK-FORWARD NAV
# ------------------------------------------------------------

wf_nav = signals_df[["timestamp","portfolio"]].copy()

wf_nav = wf_nav.set_index("timestamp")

wf_nav = wf_nav.rename(columns={"portfolio":"WF_Value"})


# ------------------------------------------------------------
# LOAD V4 SIGNAL FILE
# ------------------------------------------------------------

V4_SIGNAL_PATH = "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv"

v4 = pd.read_csv(V4_SIGNAL_PATH, parse_dates=["timestamp"])

v4 = v4.set_index("timestamp")


# ------------------------------------------------------------
# LOAD MARKET DATA
# ------------------------------------------------------------

prices = spark.table(
"pawan_quant_research_1.v4.bronze_bt_market_1h"
).filter(
"symbol IN ('SPY','QQQ')"
).toPandas()

prices["timestamp"] = pd.to_datetime(prices["timestamp"])

prices = prices.pivot_table(
index="timestamp",
columns="symbol",
values="close",
aggfunc="last"
)

prices = prices.sort_index()


# ------------------------------------------------------------
# ALIGN DATA
# ------------------------------------------------------------

v4 = v4.join(prices)

v4["SPY"] = v4["SPY"].ffill()
v4["QQQ"] = v4["QQQ"].ffill()


# ------------------------------------------------------------
# V4 BACKTEST
# ------------------------------------------------------------

capital = INITIAL_CAPITAL

portfolio = [capital]

active_asset = "START"

for i in range(1,len(v4)):

    regime = v4.predicted_regime.iloc[i-1]

    if regime == 0:
        target = "QQQ"

    elif regime == 1:
        target = "SPY"

    else:
        target = "CASH"

    if target == "QQQ":
        ret = v4.QQQ.iloc[i] / v4.QQQ.iloc[i-1] - 1

    elif target == "SPY":
        ret = v4.SPY.iloc[i] / v4.SPY.iloc[i-1] - 1

    else:
        ret = 0

    cost = 0.0005 if target != active_asset else 0

    capital *= (1 + ret - cost)

    portfolio.append(capital)

    active_asset = target

v4["V4_Value"] = portfolio


# ------------------------------------------------------------
# SPY BENCHMARK
# ------------------------------------------------------------

spy = prices.copy()

spy = spy.loc[wf_nav.index.min():]

spy["SPY_Return"] = spy["SPY"].pct_change()

spy["SPY_Value"] = INITIAL_CAPITAL * (1 + spy["SPY_Return"]).cumprod()


# ------------------------------------------------------------
# MERGE ALL SERIES
# ------------------------------------------------------------


# ------------------------------------------------------------
# ALIGN START DATE AND REBASE TO 5M
# ------------------------------------------------------------

START_DATE = wf_nav.index.min()

comparison = pd.DataFrame(index=wf_nav.index)

comparison["WF_Value"] = wf_nav["WF_Value"]
comparison["V4_Value"] = v4["V4_Value"]
comparison["SPY_Value"] = spy["SPY_Value"]

comparison = comparison.loc[START_DATE:]

comparison = comparison.dropna()

# Rebase all series to 5M
for col in ["WF_Value","V4_Value","SPY_Value"]:
    comparison[col] = INITIAL_CAPITAL * (
        comparison[col] / comparison[col].iloc[0]
    )

print("Start date:", comparison.index.min())
print("Start values:")
print(comparison.iloc[0])


# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------

def compute_metrics(series):

    returns = series.pct_change().dropna()

    years = len(returns) / TRADING_DAYS

    total_return = series.iloc[-1] / series.iloc[0] - 1

    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1

    vol = returns.std() * np.sqrt(TRADING_DAYS)

    sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)

    max_dd = (series/series.cummax() - 1).min()

    return {
        "Final Value": series.iloc[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }


metrics = pd.DataFrame([
compute_metrics(comparison["WF_Value"]),
compute_metrics(comparison["V4_Value"]),
compute_metrics(comparison["SPY_Value"])
], index=["WalkForward","V4","SPY"])

print("\nPERFORMANCE COMPARISON\n")

display(metrics)


# ------------------------------------------------------------
# PLOT EQUITY CURVES
# ------------------------------------------------------------

plt.figure(figsize=(16,7))

plt.plot(comparison["WF_Value"], label="Walk Forward Strategy")

plt.plot(comparison["V4_Value"], label="V4 Strategy")

plt.plot(comparison["SPY_Value"], label="SPY Benchmark")

plt.yscale("log")

plt.title("WalkForward vs V4 vs SPY")

plt.legend()

plt.grid(alpha=0.3)

plt.show()

# COMMAND ----------

# ============================================================
# CLEAN STRATEGY ANALYTICS
# ============================================================

TRADING_DAYS = 252

strategies = {
    "WalkForward_V4": comparison["WF_Value"],
    "V4": comparison["V4_Value"],
    "SPY": comparison["SPY_Value"]
}

# ------------------------------------------------------------
# DAILY RETURNS
# ------------------------------------------------------------

daily_returns = pd.DataFrame()

for name, series in strategies.items():
    daily_returns[name] = (
        series.resample("D").last().pct_change(fill_method=None)
    )

daily_returns = daily_returns.dropna()


# ------------------------------------------------------------
# WEEKLY / MONTHLY / YEARLY RETURNS
# ------------------------------------------------------------

weekly_returns = (
    daily_returns
    .resample("W-FRI")
    .apply(lambda x: (1+x).prod()-1)
)

monthly_returns = (
    daily_returns
    .resample("ME")
    .apply(lambda x: (1+x).prod()-1)
)

yearly_returns = (
    daily_returns
    .resample("YE")
    .apply(lambda x: (1+x).prod()-1)
)


# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------

def compute_metrics(series):

    daily = series.resample("D").last().dropna()
    r = daily.pct_change(fill_method=None).dropna()

    years = len(r)/TRADING_DAYS

    start = series.iloc[0]
    end = series.iloc[-1]

    total_return = end/start - 1
    cagr = (end/start)**(1/years) - 1

    vol_month = r.std()*np.sqrt(21)
    vol_ann = r.std()*np.sqrt(TRADING_DAYS)

    sharpe = (r.mean()/r.std())*np.sqrt(TRADING_DAYS)

    downside = r[r<0]
    sortino = (r.mean()/downside.std())*np.sqrt(TRADING_DAYS)

    dd = series/series.cummax() - 1
    max_dd = dd.min()

    return [
        start,
        end,
        total_return,
        cagr,
        vol_month,
        vol_ann,
        sharpe,
        sortino,
        max_dd
    ]


metrics = pd.DataFrame(
    [compute_metrics(s) for s in strategies.values()],
    index=strategies.keys(),
    columns=[
        "Start Value",
        "Final Value",
        "Total Return",
        "CAGR",
        "Vol Monthly",
        "Vol Annual",
        "Sharpe Annual",
        "Sortino Annual",
        "Max Drawdown"
    ]
)

print("\n================ PERFORMANCE SUMMARY ================\n")
display(metrics.round(4))


# ------------------------------------------------------------
# POSITIVE RETURN COUNTS
# ------------------------------------------------------------

positive_summary = pd.DataFrame({
    "Positive Weeks": (weekly_returns > 0).sum(),
    "Positive Months": (monthly_returns > 0).sum(),
    "Positive Years": (yearly_returns > 0).sum()
})

print("\n================ POSITIVE RETURN PERIODS ================\n")
display(positive_summary)


# ------------------------------------------------------------
# DAILY RETURNS TABLE
# ------------------------------------------------------------

print("\n================ DAILY RETURNS PREVIEW ================\n")
display(daily_returns.head())


# ------------------------------------------------------------
# EXPORT CSVs
# ------------------------------------------------------------

EXPORT_PATH = "/Volumes/pawan_quant_research_1/v4/market_data"

daily_returns.to_csv(f"{EXPORT_PATH}/strategy_daily_returns.csv")
weekly_returns.to_csv(f"{EXPORT_PATH}/strategy_weekly_returns.csv")
monthly_returns.to_csv(f"{EXPORT_PATH}/strategy_monthly_returns.csv")
yearly_returns.to_csv(f"{EXPORT_PATH}/strategy_yearly_returns.csv")

metrics.to_csv(f"{EXPORT_PATH}/strategy_performance_summary.csv")
positive_summary.to_csv(f"{EXPORT_PATH}/strategy_positive_periods.csv")

print("\nCSV files exported successfully.")

# COMMAND ----------

# ------------------------------------------------------------
# ALIGN START DATE AND REBASE TO $5M
# ------------------------------------------------------------

INITIAL_CAPITAL = 5_000_000

# Ensure datetime index
wf_nav.index = pd.to_datetime(wf_nav.index)
v4.index = pd.to_datetime(v4.index)
spy.index = pd.to_datetime(spy.index)

# Common start date
START_DATE = wf_nav.index.min()

# Merge all strategies
comparison = pd.concat([
    wf_nav["WF_Value"],
    v4["V4_Value"],
    spy["SPY_Value"]
], axis=1)

comparison = comparison.loc[START_DATE:]
comparison = comparison.dropna()

# Rebase to same starting capital
comparison = comparison / comparison.iloc[0] * INITIAL_CAPITAL

print("Start date:", comparison.index.min())
print("\nStart values (should all be 5M):")
print(comparison.iloc[0])

# COMMAND ----------

plt.figure(figsize=(16,7))

plt.plot(comparison["WF_Value"], label="Walk Forward Strategy")
plt.plot(comparison["V4_Value"], label="V4 Strategy")
plt.plot(comparison["SPY_Value"], label="SPY Benchmark")

plt.yscale("log")
plt.title("WalkForward vs V4 vs SPY (Start = $5M)")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# COMMAND ----------

# ------------------------------------------------------------
# MERGE FEATURE IMPORTANCE WITH REGIME
# ------------------------------------------------------------

# convert importance dates
fi_df["date"] = pd.to_datetime(fi_df["date"])

# create monthly regime label
signals_df = pd.concat(signals)
signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

signals_df["date"] = signals_df["timestamp"].dt.to_period("M").dt.to_timestamp()

regime_monthly = (
    signals_df
    .groupby("date")["predicted_regime"]
    .agg(lambda x: x.mode()[0])
    .reset_index()
)

# merge with importance
fi_regime = fi_df.merge(regime_monthly, on="date", how="left")

# COMMAND ----------

# ------------------------------------------------------------
# FEATURE IMPORTANCE BY REGIME
# ------------------------------------------------------------

importance_by_regime = (
    fi_regime
    .groupby(["predicted_regime","feature"])["importance"]
    .mean()
    .reset_index()
)

importance_matrix = importance_by_regime.pivot(
    index="feature",
    columns="predicted_regime",
    values="importance"
).fillna(0)

display(importance_matrix)

# COMMAND ----------

regime_labels = {
    0: "Low Vol / Risk-On",
    1: "Transition / Neutral",
    2: "High Vol / Risk-Off"
}

# COMMAND ----------

plt.figure(figsize=(14,8))

importance_matrix.plot(kind="bar", figsize=(14,8))

plt.title("Average Feature Importance by Market Regime", fontsize=16)

plt.ylabel("Importance")
plt.xlabel("Macro Features")

# Replace regime numbers with labels
handles, labels = plt.gca().get_legend_handles_labels()
labels = [regime_labels[int(l)] for l in labels]

plt.legend(handles, labels, title="Market Regime")

plt.xticks(rotation=60)

# Add explanation text on chart
plt.text(
    -0.5,
    importance_matrix.values.max()*1.05,
    "Higher bars indicate stronger influence of that macro factor\non regime classification.",
    fontsize=11
)

plt.tight_layout()

plt.show()

# COMMAND ----------

import seaborn as sns

plt.figure(figsize=(18,8))

sns.heatmap(
    fi_matrix.T,
    cmap="viridis_r",
    linewidths=0.2
)

plt.title("Feature Importance Heatmap (2021–2025)")
plt.xlabel("Date")
plt.ylabel("Feature")

plt.show()

# COMMAND ----------

top_features = (
    fi_matrix.mean()
    .sort_values(ascending=False)
    .head(8)
    .index
)

fi_top = fi_matrix[top_features]

plt.figure(figsize=(16,6))

sns.heatmap(
    fi_top.T,
    cmap="viridis_r",
    linewidths=0.3
)

plt.title("Top Feature Importance Heatmap")

plt.show()


# COMMAND ----------

importance_matrix.to_csv("feature_importance_by_regime.csv")

fi_matrix.to_csv("feature_importance_time_matrix.csv")

# COMMAND ----------

!pip install yfinance

# COMMAND ----------

# ============================================================
# WALKFORWARD PERFORMANCE ANALYSIS (V3 vs V4 vs SPY)
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

START_DATE = "2021-01-01"
INITIAL_CAPITAL = 5_000_000

# ------------------------------------------------------------
# 1. LOAD MARKET DATA
# ------------------------------------------------------------

tickers = ["QQQ","MTUM","QUAL","USMV","SPY"]

price_df = yf.download(
    tickers,
    start=START_DATE,
    auto_adjust=True,
    progress=False
)

if isinstance(price_df.columns, pd.MultiIndex):
    price_df = price_df["Close"]

# ------------------------------------------------------------
# RETURNS
# ------------------------------------------------------------

daily_returns = price_df.pct_change().dropna()

# convert daily → hourly equivalent
hourly_returns = (1 + daily_returns)**(1/24) - 1


# ------------------------------------------------------------
# 2. LOAD V4 HOURLY SIGNALS
# ------------------------------------------------------------

v4 = signals_df.copy()

v4["timestamp"] = pd.to_datetime(v4["timestamp"])

v4 = v4.set_index("timestamp")["predicted_regime"]

v4 = v4.loc[START_DATE:]


# ------------------------------------------------------------
# 3. LOAD V3 WEEKLY WALKFORWARD SIGNALS
# ------------------------------------------------------------

v3 = pd.read_csv(
    "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv",
    index_col=0,
    parse_dates=True
)["predicted_regime"]

v3 = v3.loc[START_DATE:]

# align V3 to V4 timeline
v3 = v3.reindex(v4.index, method="ffill")


# ------------------------------------------------------------
# 4. ALIGN SIGNALS
# ------------------------------------------------------------

signal_df = pd.DataFrame({
    "V3": v3,
    "V4": v4
}).dropna()

print("Signal sample")
print(signal_df.head())

print("\nOverlap length:", len(signal_df))


# ------------------------------------------------------------
# ALIGN RETURNS TO SIGNAL TIMELINE
# ------------------------------------------------------------

returns = hourly_returns.reindex(signal_df.index, method="ffill")


# ------------------------------------------------------------
# PORTFOLIO LOGIC
# ------------------------------------------------------------

def get_asset(regime):

    if regime == 0:
        return "QQQ"

    if regime == 1:
        return "SPY"

    if regime == 2:
        return "CASH"

    return "CASH"


# ------------------------------------------------------------
# STRATEGY RETURNS
# ------------------------------------------------------------

def strategy_returns(regime_series):

    strat_returns = []

    for date, regime in regime_series.items():

        if regime == 2:
            r = 0

        else:
            asset = get_asset(regime)

            if asset in returns.columns:
                r = returns.loc[date, asset]
            else:
                r = 0

        strat_returns.append(r)

    return pd.Series(strat_returns, index=regime_series.index)


# ------------------------------------------------------------
# CALCULATE RETURNS
# ------------------------------------------------------------

ret_v3 = strategy_returns(signal_df["V3"])
ret_v4 = strategy_returns(signal_df["V4"])
ret_spy = returns["SPY"].reindex(signal_df.index).fillna(0)


# ------------------------------------------------------------
# EQUITY CURVES
# ------------------------------------------------------------

equity = pd.DataFrame({
    "V3 Strategy": (1+ret_v3).cumprod()*INITIAL_CAPITAL,
    "V4 Walkforward": (1+ret_v4).cumprod()*INITIAL_CAPITAL,
    "SPY": (1+ret_spy).cumprod()*INITIAL_CAPITAL
})


# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------

def perf_metrics(ret):

    CAGR = (1+ret).prod()**(252/len(ret))-1
    vol = ret.std()*np.sqrt(252)

    sharpe = CAGR/vol if vol != 0 else np.nan

    downside = ret[ret<0].std()*np.sqrt(252)
    sortino = CAGR/downside if downside != 0 else np.nan

    drawdown = (1+ret).cumprod()/((1+ret).cumprod().cummax())-1
    maxdd = drawdown.min()

    return pd.Series({
        "CAGR":CAGR,
        "Volatility":vol,
        "Sharpe":sharpe,
        "Sortino":sortino,
        "Max Drawdown":maxdd
    })


metrics = pd.concat([
    perf_metrics(ret_v3),
    perf_metrics(ret_v4),
    perf_metrics(ret_spy)
],axis=1)

metrics.columns = ["V3","V4","SPY"]

print("\nPerformance Metrics")
display(metrics.round(3))


# ------------------------------------------------------------
# EQUITY CURVE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(14,7))

plt.plot(equity.index, equity["V3 Strategy"], label="V3 Strategy")
plt.plot(equity.index, equity["V4 Walkforward"], label="V4 Walkforward")
plt.plot(equity.index, equity["SPY"], label="SPY")

plt.title("Strategy Performance ($5M Initial Capital)")
plt.ylabel("Portfolio Value ($)")
plt.grid(alpha=0.3)
plt.legend()

plt.show()


# ------------------------------------------------------------
# FINAL VALUES
# ------------------------------------------------------------

print("\nFinal Portfolio Value")

display(equity.iloc[-1].round(0))

# COMMAND ----------

if "equity" not in globals():
    raise ValueError("Run the strategy performance section first — equity dataframe not found.")

# COMMAND ----------

# ============================================================
# MASTER STRATEGY COMPARISON
# V3 | V3 WalkForward | V4 | V4 WalkForward | SPY
# 365-DAY ANNUALIZATION (Industry Standard)
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

START_DATE = "2021-01-01"
INITIAL_CAPITAL = 5_000_000
TRADING_DAYS = 365

# ------------------------------------------------------------
# 1. MARKET DATA
# ------------------------------------------------------------

prices = yf.download(
    ["QQQ","SPY"],
    start=START_DATE,
    auto_adjust=True,
    progress=False
)["Close"]

returns = prices.pct_change().dropna()

# ------------------------------------------------------------
# 2. LOAD SIGNALS
# ------------------------------------------------------------

# ---- V4 WALKFORWARD
v4_wf = signals_df.copy()
v4_wf["timestamp"] = pd.to_datetime(v4_wf["timestamp"])
v4_wf = v4_wf.set_index("timestamp")["predicted_regime"]
v4_wf = v4_wf.loc[START_DATE:]

# ---- V4 ORIGINAL
v4_orig = pd.read_csv(
    "/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv",
    parse_dates=["timestamp"]
)

v4_orig = v4_orig.set_index("timestamp")["predicted_regime"]
v4_orig = v4_orig.loc[START_DATE:]

# ---- V3 WALKFORWARD
v3_wf = pd.read_csv(
    "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv",
    index_col=0,
    parse_dates=True
)["predicted_regime"]

v3_wf = v3_wf.loc[START_DATE:]

# ---- V3 ORIGINAL
v3_orig = pd.read_csv(
    "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv",
    index_col=0,
    parse_dates=True
)["predicted_regime"]

v3_orig = v3_orig.loc[START_DATE:]

# ------------------------------------------------------------
# ALIGN SIGNALS
# ------------------------------------------------------------

timeline = returns.index

signals = pd.DataFrame({
    "V3": v3_orig.reindex(timeline, method="ffill"),
    "V3_WF": v3_wf.reindex(timeline, method="ffill"),
    "V4": v4_orig.reindex(timeline, method="ffill"),
    "V4_WF": v4_wf.reindex(timeline, method="ffill")
})

# ------------------------------------------------------------
# STRATEGY LOGIC
# ------------------------------------------------------------

def get_asset(regime):

    if regime == 0:
        return "QQQ"

    if regime == 1:
        return "SPY"

    return None


def strategy_returns(regime_series):

    r = []

    for date, regime in regime_series.items():

        asset = get_asset(regime)

        if asset is None:
            r.append(0)
        else:
            r.append(returns.loc[date, asset])

    return pd.Series(r, index=regime_series.index)

# ------------------------------------------------------------
# CALCULATE RETURNS
# ------------------------------------------------------------

ret_v3 = strategy_returns(signals["V3"])
ret_v3wf = strategy_returns(signals["V3_WF"])
ret_v4 = strategy_returns(signals["V4"])
ret_v4wf = strategy_returns(signals["V4_WF"])
ret_spy = returns["SPY"].reindex(signals.index).fillna(0)

# ------------------------------------------------------------
# EQUITY CURVES
# ------------------------------------------------------------

equity = pd.DataFrame({
    "V3 Original": (1+ret_v3).cumprod()*INITIAL_CAPITAL,
    "V3 WalkForward": (1+ret_v3wf).cumprod()*INITIAL_CAPITAL,
    "V4 Original": (1+ret_v4).cumprod()*INITIAL_CAPITAL,
    "V4 WalkForward": (1+ret_v4wf).cumprod()*INITIAL_CAPITAL,
    "SPY": (1+ret_spy).cumprod()*INITIAL_CAPITAL
})

# ------------------------------------------------------------
# PERFORMANCE METRICS (365-DAY STANDARD)
# ------------------------------------------------------------

def perf(series):

    r = series.pct_change().dropna()

    years = (series.index[-1] - series.index[0]).days / 365

    total_return = series.iloc[-1] / series.iloc[0] - 1

    cagr = (series.iloc[-1] / series.iloc[0])**(1/years) - 1

    vol = r.std() * np.sqrt(TRADING_DAYS)

    sharpe = r.mean()/r.std() * np.sqrt(TRADING_DAYS)

    downside = r[r<0].std() * np.sqrt(TRADING_DAYS)
    sortino = r.mean()/downside * np.sqrt(TRADING_DAYS)

    dd = series/series.cummax() - 1
    maxdd = dd.min()

    return pd.Series({
        "Final Value":series.iloc[-1],
        "Total Return":total_return,
        "CAGR":cagr,
        "Volatility":vol,
        "Sharpe":sharpe,
        "Sortino":sortino,
        "Max Drawdown":maxdd
    })

metrics = pd.DataFrame({
    col: perf(equity[col])
    for col in equity.columns
}).T

metrics_out = metrics.round(4).copy()
metrics_out.insert(0, "Strategy", metrics_out.index)

print("\n================ PERFORMANCE SUMMARY ================\n")
display(metrics_out)


print("\n================ PERFORMANCE SUMMARY ================\n")
display(metrics.round(4))

# ------------------------------------------------------------
# RETURNS TABLES
# ------------------------------------------------------------

daily_returns = equity.pct_change().dropna()

weekly_returns = (
    daily_returns
    .resample("W-FRI")
    .apply(lambda x: (1+x).prod()-1)
)

monthly_returns = (
    daily_returns
    .resample("ME")
    .apply(lambda x: (1+x).prod()-1)
)

yearly_returns = (
    daily_returns
    .resample("YE")
    .apply(lambda x: (1+x).prod()-1)
)

print("\nDaily Returns Sample\n")
display(daily_returns.head())

print("\nWeekly Returns Sample\n")
display(weekly_returns.head())

print("\nMonthly Returns Sample\n")
display(monthly_returns.head())

print("\nYearly Returns\n")
display(yearly_returns)

# ------------------------------------------------------------
# EXPORT RETURNS
# ------------------------------------------------------------

EXPORT_PATH="/Volumes/pawan_quant_research_1/v4/market_data"

daily_returns.to_csv(f"{EXPORT_PATH}/daily_returns_all_models.csv")
weekly_returns.to_csv(f"{EXPORT_PATH}/weekly_returns_all_models.csv")
monthly_returns.to_csv(f"{EXPORT_PATH}/monthly_returns_all_models.csv")
yearly_returns.to_csv(f"{EXPORT_PATH}/yearly_returns_all_models.csv")

metrics.to_csv(f"{EXPORT_PATH}/strategy_comparison_metrics.csv")

print("\nCSV files exported.")

# ------------------------------------------------------------
# EQUITY CURVE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(15,7))

for col in equity.columns:
    plt.plot(equity.index,equity[col],label=col)

plt.yscale("log")

plt.title("Strategy Performance Comparison ($5M Start)")
plt.ylabel("Portfolio Value")

plt.grid(alpha=0.3)
plt.legend()

plt.show()

# COMMAND ----------
