# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %pip install xgboost alpaca-py --quiet

# COMMAND ----------

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

# DBTITLE 1,Cell 3
# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

CATALOG = "pawan_quant_research_1"
SCHEMA = "default"

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"

LABEL_PATH = "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_regime_signals-v4.csv"

TRAIN_START = "2016-01-01"
LIVE_START = "2021-01-01"
LIVE_END = "2026-04-01"  # Extended to today (was 2026-01-01)

FEATURES = [
"CPIAUCSL_z","PCEPI_z","PCE_z","PAYEMS_z","UNRATE_z",
"GDPC1_z","INDPRO_z","UMCSENT_z","VIXCLS_z","vix3m_z",
"VIX_Inversion_Z","Price_Mom_24H","CPI_Event_Active","NFP_Event_Active"
]

TARGET = "regime"

print("Configuration loaded")
print(f"LIVE_END extended to: {LIVE_END}")

# COMMAND ----------

# DBTITLE 1,Step 0a: Incremental Market Data Refresh
# ------------------------------------------------------------
# INCREMENTAL MARKET DATA REFRESH (Alpaca API)
# Fetches only missing hourly ETF data since last ingestion.
# ------------------------------------------------------------

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
import pandas as pd
import time

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/bronze_bt_market_1h"

# --- Alpaca credentials ---
try:
    API_KEY = dbutils.secrets.get(scope="alpaca", key="api_key_id")
    API_SECRET = dbutils.secrets.get(scope="alpaca", key="api_secret_key")
    print("✅ Credentials loaded from Secrets")
except:
    API_KEY = "PKPSF55ZI6V53R25GIBFTUGB4B"
    API_SECRET = "5MnZMUeKU8EFaXFV6dHGUM1uS6kG8AgPXqCnaSjHVE2U"
    print("⚠️  Using Fallback Credentials")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

# --- Check existing data range ---
existing_df = spark.read.format("delta").load(VOLUME_PATH)
existing_max = existing_df.agg(F.max("timestamp")).collect()[0][0]
ALL_TICKERS = [r["symbol"] for r in existing_df.select("symbol").distinct().collect()]

print(f"📊 Existing data up to: {existing_max}")
print(f"📊 Symbols: {ALL_TICKERS}")

# --- Fetch only from the year of existing max onwards ---
incremental_start_year = existing_max.year
END_YEAR = 2026

hourly_schema = StructType([
    StructField("symbol", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
    StructField("trade_count", DoubleType(), True),
    StructField("vwap", DoubleType(), True)
])

def fetch_hourly_data(symbol_list, year_val):
    start = datetime(year_val, 1, 1, tzinfo=ZoneInfo("UTC"))
    end = datetime(year_val, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC"))
    max_allowed_time = datetime.now(ZoneInfo("UTC")) - timedelta(minutes=16)
    if end > max_allowed_time:
        end = max_allowed_time
    if start > end:
        return None
    req = StockBarsRequest(
        symbol_or_symbols=symbol_list,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end
    )
    try:
        bars = client.get_stock_bars(req)
        if bars.df is None or bars.df.empty: return None
        return bars.df.reset_index()
    except Exception as e:
        print(f"\n      ⚠️ API Error: {str(e)}")
        return None

print(f"🚀 Incremental fetch: years {incremental_start_year}–{END_YEAR}")

for yr in range(incremental_start_year, END_YEAR + 1):
    print(f"📅 Processing {yr}...", end=" ")
    pdf = fetch_hourly_data(ALL_TICKERS, yr)
    if pdf is not None and not pdf.empty:
        sdf = spark.createDataFrame(pdf, schema=hourly_schema)
        sdf = (sdf
            .withColumn("year_partition", F.year("timestamp").cast("integer"))
            .withColumn("ingestion_ts", F.current_timestamp())
        )
        (sdf.write
            .mode("overwrite")
            .option("replaceWhere", f"year_partition = {yr}")
            .format("delta")
            .save(VOLUME_PATH)
        )
        print(f"✅ {len(pdf)} rows saved.")
    else:
        print("⊘ No data.")

# --- Verify ---
new_max = spark.read.format("delta").load(VOLUME_PATH).agg(F.max("timestamp")).collect()[0][0]
print(f"\n🎯 Market data now extends to: {new_max}")

# COMMAND ----------

# DBTITLE 1,Step 0b: Rebuild Silver Master Grid
# ------------------------------------------------------------
# VERIFY SILVER TABLE IS CURRENT
# (Rebuilt by Macro_Ingestion job with fresh market data)
# If stale, uncomment and run the full rebuild below.
# ------------------------------------------------------------

silver_check = spark.sql(f"SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as rows FROM {SILVER_TABLE}")
silver_check.show(truncate=False)
print("✅ Silver table is up to date")

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
    "pawan_quant_research_1.default.bronze_bt_market_1h"
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

V4_SIGNAL_PATH = "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_regime_signals-v4.csv"

v4 = pd.read_csv(V4_SIGNAL_PATH, parse_dates=["timestamp"])

v4 = v4.set_index("timestamp")


# ------------------------------------------------------------
# LOAD MARKET DATA
# ------------------------------------------------------------

prices = spark.table(
"pawan_quant_research_1.default.bronze_bt_market_1h"
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

EXPORT_PATH = "/Volumes/pawan_quant_research_1/default/market_data"

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
# ENSURE SIGNALS DATA EXISTS
# ============================================================

if "signals_df" not in globals():

    print("Reloading signals from Delta table...")

    SIGNAL_TABLE = "pawan_quant_research_1.default.bt_walkforward_regime_signals"

    signals_df = spark.table(SIGNAL_TABLE).toPandas()

    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

    print("Signals restored:", len(signals_df))

# COMMAND ----------

# ============================================================
# MASTER STRATEGY COMPARISON
# SPY | V3 Original | V3 WalkForward | V4 Original | V4 WalkForward
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

START_DATE = "2021-01-01"
INITIAL_CAPITAL = 5_000_000
TRADING_DAYS = 365

# ------------------------------------------------------------
# MARKET DATA
# ------------------------------------------------------------

prices = yf.download(
    ["QQQ","SPY"],
    start=START_DATE,
    auto_adjust=True,
    progress=False
)["Close"]

returns = prices.pct_change().dropna()

# ------------------------------------------------------------
# LOAD V4 WALKFORWARD SIGNALS
# ------------------------------------------------------------

v4_wf = signals_df.copy()
v4_wf["timestamp"] = pd.to_datetime(v4_wf["timestamp"])
v4_wf = v4_wf.set_index("timestamp")["predicted_regime"]
v4_wf = v4_wf.resample("D").ffill()
v4_wf = v4_wf.loc[START_DATE:]

# ------------------------------------------------------------
# LOAD V4 ORIGINAL
# ------------------------------------------------------------

v4_orig = pd.read_csv(
    "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_regime_signals-v4.csv",
    parse_dates=["timestamp"]
)

v4_orig = v4_orig.set_index("timestamp")["predicted_regime"]
v4_orig = v4_orig.resample("D").ffill()
v4_orig = v4_orig.loc[START_DATE:]

# ------------------------------------------------------------
# LOAD V3 WALKFORWARD
# ------------------------------------------------------------

v3_wf = pd.read_csv(
    "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv",
    index_col=0,
    parse_dates=True
)["predicted_regime"]

v3_wf = v3_wf.resample("D").ffill()
v3_wf = v3_wf.loc[START_DATE:]

# ------------------------------------------------------------
# LOAD V3 ORIGINAL (WEEKLY MODEL)
# ------------------------------------------------------------

v3_orig = pd.read_csv(
    "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv",
    index_col=0,
    parse_dates=True
)["predicted_regime"]

v3_orig = v3_orig.resample("W-FRI").last()
v3_orig = v3_orig.reindex(returns.index,method="ffill")

# ------------------------------------------------------------
# ALIGN SIGNALS
# ------------------------------------------------------------

timeline = returns.index

signals = pd.DataFrame({

    "V3 Original": v3_orig.reindex(timeline,method="ffill"),

    "V3 WalkForward": v3_wf.reindex(timeline,method="ffill"),

    "V4 Original": v4_orig.reindex(timeline,method="ffill"),

    "V4 WalkForward": v4_wf.reindex(timeline,method="ffill")

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

    r=[]

    for date,regime in regime_series.items():

        asset=get_asset(regime)

        if asset is None:

            r.append(0)

        else:

            r.append(returns.loc[date,asset])

    return pd.Series(r,index=regime_series.index)

# ------------------------------------------------------------
# RETURNS
# ------------------------------------------------------------

ret_v3 = strategy_returns(signals["V3 Original"])
ret_v3wf = strategy_returns(signals["V3 WalkForward"])
ret_v4 = strategy_returns(signals["V4 Original"])
ret_v4wf = strategy_returns(signals["V4 WalkForward"])
ret_spy = returns["SPY"].reindex(signals.index).fillna(0)

# ------------------------------------------------------------
# EQUITY CURVES
# ------------------------------------------------------------

equity = pd.DataFrame({

    "SPY":(1+ret_spy).cumprod()*INITIAL_CAPITAL,

    "V3 Original":(1+ret_v3).cumprod()*INITIAL_CAPITAL,

    "V3 WalkForward":(1+ret_v3wf).cumprod()*INITIAL_CAPITAL,

    "V4 Original":(1+ret_v4).cumprod()*INITIAL_CAPITAL,

    "V4 WalkForward":(1+ret_v4wf).cumprod()*INITIAL_CAPITAL

})

equity = equity.sort_index()

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

print("\nFinal Portfolio Value")

display(equity.iloc[-1].round(0))

# COMMAND ----------

# ============================================================
# PERFORMANCE METRICS
# ============================================================

def perf(series):

    r = series.pct_change().dropna()

    years = (series.index[-1] - series.index[0]).days / 365

    total_return = series.iloc[-1] / series.iloc[0] - 1

    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

    vol = r.std() * np.sqrt(TRADING_DAYS)

    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS)

    #downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)

    #sortino = r.mean() / downside * np.sqrt(TRADING_DAYS)

    # downside deviation (institutional definition)
    downside_returns = np.minimum(0, r)

    downside_dev = np.sqrt((downside_returns**2).mean()) * np.sqrt(TRADING_DAYS)

    sortino = (r.mean() * TRADING_DAYS) / downside_dev

    dd = series / series.cummax() - 1

    maxdd = dd.min()

    return pd.Series({
        "Final Value": series.iloc[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": maxdd
    })

metrics = pd.DataFrame({
    col: perf(equity[col])
    for col in equity.columns
}).T

metrics_fmt = metrics.copy()
metrics_fmt.insert(0, "Strategy", metrics_fmt.index)

def fmt(x, pct=False, comma=False, decimals=2):
    if pct:
        return f"{x*100:.{decimals}f}%"
    if comma:
        return f"{x:,.0f}"
    return f"{x:.{decimals}f}"

metrics_fmt["Final Value"] = metrics["Final Value"].apply(lambda x: fmt(x, comma=True))
metrics_fmt["Total Return"] = metrics["Total Return"].apply(lambda x: fmt(x, pct=True, decimals=2))
metrics_fmt["CAGR"] = metrics["CAGR"].apply(lambda x: fmt(x, pct=True, decimals=2))
metrics_fmt["Volatility"] = metrics["Volatility"].apply(lambda x: fmt(x, pct=True, decimals=2))
metrics_fmt["Sharpe"] = metrics["Sharpe"].apply(lambda x: fmt(x, decimals=2))
metrics_fmt["Sortino"] = metrics["Sortino"].apply(lambda x: fmt(x, decimals=2))
metrics_fmt["Max Drawdown"] = metrics["Max Drawdown"].apply(lambda x: fmt(x, pct=True, decimals=2))

display(metrics_fmt)

# COMMAND ----------

# ============================================================
# TRAINING WINDOW PERFORMANCE
# ============================================================

training_years = [2020, 2021, 2022, 2023, 2024]

results = []

for year in training_years:

    start = pd.Timestamp(f"{year+1}-01-01")

    subset = equity.loc[start:]

    years = (subset.index[-1] - subset.index[0]).days / 365

    for strat in equity.columns:

        r = subset[strat].pct_change().dropna()

        cagr = (subset[strat].iloc[-1] / subset[strat].iloc[0]) ** (1 / years) - 1

        sharpe = r.mean() / r.std() * np.sqrt(365)

        results.append({
            "Training End Year": year,
            "Strategy": strat,
            "CAGR": cagr,
            "Sharpe": sharpe
        })

training_perf = pd.DataFrame(results)

def fmt(x, pct=False, comma=False, decimals=2):
    if pct:
        return f"{x*100:.{decimals}f}%"
    if comma:
        return f"{x:,.0f}"
    return f"{x:.{decimals}f}"

training_perf_fmt = training_perf.copy()
training_perf_fmt["CAGR"] = training_perf["CAGR"].apply(lambda x: fmt(x, pct=True, decimals=2))
training_perf_fmt["Sharpe"] = training_perf["Sharpe"].apply(lambda x: fmt(x, decimals=2))

display(training_perf_fmt)

# COMMAND ----------

daily_returns = equity.pct_change().dropna()

weekly_returns = daily_returns.resample("W-FRI").apply(lambda x:(1+x).prod()-1)
monthly_returns = daily_returns.resample("ME").apply(lambda x:(1+x).prod()-1)
yearly_returns = daily_returns.resample("YE").apply(lambda x:(1+x).prod()-1)

def fmt(x):
    return f"{x*100:,.2f}%"

yearly_returns_fmt = yearly_returns.applymap(fmt)

display(yearly_returns_fmt)

# COMMAND ----------

# ============================================================
# EXPORT CSV FILES
# ============================================================

EXPORT_PATH="/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_20260307/01_Backtesting/04_Backtesting/WF_csvs"

equity.to_csv(f"{EXPORT_PATH}/equity_curves.csv")

metrics.to_csv(f"{EXPORT_PATH}/performance_metrics.csv")

daily_returns.to_csv(f"{EXPORT_PATH}/daily_returns.csv")

weekly_returns.to_csv(f"{EXPORT_PATH}/weekly_returns.csv")

monthly_returns.to_csv(f"{EXPORT_PATH}/monthly_returns.csv")

yearly_returns.to_csv(f"{EXPORT_PATH}/yearly_returns.csv")

training_perf.to_csv(f"{EXPORT_PATH}/training_window_performance.csv")

print("CSV files exported to:",EXPORT_PATH)

# COMMAND ----------

# ============================================================
# TRADE LOG — V4 WALK FORWARD
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# STEP 1: Get a clean copy of signals with prices merged in
# ------------------------------------------------------------

trade_signals = signals_df[["timestamp", "predicted_regime"]].copy()
trade_signals["timestamp"] = pd.to_datetime(trade_signals["timestamp"])
trade_signals = trade_signals.sort_values("timestamp").reset_index(drop=True)

# Load prices — use whichever source is available in your session
# Option A: from Spark (hourly, most precise)
prices_raw = spark.table(
    "pawan_quant_research_1.default.bronze_bt_market_1h"
).filter(
    "symbol IN ('SPY','QQQ')"
).toPandas()

prices_raw["timestamp"] = pd.to_datetime(prices_raw["timestamp"])

prices_pivot = prices_raw.pivot_table(
    index="timestamp",
    columns="symbol",
    values="close",
    aggfunc="last"
).reset_index()

# Merge prices onto signals
trade_signals = trade_signals.merge(prices_pivot, on="timestamp", how="left")
trade_signals["SPY"] = trade_signals["SPY"].ffill()
trade_signals["QQQ"] = trade_signals["QQQ"].ffill()

print("Columns available:", trade_signals.columns.tolist())
print("Rows:", len(trade_signals))
print(trade_signals.head())

# ------------------------------------------------------------
# STEP 2: Build trade log
# ------------------------------------------------------------

trade_log    = []
active_asset = "START"
capital      = 5_000_000

for i in range(1, len(trade_signals)):

    regime    = trade_signals["predicted_regime"].iloc[i - 1]
    timestamp = trade_signals["timestamp"].iloc[i]

    if regime == 0:
        target = "QQQ"
    elif regime == 1:
        target = "SPY"
    else:
        target = "CASH"

    price_now  = trade_signals[target].iloc[i]   if target != "CASH" else None
    price_prev = trade_signals[target].iloc[i-1] if target != "CASH" else None

    ret  = (price_now / price_prev - 1) if target != "CASH" else 0
    cost = 0.0005 if target != active_asset else 0

    if target != active_asset:
        trade_log.append({
            "timestamp"        : timestamp,
            "from_asset"       : active_asset,
            "to_asset"         : target,
            "regime"           : int(regime),
            "price_entry"      : price_now,
            "transaction_cost" : cost,
            "portfolio_value"  : round(capital, 2)
        })

    capital      *= (1 + ret - cost)
    active_asset  = target

trades_df = pd.DataFrame(trade_log)
trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

print(f"\nTotal regime switches (trades): {len(trades_df)}")
print(f"\nTrade breakdown:\n{trades_df['to_asset'].value_counts()}")
display(trades_df)

# COMMAND ----------

# ============================================================
# FAST RELOAD (NO MODEL TRAINING)
# ============================================================

import pandas as pd

# 1. LOAD SIGNALS FROM DELTA
signals_df = spark.table(SIGNAL_TABLE).toPandas()
signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

print("Signals loaded:", len(signals_df))


# 2. LOAD RETURNS (already exported earlier)
BASE_PATH = "/Volumes/pawan_quant_research_1/default/market_data"

daily_returns   = pd.read_csv(f"{BASE_PATH}/strategy_daily_returns.csv", index_col=0, parse_dates=True)
weekly_returns  = pd.read_csv(f"{BASE_PATH}/strategy_weekly_returns.csv", index_col=0, parse_dates=True)
monthly_returns = pd.read_csv(f"{BASE_PATH}/strategy_monthly_returns.csv", index_col=0, parse_dates=True)
yearly_returns  = pd.read_csv(f"{BASE_PATH}/strategy_yearly_returns.csv", index_col=0, parse_dates=True)

metrics = pd.read_csv(f"{BASE_PATH}/strategy_performance_summary.csv", index_col=0)

print("Returns + metrics loaded")

# COMMAND ----------

# ============================================================
# SAFE EXPORT BLOCK (NO RE-RUN REQUIRED)
# ============================================================

import os

DATE_PREFIX = "20260331"

BASE_DIR = os.getcwd()
EXPORT_DIR = os.path.join(BASE_DIR, f"{DATE_PREFIX}_outputs")

os.makedirs(EXPORT_DIR, exist_ok=True)

print("Export directory:", EXPORT_DIR)


# ------------------------------------------------------------
# HELPER FUNCTION
# ------------------------------------------------------------

def safe_export(df, filename, index=True):
    try:
        if df is not None:
            path = os.path.join(EXPORT_DIR, f"{DATE_PREFIX}_{filename}")
            df.to_csv(path, index=index)
            print(f"✅ Saved: {filename}")
    except Exception as e:
        print(f"❌ Skipped {filename} | Reason: {str(e)}")


# ------------------------------------------------------------
# EXPORT CHECKS (ONLY IF VARIABLE EXISTS)
# ------------------------------------------------------------

vars_available = globals()

# RETURNS
safe_export(vars_available.get("daily_returns"), "daily_returns.csv")
safe_export(vars_available.get("weekly_returns"), "weekly_returns.csv")
safe_export(vars_available.get("monthly_returns"), "monthly_returns.csv")
safe_export(vars_available.get("yearly_returns"), "yearly_returns.csv")

# PERFORMANCE
safe_export(vars_available.get("metrics"), "performance_metrics.csv")

# EQUITY
safe_export(vars_available.get("equity"), "equity_curves.csv")

# COMPARISON
safe_export(vars_available.get("comparison"), "strategy_comparison.csv")

# TRADES
safe_export(vars_available.get("trades_df"), "v4_trades.csv", index=False)
safe_export(vars_available.get("v3_trades_df"), "v3_trades.csv", index=False)

# SIGNALS
safe_export(vars_available.get("signals_df"), "walkforward_signals.csv", index=False)

# FEATURE IMPORTANCE
safe_export(vars_available.get("fi_matrix"), "feature_importance_time.csv")
safe_export(vars_available.get("importance_matrix"), "feature_importance_by_regime.csv")
safe_export(vars_available.get("fi_df"), "feature_importance_raw.csv", index=False)

# CONFUSION MATRICES (if present)
safe_export(vars_available.get("cm_overall"), "confusion_matrix_overall.csv")
safe_export(vars_available.get("cm_yearly"), "confusion_matrix_yearly.csv")
safe_export(vars_available.get("cm_monthly"), "confusion_matrix_monthly.csv")

print("\n✅ SAFE EXPORT COMPLETE")
print(f"📁 Location: {EXPORT_DIR}")

# COMMAND ----------

# DBTITLE 1,Export 20260401 — FINAL CSVs (daily/weekly/monthly/quarterly/yearly)
# ============================================================
# EXPORT 20260401 — FINAL CSVs
# daily, weekly, monthly, quarterly, yearly returns
# + performance, equity, signals, trades, feature importance
# ============================================================

import os
import pandas as pd
import numpy as np

DATE_PREFIX = "20260401"
NOTEBOOK_DIR = "/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/04_Backtesting"
EXPORT_DIR = os.path.join(NOTEBOOK_DIR, DATE_PREFIX)
os.makedirs(EXPORT_DIR, exist_ok=True)
print(f"Export directory: {EXPORT_DIR}\n")

# ── helper ────────────────────────────────────────────────────
def save(df, name, index=True):
    if df is not None and len(df) > 0:
        path = os.path.join(EXPORT_DIR, f"{DATE_PREFIX}_FINAL_{name}")
        df.to_csv(path, index=index)
        print(f"  ✅  {DATE_PREFIX}_FINAL_{name}  ({len(df)} rows)")
    else:
        print(f"  ⚠️  Skipped {name} (empty or missing)")

# ============================================================
# 1. QUARTERLY RETURNS  (not in original pipeline)
# ============================================================
quarterly_returns = (
    daily_returns
    .resample("QE")
    .apply(lambda x: (1 + x).prod() - 1)
)

# ============================================================
# 2. TRADE LOG  (rebuild from in-memory signals_df)
# ============================================================
trade_signals = signals_df[["timestamp", "predicted_regime", "SPY", "QQQ"]].copy()
trade_signals = trade_signals.sort_values("timestamp").reset_index(drop=True)

trade_log = []
active_asset = "START"
capital = 5_000_000

for i in range(1, len(trade_signals)):
    regime = trade_signals["predicted_regime"].iloc[i - 1]
    target = "QQQ" if regime == 0 else ("SPY" if regime == 1 else "CASH")
    price_now  = trade_signals[target].iloc[i]   if target != "CASH" else None
    price_prev = trade_signals[target].iloc[i-1] if target != "CASH" else None
    ret  = (price_now / price_prev - 1) if target != "CASH" else 0
    cost = 0.0005 if target != active_asset else 0
    if target != active_asset:
        trade_log.append({
            "timestamp": trade_signals["timestamp"].iloc[i],
            "from_asset": active_asset, "to_asset": target,
            "regime": int(regime), "price_entry": price_now,
            "transaction_cost": cost, "portfolio_value": round(capital, 2)
        })
    capital *= (1 + ret - cost)
    active_asset = target

trades_df = pd.DataFrame(trade_log)

# ============================================================
# 3. EXPORT ALL CSVs
# ============================================================
print("── RETURNS ──")
save(daily_returns,     "daily_returns.csv")
save(weekly_returns,    "weekly_returns.csv")
save(monthly_returns,   "monthly_returns.csv")
save(quarterly_returns, "quarterly_returns.csv")
save(yearly_returns,    "yearly_returns.csv")

print("\n── PERFORMANCE ──")
save(metrics,           "performance_metrics.csv")
save(comparison,        "equity_curves.csv")

print("\n── SIGNALS & TRADES ──")
save(signals_df,        "walkforward_signals.csv", index=False)
save(trades_df,         "v4_trades.csv",           index=False)

print("\n── FEATURE IMPORTANCE ──")
save(fi_matrix,         "feature_importance_time.csv")
save(fi_df,             "feature_importance_raw.csv", index=False)
try:
    save(importance_matrix, "feature_importance_by_regime.csv")
except NameError:
    print("  ⚠️  Skipped feature_importance_by_regime (not in session)")

# ============================================================
# 4. SUMMARY
# ============================================================
print(f"\n{'='*55}")
print(f"✅  EXPORT COMPLETE → {EXPORT_DIR}")
print(f"{'='*55}")
print(f"Signal range: {signals_df.timestamp.min()} → {signals_df.timestamp.max()}")
print(f"Files: {len([f for f in os.listdir(EXPORT_DIR) if f.endswith('.csv')])} CSVs")
for f in sorted(os.listdir(EXPORT_DIR)):
    size_kb = os.path.getsize(os.path.join(EXPORT_DIR, f)) / 1024
    print(f"  📄 {f}  ({size_kb:.0f} KB)")

# COMMAND ----------

pip install yfinance

# COMMAND ----------

# ============================================================
# BUILD FINAL ANNUAL DATASET: V4 WF vs V3 WF vs SPY
# NO MODEL RERUN REQUIRED
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os

DATE_PREFIX = "20260331"

# ------------------------------------------------------------
# 1. LOAD EXISTING ANNUAL RETURNS (V4 WF + SPY)
# ------------------------------------------------------------

BASE_DIR = os.getcwd()
INPUT_FILE = "/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/04_Backtesting/20260331_outputs/20260331_yearly_returns.csv"
#os.path.join(BASE_DIR, f"{DATE_PREFIX}_yearly_returns.csv")

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.set_index("timestamp")

print("Loaded existing annual returns:")
print(df.head())


# ------------------------------------------------------------
# 2. LOAD V3 WALKFORWARD SIGNALS
# ------------------------------------------------------------

V3_PATH = "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv"

v3 = pd.read_csv(
    V3_PATH,
    index_col=0,
    parse_dates=True
)["predicted_regime"]

v3 = v3.sort_index()

print("\nLoaded V3 signals:", len(v3))


# ------------------------------------------------------------
# 3. LOAD MARKET DATA
# ------------------------------------------------------------

prices = yf.download(
    ["QQQ", "SPY"],
    start="2021-01-01",
    auto_adjust=True,
    progress=False
)["Close"]

returns = prices.pct_change().dropna()

print("Market data loaded:", returns.shape)


# ------------------------------------------------------------
# 4. ALIGN V3 TO DAILY FREQUENCY
# ------------------------------------------------------------

v3 = v3.resample("D").ffill()
v3 = v3.reindex(returns.index, method="ffill")


# ------------------------------------------------------------
# 5. BUILD V3 WALKFORWARD RETURNS
# ------------------------------------------------------------

def get_asset(regime):
    if regime == 0:
        return "QQQ"
    elif regime == 1:
        return "SPY"
    else:
        return None

ret_v3 = []

for date, regime in v3.items():
    asset = get_asset(regime)

    if asset is None:
        r = 0
    else:
        r = returns.loc[date, asset]

    ret_v3.append(r)

ret_v3 = pd.Series(ret_v3, index=v3.index)


# ------------------------------------------------------------
# 6. CONVERT TO ANNUAL RETURNS
# ------------------------------------------------------------

v3_annual = (
    ret_v3
    .resample("YE")
    .apply(lambda x: (1 + x).prod() - 1)
)

v3_annual.name = "WalkForward_V3"

print("\nV3 Annual Returns:")
print(v3_annual)


# ------------------------------------------------------------
# 7. MERGE WITH EXISTING DATA
# ------------------------------------------------------------

df["WalkForward_V3"] = v3_annual

# Keep only required columns for PPT
final_df = df[["WalkForward_V4", "WalkForward_V3", "SPY"]]

# Drop any incomplete rows (alignment safety)
final_df = final_df.dropna()

print("\nFinal dataset:")
print(final_df)


# ------------------------------------------------------------
# 8. SAVE FINAL CSV
# ------------------------------------------------------------

OUTPUT_FILE = os.path.join(BASE_DIR, f"{DATE_PREFIX}_yearly_returns_FINAL.csv")

final_df.reset_index().to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ FINAL FILE SAVED: {OUTPUT_FILE}")


# ------------------------------------------------------------
# 9. CREATE BAR CHART (SLIDE 7 READY)
# ------------------------------------------------------------

final_df.plot(kind="bar", figsize=(12,6))

plt.title("Annual Returns: V4 WF vs V3 WF vs SPY")
plt.ylabel("Return")
plt.xlabel("Year")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# ============================================================
# BUILD D/W/M/Y RETURNS FOR:
# V4 WALKFORWARD vs V3 WALKFORWARD vs SPY
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import os

DATE_PREFIX = "20260331"
BASE_DIR = os.getcwd()

# ------------------------------------------------------------
# 1. LOAD V4 WALKFORWARD SIGNALS (DELTA)
# ------------------------------------------------------------

signals_df = spark.table(SIGNAL_TABLE).toPandas()
signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
signals_df = signals_df.sort_values("timestamp")

v4 = signals_df.set_index("timestamp")["predicted_regime"]


# ------------------------------------------------------------
# 2. LOAD V3 WALKFORWARD SIGNALS
# ------------------------------------------------------------

V3_PATH = "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv"

v3 = pd.read_csv(
    V3_PATH,
    index_col=0,
    parse_dates=True
)["predicted_regime"]

v3 = v3.sort_index()


# ------------------------------------------------------------
# 3. LOAD MARKET DATA
# ------------------------------------------------------------

prices = yf.download(
    ["QQQ", "SPY"],
    start="2021-01-01",
    auto_adjust=True,
    progress=False
)["Close"]

returns = prices.pct_change().dropna()


# ------------------------------------------------------------
# 4. ALIGN SIGNALS TO DAILY
# ------------------------------------------------------------

v4 = v4.resample("D").ffill().reindex(returns.index, method="ffill")
v3 = v3.resample("D").ffill().reindex(returns.index, method="ffill")


# ------------------------------------------------------------
# 5. STRATEGY RETURN FUNCTION
# ------------------------------------------------------------

def get_asset(regime):
    if regime == 0:
        return "QQQ"
    elif regime == 1:
        return "SPY"
    else:
        return None

def compute_strategy_returns(regime_series):
    strat_returns = []

    for date, regime in regime_series.items():
        asset = get_asset(regime)

        if asset is None:
            r = 0
        else:
            r = returns.loc[date, asset]

        strat_returns.append(r)

    return pd.Series(strat_returns, index=regime_series.index)


# ------------------------------------------------------------
# 6. COMPUTE RETURNS
# ------------------------------------------------------------

ret_v4 = compute_strategy_returns(v4)
ret_v3 = compute_strategy_returns(v3)
ret_spy = returns["SPY"].reindex(ret_v4.index).fillna(0)

strategy_returns = pd.DataFrame({
    "WalkForward_V4": ret_v4,
    "WalkForward_V3": ret_v3,
    "SPY": ret_spy
})


# ------------------------------------------------------------
# 7. AGGREGATE RETURNS
# ------------------------------------------------------------

daily_returns = strategy_returns.copy()

weekly_returns = (
    daily_returns
    .resample("W-FRI")
    .apply(lambda x: (1 + x).prod() - 1)
)

monthly_returns = (
    daily_returns
    .resample("ME")
    .apply(lambda x: (1 + x).prod() - 1)
)

yearly_returns = (
    daily_returns
    .resample("YE")
    .apply(lambda x: (1 + x).prod() - 1)
)


# ------------------------------------------------------------
# 8. SAVE FILES
# ------------------------------------------------------------

daily_returns.to_csv(f"{BASE_DIR}/{DATE_PREFIX}_daily_returns_FINAL.csv")
weekly_returns.to_csv(f"{BASE_DIR}/{DATE_PREFIX}_weekly_returns_FINAL.csv")
monthly_returns.to_csv(f"{BASE_DIR}/{DATE_PREFIX}_monthly_returns_FINAL.csv")
yearly_returns.to_csv(f"{BASE_DIR}/{DATE_PREFIX}_yearly_returns_FINAL_CHECK.csv")

print("\n✅ ALL RETURNS FILES SAVED")
print(BASE_DIR)


# ------------------------------------------------------------
# 9. QUICK PREVIEW
# ------------------------------------------------------------

print("\nYearly Returns Preview:")
print(yearly_returns.tail())

# COMMAND ----------

# ============================================================
# V4 WALK-FORWARD TRADE LOG
# Assumes signals_df already exists from the walk-forward notebook
# ============================================================

import pandas as pd
import numpy as np

INITIAL_CAPITAL = 5_000_000

# ------------------------------------------------------------
# 1. PREP SIGNALS
# ------------------------------------------------------------
trade_signals = signals_df[["timestamp", "predicted_regime"]].copy()
trade_signals["timestamp"] = pd.to_datetime(trade_signals["timestamp"])
trade_signals = trade_signals.sort_values("timestamp").reset_index(drop=True)

# ------------------------------------------------------------
# 2. LOAD PRICES
# ------------------------------------------------------------
prices_raw = (
    spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
    .filter("symbol IN ('SPY','QQQ')")
    .toPandas()
)

prices_raw["timestamp"] = pd.to_datetime(prices_raw["timestamp"])

prices_pivot = (
    prices_raw
    .pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
    .reset_index()
)

trade_signals = trade_signals.merge(prices_pivot, on="timestamp", how="left")
trade_signals["SPY"] = trade_signals["SPY"].ffill()
trade_signals["QQQ"] = trade_signals["QQQ"].ffill()

# ------------------------------------------------------------
# 3. BUILD TRADE LOG
# ------------------------------------------------------------
trade_log = []
active_asset = "START"
capital = INITIAL_CAPITAL

for i in range(1, len(trade_signals)):

    ts = trade_signals.loc[i, "timestamp"]
    regime = int(trade_signals.loc[i - 1, "predicted_regime"])

    if regime == 0:
        target = "QQQ"
    elif regime == 1:
        target = "SPY"
    else:
        target = "CASH"

    if target == "QQQ":
        price_now = trade_signals.loc[i, "QQQ"]
        price_prev = trade_signals.loc[i - 1, "QQQ"]
        ret = price_now / price_prev - 1
    elif target == "SPY":
        price_now = trade_signals.loc[i, "SPY"]
        price_prev = trade_signals.loc[i - 1, "SPY"]
        ret = price_now / price_prev - 1
    else:
        price_now = np.nan
        price_prev = np.nan
        ret = 0.0

    tx_cost = 0.0005 if target != active_asset else 0.0

    if target != active_asset:
        trade_log.append({
            "timestamp": ts,
            "year": ts.year,
            "month": ts.month,
            "from_asset": active_asset,
            "to_asset": target,
            "regime": regime,
            "price_entry": price_now,
            "transaction_cost": tx_cost,
            "portfolio_value_before_trade": round(capital, 2)
        })

    capital *= (1 + ret - tx_cost)
    active_asset = target

trades_df = pd.DataFrame(trade_log)
trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

print(f"Total trades: {len(trades_df)}")
display(trades_df.head(20))

# Optional export
EXPORT_PATH = "/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/04_Backtesting/20260331_outputs"
trades_df.to_csv(f"{EXPORT_PATH}/v4_walkforward_trade_log.csv", index=False)
print(f"Saved: {EXPORT_PATH}/v4_walkforward_trade_log.csv")

# COMMAND ----------

# ============================================================
# ANNUAL RETURNS TABLE + BAR CHART
# WalkForward V4 vs WalkForward V3 vs SPY
# LIMITED TO 2025 ONLY
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

START_DATE = "2021-01-01"
END_DATE   = "2026-03-31"

# ------------------------------------------------------------
# 1. LOAD V4 WALK-FORWARD SIGNALS
# ------------------------------------------------------------
v4 = signals_df.copy()
v4["timestamp"] = pd.to_datetime(v4["timestamp"])
v4 = v4.sort_values("timestamp")
v4 = v4.set_index("timestamp")["predicted_regime"]

# Trim V4 to 2025 end
v4 = v4.loc[:END_DATE]

# ------------------------------------------------------------
# 2. LOAD V3 WALK-FORWARD SIGNALS
# ------------------------------------------------------------
V3_PATH = "/Workspace/Users/pawan.jatale@mubama.group/backup/Global-Dynamic-Factor-Rotation/Global Factors/notebooks/v3_walkforward_predictions.csv"

v3 = pd.read_csv(
    V3_PATH,
    index_col=0,
    parse_dates=True
)["predicted_regime"].sort_index()

# Trim V3 to 2025 end
v3 = v3.loc[:END_DATE]

# ------------------------------------------------------------
# 3. LOAD DAILY MARKET DATA
# ------------------------------------------------------------
prices = yf.download(
    ["QQQ", "SPY"],
    start=START_DATE,
    end="2026-01-01",   # yfinance end is exclusive
    auto_adjust=True,
    progress=False
)["Close"]

returns = prices.pct_change().dropna()

# Extra safety trim
returns = returns.loc[:END_DATE]

# ------------------------------------------------------------
# 4. ALIGN TO DAILY
# ------------------------------------------------------------
v4 = v4.resample("D").ffill().reindex(returns.index, method="ffill")
v3 = v3.resample("D").ffill().reindex(returns.index, method="ffill")

def get_asset(regime):
    if regime == 0:
        return "QQQ"
    elif regime == 1:
        return "SPY"
    return None

def compute_strategy_returns(regime_series):
    out = []
    for dt, regime in regime_series.items():
        asset = get_asset(regime)
        if asset is None or pd.isna(regime):
            out.append(0.0)
        else:
            out.append(returns.loc[dt, asset])
    return pd.Series(out, index=regime_series.index)

ret_v4 = compute_strategy_returns(v4)
ret_v3 = compute_strategy_returns(v3)
ret_spy = returns["SPY"].reindex(ret_v4.index).fillna(0.0)

daily_ret_df = pd.DataFrame({
    "WalkForward_V4": ret_v4,
    "WalkForward_V3": ret_v3,
    "SPY": ret_spy
})

# Final safety trim
daily_ret_df = daily_ret_df.loc[:END_DATE]

# ------------------------------------------------------------
# 5. YEARLY RETURNS
# ------------------------------------------------------------
yearly_returns = (
    daily_ret_df
    .resample("YE")
    .apply(lambda x: (1 + x).prod() - 1)
)

yearly_returns.index = yearly_returns.index.year
yearly_returns.index.name = "Year"

# Keep only 2021–2025
yearly_returns = yearly_returns.loc[2021:2025]

print("Annual returns table (through 2025 only):")
display((yearly_returns * 100).round(2))

# Optional export
EXPORT_PATH = "/Volumes/pawan_quant_research_1/default/market_data"
yearly_returns.to_csv(f"{EXPORT_PATH}/yearly_returns_v4_v3_spy_till_2025.csv")
print(f"Saved: {EXPORT_PATH}/yearly_returns_v4_v3_spy_till_2025.csv")

# ------------------------------------------------------------
# 6. PRESENTATION-READY BAR CHART
# ------------------------------------------------------------
plot_df = (yearly_returns * 100).copy()

ax = plot_df.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.8
)

plt.title("Annual Returns: Walk-Forward V4 vs Walk-Forward V3 vs SPY (2021–2025)", fontsize=14)
plt.ylabel("Return (%)")
plt.xlabel("Year")
plt.axhline(0, color="black", linewidth=1)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Strategy")
plt.tight_layout()
plt.show()

# COMMAND ----------

# ============================================================
# REGIME CASE-STUDY CHARTS
# WalkForward V4 vs SPY with regime strip
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INITIAL_CAPITAL = 5_000_000

# ------------------------------------------------------------
# 1. BUILD WALK-FORWARD EQUITY SERIES
# ------------------------------------------------------------
case_df = signals_df.copy()
case_df["timestamp"] = pd.to_datetime(case_df["timestamp"])
case_df = case_df.sort_values("timestamp")

prices_raw = (
    spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
    .filter("symbol IN ('SPY','QQQ')")
    .toPandas()
)

prices_raw["timestamp"] = pd.to_datetime(prices_raw["timestamp"])

prices_pivot = (
    prices_raw
    .pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
    .reset_index()
)

case_df = case_df.merge(prices_pivot, on="timestamp", how="left")
case_df["SPY"] = case_df["SPY"].ffill()
case_df["QQQ"] = case_df["QQQ"].ffill()

portfolio = [INITIAL_CAPITAL]
active_asset = "START"

for i in range(1, len(case_df)):
    regime = int(case_df.loc[i - 1, "predicted_regime"])

    if regime == 0:
        target = "QQQ"
        ret = case_df.loc[i, "QQQ"] / case_df.loc[i - 1, "QQQ"] - 1
    elif regime == 1:
        target = "SPY"
        ret = case_df.loc[i, "SPY"] / case_df.loc[i - 1, "SPY"] - 1
    else:
        target = "CASH"
        ret = 0.0

    tx_cost = 0.0005 if target != active_asset else 0.0
    portfolio.append(portfolio[-1] * (1 + ret - tx_cost))
    active_asset = target

case_df["WF_Value"] = portfolio
case_df["SPY_Value"] = INITIAL_CAPITAL * (case_df["SPY"] / case_df["SPY"].iloc[0])

# ------------------------------------------------------------
# 2. REUSABLE PLOT FUNCTION
# ------------------------------------------------------------
def plot_regime_case_study(df, start_date, end_date, title):
    zoom = df[(df["timestamp"] >= pd.Timestamp(start_date)) & (df["timestamp"] <= pd.Timestamp(end_date))].copy()

    if zoom.empty:
        print(f"No data for {title}")
        return

    # Normalize both to 100 at window start
    zoom["WF_Norm"] = 100 * zoom["WF_Value"] / zoom["WF_Value"].iloc[0]
    zoom["SPY_Norm"] = 100 * zoom["SPY_Value"] / zoom["SPY_Value"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    ax1.plot(zoom["timestamp"], zoom["WF_Norm"], label="Walk-Forward V4", linewidth=2)
    ax1.plot(zoom["timestamp"], zoom["SPY_Norm"], label="SPY", linewidth=2, alpha=0.75)
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel("Normalized Value (Start = 100)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ts = zoom["timestamp"]
    reg = zoom["predicted_regime"]

    ax2.fill_between(ts, 0, 1, where=(reg == 0), alpha=0.35, label="Expansion / QQQ")
    ax2.fill_between(ts, 0, 1, where=(reg == 1), alpha=0.35, label="Normalization / SPY")
    ax2.fill_between(ts, 0, 1, where=(reg == 2), alpha=0.35, label="Contraction / Cash")

    ax2.set_yticks([])
    ax2.set_ylabel("Regime")
    ax2.legend(loc="upper left", ncol=3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 3. CASE STUDIES
# ------------------------------------------------------------

# 2022 bear / hiking regime
plot_regime_case_study(
    case_df,
    start_date="2022-01-01",
    end_date="2022-12-31",
    title="Case Study 1: 2022 Rate-Hike / Bear Market Regime"
)

# Apr-2025 tariffs window
plot_regime_case_study(
    case_df,
    start_date="2025-04-01",
    end_date="2025-04-30",
    title="Case Study 2: April 2025 Tariff Shock Window"
)

# Feb-2025 to present
plot_regime_case_study(
    case_df,
    start_date="2026-02-25",
    end_date=case_df["timestamp"].max(),
    title="Case Study 3: Feb 2026 to Present"
)

# COMMAND ----------

# ============================================================
# V4 WALK-FORWARD PERFORMANCE METRICS
# CAGR = calendar-day geometric return
# VOL / SHARPE / SORTINO = daily return based
# WINDOW: 2021-01-01 to 2025-12-31
# ============================================================

import pandas as pd
import numpy as np

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"
INITIAL_CAPITAL = 5_000_000
TRADING_DAYS = 252
CALENDAR_DAYS_PER_YEAR = 365.25
RISK_FREE_RATE = 0.0   # set to non-zero if needed

# ------------------------------------------------------------
# 1. PREP SIGNALS
# ------------------------------------------------------------
wf = signals_df.copy()
wf["timestamp"] = pd.to_datetime(wf["timestamp"])
wf = wf.sort_values("timestamp")
wf = wf[(wf["timestamp"] >= START_DATE) & (wf["timestamp"] <= END_DATE)].copy()

# ------------------------------------------------------------
# 2. LOAD PRICES
# ------------------------------------------------------------
prices = (
    spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
    .filter("symbol IN ('SPY','QQQ')")
    .toPandas()
)

prices["timestamp"] = pd.to_datetime(prices["timestamp"])

prices = prices.pivot_table(
    index="timestamp",
    columns="symbol",
    values="close",
    aggfunc="last"
).reset_index()

prices = prices[
    (prices["timestamp"] >= START_DATE) &
    (prices["timestamp"] <= END_DATE)
].copy()

# ------------------------------------------------------------
# 3. MERGE SIGNALS + PRICES
# ------------------------------------------------------------
wf = wf.merge(prices, on="timestamp", how="left")
wf["SPY"] = wf["SPY"].ffill()
wf["QQQ"] = wf["QQQ"].ffill()
wf = wf.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)

# ------------------------------------------------------------
# 4. BUILD WALK-FORWARD PORTFOLIO (hourly logic)
# ------------------------------------------------------------
capital = INITIAL_CAPITAL
portfolio = [capital]
active_asset = "START"

for i in range(1, len(wf)):

    regime = int(wf["predicted_regime"].iloc[i - 1])

    if regime == 0:
        target = "QQQ"
        ret = wf["QQQ"].iloc[i] / wf["QQQ"].iloc[i - 1] - 1

    elif regime == 1:
        target = "SPY"
        ret = wf["SPY"].iloc[i] / wf["SPY"].iloc[i - 1] - 1

    else:
        target = "CASH"
        ret = 0.0

    cost = 0.0005 if target != active_asset else 0.0

    capital *= (1 + ret - cost)
    portfolio.append(capital)
    active_asset = target

wf["WF_Value"] = portfolio
wf["SPY_Value"] = INITIAL_CAPITAL * (wf["SPY"] / wf["SPY"].iloc[0])

wf = wf.set_index("timestamp")

# ------------------------------------------------------------
# 5. DAILY NAV
# ------------------------------------------------------------
daily_nav = wf[["WF_Value", "SPY_Value"]].resample("D").last().dropna()

# ------------------------------------------------------------
# 6. METRICS FUNCTION
# ------------------------------------------------------------
def compute_metrics_from_daily_nav(series: pd.Series) -> pd.Series:
    daily_ret = series.pct_change(fill_method=None).dropna()

    start_val = float(series.iloc[0])
    end_val   = float(series.iloc[-1])

    total_return = end_val / start_val - 1

    calendar_days = (series.index[-1] - series.index[0]).days
    years = calendar_days / CALENDAR_DAYS_PER_YEAR if calendar_days > 0 else np.nan

    # CAGR = geometric, calendar-based
    cagr = (end_val / start_val) ** (1 / years) - 1 if pd.notna(years) and years > 0 else np.nan

    # Daily-vol annualized
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)

    # Daily Sharpe annualized
    ann_mean_excess = (daily_ret.mean() - RISK_FREE_RATE / TRADING_DAYS) * TRADING_DAYS
    sharpe = ann_mean_excess / vol if pd.notna(vol) and vol != 0 else np.nan

    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else np.nan
    sortino = ann_mean_excess / downside_vol if pd.notna(downside_vol) and downside_vol != 0 else np.nan

    dd = series / series.cummax() - 1
    max_dd = dd.min()

    # sanity ratio for debugging only
    cagr_over_vol = cagr / vol if pd.notna(vol) and vol != 0 else np.nan

    return pd.Series({
        "Start Date": series.index[0].date(),
        "End Date": series.index[-1].date(),
        "Calendar Days": calendar_days,
        "Start Value": start_val,
        "Final Value": end_val,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "CAGR / Vol (sanity check)": cagr_over_vol
    })

metrics_df = pd.DataFrame({
    "V4 WalkForward": compute_metrics_from_daily_nav(daily_nav["WF_Value"]),
    "SPY Benchmark": compute_metrics_from_daily_nav(daily_nav["SPY_Value"])
}).T

display(metrics_df)

# ------------------------------------------------------------
# 7. FORMATTED VERSION
# ------------------------------------------------------------
metrics_fmt = metrics_df.copy()

metrics_fmt["Start Value"] = metrics_fmt["Start Value"].map(lambda x: f"{x:,.0f}")
metrics_fmt["Final Value"] = metrics_fmt["Final Value"].map(lambda x: f"{x:,.0f}")
metrics_fmt["Total Return"] = metrics_fmt["Total Return"].map(lambda x: f"{x*100:.2f}%")
metrics_fmt["CAGR"] = metrics_fmt["CAGR"].map(lambda x: f"{x*100:.2f}%")
metrics_fmt["Annual Volatility"] = metrics_fmt["Annual Volatility"].map(lambda x: f"{x*100:.2f}%")
metrics_fmt["Sharpe"] = metrics_fmt["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
metrics_fmt["Sortino"] = metrics_fmt["Sortino"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
metrics_fmt["Max Drawdown"] = metrics_fmt["Max Drawdown"].map(lambda x: f"{x*100:.2f}%")
metrics_fmt["CAGR / Vol (sanity check)"] = metrics_fmt["CAGR / Vol (sanity check)"].map(
    lambda x: f"{x:.2f}" if pd.notna(x) else "NA"
)

display(metrics_fmt)

# COMMAND ----------

# ============================================================
# LOAD WALK-FORWARD SIGNALS WITHOUT RERUNNING MODEL
# ============================================================

import pandas as pd

# OPTION A — preferred: Delta table written by notebook
SIGNAL_TABLE = "pawan_quant_research_1.default.bt_walkforward_regime_signals"

signals_df = spark.table(SIGNAL_TABLE).toPandas()
signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
signals_df = signals_df.sort_values("timestamp").reset_index(drop=True)

print("Loaded signals from Delta table")
print("Rows:", len(signals_df))
print("Range:", signals_df["timestamp"].min(), "->", signals_df["timestamp"].max())
display(signals_df.head())

# COMMAND ----------

# ============================================================
# V4 WALK-FORWARD WITH 5X LEVERAGE DURING EXPANSION ONLY
# Expansion   -> 5x QQQ
# Slowdown    -> 1x SPY
# Contraction -> CASH
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"

INITIAL_CAPITAL = 5_000_000
SWITCH_COST     = 0.0005
EXPANSION_LEV   = 5.0

TRADING_DAYS = 252
CAL_DAYS     = 365.25

# ------------------------------------------------------------
# 1. PREP SIGNALS
# ------------------------------------------------------------
wf_df = signals_df.copy()
wf_df["timestamp"] = pd.to_datetime(wf_df["timestamp"])
wf_df = wf_df.sort_values("timestamp")
wf_df = wf_df[
    (wf_df["timestamp"] >= START_DATE) &
    (wf_df["timestamp"] <= END_DATE)
].copy()

# ------------------------------------------------------------
# 2. LOAD SPY / QQQ PRICES
# ------------------------------------------------------------
prices = (
    spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
    .filter("symbol IN ('SPY','QQQ')")
    .toPandas()
)

prices["timestamp"] = pd.to_datetime(prices["timestamp"])

prices = prices.pivot_table(
    index="timestamp",
    columns="symbol",
    values="close",
    aggfunc="last"
).reset_index()

prices = prices[
    (prices["timestamp"] >= START_DATE) &
    (prices["timestamp"] <= END_DATE)
].copy()

# ------------------------------------------------------------
# 3. MERGE
# ------------------------------------------------------------
wf_df = wf_df.merge(prices, on="timestamp", how="left")
wf_df["SPY"] = wf_df["SPY"].ffill()
wf_df["QQQ"] = wf_df["QQQ"].ffill()
wf_df = wf_df.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)

print("Rows:", len(wf_df))
print("Window:", wf_df["timestamp"].min(), "->", wf_df["timestamp"].max())

# ------------------------------------------------------------
# 4. BASELINE WALK-FORWARD
# ------------------------------------------------------------
def run_wf_baseline(df):
    capital = INITIAL_CAPITAL
    portfolio = [capital]
    active_asset = "START"

    for i in range(1, len(df)):
        regime = int(df.loc[i - 1, "predicted_regime"])

        if regime == 0:
            target = "QQQ"
            ret = df.loc[i, "QQQ"] / df.loc[i - 1, "QQQ"] - 1

        elif regime == 1:
            target = "SPY"
            ret = df.loc[i, "SPY"] / df.loc[i - 1, "SPY"] - 1

        else:
            target = "CASH"
            ret = 0.0

        cost = SWITCH_COST if target != active_asset else 0.0
        capital *= (1 + ret - cost)

        portfolio.append(capital)
        active_asset = target

    return portfolio

# ------------------------------------------------------------
# 5. 5X LEVERAGE DURING EXPANSION ONLY
# ------------------------------------------------------------
def run_wf_expansion_5x(df):
    capital = INITIAL_CAPITAL
    portfolio = [capital]
    active_asset = "START"

    for i in range(1, len(df)):
        regime = int(df.loc[i - 1, "predicted_regime"])

        # Expansion -> 5x QQQ
        if regime == 0:
            target = "QQQ_5X"
            raw_ret = df.loc[i, "QQQ"] / df.loc[i - 1, "QQQ"] - 1
            ret = EXPANSION_LEV * raw_ret

        # Slowdown -> 1x SPY
        elif regime == 1:
            target = "SPY"
            ret = df.loc[i, "SPY"] / df.loc[i - 1, "SPY"] - 1

        # Contraction -> CASH
        else:
            target = "CASH"
            ret = 0.0

        cost = SWITCH_COST if target != active_asset else 0.0
        capital *= (1 + ret - cost)

        portfolio.append(capital)
        active_asset = target

    return portfolio

# ------------------------------------------------------------
# 6. BUILD EQUITY CURVES
# ------------------------------------------------------------
wf_df["WF_Base"] = run_wf_baseline(wf_df)
wf_df["WF_Expansion_5x"] = run_wf_expansion_5x(wf_df)
wf_df["SPY_Value"] = INITIAL_CAPITAL * (wf_df["SPY"] / wf_df["SPY"].iloc[0])

# ------------------------------------------------------------
# 7. METRICS
# CAGR = calendar days
# VOL / SHARPE / SORTINO = daily returns
# ------------------------------------------------------------
def compute_metrics_from_daily_nav(series: pd.Series) -> pd.Series:
    daily_nav = series.resample("D").last().dropna()
    daily_ret = daily_nav.pct_change(fill_method=None).dropna()

    start_val = float(daily_nav.iloc[0])
    end_val   = float(daily_nav.iloc[-1])

    total_return = end_val / start_val - 1
    cal_days = (daily_nav.index[-1] - daily_nav.index[0]).days
    years = cal_days / CAL_DAYS if cal_days > 0 else np.nan

    cagr = (end_val / start_val) ** (1 / years) - 1 if pd.notna(years) and years > 0 else np.nan
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)

    sharpe = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS)
        if daily_ret.std() != 0 else np.nan
    )

    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std()
    sortino = (
        daily_ret.mean() / downside_std * np.sqrt(TRADING_DAYS)
        if pd.notna(downside_std) and downside_std != 0 else np.nan
    )

    dd = daily_nav / daily_nav.cummax() - 1
    max_dd = dd.min()

    return pd.Series({
        "Start Value": start_val,
        "Final Value": end_val,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd
    })

nav_df = wf_df.set_index("timestamp")[["WF_Base", "WF_Expansion_5x", "SPY_Value"]]

metrics = pd.DataFrame({
    "V4 WalkForward": compute_metrics_from_daily_nav(nav_df["WF_Base"]),
    "V4 Expansion 5x": compute_metrics_from_daily_nav(nav_df["WF_Expansion_5x"]),
    "SPY": compute_metrics_from_daily_nav(nav_df["SPY_Value"])
}).T

print("\nRaw metrics")
display(metrics)

metrics_fmt = metrics.copy()
metrics_fmt["Start Value"] = metrics_fmt["Start Value"].map(lambda x: f"{x:,.0f}")
metrics_fmt["Final Value"] = metrics_fmt["Final Value"].map(lambda x: f"{x:,.0f}")
for c in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown"]:
    metrics_fmt[c] = metrics_fmt[c].map(lambda x: f"{x*100:.2f}%")
for c in ["Sharpe", "Sortino"]:
    metrics_fmt[c] = metrics_fmt[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")

print("\nFormatted metrics")
display(metrics_fmt)

# ------------------------------------------------------------
# 8. PLOT
# ------------------------------------------------------------
plt.figure(figsize=(16, 7))

plt.plot(nav_df.index, nav_df["WF_Base"], label="V4 WalkForward", linewidth=1.8)
plt.plot(nav_df.index, nav_df["WF_Expansion_5x"], label="V4 Expansion 5x", linewidth=1.8)
plt.plot(nav_df.index, nav_df["SPY_Value"], label="SPY", linewidth=1.5, alpha=0.8)

plt.yscale("log")
plt.title("V4 WalkForward vs Expansion-Only 5x vs SPY")
plt.ylabel("Portfolio Value")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. OPTIONAL EXPORT
# ------------------------------------------------------------
EXPORT_PATH = "/Volumes/pawan_quant_research_1/default/market_data"

nav_df.to_csv(f"{EXPORT_PATH}/v4_expansion_5x_nav.csv")
metrics.to_csv(f"{EXPORT_PATH}/v4_expansion_5x_metrics.csv")

print(f"Saved: {EXPORT_PATH}/v4_expansion_5x_nav.csv")
print(f"Saved: {EXPORT_PATH}/v4_expansion_5x_metrics.csv")

# COMMAND ----------

# ============================================================
# LEVERAGE OPTIMIZATION SWEEP
# Expansion leverage only
# Slowdown = 1x SPY
# Contraction = CASH
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"

INITIAL_CAPITAL = 5_000_000
SWITCH_COST = 0.0005
TRADING_DAYS = 252
CAL_DAYS = 365.25

LEVERAGE_GRID = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# ------------------------------------------------------------
# 1. LOAD SIGNALS (NO MODEL RERUN)
# ------------------------------------------------------------
SIGNAL_TABLE = "pawan_quant_research_1.default.bt_walkforward_regime_signals"

wf_df = spark.table(SIGNAL_TABLE).toPandas()
wf_df["timestamp"] = pd.to_datetime(wf_df["timestamp"], utc=True).dt.tz_localize(None)
wf_df = wf_df.sort_values("timestamp")
wf_df = wf_df[
    (wf_df["timestamp"] >= START_DATE) &
    (wf_df["timestamp"] <= END_DATE)
].copy()

# ------------------------------------------------------------
# 2. LOAD PRICES
# ------------------------------------------------------------
prices = (
    spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
    .filter("symbol IN ('SPY','QQQ')")
    .toPandas()
)

prices["timestamp"] = pd.to_datetime(prices["timestamp"])
if str(prices["timestamp"].dtype).startswith("datetime64[ns,"):
    prices["timestamp"] = prices["timestamp"].dt.tz_localize(None)

prices = (
    prices.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last"
    )
    .reset_index()
)

prices = prices[
    (prices["timestamp"] >= START_DATE) &
    (prices["timestamp"] <= END_DATE)
].copy()

# ------------------------------------------------------------
# 3. MERGE
# ------------------------------------------------------------
wf_df = wf_df.merge(prices, on="timestamp", how="left")
wf_df["SPY"] = wf_df["SPY"].ffill()
wf_df["QQQ"] = wf_df["QQQ"].ffill()
wf_df = wf_df.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)

# ------------------------------------------------------------
# 4. STRATEGY RUNNER
# ------------------------------------------------------------
def run_expansion_leverage(df, expansion_lev=1.0):
    capital = INITIAL_CAPITAL
    portfolio = [capital]
    active_asset = "START"

    for i in range(1, len(df)):
        regime = int(df.loc[i - 1, "predicted_regime"])

        if regime == 0:
            target = f"QQQ_{expansion_lev}x"
            raw_ret = df.loc[i, "QQQ"] / df.loc[i - 1, "QQQ"] - 1
            ret = expansion_lev * raw_ret

        elif regime == 1:
            target = "SPY"
            ret = df.loc[i, "SPY"] / df.loc[i - 1, "SPY"] - 1

        else:
            target = "CASH"
            ret = 0.0

        cost = SWITCH_COST if target != active_asset else 0.0
        capital *= (1 + ret - cost)
        portfolio.append(capital)
        active_asset = target

    return pd.Series(portfolio, index=df["timestamp"])

# ------------------------------------------------------------
# 5. METRICS
# CAGR = calendar days
# VOL/SHARPE/SORTINO = daily-return based
# ------------------------------------------------------------
def compute_metrics_from_nav(nav: pd.Series):
    daily_nav = nav.resample("D").last().dropna()
    daily_ret = daily_nav.pct_change(fill_method=None).dropna()

    start_val = float(daily_nav.iloc[0])
    end_val = float(daily_nav.iloc[-1])

    total_return = end_val / start_val - 1
    cal_days = (daily_nav.index[-1] - daily_nav.index[0]).days
    years = cal_days / CAL_DAYS if cal_days > 0 else np.nan

    cagr = (end_val / start_val) ** (1 / years) - 1 if pd.notna(years) and years > 0 else np.nan
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)

    sharpe = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS)
        if daily_ret.std() != 0 else np.nan
    )

    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std()
    sortino = (
        daily_ret.mean() / downside_std * np.sqrt(TRADING_DAYS)
        if pd.notna(downside_std) and downside_std != 0 else np.nan
    )

    dd = daily_nav / daily_nav.cummax() - 1
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if pd.notna(max_dd) and max_dd != 0 else np.nan

    return {
        "Start Value": start_val,
        "Final Value": end_val,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar
    }

# ------------------------------------------------------------
# 6. RUN GRID
# ------------------------------------------------------------
navs = {}
rows = []

for lev in LEVERAGE_GRID:
    nav = run_expansion_leverage(wf_df, expansion_lev=lev)
    navs[lev] = nav

    m = compute_metrics_from_nav(nav)
    m["Expansion Leverage"] = lev
    rows.append(m)

# SPY benchmark
spy_nav = INITIAL_CAPITAL * (
    wf_df.set_index("timestamp")["SPY"] / wf_df["SPY"].iloc[0]
)
spy_metrics = compute_metrics_from_nav(spy_nav)
spy_metrics["Expansion Leverage"] = "SPY"
rows.append(spy_metrics)

results = pd.DataFrame(rows)

# ------------------------------------------------------------
# 7. SORT / REVIEW
# ------------------------------------------------------------
results = results[
    ["Expansion Leverage", "Final Value", "Total Return", "CAGR",
     "Annual Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar"]
]

print("Raw leverage sweep results")
display(results.sort_values("Sharpe", ascending=False))

# ------------------------------------------------------------
# 8. FORMAT
# ------------------------------------------------------------
results_fmt = results.copy()

results_fmt["Final Value"] = results_fmt["Final Value"].apply(
    lambda x: f"{x:,.0f}" if pd.notna(x) and not isinstance(x, str) else x
)

for col in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown"]:
    results_fmt[col] = results_fmt[col].apply(
        lambda x: f"{x*100:.2f}%" if pd.notna(x) and not isinstance(x, str) else x
    )

for col in ["Sharpe", "Sortino", "Calmar"]:
    results_fmt[col] = results_fmt[col].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) and not isinstance(x, str) else x
    )

print("Formatted leverage sweep results")
display(results_fmt.sort_values("Sharpe", ascending=False))

# ------------------------------------------------------------
# 9. PLOT EFFICIENT FRONTIER STYLE
# ------------------------------------------------------------
plot_df = results[results["Expansion Leverage"] != "SPY"].copy()
plot_df["Expansion Leverage"] = plot_df["Expansion Leverage"].astype(float)

plt.figure(figsize=(12, 7))
plt.scatter(plot_df["Annual Volatility"], plot_df["CAGR"], s=80)

for _, row in plot_df.iterrows():
    plt.annotate(
        f"{row['Expansion Leverage']}x",
        (row["Annual Volatility"], row["CAGR"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

# SPY marker
plt.scatter(
    results.loc[results["Expansion Leverage"] == "SPY", "Annual Volatility"],
    results.loc[results["Expansion Leverage"] == "SPY", "CAGR"],
    s=120,
    marker="X"
)

plt.xlabel("Annual Volatility")
plt.ylabel("CAGR")
plt.title("Expansion-Leverage Frontier: CAGR vs Volatility")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. OPTIONAL FILTER: ACCEPTABLE DRAWDOWN
# ------------------------------------------------------------
dd_cap = -0.35

acceptable = results[
    (results["Expansion Leverage"] != "SPY") &
    (results["Max Drawdown"] >= dd_cap)
].copy()

print(f"Strategies with Max Drawdown >= {dd_cap:.0%}")
display(acceptable.sort_values("Calmar", ascending=False))

# COMMAND ----------

# DBTITLE 1,Leverage Sweep WITH Borrowing Cost (10yr UST + 50bps)
# ============================================================
# LEVERAGE SWEEP — WITH COST OF LEVERAGE
# Borrow rate = avg 10-year US Treasury yield (per year) + 50 bps
# Sourced from yfinance (^TNX)
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"

INITIAL_CAPITAL = 5_000_000
SWITCH_COST     = 0.0005
SPREAD_BPS      = 0.0050          # 50 bps over 10yr
TRADING_DAYS    = 252
CAL_DAYS        = 365.25

LEVERAGE_GRID = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# ── 1. DOWNLOAD 10-YEAR TREASURY YIELD ───────────────────────
tnx = yf.download("^TNX", start="2021-01-01", end="2026-01-01", progress=False)
tnx_close = tnx["Close"].dropna()
if isinstance(tnx_close, pd.DataFrame):
    tnx_close = tnx_close.squeeze()

# Average annual yield (^TNX is in %, e.g. 4.25 = 4.25%)
avg_yield_by_year = tnx_close.groupby(tnx_close.index.year).mean() / 100  # decimal
borrow_rate_by_year = avg_yield_by_year + SPREAD_BPS

print("Average 10yr UST yield + 50bps (borrow rate):")
for yr, rate in borrow_rate_by_year.items():
    print(f"  {yr}: {avg_yield_by_year[yr]*100:.2f}% + 50bps = {rate*100:.2f}%")

# ── 2. LOAD SIGNALS + PRICES (reuse from cell 55) ───────────
SIGNAL_TABLE = "pawan_quant_research_1.default.bt_walkforward_regime_signals"

wf_df = spark.table(SIGNAL_TABLE).toPandas()
wf_df["timestamp"] = pd.to_datetime(wf_df["timestamp"])
if str(wf_df["timestamp"].dtype).startswith("datetime64[ns,"):
    wf_df["timestamp"] = wf_df["timestamp"].dt.tz_localize(None)
wf_df = wf_df.sort_values("timestamp")
wf_df = wf_df[
    (wf_df["timestamp"] >= START_DATE) &
    (wf_df["timestamp"] <= END_DATE)
].copy()

prices = (
    spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
    .filter("symbol IN ('SPY','QQQ')")
    .toPandas()
)
prices["timestamp"] = pd.to_datetime(prices["timestamp"])
if str(prices["timestamp"].dtype).startswith("datetime64[ns,"):
    prices["timestamp"] = prices["timestamp"].dt.tz_localize(None)
prices = (
    prices.pivot_table(index="timestamp", columns="symbol",
                       values="close", aggfunc="last")
    .reset_index()
)
prices = prices[
    (prices["timestamp"] >= START_DATE) &
    (prices["timestamp"] <= END_DATE)
].copy()

wf_df = wf_df.merge(prices, on="timestamp", how="left")
wf_df["SPY"] = wf_df["SPY"].ffill()
wf_df["QQQ"] = wf_df["QQQ"].ffill()
wf_df = wf_df.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)

# Count trading hours per day (for hourly cost conversion)
hours_per_day = wf_df.groupby(wf_df["timestamp"].dt.date).size().median()
print(f"\nMedian trading hours/day: {hours_per_day}")
print(f"Rows: {len(wf_df)}")

# ── 3. STRATEGY RUNNER WITH BORROW COST ──────────────────────
def run_expansion_leverage_with_cost(df, expansion_lev=1.0):
    capital = INITIAL_CAPITAL
    portfolio = [capital]
    active_asset = "START"

    for i in range(1, len(df)):
        regime = int(df.loc[i - 1, "predicted_regime"])
        ts = df.loc[i, "timestamp"]
        yr = ts.year

        if regime == 0:  # Expansion → leveraged QQQ
            target = f"QQQ_{expansion_lev}x"
            raw_ret = df.loc[i, "QQQ"] / df.loc[i - 1, "QQQ"] - 1
            lev_ret = expansion_lev * raw_ret

            # Hourly borrow cost on the leveraged portion
            annual_rate = borrow_rate_by_year.get(yr, borrow_rate_by_year.iloc[-1])
            hourly_cost = (expansion_lev - 1) * annual_rate / (TRADING_DAYS * hours_per_day)
            ret = lev_ret - hourly_cost

        elif regime == 1:  # Slowdown → 1x SPY (no leverage cost)
            target = "SPY"
            ret = df.loc[i, "SPY"] / df.loc[i - 1, "SPY"] - 1

        else:  # Contraction → CASH
            target = "CASH"
            ret = 0.0

        cost = SWITCH_COST if target != active_asset else 0.0
        capital *= (1 + ret - cost)
        portfolio.append(capital)
        active_asset = target

    return pd.Series(portfolio, index=df["timestamp"])

# ── 4. METRICS ───────────────────────────────────────────────
def compute_metrics(nav: pd.Series):
    daily_nav = nav.resample("D").last().dropna()
    daily_ret = daily_nav.pct_change(fill_method=None).dropna()

    start_val = float(daily_nav.iloc[0])
    end_val   = float(daily_nav.iloc[-1])
    total_return = end_val / start_val - 1
    cal_days = (daily_nav.index[-1] - daily_nav.index[0]).days
    years = cal_days / CAL_DAYS

    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else np.nan
    vol  = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS) if daily_ret.std() else np.nan

    down = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / down.std() * np.sqrt(TRADING_DAYS) if len(down) and down.std() else np.nan

    dd = daily_nav / daily_nav.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Final Value": end_val,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar
    }

# ── 5. RUN GRID ──────────────────────────────────────────────
rows = []
for lev in LEVERAGE_GRID:
    nav = run_expansion_leverage_with_cost(wf_df, expansion_lev=lev)
    m = compute_metrics(nav)
    m["Expansion Leverage"] = lev
    rows.append(m)

# SPY benchmark
spy_nav = INITIAL_CAPITAL * (wf_df.set_index("timestamp")["SPY"] / wf_df["SPY"].iloc[0])
spy_m = compute_metrics(spy_nav)
spy_m["Expansion Leverage"] = "SPY"
rows.append(spy_m)

results_cost = pd.DataFrame(rows)
results_cost = results_cost[
    ["Expansion Leverage", "Final Value", "Total Return", "CAGR",
     "Annual Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar"]
]

# ── 6. FORMAT ────────────────────────────────────────────────
fmt = results_cost.copy()
fmt["Expansion Leverage"] = fmt["Expansion Leverage"].apply(
    lambda x: f"{x:.2f}x" if isinstance(x, float) else x
)
fmt["Final Value"] = fmt["Final Value"].apply(
    lambda x: f"{x:,.0f}" if pd.notna(x) else x
)
for c in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown"]:
    fmt[c] = fmt[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else x)
for c in ["Sharpe", "Sortino", "Calmar"]:
    fmt[c] = fmt[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)

print("\n=== LEVERAGE SWEEP WITH BORROWING COST ===")
print(f"    Borrow rate = 10yr UST avg + 50bps")
print(f"    Cost applied to (N-1) leveraged portion, hourly\n")
display(fmt)

# ── 7. PPT TABLE IMAGE ───────────────────────────────────────
fmt_table = fmt[fmt["Expansion Leverage"] != "SPY"].copy()
table_cols = ["Expansion Leverage", "Final Value", "CAGR",
              "Annual Volatility", "Sharpe", "Max Drawdown", "Calmar"]
table_data = fmt_table[table_cols]

fig, ax = plt.subplots(figsize=(16, 5))
ax.axis("off")

tbl = ax.table(
    cellText=table_data.values,
    colLabels=table_cols,
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

for c in range(len(table_cols)):
    tbl[(0, c)].set_text_props(weight="bold")
    tbl[(0, c)].set_height(0.12)

highlight = {1: "#f2f2f2", 2: "#f2f2f2", 3: "#eaf4ea"}
for r, color in highlight.items():
    for c in range(len(table_cols)):
        tbl[(r, c)].set_facecolor(color)

plt.title("Expansion Leverage Sensitivity (incl. Borrowing Cost: 10yr UST + 50bps)",
          fontsize=14, pad=16)
plt.tight_layout()

EXPORT = "/Volumes/pawan_quant_research_1/default/market_data"
plt.savefig(f"{EXPORT}/ppt_leverage_with_borrow_cost.png", dpi=220, bbox_inches="tight")
plt.show()

# ── 8. RISK APPETITE TABLE ───────────────────────────────────
numeric = results_cost[results_cost["Expansion Leverage"] != "SPY"].copy()
numeric["Expansion Leverage"] = numeric["Expansion Leverage"].astype(float)

def bucket(lev):
    if lev <= 1.25: return "Conservative"
    elif lev <= 1.50: return "Balanced"
    else: return "Aggressive"

numeric["Bucket"] = numeric["Expansion Leverage"].map(bucket)

risk_agg = numeric.groupby("Bucket").agg(
    lev_min=("Expansion Leverage", "min"),
    lev_max=("Expansion Leverage", "max"),
    best_cagr=("CAGR", "max"),
    max_vol=("Annual Volatility", "max"),
    worst_dd=("Max Drawdown", "min"),
    best_sharpe=("Sharpe", "max")
).reindex(["Aggressive", "Balanced", "Conservative"])

risk_table = pd.DataFrame({
    "Risk Appetite": risk_agg.index,
    "Leverage Range": [f"{r.lev_min:.2f}x \u2013 {r.lev_max:.2f}x" for _, r in risk_agg.iterrows()],
    "Best CAGR": [f"{r.best_cagr*100:.2f}%" for _, r in risk_agg.iterrows()],
    "Max Volatility": [f"{r.max_vol*100:.2f}%" for _, r in risk_agg.iterrows()],
    "Worst Max Drawdown": [f"{r.worst_dd*100:.2f}%" for _, r in risk_agg.iterrows()],
    "Best Sharpe": [f"{r.best_sharpe:.2f}" for _, r in risk_agg.iterrows()]
})

print("\n=== RISK APPETITE SUMMARY (with borrow cost) ===")
display(risk_table)

# ── 9. SAVE TO 20260401 FOLDER ───────────────────────────────
EXPORT_DIR = "/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/04_Backtesting/20260401"
results_cost.to_csv(f"{EXPORT_DIR}/20260401_FINAL_leverage_sweep_with_borrow_cost.csv", index=False)
print(f"\nSaved: {EXPORT_DIR}/20260401_FINAL_leverage_sweep_with_borrow_cost.csv")

# COMMAND ----------

# ============================================================
# PPT-READY LEVERAGE TABLE + CHARTS
# Assumes `results` already exists from the leverage sweep
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CLEAN / SORT
# ------------------------------------------------------------
plot_df = results.copy()

# keep leverage rows separate from SPY
lev_df = plot_df[plot_df["Expansion Leverage"] != "SPY"].copy()
lev_df["Expansion Leverage"] = lev_df["Expansion Leverage"].astype(float)
lev_df = lev_df.sort_values("Expansion Leverage").reset_index(drop=True)

spy_df = plot_df[plot_df["Expansion Leverage"] == "SPY"].copy()

# ------------------------------------------------------------
# 2. FORMATTED TABLE FOR DECK
# ------------------------------------------------------------
table_df = lev_df.copy()

table_df["Expansion Leverage"] = table_df["Expansion Leverage"].map(lambda x: f"{x:.2f}x")
table_df["Final Value"] = table_df["Final Value"].map(lambda x: f"{x:,.0f}")
table_df["Total Return"] = table_df["Total Return"].map(lambda x: f"{x*100:.2f}%")
table_df["CAGR"] = table_df["CAGR"].map(lambda x: f"{x*100:.2f}%")
table_df["Annual Volatility"] = table_df["Annual Volatility"].map(lambda x: f"{x*100:.2f}%")
table_df["Sharpe"] = table_df["Sharpe"].map(lambda x: f"{x:.2f}")
table_df["Sortino"] = table_df["Sortino"].map(lambda x: f"{x:.2f}")
table_df["Max Drawdown"] = table_df["Max Drawdown"].map(lambda x: f"{x*100:.2f}%")
table_df["Calmar"] = table_df["Calmar"].map(lambda x: f"{x:.2f}")

table_df = table_df[
    [
        "Expansion Leverage",
        "Final Value",
        "CAGR",
        "Annual Volatility",
        "Sharpe",
        "Max Drawdown",
        "Calmar"
    ]
]

print("=== LEVERAGE SENSITIVITY TABLE ===")
display(table_df)

# Optional export for deck
EXPORT_PATH = "/Volumes/pawan_quant_research_1/default/market_data"
table_df.to_csv(f"{EXPORT_PATH}/ppt_leverage_sensitivity_table.csv", index=False)
print(f"Saved: {EXPORT_PATH}/ppt_leverage_sensitivity_table.csv")

# ------------------------------------------------------------
# 3. FRONTIER CHART: CAGR vs VOL
# ------------------------------------------------------------
plt.figure(figsize=(12, 7))

plt.scatter(
    lev_df["Annual Volatility"] * 100,
    lev_df["CAGR"] * 100,
    s=120
)

for _, row in lev_df.iterrows():
    plt.annotate(
        f"{row['Expansion Leverage']:.2f}x",
        (row["Annual Volatility"] * 100, row["CAGR"] * 100),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10
    )

if len(spy_df) > 0:
    plt.scatter(
        spy_df["Annual Volatility"] * 100,
        spy_df["CAGR"] * 100,
        s=160,
        marker="X",
        label="SPY"
    )
    for _, row in spy_df.iterrows():
        plt.annotate(
            "SPY",
            (row["Annual Volatility"] * 100, row["CAGR"] * 100),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=10
        )

plt.xlabel("Annual Volatility (%)")
plt.ylabel("CAGR (%)")
plt.title("Expansion Leverage Frontier: Return vs Risk")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. LINE CHART: LEVERAGE VS KEY METRICS
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

x = lev_df["Expansion Leverage"]

ax.plot(x, lev_df["CAGR"] * 100, marker="o", label="CAGR (%)")
ax.plot(x, lev_df["Annual Volatility"] * 100, marker="o", label="Volatility (%)")
ax.plot(x, lev_df["Max Drawdown"] * -100, marker="o", label="Max Drawdown (%)")

ax.set_xlabel("Expansion Leverage (x)")
ax.set_ylabel("Percent (%)")
ax.set_title("Leverage Sensitivity: CAGR, Volatility, and Drawdown")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. SHARPE / CALMAR CHART
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x, lev_df["Sharpe"], marker="o", label="Sharpe")
ax.plot(x, lev_df["Calmar"], marker="o", label="Calmar")

ax.set_xlabel("Expansion Leverage (x)")
ax.set_ylabel("Ratio")
ax.set_title("Leverage Sensitivity: Risk-Adjusted Performance")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. RISK-APPETITE BUCKET TABLE
# ------------------------------------------------------------
def risk_bucket(lev):
    if lev <= 1.25:
        return "Conservative"
    elif lev <= 1.50:
        return "Balanced"
    else:
        return "Aggressive"

bucket_df = lev_df.copy()
bucket_df["Risk Appetite"] = bucket_df["Expansion Leverage"].apply(risk_bucket)

bucket_summary = (
    bucket_df.groupby("Risk Appetite")
    .agg({
        "Expansion Leverage": ["min", "max"],
        "CAGR": "max",
        "Annual Volatility": "max",
        "Max Drawdown": "min",
        "Sharpe": "max"
    })
)

bucket_summary.columns = [
    "Min Leverage",
    "Max Leverage",
    "Best CAGR",
    "Max Volatility",
    "Worst Max Drawdown",
    "Best Sharpe"
]

bucket_summary = bucket_summary.reset_index()

bucket_summary_fmt = bucket_summary.copy()
bucket_summary_fmt["Leverage Range"] = (
    bucket_summary_fmt["Min Leverage"].map(lambda x: f"{x:.2f}x") +
    " – " +
    bucket_summary_fmt["Max Leverage"].map(lambda x: f"{x:.2f}x")
)
bucket_summary_fmt["Best CAGR"] = bucket_summary_fmt["Best CAGR"].map(lambda x: f"{x*100:.2f}%")
bucket_summary_fmt["Max Volatility"] = bucket_summary_fmt["Max Volatility"].map(lambda x: f"{x*100:.2f}%")
bucket_summary_fmt["Worst Max Drawdown"] = bucket_summary_fmt["Worst Max Drawdown"].map(lambda x: f"{x*100:.2f}%")
bucket_summary_fmt["Best Sharpe"] = bucket_summary_fmt["Best Sharpe"].map(lambda x: f"{x:.2f}")

bucket_summary_fmt = bucket_summary_fmt[
    ["Risk Appetite", "Leverage Range", "Best CAGR", "Max Volatility", "Worst Max Drawdown", "Best Sharpe"]
]

print("=== RISK APPETITE TABLE ===")
display(bucket_summary_fmt)

bucket_summary_fmt.to_csv(f"{EXPORT_PATH}/ppt_risk_appetite_table.csv", index=False)
print(f"Saved: {EXPORT_PATH}/ppt_risk_appetite_table.csv")

# COMMAND ----------

# ============================================================
# PPT TABLE CHART — EXPANSION LEVERAGE SENSITIVITY
# Updated: uses results_cost (incl. borrowing cost: 10yr UST + 50bps)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# Build table dynamically from results_cost
lev_only = results_cost[results_cost["Expansion Leverage"] != "SPY"].copy()
lev_only["Expansion Leverage"] = lev_only["Expansion Leverage"].astype(float)
lev_only = lev_only.sort_values("Expansion Leverage").reset_index(drop=True)

exp_table = pd.DataFrame({
    "Expansion Leverage": lev_only["Expansion Leverage"].map(lambda x: f"{x:.2f}x"),
    "Final Value":        lev_only["Final Value"].map(lambda x: f"{x:,.0f}"),
    "CAGR":               lev_only["CAGR"].map(lambda x: f"{x*100:.2f}%"),
    "Annual Volatility":  lev_only["Annual Volatility"].map(lambda x: f"{x*100:.2f}%"),
    "Sharpe":             lev_only["Sharpe"].map(lambda x: f"{x:.2f}"),
    "Max Drawdown":       lev_only["Max Drawdown"].map(lambda x: f"{x*100:.2f}%"),
    "Calmar":             lev_only["Calmar"].map(lambda x: f"{x:.2f}")
})

print("Leverage Sensitivity (incl. borrow cost: 10yr UST + 50bps):")
display(exp_table)

fig, ax = plt.subplots(figsize=(16, 4.8))
ax.axis("off")

tbl = ax.table(
    cellText=exp_table.values,
    colLabels=exp_table.columns,
    cellLoc="center",
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

# Header styling
for c in range(len(exp_table.columns)):
    cell = tbl[(0, c)]
    cell.set_text_props(weight="bold")
    cell.set_height(0.12)

# Highlight conservative / balanced rows
highlight_rows = {
    1: "#f2f2f2",   # 1.00x
    2: "#f2f2f2",   # 1.25x
    3: "#eaf4ea"    # 1.50x
}
for r, color in highlight_rows.items():
    for c in range(len(exp_table.columns)):
        tbl[(r, c)].set_facecolor(color)

plt.title("Expansion Leverage Sensitivity (incl. Borrowing Cost: 10yr UST + 50bps)",
          fontsize=14, pad=16)
plt.tight_layout()

EXPORT_PATH = "/Volumes/pawan_quant_research_1/default/market_data"
plt.savefig(f"{EXPORT_PATH}/ppt_expansion_leverage_table.png", dpi=220, bbox_inches="tight")
plt.show()

print(f"Saved: {EXPORT_PATH}/ppt_expansion_leverage_table.png")

# COMMAND ----------

# ============================================================
# PPT TABLE CHART — RISK APPETITE FRAMING
# Updated: uses results_cost from leverage sweep WITH borrow cost
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# Build risk appetite table from latest leverage sweep (with borrowing cost)
numeric_ra = results_cost[results_cost["Expansion Leverage"] != "SPY"].copy()
numeric_ra["Expansion Leverage"] = numeric_ra["Expansion Leverage"].astype(float)

def _bucket(lev):
    if lev <= 1.25: return "Conservative"
    elif lev <= 2.50: return "Balanced"
    else: return "Aggressive"

numeric_ra["Bucket"] = numeric_ra["Expansion Leverage"].map(_bucket)

ra_agg = numeric_ra.groupby("Bucket").agg(
    lev_min=("Expansion Leverage", "min"),
    lev_max=("Expansion Leverage", "max"),
    best_cagr=("CAGR", "max"),
    max_vol=("Annual Volatility", "max"),
    worst_dd=("Max Drawdown", "min"),
    best_sharpe=("Sharpe", "max")
).reindex(["Aggressive", "Balanced", "Conservative"])

risk_table = pd.DataFrame({
    "Risk Appetite":       ra_agg.index.tolist(),
    "Leverage Range":      [f"{r.lev_min:.2f}x \u2013 {r.lev_max:.2f}x" for _, r in ra_agg.iterrows()],
    "Best CAGR":           [f"{r.best_cagr*100:.2f}%" for _, r in ra_agg.iterrows()],
    "Max Volatility":      [f"{r.max_vol*100:.2f}%" for _, r in ra_agg.iterrows()],
    "Worst Max Drawdown":  [f"{r.worst_dd*100:.2f}%" for _, r in ra_agg.iterrows()],
    "Best Sharpe":         [f"{r.best_sharpe:.2f}" for _, r in ra_agg.iterrows()]
})

print("Risk Appetite Table (incl. borrow cost: 10yr UST + 50bps):")
display(risk_table)

fig, ax = plt.subplots(figsize=(12, 3.2))
ax.axis("off")

tbl = ax.table(
    cellText=risk_table.values,
    colLabels=risk_table.columns,
    cellLoc="center",
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.9)

# Header styling
for c in range(len(risk_table.columns)):
    cell = tbl[(0, c)]
    cell.set_text_props(weight="bold")
    cell.set_height(0.18)

# Row colors
row_colors = {
    1: "#fdeaea",   # Aggressive
    2: "#eaf4ea",   # Balanced
    3: "#f2f2f2"    # Conservative
}
for r, color in row_colors.items():
    for c in range(len(risk_table.columns)):
        tbl[(r, c)].set_facecolor(color)

plt.title("Leverage as a Function of Risk Appetite (incl. Borrowing Cost)", fontsize=15, pad=16)
plt.tight_layout()

EXPORT_PATH = "/Volumes/pawan_quant_research_1/default/market_data"
plt.savefig(f"{EXPORT_PATH}/ppt_risk_appetite_table.png", dpi=220, bbox_inches="tight")
plt.show()

print(f"Saved: {EXPORT_PATH}/ppt_risk_appetite_table.png")

# COMMAND ----------

# DBTITLE 1,PPT — Annual Returns Bar Chart (2021–2026 YTD)
# ============================================================
# ANNUAL RETURNS BAR CHART: V4 WF vs SPY
# 2021 through 2026 YTD (Jan–Mar)
# Uses bronze_bt_market_1h for prices (no yfinance dependency)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START = "2021-01-01"
END   = "2026-03-31"

# ── 1. LOAD V4 SIGNALS (Delta) ──────────────────────────────
v4 = (
    spark.table("pawan_quant_research_1.default.bt_walkforward_regime_signals")
    .toPandas()
)
v4["timestamp"] = pd.to_datetime(v4["timestamp"])
v4 = v4.set_index("timestamp").sort_index()["predicted_regime"]
v4 = v4.loc[START:END]

# ── 2. LOAD DAILY PRICES (bronze hourly → daily close) ─────
prices_h = (
    spark.table("pawan_quant_research_1.default.bronze_bt_market_1h")
    .filter("symbol IN ('SPY','QQQ')")
    .toPandas()
)
prices_h["timestamp"] = pd.to_datetime(prices_h["timestamp"])
prices_daily = (
    prices_h
    .pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
    .resample("D").last()
    .dropna()
)
returns = prices_daily.pct_change().dropna()
returns = returns.loc[START:END]

# ── 3. ALIGN SIGNALS TO DAILY ─────────────────────────────
v4_d = v4.resample("D").ffill().reindex(returns.index, method="ffill")

def strategy_daily_returns(regime_series):
    out = []
    for dt, reg in regime_series.items():
        if reg == 0:
            out.append(returns.loc[dt, "QQQ"])
        elif reg == 1:
            out.append(returns.loc[dt, "SPY"])
        else:
            out.append(0.0)
    return pd.Series(out, index=regime_series.index)

ret_v4  = strategy_daily_returns(v4_d)
ret_spy = returns["SPY"].reindex(ret_v4.index).fillna(0.0)

daily_ret = pd.DataFrame({
    "WalkForward_V4": ret_v4,
    "SPY": ret_spy
})

# ── 4. YEARLY RETURNS ─────────────────────────────────────
full_years = daily_ret.loc[:"2025-12-31"].resample("YE").apply(lambda x: (1 + x).prod() - 1)
full_years.index = full_years.index.year

ytd_2026 = (1 + daily_ret.loc["2026-01-01":END]).prod() - 1
ytd_2026.name = 2026

yearly = pd.concat([full_years, ytd_2026.to_frame().T])
yearly.index.name = "Year"

print("Annual returns (%) including 2026 YTD:")
display((yearly * 100).round(2))

# ── 5. PPT BAR CHART ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

plot_data = (yearly * 100).copy()
labels = [str(y) if y != 2026 else "2026 YTD" for y in plot_data.index]

x = np.arange(len(labels))
width = 0.35

colors_map = {"WalkForward_V4": "#5B2C8C", "SPY": "#B088D4"}

offset = -width / 2
for col in plot_data.columns:
    vals = plot_data[col].values
    bars = ax.bar(x + offset, vals, width, label=col, color=colors_map.get(col), edgecolor="white")
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            y_pos = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + (1.0 if y_pos >= 0 else -2.5),
                f"{val:.1f}%",
                ha="center", va="bottom" if y_pos >= 0 else "top",
                fontsize=9, fontweight="bold"
            )
    offset += width

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Return (%)", fontsize=12)
ax.set_title("Annual Returns: Walk-Forward V4 vs SPY (2021–2026 YTD)", fontsize=15, pad=12)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=11, loc="upper left")
plt.tight_layout()

EXPORT = "/Volumes/pawan_quant_research_1/default/market_data"
plt.savefig(f"{EXPORT}/ppt_annual_returns_2021_2026ytd.png", dpi=220, bbox_inches="tight")
plt.show()

EXPORT_DIR = "/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/04_Backtesting/20260401"
yearly.to_csv(f"{EXPORT_DIR}/20260401_FINAL_yearly_returns_v4_spy.csv")
print(f"\nSaved chart: {EXPORT}/ppt_annual_returns_2021_2026ytd.png")
print(f"Saved CSV:   {EXPORT_DIR}/20260401_FINAL_yearly_returns_v4_spy.csv")

# COMMAND ----------

# DBTITLE 1,PPT — Trailing 1yr / 3yr / 5yr Performance
# ============================================================
# TRAILING 1-YEAR, 3-YEAR, 5-YEAR PERFORMANCE
# V4 WF vs SPY — As of March 31, 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AS_OF = "2026-03-31"
TRADING_DAYS = 252
CAL_DAYS = 365.25

# ── 1. BUILD CUMULATIVE NAV (from daily_ret in prev cell) ───
nav = (1 + daily_ret).cumprod()

windows = {
    "1 Year":  ("2025-04-01", AS_OF),
    "3 Year":  ("2023-04-01", AS_OF),
    "5 Year":  ("2021-01-01", AS_OF),
}

# ── 2. METRICS FUNCTION ───────────────────────────────────
def trailing_metrics(daily_rets, label):
    cum = (1 + daily_rets).cumprod()
    start_val = cum.iloc[0]
    end_val   = cum.iloc[-1]
    total_ret = end_val / start_val - 1
    cal_days  = (cum.index[-1] - cum.index[0]).days
    years     = cal_days / CAL_DAYS

    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else np.nan
    vol  = daily_rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(TRADING_DAYS)
              if daily_rets.std() != 0 else np.nan)
    down = daily_rets[daily_rets < 0]
    sortino = (daily_rets.mean() / down.std() * np.sqrt(TRADING_DAYS)
               if len(down) > 0 and down.std() != 0 else np.nan)
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()

    return {
        "Window": label,
        "Total Return": total_ret,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd
    }

# ── 3. COMPUTE FOR EACH WINDOW × STRATEGY ─────────────────
rows = []
for wlabel, (ws, we) in windows.items():
    slice_ret = daily_ret.loc[ws:we]
    if len(slice_ret) == 0:
        continue
    for strat in daily_ret.columns:
        m = trailing_metrics(slice_ret[strat], wlabel)
        m["Strategy"] = strat
        rows.append(m)

trailing = pd.DataFrame(rows)

# ── 4. FORMATTED TABLE ────────────────────────────────────
trailing_fmt = trailing.copy()
for c in ["Total Return", "CAGR", "Volatility", "Max Drawdown"]:
    trailing_fmt[c] = trailing_fmt[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
for c in ["Sharpe", "Sortino"]:
    trailing_fmt[c] = trailing_fmt[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

print("Trailing Performance as of", AS_OF)
display(trailing_fmt[["Window","Strategy","Total Return","CAGR","Volatility","Sharpe","Sortino","Max Drawdown"]])

# ── 5. PPT CHART: CAGR / SHARPE / MAX DD GROUPED BAR ───────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

metric_cols   = ["CAGR", "Sharpe", "Max Drawdown"]
metric_titles = ["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)"]
metric_mult   = [100, 1, 100]

colors_map = {"WalkForward_V4": "#5B2C8C", "SPY": "#B088D4"}
strategies = daily_ret.columns.tolist()
win_labels = list(windows.keys())

for idx, (metric, title, mult) in enumerate(zip(metric_cols, metric_titles, metric_mult)):
    ax = axes[idx]
    x = np.arange(len(win_labels))
    width = 0.35
    offset = -width / 2

    for strat in strategies:
        vals = []
        for wl in win_labels:
            row = trailing[(trailing["Window"] == wl) & (trailing["Strategy"] == strat)]
            if len(row) > 0:
                vals.append(float(row[metric].iloc[0]) * mult)
            else:
                vals.append(0)
        bars = ax.bar(x + offset, vals, width, label=strat,
                      color=colors_map.get(strat), edgecolor="white")
        for bar, v in zip(bars, vals):
            y = bar.get_height()
            fmt = f"{v:.1f}%" if mult == 100 else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.3 if y >= 0 else -0.8),
                    fmt, ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=9, fontweight="bold")
        offset += width

    ax.set_xticks(x)
    ax.set_xticklabels(win_labels, fontsize=11)
    ax.set_title(title, fontsize=13, pad=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=10, loc="upper left")

plt.suptitle(f"Trailing Performance: V4 Walk-Forward vs SPY — as of {AS_OF}", fontsize=16, y=1.02)
plt.tight_layout()

EXPORT = "/Volumes/pawan_quant_research_1/default/market_data"
plt.savefig(f"{EXPORT}/ppt_trailing_1yr_3yr_5yr.png", dpi=220, bbox_inches="tight")
plt.show()

# ── 6. PPT TABLE IMAGE ────────────────────────────────────
table_data = []
for wl in win_labels:
    for strat in strategies:
        row = trailing_fmt[(trailing_fmt["Window"] == wl) & (trailing_fmt["Strategy"] == strat)]
        if len(row) > 0:
            table_data.append([
                wl, strat,
                row["CAGR"].iloc[0], row["Volatility"].iloc[0],
                row["Sharpe"].iloc[0], row["Sortino"].iloc[0],
                row["Max Drawdown"].iloc[0]
            ])

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

tbl = ax.table(
    cellText=table_data,
    colLabels=["Window", "Strategy", "CAGR", "Volatility", "Sharpe", "Sortino", "Max DD"],
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)

for c in range(7):
    tbl[(0, c)].set_text_props(weight="bold")
    tbl[(0, c)].set_height(0.1)

# Alternate row colors by window (purple theme)
row_idx = 1
for wl in win_labels:
    color = "#F0E6F6" if win_labels.index(wl) % 2 == 0 else "#ffffff"
    for strat in strategies:
        for c in range(7):
            tbl[(row_idx, c)].set_facecolor(color)
        row_idx += 1

plt.title(f"Trailing Performance Summary: V4 WF vs SPY — as of {AS_OF}", fontsize=14, pad=14)
plt.tight_layout()
plt.savefig(f"{EXPORT}/ppt_trailing_performance_table.png", dpi=220, bbox_inches="tight")
plt.show()

# Save to 20260401 folder
EXPORT_DIR = "/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/04_Backtesting/20260401"
trailing.to_csv(f"{EXPORT_DIR}/20260401_FINAL_trailing_performance.csv", index=False)
print(f"\nSaved charts + CSV to {EXPORT_DIR}")
