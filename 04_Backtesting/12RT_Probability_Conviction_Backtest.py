# Databricks notebook source
# MAGIC %md
# MAGIC # Probability-Based Allocation with Conviction Scoring
# MAGIC 
# MAGIC **Objective:** Test allocation logic based on regime probabilities + conviction score
# MAGIC 
# MAGIC **Logic:**
# MAGIC - Conviction = max(probability across 3 regimes)
# MAGIC - High conviction (≥0.6): Full allocation (100% to regime asset)
# MAGIC - Medium conviction (0.4-0.6): 70% allocation
# MAGIC - Low conviction (<0.4): 40% allocation
# MAGIC - Contraction regime: Scale equity exposure based on conviction

# COMMAND ----------

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix

# Conviction thresholds
CONVICTION_HIGH = 0.6
CONVICTION_MED = 0.4

# Date ranges
TRAIN_START = "2016-01-01"
LIVE_START = "2021-01-01"
LIVE_END = "2026-01-01"

# Features
FEATURES = [
    "CPIAUCSL_z","PCEPI_z","PCE_z","PAYEMS_z","UNRATE_z",
    "GDPC1_z","INDPRO_z","UMCSENT_z","VIXCLS_z","vix3m_z",
    "VIX_Inversion_Z","Price_Mom_24H","CPI_Event_Active","NFP_Event_Active"
]

CATALOG = "pawan_quant_research_1"
SCHEMA = "default"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
LABEL_PATH = "/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_regime_signals-v4.csv"

print("✅ Configuration loaded")
print(f"   Conviction thresholds: HIGH≥{CONVICTION_HIGH}, MED≥{CONVICTION_MED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

# Load silver features (include SPY_close for returns)
spark_df = (
    spark.table(SILVER_TABLE)
    .filter(f"timestamp >= '{TRAIN_START}'")
    .select(["timestamp", "SPY_close"] + FEATURES)
)
silver_df = spark_df.toPandas()
silver_df["timestamp"] = pd.to_datetime(silver_df["timestamp"])
silver_df = silver_df.sort_values("timestamp")

# Load labels
labels_df = pd.read_csv(LABEL_PATH, parse_dates=["timestamp"])
labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"])
labels_df = labels_df[["timestamp","predicted_regime"]]
labels_df = labels_df.rename(columns={"predicted_regime":"regime"})

# Merge
silver_df = silver_df.merge(labels_df, on="timestamp", how="left")
silver_df = silver_df.dropna(subset=["regime"])
silver_df["regime"] = silver_df["regime"].astype(int)

print(f"✅ Dataset: {len(silver_df)} rows")
print(f"   Range: {silver_df.timestamp.min()} → {silver_df.timestamp.max()}")
print(f"   Class distribution:\n{silver_df.regime.value_counts()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load SPY/QQQ Returns for Backtest

# COMMAND ----------

# Calculate forward returns (1-hour ahead to match hourly bars)
silver_df["SPY_ret_1h"] = silver_df["SPY_close"].shift(-1) / silver_df["SPY_close"] - 1

# Optional QQQ support if present in silver table
HAS_QQQ = "QQQ_close" in silver_df.columns
if HAS_QQQ:
    silver_df["QQQ_ret_1h"] = silver_df["QQQ_close"].shift(-1) / silver_df["QQQ_close"] - 1
    print("✅ Returns calculated (1-hour forward) for SPY and QQQ")
else:
    silver_df["QQQ_ret_1h"] = np.nan
    print("⚠️ QQQ_close not found in silver table. Expansion will use SPY returns unless QQQ is added.")

# Quick sanity check
non_null_spy = silver_df["SPY_ret_1h"].notna().mean()
print(f"   SPY_ret_1h non-null rate: {non_null_spy:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Walk-Forward Schedule

# COMMAND ----------

train_start = pd.Timestamp(TRAIN_START)
live_start = pd.Timestamp(LIVE_START)
live_end = pd.Timestamp(LIVE_END)

retrain_dates = []
d = live_start
while d <= live_end:
    retrain_dates.append(d)
    d += relativedelta(months=1)

print(f"✅ {len(retrain_dates)} monthly retrain cycles")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training Function

# COMMAND ----------

def train_model(train_df):
    """Train XGBoost with probability output"""
    X = train_df[FEATURES]
    y = train_df["regime"]
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",  # Output probabilities
        num_class=3,
        tree_method="hist",
        random_state=42
    )
    
    model.fit(X, y)
    return model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Allocation Logic Functions

# COMMAND ----------

def get_conviction_tier(proba):
    """Map max probability to conviction tier"""
    conviction = max(proba)
    if conviction >= CONVICTION_HIGH:
        return "HIGH", conviction
    elif conviction >= CONVICTION_MED:
        return "MEDIUM", conviction
    else:
        return "LOW", conviction

def get_allocation_conviction(regime_int, proba):
    """
    Probability-based allocation with conviction scoring
    All regimes scale allocation based on conviction score.
    
    Returns:
        equity_allocation: float (0-1)
        position_str: string description
    """
    conviction_tier, conviction = get_conviction_tier(proba)
    
    # Map conviction to allocation multiplier
    if conviction_tier == "HIGH":
        allocation_mult = 1.0
    elif conviction_tier == "MEDIUM":
        allocation_mult = 0.7
    else:  # LOW
        allocation_mult = 0.4
    
    # Regime-specific allocation
    if regime_int == 0:  # Expansion → QQQ
        equity_allocation = allocation_mult
        position_str = f"{int(equity_allocation*100)}% QQQ / {int((1-equity_allocation)*100)}% CASH"
        
    elif regime_int == 1:  # Slowdown → SPY
        equity_allocation = allocation_mult
        position_str = f"{int(equity_allocation*100)}% SPY / {int((1-equity_allocation)*100)}% CASH"
        
    else:  # Contraction → CASH (inverse scaling)
        # High conviction contraction → 100% CASH (0% equity)
        # Medium conviction → 30% SPY / 70% CASH
        # Low conviction → 40% SPY / 60% CASH (uncertain, keep some exposure)
        if conviction_tier == "HIGH":
            equity_allocation = 0.0
            position_str = "100% CASH"
        elif conviction_tier == "MEDIUM":
            equity_allocation = 0.3
            position_str = "30% SPY / 70% CASH"
        else:  # LOW conviction contraction
            equity_allocation = 0.4
            position_str = "40% SPY / 60% CASH"
    
    return equity_allocation, position_str, conviction_tier, conviction

def get_allocation_baseline(regime_int):
    """
    Baseline: 100% allocation based on predicted regime
    """
    if regime_int == 0:
        return 1.0, "100% QQQ", "N/A", 1.0
    elif regime_int == 1:
        return 1.0, "100% SPY", "N/A", 1.0
    else:
        return 0.0, "100% CASH", "N/A", 1.0

# COMMAND ----------

# MAGIC %md
# MAGIC ### Walk-Forward Backtest Loop

# COMMAND ----------

# Storage for results
results_conviction = []
results_baseline = []
conviction_details = []

feature_importance_history = []

for i in range(len(retrain_dates)-1):
    train_end = retrain_dates[i]
    test_end = retrain_dates[i+1]
    
    if i % 3 == 0:  # Print every 3 months
        print(f"\n{'='*50}")
        print(f"Cycle {i}: TRAIN until {train_end.date()}, TEST until {test_end.date()}")
    
    # Train set
    train_df = silver_df[
        (silver_df.timestamp >= train_start) &
        (silver_df.timestamp < train_end)
    ].dropna(subset=["regime"] + FEATURES)
    
    # Test set
    test_df = silver_df[
        (silver_df.timestamp >= train_end) &
        (silver_df.timestamp < test_end)
    ].dropna(subset=FEATURES)
    
    if len(test_df) == 0 or len(train_df) == 0:
        continue
    
    # Train model
    model = train_model(train_df)
    
    # Store feature importance
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": list(importance_dict.keys()),
        "importance": list(importance_dict.values()),
        "date": train_end
    })
    feature_importance_history.append(imp_df)
    
    # Predict on test set
    X_test = test_df[FEATURES]
    
    # Get predictions AND probabilities
    y_pred = model.predict(X_test)
    proba_pred = model.predict_proba(X_test)
    
    # Process each timestamp
    for idx, row in test_df.iterrows():
        ts = row["timestamp"]
        regime_pred = int(y_pred[test_df.index.get_loc(idx)])
        proba = proba_pred[test_df.index.get_loc(idx)]
        
        # Get allocations
        eq_alloc_conv, pos_conv, conv_tier, conviction = get_allocation_conviction(regime_pred, proba)
        eq_alloc_base, pos_base, _, _ = get_allocation_baseline(regime_pred)
        
        # Get actual returns
        spy_ret = row.get("SPY_ret_1h", 0) if not pd.isna(row.get("SPY_ret_1h")) else 0
        qqq_ret = row.get("QQQ_ret_1h", np.nan)
        
        # Calculate strategy returns
        # Conviction-based
        if regime_pred == 0:  # Expansion
            # If QQQ isn't available, fall back to SPY to avoid silent zeros
            expansion_ret = qqq_ret if HAS_QQQ and not pd.isna(qqq_ret) else spy_ret
            strat_ret_conv = eq_alloc_conv * expansion_ret
        elif regime_pred == 1:  # Slowdown
            strat_ret_conv = eq_alloc_conv * spy_ret
        else:  # Contraction
            strat_ret_conv = eq_alloc_conv * spy_ret  # SPY for partial equity, 0 for cash
        
        # Baseline
        if regime_pred == 0:
            expansion_ret = qqq_ret if HAS_QQQ and not pd.isna(qqq_ret) else spy_ret
            strat_ret_base = eq_alloc_base * expansion_ret
        elif regime_pred == 1:
            strat_ret_base = eq_alloc_base * spy_ret
        else:
            strat_ret_base = 0  # 100% cash
        
        # Store results
        results_conviction.append({
            "timestamp": ts,
            "regime": regime_pred,
            "equity_allocation": eq_alloc_conv,
            "position": pos_conv,
            "conviction_tier": conv_tier,
            "conviction": conviction,
            "prob_expansion": proba[0],
            "prob_slowdown": proba[1],
            "prob_contraction": proba[2],
            "strategy_return": strat_ret_conv,
            "spy_return": spy_ret,
            "qqq_return": qqq_ret if not pd.isna(qqq_ret) else np.nan
        })
        
        results_baseline.append({
            "timestamp": ts,
            "regime": regime_pred,
            "equity_allocation": eq_alloc_base,
            "position": pos_base,
            "strategy_return": strat_ret_base,
            "spy_return": spy_ret,
            "qqq_return": qqq_ret if not pd.isna(qqq_ret) else np.nan
        })

print(f"\n✅ Backtest complete: {len(results_conviction)} periods")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert to DataFrames

# COMMAND ----------

results_conv_df = pd.DataFrame(results_conviction).set_index("timestamp")
results_base_df = pd.DataFrame(results_baseline).set_index("timestamp")

print("Conviction-based results:")
print(results_conv_df.head())
print(f"\nConviction distribution:\n{results_conv_df.conviction_tier.value_counts()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Equity Curves

# COMMAND ----------

INITIAL_CAPITAL = 10000

# Cumulative returns
equity_conv = (1 + results_conv_df["strategy_return"]).cumprod() * INITIAL_CAPITAL
equity_base = (1 + results_base_df["strategy_return"]).cumprod() * INITIAL_CAPITAL
equity_spy = (1 + results_conv_df["spy_return"]).cumprod() * INITIAL_CAPITAL
equity_qqq = None
if HAS_QQQ:
    equity_qqq = (1 + results_conv_df["qqq_return"]).cumprod() * INITIAL_CAPITAL

# Plot
plt.figure(figsize=(14, 6))

plt.plot(equity_conv.index, equity_conv.values, label="Conviction-Based", linewidth=2, color="blue")
plt.plot(equity_base.index, equity_base.values, label="Baseline (100% alloc)", linewidth=2, color="orange", linestyle="--")
plt.plot(equity_spy.index, equity_spy.values, label="SPY Buy&Hold", linewidth=1, color="gray", alpha=0.5)
if equity_qqq is not None:
    plt.plot(equity_qqq.index, equity_qqq.values, label="QQQ Buy&Hold", linewidth=1, color="purple", alpha=0.5)

plt.title("Equity Curves: Conviction-Based vs Baseline")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Metrics

# COMMAND ----------

def _estimate_periods_per_year(index):
    """Estimate periods/year from timestamp index (assumes regular spacing)."""
    if len(index) < 2:
        return 252
    diffs = index.to_series().diff().dropna()
    median_seconds = diffs.dt.total_seconds().median()
    if not median_seconds or median_seconds <= 0:
        return 252
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / median_seconds

def calc_metrics(equity_curve, name):
    """Calculate performance metrics"""
    returns = equity_curve.pct_change().dropna()
    periods_per_year = _estimate_periods_per_year(equity_curve.index)
    
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (periods_per_year / len(returns)) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    print(f"\n{name}:")
    print(f"  Total Return: {total_ret:.2%}")
    print(f"  Ann. Return:  {ann_ret:.2%}")
    print(f"  Ann. Vol:     {vol:.2%}")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Max DD:       {max_dd:.2%}")
    
    return {
        "name": name,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd
    }

print("="*50)
print("PERFORMANCE METRICS")
print("="*50)

metrics_conv = calc_metrics(equity_conv, "Conviction-Based")
metrics_base = calc_metrics(equity_base, "Baseline")
metrics_spy = calc_metrics(equity_spy, "SPY Buy&Hold")
metrics_qqq = None
if equity_qqq is not None:
    metrics_qqq = calc_metrics(equity_qqq, "QQQ Buy&Hold")

# Comparison table
metrics_rows = [metrics_conv, metrics_base, metrics_spy]
if metrics_qqq is not None:
    metrics_rows.append(metrics_qqq)
metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.set_index("name")
print("\n" + "="*50)
print("COMPARISON TABLE")
print("="*50)
print(metrics_df.round(4).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conviction Analysis

# COMMAND ----------

# Analyze performance by conviction tier
conv_tier_returns = results_conv_df.groupby("conviction_tier")["strategy_return"].agg(["mean", "std", "count"])
    conv_tier_returns["sharpe"] = conv_tier_returns["mean"] / conv_tier_returns["std"] * np.sqrt(
        _estimate_periods_per_year(results_conv_df.index)
    )

print("Performance by Conviction Tier:")
print(conv_tier_returns.round(4))

# Plot
plt.figure(figsize=(10, 5))
bars = plt.bar(conv_tier_returns.index, conv_tier_returns["mean"] * 100)
plt.title("Average Return by Conviction Tier")
plt.ylabel("Avg Return (%)")
plt.xlabel("Conviction Tier")

for bar, val in zip(bars, conv_tier_returns["mean"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f"{val*100:.2f}%", ha='center')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regime-Specific Analysis

# COMMAND ----------

# Performance by regime
regime_returns = results_conv_df.groupby("regime")["strategy_return"].agg(["mean", "std", "count"])
regime_returns["sharpe"] = regime_returns["mean"] / regime_returns["std"] * np.sqrt(
    _estimate_periods_per_year(results_conv_df.index)
)
regime_returns = regime_returns.rename(index={0: "Expansion", 1: "Slowdown", 2: "Contraction"})

print("Performance by Regime:")
print(regime_returns.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drawdown Comparison

# COMMAND ----------

# Calculate drawdowns
def calc_drawdown(equity):
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return drawdown

dd_conv = calc_drawdown(equity_conv)
dd_base = calc_drawdown(equity_base)

plt.figure(figsize=(14, 5))

plt.fill_between(dd_conv.index, dd_conv.values, 0, alpha=0.5, label="Conviction-Based", color="blue")
plt.fill_between(dd_base.index, dd_base.values, 0, alpha=0.5, label="Baseline", color="orange")

plt.title("Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance (Latest)

# COMMAND ----------

# Plot latest feature importance
latest_fi = feature_importance_history[-1]
latest_fi = latest_fi.sort_values("importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(latest_fi["feature"], latest_fi["importance"])
plt.title("Feature Importance (Latest Cycle)")
plt.xlabel("Gain")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary & Recommendation

# COMMAND ----------

print("="*60)
print("BACKTEST SUMMARY")
print("="*60)

print(f"""
Test Period: {results_conv_df.index.min().date()} to {results_conv_df.index.max().date()}

CONVICTION-BASED ALLOCATION:
  Total Return: {metrics_conv['total_return']:.2%}
  Sharpe Ratio: {metrics_conv['sharpe']:.2f}
  Max Drawdown: {metrics_conv['max_drawdown']:.2%}

BASELINE (100% ALLOCATION):
  Total Return: {metrics_base['total_return']:.2%}
  Sharpe Ratio: {metrics_base['sharpe']:.2f}
  Max Drawdown: {metrics_base['max_drawdown']:.2%}

OUTPERFORMANCE:
  Return Delta: {(metrics_conv['total_return'] - metrics_base['total_return'])*100:.2f} pp
  Sharpe Delta: {metrics_conv['sharpe'] - metrics_base['sharpe']:.2f}
  DD Improvement: {(metrics_base['max_drawdown'] - metrics_conv['max_drawdown'])*100:.2f} pp

CONVICTION DISTRIBUTION:
{results_conv_df.conviction_tier.value_counts().to_string()}
""")

# Recommendation
if metrics_conv['sharpe'] > metrics_base['sharpe'] and metrics_conv['max_drawdown'] > metrics_base['max_drawdown']:
    print("\n✅ RECOMMENDATION: Conviction-based allocation shows IMPROVED risk-adjusted returns")
    print("   → Proceed with RT implementation")
elif metrics_conv['total_return'] > metrics_base['total_return']:
    print("\n⚠️  RECOMMENDATION: Higher returns but similar/higher risk")
    print("   → Consider optimization or further testing")
else:
    print("\n❌ RECOMMENDATION: No improvement over baseline")
    print("   → Optimize conviction thresholds or keep current approach")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Results

# COMMAND ----------

# Save for reference
results_conv_df.to_csv("/Volumes/pawan_quant_research_1/default/market_data/conviction_backtest_results.csv")
metrics_df.to_csv("/Volumes/pawan_quant_research_1/default/market_data/conviction_backtest_metrics.csv")

print("✅ Results saved to Volumes")
