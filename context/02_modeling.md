# Modeling Documentation

## Model Type
**XGBoost Classifier** (3-class regime detection)

## Target Definition

```python
# Forward 1-week return on SPY
df["fwd_return_1w"] = df["SPY_close"].shift(-168) / df["SPY_close"] - 1

# Regime labels
def label_regime(ret):
    if pd.isna(ret): return None
    if ret > 0.005:   return 0  # Expansion (Risk-On)
    if ret < -0.005:  return 2  # Recession (Risk-Off)
    return 1                    # Slowdown (Neutral)
```

## Features (14 total)

| Feature | Source | Description |
|---------|--------|-------------|
| CPIAUCSL_z | FRED | CPI YoY z-score |
| PCEPI_z | FRED | PCE Price Index z-score |
| PCE_z | FRED | Personal Consumption z-score |
| PAYEMS_z | FRED | Payrolls z-score |
| UNRATE_z | FRED | Unemployment z-score |
| GDPC1_z | FRED | GDP z-score |
| INDPRO_z | FRED | Industrial Production z-score |
| UMCSENT_z | FRED | Consumer Sentiment z-score |
| VIXCLS_z | FRED | VIX z-score |
| vix3m_z | Portara/FRED | 3M VIX z-score |
| VIX_Inversion_Z | Derived | VIX term structure inversion |
| Price_Mom_24H | Alpaca | 24H price momentum |
| CPI_Event_Active | Derived | CPI release window flag |
| NFP_Event_Active | Derived | NFP release window flag |

## Model Configuration

```python
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.03,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)
```

## Conviction-Based Allocation (Experimental)

**Script:** `04_Backtesting/12RT_Probability_Conviction_Backtest.py`

```python
# Conviction thresholds
CONVICTION_HIGH = 0.6   # ≥60% → Full allocation
CONVICTION_MED  = 0.4   # 40-60% → 70% allocation
# <40% → 40% allocation

# Allocation by regime
if regime == 0:  # Expansion
    equity = allocation_mult  # QQQ + CASH
elif regime == 1:  # Slowdown
    equity = allocation_mult  # SPY + CASH
else:  # Contraction
    if conviction == HIGH: equity = 0.0    # 100% CASH
    elif conviction == MED:  equity = 0.7   # 70% SPY
    else:                    equity = 0.6   # 60% SPY (low conviction)
```

**Backtest Comparison:**
- Baseline: 100% allocation to regime ETF
- Conviction-based: Scales exposure based on model confidence

## Training Logic

**Script:** `03_Modeling/06RT_XGBoost_Regime_V4_Training.py`

1. **Data Load:** Silver master grid from Unity Catalog
2. **Label Creation:** Forward 1-week SPY returns → regime labels
3. **Train/Test Split:** TimeSeriesSplit (walk-forward)
4. **Training:** XGBoost with class weights
5. **Evaluation:** Classification report, confusion matrix
6. **Feature Importance:** Saved for alpha analysis
7. **Model Export:** `bt_fred_1d_xgboost_regime.joblib`

## Inference Logic

**Script:** `02_RealTime/RT_04_Model_Inference_Engine.py`

```python
# Load latest silver row
latest_row = spark.sql("SELECT * FROM silver_rt_1h ORDER BY timestamp DESC LIMIT 1")

# Prepare features
X_rt = latest_row[FEATURES].fillna(0)

# Predict
regime_int = int(model.predict(X_rt)[0])
proba = model.predict_proba(X_rt)[0]

# Map to position
REGIME_MAP   = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
POSITION_MAP = {0: "100% QQQ",  1: "100% SPY",  2: "100% CASH"}
```

## Output Schema

**Table:** `bronze_rt_regime_signal`

| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMP | Inference timestamp |
| regime_predicted | INTEGER | 0/1/2 |
| regime_name | STRING | Expansion/Slowdown/Recession |
| position_recommended | STRING | 100% QQQ/SPY/CASH |
| prob_expansion | DOUBLE | P(regime=0) |
| prob_slowdown | DOUBLE | P(regime=1) |
| prob_recession | DOUBLE | P(regime=2) |
| [macro features...] | DOUBLE | All 14 features logged |

## Retraining Cadence

- **Frequency:** Weekly (recommended)
- **Trigger:** New macro data releases
- **Validation:** Walk-forward on expanding window
- **Model Registry:** Unity Catalog (pending implementation)
