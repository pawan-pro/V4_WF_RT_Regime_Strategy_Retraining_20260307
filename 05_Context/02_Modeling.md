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
    objective="multi:softmax",
    num_class=3,
    eval_metric="merror",
    random_state=42
)
```

## Conviction-Based Allocation (Research Only)

**Script:** `03_Research/01_Testing/04_Probability_Conviction_Backtest.ipynb`

This notebook is not part of the production model or live execution path. The production system remains fixed allocation by predicted regime only.

```python
# Conviction thresholds
CONVICTION_HIGH = 0.6   # ≥60% → Full allocation
CONVICTION_MED  = 0.4   # 40-60% → 70% allocation
# <40% → 40% allocation

# Allocation by regime and conviction
# ┌─────────────┬───────────┬────────────┬──────────────┐
# │ Regime      │ Conviction│ Equity %   │ Position     │
# ├─────────────┼───────────┼────────────┼──────────────┤
# │ Expansion   │ HIGH      │ 100% QQQ   │ Full risk    │
# │ (QQQ)       │ MEDIUM    │ 70% QQQ    │ Moderate     │
# │             │ LOW       │ 40% QQQ    │ Conservative │
# ├─────────────┼───────────┼────────────┼──────────────┤
# │ Slowdown    │ HIGH      │ 100% SPY   │ Full hedge   │
# │ (SPY)       │ MEDIUM    │ 70% SPY    │ Moderate     │
# │             │ LOW       │ 40% SPY    │ Conservative │
# ├─────────────┼───────────┼────────────┼──────────────┤
# │ Contraction │ HIGH      │ 0% (CASH)  │ Full defense │
# │ (CASH)      │ MEDIUM    │ 30% SPY    │ Light exposure│
# │             │ LOW       │ 40% SPY    │ Uncertain    │
# └─────────────┴───────────┴────────────┴──────────────┘
```

**Backtest Comparison:**
- Baseline: 100% allocation to regime ETF (binary)
- Conviction-based: Scales exposure based on model confidence (probabilistic)

## Training Logic

**Production Script:** `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py`
**Research Baseline:** `03_Research/01_Testing/01_XGBoost_Regime_V4_Training.py`

1. **Data Load:** Silver master grid from Unity Catalog
2. **Label Creation:** Forward 1-week SPY returns → regime labels
3. **Train/Test Split:** Monthly expanding-window walk-forward
4. **Training:** XGBoost with monthly re-estimation on completed data
5. **Evaluation:** Monthly walk-forward validation metrics written to `/performance/summary/`
6. **Model Export:** Versioned monthly artifact under `/Volumes/pawan_quant_research_1/v4/market_data/models/`
7. **Promotion:** Latest validated model is copied to the stable live alias consumed by RT inference

## Inference Logic

**Script:** `01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py`

```python
# Load latest silver row
latest_row = spark.sql("SELECT * FROM silver_rt_1h ORDER BY timestamp DESC LIMIT 1")

# Prepare features
X_rt = latest_row[FEATURES].fillna(0)

# Predict
regime_int = int(model.predict(X_rt)[0])

# Map to position
REGIME_MAP   = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
POSITION_MAP = {0: "100% QQQ",  1: "100% SPY",  2: "100% CASH"}
```

## Output Schema

**Table:** `bronze_rt_regime_signal`

| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMP | Inference timestamp |
| regime_int | INTEGER | 0/1/2 |
| regime_name | STRING | Expansion/Slowdown/Recession |
| position | STRING | 100% QQQ/SPY/CASH |
| [macro features...] | DOUBLE | Full feature vector logged |
| model_version | STRING | Promoted monthly model version |

## Retraining Cadence

- **Frequency:** Monthly, on the 1st day of the month
- **Trigger:** Prior month close plus fresh historical slices
- **Validation:** Monthly expanding-window walk-forward
- **Model Registry:** `pawan_quant_research_1.v4.model_registry_current`
