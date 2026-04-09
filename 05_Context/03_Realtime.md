# Real-Time Pipeline

## Pipeline Stages

### Stage 1: Market Data Ingestion
**Script:** `01_Processing/03_Realtime/01_RT_Live_Market_Ingestion.py`

- **Source:** Alpaca WebSocket (live streaming)
- **Tickers:** SPY, QQQ, MTUM, USMV, IQLT, EFV, AGG, HYG
- **Frequency:** 1-minute bars → aggregated to 1H
- **Output:** `bronze_rt_market_1h` (Delta table)

```python
# Config
CATALOG = "pawan_quant_research_1"
SCHEMA  = "v4"
VOLUME  = "market_data"
```

---

### Stage 2: Macro & VIX Ingestion
**Scripts:**
- `01_Processing/03_Realtime/02_RT_Live_Rates_VIX_Ingestion.py` → RT VIX spot
- `01_Processing/01_Ingestion/03_BT_Macro_Incremental_Update.py` → Economic indicators
- `01_Processing/01_Ingestion/04_BT_Firstrate_Rates_Ingestion.py` → Interest rates
- `01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py` → VIX3M term structure

**Output Tables:**
- `bronze_rt_macro_events`
- `bronze_rt_vix_1h`
- `bronze_rt_rates_1h`

---

### Stage 3: Silver RT Enrichment
**Script:** `01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py`

Applies the **same feature logic** as batch training:

1. **Time Grid:** Hourly backbone aligned to market hours
2. **PIT Adjustments:**
   - Macro releases: +2H lag
   - VIX3M: +1D lag (if Portara source)
3. **Z-Scores:** Rolling normalization vs. historical mean/std
4. **Event Flags:** CPI/NFP active windows
5. **Technical Features:** VIX inversion, 24H momentum

**Output:** `silver_rt_1h` (single latest row ready for inference)

---

### Stage 4: Model Inference
**Script:** `01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py`

```python
# Load model
model = joblib.load("/Volumes/.../models/bt_fred_1d_xgboost_regime_live.joblib")

# Get latest features
latest_row = spark.sql("SELECT * FROM silver_rt_1h ORDER BY timestamp DESC LIMIT 1")

# Predict
regime = model.predict(X_rt)[0]

# Write signal
spark.sql(f"""
    INSERT INTO bronze_rt_regime_signal 
    VALUES (timestamp, regime, ...)
""")
```

**Output:** `bronze_rt_regime_signal`

---

### Stage 5: Trade Execution
**Scripts:**
- `01_Processing/03_Realtime/05_RT_Paper_Trade_Execution.py` → Alpaca Paper Trading
- `01_Processing/03_Realtime/08_RT_Saxo_Trade_Execution.py` → Saxo Markets (production)

**Logic:**
```python
POSITION_MAP = {
    0: "100% QQQ",   # Expansion
    1: "100% SPY",   # Slowdown
    2: "100% CASH"   # Recession
}
```

---

### Stage 6: Signal Audit Log
**Script:** `01_Processing/03_Realtime/06_RT_Signal_Audit_Log.py`

Appends every signal + portfolio state to Delta table for:
- Compliance tracking
- Performance attribution
- Debugging

**Output:** `bronze_rt_signal_audit`

---

### Stage 7: Dashboard Feed
**Script:** `01_Processing/03_Realtime/07_RT_Dashboard_Feed.py`

Writes PowerBI-ready Delta tables:
- Current regime + full feature context
- Position history
- P&L tracking (if live)

**Output:** `gold_rt_dashboard_feed`

---

## Execution Order

```
RT_01 → RT_02 → RT_03 → RT_04 → RT_05 → RT_06 → RT_07
  ↓       ↓       ↓       ↓       ↓       ↓       ↓
Market  Macro   Silver  Infer   Trade   Audit   Dashboard
```

## Orchestration

Managed via **Databricks Workflows** in `04_Orchestration/01_Workflows/`

- **Schedule:** Every hour during market hours (9:30 AM - 4:00 PM ET)
- **Dependencies:** RT_02 depends on RT_01, RT_03 depends on RT_01+RT_02, etc.
- **Retry Logic:** 3 retries with exponential backoff
- **Alerting:** Email/Slack on failure
- **Macro Maintenance:** `02_V4_Macro_Hourly_Maintenance.json` runs `03_BT_Macro_Incremental_Update` and `02_BT_Incremental_Silver_Refresh` hourly on weekdays so new macro releases can flow into refreshed silver state before later inference runs

## Latency Targets

| Stage | Target Latency |
|-------|----------------|
| Ingestion | < 5 seconds |
| Silver Enrichment | < 30 seconds |
| Inference | < 5 seconds |
| Execution | < 10 seconds |
| **End-to-End** | **< 1 minute** |
