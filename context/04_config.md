# Configuration & Environment

## Databricks Environment

```python
CATALOG = "pawan_quant_research_1"
SCHEMA  = "default"
VOLUME  = "market_data"
```

## Unity Catalog Paths

### Bronze Tables (Raw Data)
| Table | Path | Description |
|-------|------|-------------|
| `bronze_bt_market_1h` | `/Volumes/{catalog}/{schema}/{volume}/bronze_bt_market_1h` | Historical hourly OHLCV |
| `bronze_bt_macro_events` | Delta table | Historical macro releases |
| `bronze_bt_vix_1h` | Delta table | Historical VIX spot |
| `bronze_bt_vix3m_1h` | Delta table | Historical VIX3M |
| `bronze_rt_market_1h` | Delta table | Live market data |
| `bronze_rt_regime_signal` | Delta table | Inference signals |

### Silver Tables (Features)
| Table | Description |
|-------|-------------|
| `silver_bt_master_grid_1h` | Batch training feature grid |
| `silver_rt_1h` | Real-time feature row |

### Gold Tables (Analytics)
| Table | Description |
|-------|-------------|
| `gold_rt_dashboard` | PowerBI-ready analytics |

## Model Artifacts

| File | Path | Description |
|------|------|-------------|
| Training Model | `/Volumes/pawan_quant_research_1/default/market_data/bt_fred_1d_xgboost_regime.joblib` | Batch-trained XGBoost |
| Inference Model | Same as above | Loaded by RT_04 |

## API Integrations

| Provider | Purpose | Secrets Scope |
|----------|---------|---------------|
| Alpaca | Market data + execution | `alpaca` (api_key_id, api_secret_key) |
| FinnHub | Macro economic calendar | `finnhub` |
| Firstrate | Interest rates | `firstrate` |
| Portara | VIX3M term structure | `portara` |
| Databento | Alternative rates/VIX | `databento` |
| Polygon | Backup market data | `polygon` |

## Environment Variables

```bash
# Databricks Secrets (configure via UI)
dbutils.secrets.get(scope="alpaca", key="api_key_id")
dbutils.secrets.get(scope="alpaca", key="api_secret_key")

# Timezone
TZ = "America/New_York"  # Market hours in ET
```

## Key Constants

```python
# Tickers
ETFS = ["SPY", "QQQ", "MTUM", "USMV", "IQLT", "EFV", "AGG"]
CREDIT_PROXY = ["HYG"]
ALL_TICKERS = ETFS + CREDIT_PROXY

# Date Range
START_YEAR = 2016
END_YEAR = 2026

# Regime Thresholds
EXPANSION_THRESHOLD = 0.005    # > 0.5% forward return
RECESSION_THRESHOLD = -0.005   # < -0.5% forward return

# PIT Lags
MACRO_LAG_HOURS = 2    # Macro releases +2H after announcement
VIX3M_LAG_DAYS = 1     # VIX3M +1D (if Portara source)
```

## VIX Source Configuration

```python
# In 05RT_Master_Time_Grid_Silver.py
VIX_SOURCE = "fred_1d"   # Options: "fred_1d" | "portara_1h"

# fred_1d:    Daily FRED data, PIT already applied at ingestion
# portara_1h: Hourly Portara data, apply 1-day shift in Silver
```

## Databricks Workflow Config

**File:** `04_Orchestration/Workflow_Definition.json`

```json
{
  "schedule": "0 9-16 * * 1-5",  # Every hour, 9 AM - 4 PM, Mon-Fri
  "timezone": "America/New_York",
  "tasks": ["RT_01", "RT_02", "RT_03", "RT_04", "RT_05", "RT_06", "RT_07"],
  "retry_policy": {"max_retries": 3, "backoff_multiplier": 2}
}
```

## Python Dependencies

```
# Core
pandas
numpy
pyspark (Databricks Runtime 13.3+)

# ML
xgboost
scikit-learn
joblib

# Data Sources
alpaca-py
databento

# Utilities
zoneinfo
datetime
```

## Directory Structure

```
V4_WF_RT_Regime_Strategy_Retraining_20260307/
├── 00_Config/
│   └── Global_Constants.py
├── 01_Data_Factory/
│   ├── 01RT_Alpaca_Market_Ingestion.py
│   ├── 02RT_FinnHub_Macro_Ingestion.py
│   ├── 03RT_Firstrate_Ingestion.py
│   └── 04RT_Portara_VIX_Ingestion.py
├── 02_Feature_Engineering/
│   └── 05RT_Master_Time_Grid_Silver.py
├── 02_RealTime/
│   ├── RT_01_Live_Market_Ingestion.py
│   ├── RT_02_Live_Rates_VIX_Ingestion.py
│   ├── RT_03_Silver_RT_Enrichment.py
│   ├── RT_04_Model_Inference_Engine.py
│   ├── RT_05_Paper_Trade_Execution.py
│   ├── RT_05A_Saxo_Trade_Execution.py
│   ├── RT_06_Signal_Audit_Log.py
│   └── RT_07_Dashboard_Feed.py
├── 03_Modeling/
│   └── 06RT_XGBoost_Regime_V4_Training.py
├── 04_Orchestration/
│   └── Workflow_Definition.json
└── context/
    ├── overview.md
    ├── modeling.md
    ├── realtime.md
    └── config.md
```
