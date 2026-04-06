# Configuration & Environment

## Databricks Environment

```python
CATALOG = "pawan_quant_research_1"
SCHEMA  = "v4"
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
| `gold_rt_dashboard_feed` | PowerBI-ready analytics |

## Model Artifacts

| File | Path | Description |
|------|------|-------------|
| Versioned Monthly Model | `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_YYYY_MM_01.joblib` | Monthly walk-forward promoted artifact |
| Live Inference Alias | `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib` | Stable RT_04 load target |
| Performance Exports | `/Volumes/pawan_quant_research_1/v4/market_data/performance/` | Daily/weekly/monthly/quarterly/yearly/summary CSV outputs |

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
# In 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py
VIX_SOURCE = "fred_1d"   # Options: "fred_1d" | "portara_1h"

# fred_1d:    Daily FRED data, PIT already applied at ingestion
# portara_1h: Hourly Portara data, apply 1-day shift in Silver
```

## Databricks Workflow Config

**Files:** `04_Orchestration/01_Workflows/01_V4_RT_Hourly_Chain.json`, `04_Orchestration/01_Workflows/02_V4_Macro_Hourly_Maintenance.json`, `04_Orchestration/01_Workflows/03_V4_Monthly_Retraining.json`

```json
{
  "hourly_chain": {
    "schedule": "hourly on trading days shortly after bar close",
    "tasks": ["RT_01", "RT_02", "RT_03", "RT_04", "RT_05", "RT_06", "RT_07"]
  },
  "macro_hourly_maintenance": {
    "schedule": "hourly on weekdays during US release windows",
    "tasks": ["macro_incremental", "silver_refresh"]
  },
  "monthly_retraining": {
    "schedule": "1st day of month at end of US trading day",
    "tasks": ["macro_incremental", "silver_refresh", "monthly_walkforward_retrain", "performance_exports"]
  }
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
│   ├── 01_Global_Constants.py
│   └── 02_Databricks_Frozen_Manifest.py
├── 01_Processing/
│   ├── 01_Ingestion/
│   ├── 02_Features/
│   ├── 03_Realtime/
│   └── 04_Retraining/
├── 02_Analysis/
│   ├── 01_Performance/
│   └── 02_Dashboard/
├── 03_Research/
│   ├── 01_Testing/
│   ├── 02_Archive/
│   └── 03_Artifacts/
├── 04_Orchestration/
│   └── 01_Workflows/
├── 05_Context/
├── 06_Databricks/
├── 07_Docs/
└── 98_Legacy/
```
