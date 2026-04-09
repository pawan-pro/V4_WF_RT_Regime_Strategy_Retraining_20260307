# Architecture Overview

## System Purpose
Real-time regime detection system for ETF rotation strategy using XGBoost classification.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION (Bronze Layer)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 01_BT_Alpaca_Market_Ingestion.py     → SPY, QQQ, MTUM, USMV, IQLT, EFV, AGG │
│ 03_BT_Macro_Incremental_Update.py    → CPI, PCE, NFP, macro releases        │
│ 04_BT_Firstrate_Rates_Ingestion.py   → Interest rates                        │
│ 05_BT_Portara_VIX_Ingestion.py       → VIX3M term structure                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING (Silver Layer)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 01_BT_Master_Time_Grid_Silver.py                                            │
│   • Hourly backbone (2016–present)                                          │
│   • Point-in-time macro releases (+2H lag)                                  │
│   • VIX3M 1-day PIT shift                                                   │
│   • Z-score normalization                                                   │
│   • Event flags (CPI_Event_Active, NFP_Event_Active)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODELING (Gold Layer)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 01_Monthly_Walkforward_Retrain.py                                           │
│   • Target: 1-week forward SPY return                                       │
│   • Labels: 0=Expansion, 1=Slowdown, 2=Recession                            │
│   • Walk-forward validation                                                 │
│   • Output: bt_fred_1d_xgboost_regime.joblib                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 01_RT_Live_Market_Ingestion.py       → Alpaca WebSocket → bronze_rt_market  │
│ 02_RT_Live_Rates_VIX_Ingestion.py    → FRED PIT VIX                         │
│ 03_RT_Silver_RT_Enrichment.py        → Apply same feature logic as batch    │
│ 04_RT_Model_Inference_Engine.py      → Load model → predict → signal        │
│ 05_RT_Paper_Trade_Execution.py       → Alpaca Paper → orders                │
│ 06_RT_Signal_Audit_Log.py            → Delta audit table                    │
│ 07_RT_Dashboard_Feed.py              → PowerBI-ready tables                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Regime Logic

| Regime | Label | Forward Return | Position |
|--------|-------|----------------|----------|
| Expansion | 0 | > 0.5% | 100% QQQ |
| Slowdown  | 1 | -0.5% to 0.5% | 100% SPY |
| Recession | 2 | < -0.5% | 100% CASH |

## Key Features (14 total)

**Macro Z-scores:** CPIAUCSL, PCEPI, PCE, PAYEMS, UNRATE, GDPC1, INDPRO, UMCSENT, VIXCLS, v3m
**Technical:** VIX_Inversion_Z, Price_Mom_24H
**Event Flags:** CPI_Event_Active, NFP_Event_Active

## Orchestration

Databricks workflows are versioned in `04_Orchestration/01_Workflows/`

- `01_V4_RT_Hourly_Chain.json` runs the hourly live inference path
- `02_V4_Macro_Hourly_Maintenance.json` updates macro and refreshes batch silver during the trading day
- `03_V4_Monthly_Retraining.json` runs end-of-day on the 1st of the month

## Backtesting

- `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py` – Production monthly walk-forward retraining and live-model promotion
- `03_Research/01_Testing/04_Probability_Conviction_Backtest.ipynb` – Research-only conviction experiment, not used in production
