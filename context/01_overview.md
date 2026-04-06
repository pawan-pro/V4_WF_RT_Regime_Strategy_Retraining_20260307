# Architecture Overview

## System Purpose
Real-time regime detection system for ETF rotation strategy using XGBoost classification.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION (Bronze Layer)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 01RT_Alpaca_Market_Ingestion.py      → SPY, QQQ, MTUM, USMV, IQLT, EFV, AGG │
│ 02RT_FinnHub_Macro_Ingestion.py      → CPI, PCE, NFP, VIX, etc.             │
│ 03RT_Firstrate_Ingestion.py          → Interest rates                        │
│ 04RT_Portara_VIX_Ingestion.py        → VIX3M term structure                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING (Silver Layer)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 05RT_Master_Time_Grid_Silver.py                                             │
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
│ 06RT_XGBoost_Regime_V4_Training.py                                          │
│   • Target: 1-week forward SPY return                                       │
│   • Labels: 0=Expansion, 1=Slowdown, 2=Recession                            │
│   • Walk-forward validation                                                 │
│   • Output: bt_fred_1d_xgboost_regime.joblib                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ RT_01_Live_Market_Ingestion.py       → Alpaca WebSocket → bronze_rt_market  │
│ RT_02_Live_Rates_VIX_Ingestion.py    → Databento/Polygon → rates + VIX      │
│ RT_03_Silver_RT_Enrichment.py        → Apply same feature logic as batch    │
│ RT_04_Model_Inference_Engine.py      → Load model → predict → signal        │
│ RT_05_Paper_Trade_Execution.py       → Alpaca Paper → orders                │
│ RT_06_Signal_Audit_Log.py            → Delta audit table                    │
│ RT_07_Dashboard_Feed.py              → PowerBI-ready tables                 │
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

Databricks workflows are versioned in `04_Orchestration/workflows/`

- `v4_rt_hourly_chain.json` runs the hourly live inference path
- `v4_macro_hourly_maintenance.json` updates macro and refreshes batch silver during the trading day
- `v4_monthly_retraining.json` runs end-of-day on the 1st of the month

## Backtesting

- `processing/retraining/Monthly_Walkforward_Retrain.py` – Production monthly walk-forward retraining and live-model promotion
- `research/testing/12RT_Probability_Conviction_Backtest.ipynb` – Research-only conviction experiment, not used in production
