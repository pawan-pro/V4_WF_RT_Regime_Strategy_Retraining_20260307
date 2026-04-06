# Current-State Databricks Inventory

As of 2026-04-06, this repository mixes three concerns in one catalog/schema and one primary volume family:

- Historical training/backtest data
- Frozen model artifacts and CSV exports
- Live real-time serving tables

That mixing is the main reason the repo is hard to reason about.

## Main Locations

| Location | Purpose | Notes |
|---|---|---|
| `/Volumes/pawan_quant_research_1/v4/market_data` | Main Unity Catalog volume used for BT market data, model artifacts, signal CSVs, and Finnhub intermediate CSVs | This is acting as both lake storage and artifact storage |
| `/Volumes/pawan_quant_research/default/firstratedata/futures_data/inception_till_10FEB26` | Raw FirstRate futures source files | Read-only external raw source for rates |
| `/Volumes/pawan_quant_research/default/portaradata/vix_data` | Raw Portara VIX source zip | Read-only external raw source for VIX futures |
| `pawan_quant_research_1.v4.*` | Current Unity Catalog schema for almost every managed table | Batch, RT, audit, and dashboard objects all coexist here |

## Current Object Inventory

### Historical / Training Objects

| Object | Kind | Current location | Created by | Upstream | Downstream |
|---|---|---|---|---|---|
| `bronze_bt_market_1h` | View over Delta path | `pawan_quant_research_1.v4.bronze_bt_market_1h` backed by `/Volumes/pawan_quant_research_1/v4/market_data/bronze_bt_market_1h` | `01_Data_Factory/01RT_Alpaca_Market_Ingestion.py` | Alpaca historical bars | `silver_bt_master_grid_1h`, walk-forward backtests, performance notebooks |
| `bronze_bt_macro_events` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_macro_events` | `01_Data_Factory/02RT_FinnHub_Macro_Ingestion.py` | Finnhub calendar + ALFRED first releases + CSV intermediates | `silver_bt_master_grid_1h` |
| `bronze_bt_rates_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_rates_1h` | `01_Data_Factory/03RT_Firstrate_Ingestion.py` | FirstRate futures zip | Primarily research/audit today; not wired into current silver build |
| `bronze_bt_vix_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_vix_1h` | `01_Data_Factory/04RT_Portara_VIX_Ingestion.py` | FRED or Portara, depending on `VIX_SOURCE` | `silver_bt_master_grid_1h` |
| `bronze_bt_vix3m_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_vix3m_1h` | `01_Data_Factory/04RT_Portara_VIX_Ingestion.py` | FRED or Portara/FRED mix, depending on `VIX_SOURCE` | `silver_bt_master_grid_1h` |
| `silver_bt_master_grid_1h` | Managed Delta table | `pawan_quant_research_1.v4.silver_bt_master_grid_1h` | `02_Feature_Engineering/05RT_Master_Time_Grid_Silver.py` | BT bronze market, macro, VIX, VIX3M | Model training, RT feature carry-forward, backtests |
| `bt_walkforward_regime_signals` | Managed Delta table | `pawan_quant_research_1.v4.bt_walkforward_regime_signals` | `04_Backtesting/11RT_WalkForward_Simulation.py` | `silver_bt_master_grid_1h` | Backtests and research notebooks |
| `gold_bt_returns_*` | Managed Delta tables | `pawan_quant_research_1.v4.gold_bt_returns_*` | `04_Backtesting/09RT_Performance_Deep_Dive.py`, `04_Backtesting/10RT_Performance_analysis.py` | Backtest outputs + market prices | Analytics only |
| `bt_fred_1d_xgboost_regime.joblib` | Volume artifact | `/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_xgboost_regime.joblib` | `03_Modeling/06RT_XGBoost_Regime_V4_Training.py` | `silver_bt_master_grid_1h` | RT inference |
| `bt_fred_1d_regime_signals-v4.csv` | Volume artifact | `/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv` | `03_Modeling/06RT_XGBoost_Regime_V4_Training.py` | Trained model + full silver history | Backtests and analysis |

### Real-Time Serving Objects

| Object | Kind | Current location | Created by | Upstream | Downstream |
|---|---|---|---|---|---|
| `bronze_rt_market_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_market_1h` | `02_RealTime/RT_01_Live_Market_Ingestion.py` | Alpaca latest bar / websocket feed | `silver_rt_1h`, paper trading, dashboard |
| `bronze_rt_vix_1d` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_vix_1d` | `02_RealTime/RT_02_Live_Rates_VIX_Ingestion.py` | FRED PIT-corrected VIXCLS/VXVCLS | `silver_rt_1h`, audit cells in RT_04/RT_06 |
| `silver_rt_1h` | Managed Delta table | `pawan_quant_research_1.v4.silver_rt_1h` | `02_RealTime/RT_03_Silver_RT_Enrichment.py` | RT market/VIX plus carry-forward values from `silver_bt_master_grid_1h` | `bronze_rt_regime_signal`, dashboard, audit |
| `bronze_rt_regime_signal` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_regime_signal` | `02_RealTime/RT_04_Model_Inference_Engine.py` | `silver_rt_1h` + frozen joblib model | Paper trading, audit, dashboard |
| `bronze_rt_paper_trade_log` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_paper_trade_log` | `02_RealTime/RT_05_Paper_Trade_Execution.py` | Regime signal + latest RT prices + Alpaca paper account | Audit, dashboard |
| `bronze_rt_signal_audit_log` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_signal_audit_log` | `02_RealTime/RT_06_Signal_Audit_Log.py` | Regime signal + silver RT + paper trade log + RT VIX source date | Dashboard, ops review |
| `gold_rt_dashboard_feed` | Managed Delta table | `pawan_quant_research_1.v4.gold_rt_dashboard_feed` | `02_RealTime/RT_07_Dashboard_Feed.py` | Signal, RT silver, trade log, audit log, latest prices | Power BI / dashboard |

## Actual End-to-End Lineage

```text
Alpaca historical bars -> bronze_bt_market_1h -> silver_bt_master_grid_1h
Finnhub + ALFRED -> bronze_bt_macro_events -> silver_bt_master_grid_1h
FRED/Portara VIX -> bronze_bt_vix_1h + bronze_bt_vix3m_1h -> silver_bt_master_grid_1h
silver_bt_master_grid_1h -> model training -> joblib + BT signal CSV

Alpaca live bars -> bronze_rt_market_1h -> silver_rt_1h
FRED live PIT VIX -> bronze_rt_vix_1d -> silver_rt_1h
silver_bt_master_grid_1h -> silver_rt_1h (macro and z-score carry-forward)
silver_rt_1h + joblib -> bronze_rt_regime_signal
bronze_rt_regime_signal -> bronze_rt_paper_trade_log
bronze_rt_regime_signal + silver_rt_1h + bronze_rt_paper_trade_log -> bronze_rt_signal_audit_log
bronze_rt_regime_signal + bronze_rt_paper_trade_log + bronze_rt_signal_audit_log -> gold_rt_dashboard_feed
```

## Drift And Risks Found During Inventory

- The repo documents `gold_rt_dashboard` in context files, but the RT notebook writes `gold_rt_dashboard_feed`.
- `04_Orchestration/Workflow_Definition.json.py` is effectively empty, so the deployed Databricks workflow is not actually versioned in the repo.
- `02_RealTime/RT_07_Dashboard_Feed.py` had a live-field mapping bug where `PAYEMS_z` was populated from `UNRATE_z`. This branch fixes that bug.
- `02_RealTime/RT_03_Silver_RT_Enrichment.py` does not fully recompute z-scores from live raw inputs. It carries forward macro and VIX z-scores from historical silver, which is operationally convenient but should be treated as a design choice, not a true standalone RT feature pipeline.
- Multiple notebooks contain hardcoded API keys, Databricks tokens, and secrets. Those should be removed before any production Databricks deployment.
- Training artifacts, intermediate CSVs, and live-serving objects all share the same logical namespace and partially the same storage root, which makes freeze, rollback, and audit difficult.
