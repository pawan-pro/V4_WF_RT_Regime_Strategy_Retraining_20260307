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
| `bronze_bt_market_1h` | View over Delta path | `pawan_quant_research_1.v4.bronze_bt_market_1h` backed by `/Volumes/pawan_quant_research_1/v4/market_data/bronze_bt_market_1h` | `01_Processing/01_Ingestion/01_BT_Alpaca_Market_Ingestion.py` | Alpaca historical bars | `silver_bt_master_grid_1h`, monthly retraining, performance exports |
| `bronze_bt_macro_events` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_macro_events` | `01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py`, `01_Processing/01_Ingestion/03_BT_Macro_Incremental_Update.py` | Finnhub calendar + ALFRED first releases | `silver_bt_master_grid_1h`, monthly retraining, RT feature carry-forward |
| `bronze_bt_rates_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_rates_1h` | `01_Processing/01_Ingestion/04_BT_Firstrate_Rates_Ingestion.py` | FirstRate futures zip | Primarily research/audit today; not wired into current silver build |
| `bronze_bt_vix_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_vix_1h` | `01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py` | FRED or Portara, depending on `VIX_SOURCE` | `silver_bt_master_grid_1h` |
| `bronze_bt_vix3m_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_bt_vix3m_1h` | `01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py` | FRED or Portara/FRED mix, depending on `VIX_SOURCE` | `silver_bt_master_grid_1h` |
| `silver_bt_master_grid_1h` | Managed Delta table | `pawan_quant_research_1.v4.silver_bt_master_grid_1h` | `01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py`, `01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py` | BT bronze market, macro, VIX, VIX3M | Monthly training, RT feature carry-forward, backtests |
| `bt_walkforward_regime_signals` | Managed Delta table | `pawan_quant_research_1.v4.bt_walkforward_regime_signals` | `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py` | `silver_bt_master_grid_1h` | Performance exports and research notebooks |
| `gold_bt_returns_*` | Managed Delta tables | `pawan_quant_research_1.v4.gold_bt_returns_*` | `02_Analysis/01_Performance/03_Performance_Exports.py` | Walk-forward outputs + market prices | Analytics and reporting |
| `model_registry_current` | Managed Delta table | `pawan_quant_research_1.v4.model_registry_current` | `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py` | Monthly retraining output | RT inference, audit, dashboard |
| `bt_fred_1d_xgboost_regime_live.joblib` | Volume artifact | `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib` | `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py` | Monthly promoted model | RT inference |
| `bt_fred_1d_regime_signals-v4.csv` | Volume artifact | `/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv` | `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py` | Trained model + full silver history | Backtests and analysis |

### Real-Time Serving Objects

| Object | Kind | Current location | Created by | Upstream | Downstream |
|---|---|---|---|---|---|
| `bronze_rt_market_1h` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_market_1h` | `01_Processing/03_Realtime/01_RT_Live_Market_Ingestion.py` | Alpaca latest bar / websocket feed | `silver_rt_1h`, paper trading, dashboard |
| `bronze_rt_vix_1d` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_vix_1d` | `01_Processing/03_Realtime/02_RT_Live_Rates_VIX_Ingestion.py` | FRED PIT-corrected VIXCLS/VXVCLS | `silver_rt_1h`, audit, dashboard |
| `silver_rt_1h` | Managed Delta table | `pawan_quant_research_1.v4.silver_rt_1h` | `01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py` | RT market/VIX plus carry-forward values from `silver_bt_master_grid_1h` | `bronze_rt_regime_signal`, dashboard, audit |
| `bronze_rt_regime_signal` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_regime_signal` | `01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py` | `silver_rt_1h` + promoted live model alias | Paper trading, audit, dashboard |
| `bronze_rt_paper_trade_log` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_paper_trade_log` | `01_Processing/03_Realtime/05_RT_Paper_Trade_Execution.py` | Regime signal + latest RT prices + Alpaca paper account snapshot | Audit, dashboard |
| `bronze_rt_signal_audit_log` | Managed Delta table | `pawan_quant_research_1.v4.bronze_rt_signal_audit_log` | `01_Processing/03_Realtime/06_RT_Signal_Audit_Log.py` | Regime signal + silver RT + paper trade log + RT VIX source date | Dashboard, ops review |
| `gold_rt_dashboard_feed` | Managed Delta table | `pawan_quant_research_1.v4.gold_rt_dashboard_feed` | `01_Processing/03_Realtime/07_RT_Dashboard_Feed.py` | Signal, RT silver, trade log, audit log, latest prices, macro freshness | Power BI / dashboard |

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
- `04_Orchestration/Workflow_Definition.json.py` is still empty, but versioned workflow JSON now exists under `04_Orchestration/01_Workflows/`.
- Live operations are now split across three versioned workflow definitions under `04_Orchestration/01_Workflows/`: hourly RT inference, hourly macro maintenance, and monthly retraining.
- `01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py` does not fully recompute z-scores from live raw inputs. It carries forward macro and VIX z-scores from historical silver, which is operationally convenient but should be treated as a design choice, not a true standalone RT feature pipeline.
- Multiple notebooks contain hardcoded API keys, Databricks tokens, and secrets. Those should be removed before any production Databricks deployment.
- Training artifacts, intermediate CSVs, and live-serving objects now have a cleaner `v4` split, but historical and live outputs still share the same schema and volume root. Naming discipline remains important for rollback and audit.
