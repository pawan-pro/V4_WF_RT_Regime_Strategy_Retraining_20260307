# Training vs Production Data Inventory

## 1. Executive Summary
The current V4 model trains from the historical batch silver table `pawan_quant_research_1.v4.silver_bt_master_grid_1h`, which is built from historical market, macro, and VIX bronze tables. The monthly retraining notebook derives labels internally from forward 1-week SPY returns and fits the XGBoost regime classifier on the 14 canonical engineered features.

The live production inference path does **not** rebuild the model from scratch. Instead it assembles one live feature row in `pawan_quant_research_1.v4.silver_rt_1h`, loads the promoted model referenced by `pawan_quant_research_1.v4.model_registry_current`, and writes live predictions to `pawan_quant_research_1.v4.bronze_rt_regime_signal`.

Training and production overlap in the **feature names and intended semantics**: both use the same 14 feature columns. They differ in how those columns are sourced at runtime. Training uses the fully historical batch silver grid; production uses a hybrid RT row that carries forward most historical silver state, recomputes live VIX and price momentum, and only allows live macro overrides for a small subset of indicators (`CPIAUCSL`, `UMCSENT`).

Dashboard and monitoring objects sit downstream of inference. They read the live signal table, live trade/audit logs, dashboard gold tables, and selected historical/raw reference tables for validation and audit. They are not part of model fitting and do not change the model prediction path.

A practical distinction to keep in mind:
- **Training / retraining** = batch historical feature grid + historical labels + model artifacts/validation outputs.
- **Production inference** = live RT feature row + promoted model artifact/registry + live signal output.
- **Dashboard / monitoring** = read-only views over live signal, execution, audit, health, performance, and selected historical reference tables.

Per project operating memory, the canonical live producer is `v4_saxo_live_hourly_chain`. The older `v4_rt_hourly_chain` remains a downstream Alpaca simulation path and should not be treated as the primary live production model path.

## 2. Canonical Model Feature List
The canonical model features are exactly:
- `CPIAUCSL_z`
- `PCEPI_z`
- `PCE_z`
- `PAYEMS_z`
- `UNRATE_z`
- `GDPC1_z`
- `INDPRO_z`
- `UMCSENT_z`
- `VIXCLS_z`
- `vix3m_z`
- `VIX_Inversion_Z`
- `Price_Mom_24H`
- `CPI_Event_Active`
- `NFP_Event_Active`

| Feature | Immediate training source | Immediate production source | Raw source family | Raw lineage summary | Used in both training and production |
| --- | --- | --- | --- | --- | --- |
| CPIAUCSL_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.CPIAUCSL_z | pawan_quant_research_1.v4.silver_rt_1h.CPIAUCSL_z | macro | Training: bronze_bt_macro_events.actual where indicator_id='CPIAUCSL' -> +2h PIT join -> forward fill -> 1y z-score. Production: same historical carry-forward, with optional live override from bronze_rt_macro_events_live.observed_value for CPI releases. | yes |
| PCEPI_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.PCEPI_z | pawan_quant_research_1.v4.silver_rt_1h.PCEPI_z | macro | bronze_bt_macro_events.actual where indicator_id='PCEPI' -> +2h PIT join -> forward fill -> 1y z-score. RT path carries the latest historical silver value forward; no dedicated live override. | yes |
| PCE_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.PCE_z | pawan_quant_research_1.v4.silver_rt_1h.PCE_z | macro | bronze_bt_macro_events.actual where indicator_id='PCE' -> +2h PIT join -> forward fill -> 1y z-score. RT path carries the latest historical silver value forward. | yes |
| PAYEMS_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.PAYEMS_z | pawan_quant_research_1.v4.silver_rt_1h.PAYEMS_z | macro | bronze_bt_macro_events.actual where indicator_id='PAYEMS' -> +2h PIT join -> forward fill -> 1y z-score. RT path carries the latest historical silver value forward. | yes |
| UNRATE_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.UNRATE_z | pawan_quant_research_1.v4.silver_rt_1h.UNRATE_z | macro | bronze_bt_macro_events.actual where indicator_id='UNRATE' -> +2h PIT join -> forward fill -> 1y z-score. RT path carries the latest historical silver value forward. | yes |
| GDPC1_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.GDPC1_z | pawan_quant_research_1.v4.silver_rt_1h.GDPC1_z | macro | bronze_bt_macro_events.actual where indicator_id='GDPC1' -> +2h PIT join -> forward fill -> 1y z-score. RT path carries the latest historical silver value forward. | yes |
| INDPRO_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.INDPRO_z | pawan_quant_research_1.v4.silver_rt_1h.INDPRO_z | macro | bronze_bt_macro_events.actual where indicator_id='INDPRO' -> +2h PIT join -> forward fill -> 1y z-score. RT path carries the latest historical silver value forward. | yes |
| UMCSENT_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.UMCSENT_z | pawan_quant_research_1.v4.silver_rt_1h.UMCSENT_z | macro | Training: bronze_bt_macro_events.actual where indicator_id='UMCSENT' -> +2h PIT join -> forward fill -> 1y z-score. Production: same carry-forward with optional final-only live override from bronze_rt_macro_events_live.observed_value. | yes |
| VIXCLS_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.VIXCLS_z | pawan_quant_research_1.v4.silver_rt_1h.VIXCLS_z | VIX | Training: bronze_bt_vix_1h.close -> forward fill -> 1y z-score. Production: bronze_rt_vix_1d.close where series='VIXCLS', re-zscored against historical bronze_bt_vix_1h context. | yes |
| vix3m_z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.vix3m_z | pawan_quant_research_1.v4.silver_rt_1h.vix3m_z | VIX | Training: bronze_bt_vix3m_1h.close/vix3m -> forward fill -> 1y z-score. Production: bronze_rt_vix_1d.close where series='vix3m', re-zscored against historical bronze_bt_vix3m_1h context. | yes |
| VIX_Inversion_Z | pawan_quant_research_1.v4.silver_bt_master_grid_1h.VIX_Inversion_Z | pawan_quant_research_1.v4.silver_rt_1h.VIX_Inversion_Z | derived | Derived in both paths as VIXCLS_z - vix3m_z after the individual VIX z-scores are computed. | yes |
| Price_Mom_24H | pawan_quant_research_1.v4.silver_bt_master_grid_1h.Price_Mom_24H | pawan_quant_research_1.v4.silver_rt_1h.Price_Mom_24H | market / derived | Training: derived from silver_bt_master_grid_1h.SPY_close using 1-hour and 25-hour lags. Production: derived from current bronze_rt_market_1h SPY close plus 24h price context assembled from bronze_bt_market_1h and bronze_rt_market_1h. | yes |
| CPI_Event_Active | pawan_quant_research_1.v4.silver_bt_master_grid_1h.CPI_Event_Active | pawan_quant_research_1.v4.silver_rt_1h.CPI_Event_Active | derived macro event flag | Training: pulse window over CPI shock computed from consecutive CPI levels in silver build. Production: carried forward from latest historical silver row; not recomputed from the live macro schedule table. | yes |
| NFP_Event_Active | pawan_quant_research_1.v4.silver_bt_master_grid_1h.NFP_Event_Active | pawan_quant_research_1.v4.silver_rt_1h.NFP_Event_Active | derived macro event flag | Training: pulse window over PAYEMS shock computed from consecutive payroll levels in silver build. Production: carried forward from latest historical silver row; not recomputed from live NFP schedule rows. | yes |

## 3. Training Data Inventory
The main training/retraining objects are below. "Training" here includes direct model-fitting inputs and the monthly retraining outputs that are generated alongside the promoted model.

| Data object | Type | Frequency | Purpose | Producer | Consumer | Direct training use |
| --- | --- | --- | --- | --- | --- | --- |
| /Volumes/pawan_quant_research_1/v4/market_data/raw_finnhub_haystack.csv | file | historical event snapshot | Historical Finnhub US macro event haystack used to map release dates to ALFRED/FRED vintages during the bootstrap build. | 01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py | 01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py | indirect |
| /Volumes/pawan_quant_research_1/v4/market_data/first_releases_<indicator>.csv | file_group | event-driven historical bootstrap | Per-indicator ALFRED/FRED first-release extracts written before bronze consolidation. | 01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py | 01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py | indirect |
| pawan_quant_research_1.v4.bronze_bt_macro_events | table | event-driven; refreshed hourly in maintenance jobs | Canonical batch macro release history used to build point-in-time macro features. | 01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py; 01_Processing/01_Ingestion/03_BT_Macro_Incremental_Update.py; workflow 02_V4_Macro_Hourly_Maintenance.json | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; dashboard/feed notebooks | indirect |
| pawan_quant_research_1.v4.bronze_bt_market_1h | table | 1h historical bars | Historical hourly ETF price backbone and label context. | 01_Processing/01_Ingestion/01_BT_Alpaca_Market_Ingestion.py | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py; 02_Analysis/01_Performance/03_Performance_Exports.py; dashboard notebook | indirect |
| pawan_quant_research_1.v4.bronze_bt_vix_1h | table | 1h PIT-expanded VIX spot | Historical VIX spot context for VIXCLS_z and VIX inversion features. | 01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; dashboard notebook | indirect |
| pawan_quant_research_1.v4.bronze_bt_vix3m_1h | table | 1h PIT-expanded VIX 3M term structure | Historical VIX 3M context for vix3m_z and VIX inversion features. | 01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; dashboard notebook | indirect |
| pawan_quant_research_1.v4.silver_bt_master_grid_1h | table | 1h historical feature grid | Canonical historical feature table for training and the carry-forward reference for RT enrichment. | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; dashboard notebook | direct |
| pawan_quant_research_1.v4.bt_walkforward_regime_signals | table | 1h historical walk-forward predictions | Walk-forward validation output from monthly retraining folds. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | 02_Analysis/01_Performance/03_Performance_Exports.py; research/comparison notebooks | output |
| /Volumes/pawan_quant_research_1/v4/market_data/bt_walkforward_regime_signals.csv | file | monthly retrain export | CSV export of walk-forward regime signals. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | analysis/comparison notebooks and exports | output |
| /Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv | file | monthly retrain export | Full-history signal export from the final retrained model. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | analysis/comparison notebooks | output |
| /Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_<YYYY_MM_DD>.joblib | file | monthly when retraining promotes a new version | Versioned model artifact preserved for audit and rollback. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | rollback/reference via model_registry_current.versioned_model_path | output |
| /Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib | file | monthly when a model is promoted | Promoted live model alias used by production inference and dashboard inspection. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | 01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py; 02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb | output |
| pawan_quant_research_1.v4.model_registry_current | table | monthly promotion overwrite | Stores the currently promoted model metadata and pointers to active artifacts/exports. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | 01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py; 01_Processing/03_Realtime/07_RT_Dashboard_Feed.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook | output |
| /Volumes/pawan_quant_research_1/v4/market_data/performance/summary/monthly_walkforward_validation.csv | file | monthly retrain export | Monthly fold accuracy summary for walk-forward validation. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | human review / comparison notebooks | output |
| pawan_quant_research_1.v4.gold_bt_returns_{daily,weekly,monthly,quarterly,yearly} | table_group | monthly export refresh | Canonical walk-forward performance return tables derived from bt_walkforward_regime_signals and bronze_bt_market_1h. | 02_Analysis/01_Performance/03_Performance_Exports.py | performance analysis / reporting | output |
| pawan_quant_research_1.v4.gold_bt_performance_summary | table | monthly export refresh | Summary performance table for walk-forward strategy and SPY benchmark. | 02_Analysis/01_Performance/03_Performance_Exports.py | performance analysis / reporting | output |

Additional training notes:
- The retraining notebook reads `silver_bt_master_grid_1h` into pandas, then creates `fwd_return_1w` from `SPY_close.shift(-168) / SPY_close - 1`.
- Labels are derived as: `0` if 1-week forward SPY return > 0.5%, `2` if < -0.5%, else `1`.
- `QQQ_close` exists in the batch silver table for strategy evaluation/backtests, but it is **not** part of the feature list used for model fitting.
- `bronze_bt_rates_1h` exists in the repo but is not consumed by the current silver build or retraining code.

## 4. Production Inference Data Inventory
These are the main data/config objects used in the current live inference path.

| Data object | Type | Frequency / cadence | Purpose | Producer | Consumer | Live mode |
| --- | --- | --- | --- | --- | --- | --- |
| pawan_quant_research_1.v4.bronze_bt_macro_events | table | event-driven; refreshed hourly in maintenance jobs | Canonical batch macro release history used to build point-in-time macro features. | 01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py; 01_Processing/01_Ingestion/03_BT_Macro_Incremental_Update.py; workflow 02_V4_Macro_Hourly_Maintenance.json | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; dashboard/feed notebooks | hybrid |
| pawan_quant_research_1.v4.bronze_bt_market_1h | table | 1h historical bars | Historical hourly ETF price backbone and label context. | 01_Processing/01_Ingestion/01_BT_Alpaca_Market_Ingestion.py | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py; 02_Analysis/01_Performance/03_Performance_Exports.py; dashboard notebook | hybrid |
| pawan_quant_research_1.v4.bronze_bt_vix_1h | table | 1h PIT-expanded VIX spot | Historical VIX spot context for VIXCLS_z and VIX inversion features. | 01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; dashboard notebook | hybrid |
| pawan_quant_research_1.v4.bronze_bt_vix3m_1h | table | 1h PIT-expanded VIX 3M term structure | Historical VIX 3M context for vix3m_z and VIX inversion features. | 01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; dashboard notebook | hybrid |
| pawan_quant_research_1.v4.silver_bt_master_grid_1h | table | 1h historical feature grid | Canonical historical feature table for training and the carry-forward reference for RT enrichment. | 01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py; 01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py; 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; dashboard notebook | metadata/output |
| /Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib | file | monthly when a model is promoted | Promoted live model alias used by production inference and dashboard inspection. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | 01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py; 02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb | metadata/output |
| pawan_quant_research_1.v4.model_registry_current | table | monthly promotion overwrite | Stores the currently promoted model metadata and pointers to active artifacts/exports. | 01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py | 01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py; 01_Processing/03_Realtime/07_RT_Dashboard_Feed.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook | metadata/output |
| pawan_quant_research_1.v4.rt_runtime_instrument_config | table | as-needed config updates | Defines live symbols, primary/secondary tickers, and regime-to-position mapping. | 06_Databricks/02_Scripts/02_Runtime_Instrument_Config (helper) | 01_Processing/03_Realtime/01_RT_Live_Market_Ingestion.py; 01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py; 01_Processing/03_Realtime/08_RT_Saxo_Trade_Execution.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py | live-only |
| pawan_quant_research_1.v4.bronze_rt_market_1h | table | hourly live snapshot | Live hourly market bars for current SPY/QQQ state. | 01_Processing/03_Realtime/01_RT_Live_Market_Ingestion.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; 01_Processing/03_Realtime/08_RT_Saxo_Trade_Execution.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook | live-only |
| pawan_quant_research_1.v4.bronze_rt_vix_1d | table | daily PIT-safe live snapshot, refreshed hourly | Live VIX and VIX 3M inputs with PIT shift preserved for RT feature recomputation. | 01_Processing/03_Realtime/02_RT_Live_Rates_VIX_Ingestion.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; 01_Processing/03_Realtime/09_RT_Saxo_Signal_Audit_Log.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook | live-only |
| pawan_quant_research_1.v4.bronze_rt_macro_events_live | table | event-driven live schedule / observation rows | Live macro schedule and observed-value capture for the small set of indicators currently allowed to override RT features. | 01_Processing/03_Realtime/02_RT_Live_Macro_Events.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; dashboard notebook | live-only |
| pawan_quant_research_1.v4.silver_rt_1h | table | hourly live feature row | Inference-ready live feature row aligned to the historical feature schema. | 01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py; 01_Processing/03_Realtime/09_RT_Saxo_Signal_Audit_Log.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook | hybrid |
| pawan_quant_research_1.v4.bronze_rt_regime_signal | table | hourly live prediction | Stores the latest live regime prediction and the exact feature snapshot used by the live model. | 01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 01_Processing/03_Realtime/08_RT_Saxo_Trade_Execution.py; 01_Processing/03_Realtime/09_RT_Saxo_Signal_Audit_Log.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook | live-only |

Key live-path behavior:
- `01_RT_Live_Market_Ingestion.py` writes the latest live market bars to `bronze_rt_market_1h`.
- `02_RT_Live_Rates_VIX_Ingestion.py` writes PIT-safe daily VIX/VIX3M values to `bronze_rt_vix_1d`.
- `02_RT_Live_Macro_Events.py` writes live macro schedule/observed rows to `bronze_rt_macro_events_live`, but the current override scope is intentionally narrow.
- `03_RT_Silver_RT_Enrichment.py` creates one live row in `silver_rt_1h` by combining:
  - live SPY close from `bronze_rt_market_1h`
  - live VIX/VIX3M from `bronze_rt_vix_1d`
  - historical carry-forward from `silver_bt_master_grid_1h`
  - optional live macro override only for `CPIAUCSL_z` and `UMCSENT_z`
- `04_RT_Model_Inference_Engine.py` loads the promoted model from `model_registry_current` / live alias and predicts on `silver_rt_1h` only.

## 5. Dashboard / Monitoring-Only Data Inventory
These objects exist primarily to support monitoring, execution-state visibility, or dashboard presentation.

| Data object | Type | Frequency | Purpose | Producer | Consumer |
| --- | --- | --- | --- | --- | --- |
| pawan_quant_research_1.v4.bronze_rt_saxo_trade_log | table | hourly live execution snapshot | Execution/account-state log for the live Saxo strategy after inference has already happened. | 01_Processing/03_Realtime/08_RT_Saxo_Trade_Execution.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 01_Processing/03_Realtime/09_RT_Saxo_Signal_Audit_Log.py; 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook |
| pawan_quant_research_1.v4.bronze_rt_saxo_signal_audit_log | table | hourly live audit row | Auditable join of live signal, live silver features, execution state, and VIX bias-gate metadata. | 01_Processing/03_Realtime/09_RT_Saxo_Signal_Audit_Log.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; dashboard notebook |
| pawan_quant_research_1.v4.gold_rt_saxo_dashboard_feed | table | hourly live dashboard snapshot | Single latest snapshot table for the institutional dashboard. | 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py; workflow 06_V4_Saxo_Live_Hourly_Chain.json | 02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb |
| pawan_quant_research_1.v4.gold_rt_saxo_dashboard_health_rollup | table | hourly health rollup | Dashboard health/freshness monitoring rollup. | 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py | 02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb |
| pawan_quant_research_1.v4.gold_rt_saxo_dashboard_performance_rollup | table | hourly performance rollup | Dashboard-friendly strategy/SPY performance rollup history. | 01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py | 02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb |
| Alpaca simulation monitoring tables (pawan_quant_research_1.v4.gold_rt_dashboard_feed; pawan_quant_research_1.v4.bronze_rt_paper_trade_log; pawan_quant_research_1.v4.bronze_rt_signal_audit_log) | table_group | hourly in downstream simulation chain | Parallel Alpaca simulation path kept for downstream-only monitoring. | 01_Processing/03_Realtime/05_RT_Paper_Trade_Execution.py; 01_Processing/03_Realtime/06_RT_Signal_Audit_Log.py; 01_Processing/03_Realtime/07_RT_Dashboard_Feed.py; workflow 01_V4_RT_Hourly_Chain.json | legacy/paper dashboards and monitoring |

The institutional dashboard notebook (`02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb`) also reads several shared reference objects directly for audit/monitoring context, including:
- `gold_rt_saxo_dashboard_feed`
- `gold_rt_saxo_dashboard_health_rollup`
- `gold_rt_saxo_dashboard_performance_rollup`
- `bronze_rt_regime_signal`
- `silver_rt_1h`
- `silver_bt_master_grid_1h`
- `bronze_bt_macro_events`
- `bronze_rt_macro_events_live`
- `bronze_bt_market_1h`
- `bronze_rt_market_1h`
- `bronze_bt_vix_1h`
- `bronze_bt_vix3m_1h`
- `bronze_rt_vix_1d`
- `bronze_rt_saxo_trade_log`
- `model_registry_current`
- `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`

That dashboard notebook is read-only presentation logic; it does not participate in prediction generation.

## 6. Training vs Production Comparison
**Shared across training and production**
- Same 14 canonical feature names.
- Same historical macro/VIX market lineage as the base truth set.
- Same promoted model family (`XGBoostClassifier`) and model registry concept.
- Same historical silver table semantics for macro/VIX-derived features.

**Different between training and production**
- Training uses the full historical batch feature table `silver_bt_master_grid_1h`; production uses a single live row in `silver_rt_1h`.
- Training labels are created from future SPY returns; production does not create labels.
- Production inference is hybrid: it carries forward historical batch silver state, then overlays live market/VIX data and a narrow live macro override set.
- Training never needs `bronze_rt_market_1h`, `bronze_rt_vix_1d`, or `bronze_rt_macro_events_live`; production relies on them.
- Production execution/dashboard state lives in downstream RT tables (`bronze_rt_regime_signal`, `bronze_rt_saxo_trade_log`, `bronze_rt_saxo_signal_audit_log`, dashboard gold tables). These are not part of model fitting.

**Where batch vs live meet**
- `silver_bt_master_grid_1h` is the bridge object: it is the direct historical training input and also the carry-forward reference for live RT enrichment.
- `model_registry_current` is the bridge between retraining and production inference: retraining writes it, inference reads it.

## 7. Key Notes / Caveats
- Training primarily depends on the historical batch silver table, not on raw bronze tables directly.
- The current live inference engine does **not** read `QQQ_close` as a model feature. `QQQ` matters for downstream strategy allocation and dashboard/performance context, not for the classifier inputs themselves.
- Live macro override logic is intentionally limited. Only `CPIAUCSL` and `UMCSENT` can currently override the RT silver row, and `UMCSENT` requires final validated rows.
- `CPI_Event_Active` and `NFP_Event_Active` are derived event flags from the batch silver pipeline; the RT path carries them forward from historical silver rather than rebuilding them from the live macro schedule table.
- `bronze_bt_rates_1h` is currently outside the canonical model path. It exists in the repo but is not used by `silver_bt_master_grid_1h`, live inference, or the canonical dashboard.
- The repo still contains an Alpaca paper/simulation chain (`v4_rt_hourly_chain`) with its own dashboard feed objects. That path is not the canonical live producer for the Saxo production stack.
- The institutional dashboard notebook is broader than the inference path: it reads both live state and historical reference objects for monitoring, audit, and public-macro export checks.
