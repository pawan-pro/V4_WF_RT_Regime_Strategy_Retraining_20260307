# 2026-05-08 Teacher-Student Promotion and Macro Audit

## Teacher-Student Promotion

- Promoted model version: `bt_teacher_student_smoothed_v4_2026_05_08`
- Candidate artifact path: `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/faithful_legacy_grid_model_replay_2026-04-30/models/folds/2025_12/model.joblib`
- Candidate SHA256: `7dd1949058c4af0e6525dff1daea095603e108cba2b0dc9fd5a4756968b44f6b`
- Versioned model path: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_teacher_student_smoothed_v4_2026_05_08.joblib`
- Live alias path: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- Backup alias path: `/Volumes/pawan_quant_research_1/v4/market_data/models/backups/bt_fred_1d_xgboost_regime_live_pre_bt_teacher_student_smoothed_v4_2026_05_08_20260508T121613Z.joblib`
- Candidate signal path: `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_b_vs_c_lineage_diff_2026-05-04/signals/recreated_C5_cached_models_if_available.csv`
- C2 signal reference path: `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_b_vs_c_lineage_diff_2026-05-04/signals/recreated_C2_recovered_preprocessing.csv`
- Candidate manifest path: `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/faithful_legacy_grid_model_replay_2026-04-30/manifests/fold_manifest.csv`

## Registry Before / After

Before:

- `model_version`: `bt_fred_1d_xgboost_regime_2026_05_01`
- `model_path`: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- `versioned_model_path`: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_05_01.joblib`
- `promoted_ts`: `2026-05-02 00:32:22.396833`
- `signal_path`: `/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv`
- `wf_signal_table`: `pawan_quant_research_1.v4.bt_walkforward_regime_signals`

After:

- `model_version`: `bt_teacher_student_smoothed_v4_2026_05_08`
- `model_path`: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- `versioned_model_path`: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_teacher_student_smoothed_v4_2026_05_08.joblib`
- `promoted_ts`: `2026-05-08 12:16:31.290194`
- `train_start`: `2016-01-01`
- `train_end`: `2025-12-01 00:00:00`
- `signal_path`: `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_b_vs_c_lineage_diff_2026-05-04/signals/recreated_C5_cached_models_if_available.csv`
- `wf_signal_table`: `null`

## Validation Artifacts

- Shadow table: `pawan_quant_research_1.v4.bronze_rt_regime_signal_teacher_student_shadow`
- Validation table: `pawan_quant_research_1.v4.model_promotion_validation_teacher_student_2026_05_08`
- Shadow validation source: `silver_rt_1h`
- Shadow rows: `100`
- Shadow timestamp range: `2026-04-21 16:00:00` through `2026-05-07 20:00:00`
- Latest-100 feature null counts: all 14 production model features were `0`
- Latest-100 candidate distribution: `candidate_regime=2` for `100/100` rows
- Latest-100 previous live distribution: `regime 0: 60`, `regime 1: 19`, `regime 2: 21`

## Manual Runs

- Promotion one-time run: `617587535161840`, task run `1098660057841823`, success
- Inference-only validation run: `574605272938530`, task run `143341971385553`, success
- Full manual `v4_saxo_live_hourly_chain` run: `410988321035754`, success
- Full chain task run ids:
  - `rt_01_market`: `960833506438719`
  - `rt_01b_macro_events`: `729902420797704`
  - `rt_02_vix`: `283687045290107`
  - `rt_03_silver`: `680318312525541`
  - `rt_04_inference`: `1061310681902188`
  - `rt_08_saxo_execution`: `846490729507178`
  - `rt_09_saxo_audit`: `702276484618420`
  - `rt_10_saxo_dashboard`: `688470650797889`

## Latest Post-Promotion Live State

Latest live signal after manual full Saxo chain:

- `timestamp`: `2026-05-08 12:00:00`
- `regime_int`: `2`
- `regime_name`: `Recession`
- `position`: `100% CASH`
- `model_version`: `bt_teacher_student_smoothed_v4_2026_05_08`
- `model_path`: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`

Latest Saxo audit row:

- `audit_ts`: `2026-05-08 12:19:49.160148`
- `signal_ts`: `2026-05-08 12:00:00`
- `regime_int`: `2`
- `regime_name`: `Recession`
- `position`: `100% CASH`
- `model_version`: `bt_teacher_student_smoothed_v4_2026_05_08`
- `execution_mode`: `live`

## Rollback Path

To roll back the model promotion, copy the backup alias back over the live alias and restore `pawan_quant_research_1.v4.model_registry_current` to the previous row:

- Source backup: `/Volumes/pawan_quant_research_1/v4/market_data/models/backups/bt_fred_1d_xgboost_regime_live_pre_bt_teacher_student_smoothed_v4_2026_05_08_20260508T121613Z.joblib`
- Destination live alias: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- Previous registry model version: `bt_fred_1d_xgboost_regime_2026_05_01`
- Previous versioned model path: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_05_01.joblib`

Run inference-only validation before any downstream execution after rollback.

## NFP Patch State

The PAYEMS/NFP patch was not committed, merged, deployed, or imported to Databricks.

- Local branch only: `fix/live-payems-nfp-immediate-alfred-2026-05-08`
- Local worktree only: `/Users/pawan/Mubama/Github/V4_WF_RT_Regime_Strategy_Retraining_20260307_nfp_patch`
- Files modified only in that worktree:
  - `01_Processing/03_Realtime/02_RT_Live_Macro_Events.py`
  - `01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py`

Production PAYEMS/NFP remains unchanged until a separate review/deploy step.
