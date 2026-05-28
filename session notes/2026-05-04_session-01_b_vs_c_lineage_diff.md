# 2026-05-04 Session 01 - B vs C Lineage Difference

## Scope

Research-only lineage audit for why Variant C newly trained full-sample teacher underperformed the B recovered control on the same faithful legacy master grid.

No production Delta tables, Saxo execution, token refresh, realtime notebooks, dashboard notebooks, workflows, schedules, or model registry assets were modified.

## Inputs

- Faithful legacy grid: `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/grids/full_legacy_grid_clone.csv`
- Expected grid SHA256: `b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1`
- B recovered replay: `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/faithful_legacy_grid_model_replay_2026-04-30/`
- C package: `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_c_full_sample_teacher_package_2026-05-04/`
- Recovered reference signals:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/corrected_variant/full_signal_export.csv`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/corrected_variant/walkforward_signals.csv`

## Output

- Root: `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_b_vs_c_lineage_diff_2026-05-04/`
- Script: `03_Research/02_Models/05_V4_2_B_vs_C_Lineage_Diff.py`
- Databricks run id: `653294628821948`
- Task run id: `77175825505828`

## Reproducibility Ladder

| Variant | Signal Match vs B | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switches | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C0 current | 71.4563% | 14.6138% | 0.8034 | 1.0186 | -24.1275% | $9.889M | 906 | Current C package, fresh 500/depth6 teacher and fresh monthly students. |
| C1 recovered params | 92.3877% | 23.6492% | 1.1197 | 1.5146 | -21.8160% | $14.452M | 276 | Recovered 200/depth4 teacher and recovered student params, but current row filtering. |
| C2 recovered preprocessing | 100.0000% | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12.883M | 296 | Fresh retraining with recovered params plus recovered fillna/merge/schedule semantics. |
| C3 recovered teacher labels | 100.0000% | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12.883M | 296 | Recovered full_signal_export labels, fresh monthly students. |
| C4 recovered walk-forward signals | 100.0000% | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12.883M | 296 | Direct recovered signal replay. |
| C5 cached models | 100.0000% | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12.883M | 296 | Recovered fold model binaries loaded and rerun on faithful grid. |

## Diagnosis

C current diverged from B because it changed the recovered full-sample teacher configuration and preprocessing path.

The recovered code path is independently reproducible from the faithful legacy grid: C2 used fresh training, did not use recovered teacher labels or recovered walk-forward signals, and still matched B at 100%.

Minimum reproduction path:

- Use recovered teacher params: `n_estimators=200`, `max_depth=4`, `learning_rate=0.03`, `objective=multi:softmax`, `num_class=3`, `tree_method=hist`, `random_state=42`.
- Use recovered student params: `n_estimators=500`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.9`, `colsample_bytree=0.9`, `objective=multi:softmax`, `num_class=3`, `tree_method=hist`, `random_state=42`.
- Use recovered teacher preprocessing: train on rows with non-null forward label, predict across the full feature frame with `FEATURES.fillna(0)`.
- Use recovered walk-forward merge/schedule semantics: labels joined by timestamp, monthly folds with train rows before fold start and test rows in the fold month.

## Key Artifacts

- `forensics/artifact_inventory.csv`
- `forensics/input_grid_validation.json`
- `forensics/teacher_logic_diff.md`
- `forensics/walkforward_logic_diff.md`
- `forensics/reproducibility_ladder.csv`
- `performance/recreated_variant_summary.csv`
- `forensics/b_vs_c_lineage_diff_summary.md`
- `forensics/b_vs_c_lineage_diff_summary.json`

## Validation

- `python3 -m py_compile 03_Research/02_Models/05_V4_2_B_vs_C_Lineage_Diff.py` passed.
- Safety scan found no production write APIs; matches were only no-write/safety text.
- All generated outputs are under `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_b_vs_c_lineage_diff_2026-05-04/`.
- Production live assets were not changed.

## Governance

B/C2 are reproducible from code and faithful legacy data, but both remain full-sample/non-PIT forensic research baselines. They are not strict PIT production candidates. The next research step remains an independent strict rolling model path, such as TFT, compared against B, C, D, SPY, and production V4 read-only reference.
