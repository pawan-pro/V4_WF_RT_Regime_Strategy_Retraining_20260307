# 2026-05-04 Session 01 - PIT Teacher-Student Research Plan

## Request

Build a research-only strict point-in-time teacher-student experiment for V4.2 using the existing V4 historical silver grid as input. The goal is to convert the recovered full-sample teacher pseudo-label idea into a production-candidate research path without modifying production V4 assets.

## Branch

Created branch:

`research/v4-2-pit-teacher-student-2026-05-04`

Initial status before branch/change showed pre-existing dirty files:

- `.gitignore`
- `05_Context/04_Codex_Project_Memory.md`
- `model_feature_upstream_lineage.csv`
- `session notes/2026-04-30_session-01.md`

Those files were not part of this task.

## Files Added

- `03_Research/02_Models/01_V4_2_PIT_Teacher_Student.py`
- `03_Research/02_Models/02_V4_2_PIT_Teacher_Student_Finalize.py`
- `03_Research/02_Models/README_V4_2_PIT_Teacher_Student.md`
- `session notes/2026-05-04_session-01_pit_teacher_student_research_plan.md`

## Research Output Root

All experiment writes are constrained to:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/`

Expected outputs:

- `signals/`
- `models/`
- `manifests/`
- `performance/`
- `forensics/`

## Read-Only Inputs

Primary input:

`pawan_quant_research_1.v4.silver_bt_master_grid_1h`

Optional read-only comparison:

`pawan_quant_research_1.v4.bt_walkforward_regime_signals`

## Variants Implemented

| Variant | Purpose | PIT status |
|---|---|---|
| A_raw_label_monthly_wf | Raw forward-return label monthly walk-forward baseline | Strict PIT |
| B_retrospective_full_sample_teacher_reference | Recovered-style full-sample teacher pseudo-label reference | Not PIT; comparison only |
| C_pit_safe_teacher_student | Fold-local teacher trained only before fold start, then student trained on fold-safe pseudo-labels | Strict PIT |
| D_conservative_hybrid | Full allocation only when A and C agree; otherwise more defensive regime at 50% allocation | Strict PIT derived signal |

## Performance Methodology

Implemented methodology requested for comparison:

- prior-bar / next-bar allocation
- price-update hourly rows only
- 5 bps switch cost
- daily-last equity reporting
- CAGR from `2021-01-01` to `2025-12-31` using exactly 5.0 years
- Sharpe/Sortino on daily returns with 252 annualization and risk-free 0

## Fold Manifest Requirements

For model-producing variants, the script persists per-fold:

- fold start/end
- teacher train start/end when applicable
- student train start/end
- train/test row counts
- teacher label distribution
- student label distribution
- student prediction distribution
- feature list
- model parameters
- input data hash
- output signal hash
- model artifact paths and hashes

For the hybrid variant, the fold manifest records the deterministic hybrid rule, agreement rate, input hash, and output signal hash.

## Scaling Diagnostics

The script computes:

- turnover / switch count
- average and median holding period
- max drawdown
- worst 1-day / 5-day / 20-day return
- crisis-window returns and drawdowns
- slippage sensitivity at 5 bps, 10 bps, and 25 bps
- notional readiness notes for `$100K`, `$1M`, and `$100M`

## Governance

Verdicts supported:

- `research_only`
- `shadow_candidate`
- `eligible_for_100k`
- `not_eligible`

The retrospective full-sample teacher reference is forced to `research_only` because it is not strict PIT.

## Production Safety

The script does not:

- call `saveAsTable`
- write Delta tables
- write the model registry
- overwrite `bt_walkforward_regime_signals`
- overwrite `bronze_rt_regime_signal`
- overwrite `silver_rt_1h`
- touch Saxo tables
- touch dashboard feed tables
- change workflows or schedules

This task created research code and documentation only. It did not run a production promotion.

## Execution Notes

Databricks one-off research runs used existing cluster:

`0311-005159-pm2x3pce`

Observed run sequence:

- Initial full experiment run wrote the fold models but failed late in diagnostics because `position_changed` was not persisted into the hourly equity frame.
- Script was patched to persist `position_changed` and to reuse existing research fold models when rerun.
- Resume run wrote signals/performance and failed at the Excel audit pack step because `openpyxl` was unavailable.
- Dependency handling was patched to install `openpyxl` explicitly.
- A duplicate long-running resume run was canceled after the required CSV outputs already existed.
- Finalizer run `1034898786598747` completed successfully and wrote `runtime_manifest.json`, `audit_pack.xlsx`, and `pit_teacher_student_research_summary.md`.
- Manifest refresh finalizer run `932706087086157` completed successfully after syncing the final patched scripts into the isolated `code/` folder.

The finalizer does not train models. It only reads existing research CSV outputs and writes final summary/manifest artifacts under the isolated output root.

## Final Outputs

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/signals/`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/models/`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/manifests/fold_manifest.csv`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/manifests/fold_manifest.json`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/manifests/runtime_manifest.json`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/performance/variant_summary.csv`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/performance/audit_pack.xlsx`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/forensics/pit_teacher_student_research_summary.md`

## Final Variant Summary

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switch Count | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A_raw_label_monthly_wf | 6.8925% | 0.4733 | 0.5834 | -38.4318% | $6,977,589.08 | 1058 | shadow_candidate |
| B_retrospective_full_sample_teacher_reference | 6.0667% | 0.4290 | 0.5483 | -40.1091% | $6,712,199.89 | 904 | research_only |
| C_pit_safe_teacher_student | 6.4963% | 0.4542 | 0.5835 | -38.0591% | $6,849,254.53 | 924 | shadow_candidate |
| D_conservative_hybrid | 5.3296% | 0.4009 | 0.4962 | -36.1099% | $6,482,197.35 | 1259 | research_only |
| production_v4_read_only_reference | 5.9886% | 0.4141 | 0.5146 | -39.7854% | $6,687,537.12 | 1002 | read_only_reference |
| SPY_benchmark_same_price_update_rows | 12.7658% | 0.8431 | 1.1842 | -25.2094% | $9,117,097.14 | 0 | benchmark |

## Recommendation

The strict PIT-safe teacher-student path is a research `shadow_candidate`, not ready for promotion. It slightly outperformed the current production V4 read-only reference in this test but materially underperformed SPY over the requested 2021-2025 window. The smallest next step is to inspect why both strict PIT model variants lag SPY before any shadow deployment discussion.

## Same-Data Correction

The first V4.2 teacher-student comparison was not apples-to-apples against the recovered 20.840487% forensic reference because it used current production silver:

`pawan_quant_research_1.v4.silver_bt_master_grid_1h`

A corrected same-data run was created with script:

`03_Research/02_Models/03_V4_2_Same_Data_Teacher_Student.py`

Output root:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_same_data_teacher_student_2026-05-04/`

Input grid:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/grids/full_legacy_grid_clone.csv`

Validation:

- Faithful grid rows: `89,991`
- Faithful grid timestamp range: `2016-01-01 00:00:00` to `2026-04-07 14:00:00`
- Canonical signal rows: `43,824`
- All variants use the same timestamp hash.
- B recovered replay control matches the recovered reference signals at `100.0%`.
- B recovered replay control reproduces the 20.840487% reference exactly on CAGR/final equity/max drawdown and matches Sharpe/Sortino after aligning switch-cost and daily grouping semantics.

Final same-data summary:

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Signal Match vs 20.84 Ref |
|---|---:|---:|---:|---:|---:|---:|
| A_raw_label_monthly_wf_faithful_grid | 5.4884% | 0.3669 | 0.4735 | -40.9836% | $6,531,196.89 | 69.8544% |
| B_recovered_full_sample_teacher_replay_control | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12,883,454.99 | 100.0000% |
| C_new_full_sample_teacher_faithful_grid | 14.6138% | 0.8034 | 1.0186 | -24.1275% | $9,889,062.68 | 71.4563% |
| D_pit_teacher_student_faithful_grid | 8.4885% | 0.5121 | 0.6663 | -26.2756% | $7,514,283.85 | 69.9388% |
| E_conservative_hybrid_raw_and_pit_agreement | 5.8086% | 0.3917 | 0.5025 | -28.7781% | $6,630,926.02 | 66.4841% |
| F_SPY_benchmark_same_price_rows | 12.7546% | 0.7703 | 1.0480 | -25.1969% | $9,112,563.90 | 2.1929% |

Answer to the key question:

- The prior `6.0667%` B result was mainly caused by using current V4 silver rather than the faithful legacy grid/control lineage.
- On the exact faithful legacy dataset, the recovered control returns to `20.840487%`.
- The strict PIT teacher-student method still materially degrades on the same data: `8.4885%` CAGR versus `20.8405%` for the recovered non-PIT control and `12.7546%` for SPY.
- The newly trained full-sample teacher on the faithful grid reaches `14.6138%`, which indicates some of the recovered reference edge is in the non-PIT full-sample/cached lineage, but the strict PIT distillation does not preserve it.
