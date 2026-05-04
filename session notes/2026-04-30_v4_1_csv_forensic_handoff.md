# V4.1 CSV Forensic Handoff - 2026-04-30

This note checkpoints the isolated V4.1 CSV-only forensic/research work completed during the long reconciliation session. It is a handoff note only. It does not authorize changes to production V4.

## 1. Current Git State

- Checkpoint branch created from current local state: `checkpoint/v4-1-csv-forensic-handoff-2026-04-30`
- Starting branch before checkpoint: `feat/live-macro-phase1-pce-gdp-overrides`
- Latest local commit at branch creation: `8df00da docs: record phase1 macro recovery validation`
- Recent local commits:
  - `8df00da docs: record phase1 macro recovery validation`
  - `5b1cfb6 market: merge incremental refresh into managed table`
  - `23afad7 market: add bounded incremental Alpaca refresh`
  - `b87af93 docs: record live macro semantic guard`
  - `ee75ed9 guard live macro semantic overrides`

Dirty files observed before this checkpoint branch was created:

- Modified and intentionally not treated as part of this checkpoint unless separately staged later:
  - `.gitignore`
  - `05_Context/04_Codex_Project_Memory.md`
  - `model_feature_upstream_lineage.csv`
  - `session notes/2026-04-30_session-01.md`
- Untracked files observed:
  - `04_Orchestration/02_Symphony/run_symphony_local.sh`
  - `07_Docs/02_Symphony_Local_Setup.md`
  - `07_V4_1_CSV_Baseline/README.md`
  - `07_V4_1_CSV_Baseline/code/v4_1_backtest_performance_from_csv.py`
  - `07_V4_1_CSV_Baseline/code/v4_1_build_grid_from_csv.py`
  - `07_V4_1_CSV_Baseline/code/v4_1_common.py`
  - `07_V4_1_CSV_Baseline/code/v4_1_databricks_runner.py`
  - `07_V4_1_CSV_Baseline/code/v4_1_run_all.py`
  - `07_V4_1_CSV_Baseline/code/v4_1_setup_inputs_from_sources.py`
  - `07_V4_1_CSV_Baseline/code/v4_1_train_walkforward_from_csv.py`
  - `07_V4_1_CSV_Baseline/code/v4_1_write_reports.py`
  - `WORKFLOW.md`
  - `context.md`
  - `databricks.yml`
  - `tmp_build_promoted_corrected_alfred_xlsx.mjs`
  - `tmp_build_v4_daily_eod_xlsx.mjs`
  - `tmp_codex_corrected_alfred_confirm.py`
  - `tmp_codex_corrected_alfred_promote.py`
  - `tmp_codex_prepare_promoted_corrected_alfred_pack.py`
  - `tmp_codex_v4_daily_eod_audit_pack.py`
  - `tmp_verify_promoted_corrected_alfred_xlsx.mjs`

Databricks repo/workspace state known from the session:

- Workspace repo path: `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github`
- Repo id: `2360562195327963`
- Databricks repo branch reported during the session: `feat/live-macro-phase1-pce-gdp-overrides`
- Workspace repo head reported during the session: `f8613656039839e01cdf7e8947def9c0ccc7823e`
- Local Git head and Databricks repo head were known to differ.
- Temporary isolated runner notebook created during V4.1 CSV work: `/Users/pawan.jatale@mubama.group/v4_1_csv_only_runner`
- Recovered corrected-generator workspace notebook was exported and matched recovered git source byte-for-byte.
- Production repo notebooks were not edited as part of the forensic work.

## 2. Production Safety Statement

No production V4 assets were intentionally changed during this isolated V4.1 CSV forensic work:

- No production V4 Delta tables changed.
- No Saxo execution changed.
- No Saxo token refresh changed.
- No RT notebooks changed.
- No dashboard notebooks changed.
- No model registry changed.
- No Databricks workflows, schedules, or live schedules changed.

The work was scoped to isolated CSV/research artifacts under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/`

## 3. V4.1 Artifact Folders Created

All folders below live under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/`

- `forensic_clone_macro_reference_2026-04-30`
- `forensic_clone_vix_zscore_reference_2026-04-30`
- `forensic_feature_freeze_ladder_2026-04-30`
- `forensic_model_fold_reconciliation_2026-04-30`
- `forensic_corrected_training_artifact_recovery_2026-04-30`
- `recovered_generator_replay_2026-04-30`
- `pit_safe_teacher_student_2026-04-30`
- `grid_spot_audit_2026-04-30`
- `raw_lineage_zscore_decomposition_2026-04-30`

Earlier isolated V4.1 baseline artifacts also remain under the same root, including `inputs/`, `grids/`, `signals/`, `models/`, `performance/`, `manifests/`, and `code/`.

## 4. Key Conclusions

- The non-frozen V4.1 CSV grid does not match the corrected-ALFRED reference features. Spot audit feature match was about `20.81%`; lineage/z-score comparison was about `22.3059%`.
- The full corrected-feature freeze grid matches the corrected reference features at `100.00%`.
- ALFRED/timing lateness is not the dominant issue. Macro timing drift audit found `476` same-timestamp cases, `6` early cases, and `0` late cases greater than 1 hour.
- Remaining non-frozen grid divergence is concentrated in raw lineage / forward-fill / bootstrap-history behavior for `GDPC1`, `UNRATE`, `UMCSENT`, `INDPRO`, `VIXCLS`, and `vix3m`.
- Event flags also diverge: `CPI_Event_Active` and `NFP_Event_Active` appear tied to pulse/window semantics.
- The full corrected-feature freeze still does not recreate corrected reference signals when ordinary monthly walk-forward is trained directly on raw forward-return labels.
- The exact corrected generator was recovered. It uses full-sample teacher pseudo-label distillation:
  - Train a full-sample teacher XGBoost on raw forward-return labels.
  - Write `full_signal_export.csv`.
  - Train monthly walk-forward models using `full_signal_export.csv.predicted_regime` as the label.
- That method is not strict point-in-time walk-forward. It has leakage risk and should be treated as a retrospective research/audit baseline, not strict production performance.
- PIT-safe teacher-student V4.1 removes the full-sample teacher leakage by training the teacher only on fold-local historical data, but performance is weak:
  - CAGR `5.0069%`
  - Sharpe `0.2794`
  - Max drawdown `-37.8312%`
  - Final equity `$6.380M`
- Raw-label V4.1 monthly walk-forward was around `9.9128%` CAGR.
- SPY benchmark was around `12.7658%` CAGR.
- Therefore the corrected-ALFRED `20.8405%` should not be promoted or described as strict production V4 performance.

## 5. Important Metrics

| Variant | CAGR | Sharpe | Max DD | Final Equity | Notes |
|---|---:|---:|---:|---:|---|
| Current V4 production daily-EOD | ~4.06% | n/a | n/a | n/a | Separate production stream; not a sampling-frequency issue |
| Raw-label V4.1 monthly WF | 9.9128% | 0.5821 | -39.8579% | $8.021M | CSV-only, strict raw-label monthly walk-forward |
| Full-sample teacher corrected-ALFRED | 20.8405% | 1.0024 | -25.5385% | $12.883M | Retrospective teacher pseudo-label baseline with leakage risk |
| PIT-safe teacher-student V4.1 | 5.0069% | 0.2794 | -37.8312% | $6.380M | Strict PIT teacher-student implementation |
| SPY benchmark | 12.7658% | 0.6518 | -25.1969% | $9.117M | Aligned benchmark |

Additional forensic metrics:

- Macro-reference clone:
  - Signal match improved from `68.04%` to `71.48%`.
  - CAGR `9.3160%`, Sharpe `0.5439`, Sortino `0.7227`, max DD `-37.0097%`, final equity `$7.805M`.
- VIX/z-score-reference clone:
  - Signal match `70.70%`.
  - CAGR `10.1702%`, Sharpe `0.5941`, Sortino `0.7810`, max DD `-37.1032%`, final equity `$8.115M`.
- Full feature freeze:
  - Signal match `71.45%`.
  - CAGR `10.1066%`, Sharpe `0.5912`, Sortino `0.7841`, max DD `-32.9947%`, final equity `$8.092M`.
- Recovered-generator replay:
  - Persisted `60` monthly fold models.
  - Teacher retrain match vs corrected `full_signal_export.csv`: `94.0739%`.
  - Walk-forward signal match vs corrected `walkforward_signals.csv`: `92.0706%`.
  - CAGR `24.1697%`, Sharpe `0.9689`, Sortino `1.1316`, max DD `-21.8160%`, final equity `$14.751M`.
- Raw lineage/z-score patched grid:
  - Feature match improved only to `22.6547%`.
  - Signal match `67.9217%`.
  - CAGR `1.1921%`, Sharpe `0.1299`, Sortino `0.1429`, max DD `-46.4342%`, final equity `$5.292M`.

## 6. Next Recommended Task

Run an ALFRED event-selector audit before more model experiments.

The next task should:

- Build an ALFRED candidate-row dump.
- Test selector variants against corrected reference behavior.
- Compare selected raw rows to corrected reference behavior.
- Recover or recreate the corrected raw macro and VIX level series before z-scoring.
- Only then rebuild the non-frozen V4.1 grid.

The priority is to explain raw lineage / forward-fill / bootstrap-history mismatch, not to tune XGBoost or fold definitions.

## 7. Next-Chat Prompt

Use this prompt to continue in a fresh Codex chat:

```text
You are working in the isolated V4.1 CSV-only forensic environment.

Do not mutate production V4.
Do not change production Delta tables, Saxo execution, token refresh, RT notebooks, dashboard notebooks, live schedules, workflows, or model registry.

Current checkpoint:
- The corrected-ALFRED 20.8405% result is a retrospective research baseline produced by full-sample teacher pseudo-label distillation, not strict PIT walk-forward.
- Full corrected-feature freeze matches corrected reference features 100%, but ordinary raw-label monthly walk-forward still does not reproduce corrected signals.
- The non-frozen V4.1 CSV grid does not match corrected reference features: spot-audit match about 20.81%, lineage/z-score comparison about 22.3059%.
- ALFRED lateness is not the dominant issue: macro timing drift had 476 same-timestamp, 6 early, and 0 late >1h cases.
- Remaining feature divergence appears to be raw lineage / forward-fill / bootstrap-history mismatch for GDPC1, UNRATE, UMCSENT, INDPRO, VIXCLS, and vix3m, plus event pulse semantics for CPI_Event_Active and NFP_Event_Active.
- Raw lineage/z-score patch only improved feature match to about 22.6547% and worsened performance, so simple z-score/timing patch is insufficient.

Goal:
Run an ALFRED event-selector and raw-lineage audit to recover/recreate the corrected raw macro and VIX level series before z-scoring, then rebuild the non-frozen V4.1 grid only if the raw lineage match improves.

Create output folder:
/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/alfred_event_selector_audit_2026-04-30/

Inputs:
1. V4.1 macro input:
   /Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/inputs/v4_1_macro_alfred_first_releases.csv
2. V4.1 VIX input:
   /Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/inputs/v4_1_vix_vix3m_1h_pit.csv
3. Corrected reference full signal export:
   /Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/corrected_variant/full_signal_export.csv
4. Full corrected-feature freeze grid:
   /Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/forensic_feature_freeze_ladder_2026-04-30/grids/D_full_feature_freeze/v4_1_silver_grid.csv
5. Main non-frozen V4.1 grid:
   /Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/grids/macro_market_open_or_30m_buffer__vix_next_day_10am/v4_1_silver_grid.csv
6. Raw lineage/z-score decomposition outputs:
   /Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/raw_lineage_zscore_decomposition_2026-04-30/
7. Corrected comparison folder, especially macro-selection artifacts if present:
   /Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/

Task 1 - Inventory corrected raw lineage artifacts:
- Search corrected comparison folders for macro_selection_audit.csv, raw macro level series, VIX/VIX3M raw series, first-release dumps, bootstrap/fallback rows, notebook manifests, and model/run metadata.
- Write artifact_inventory.csv with path, type, size, mtime, hash, and relevance.

Task 2 - Build ALFRED candidate-row dump:
- For GDPC1, UNRATE, UMCSENT, INDPRO, CPIAUCSL, PAYEMS, PCEPI, and PCE, dump all candidate ALFRED/FRED vintage rows for 2016-2025 release windows.
- Include realtime_start, realtime_end, observation_date, value, inferred release_date, source URL/API query metadata, and any fallback/bootstrap flags.
- Write candidate_rows/<indicator>_alfred_candidates.csv and a combined candidate dump.

Task 3 - Test selector variants:
- Implement selector variants including:
  - exact realtime_start == release_date, latest non-null observation_date
  - nearest prior realtime_start fallback
  - nearest next realtime_start fallback
  - latest observation_date per release bucket
  - observation_date constrained to release-period expectation
  - bootstrap/fallback early-history row injection
  - corrected macro_selection_audit behavior if available
- For each variant, create selected raw macro release rows and forward-filled hourly raw levels.

Task 4 - Compare selected raw lineage to corrected reference behavior:
- Compare selected release values/timestamps against macro_selection_audit.csv if available.
- If raw corrected levels are unavailable, infer/proxy corrected raw-level changes from corrected z-score changes and documented rolling semantics.
- For each indicator, report raw value mismatch count, release timestamp mismatch, availability timestamp mismatch, first divergence, and candidate selector that best matches corrected behavior.

Task 5 - VIX/VIX3M raw lineage audit:
- Search for corrected raw VIX/VIX3M sources.
- Compare V4.1 FRED VIXCLS/VXVCLS raw values, observation dates, and availability timestamps to any corrected reference source.
- If corrected raw source is unavailable, infer update/change timing from corrected VIXCLS_z and vix3m_z.
- Classify whether VIX divergence is raw source, PIT availability, bootstrap history, or z-score semantics.

Task 6 - Recovered raw series:
- Create recovered hourly raw level series for each macro indicator and VIX/VIX3M using the best selector/lineage behavior.
- Write:
  - recovered_raw_levels/macro_hourly_raw_levels.csv
  - recovered_raw_levels/vix_hourly_raw_levels.csv
  - recovered_raw_levels/recovered_raw_lineage_manifest.json

Task 7 - Rebuild non-frozen grid from recovered raw levels:
- Recompute features from recovered raw levels without freezing corrected reference feature columns directly.
- Use recovered semantics from the decomposition audit:
  - macro z-score best candidate: 8761-hour rolling, current row included, sample std
  - VIX/VIX3M z-score best candidate: 8760-hour rolling, current row included, sample std unless raw audit finds otherwise
  - event flags best candidate: hourly level delta, 49-hour pulse, current row excluded
- Write grids/v4_1_silver_grid_recovered_raw_lineage.csv and validation JSON.

Task 8 - Compare and decide:
- Compare recovered grid features to corrected reference.
- Report feature match %, MAE by feature, first divergences, and whether raw lineage match materially improved from about 22%.
- Do not retrain unless feature match materially improves.
- If feature match remains low, stop and report the missing raw artifact or selector ambiguity.

Outputs:
- artifact_inventory.csv
- candidate_rows/*.csv
- selector_variant_matrix.csv
- raw_lineage_comparison.csv
- vix_raw_lineage_audit.csv
- recovered_raw_levels/*.csv
- grids/v4_1_silver_grid_recovered_raw_lineage.csv if justified
- forensics/alfred_event_selector_audit_summary.md
- forensics/alfred_event_selector_audit_summary.json

Return:
1. Output folder path.
2. Whether corrected raw lineage artifacts were found.
3. Best selector per macro indicator.
4. VIX/VIX3M raw lineage diagnosis.
5. Recovered grid feature match if rebuilt.
6. Remaining unresolved raw lineage gaps.
7. Next smallest action.
8. Explicit no-production-change confirmation.
```

## 8. Updates Since Original Checkpoint - 2026-05-04

This section records the completed follow-on forensic work after the original 2026-04-30 checkpoint. It supersedes the "next recommended task" above, which was completed and then followed by additional reconciliation tasks. The production safety scope remained unchanged.

### 8.1 Raw Macro / Bootstrap / Z-Score Findings

- Raw macro lineage mismatch was ruled out for comparable V4 vs V4.1 rows.
  - Comparable raw macro release rows matched exactly: `1,000`.
  - Absolute value difference: `0.0`.
  - Timestamp difference: `0.0` hours.
  - Only raw macro difference found: eight V4.1 bootstrap rows dated `2016-01-01`, one per indicator, missing from production raw macro.
- Bootstrap-row impact was small.
  - Removing the eight bootstrap rows improved feature match only from about `21.4592%` to `22.4586%`.
  - Bootstrap rows do not explain the main mismatch.
- Z-score construction variants also did not explain the feature mismatch.
  - Tested window lengths, endpoint inclusion/exclusion, sample vs population std, min periods, bootstrap behavior, and backbone variants.
  - Z-score math changes alone produced negligible improvement.
- Conclusion at this stage:
  - Raw macro release-row selection and simple z-score math were not the dominant issue.
  - Remaining issue was hidden legacy source/logic/cached-artifact lineage.

Key folders:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/alfred_event_selector_audit_2026-04-30/`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/bootstrap_row_impact_audit_2026-04-30/`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/zscore_construction_audit_2026-04-30/`

### 8.2 Recovered Code Lineage Diagnosis

- The recovered corrected generator was audited line-by-line.
- Decisive finding:
  - The replay runner loaded a cached full-feature-freeze grid CSV and did not rebuild corrected silver from raw data.
  - The recovered generator used legacy/default data paths and legacy grid semantics:
    - `first_releases_<indicator>.csv`
    - macro join clock of `timestamp + 2h`
    - 24/7 hourly backbone
    - 24/7 forward-fill
    - default-schema market/VIX Delta tables
- VIX/VIX3M source was identified exactly:
  - `pawan_quant_research_1.default.bronze_bt_vix_1h`
  - `pawan_quant_research_1.default.bronze_bt_vix3m_1h`
- V4.1 had used newer FRED daily CSV/PIT expansion for VIX/VIX3M, which explained the VIX raw/source mismatch.

Key folder:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/recovered_code_lineage_audit_2026-04-30/`

### 8.3 Legacy Default VIX Source Rebuild

- Default VIX/VIX3M tables were exported read-only to isolated CSVs:
  - `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/legacy_default_vix_source_rebuild_2026-04-30/inputs/default_bronze_bt_vix_1h.csv`
  - `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/legacy_default_vix_source_rebuild_2026-04-30/inputs/default_bronze_bt_vix3m_1h.csv`
- Rebuilding only VIX/VIX3M columns with recovered legacy logic fully matched corrected reference VIX features:
  - `VIXCLS_z`: `100.0%`
  - `vix3m_z`: `100.0%`
  - `VIX_Inversion_Z`: `100.0%`
- Overall 14-feature match improved:
  - original non-frozen V4.1: `22.3397%`
  - non-bootstrap V4.1: `22.4586%`
  - legacy-default-VIX patched grid: `43.9411%`
- Verdict:
  - Legacy default VIX source fully explained the VIX/VIX3M feature mismatch.

Key folder:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/legacy_default_vix_source_rebuild_2026-04-30/`

### 8.4 Full Faithful Legacy Grid Clone

- A full faithful legacy-grid clone was built from isolated CSVs using:
  - default market source
  - default VIX/VIX3M source
  - original `first_releases_<indicator>.csv` files
  - recovered `build_legacy_silver()` semantics exactly
- Faithful grid:
  - `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/grids/full_legacy_grid_clone.csv`
- Result against corrected reference/full-feature-freeze grid:
  - Overall 14-feature match: `100.000000%`
  - All 14 features matched at `100%`
  - SPY/QQQ close matched `99.998889%`
  - Candidate grid rows: `89,991`
  - Reference rows: `90,544`
  - Inner matched rows: `89,991`
  - Reference-only timestamps: `553`, because legacy default market source ended at `2026-04-07 14:00:00`
- Verdict:
  - Corrected reference features were faithfully reproducible from recovered legacy inputs and recovered legacy grid semantics.

Key folder:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/`

### 8.5 Faithful Legacy Grid Model Replay

- The recovered corrected-generator model replay was run on the 100%-feature-matched faithful grid.
- Results:
  - Teacher `full_signal_export` match vs corrected reference: `100.000000%`
  - Walk-forward signals match vs corrected reference: `100.000000%`
  - Walk-forward mismatches: `0`
  - All 14 features on prediction timestamps: `100.000000%`
- Replay performance from the first replay method:
  - CAGR: `19.281820%`
  - Sharpe: `0.794532`
  - Sortino: `0.922043`
  - Max DD: `-26.983086%`
  - Final equity: `$12,067,688.21`
- Corrected reference performance target:
  - CAGR: `20.8405%`
  - Sharpe: `1.0024`
  - Sortino: `1.3729`
  - Max DD: `-25.5385%`
  - Final equity: `$12.883M`
- Conclusion:
  - Grid, features, teacher predictions, and walk-forward signals were fully reproduced.
  - Remaining mismatch was performance methodology / price-path treatment, not model or feature generation.

Key folder:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/faithful_legacy_grid_model_replay_2026-04-30/`

### 8.6 Performance Methodology Reconciliation

- The corrected reference `20.8405%` performance was reconciled from the exact matched signal stream.
- Decisive result:
  - The reference performance was reproducible when using the corrected reference price/equity methodology rather than the first replay method.
  - The mismatch was caused by allocation timing / price-path / daily resampling methodology, not signal mismatch.
- Reconciled conclusion:
  - Corrected `20.8405%` result can be regenerated from matched signals with the reference methodology.
  - It remains a retrospective teacher pseudo-label baseline with leakage risk because the signal stream depends on full-sample teacher pseudo-labels.
  - It should be used as forensic/reference baseline, not strict PIT production baseline.

Key folder:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/performance_methodology_reconciliation_2026-04-30/`

### 8.7 FRED VIX/VIX3M Exact Recreation

- The V4 FRED `fred_1d` VIX ingestion logic was recreated and compared row-by-row to legacy default VIX/VIX3M exports.
- Logic tested:
  - FRED daily `VIXCLS` / `VXVCLS`
  - `close.shift(1)`
  - expand each date to 24 hourly rows
  - timestamp = normalized date + hour
  - `VIXCLS -> VX`
  - `VXVCLS -> VIX3M`
  - source fields matching V4 notebook semantics
- Result:
  - Current FRED V4 logic matched legacy default VIX/VIX3M exactly on overlapping timestamps.
  - Not file-identical only because current FRED had additional later rows beyond the legacy default snapshot end.
  - Grid feature match with recreated FRED VIX was about `99.767%`.
  - Model replay was not run because feature match was below the optional replay threshold.
- Final VIX answer:
  - Legacy default VIX is reproducible from current FRED using the exact V4 logic for the overlapping period.
  - Prior V4.1 mismatch was due to implementation/source-path difference, not unresolved FRED revision mystery.

Key folder:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/vix_vix3m_exact_recreation_2026-04-30/`

### 8.8 V4.2 Long-History Data Availability Audit

- A read-only V4.2 data availability audit was run for start-date candidates:
  - `1996-01-01`
  - `2006-01-01`
  - `2010-01-01`
  - `2016-01-01`
- Existing V4/default hourly market data:
  - SPY hourly first timestamp: `2016-01-01 00:00:00`
  - QQQ hourly first timestamp: `2016-01-01 00:00:00`
  - No inspected V4/default table or targeted CSV artifact contained SPY/QQQ before 2016.
- Current FRED volatility availability under V4 `shift(1)` logic:
  - VIXCLS usable from `1990-01-03`
  - VXVCLS/VIX3M usable from `2007-12-05`
- Macro ALFRED feasibility:
  - Macro first-release reconstruction is feasible for `2006`, `2010`, and `2016`.
  - `1996` is blocked mainly by PCEPI/UMCSENT vintage availability.
  - UMCSENT has preliminary/final release-stage ambiguity and needs explicit final-only semantics.
- Feasibility summary:
  - Best clean hourly V4.2 start from existing artifacts: `2016-01-01`
  - Best practical longer-history daily V4.2 start with new external daily SPY/QQQ feed: `2010-01-01`
  - `1996` blocked by SPY/QQQ, VIX3M, and macro first-release history.
  - `2006` blocked by SPY/QQQ and VIX3M.
  - `2010` feasible only as daily with new market data, not hourly from current artifacts.

Key folder:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_data_availability_audit_2026-04-30/`

### 8.9 Massive SPY/QQQ Hourly Data Audit

- Massive API key was loaded into Databricks secret:
  - scope: `massive`
  - key: `api_key`
- The key itself is intentionally not written in this note.
- The Massive hourly SPY/QQQ audit was run under:
  - `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/`
- Requested range:
  - `2007-12-05` to `2026-04-30`
- Key finding:
  - The current Massive entitlement does not support the requested long-history timeframe.
  - Massive returned `NOT_AUTHORIZED` for historical chunks including `2007`, `2008`, `2010`, and `2016`.
  - Error class/message recorded in `forensics/massive_fetch_chunk_status.csv`.
- Accessible partial data only:
  - SPY: `4,004` rows, `2024-05-01 08:00:00` to `2026-04-30 23:00:00`
  - QQQ: `6,302` rows, `2024-05-01 08:00:00` to `2025-11-26 10:00:00`
  - This is not complete enough for any requested V4.2 long-history hourly baseline.
- Overlap vs V4 where Massive data exists:
  - Best session filter: `all_sessions`
  - Best alignment: `same_timestamp`
  - No one-hour shift needed.
  - QQQ same-timestamp MAE: `0.0001206`, near-match rate `99.9365%`
  - SPY same-timestamp MAE: `0.0004825`, near-match rate `99.6024%`
- Recommendation:
  - Massive with the current entitlement cannot support `2007-12-05`, `2008-01-01`, `2010-01-01`, or `2016-01-01` as an hourly start.
  - To pursue V4.2 hourly long history, upgrade Massive historical access or use a different intraday vendor/backfill.
  - If a deeper Massive plan is obtained later, use `all_sessions` and `same_timestamp` alignment as the first candidate because it best matches V4 where overlap exists.

Key outputs:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/inputs/raw/SPY_hourly_massive_2007_2026.csv`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/inputs/raw/QQQ_hourly_massive_2007_2026.csv`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/inputs/v4_2_market_spy_qqq_1h_massive_2007_2026.csv`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/forensics/massive_fetch_chunk_status.csv`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/forensics/massive_vs_v4_market_overlap_comparison.csv`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/manifests/massive_market_feed_manifest.json`

## 9. Updated Current State

- The original V4.1 corrected-reference mystery is now explained:
  - Corrected reference features and signals are reproducible exactly from recovered legacy sources and recovered legacy generator semantics.
  - The original V4.1 CSV rebuild diverged because it used newer/different VIX source handling and did not exactly match the legacy/default market, VIX, macro first-release, join-clock, ffill, event, and cached-artifact lineage.
- The corrected `20.8405%` result is reproducible as a forensic/reference result, but it is still not a strict PIT production baseline because it depends on full-sample teacher pseudo-label flow.
- Production V4 remained untouched throughout these forensic/research runs.
- For strict V4.2:
  - Clean hourly baseline from existing data remains `2016-01-01`.
  - Longer daily research baseline around `2010-01-01` is plausible if a clean daily SPY/QQQ feed is added.
  - Longer hourly baseline around `2007/2008` requires a new/expanded intraday market feed; the tested Massive entitlement is insufficient.

## 10. Updated Next Recommended Task

Do not train models yet.

Recommended next step depends on intended V4.2 direction:

- If staying hourly:
  - Obtain/approve a market-data source with SPY/QQQ hourly history back to at least `2007-12-05` or `2008-01-01`.
  - Re-run the Massive audit or equivalent vendor audit.
  - Require coverage, duplicate/null/gap checks, and overlap alignment against V4 before any grid build.
- If moving to daily long-history V4.2:
  - Source daily SPY/QQQ data back to at least `2010-01-01`.
  - Build a daily strict-PIT feature grid design that uses VIXCLS/VXVCLS and ALFRED macro availability semantics.
  - Keep it explicitly separate from the legacy V4.1 hourly/reference baseline.
- If documenting corrected V4.1:
  - Treat the faithful legacy clone as a forensic reference baseline.
  - Label the `20.8405%` result as retrospective teacher pseudo-label/reference methodology, not strict production performance.

## 11. Updated Production Safety Statement

No production V4 assets were intentionally changed by the follow-on forensic/research work:

- No production V4 Delta tables were overwritten.
- No model training was promoted to production.
- No Saxo execution changed.
- No Saxo token refresh changed.
- No RT notebooks changed.
- No dashboard notebooks changed.
- No model registry changed.
- No Databricks production workflows, schedules, or live schedules changed.

One non-production secret-management action was performed for research API access:

- Databricks secret scope/key `massive/api_key` was created/updated so the Massive API key would not be hardcoded into scripts, manifests, logs, or this handoff note.

All forensic outputs were written under isolated research folders:

- `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_data_availability_audit_2026-04-30/`
- `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026/`
