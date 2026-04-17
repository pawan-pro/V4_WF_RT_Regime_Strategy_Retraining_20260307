# 2026-04-17 Session 01 — Dashboard Validation and Macro Lineage Fix

## 1. Objective

- Validate institutional Saxo dashboard lineage from displayed feature values back to authoritative macro source data.
- Ensure macro features used by the dashboard and model inputs match authoritative ALFRED release-bucket data.
- Identify and resolve any discrepancies without changing live workflow topology, RT ownership, or production execution behavior.
- Confirm post-fix model correctness, dashboard stability, and production safety before any promotion decision.

## 2. Initial State

- Institutional dashboard proof initially showed:
  - Total features checked: `14`
  - Exact matches: `12`
  - Close matches: `0`
  - Mismatches: `2`
- The two mismatches were:
  - `PAYEMS` (Payrolls)
  - `INDPRO` (Industrial Production)
- Read-only ALFRED comparison inside the dashboard showed the discrepancy was not a dashboard math issue; it was an upstream macro-source selection issue inside release buckets.
- Initial working hypothesis was that the ALFRED selection logic was choosing the wrong row when multiple observations existed for the same release timestamp.

## 3. Root Cause

- The upstream ALFRED ingestion logic in `01_Processing/01_Ingestion/03_BT_Macro_Incremental_Update.py` filtered rows by release bucket using `realtime_start`, then selected the first remaining row with `iloc[0]`.
- This was unsafe because ALFRED can return multiple rows for the same release timestamp with different `observation_date` values.
- `iloc[0]` did not guarantee selection of the latest valid observation inside the release bucket.
- As a result, an older observation could be selected and propagated downstream.
- The incorrect values flowed through:
  - `pawan_quant_research_1.v4.bronze_bt_macro_events`
  - `pawan_quant_research_1.v4.silver_bt_master_grid_1h`
  - `pawan_quant_research_1.v4.silver_rt_1h`
  - model input features
  - institutional dashboard validation outputs

## 4. Fix Implemented

### 4.1 Logic Fix

- The ALFRED selection helper was corrected on branch `feat/macro-alfred-latest-observation-selection-fix-2026-04-16-s2`.
- Commit `9e9db8c` updated the logic to select the row with the maximum `observation_date` within the matched release bucket.
- This preserved the existing function interface and downstream shape while correcting the selection semantics.

### 4.2 Data Repair

- A normal rerun was not sufficient because historical bronze rows for already-seen release timestamps were not being rewritten by incremental logic.
- A targeted bronze repair was therefore applied on branch `feat/macro-targeted-bronze-backfill-payems-indpro-2026-04-17`.
- Scope was intentionally limited to the two proven stale rows in `pawan_quant_research_1.v4.bronze_bt_macro_events`:
  - `PAYEMS` at `2026-04-03 12:30:00Z`
  - `INDPRO` at `2026-03-16 13:15:00Z`
- Corrected values applied:
  - `PAYEMS`: `158592` -> `158637`
  - `INDPRO`: `101.668` -> `102.551`
- The repair was surgical:
  - exact existing row updated
  - schema preserved
  - release timestamp preserved
  - `ingestion_ts` updated for auditability
  - no broad historical overwrite performed

### 4.3 Downstream Rebuild

- After bronze repair, downstream refresh steps were rerun:
  - `01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py`
  - `01_Processing/02_Features/03_RT_Silver_RT_Enrichment.py`
- Result:
  - corrected bronze values propagated into downstream silver outputs
  - `silver_bt_master_grid_1h` and `silver_rt_1h` realigned with authoritative macro values

## 5. Dashboard Fix

- After macro lineage repair, the final dashboard validation run was blocked by notebook runtime issues in `02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb`.
- A minimal notebook-only fix was applied on branch `feat/dashboard-fix-as-text-runtime-2026-04-17`.
- Runtime issues corrected:
  - missing `_as_text` helper
  - missing `RAW_FEATURE_SPECS` mapping
- The fix restored end-to-end notebook execution without changing dashboard logic, ingestion, silver, RT, workflow JSON, or jobs.

## 6. Final Validation

- The institutional dashboard proof was restored to:
  - Total features: `14`
  - Exact matches: `12`
  - Close matches: `0`
  - Mismatches: `0`
  - Derived / unresolved: `2`
- Remaining derived / unresolved items:
  - `CPI_Event_Active`
  - `NFP_Event_Active`
- ALFRED diagnostic section `4c` showed:
  - `INDPRO`: latest observation selected
  - `PAYEMS`: ambiguous duplicate values, but selected value consistent with authoritative data
  - `UNRATE`: ambiguous duplicate values
  - `GDPC1`: latest observation date selected, but value mismatch remains
- Conclusion:
  - macro lineage issue for the validated mismatches was resolved
  - dashboard validation was restored to a correct and auditable state

## 7. Retrain Comparison

- A full before/after comparison workflow was then built on branch `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`.
- New comparison notebooks added:
  - `01_Processing/04_Retraining/02_Baseline_Snapshot_Comparison.py`
  - `01_Processing/04_Retraining/03_Corrected_Grid_Comparison_Retrain.py`
  - `01_Processing/04_Retraining/04_Baseline_vs_Corrected_Comparison.py`
- Comparison output root:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_baseline_vs_corrected`

### 7.1 Execution Record

- Initial one-time Databricks submit run `549578963038936` failed because cluster creation permission was not available.
- Safe fallback used the approved existing cluster `0311-005159-pm2x3pce`.
- Existing-cluster submit run `94007764412417` executed:
  - baseline snapshot task `308402930586609`: `SUCCESS`
  - corrected retrain task `536958567073330`: `SUCCESS`
  - first comparison task `216954812415301`: `FAILED`
- Failure cause in the first comparison run:
  - notebook assumed a `regime` column existed in `silver_bt_master_grid_1h`
  - that assumption was invalid and was removed in follow-up commit `28b7292`
- Comparison-only rerun:
  - submit run `420098833423272`
  - comparison task `437698600576448`: `SUCCESS`

### 7.2 Comparison Results

- Baseline model version:
  - `bt_fred_1d_xgboost_regime_2026_04_01`
- Corrected comparison model version:
  - `bt_fred_1d_xgboost_regime_comparison_2026_04_01`
- Mean walk-forward accuracy:
  - baseline: `0.421968064458339`
  - corrected: `0.42224523228749594`
  - difference: `+0.00027716782915697733`
- Model-output differences observed:
  - recent walk-forward differing timestamps: `109`
  - repaired-window differing timestamps: `228`
  - recent RT differing timestamps: `13`
- Latest RT regime remained unchanged:
  - baseline: `Recession`
  - corrected: `Recession`

### 7.3 Safety Outcome

- The comparison retrain was explicitly non-promoting.
- Verified unchanged during comparison execution:
  - `pawan_quant_research_1.v4.model_registry_current`
  - live model alias `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- Production state was not overwritten during the comparison phase.

## 8. Interpretation

- The ALFRED selection bug was a real historical data-quality defect.
- Observed impact was not system-wide destabilization; it was concentrated around recent macro release windows where the wrong release-bucket observation had been selected.
- The downstream model remained broadly robust because:
  - the issue affected a small number of macro release points
  - z-score feature construction dampened raw-value shocks
  - model behavior was not singularly dependent on one corrected macro print
- Even with modest performance delta, the correction materially improves:
  - source-data integrity
  - point-in-time consistency
  - audit defensibility
  - interpretation of model confidence around repaired windows

## 9. Decision

- Recommendation: `promote after approval`
- Rationale:
  - corrected macro dataset is more internally and externally consistent
  - no destabilization was observed in dashboard validation or comparison retrain outputs
  - measured model change is small but directionally positive
  - governance and audit benefits outweigh the limited operational risk

## 10. Open Questions

- `GDPC1` remains a separate issue because the latest observation date appears selected while the value still differs.
- Event features remain intentionally unresolved in the dashboard proof:
  - `CPI_Event_Active`
  - `NFP_Event_Active`
- `UNRATE` remains an ambiguous duplicate-value case and does not yet justify a separate repair action.

## 11. Next Step

- Proceed to a full historical rebuild using the corrected ALFRED latest-observation selection logic.
- Scope of the next phase:
  - rebuild historical macro ingestion across the full `2016+` period
  - regenerate bronze, silver, and retraining outputs from the corrected point-in-time macro base
  - compare the fully rebuilt model outputs against current production
- Objective of the next phase:
  - achieve full PIT consistency across the backtest history
  - produce a complete audit-grade historical dataset rather than a targeted repair subset

## 12. State Separation

- Git state:
  - updated across multiple feature branches during this session sequence:
    - `feat/dashboard-alfred-selection-diagnostic-cleanup-2026-04-16-s2`
    - `feat/macro-alfred-latest-observation-selection-fix-2026-04-16-s2`
    - `feat/macro-targeted-bronze-backfill-payems-indpro-2026-04-17`
    - `feat/dashboard-fix-as-text-runtime-2026-04-17`
    - `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`
- Databricks repo state:
  - repo `2079003379491234` was aligned to branch `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`
  - Databricks repo head was updated to the comparison branch commit used for execution
- Actual workspace notebook state:
  - workspace notebooks for baseline capture, corrected retrain, and comparison were updated from the Databricks repo and executed
  - workspace execution state was therefore verified separately from local Git state
- Production workflows:
  - unchanged
  - no workflow JSON edits
  - no schedule changes
  - no RT topology changes
- Live model state:
  - live model alias was **not** overwritten
  - `model_registry_current` production row was **not** overwritten during the comparison workflow

