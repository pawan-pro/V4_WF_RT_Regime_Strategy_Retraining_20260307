# 2026-04-20 Improved ALFRED Model Update Bug Addressal Report

## 1. Executive Summary

This report is the consolidated handoff for the multi-day model/data investigation and release work completed around April 17-20, 2026.

In plain English:

- A real macro-data bug was identified in the ALFRED/FRED selection logic used in the legacy and production macro pipelines.
- The bug was not about choosing the wrong release bucket. It was about choosing the wrong row **inside** the correct bucket:
  - old behavior: first matching row / `iloc[0]`
  - intended behavior: row with the **latest `observation_date`** in that same release bucket
- That bug mattered because the selected macro value feeds:
  - macro bronze
  - silver feature history
  - model inputs
  - dashboard lineage checks
  - any downstream retrain or walk-forward comparison
- The issue was investigated through several distinct workstreams:
  - institutional dashboard lineage validation
  - targeted historical bronze repair
  - non-promoting targeted retrain comparison
  - full historical rebuild comparison
  - walk-forward PnL methodology parity work
  - walk-forward methodology reconciliation against the old reference notebook
  - legacy baseline recovery
  - canonical legacy-V4 performance generator creation
  - ALFRED-only recreated-signal comparison
  - final controlled production adoption
- The final production decision was:
  - adopt improved ALFRED logic into production
  - refresh production data layers broadly for consistency
  - retrain the production model
  - promote the retrained model
  - rerun RT inference and monitoring so live state matches the corrected data/model stack
- One residual historical validation caveat remains open and should be investigated separately:
  - historical silver PIT-join spot-checks for `CPIAUCSL`, `GDPC1`, and `INDPRO`

This release is considered successful overall. No rollback is currently indicated.

## 2. Initial Problem Statement

The original issue surfaced during institutional dashboard lineage validation. The dashboard proof logic was checking whether live-facing feature values could be explained cleanly from the production upstream data layers.

The main suspicion centered on macro features, especially:

- `PAYEMS`
- `INDPRO`

The root-cause hypothesis was:

- ALFRED release-bucket selection was using the first matching row rather than the latest observation month contained in the release bucket.

Why this matters:

- A single ALFRED release bucket can contain multiple observation months or revisions.
- For a release such as an early-March payrolls print, the intended value is the **latest observation month actually released in that bucket** (for example, February 2025), not an older revised month that happens to appear first in row order.
- If the wrong row is selected:
  - the bronze `actual` value is wrong
  - the silver feature carry-forward is wrong
  - the model input lineage is wrong
  - dashboard validation fails even when the realtime pipeline itself is behaving correctly

## 3. Phased Investigation Timeline

### Phase A — Dashboard lineage validation

**Goal**

Prove whether the institutional dashboard’s macro feature values could be traced cleanly from production upstream sources, and isolate whether the issue lived in bronze, silver, realtime enrichment, or the dashboard proof logic itself.

**Primary branches / notebooks / artifacts**

- Working lineage/debug effort ultimately rolled into `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`
- Git notebook: `02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb`
- Workspace execution path used by Databricks: `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard`

**What was learned**

- The dashboard issue was not a generic RT failure.
- The proof logic was strongest when treated as **lineage-first validation**:
  1. identify the expected macro release
  2. prove the bronze row
  3. prove the silver PIT value
  4. prove the realtime carry-forward
  5. then prove the dashboard/model input
- The tracked proof status was being summarized in shorthand counts of the form `14 / 12 / 0 / 0 / 2`:
  - fourteen tracked features or proof checks
  - twelve cleared after repair work
  - zero confirmed live-override defects in the resolved macro set
  - zero remaining core ALFRED-value mismatches after the targeted bronze fix
  - two unresolved event-pulse items kept separate from the ALFRED value-selection defect
- The two unresolved items were the event-pulse style features, not raw macro values:
  - `CPI_Event_Active`
  - `NFP_Event_Active`
- Indicator nuances became clear:
  - `PAYEMS`: a clean example of why bucket-internal row order matters
  - `INDPRO`: another strong example, complicated by its `09:15` release time
  - `GDPC1`: quarterly and estimate-heavy, so multiple release-bucket rows and observation periods make it more subtle than payrolls
  - `UNRATE`: tied to the payroll release schedule; useful as a paired control for payroll validation

**Outcome**

- The dashboard proof work identified the ALFRED selection bug as a real upstream lineage issue.
- The event-pulse features remained a separate downstream validation topic rather than the root bug.

### Phase B — Targeted bronze repair

**Goal**

Repair the clearly stale macro bronze rows first, restore dashboard proof, and avoid a broad production mutation before the bug was understood.

**Primary branches / notebooks / artifacts**

- Working branch lineage ultimately fed `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`
- Targeted repair notebook: `01_Processing/01_Ingestion/repair/01_BT_Macro_Bronze_Targeted_Backfill_2026-04-17.py`
- Silver refresh notebook: `01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py`
- RT refresh notebook: `01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py`
- Dashboard validation notebook: `02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb`

**What was learned**

- A narrow backfill could repair the obvious stale rows without rewriting the full history immediately.
- `PAYEMS` and `INDPRO` were the highest-value targets for a fast lineage restore.
- After the targeted repair, silver refresh, and RT refresh, the dashboard proof recovered materially.

**Outcome**

- The targeted repair restored trust in the dashboard lineage path.
- This phase justified moving to a **non-promoting retrain comparison** rather than jumping straight to production adoption.

### Phase C — Non-promoting retrain comparison

**Goal**

Measure what changes if the corrected data logic is used for retraining, without mutating production registry state or the live alias.

**Primary branches / notebooks / artifacts**

- Branch: `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`
- Notebooks:
  - `01_Processing/04_Retraining/02_Baseline_Snapshot_Comparison.py`
  - `01_Processing/04_Retraining/03_Corrected_Grid_Comparison_Retrain.py`
  - `01_Processing/04_Retraining/04_Baseline_vs_Corrected_Comparison.py`
- Output root: `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_baseline_vs_corrected`

**What was learned**

- The targeted corrected-grid retrain improved mean walk-forward accuracy slightly:
  - baseline mean accuracy: `0.421968`
  - corrected mean accuracy: `0.422245`
  - diff: `+0.000277`
- Recent RT output differences existed but were modest:
  - recent walk-forward differing timestamps: `109`
  - repaired-window differing timestamps: `228`
  - recent RT differing timestamps: `13`
- Latest RT regime did **not** change:
  - baseline: `Recession`
  - corrected: `Recession`

**Outcome**

- That phase-level recommendation was `promote after approval`.
- It was still explicitly non-promoting and was treated as an intermediate result, not final release evidence.

### Phase D — Full historical rebuild comparison

**Goal**

Test the stronger intervention: adopt corrected ALFRED logic across the entire historical dataset from 2016 onward, rebuild macro bronze and silver, retrain in comparison mode, and compare against baseline.

**Primary branches / notebooks / artifacts**

- Branches:
  - `feat/full-historical-alfred-rebuild-2016-2026`
  - later consolidated into `feat/full-historical-alfred-walkforward-decision-2026-04-20`
- Notebooks:
  - `01_Processing/01_Ingestion/repair/02_BT_Macro_Bronze_Full_Historical_Rebuild_2016_2026.py`
  - `01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py`
  - `01_Processing/04_Retraining/05_Full_Rebuild_Comparison.py`
- Output roots:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_full_rebuild`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_full_historical_adoption`

**What was learned**

- The full rebuild improved classification-style walk-forward accuracy materially:
  - baseline mean accuracy: `0.421968`
  - full rebuild mean accuracy: `0.448862`
- But it also changed the strategy object materially:
  - full-history regime changes vs baseline: `4240`
  - walk-forward changes vs baseline: `16623`
  - probability-diff rows: `43836` in the later full-historical decision pack
- Once the comparison was moved into the parity-correct walk-forward PnL frame, the full rebuild performed much worse as an investable strategy:
  - baseline final equity: `7124349.78`
  - rebuild final equity: `4252157.63`
  - baseline CAGR: `6.99%`
  - rebuild CAGR: `-3.04%`
  - baseline Sharpe: `0.4091`
  - rebuild Sharpe: `-0.0720`

**Why it looked better on accuracy but worse on PnL**

- The rebuild improved agreement with the label stream used in the walk-forward validation.
- But it changed the inferred regime path enough that the SPY/QQQ/CASH execution layer did not benefit on the actual invested period.
- Accuracy alone was therefore not decision-grade evidence.

**Outcome**

- Initial accuracy-only comparison briefly favored the full rebuild.
- Final decision-grade walk-forward evidence later reversed that conclusion and recommended `hold current V4`.

### Phase E — Walk-forward PnL methodology parity work

**Goal**

Determine whether the new PnL comparison notebook was actually faithful to the accepted walk-forward production methodology before using it to accept or reject any model variant.

**Primary branches / notebooks / artifacts**

- Branches:
  - `feat/pnl-baseline-methodology-parity-check-2026-04-18`
  - `feat/walkforward-pnl-engine-parity-correct-2026-04-18`
  - `feat/walkforward-methodology-reconciliation-2026-04-20`
- Notebooks and reference assets:
  - `01_Processing/04_Retraining/06_PnL_Comparison_V4_vs_V4_1.py`
  - branch-local parity notebook `01_Processing/04_Retraining/07_PnL_Baseline_Methodology_Parity_Check.py` (not present on the current local release branch)
  - branch-local corrected engine notebooks `08_Walkforward_PnL_Parity_Correct_Comparison.py` and `09_Walkforward_PnL_Baseline_vs_Rebuild.py` (not present on the current local release branch)
  - legacy reference notebook: `/Users/pawan/Downloads/11RT_WalkForward_Simulation 2026-03-12 08_50_27 (2).py`
- Output roots:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_pnl_parity_check`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_walkforward_pnl_parity_correct`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_walkforward_methodology_reconciliation`

**What was learned**

- The initial PnL notebook was **not** parity-correct.
- The parity audit found these material divergences:
  - same-row signal application instead of prior-bar application
  - hourly risk metrics instead of daily-resampled-equity metrics
  - multiplicative cost logic instead of subtractive return-minus-cost logic
  - baseline truncation to the overlap horizon with rebuild outputs
- The old reference notebook also did **not** survive direct reconciliation as a canonical decision tool:
  - it used a different signal-generation lineage
  - it depended on the `default` namespace
  - it relied on full-signal / teacher-label objects rather than the current walk-forward baseline stream
- The reconciliation verdict was:
  - `new_engine_is_canonical`
  - dominant gap driver: `signal_source_and_namespace`

**Outcome**

- The parity-correct walk-forward engine became the canonical evaluation method for current V4-vs-V4.1 comparison work.
- The legacy notebook was retired from decision use and kept only as archival research context.

### Phase F — Legacy baseline recovery work

**Goal**

Recreate the old V4 stack faithfully enough that a controlled ALFRED-only comparison would be meaningful.

**Primary branches / notebooks / artifacts**

- Branches:
  - `feat/legacy-v4-stack-rebuild-and-alfred-swap-2026-04-20`
  - `feat/repair-legacy-macro-series-recovery-2026-04-20`
  - `feat/close-final-legacy-baseline-gap-2026-04-20`
- Legacy reference notebooks supplied from `/Users/pawan/Downloads/`:
  - `V4_02RT_FinnHub_Macro_Ingestion.py`
  - `V4_05RT_Master_Time_Grid_Silver.py`
  - `V4_06RT_XGBoost_Regime_V4_Training.py`
  - `11RT_WalkForward_Simulation 2026-03-12 08_50_27 (2).py`
- Output roots:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_legacy_macro_repair`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_legacy_baseline_gap_closure`

**What was learned**

- Early recovery failed or was only partial because the recreated legacy stack was missing or collapsing key macro information:
  - haystack release-date gaps
  - ALFRED bucket collapse
  - incomplete series coverage
- The repaired macro audit showed the expected legacy macro stack had to include:
  - `CPIAUCSL`
  - `PCEPI`
  - `PCE`
  - `PAYEMS`
  - `UNRATE`
  - `GDPC1`
  - `INDPRO`
  - `UMCSENT`
- The authoritative legacy CSV layer mattered.
- The authoritative legacy label/signal lineage also mattered. This is why the old reference notebook and the current production walk-forward stream diverged.
- After the gap-closure phase, the recreated legacy baseline became close enough to proceed:
  - signal match rate: `0.970313`
  - regime mismatch count: `1301`
  - recovery verdict: `recovery_succeeded`

**Outcome**

- Legacy recovery moved from `recovery_partial` to `recovery_succeeded`.
- That unlocked a controlled ALFRED-only comparison on recreated signals.

### Phase G — Canonical legacy-V4 performance-table generator

**Goal**

Stop re-litigating performance methodology and create one stable, reusable reporting engine for the recreated legacy V4 strategy.

**Primary branches / notebooks / artifacts**

- Branch: `feat/canonical-legacy-v4-performance-table-generator-2026-04-20`
- Generator notebook:
  - logical repo path used by the control manifest: `01_Processing/04_Retraining/17_Canonical_Legacy_V4_Performance_Table_Generator.py`
  - workspace path used during execution: `/Users/pawan.jatale@mubama.group/tmp/freeze_canonical_baseline_and_rerun_alfred_corrected_2026_04_20/01_Processing/04_Retraining/17_Canonical_Legacy_V4_Performance_Table_Generator`
  - note: the file is referenced by the output manifests but is **not** present on the current local release branch
- Output root: `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_performance`

**Fixed assumptions**

- hourly signals
- hourly market prices
- prior-bar execution
- `0 = Expansion -> QQQ`
- `1 = Slowdown -> SPY`
- `2 = Recession/Contraction -> CASH`
- 5 bps subtractive switch cost
- hourly equity path
- daily-resampled reporting
- 365-day CAGR convention

**What was learned**

- The generator created a stable control object that future comparisons can reuse.
- Frozen control performance:
  - final value: `12800849.77`
  - CAGR: `20.6851%`
  - Sharpe: `1.0231`
  - Sortino: `1.3932`
  - max drawdown: `-21.8173%`

**Outcome**

- This generator became the canonical legacy-V4 baseline reporting engine for later ALFRED-only comparisons.

### Phase H — ALFRED-only recreated-signal comparison

**Goal**

Run a fair comparison where the recreated legacy baseline and the corrected ALFRED variant use the **same recreated-signal stack** and the **same performance generator**.

**Primary branches / notebooks / artifacts**

- Branches:
  - `feat/legacy-recreated-signals-alfred-only-comparison-2026-04-20`
  - `feat/freeze-canonical-baseline-and-rerun-alfred-corrected-2026-04-20`
- Historical branch-local notebooks referenced during this work:
  - `01_Processing/04_Retraining/16_Legacy_Recreated_Signals_ALFRED_Only_Comparison.py`
  - `01_Processing/04_Retraining/18_Freeze_Canonical_Baseline_And_Rerun_ALFRED_Corrected.py`
  - note: those file paths were used in session scope, but the files are not present on the current local release branch; the output folders are the durable evidence
- Output roots:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_legacy_recreated_signals_alfred_only`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected`

**What was learned**

- There were two useful readings of the ALFRED-only effect:

1. **Raw recreated-signal comparison**
   - baseline CAGR: `13.6527%`
   - corrected CAGR: `12.9374%`
   - recommendation at that stage: `keep recreated baseline`

2. **Frozen canonical control comparison**
   - baseline final value: `12800849.77`
   - corrected final value: `12883454.99`
   - baseline CAGR: `20.6851%`
   - corrected CAGR: `20.8405%`
   - baseline Sharpe: `1.0231`
   - corrected Sharpe: `1.0024`
   - baseline Sortino: `1.3932`
   - corrected Sortino: `1.3729`
   - baseline max drawdown: `-21.8173%`
   - corrected max drawdown: `-25.5385%`
   - switch count improved from `326` to `296`
   - average holding bars increased from `61.02` to `67.21`

- Interpretation:
  - corrected ALFRED was **not** a clean across-the-board win
  - it slightly improved final value and CAGR
  - it reduced switching and lengthened holds
  - it slightly worsened Sharpe/Sortino and max drawdown

**Outcome**

- The final canonical comparison verdict was `further review needed`, not a research-only promotion order.
- That nuance mattered in the final production decision: the adoption case was driven partly by data correctness and monitoring fidelity, not only by a single backtest score.

### Phase I — Final production adoption

**Goal**

Adopt the improved ALFRED logic into production in a controlled way, refresh the production data/model stack consistently, and validate live-facing outputs.

**Primary branches / notebooks / artifacts**

- Final release branch: `feat/promote-improved-alfred-and-retrain-production-2026-04-20`
- Databricks repo branch **before** release: `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`
- Databricks repo head before release: `28b7292c04019e7e97475f44d31f675461612ca6`
- Databricks repo head after release: `688505a91ef14c5d64dbf576ad122ac664b8a1d6`
- Production helper notebook: `01_Processing/04_Retraining/19_Improved_ALFRED_Production_Release.py`
- Release artifact root: `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_improved_alfred_production_release`

**What was learned**

- The production incremental macro notebook path already contained the improved ALFRED logic on the Databricks repo branch that was live before the final release; the release branch was therefore based on that branch rather than on `main`.
- The missing pieces for a controlled production adoption were:
  - historical/bootstrap consistency
  - validation packaging
  - clean release bookkeeping
- Several repair steps were needed during the release itself:
  - the old historical bootstrap path failed on timezone localization and was fixed
  - the first silver rebuild attempt failed because the old bootstrap path did not produce full macro coverage
  - the validation helper initially failed because `pandas.to_markdown()` needed `tabulate`
  - the monthly retrain updated the live alias in place but reused the month-bucket version label `bt_fred_1d_xgboost_regime_2026_04_01`, so a one-off registry/version fix was added

**Outcome**

- Production macro selection now uses the corrected ALFRED rule end to end.
- Production data layers were refreshed broadly:
  - historical macro overwrite
  - full silver rebuild
  - monthly retrain
  - RT inference refresh
  - dashboard feed refresh
  - institutional dashboard refresh
- Final promoted model version:
  - `bt_fred_1d_xgboost_regime_2026_04_20`
- Live alias path:
  - `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- Final release status:
  - successful overall
  - one residual historical silver PIT-join caveat remains open

## 4. Important Files / Notebooks / Tables

This section lists the most important code, notebooks, tables, and artifact roots. Where a historical notebook path is not present on the current local release branch, that is stated explicitly.

### Core production notebooks

- `01_Processing/01_Ingestion/03_BT_Macro_Incremental_Update.py`
- `01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py`
- `01_Processing/01_Ingestion/repair/01_BT_Macro_Bronze_Targeted_Backfill_2026-04-17.py`
- `01_Processing/01_Ingestion/repair/02_BT_Macro_Bronze_Full_Historical_Rebuild_2016_2026.py`
- `01_Processing/02_Features/01_BT_Master_Time_Grid_Silver.py`
- `01_Processing/02_Features/02_BT_Incremental_Silver_Refresh.py`
- `01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment.py`
- `01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py`
- `01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py`
- `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py`
- `01_Processing/04_Retraining/19_Improved_ALFRED_Production_Release.py`
- Git dashboard file: `02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb`
- Workspace dashboard execution path: `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard`

### Comparison / recovery notebooks

- `01_Processing/04_Retraining/02_Baseline_Snapshot_Comparison.py`
- `01_Processing/04_Retraining/03_Corrected_Grid_Comparison_Retrain.py`
- `01_Processing/04_Retraining/04_Baseline_vs_Corrected_Comparison.py`
- `01_Processing/04_Retraining/05_Full_Rebuild_Comparison.py`
- `01_Processing/04_Retraining/06_PnL_Comparison_V4_vs_V4_1.py`
- `01_Processing/04_Retraining/17_Canonical_Legacy_V4_Performance_Table_Generator.py`
  - referenced by `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_performance/control_manifest.json`
  - workspace path used: `/Users/pawan.jatale@mubama.group/tmp/freeze_canonical_baseline_and_rerun_alfred_corrected_2026_04_20/01_Processing/04_Retraining/17_Canonical_Legacy_V4_Performance_Table_Generator`
  - not present on the current local release branch
- `01_Processing/04_Retraining/18_Freeze_Canonical_Baseline_And_Rerun_ALFRED_Corrected.py`
  - historical session notebook name
  - current local branch does not contain the file
  - durable output root: `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected`
- `01_Processing/04_Retraining/07_PnL_Baseline_Methodology_Parity_Check.py`
- `01_Processing/04_Retraining/08_Walkforward_PnL_Parity_Correct_Comparison.py`
- `01_Processing/04_Retraining/09_Walkforward_PnL_Baseline_vs_Rebuild.py`
- `01_Processing/04_Retraining/13_Legacy_V4_Stack_ALFRED_Swap_Comparison.py`
- `01_Processing/04_Retraining/14_Legacy_Macro_Series_Recovery.py`
- `01_Processing/04_Retraining/15_Legacy_Baseline_Gap_Closure.py`
- `01_Processing/04_Retraining/16_Legacy_Recreated_Signals_ALFRED_Only_Comparison.py`
  - the `07`-`09` and `13`-`16` files were used in session scope and are represented by output artifacts, but are not present on the current local release branch

### Legacy reference notebooks

- `/Users/pawan/Downloads/V4_02RT_FinnHub_Macro_Ingestion.py`
- `/Users/pawan/Downloads/V4_05RT_Master_Time_Grid_Silver.py`
- `/Users/pawan/Downloads/V4_06RT_XGBoost_Regime_V4_Training.py`
- `/Users/pawan/Downloads/11RT_WalkForward_Simulation 2026-03-12 08_50_27 (2).py`

### Production tables

- `pawan_quant_research_1.v4.bronze_bt_macro_events`
- `pawan_quant_research_1.v4.silver_bt_master_grid_1h`
- `pawan_quant_research_1.v4.silver_rt_1h`
- `pawan_quant_research_1.v4.model_registry_current`
- `pawan_quant_research_1.v4.bt_walkforward_regime_signals`
- `pawan_quant_research_1.v4.bronze_rt_regime_signal`
- `pawan_quant_research_1.v4.gold_bt_returns_daily`
- `pawan_quant_research_1.v4.gold_bt_returns_weekly`
- `pawan_quant_research_1.v4.gold_bt_returns_monthly`
- `pawan_quant_research_1.v4.gold_bt_returns_quarterly`
- `pawan_quant_research_1.v4.gold_bt_returns_yearly`
- `pawan_quant_research_1.v4.gold_bt_performance_summary`
- Production walk-forward price source used by parity-correct engine: `pawan_quant_research_1.v4.bronze_bt_market_1h`

### Comparison-safe tables

- `pawan_quant_research_1.v4.bronze_bt_macro_events_rebuild`
- `pawan_quant_research_1.v4.silver_bt_master_grid_1h_rebuild`
- `pawan_quant_research_1.v4.bronze_bt_macro_events_legacy_recreated_baseline`
- `pawan_quant_research_1.v4.bronze_bt_macro_events_legacy_recreated_corrected`
- `pawan_quant_research_1.v4.silver_bt_master_grid_1h_legacy_recreated_baseline`
- `pawan_quant_research_1.v4.silver_bt_master_grid_1h_legacy_recreated_corrected`
- `pawan_quant_research_1.v4.bt_walkforward_regime_signals_legacy_recreated_baseline`
- `pawan_quant_research_1.v4.bt_walkforward_regime_signals_legacy_recreated_corrected`
- Canonical legacy performance generator price source: `pawan_quant_research_1.default.bronze_bt_market_1h`

### Model artifacts and registry objects

- Current live alias path: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- Pre-release versioned model: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_04_01.joblib`
- Post-release versioned model: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_04_20.joblib`
- Registry table: `pawan_quant_research_1.v4.model_registry_current`

### Output artifact folders

- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_baseline_vs_corrected`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_full_rebuild`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_pnl_parity_check`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_walkforward_pnl_parity_correct`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_full_historical_adoption`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_walkforward_methodology_reconciliation`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_legacy_macro_repair`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_legacy_baseline_gap_closure`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_legacy_recreated_signals_alfred_only`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_performance`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected`
- `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_improved_alfred_production_release`

## 5. Key Decisions Made

### Why targeted bronze repair was done first

- The dashboard lineage problem was visible on a small number of known stale rows.
- A targeted repair was the fastest way to separate a **real upstream data defect** from a broader modeling or RT issue.
- It restored dashboard proof without forcing an immediate production promotion decision.

### Why the full historical rebuild was treated as research/comparison before promotion

- A full historical overwrite changes the entire feature history and therefore the entire retrain state.
- That kind of change can improve classification metrics while still hurting investable walk-forward performance.
- It had to be tested non-promotingly first.

### Why walk-forward PnL methodology had to be reconciled

- The early accuracy-only comparison and the early PnL comparison disagreed.
- The disagreement was not trustworthy until the PnL engine was proven methodology-consistent.
- The parity audit showed that the original notebook was not consistent enough to drive a promotion decision.

### Why the old legacy baseline had to be recovered before trusting ALFRED-only comparison

- A controlled A/B comparison is only as good as its control.
- The early recreated legacy baseline was too weak because the recreated macro stack was incomplete.
- Missing or collapsed legacy macro series would have made any ALFRED-only effect estimate unreliable.

### Why a canonical legacy-V4 performance table generator was created

- Too many notebooks were reporting performance with slightly different assumptions.
- The canonical generator froze one set of assumptions and made later comparisons auditable.
- It stopped the comparison from drifting every time a notebook changed timing, cost, or annualization conventions.

### Why the improved ALFRED strategy was still adopted into production

- The underlying bug was real and lived in production lineage, not just research code.
- Monitoring fidelity mattered. If production data logic stayed old while comparison or validation logic used corrected ALFRED semantics, train/serve/monitor mismatch would persist.
- The canonical ALFRED-only comparison was not a clean performance win, but it did show:
  - slightly higher final value
  - slightly higher CAGR
  - fewer switches
  - longer average holding periods
- The downside was:
  - slightly worse Sharpe and Sortino
  - deeper max drawdown
- The decision was therefore not “promote because backtest clearly dominates.”
- The actual rationale was:
  - fix the real data bug
  - keep the production model aligned with corrected data
  - keep monitoring lineage coherent
  - accept the mild performance trade-off as operationally reasonable

### Why no rollback was triggered after release

- Macro selection validation passed for the key release examples.
- RT silver alignment passed.
- Live alias, registry version, latest regime snapshot, and dashboard state were coherent after the one-off registry fix.
- The remaining caveat is historical silver PIT-join validation, not a proven live-state failure.

## 6. Metrics Snapshot

The table below captures the load-bearing metrics. It is intentionally selective.

### Legacy reference and canonical legacy comparison

| Object | Final Value | CAGR | Volatility | Sharpe | Sortino | Max Drawdown | Switches | Avg Holding Bars | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Trusted legacy reference | 12787240.33 | 18.7642% | 19.6924% | 0.9911 | 1.3801 | -23.7429% | — | — | Historical benchmark used during recovery and reconciliation |
| Canonical legacy baseline control | 12800849.77 | 20.6851% | 19.6191% | 1.0231 | 1.3932 | -21.8173% | 326 | 61.02 | Frozen reusable control from canonical generator |
| Canonical corrected-ALFRED variant | 12883454.99 | 20.8405% | 20.2798% | 1.0024 | 1.3729 | -25.5385% | 296 | 67.21 | Same generator, only ALFRED selection changed |

### Walk-forward production comparison frame

| Model | Final Equity | CAGR | Annualized Vol | Sharpe | Max Drawdown | Calmar | Switches | Avg Holding Bars | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current V4 baseline | 7124349.78 | 6.9905% | 13.7116% | 0.4091 | 16.3622% | 0.4272 | 1282 | 35.84 | Parity-correct walk-forward engine |
| Targeted fix | 7114719.91 | 6.9628% | 14.7204% | 0.3890 | 19.6075% | 0.3551 | 1261 | 36.44 | Parity-correct walk-forward engine |
| Full rebuild V4.1 | 4252157.63 | -3.0444% | 14.6249% | -0.0720 | 48.8203% | -0.0624 | 1246 | 36.88 | Accuracy improved, investable PnL deteriorated sharply |

### Legacy recovery checkpoint metrics

| Checkpoint | Final Equity | CAGR | Sharpe | Max Drawdown | Signal Match Rate | Regime Mismatch Count | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy macro repair | 11956049.22 | 12.7925% | 0.7925 | -29.1828% | 0.925041 | 3285 | `recovery_partial` |
| Legacy baseline gap closure | 15015893.94 | 16.3980% | 0.9970 | -22.9391% | 0.970313 | 1301 | `recovery_succeeded` |

### Signal-comparison highlights

- Recreated baseline vs trusted legacy reference signal match: `0.895674`
- Recreated corrected-vs-baseline signal match: `0.844263`
- Recreated corrected-vs-baseline regime mismatches: `6825`
- Bronze rows changed by ALFRED-only swap in recreated legacy comparison: `39`

## 7. Production Release Details

### Release branch and commits

- Release branch: `feat/promote-improved-alfred-and-retrain-production-2026-04-20`
- Release branch head used for final Databricks repo sync: `688505a91ef14c5d64dbf576ad122ac664b8a1d6`
- Notable release-branch commits:
  - `81ad88c` `Add production full historical bronze rebuild helper`
  - `bb265c3` `Fix historical macro bootstrap timezone localization`
  - `8c04de1` `Promote improved ALFRED historical bootstrap release helper`
  - `71fb929` `Fix release validation markdown rendering`
  - `688505a` `Add one-off release registry version fix`

### What code changed

- `01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py`
  - changed ALFRED selection from first-row behavior to max-`observation_date` within bucket
  - added `release_date` to generated CSV rows
  - fixed timezone localization bug in historical bootstrap path
- `01_Processing/01_Ingestion/repair/02_BT_Macro_Bronze_Full_Historical_Rebuild_2016_2026.py`
  - added/used as the full historical bronze rebuild helper for production consistency
- `01_Processing/04_Retraining/19_Improved_ALFRED_Production_Release.py`
  - added snapshot mode
  - added validate mode
  - added dependency-free markdown rendering
  - added one-off registry/version-fix mode
- No workflow JSON was changed.
- No RT notebook code was changed.

### What ran in Databricks

- Existing cluster used for the release and validation runs: `0311-005159-pm2x3pce`
- Pre-release snapshot
  - run `897151326104232`
- Historical/bootstrap attempts
  - old bootstrap attempt `1120721712636801` failed on timezone localization
  - old bootstrap retry `530155191797256` succeeded but was not sufficient for full macro coverage
  - full production bronze rebuild `690935609075207` succeeded
- Silver rebuild
  - first attempt `395166435277364` failed because the upstream bronze was incomplete
  - full silver rebuild `358189648810498` succeeded
- Production monthly retraining workflow
  - workflow run `782840605939610` succeeded
  - `macro_incremental_update` `235237763103214`
  - `silver_incremental_refresh` `447993380452536`
  - `monthly_walkforward_retrain` `371193849247293`
  - `performance_exports` `562562030968491`
- RT and dashboard refresh pass 1
  - RT silver `353078018372181`
  - RT inference `522087985945025`
  - dashboard feed `652911351992280`
  - institutional dashboard `111281500996101`
- Validation helper
  - first post-release validation failed because `tabulate` was missing: `624005266215520`
  - second validation after helper patch succeeded: `441431362158484`
- Registry bookkeeping fix
  - one-off registry fix `743396843982859`
- RT and dashboard refresh pass 2 after registry fix
  - RT inference `964229653014430`
  - dashboard feed `882322785638385`
  - institutional dashboard `908928575044461`
- Final validation
  - final validation run `13657699796531`

### What tables were refreshed

- `pawan_quant_research_1.v4.bronze_bt_macro_events`
- `pawan_quant_research_1.v4.silver_bt_master_grid_1h`
- `pawan_quant_research_1.v4.model_registry_current`
- `pawan_quant_research_1.v4.bt_walkforward_regime_signals`
- `pawan_quant_research_1.v4.gold_bt_returns_*`
- `pawan_quant_research_1.v4.gold_bt_performance_summary`
- `pawan_quant_research_1.v4.silver_rt_1h`
- `pawan_quant_research_1.v4.bronze_rt_regime_signal`

### Refresh shape

- This was **broad**, not targeted:
  - broad historical macro overwrite
  - broad silver rebuild
  - retrain and promotion
  - RT/monitoring refresh

### Model version before and after release

- Before release:
  - model version: `bt_fred_1d_xgboost_regime_2026_04_01`
  - versioned artifact: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_04_01.joblib`
- After release:
  - model version: `bt_fred_1d_xgboost_regime_2026_04_20`
  - versioned artifact: `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_04_20.joblib`
- Live alias path remained:
  - `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`

### Pre-release snapshot details

- Snapshot file root: `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_improved_alfred_production_release`
- Pre-release model version: `bt_fred_1d_xgboost_regime_2026_04_01`
- Pre-release promoted timestamp: `2026-04-06T16:04:37.608268`
- Pre-release latest regime snapshot:
  - timestamp: `2026-04-17T20:00:00`
  - regime: `Recession`
  - position: `100% CASH`

### Post-release validation results

- Passed:
  - `macro_release_selection_CPIAUCSL`
  - `macro_release_selection_GDPC1`
  - `macro_release_selection_INDPRO`
  - `macro_release_selection_PAYEMS`
  - `macro_release_selection_UNRATE`
  - `rt_silver_alignment`
  - `model_registry_promoted`
  - `live_model_alias_exists`
  - `latest_regime_matches_registry`
  - `latest_feature_snapshot_present`
  - `dashboard_run_success`
- Failed:
  - `silver_pit_join_CPIAUCSL`
  - `silver_pit_join_GDPC1`
  - `silver_pit_join_INDPRO`

### Residual validation caveat

- The only remaining flagged validation items are the historical silver PIT-join spot-checks for:
  - `CPIAUCSL`
  - `GDPC1`
  - `INDPRO`
- Importantly:
  - the macro release-bucket selection itself passed
  - RT silver alignment passed
  - the latest regime snapshot matched the promoted registry version

## 8. Rollback Context

Rollback is **not** currently recommended, but the references are preserved.

### Pre-release rollback anchors

- Pre-release model version:
  - `bt_fred_1d_xgboost_regime_2026_04_01`
- Pre-release versioned model path:
  - `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_04_01.joblib`
- Pre-release live alias path:
  - `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`
- Pre-release latest regime snapshot:
  - timestamp: `2026-04-17T20:00:00`
  - regime: `Recession`
  - position: `100% CASH`
- Pre-release release artifacts:
  - `pre_release_model_metadata.json`
  - `pre_release_registry_snapshot.csv`
  - `pre_release_latest_regime_snapshot.json`
  - `pre_release_latest_feature_snapshot.csv`

### If rollback were ever required

The minimum rollback steps would be:

1. restore the versioned model `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_2026_04_01.joblib` back to the live alias path
2. restore `model_registry_current` to the pre-release row captured in `pre_release_registry_snapshot.csv` / `pre_release_model_metadata.json`
3. rerun:
   - `01_Processing/03_Realtime/04_RT_Model_Inference_Engine.py`
   - `01_Processing/03_Realtime/10_RT_Saxo_Dashboard_Feed.py`
   - `02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb`
4. if a data rollback were required, restore bronze/silver using Delta time travel or rerun the pre-release branch logic

### Important rollback nuance

- The release pack preserves model and regime rollback context cleanly.
- It does **not** include a separate pre-release bronze/silver clone table pack.
- Because the release used a broad overwrite/rebuild shape, table rollback would rely on:
  - Delta time travel, if available
  - or rerunning the pre-release Databricks repo branch logic

### Rollback recommendation

- Current recommendation: **do not roll back**
- Reason:
  - release is successful overall
  - live-facing model and monitoring are coherent
  - open issue is a historical PIT-join spot-check, not a proven live-state defect

## 9. What Remains Open

### Open production validation items

1. Investigate the historical silver PIT-join validation caveat for:
   - `CPIAUCSL`
   - `GDPC1`
   - `INDPRO`
2. Determine whether those failed checks are caused by:
   - a timestamp expectation mismatch in the validation helper
   - a legitimate historical silver join issue
   - or a release-time rounding / floor / +2 hour alignment nuance
3. Do **not** change production behavior on the basis of that caveat alone until a real defect is confirmed.

### Open research items

1. The recreated legacy baseline is closer to the historical reference than before, but it is still not a bit-for-bit clone of every historical legacy notebook behavior.
2. The full-dataset / full-signal high-CAGR framing remains a separate research topic and was explicitly kept out of the production decision frame.
3. Broader live macro rollout beyond the currently validated scope remains deferred.
4. Any broader model or feature redesign should be treated as new work, not as an extension of this bug-addressal release.

## 10. Recommended Next Steps

1. Investigate the residual historical silver PIT-join validation caveat for `CPIAUCSL`, `GDPC1`, and `INDPRO` without changing production behavior unless a true defect is confirmed.
2. Preserve and reference the canonical legacy-V4 performance generator and its frozen control manifest for future comparison work.
3. Use the improved ALFRED production state as the new monitoring baseline.
4. Keep any future model/data enhancements separate from this bug-addressal release.
5. Only revisit broader model or feature improvements after the PIT-join caveat is understood.

## 11. State Separation

This section is explicit because multiple states diverged during validation and release work.

### Git state

- Final production release branch: `feat/promote-improved-alfred-and-retrain-production-2026-04-20`
- Release head committed for deployment: `688505a91ef14c5d64dbf576ad122ac664b8a1d6`
- Important nuance:
  - this report may later exist on the same branch as a documentation-only change
  - the actual production release state is the branch head used during Databricks repo sync and deployment, not any later documentation commit

### Databricks repo git state

- Workspace repo path: `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github`
- Repo id: `2079003379491234`
- Repo branch before release: `feat/retrain-baseline-vs-corrected-grid-comparison-2026-04-18`
- Repo head before release: `28b7292c04019e7e97475f44d31f675461612ca6`
- Repo branch after release: `feat/promote-improved-alfred-and-retrain-production-2026-04-20`
- Repo head after release: `688505a91ef14c5d64dbf576ad122ac664b8a1d6`

### Workspace notebook state

- Git files and Databricks workspace notebooks are not identical objects.
- Databricks repo-backed notebook execution used workspace paths such as:
  - `/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/01_Processing/04_Retraining/19_Improved_ALFRED_Production_Release`
  - `/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard`
- During release prep, a temporary helper notebook also existed at:
  - `/Users/pawan.jatale@mubama.group/tmp/improved_alfred_production_release_2026_04_20/19_Improved_ALFRED_Production_Release`
- Final validation authority was the **repo-backed** notebook, not the temp copy.
- The dashboard git file is `04_Saxo_Institutional_Dashboard.ipynb`, but the executed workspace notebook path does not carry the `.ipynb` suffix.

### Production/live state

- Production/live state after release is not just “what git says.”
- Production/live state means:
  - production bronze and silver tables refreshed
  - `model_registry_current` advanced to `bt_fred_1d_xgboost_regime_2026_04_20`
  - live alias still points to `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`, but now represents the corrected release
  - RT inference rerun after registry fix
  - dashboard feed rerun
  - institutional dashboard rerun
  - latest live-facing regime snapshot moved to:
    - timestamp `2026-04-20T12:00:00`
    - regime `Recession`
    - position `100% CASH`
    - model version `bt_fred_1d_xgboost_regime_2026_04_20`

## 12. Handoff Notes for Future Codex Sessions

- Treat this report as the main starting document for the April 2026 ALFRED bug-addressal and model-update work.
- Trust the canonical legacy baseline reporting engine captured in:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_performance`
  - especially `control_manifest.json`, `performance_metrics.csv`, and `methodology_notes.md`
- Trust the production model version after release as:
  - `bt_fred_1d_xgboost_regime_2026_04_20`
- Do **not** redo these workstreams unless the comparison frame changes materially:
  - non-parity PnL reconciliation
  - legacy baseline recovery
  - canonical generator construction
  - broad full-historical adoption comparison
- The key evidence folders to read first are:
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_pnl_parity_check`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-18_walkforward_pnl_parity_correct`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_walkforward_methodology_reconciliation`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_legacy_baseline_gap_closure`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_performance`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected`
  - `/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_improved_alfred_production_release`
- The first unresolved issue to investigate is:
  - the historical silver PIT-join validation caveat for `CPIAUCSL`, `GDPC1`, and `INDPRO`
- For future sessions that touch live jobs, also read:
  - `AGENTS.md`
  - `05_Context/04_Codex_Project_Memory.md`
