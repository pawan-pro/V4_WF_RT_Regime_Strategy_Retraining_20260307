# MUB-27 Model Research Summary - 2026-05-14

## Current Decision

Keep the current XGBoost live/primary. Do not productionize or shadow-track the current TFT, TFT/XGBoost overlay, compact meta-label tree, compact model-family comparison, or compact direct-allocation research outputs.

The recent MUB-27 research sequence did not find a strict-PIT candidate that beats SPY/S&P 500 on both CAGR and Sharpe while respecting the drawdown and zero-equity constraints. Several candidates beat the weak strict-PIT XGBoost reference, but those results remain below SPY and therefore are not accepted.

No production V4 tables, jobs, Saxo/token objects, RT/dashboard notebooks, model registry entries, schedules, or trading logic were changed.

## Research Guardrails

- All work is research-only.
- Do not mutate production V4 Delta tables, Saxo execution/token refresh, RT notebooks, dashboard notebooks, Databricks workflows/jobs, schedules, model registry entries, live model aliases, or production trading logic.
- No model result can be accepted unless grid validation passes.
- Required reconciled grid:
  `/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/grids/full_legacy_grid_clone.csv`
- Required grid SHA-256:
  `b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1`
- Required feature set is the original 14 V4 features only for first-pass runs; do not add calendar/time features because prior ablation hurt.
- B/C2 teacher-smoothed references are non-PIT reference only.

## Completed Research Legs

| Leg | Branch | Commit / Run | Output Root | Result |
| --- | --- | --- | --- | --- |
| TFT broad improvement grid | `research/v4-2-tft-only-improvement-grid-2026-05-11` | run `560935970222749`; commit used `d72640283e7bc11a2337459c9e362d7b5a0a7424` | `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_only_improvement_grid_2026-05-11/` | Broad run wrote core summaries but final artifact generation failed after driver stop. `candidate_rankings.csv` was empty because no row passed full seed-stability acceptance. Pre-stability diagnostics had 511 rows beating SPY on CAGR and Sharpe. Strong diagnostic family: `72h + dd_075_ret_050 + conservative_uncertainty_override + BIL`, but not stable. |
| TFT targeted stability | `research/v4-2-tft-only-improvement-grid-2026-05-11` | run `941935482517226`; commit `1374659393eba51430d91cc42585e08f664fb856` | `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_targeted_stability_2026-05-12/` | No seed-stable accepted standalone TFT candidate. |
| Original TFT leverage candidate audit | TFT audit branch / TFT research branch | see MUB-27 Linear history | `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_original_candidate_audit_2026-05-12/` | Original `tft72_bil_drawdown_aware_dd_010_ret_005` leverage table remains valid historical evidence only. No model/scaler artifacts were found, so it is not replayable as a live/shadow model. Retraining the exact recipe was not seed-stable. |
| TFT/XGBoost overlay | TFT overlay research | see MUB-27 Linear history | `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_xgboost_overlay_2026-05-12/` | TFT as an XGBoost risk/conviction overlay did not beat SPY on both CAGR and Sharpe. Do not productionize TFT overlay. |
| Compact meta-label tree | `research/v4-2-strict-pit-meta-label-tree-2026-05-13` | run `234250509108143`; commit `1372fc1ff03a5525f0c2c6ff9eab00235ba3d1df` | `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_strict_pit_meta_label_tree_2026-05-13/` | No accepted candidates. Best row was hard veto with SPY fallback at threshold 0.65: CAGR 3.66%, Sharpe 0.301, Max DD -34.52%. It beat weak strict-PIT XGBoost but not SPY. |
| Compact model-family comparison | `research/v4-2-model-family-comparison-2026-05-13` | run `232246246505361`; commit `9a6ce9c9ca4cd9cc8f0803965e055057ddb751d7` | `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_model_family_comparison_2026-05-13/` | Grid validation passed. No accepted candidates. Best compact row was XGBoost general-trust hard veto: CAGR 4.80%, Sharpe 0.373, Max DD -32.73%. It did not beat SPY. |
| Compact direct-allocation model family | `research/v4-2-direct-allocation-model-family-2026-05-14` | run `932948816822416`; commit `6e3a9e4bdb4593d35c27e880587bd1caf6c92872` | `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_direct_allocation_model_family_2026-05-14/` | Grid validation passed. No accepted candidates. Best headline-financing row reached 7.07% CAGR, 0.508 Sharpe, -34.17% Max DD. Best 0% financing row reached 7.26% CAGR, ~0.51 Sharpe. No row approached 15-20% CAGR or beat SPY on Sharpe. |

## Detailed Conclusions

### TFT

- Current TFT recipe is not seed-stable.
- The original TFT leverage table is valid historical research evidence only.
- No original TFT model/scaler artifacts were found.
- The exact original TFT recipe is `historical_artifact_only`, not replayable as a live or shadow model.
- TFT/XGBoost overlay did not beat SPY on both CAGR and Sharpe.
- Do not productionize or shadow-track the current TFT recipe.

### Defensive Veto / Meta-Label Direction

- Defensive veto-style policies reduce exposure and can improve against weak strict-PIT XGBoost references, but they crush CAGR relative to SPY.
- Compact meta-label tree first pass did not beat SPY.
- Compact model-family comparison with general-trust and risk-off targets did not beat SPY.
- This direction is not sufficient for the 15-20% CAGR research objective.

### Direct Allocation Direction

- Direct allocation improved over defensive veto/meta-label families but still did not meet the objective.
- Top compact direct-allocation policies were too defensive in realized exposure:
  - QQQ share: ~3.7%
  - SPY share: ~66.8%
  - BIL share: ~29.5%
- Best compact result under headline financing: 7.07% CAGR, 0.508 Sharpe, -34.17% Max DD.
- No compact direct-allocation candidate beat SPY on both CAGR and Sharpe.
- No compact direct-allocation candidate reached 15% CAGR.

## Grid Validation Status

The two most recent branches generated `grid_validation_report.csv` and passed the reconciled-grid guardrail:

- `v4_2_model_family_comparison_2026-05-13`
- `v4_2_direct_allocation_model_family_2026-05-14`

Observed validation facts:

- Grid rows: 89,991
- Timestamp range: 2016-01-01 through 2026-04-07 14:00
- 2021-2025 evaluation rows: 43,824
- Original 14 V4 features present.
- VIX/VIX3M feature coverage:
  - `VIXCLS_z`: 100%
  - `vix3m_z`: 100%
  - `VIX_Inversion_Z`: 100%
- SPY/QQQ/BIL alignment passed.
- Benchmark/reference signal alignment passed.
- Label availability strict-PIT check passed.
- Defensive BIL price hash:
  `3440dd73d7cd4c4980995d770b504448a2a65f46995c14ea5949ebc8803140d2`
- B/C2 reference signal hash:
  `603b92155f46bcd45344f777f5732fa903b7342bd6608d5bee27abf871b78610`

## Package Availability Observed

From the compact direct-allocation run:

- XGBoost: available
- LightGBM: available, not used in compact default
- CatBoost: unavailable
- hmmlearn: unavailable

## Current Recommendation

Do not continue with defensive veto/meta-label targets as the primary research path. They are useful diagnostics but structurally too conservative for the stated objective.

Do not continue with conservative utility labels as the primary direct-allocation target. They also became too defensive and did not create enough QQQ/upside capture.

Keep current XGBoost live/primary.

## Next Research Leg

The next intended MUB-27 leg is a compact temporal model study using TCN / 1D CNN architectures. It should explicitly target upside capture first, then apply drawdown and financing guardrails after the model has identified risk-on opportunities.

Suggested direction:

1. Create a compact TCN / 1D CNN upside-capture research branch.
2. Keep the same grid-validation artifact and strict-PIT fold discipline.
3. Use short-window temporal pattern recognition over validated faithful-grid sequences:
   - 24h
   - 48h
   - 72h
   - 96h
4. Test targets that can plausibly produce 15-20% CAGR:
   - QQQ vs SPY forward relative-return spread labels.
   - Top-decile or top-quintile QQQ opportunity classifiers.
   - Learning-to-rank objective over QQQ/SPY/BIL, ranked by forward return first and risk penalty second.
   - Separate risk-on opportunity model and risk-control model rather than a single conservative utility label.
5. Target upside capture / direct allocation, not defensive veto only.
6. Explicitly measure upside capture:
   - QQQ exposure share.
   - QQQ hit rate.
   - QQQ/SPY spread capture.
   - Best-year and worst-year contribution.
   - Whether CAGR comes from broad consistency or one lucky year.
7. Only after upside capture exists, test:
   - BIL/SPY guardrails.
   - Minimum-hold smoothing.
   - 1.25x/1.5x/2.0x leverage with 10Y + 50 bps financing.
   - Seed and fold stability.

Suggested branch name:

`research/v4-2-compact-temporal-upside-capture-2026-05-14`

Suggested output root:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_compact_temporal_upside_capture_2026-05-14/`

Acceptance remains:

- Grid validation passes.
- Strict PIT manifest complete.
- CAGR > SPY CAGR.
- Sharpe > SPY Sharpe.
- Max DD >= -75%.
- No zero-equity breach.
- If leveraged, survives headline 10Y Treasury + 50 bps financing.
- Not dependent on one lucky seed/fold/year.
