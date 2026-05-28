# V4.2 Independent TFT Research Plan

This is the next-leg research plan after packaging Variant C. It is research-only and must not modify production Delta tables, Saxo execution, token refresh, RT notebooks, dashboard notebooks, workflows, schedules, or the model registry.

## Objective

Build an independent Temporal Fusion Transformer style model path that does not depend on XGBoost, recovered teacher labels, or teacher-student distillation.

## Inputs

Primary starting point:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/grids/full_legacy_grid_clone.csv`

Comparison references:

- SPY benchmark on identical price rows
- B recovered full-sample teacher replay control
- C full-sample teacher package
- D strict PIT teacher-student
- current production V4 signal as read-only reference, if available

## Model Independence Requirements

- No XGBoost dependency.
- No recovered teacher-label dependency.
- No full-sample teacher pseudo-labels.
- Labels must be defined from strict point-in-time future-return targets within each training fold only.
- Architecture and preprocessing must be reproducible from code and manifests.

## Validation Design

- Use strict rolling or expanding walk-forward validation.
- Each fold trains only on rows strictly before the validation/test period.
- Feature scaling, normalization, embedding/category maps, and any target transformations must be fit on the training window only.
- Persist fold membership, input hashes, model config, model checkpoints, predictions, and performance artifacts.

## Candidate TFT Setup

- Temporal encoder length: test a small grid such as 24h, 72h, 168h, 336h.
- Prediction horizon: align with existing next-week return regime target first, then test shorter horizons only as a labeled variant.
- Static covariates: none initially unless justified.
- Known time-varying inputs: calendar/hour/day features, event flags if PIT-safe.
- Observed time-varying inputs: 14 feature columns and SPY/QQQ price-derived features available at timestamp.
- Output: regime probabilities or direct allocation logits.

## Comparisons

Every run should report the same table format:

`Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switch Count | Signal Match vs B | Verdict`

Required comparison variants:

- SPY benchmark
- B recovered control
- C full-sample teacher package
- D strict PIT teacher-student
- production V4 read-only reference, if available

## Promotion Boundary

No promotion is allowed from this research leg. A model can only be marked as a future shadow candidate if it beats SPY on risk-adjusted metrics and has defensible turnover, drawdown, and slippage sensitivity under strict PIT validation.
