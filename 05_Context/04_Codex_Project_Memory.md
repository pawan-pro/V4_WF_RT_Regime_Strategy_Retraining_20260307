# Codex Project Memory

Update rule for future chats:
- At the end of any chat session, update this file if any relevant operating fact changed since the last update.
- Relevant changes include job ids, schedules, active/paused ownership, token-writer ownership, repo id/path/branch, cluster strategy, or other durable workflow decisions.
- Do not update this file for transient logs, temporary failures, or routine successful runs unless they changed the durable operating state.

This file exists so future Codex chats can recover the operating context for this repo quickly.

## Non-Negotiable Invariants

1. Saxo is primary. Alpaca is downstream sim only.
2. Only one token refresher may be active for the live Saxo token file.
3. Realtime job changes must be checked against Databricks job ids, schedules, and existing-cluster requirements before enabling anything.

## Current Live Job Mapping

- `764399962399671` `v4_saxo_live_hourly_chain`
- `956566325960795` `v4_rt_hourly_chain`
- `233851783017472` `v4_saxo_token_refresher`
- `757971858926245` `V4.1-Token_Refresher_Job_SB` -> must remain paused
- `516523889858332` `v4_saxo_institutional_dashboard_daily_pdf` -> daily 06:00 Asia/Singapore PDF report job
- `559657131456932` `v4_saxo_institutional_circulation_report_daily_pdf` -> daily 06:00 Asia/Singapore circulation PDF report job
- Avik (`avik@mubama.group`) has job-level `CAN_MANAGE_RUN` access to both institutional dashboard PDF jobs. Keep Pawan's single-user cluster `0311-005159-pm2x3pce` unchanged for notebook execution; Avik's supported path is triggering the jobs or using separate dashboard-safe compute.

## Live Macro Findings To Carry Forward

- Finnhub-first live macro capture flows through `bronze_rt_macro_events_live` and workflow task `rt_01b_macro_events`.
- Durable guardrail from validation: before `scheduled_release_ts`, live macro rows must stay scheduled-only with null observed fields; the pre-release observation leak was real and is now fixed on that branch.
- Finnhub calendar rows for `PCEPI`, `PCE`, and `GDPC1` are release-detector/diagnostic only because they are rate/percent values, not FRED/ALFRED level values; they must not override RT silver until a same-semantics level source is implemented.
- `UMCSENT` live override is final-only; `Michigan Consumer Sentiment Prel` must fall back to historical silver carry-forward.

## Improved ALFRED Production State

- As of `2026-04-20`, production ALFRED selection uses the latest `observation_date` within the matched release bucket; the old first-row selection behavior is retired for the active production path.
- The corrected ALFRED production release promoted model version `bt_fred_1d_xgboost_regime_2026_04_20`; the live alias path remains `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`.
- Release validation passed overall, but a historical silver PIT-join caveat remains open for `CPIAUCSL`, `GDPC1`, and `INDPRO`; RT silver alignment and dashboard validation passed.
- Post-release cleanup aligned local Git `main`, the Databricks repo branch `main`, and the repo-backed workspace path `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github` at repo id `2360562195327963`.
- For full release context and rollback anchors, read `2026-04-20_improved_alfred_model_update_bug_addressal_report.md`.

## Official V4 Backtest Baseline

- As of `2026-04-30`, the official V4 backtest / management-reporting baseline is the promoted corrected-ALFRED recreated-legacy variant.
- The promoted slide-ready baseline reproduces from the corrected artifact reporting layer at `20.8405%` CAGR, `1.0024` Sharpe, `1.3729` Sortino, `-25.5385%` max drawdown, and final equity `12883454.99` for `2021-01-01` to `2025-12-31`.
- This baseline promotion changed canonical reporting artifacts only: `pawan_quant_research_1.v4.bt_walkforward_regime_signals`, `/Volumes/pawan_quant_research_1/v4/market_data/bt_walkforward_regime_signals.csv`, and `/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_regime_signals-v4.csv`.
- `pawan_quant_research_1.v4.silver_bt_master_grid_1h` was **not** overwritten because the corrected recreated-legacy silver lineage is not schema-compatible with the canonical alias (`QQQ_close` is missing). The promoted production-owned preserved copy is `pawan_quant_research_1.v4.silver_bt_master_grid_1h_corrected_alfred_baseline_20260430`.
- The canonical walk-forward signal alias was rebuilt in compatibility form from corrected walk-forward predictions plus corrected labels so downstream analysis expecting `timestamp, regime, predicted_regime` continues to work.
- Pre-promotion rollback anchors were captured under suffix `pre_corrected_alfred_promotion_20260430`; see `/Volumes/pawan_quant_research_1/v4/market_data/reports/v4_corrected_alfred_promotion_manifest_2026-04-30.json`.
- `model_registry_current` was not edited on `2026-04-30`; it already pointed at confirmed model version `bt_fred_1d_xgboost_regime_2026_04_20`, and live Saxo execution / token refresh / RT schedules were left unchanged.

## Why This Exists

Previous chats created overlapping token refreshers and overlapping RT writers. That caused:

- Delta concurrency failures in Saxo live runs
- unnecessary risk of Saxo token rotation conflicts
- cluster permission failures when the downstream Alpaca job tried to create a new cluster

## Required Checks Before Any Job Change

1. Is there already another job writing the same table or token file?
2. Is the job using the correct existing cluster?
3. Is the schedule aligned with the primary/downstream split?
4. If touching order placement logic, does the execution universe still match ETF trading hours?

## Recommended Prompt For Future Codex Chats

Start by reading `AGENTS.md` and `05_Context/04_Codex_Project_Memory.md` before proposing any Databricks workflow or Saxo auth changes.
