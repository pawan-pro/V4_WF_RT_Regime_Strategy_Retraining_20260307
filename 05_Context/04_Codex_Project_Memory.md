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

## Live Macro Findings To Carry Forward

- Feature branch `feat/rt-live-finnhub-core-macro-cpi-umcsent` adds Finnhub-first live macro capture for CPI/UMCSENT via `bronze_rt_macro_events_live` and workflow task `rt_01b_macro_events`.
- Durable guardrail from validation: before `scheduled_release_ts`, live macro rows must stay scheduled-only with null observed fields; the pre-release observation leak was real and is now fixed on that branch.
- CPI live capture/validation is operationally faster than the old batch-only path, but any RT feature impact is still gated by the next hourly RT bucket.
- `UMCSENT` live override is final-only; `Michigan Consumer Sentiment Prel` must fall back to historical silver carry-forward.
- Broader macro rollout and the model's future event-awareness path are deferred to a later session.
- Merge blocker from this session: the branch did not yet prove CPI override impact in `silver_rt_1h`; fallback and live CPI values matched on the tested release, and the explicit live z-score recompute path needs a second-pass review before production merge.

## Improved ALFRED Production State

- As of `2026-04-20`, production ALFRED selection uses the latest `observation_date` within the matched release bucket; the old first-row selection behavior is retired for the active production path.
- The corrected ALFRED production release promoted model version `bt_fred_1d_xgboost_regime_2026_04_20`; the live alias path remains `/Volumes/pawan_quant_research_1/v4/market_data/models/bt_fred_1d_xgboost_regime_live.joblib`.
- Release validation passed overall, but a historical silver PIT-join caveat remains open for `CPIAUCSL`, `GDPC1`, and `INDPRO`; RT silver alignment and dashboard validation passed.
- Post-release cleanup aligned local Git `main`, the Databricks repo branch `main`, and the repo-backed workspace path `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github` at repo id `2360562195327963`.
- For full release context and rollback anchors, read `2026-04-20_improved_alfred_model_update_bug_addressal_report.md`.

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
