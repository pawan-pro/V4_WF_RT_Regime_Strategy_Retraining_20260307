# Project Operating Memory

Update rule for future chats:
- At the end of any chat session, update this file only if a durable project rule or operating constraint changed since the last update.
- Do not update this file for temporary debugging, one-off run results, or minor implementation detail changes.
- If nothing durable changed, leave this file unchanged.

Read this before changing Databricks jobs, Saxo auth, or realtime pipelines in this repo.

## Core Ownership

- `v4_saxo_live_hourly_chain` is the primary hourly producer.
- `v4_rt_hourly_chain` is downstream-only Alpaca sim and must consume Saxo-produced state.
- Do not reintroduce duplicate RT feature/signal generation into the Alpaca job.

## Token Safety

- Only one automated job may write the live Saxo token file:
  `dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json`
- Active writer: `v4_saxo_token_refresher` (`233851783017472`)
- Legacy writer that must stay paused: `V4.1-Token_Refresher_Job_SB` (`757971858926245`)
- Multiple token refresh jobs against the same file can rotate refresh tokens out from under each other and cause `401` / invalid-session failures.

## Job Topology

- `v4_saxo_live_hourly_chain` schedule: `:01` every hour, `America/New_York`
- `v4_rt_hourly_chain` schedule: `:06` every hour, `America/New_York`
- `v4_saxo_token_refresher` schedule: every 10 minutes
- `v4_saxo_institutional_dashboard_daily_pdf` schedule: `06:00` daily, `Asia/Singapore`

## Compute Constraint

- Use existing cluster `0311-005159-pm2x3pce` for the live/downstream jobs.
- Do not switch the downstream Alpaca job back to a new job cluster unless cluster-create permissions are confirmed.

## Execution Guard

- Current live execution logic assumes ETF trading hours.
- Off-hours runs should still compute features/signals, but order placement is skipped outside regular market hours.
- If execution shifts to futures or CFDs, update that market-hours guard.

## Delta Write Safety

- Realtime Delta writes use `06_Databricks/02_Scripts/03_Delta_Retry_Utils.py`.
- `%run` of that helper must come after any `%pip` cells in notebooks that use it.

## Databricks Repo

- Workspace repo path: `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github`
- Repo id: `2360562195327963`
- Git repo branch, Databricks repo branch, and actual workspace notebook contents can diverge during validation; verify all three before changing live workflows.

## Live Macro Guardrails

- `bronze_rt_macro_events_live` is the live macro capture table for Finnhub-first schedule/value rows used by the Saxo hourly chain.
- Before `scheduled_release_ts`, live macro rows must remain scheduled-only: `status=scheduled`, `observed_value=null`, `observed_timestamp=null`, `validation_status=not_checked`.
- Live macro override Phase 1 expands scope to `PCEPI`, `PCE`, and `GDPC1`, preserving `CPIAUCSL` and final-only `UMCSENT` guardrails.
- `UMCSENT` live override is final-only. Preliminary Michigan sentiment releases must not override RT silver.

## Change Discipline

- Before enabling or creating any new refresher, execution, or RT job, check for an existing job that already owns that function.
- Before changing schedules, verify there is exactly one writer per Delta table and one writer per Saxo token state file.
