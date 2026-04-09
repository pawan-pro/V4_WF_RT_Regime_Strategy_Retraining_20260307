# Saxo Shadow Cutover Runbook

## Purpose

This runbook defines the safe migration path from the current Saxo live job to the new V4 Saxo architecture on branch `codex/saxo-live-v4-shadow`.

## Current Safety Rules

- Keep the current live Saxo execution job active until the new path is validated.
- Keep the current token chain unchanged during phase 1.
- Do not disturb Monday manual OAuth unless explicitly required.
- The standalone refresher remains the only automated token writer.

## Token Chain

Current token state path:

`dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json`

Current manual bootstrap notebook:

`01_Processing/05_Utilities/08_RT_Saxo_OAuth_Bootstrap_LIVE`

Current standalone refresher notebook:

`01_Processing/05_Utilities/09_RT_Saxo_Token_Refresher_Standalone_Job`

## New Workflow Files

- `04_Orchestration/01_Workflows/04_V4_Saxo_Shadow_Hourly_Chain.json`
- `04_Orchestration/01_Workflows/05_V4_Saxo_Token_Refresher.json`
- `04_Orchestration/01_Workflows/06_V4_Saxo_Live_Hourly_Chain.json`

Important:

- All new Saxo jobs are created with `pause_status = PAUSED`
- Shadow validation should be run manually first
- The new refresher job should remain paused until final cutover

## Shadow Validation Flow

The new shadow hourly chain runs:

1. `RT_01_Live_Market_Ingestion`
2. `RT_02_Live_Rates_VIX_Ingestion`
3. `RT_03_Silver_RT_Enrichment`
4. `RT_04_Model_Inference_Engine`
5. `RT_08_RT_Saxo_Trade_Execution` in `shadow` mode
6. `RT_09_RT_Saxo_Signal_Audit_Log`
7. `RT_10_RT_Saxo_Dashboard_Feed`

Shadow mode reads the live Saxo account and logs account/position state, but places no orders.

## New Saxo Output Tables

- `pawan_quant_research_1.v4.bronze_rt_saxo_trade_log`
- `pawan_quant_research_1.v4.bronze_rt_saxo_signal_audit_log`
- `pawan_quant_research_1.v4.gold_rt_saxo_dashboard_feed`

## CLI Create Commands

```bash
cd /Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github
git pull origin codex/saxo-live-v4-shadow

WF_DIR="./04_Orchestration/01_Workflows"

databricks jobs create --json @"$WF_DIR/04_V4_Saxo_Shadow_Hourly_Chain.json"
databricks jobs create --json @"$WF_DIR/05_V4_Saxo_Token_Refresher.json"
databricks jobs create --json @"$WF_DIR/06_V4_Saxo_Live_Hourly_Chain.json"
```

## Validation Sequence

1. Create the new paused Saxo jobs
2. Manually run the shadow hourly chain
3. Confirm new Saxo tables are populated
4. Confirm no orders were placed by the shadow run
5. Confirm the existing live Saxo execution remains unaffected

## Final Cutover Sequence

1. Pause the current live Saxo execution job
2. Leave the current refresher running until the new live chain is proven
3. Manually run the new live Saxo hourly chain once
4. Confirm exactly one execution path is placing orders
5. If stable, enable the new live Saxo schedule
6. Only then consider switching the refresher to the new job definition
