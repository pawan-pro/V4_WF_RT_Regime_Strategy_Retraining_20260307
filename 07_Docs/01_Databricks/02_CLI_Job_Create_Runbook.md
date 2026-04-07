# Databricks CLI Job Create Runbook

Use these commands from a Databricks workspace terminal to create the three V4 jobs from the checked-out repo.

## Current Operating Method

At the moment, the working pattern is:

1. Delete the old jobs in Databricks
2. Pull the latest repo updates in the workspace repo
3. Re-create the jobs from the workflow JSONs using Databricks CLI

This is the simplest and most reliable method currently in use.

## Workspace Repo

`/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github`

## Commands

```bash
cd /Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github
git pull origin codex-freeze-databricks-reorg
WF_DIR="./04_Orchestration/01_Workflows"

databricks jobs create --json @"$WF_DIR/01_V4_RT_Hourly_Chain.json"
databricks jobs create --json @"$WF_DIR/02_V4_Macro_Hourly_Maintenance.json"
databricks jobs create --json @"$WF_DIR/03_V4_Monthly_Retraining.json"
```

## Current Hourly Schedule

- `01_V4_RT_Hourly_Chain.json` runs at `5` minutes past every hour in `America/New_York`
- `02_V4_Macro_Hourly_Maintenance.json` runs at `15` minutes past every hour in `America/New_York`
- Each task now retries up to `2` times with a `5` minute wait between attempts and retries on timeout enabled

## Runtime Instrument Config

The live hourly chain now reads its active instrument mapping from:

`pawan_quant_research_1.v4.rt_runtime_instrument_config`

That means you can switch the live symbols later without editing the notebooks or the job JSONs.

Example: promote a futures or CFD-oriented mapping by inserting a newer config row:

```sql
INSERT INTO pawan_quant_research_1.v4.rt_runtime_instrument_config
VALUES (
  'futures_live',
  'ES,NQ',
  'ES',
  'NQ',
  'NQ',
  'ES',
  NULL,
  'provider_to_be_wired',
  'provider_to_be_wired',
  'Prepared runtime mapping for futures rollout',
  current_timestamp(),
  current_user()
);
```

The latest `updated_ts` row becomes active automatically on the next scheduled run.

## Delete Then Recreate

Delete the older jobs first, then run the create commands above.

## Alternative

There may be another way to do the same thing by updating or resetting the existing jobs instead of deleting and recreating them. That path is possible, but the current documented process is delete, pull, and recreate.

## UI Note

If the Databricks CLI terminal is not obvious in the workspace UI, open any notebook first and then open the terminal from the notebook/editor area.

## Workflows

- `01_V4_RT_Hourly_Chain.json`
- `02_V4_Macro_Hourly_Maintenance.json`
- `03_V4_Monthly_Retraining.json`
