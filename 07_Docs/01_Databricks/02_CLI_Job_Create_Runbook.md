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
