# Databricks CLI Job Create Runbook

Use these commands from a Databricks workspace terminal to create the three V4 jobs from the checked-out repo.

## Workspace Repo

`/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github`

## Commands

```bash
cd /Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github
WF_DIR="./04_Orchestration/01_Workflows"

databricks jobs create --json @"$WF_DIR/01_V4_RT_Hourly_Chain.json"
databricks jobs create --json @"$WF_DIR/02_V4_Macro_Hourly_Maintenance.json"
databricks jobs create --json @"$WF_DIR/03_V4_Monthly_Retraining.json"
```

## UI Note

If the Databricks CLI terminal is not obvious in the workspace UI, open any notebook first and then open the terminal from the notebook/editor area.

## Workflows

- `01_V4_RT_Hourly_Chain.json`
- `02_V4_Macro_Hourly_Maintenance.json`
- `03_V4_Monthly_Retraining.json`
