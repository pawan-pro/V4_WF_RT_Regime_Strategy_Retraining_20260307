# Codex Project Memory

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
