# Symphony Operating Memory

Update rule for future chats:
- Update this file only when the Symphony/Linear/Codex operating workflow changes or when issue-state conventions change.
- Do not update this file for routine issue comments, transient Symphony runtime logs, or one-off debugging unless they change the durable workflow.
- This file complements `AGENTS.md` and `05_Context/04_Codex_Project_Memory.md`; it does not replace them.

## Purpose

This file gives Symphony and manual Codex sessions a compact context source for how the V4 project is using Linear, Symphony, ChatGPT Projects, Git, and Databricks together.

Use it to avoid confusing:

1. GitHub/local Git branch state
2. Databricks Repo branch/commit metadata
3. actual Databricks workspace notebook content executed by Workflows
4. Linear issue state
5. ChatGPT Project/manual Codex session notes

## Current Workflow Model

- Linear issues are task memory.
- Git commits and PRs are source-control memory.
- Session notes are project narrative and cross-issue memory.
- Symphony is orchestration/runtime memory.
- ChatGPT Project is the human review and synthesis room.

The current preferred workflow remains:

1. Use ChatGPT Project to produce daily/session notes.
2. Push session notes to Git under `session notes/`.
3. Use Linear comments for per-issue branch/commit/validation handoffs.
4. Use Symphony gradually for issue execution once the Linear board is clean.
5. Use ChatGPT Project for human review before moving production-sensitive issues to `Done`.

## Symphony Setup Summary

Local repo root:

```text
/Users/pawan/Mubama/Github/V4_WF_RT_Regime_Strategy_Retraining_20260307
```

Symphony checkout/binary:

```text
/Users/pawan/code/symphony/elixir/bin/symphony
```

Local runner:

```text
04_Orchestration/02_Symphony/run_symphony_local.sh
```

Workspace root:

```text
~/code/v4-strategy-symphony-workspaces
```

Tracker:

```text
Linear project slug: v4-strategy-5efe6af9374f
Team key: MUB
```

One issue should map to:

- one isolated Symphony workspace
- one branch
- one PR/review handoff when code changes are ready

Symphony/Codex should move code-ready work to human review. Do not rely on Symphony to auto-close production-sensitive issues.

## Linear Board Hygiene

Only executable work should enter `Todo`.

Recommended lane meanings:

- `Backlog`: idea / planning / not ready for Symphony
- `Todo`: ready for Symphony/Codex execution
- `In Progress`: actively being worked by Symphony/Codex or manual Codex
- `In Review`: human review required
- `Done`: accepted after validation
- `Canceled`: onboarding/noise/not needed

Important lesson from 2026-04-28:

- Symphony picked up Linear starter/onboarding issues (`MUB-1` through `MUB-4`) because they were eligible cards.
- This was not a production error, but it wasted orchestration attention.
- Move onboarding/tutorial issues to `Canceled` or `Done` before running Symphony unattended.

## Current Production State From MUB-10 Reconciliation

As of the 2026-04-28 MUB-10 reconciliation:

- Git `main`: `9b825eec918bbaeeb07bee0958e1b47d8ef1b547`
- Databricks Repo path: `/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github`
- Databricks Repo id: `2360562195327963`
- Databricks Repo branch/head: `feat/live-macro-phase1-pce-gdp-overrides` at `f8613656039839e01cdf7e8947def9c0ccc7823e`
- Actual workspace exports for:
  - `02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard`
  - `01_Processing/03_Realtime/02_RT_Live_Macro_Events`
  - `01_Processing/03_Realtime/03_RT_Silver_RT_Enrichment`
  match the MUB-7 branch, not `main`.

Interpretation:

- Databricks production workspace is currently ahead of Git `main`.
- This is not automatically a production failure, but it must be explicit in reviews.
- Any merge/release-alignment decision should happen through MUB-7 or a narrow follow-up issue/PR.

## Current Issue State After 2026-04-28 Human Review

Closed / Done:

- `MUB-9`: 06:00 SGT PDF job issue completed.
- `MUB-10`: Git/Databricks/workspace reconciliation ledger completed.
- `MUB-8`: CPI/UMCSENT issue closed as dashboard semantics/documentation fix; no inference-feed change required.

Open / Review:

- `MUB-7`: Phase 1 live macro expansion for `PCEPI`, `PCE`, and `GDPC1`; keep open until after 2026-04-30 18:00 IST / 12:30 UTC post-release validation.
- `MUB-6`: dashboard actual model macro input verification table; review after state alignment and dashboard semantics are clear.
- `MUB-11`: dashboard/PDF polish for live PIT wording, text overflow, and Times New Roman; related to MUB-9.
- `MUB-5`: broader dashboard institutional clarity bucket.

Noise / onboarding:

- `MUB-1` through `MUB-4` are Linear starter cards and should be canceled/done, not used as Symphony work.

## MUB-7 Closure Gate

Do not close MUB-7 until after the relevant release window and validation.

Target checkpoint:

```text
2026-04-30 18:00 IST / 12:30 UTC
```

Required validation:

1. Confirm actual Databricks workspace is still on the expected branch/state or explicitly record any drift.
2. Inspect `pawan_quant_research_1.v4.bronze_rt_macro_events_live` for `PCEPI`, `PCE`, and `GDPC1` rows.
3. Confirm no pre-release `observed_value` leak.
4. Confirm live values, if present, are semantically compatible with the model’s raw level/index lineage, not merely calendar percentage-change values.
5. Inspect latest `pawan_quant_research_1.v4.silver_rt_1h` for `PCEPI_z`, `PCE_z`, `GDPC1_z`, and `source`.
6. Confirm `source` includes a series only when a valid live override actually applied.
7. Inspect `pawan_quant_research_1.v4.bronze_rt_regime_signal` and confirm matching z-score values for the same timestamp.

If rows are absent or units mismatch, document and keep MUB-7 open or create a narrow follow-up. Do not patch production logic without review.

## Session Notes Policy

Two session-note locations exist:

- `session notes/`: current active daily/session note folder for recent April 2026 work.
- `05_Context/99_Session_Notes/`: older/legacy archive.

Use `session notes/` for new ChatGPT Project session notes unless a separate repo-structure issue deliberately changes this.

Do not paste every chat transcript into Linear. Keep durable cross-issue summaries in session notes and key per-issue handoffs in Linear comments.

## Production Guardrails Reminder

For all Symphony/Codex work, read `AGENTS.md` and `05_Context/04_Codex_Project_Memory.md` before touching Databricks jobs, Saxo auth, realtime pipelines, or workflow JSONs.

Non-negotiables:

- Saxo is primary.
- Alpaca is downstream simulation only.
- Only one automated Saxo token writer may be active.
- Live/downstream jobs should use existing cluster `0311-005159-pm2x3pce` unless cluster-create permissions are explicitly confirmed.
- Verify Git state, Databricks Repo state, and actual workspace notebook state separately before production-sensitive changes.
- Workflow JSONs and production notebooks are the operational source for schedules and task ordering.
- Avoid unnecessary refactors.
- Prefer smallest safe changes with explicit validation.
