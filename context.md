# Symphony Local Orchestration Context

This repository has a minimal local Symphony setup for a solo-developer workflow using Linear as the issue tracker and Codex app-server as the coding-agent runtime. Keep this context secret-free when sharing it with a ChatGPT Project.

## Repository

- Local repo root: `/Users/pawan/Mubama/Github/V4_WF_RT_Regime_Strategy_Retraining_20260307`
- GitHub remote used by Symphony workspaces: `https://github.com/pawan-pro/V4_WF_RT_Regime_Strategy_Retraining_20260307.git`
- Repo-owned workflow policy: `WORKFLOW.md`
- Local setup doc: `07_Docs/02_Symphony_Local_Setup.md`
- Local runner: `04_Orchestration/02_Symphony/run_symphony_local.sh`
- Runtime scratch/log directory: `.symphony/` (git-ignored)

## Tracker

- Tracker kind: `linear`
- Linear project slug: `v4-strategy-5efe6af9374f`
- API key source: `LINEAR_API_KEY` environment variable only
- Never print, hardcode, commit, or paste the Linear API key into docs, prompts, logs, PRs, or comments.

Symphony is configured to use active Linear states:

- `Todo`
- `In Progress`
- `Rework`

Terminal / handoff states:

- `Human Review` preferred
- `In Review` fallback when the Linear team does not have `Human Review`
- `Done`
- `Closed`
- `Cancelled`
- `Canceled`
- `Duplicate`

The workflow intentionally avoids auto-closing issues. Code-ready work should move to a human-review state and use a PR as the merge-tracking object.

## Runtime

Symphony is built outside this repo:

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/openai/symphony
cd symphony/elixir
brew install mise # only if mise is missing
mise trust
mise install
mise exec -- mix setup
mise exec -- mix build
```

Expected Symphony binary:

```bash
/Users/pawan/code/symphony/elixir/bin/symphony
```

The local runner supports:

- `LINEAR_API_KEY`: required
- `SYMPHONY_BIN`: optional, defaults to `symphony` on `PATH`
- `SYMPHONY_LOGS_ROOT`: optional, defaults to `.symphony/logs`
- `SYMPHONY_PORT`: optional dashboard/API port, for example `4000`

Start from the repo root:

```bash
cd /Users/pawan/Mubama/Github/V4_WF_RT_Regime_Strategy_Retraining_20260307
export LINEAR_API_KEY='<set in shell, do not paste into shared context>'
export SYMPHONY_BIN=/Users/pawan/code/symphony/elixir/bin/symphony
export SYMPHONY_PORT=4000
./04_Orchestration/02_Symphony/run_symphony_local.sh
```

The runner:

- Requires `LINEAR_API_KEY`.
- Trims accidental surrounding whitespace without printing the key.
- Requires `codex` on `PATH`.
- Passes Symphony's preview acknowledgement flag.
- Uses `mise exec --` automatically when the Symphony checkout needs the mise-managed Elixir/Erlang runtime.

## Codex Worker Defaults

`WORKFLOW.md` launches:

```bash
codex app-server \
  -c model=\"gpt-5.4\" \
  -c model_reasoning_effort=\"medium\" \
  -c 'plugins."linear@openai-curated".enabled=false' \
  -c 'plugins."computer-use@openai-bundled".enabled=false' \
  -c 'plugins."browser-use@openai-bundled".enabled=false' \
  -c 'plugins."documents@openai-primary-runtime".enabled=false' \
  -c 'plugins."spreadsheets@openai-primary-runtime".enabled=false' \
  -c 'plugins."presentations@openai-primary-runtime".enabled=false'
```

Rationale:

- Codex CLI `0.123.0` rejected the local default `gpt-5.5`, so Symphony pins workers to `gpt-5.4`.
- Medium reasoning keeps local issue runs more cost-conscious.
- Nonessential app connectors are disabled because they increased tool-context size and one Linear connector path triggered an interactive MCP elicitation flow.
- GitHub tooling remains available for private-repo reads and PR work.

Sandbox and timeouts:

- `approval_policy: never`
- `thread_sandbox: workspace-write`
- `turn_sandbox_policy.type: workspaceWrite`
- `turn_sandbox_policy.networkAccess: true`
- `read_timeout_ms: 30000`
- `turn_timeout_ms: 5400000` (90 minutes)
- `stall_timeout_ms: 600000` (10 minutes)
- `agent.max_concurrent_agents: 1`
- `agent.max_turns: 6`
- `agent.max_retry_backoff_ms: 300000` (5 minutes)
- `polling.interval_ms: 60000` (60 seconds)

## Workspace Model

- Workspace root: `~/code/v4-strategy-symphony-workspaces`
- One Linear issue gets one isolated workspace.
- `after_create` hook clones the GitHub repo into that workspace:

```bash
git clone --depth 1 https://github.com/pawan-pro/V4_WF_RT_Regime_Strategy_Retraining_20260307.git .
```

Preferred delivery pattern:

- One issue -> one branch.
- Branch names should use the issue identifier under `codex/`, for example `codex/MUB-10`.
- One branch -> one PR when code changes are ready.
- PRs are merge-tracking objects; avoid direct blind merges.

## Agent Instructions

The agent prompt in `WORKFLOW.md` instructs Codex to:

- Read the Linear issue carefully before changing code.
- Work only inside the assigned Symphony workspace.
- Read `AGENTS.md` before touching Databricks jobs, Saxo auth, realtime pipelines, or workflow JSON.
- Preserve Saxo token-writer and Databricks workflow ownership rules.
- Add meaningful Linear comments for interpretation, progress, findings, validation, blockers, and handoff.
- Move issue status deliberately.
- Use `Human Review` when ready for review; use `In Review` if that is the available team state.
- Create or update a PR when code is ready.
- Stop for human review when confidence is insufficient, secrets/permissions are missing, production risk is unclear, or acceptance criteria are ambiguous.
- Keep investigations bounded with `rg`, `git diff`, targeted file reads, and focused notebook cell extraction.
- Avoid fetching or pasting entire notebooks, large generated artifacts, or broad repo dumps unless directly required.

Linear-specific instruction:

- Use Symphony's `linear_graphql` dynamic tool for Linear reads, comments, and status updates.
- Do not use Codex Linear app tools such as `_research`, because they can require interactive MCP elicitation and block unattended runs.
- For issue lookup, use GraphQL `issue(id: "...")`, not `issueV2`.

## Operational Notes From First Run

Observed and resolved setup issues:

- Symphony required the preview acknowledgement CLI flag; the local runner now passes it.
- `codex app-server` initially failed with the default `gpt-5.5`; `WORKFLOW.md` now pins `gpt-5.4`.
- The Codex Linear app connector triggered `mcpServer/elicitation/request`; the workflow now disables that connector and tells the agent to use `linear_graphql`.
- A test run on `MUB-10` reached the review handoff, posted Linear comments, and moved the issue to `In Review`.
- Token usage was high during that run because large connector/tool context and broad notebook/file reads were pulled into a single turn. The workflow now disables more nonessential app connectors, uses medium reasoning, caps turns at 6, and asks the agent to keep investigations bounded.

Dashboard/API:

- When running with `SYMPHONY_PORT=4000`, the dashboard is at `http://127.0.0.1:4000/`.
- Issue endpoints may return 404 after Symphony stops tracking an issue because it has reached a terminal state.

## Project Guardrails

Important durable rules from `AGENTS.md`:

- `v4_saxo_live_hourly_chain` is the primary hourly producer.
- `v4_rt_hourly_chain` is downstream-only Alpaca simulation and must consume Saxo-produced state.
- Do not reintroduce duplicate RT feature/signal generation into the Alpaca job.
- Only one automated job may write the live Saxo token file.
- Active Saxo token writer: `v4_saxo_token_refresher` (`233851783017472`).
- Legacy token refresher `V4.1-Token_Refresher_Job_SB` (`757971858926245`) must stay paused.
- Use existing Databricks cluster `0311-005159-pm2x3pce` for live/downstream jobs unless cluster-create permissions are confirmed.
- Verify Git repo branch, Databricks repo branch, and actual workspace notebook contents before changing live workflows.
- Realtime Delta writes use `06_Databricks/02_Scripts/03_Delta_Retry_Utils.py`; `%run` of that helper must come after `%pip` cells.

## Current Local Caveats

- `model_feature_upstream_lineage.csv` has a pre-existing unrelated local modification in the main repo. Do not revert or overwrite it unless explicitly requested.
- Symphony workspaces are separate clones under `~/code/v4-strategy-symphony-workspaces`; do not confuse workspace changes with main-repo setup changes.
- The runner and workflow files are intended for local single-machine orchestration only. Do not invent cloud infra, background services, or multi-host deployment unless the repo later adds that requirement.
