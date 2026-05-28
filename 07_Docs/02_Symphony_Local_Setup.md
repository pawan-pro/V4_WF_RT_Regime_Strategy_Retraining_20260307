# Symphony Local Setup

This repo uses a minimal local Symphony setup for a solo-developer Linear workflow. Symphony polls one Linear project, creates one workspace per issue, and launches `codex app-server` in that workspace.
The workflow pins the Symphony-launched Codex runtime to `gpt-5.4` because Codex CLI `0.123.0` rejects the newer `gpt-5.5` default. It also sets medium reasoning effort and disables nonessential Codex app connectors for the Symphony worker so unattended runs use Symphony's own `linear_graphql` tracker tool instead of an interactive MCP elicitation flow.

## Required Environment

- `LINEAR_API_KEY`: required. Store it in your shell environment or a local secret manager; do not commit it.
- `SYMPHONY_BIN`: optional path to the Symphony executable. Defaults to `symphony` on `PATH`.
- `SYMPHONY_LOGS_ROOT`: optional logs directory. Defaults to `.symphony/logs` in this repo and is git-ignored.
- `SYMPHONY_PORT`: optional dashboard/API port. Leave unset to run without the local dashboard.

Codex must also be authenticated locally and available as `codex` on `PATH`.

## Start Locally

First build Symphony outside this repository:

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/openai/symphony
cd symphony/elixir
brew install mise # only needed if mise is not already installed
mise trust
mise install
mise exec -- mix setup
mise exec -- mix build
```

The OpenAI reference implementation recommends `mise`; Symphony Elixir requires Elixir `1.19+` and Erlang/OTP `28`. `mise install` reads Symphony's `mise.toml` and installs the pinned runtime versions. The build should create `./bin/symphony` in the `symphony/elixir` directory.

From the repository root:

```bash
export LINEAR_API_KEY='<your Linear API key>'
export SYMPHONY_BIN=~/code/symphony/elixir/bin/symphony
./04_Orchestration/02_Symphony/run_symphony_local.sh
```

The local runner detects this `mise`-managed Symphony checkout and runs the binary through `mise exec` when `escript` is not already active in your shell.
It also passes Symphony's required preview acknowledgement flag and trims accidental surrounding whitespace from `LINEAR_API_KEY` without printing the key.

If your shell is in another directory, use the absolute script path:

```bash
/Users/pawan/Mubama/Github/V4_WF_RT_Regime_Strategy_Retraining_20260307/04_Orchestration/02_Symphony/run_symphony_local.sh
```

## Issue State Mapping

- `Todo`: eligible for pickup.
- `In Progress`: active or resumable agent work.
- `Rework`: active follow-up after human review.
- `Human Review`: preferred handoff state when implementation is ready or human judgment is needed. This is terminal for Symphony so the agent stops instead of auto-closing the issue.
- `In Review`: alternate terminal handoff state for Linear teams that do not have `Human Review`.
- `Done`, `Closed`, `Cancelled`, `Canceled`, `Duplicate`: terminal states that stop orchestration for the issue.

The workflow intentionally does not auto-close issues. Code-ready work should have a branch and PR, then move to `Human Review`.

## Workspaces And Logs

- Workspaces live under `~/code/v4-strategy-symphony-workspaces`.
- Each issue gets its own cloned copy of this GitHub repo.
- Logs default to `.symphony/logs` unless `SYMPHONY_LOGS_ROOT` is set.
- `.symphony/` is ignored by Git.

## Defaults And Assumptions

- Concurrency is `1` because this is a solo-developer setup on one machine.
- Polling is every `60` seconds to avoid aggressive API churn.
- Retry backoff is capped at `5` minutes.
- Codex turn timeout is `90` minutes, stall timeout is `10` minutes, and each issue is capped at `6` Codex turns before retry/handoff policy takes over.
- Codex worker turns allow network access because Linear, GitHub, and Databricks validation may require outbound API calls.
- The Symphony worker keeps GitHub tooling available for private-repo reads and PR work, but disables Linear, browser, desktop, document, spreadsheet, and presentation app connectors to keep local runs leaner.
- The only workspace hook is `after_create`, which clones `https://github.com/pawan-pro/V4_WF_RT_Regime_Strategy_Retraining_20260307.git`.
- The hook assumes local Git credentials can clone and push branches to that GitHub remote.
- PR creation is handled by the agent inside the workspace using available GitHub tooling; Symphony remains the tracker reader and runner.
