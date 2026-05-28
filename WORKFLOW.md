---
tracker:
  kind: linear
  api_key: $LINEAR_API_KEY
  project_slug: v4-strategy-5efe6af9374f
  active_states:
    - Todo
    - In Progress
    - Rework
  terminal_states:
    - Human Review
    - In Review
    - Done
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
polling:
  interval_ms: 60000
workspace:
  root: ~/code/v4-strategy-symphony-workspaces
hooks:
  timeout_ms: 120000
  after_create: |
    git clone --depth 1 https://github.com/pawan-pro/V4_WF_RT_Regime_Strategy_Retraining_20260307.git .
agent:
  max_concurrent_agents: 1
  max_turns: 6
  max_retry_backoff_ms: 300000
  max_concurrent_agents_by_state:
    todo: 1
    in progress: 1
    rework: 1
codex:
  command: >-
    codex app-server
    -c model=\"gpt-5.4\"
    -c model_reasoning_effort=\"medium\"
    -c 'plugins."linear@openai-curated".enabled=false'
    -c 'plugins."computer-use@openai-bundled".enabled=false'
    -c 'plugins."browser-use@openai-bundled".enabled=false'
    -c 'plugins."documents@openai-primary-runtime".enabled=false'
    -c 'plugins."spreadsheets@openai-primary-runtime".enabled=false'
    -c 'plugins."presentations@openai-primary-runtime".enabled=false'
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
    networkAccess: true
  read_timeout_ms: 30000
  turn_timeout_ms: 5400000
  stall_timeout_ms: 600000
---
You are working on Linear issue `{{ issue.identifier }}`.

Issue context:
- Title: {{ issue.title }}
- Status: {{ issue.state }}
- URL: {{ issue.url }}
- Labels: {{ issue.labels }}

Description:
{% if issue.description %}{{ issue.description }}{% else %}No description provided.{% endif %}

{% if attempt %}
This is retry attempt #{{ attempt }}. Resume from the existing workspace state and do not repeat completed work unless the issue changed.
{% endif %}

Operational workflow:
1. Read the Linear issue carefully before changing code. Confirm the request, constraints, acceptance criteria, and any linked context.
2. Work only inside the assigned Symphony workspace. Do not touch unrelated local paths or shared production state.
3. Use one branch per issue, named with the issue identifier under the `codex/` prefix. Keep one PR per branch when code changes are ready.
4. Add meaningful Linear comments as progress notes: initial interpretation, material findings, validation results, blockers, and final handoff.
5. Move status deliberately: set `In Progress` only when actively working; use `Rework` only for review-driven fixes; move to `Human Review` when implementation is ready for a human. If `Human Review` does not exist for this Linear team, use `In Review`.
6. Create or update the PR when code is ready. Use the PR as the merge-tracking object; do not merge directly and do not auto-close the issue.
7. Run the smallest reliable validation for the change and report exactly what passed or could not be run.
8. Stop at a human-review handoff when confidence is insufficient, required secrets/permissions are missing, production risk is unclear, or acceptance criteria are ambiguous.
9. Keep investigation bounded: prefer `rg`, `git diff`, targeted file reads, and focused notebook cell extraction. Do not fetch or paste whole notebooks, large generated artifacts, or broad repo dumps unless the issue explicitly requires it.

Repository guardrails:
- Read `AGENTS.md` before changing Databricks jobs, Saxo auth, realtime pipelines, or workflow JSON.
- Preserve the Saxo token-writer and Databricks workflow ownership rules documented in `AGENTS.md`.
- Do not expose secrets in comments, commits, logs, prompts, or PR descriptions.
- Use Symphony's `linear_graphql` dynamic tool for Linear reads, comments, and status updates. Do not use Codex Linear app tools such as `_research`, because they can require interactive MCP elicitation and block unattended runs.
- For Linear issue lookup, use GraphQL `issue(id: "...")`, not `issueV2`.
- Use GitHub tooling only when it is needed for private-repo reads or PR work; prefer local Git commands when the workspace already has the necessary data.
