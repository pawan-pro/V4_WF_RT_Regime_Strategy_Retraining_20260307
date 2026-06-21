# 2026-06-21 Session Notes — Persistent-Memory Short/Gated-BFS Offline Warning

## Session context

We continued ARC Prize 2026 / ARC-AGI-3 experiments from the persistent-memory solver line.

Validated experiment ladder before this session:

```text
FORGE V1 instrumented baseline:          0.26
FORGE V1 no-BFS ablation:                0.28
Persistent Memory BFS exact replication: 0.30
Persistent Memory no-BFS ablation:       0.17
```

Interpretation before today:

- In FORGE V1, ungated BFS appeared harmful because removing it improved score from `0.26` to `0.28`.
- In the Persistent Memory solver, BFS appeared important because removing it dropped score from `0.30` to `0.17`.
- Therefore, BFS is not universally good or bad; its value depends on interaction with memory, replay, and fallback learning.

## What was done today

### 1. Selected short/gated-BFS experiment

We chose the next experiment:

```text
Persistent Memory short/gated-BFS ablation
```

Purpose:

```text
Keep the useful BFS signal while reducing wasted search.
```

Design intent:

- Keep BFS enabled.
- Keep BFS transfer from previous level.
- Keep BFS solution replay.
- Keep BFS expert demonstrations.
- Keep CLTI / replay integration.
- Keep PersistentAEM.
- Keep PER.
- Keep CLEAR archive.
- Keep TTT.
- Keep CNN fallback.
- Reduce only expensive BFS budget.

Approximate intended budget changes:

```text
scan_timeout: 5s  -> 2s
bfs_timeout: 180s -> 35s
max_states: 500000 -> 75000
max_depth: 30 -> 18
hidden retry capped within remaining BFS budget
```

### 2. Created scoring notebook

Generated notebook artifact:

```text
persistent-memory-solver-v1-short-gated-bfs-ablation.ipynb
```

The user submitted this notebook to Kaggle scoring.

Current scoring status:

```text
Persistent Memory short/gated-BFS: submitted for scoring
Public score: pending
```

### 3. Created offline evaluator notebook

Generated offline evaluator notebook:

```text
arc3-offline-evaluator-persistent-memory-short-gated-bfs.ipynb
```

Purpose:

```text
Run the short/gated-BFS agent offline through the official main.py path and produce behavior diagnostics.
```

Expected artifacts:

```text
summary.csv
summary.txt
scorecard.json
run.log
recordings/
diagnostic_summary.csv
```

### 4. Fixed offline evaluator action cap

The first offline evaluator run showed all games still running after more than 1,000 actions because the agent had:

```python
class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10
```

This meant no practical action cap, causing long runs and large recordings.

We decided to use a diagnostic cap:

```python
MAX_ACTIONS = 501
```

This made the offline run comparable to earlier diagnostics and prevented runaway recordings.

### 5. Recorded offline evaluator result

The capped offline evaluator completed all 25 games.

Offline result:

```text
Overall offline score: 0.19047619047619047
Finished: 25/25
Running: 0
Nonzero-score games: 1 / 25
Games with completed levels: 1 / 25
Games at/near 501 actions: 25 / 25
```

Only game with progress:

```text
sp80
levels_completed: 1
action_count: 502
score: 4.761905
state: done
```

All other games showed:

```text
levels_completed: 0
score: 0.0000
action_count: 502
state: done
```

## Interpretation

The offline result is a warning signal against the short/gated-BFS patch.

Working interpretation:

```text
The short/gated-BFS patch may be too aggressive and may be cutting off useful BFS traces before PersistentAEM / replay / CLEAR can benefit.
```

Likely failure mode:

- BFS is still structurally enabled.
- But the reduced depth/state/time budgets may prevent it from generating useful complete or partial demonstrations.
- Without enough useful BFS-generated experience, the persistent-memory/replay system behaves closer to the failed no-BFS ablation.

This offline result is closer to the `0.17` no-BFS failure than to the `0.30` persistent-memory BFS baseline.

## Current decision

Do not promote short/gated-BFS yet.

Current best remains:

```text
Persistent Memory BFS exact replication: 0.30
```

But do not cancel the Kaggle scoring submission. Wait for official scoring before final decision.

## Decision matrix for pending short/gated-BFS score

| Public score | Interpretation | Next action |
|---:|---|---|
| > 0.30 | Offline evaluator underestimated; short/gated-BFS improves over full BFS. | Promote as new baseline and inspect why offline disagreed. |
| ~0.30 | Full BFS budget was not the main issue. | Keep full/persistent BFS baseline or test mild gating. |
| 0.20–0.29 | Gating is too aggressive but not catastrophic. | Try medium-BFS budget. |
| ~0.17–0.20 | Confirms offline warning; patch behaves close to no-BFS. | Reject short/gated-BFS and restore more BFS. |
| Kaggle Error | Wrapper or notebook issue. | Inspect log and fix wrapper only. |

## Recommended next experiment after scoring

If public score confirms the offline warning, the next experiment should be:

```text
Persistent Memory medium-BFS ablation
```

Proposed less aggressive budget:

```text
scan_timeout: 5s -> 3s
bfs_timeout: 180s -> 75s or 90s
max_states: 500000 -> 200000
max_depth: 30 -> 24 or 26
```

Preserve:

- BFS replay.
- CLTI.
- PersistentAEM.
- PER.
- CLEAR.
- TTT.
- CNN fallback.

## What worked

- We created and submitted the short/gated-BFS scoring notebook.
- We created an offline evaluator version for behavior diagnostics.
- We identified and corrected the infinite-action offline evaluation problem.
- The capped offline evaluator produced a clean diagnostic result.
- The offline result provided a useful warning before the public score returned.

## What failed / unresolved

- Offline short/gated-BFS score was weak: `0.190476`.
- Only one game, `sp80`, showed progress.
- All games hit the action cap, indicating widespread stuck behavior.
- Public score for short/gated-BFS is still pending.
- We still need to compare with an offline run of the full Persistent Memory BFS baseline using the same 501 action cap.

## Current best score/result

```text
Current best reproduced public score: 0.30
Current best baseline: Persistent Memory BFS exact replication
```

Pending:

```text
Persistent Memory short/gated-BFS public score: pending
```

Offline diagnostic result:

```text
Persistent Memory short/gated-BFS offline score: 0.190476
```

## Files / artifacts referenced

Generated scoring notebook:

```text
persistent-memory-solver-v1-short-gated-bfs-ablation.ipynb
```

Generated offline evaluator notebook:

```text
arc3-offline-evaluator-persistent-memory-short-gated-bfs.ipynb
```

Offline run directory:

```text
/kaggle/working/runs-20260621-165734-persistent-memory-short-gated-bfs-v1
```

Offline artifacts:

```text
summary.csv
scorecard.json
run.log
recordings/
diagnostic_summary.csv
```

Session note created:

```text
session_notes/2026-06-21_persistent_memory_short_gated_bfs_offline_warning.md
```

## Next session plan

1. Wait for Kaggle public score for short/gated-BFS.
2. Compare public score with offline evaluator result.
3. If public score is weak, reject short/gated-BFS as too aggressive.
4. Run offline evaluator on the full Persistent Memory BFS baseline with `MAX_ACTIONS = 501` for direct game-level comparison.
5. If short/gated-BFS fails, create a medium-BFS ablation with less aggressive budgets.
6. Continue to avoid claims of improvement unless validated by Kaggle score or controlled offline comparison.
