# 2026-06-23 Session Notes — Loop/No-Change Pruning Experiment Running

## Session context

We continued ARC Prize 2026 / ARC-AGI-3 experiments from the current strongest reproduced baseline:

```text
Persistent Memory short/gated-BFS v1: 0.38
```

Current validated ladder before this session:

| Experiment | Public Score | Decision |
|---|---:|---|
| FORGE V1 instrumented | 0.26 | validated baseline |
| FORGE V1 no-BFS | 0.28 | useful ablation |
| Persistent Memory BFS exact replication | 0.30 | older baseline |
| Persistent Memory no-BFS | 0.17 | rejected |
| Persistent Memory short/gated-BFS v1 | 0.38 | current best |
| Persistent Memory gated-BFS v2 | 0.18 | rejected |

The `0.38` result moved the project to approximately rank `100 / 1355`, above the reported public bronze range and near the reported silver range. The user noted the public/private leaderboard caveat: public leaderboard uses approximately 50% of test data, while final results use the other 50%.

## What was reviewed today

### 1. Reviewed latest failed v2 score

Latest scoring result reviewed:

```text
notebook015c82ed33 - Version 1
Status: Succeeded
Public Score: 0.18
```

Interpretation:

- This was the Persistent Memory gated-BFS v2 experiment.
- It increased BFS budget relative to v1.
- The score collapsed from `0.38` to `0.18`.
- This result is close to the failed Persistent Memory no-BFS score of `0.17`, so v2 was rejected.

Conclusion:

```text
More BFS budget is not better. The validated v1 budget appears to be a narrow but effective operating point.
```

### 2. Reviewed ARC Prize 2026 / ARC-AGI-3 discussion and 30-day learnings

The user shared a Kaggle discussion asking about common ARC-AGI-3 approaches, with a reply from the competition host pointing to the ARC Prize 30-day learnings post.

Main takeaways from the post:

- ARC-AGI-3 is an interactive reasoning benchmark.
- Agents are evaluated on exploration, memory, goal acquisition, on-the-fly learning, and action efficiency.
- Action efficiency is foundational: score is not just about whether a level is completed, but how many actions are used compared with human baselines.
- Preview winners were mostly smart exploration agents rather than pure symbolic or LLM-only systems.
- Strong preview approaches included:
  - CNN action-learning agents that learn which actions cause frame changes.
  - state-graph agents that prune loops and no-change actions.
  - small value models or replay mechanisms that improve action selection after progress.

Interpretation for our work:

```text
The current Persistent Memory + gated-BFS + replay/adaptation line is aligned with the documented successful preview-agent direction.
```

The post supports prioritizing:

- action efficiency,
- no-change action pruning,
- short-loop detection,
- persistent memory,
- state/action-effect learning,
- restrained BFS rather than larger brute-force BFS.

## What was done today

### 1. Selected next experiment

Based on the `0.38` v1 success, the `0.18` v2 failure, and the ARC Prize learning post, we selected the next experiment:

```text
Persistent Memory short/gated-BFS v1 + loop/no-change pruning
```

Purpose:

```text
Improve action efficiency without changing the validated v1 BFS budget.
```

Design principle:

```text
Keep the validated 0.38 BFS timing. Add only a soft action-efficiency layer that discourages repeated no-change actions and short state-action loops.
```

### 2. Created updated notebook artifact

Generated notebook:

```text
persistent-memory-solver-v1-gated-bfs-v1-loop-pruning.ipynb
```

The notebook is based on the validated `0.38` short/gated-BFS v1 baseline.

It preserves:

- v1 BFS budget,
- BFS replay,
- CLTI / replay integration,
- PersistentAEM,
- PER,
- CLEAR,
- TTT,
- CNN fallback,
- heuristic fallback,
- Kaggle submission wrapper.

It adds a soft action-efficiency layer intended to:

- penalize actions that repeatedly cause no frame change,
- penalize state-action pairs that revisit short loops,
- softly downweight repeated useless clicks/actions,
- preserve the underlying fallback and BFS mechanisms.

### 3. User action

The user reported that the loop-pruning notebook is currently running.

Current state:

```text
Notebook: persistent-memory-solver-v1-gated-bfs-v1-loop-pruning.ipynb
Notebook status: running
Scoring status: pending
```

## Decision matrix for loop-pruning experiment

| Public score | Interpretation | Next action |
|---:|---|---|
| > 0.38 | Loop/no-change pruning improves action efficiency. | Promote as new baseline. |
| ~0.38 | Neutral; v1 remains strong. | Keep v1 or run repeat/control. |
| 0.30–0.37 | Pruning is too restrictive or noisy. | Reject or soften pruning. |
| < 0.30 | Pruning disrupted useful policy. | Reject and return to v1. |
| >= 0.41 | Public silver-zone candidate. | Promote cautiously and validate because public/private split may differ. |
| Kaggle Error | Wrapper or code issue. | Inspect log and patch wrapper only. |

## What worked

- We rejected the larger gated-BFS v2 after its `0.18` score.
- We used competition-host guidance and ARC Prize learning notes to choose an efficiency-focused experiment.
- We avoided changing the successful v1 BFS budget.
- We created a controlled loop/no-change pruning notebook based on the validated `0.38` baseline.

## What failed / unresolved

- Gated-BFS v2 failed badly at `0.18`.
- The current loop-pruning notebook score is pending.
- It is unknown whether the soft pruning layer will improve action efficiency or disrupt useful exploration.
- Offline evaluator calibration remains unresolved; the 501-action cap previously underpredicted the successful `0.38` run.

## Current best score/result

```text
Current best reproduced public score: 0.38
Current best baseline: Persistent Memory short/gated-BFS v1
Approximate public rank: ~100 / 1355
```

Pending experiment:

```text
Persistent Memory short/gated-BFS v1 + loop/no-change pruning
Status: running / scoring pending
```

## Files / artifacts referenced

Generated notebook artifact:

```text
persistent-memory-solver-v1-gated-bfs-v1-loop-pruning.ipynb
```

Session note created:

```text
session_notes/2026-06-23_loop_pruning_experiment_running.md
```

## Next session plan

1. Wait for loop-pruning notebook run and public scoring result.
2. Compare against current best `0.38`.
3. If improved, promote loop-pruning as new baseline.
4. If neutral, keep v1 and consider repeat/control submission.
5. If worse, reject pruning and return to v1.
6. Avoid further BFS-budget expansion unless supported by controlled evidence.
7. Continue focusing on action efficiency, memory, and loop/no-change pruning rather than brute-force BFS expansion.
