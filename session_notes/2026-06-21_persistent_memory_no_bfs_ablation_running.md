# 2026-06-21 Session Notes — Persistent-Memory No-BFS Ablation Running

## Session context

We continued ARC Prize 2026 / ARC-AGI-3 experiments from the current best reproduced line.

Validated experiment ladder before this session:

```text
FORGE V1 instrumented baseline:        0.26
FORGE V1 no-BFS ablation:              0.28
Persistent Memory BFS exact replication: 0.30
```

The earlier public reference for the persistent-memory solver advertised `0.46`, but our exact replication scored `0.30`. This was treated as a positive partial replication and promoted to the current best reproduced baseline.

## What was done today

### 1. Interpreted the persistent-memory exact replication result

The uploaded / copied notebook:

```text
[0.46] ARC-AGI-3: Persistent Memory BFS ARC Solver
```

scored in our run:

```text
Public Score: 0.30
```

Interpretation:

- The advertised `0.46` did not reproduce.
- The notebook still improved over the `0.28` FORGE V1 no-BFS baseline.
- The current best reproduced score became `0.30`.
- The result should be treated as a valid score, not a Kaggle error, because the notebook succeeded.

Working hypothesis:

```text
The improvement may come from persistent memory / replay / adaptation components rather than BFS itself.
```

Key components suspected to matter:

- Persistent Action Effect Memory / PersistentAEM.
- Prioritized Experience Replay / PER.
- CLEAR-style cross-level replay archive.
- Test-time training / TTT.
- CNN fallback and heuristic fallback behavior.

### 2. Selected next ablation

We selected the next controlled experiment:

```text
Persistent Memory Solver — no-BFS ablation
```

Purpose:

```text
Measure whether the 0.30 score comes from BFS or from the persistent memory / replay / adaptation stack.
```

This mirrors the previous FORGE V1 finding:

```text
FORGE V1 full/instrumented: 0.26
FORGE V1 no-BFS:           0.28
```

That previous result showed that ungated FORGE V1 BFS was net harmful. The current experiment tests whether the same is true inside the stronger persistent-memory architecture.

### 3. Generated updated notebook artifact

Created downloadable ablation notebook:

```text
persistent-memory-solver-v1-no-bfs-ablation.ipynb
```

The ablation is intended to disable:

- BFSSolver initialization.
- BFS solve attempts.
- BFS replay actions.
- CLTI injection from BFS expert demonstrations.

The ablation is intended to preserve:

- PersistentAEM.
- PER / prioritized replay.
- CLEAR archive.
- TTT / test-time adaptation.
- CNN fallback.
- heuristic fallback.
- normal Kaggle submission wrapper.

### 4. Current user action

The user reported that the notebook is currently running in Kaggle.

Current state:

```text
Persistent Memory Solver no-BFS ablation: running in Kaggle notebook
Submission for scoring: planned after notebook run completes
Public score: pending
```

## Decision matrix for persistent-memory no-BFS score

| Score | Interpretation | Next action |
|---:|---|---|
| > 0.30 | BFS is hurting even in the persistent-memory architecture. | Promote no-BFS persistent-memory line; consider tuning persistent memory / replay. |
| ~0.30 | BFS is mostly neutral; persistent memory / replay stack is likely the main driver. | Keep simpler no-BFS version or test no-TTT / no-CLEAR / no-PER. |
| 0.26–0.29 | BFS contributes some value in this architecture. | Consider gated or short-BFS instead of fully removing BFS. |
| < 0.25 | BFS is important for this solver. | Revert to persistent-memory BFS and profile / gate BFS more carefully. |
| Kaggle Error | Ablation broke wrapper or code path. | Inspect log and fix wrapper only before algorithm changes. |

## What worked

- We promoted the exact persistent-memory replication result to the current best reproduced baseline at `0.30`.
- We avoided overclaiming the advertised `0.46`, because it did not reproduce.
- We created a targeted no-BFS ablation for the stronger persistent-memory solver.
- The user started running the ablation notebook and plans to submit after the run completes.

## What failed / unresolved

- The advertised `0.46` score did not replicate; our reproduced score was `0.30`.
- The cause of the `0.46` versus `0.30` gap remains unknown.
- It is not yet known whether BFS helps, hurts, or is neutral in the persistent-memory architecture.
- Scoring-time JSON diagnostics remain unavailable / unreliable, so score-delta ablations remain the primary evaluation method.

## Current best score/result

```text
Current best reproduced public score: 0.30
Current best reproduced baseline: Persistent Memory BFS exact replication
```

Pending experiment:

```text
Persistent Memory Solver no-BFS ablation
Notebook status: running
Scoring status: pending
```

## Files / artifacts referenced

Generated notebook artifact:

```text
persistent-memory-solver-v1-no-bfs-ablation.ipynb
```

Session note created:

```text
session_notes/2026-06-21_persistent_memory_no_bfs_ablation_running.md
```

## Next session plan

1. Wait for Kaggle notebook run completion.
2. Submit the persistent-memory no-BFS ablation for scoring.
3. Record public score.
4. Compare against the current `0.30` persistent-memory BFS baseline.
5. If no-BFS improves or ties, focus next on persistent memory / replay ablations:
   - no-TTT
   - no-CLEAR
   - no-PER
6. If no-BFS drops, design a gated-BFS or short-BFS version rather than fully removing BFS.
