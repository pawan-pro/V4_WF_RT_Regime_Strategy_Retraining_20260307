# 2026-06-19 Session Notes — No-BFS 0.28 Result and Persistent-Memory BFS Submission

## Session context

We continued ARC Prize 2026 / ARC-AGI-3 experiments from the validated FORGE V1 line.

Previous validated baseline:

```text
FORGE V1 instrumented baseline: 0.26
```

The instrumented baseline matched the clean FORGE V1 reproduction and confirmed that instrumentation itself was not degrading behavior.

Prior invalid regression:

```text
FORGE V2 / broken instrumentation: 0.10
Reason: accidental random-only fallback path replaced the original fallback behavior.
```

## What was done today

### 1. Recorded no-BFS ablation result

The user submitted the FORGE V1 no-BFS ablation to Kaggle scoring.

Result:

```text
FORGE V1 no-BFS ablation
Public Score: 0.28
Best Score: 0.28 V1
Approximate rank reported by user: ~330 / 1000
```

Comparison:

| Experiment | Public Score | Interpretation |
|---|---:|---|
| FORGE V1 instrumented baseline | 0.26 | Validated baseline with original BFS path preserved |
| FORGE V1 no-BFS ablation | 0.28 | Removing BFS improved score |
| Broken V2 random fallback | 0.10 | Invalid regression, not a valid ablation |

### 2. Interpreted the no-BFS result

The no-BFS result improved from `0.26` to `0.28`.

Working interpretation:

```text
Ungated BFS in the FORGE V1 implementation is likely net harmful in the current public scoring environment.
```

Possible causes:

- BFS consumes runtime/action budget without enough successful solves.
- BFS may search from unstable or incorrect state assumptions.
- BFS may create bad transfer/replay demonstrations.
- BFS may delay or interfere with the CNN/replay fallback path.
- Public evaluation may reward robust fallback behavior more than expensive source-search attempts.

Updated current best reproduced result:

```text
Current best reproduced score: 0.28
Current operational baseline: FORGE V1 no-BFS ablation
```

### 3. Analyzed new candidate notebook: 0.46 Persistent Memory BFS ARC Solver

The user shared a new notebook candidate:

```text
[0.46] ARC-AGI-3: Persistent Memory BFS ARC Solver
Version 3 of 8
Runtime: 13s
GPU: RTX Pro 6000
Public Score: 0.46
Best Score: 0.46 V3
```

Initial analysis:

This appears to be more than a pure BFS notebook. The likely important components are:

- BFS solver.
- Persistent Action Effect Memory / persistent AEM.
- Prioritized Experience Replay / PER.
- CLEAR-style cross-level replay archive.
- Test-time training / adaptation.

Interpretation:

```text
The score may come from persistent memory and replay/adaptation mechanisms, not from BFS alone.
```

This matters because our no-BFS ablation showed that ungated FORGE V1 BFS hurts. Therefore the key question is not simply whether BFS is good or bad, but whether persistent memory plus replay can make a BFS/fallback hybrid stronger.

### 4. User submitted the 0.46 candidate exactly

The user submitted the 0.46 Persistent Memory BFS notebook exactly as a replication test.

Current state:

```text
Experiment: 0.46 Persistent Memory BFS ARC Solver exact replication
Submission status: submitted for scoring
Score: pending
```

## Decision matrix for the 0.46 exact replication

| Replication score | Interpretation | Next action |
|---:|---|---|
| ~0.42–0.46 | Public score replicates; promote it to the new strongest baseline. | Create controlled ablations: no-BFS, no-PER, no-CLEAR, no-TTT. |
| ~0.28–0.35 | Some improvement, but public 0.46 does not fully reproduce. | Inspect runtime and determine which components are stable. |
| ~0.20–0.28 | Similar to current baseline; architecture does not reproduce in our environment. | Continue from 0.28 no-BFS line. |
| <0.20 | Likely dependency/runtime/hidden-state mismatch. | Inspect log and do not adopt. |
| Kaggle Error | Notebook wrapper/environment failure. | Fix wrapper only, no algorithm changes. |

## What worked

- No-BFS ablation improved score from 0.26 to 0.28.
- We now have a stronger reproduced operational baseline.
- The result gave a clear module-level signal: ungated FORGE V1 BFS is likely harmful.
- A stronger external candidate notebook was identified and submitted exactly, without modification, for clean replication.

## What failed / unresolved

- Scoring-time JSON diagnostics remain unavailable or unreliable from Kaggle rerun outputs.
- The exact 0.46 candidate score is pending.
- It is not yet known whether the 0.46 score is reproducible in the user's submitted environment.
- It is not yet known whether the improvement comes from BFS, persistent AEM, PER, CLEAR, TTT, or their interaction.

## Current best score/result

```text
Current best reproduced score: 0.28
Experiment: FORGE V1 no-BFS ablation
Approximate rank: ~330 / 1000
```

Pending:

```text
0.46 Persistent Memory BFS ARC Solver exact replication
Score: pending
```

## Files / artifacts referenced

User-uploaded candidate notebook:

```text
0-46-arc-agi-3-persistent-memory-bfs-arc-solver.ipynb
```

Session note created:

```text
session_notes/2026-06-19_arc_agi3_no_bfs_028_and_persistent_memory_submission.md
```

## Next session plan

1. Record the score of the exact 0.46 Persistent Memory BFS replication.
2. If it reproduces near 0.42–0.46, promote it to the new strongest baseline.
3. If it reproduces, run ablations in this order:
   - persistent-memory no-BFS
   - no-TTT
   - no-CLEAR
   - no-PER
4. If it does not reproduce, continue improving the 0.28 no-BFS baseline.
5. Avoid claiming architectural improvement until score is validated by Kaggle submission.
