# 2026-06-22 Session Notes — Gated-BFS 0.38 Result and v2 Submission Plan

## Session context

We continued ARC Prize 2026 / ARC-AGI-3 experiments from the Persistent Memory solver line.

Experiment ladder entering the session:

```text
FORGE V1 instrumented baseline:          0.26
FORGE V1 no-BFS ablation:                0.28
Persistent Memory BFS exact replication: 0.30
Persistent Memory no-BFS ablation:       0.17
Persistent Memory short/gated-BFS v1:    submitted, score pending
```

Earlier offline evaluator signal for short/gated-BFS v1 was weak:

```text
Offline short/gated-BFS score: 0.190476
Nonzero-score games: 1 / 25
Only progress game: sp80
MAX_ACTIONS cap: 501
```

The offline result suggested that short/gated-BFS might be too aggressive. We decided to wait for the official Kaggle public score before rejecting it.

## What was done today

### 1. Recorded official public score for short/gated-BFS v1

The user reported the Kaggle scoring result:

```text
Submission: notebook3c4cc85e1e - Version 1
Status: Succeeded
Public Score: 0.38
```

This is the strongest reproduced result so far.

Updated ladder:

| Experiment | Public Score | Status |
|---|---:|---|
| FORGE V1 instrumented | 0.26 | validated baseline |
| FORGE V1 no-BFS | 0.28 | improved over FORGE full BFS |
| Persistent Memory BFS exact replication | 0.30 | previous best |
| Persistent Memory no-BFS | 0.17 | rejected |
| Persistent Memory short/gated-BFS v1 | 0.38 | new best |

Current best reproduced public score:

```text
0.38
```

### 2. Interpreted offline-vs-public discrepancy

Offline evaluator result:

```text
0.190476
```

Official public score:

```text
0.38
```

Interpretation:

- The offline evaluator with `MAX_ACTIONS = 501` was too strict for this agent.
- The official scoring path allowed enough runtime/action behavior for short/gated-BFS to outperform the full Persistent Memory BFS baseline.
- Offline evaluator remains useful for qualitative debugging, but its action cap needs calibration.

Working conclusion:

```text
Short/gated-BFS preserved enough useful BFS signal while reducing wasteful full-BFS overhead.
```

This reversed the earlier offline warning and validated the short/gated-BFS direction.

### 3. Updated competitive context

User reported current leaderboard context:

```text
Current rank: ~100 / 1355
Current score: 0.38
Bronze range: up to approximately 0.35
Silver ends around 0.41
Gold around 0.59
```

Top five leaderboard scores reported:

```text
1. Tufa Labs:          1.21
2. Tong Hui Kang:      0.70
3. Redfield Rentals:   0.68
4. Barada Sahu:        0.66
5. Kevin E R MILLE:    0.66
```

Important leaderboard caveat:

```text
Public leaderboard is calculated with approximately 50% of the test data. Final results will be based on the other 50%, so final standings may differ.
```

Interpretation:

- The project is now above the reported bronze range on the public leaderboard.
- The score is close to the silver band but not yet safely silver.
- Because of the public/private split, we should avoid overfitting to public score and maintain reproducible experiment notes.

### 4. Created next experiment: gated-BFS v2

We created the next notebook:

```text
persistent-memory-solver-v1-gated-bfs-v2.ipynb
```

Purpose:

```text
Test whether modestly increasing BFS budget from v1 can improve beyond 0.38 and approach or exceed the silver-score region around 0.41.
```

Budget comparison:

| Parameter | Full Persistent BFS | Short/gated v1 | New gated v2 |
|---|---:|---:|---:|
| scan_timeout | 5s | 2s | 3s |
| bfs_timeout | 180s | 35s | 60s |
| max_states | 500,000 | 75,000 | 125,000 |
| max_depth | 30 | 18 | 22 |

Design principle:

```text
Increase BFS budget slightly versus v1, but do not return to full 180s / 500k / depth 30 BFS.
```

Preserve:

- BFS enabled.
- BFS replay.
- CLTI / replay integration.
- PersistentAEM.
- PER.
- CLEAR.
- TTT.
- CNN fallback.
- heuristic fallback.

## Decision matrix for gated-BFS v2

| v2 public score | Interpretation | Next action |
|---:|---|---|
| > 0.38 | More BFS budget helps; v1 was slightly too tight. | Promote v2 as new baseline. |
| ~0.38 | v1 budget is near optimal. | Keep v1; test orthogonal ablations. |
| 0.30–0.37 | v2 adds wasteful BFS overhead. | Return to v1. |
| < 0.30 | v2 disrupts useful policy. | Reject v2. |
| >= 0.41 | Enters reported silver-score region on public LB. | Promote but validate with caution due to public/private split. |
| Kaggle Error | Notebook wrapper/environment issue. | Inspect log and patch wrapper only. |

## What worked

- Official Kaggle score validated the short/gated-BFS direction.
- Current best improved from `0.30` to `0.38`.
- Leaderboard position improved to approximately `100 / 1355`.
- The project is above the reported bronze public-score threshold.
- We created a controlled gated-BFS v2 notebook rather than making broad architecture changes.

## What failed / unresolved

- Offline evaluator with `MAX_ACTIONS = 501` underpredicted public score severely.
- Need to calibrate offline evaluator caps, likely testing `501`, `1000`, and `2000` action caps.
- The final/private leaderboard may differ from public LB because public uses about 50% of test data.
- It is not yet known whether v2 improves over v1 or overfits/wastes additional BFS time.

## Current best score/result

```text
Current best reproduced public score: 0.38
Current best notebook: Persistent Memory short/gated-BFS v1
Approximate public rank: ~100 / 1355
```

Pending experiment:

```text
persistent-memory-solver-v1-gated-bfs-v2.ipynb
Scoring status: to be run/submitted
```

## Files / artifacts referenced

Generated notebook artifact:

```text
persistent-memory-solver-v1-gated-bfs-v2.ipynb
```

Session note created:

```text
session_notes/2026-06-22_gated_bfs_038_and_v2_submission_plan.md
```

## Next session plan

1. Upload and run `persistent-memory-solver-v1-gated-bfs-v2.ipynb` on Kaggle.
2. Submit to scoring.
3. Compare v2 public score against current best `0.38`.
4. Keep v1 selected unless v2 scores higher.
5. If v2 improves, consider a small budget sweep around v2.
6. If v2 weakens, return to v1 and test orthogonal changes such as no-TTT, no-CLEAR, no-PER, or hidden-retry ablation.
7. Calibrate offline evaluator with action caps `501`, `1000`, and `2000` before relying on offline score as a proxy.
