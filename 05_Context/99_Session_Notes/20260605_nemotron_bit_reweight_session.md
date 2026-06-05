# 2026-06-05 — NVIDIA Nemotron Bit-Manipulation Reweight Session Notes

## Context

Competition: NVIDIA Nemotron Model Reasoning Challenge.

Session objective: clarify the provenance issue around the 0.86 copied notebook, confirm the correct 0.82 baseline notebook, and define the exact minimal notebook edits for a clean bit-manipulation reweight experiment.

GPU budget at end of session:

- ~4 hours compute remaining.
- Refresh expected in ~13 hours.
- Decision: patch notebook now, avoid further GPU-heavy experimentation until the current run/refresh situation is clear.

## Key correction: 0.86 was not our independent result

The 0.86 result came from Taha's copied Kaggle notebook. Copy & Edit preserved the notebook's private dataset input. Therefore, the 0.86 result should be treated as a reference / diagnostic ceiling, not as our independently reproducible baseline.

Current provenance labels:

| Source | Notebook / data | Score | Status |
|---|---|---:|---|
| Tong Hui Kang public notebook replication | Hui Kang public repository snapshot | 0.82 | Our clean reproducible baseline |
| Exact rerun of 0.82 notebook | Same public notebook | 0.75 | Rejected due to instability |
| Continuation / LR1e-4 experiment | Public/Tinker adapter continuation path | 0.64 | Rejected |
| Taha copied notebook | Taha notebook + copied private dataset input | 0.86 | Reference only, not claimed as independent |

## Correct baseline notebook confirmed

The correct notebook to copy/edit is the Kaggle notebook with:

```text
Public Score: 0.82
Best Score: 0.82 V1
Runtime: 3h 35m 38s, GPU RTX Pro 6000
Version: 1 of 6
```

The uploaded log confirmed it is the clean Hui Kang public snapshot baseline:

```text
Loaded 7830 problem_ids in training order from:
/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/logprobs/index.jsonl

Loaded 7830 examples, 27,842,873 tokens (unmasked=26,568,807)
```

Training config from the same log:

```text
RESET_WEIGHTS=True
IN_PROJ_ONLY=False
SHUFFLE_DATASET=False
NUM_STEPS=1000 clamped to 244
Training: 244 steps, batch_size=32, micro_batch_size=4, lr=0.0002
```

This confirms the base notebook is valid for a clean public-data experiment.

## Core issue: bit manipulation

Question answered: does the 0.82 notebook take Tong's bit-manipulation post into account?

Answer: only indirectly.

The 0.82 notebook trains on pre-tokenized synthetic SFT traces from Hui Kang's public repository snapshot. It does not implement Tong's bit-manipulation solver as executable code. It may contain some bit-style traces, but it does not actively regenerate, verify, or upweight Tong-style bit operator-matching traces.

Observed problem from prior validation:

```text
V1 0.82 adapter, 12-row local validation:
bit_manipulation: 0/2 correct
bit boxed rate: 0/2
outputs: very long, no final boxed binary answer
```

Interpretation:

- Model has seen some relevant traces but underlearns or fails to terminate correctly for bit tasks.
- The first clean improvement should be data exposure, not hyperparameter changes.

## Experiment selected: duplicate existing bit traces

Notebook copy name:

```text
End-to-end finetuning LB082 - bit reweight v1
```

Slug:

```text
end-to-end-finetuning-lb082-bit-reweight-v1
```

Experiment goal:

- Start from clean 0.82 Hui Kang public baseline.
- Preserve public-data provenance.
- Keep all model/training hyperparameters unchanged.
- Duplicate a small deterministic subset of existing bit_manipulation tokenized traces.

## Exact notebook edits made / requested

### Edit 1 — shared config block

In the shared config cell, keep all original values unchanged:

```python
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LEN = 8192
NUM_STEPS = 1000
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
RESET_WEIGHTS = True
IN_PROJ_ONLY = False
MOE_TIE_WEIGHTS = True
ORIGINAL_PROBLEMS_ONLY = False
SHUFFLE_DATASET = False
```

Add immediately below:

```python
# ── Bit-manipulation reweight experiment ─────────────────────────────
# Conservative first experiment:
# duplicate a small number of existing bit_manipulation tokenized traces.
# Do not change model architecture or training hyperparameters.
BIT_REWEIGHT = True
N_EXTRA_BIT_EXAMPLES = 200
```

### Edit 2 — corpus loading block inside `run_training()`

Insert after the `ORIGINAL_PROBLEMS_ONLY` block and before:

```python
total_unmasked = sum(sum(e["weights"]) for e in examples)
total_tokens = sum(len(e["tokens"]) for e in examples)
```

Patch:

```python
    # ── Bit-manipulation reweighting ─────────────────────────────────
    # This notebook uses pre-tokenized traces. We therefore do not edit text.
    # Instead, we duplicate existing bit_manipulation examples by problem_id.
    #
    # Goal:
    # - improve exposure to bit-manipulation traces
    # - preserve Hui Kang public corpus provenance
    # - keep model config and packaging unchanged

    if BIT_REWEIGHT:
        import csv
        import copy

        with open(TRAIN_CSV_PATH) as f:
            train_rows = list(csv.DictReader(f))

        bit_id_set = {
            row["id"]
            for row in train_rows
            if "bit manipulation" in row["prompt"].lower()
            or "8-bit binary" in row["prompt"].lower()
        }

        bit_examples = [e for e in examples if e["problem_id"] in bit_id_set]

        print(
            f"BIT_REWEIGHT=True: found {len(bit_id_set)} bit ids in train.csv; "
            f"{len(bit_examples)} matching examples in loaded corpus"
        )

        if len(bit_examples) == 0:
            raise RuntimeError(
                "BIT_REWEIGHT=True but no bit_manipulation examples matched. "
                "Check TRAIN_CSV_PATH, problem_id format, and corpus ids."
            )

        n_extra = min(N_EXTRA_BIT_EXAMPLES, len(bit_examples))

        # Copy only the first N matching bit examples, preserving deterministic order.
        # Copy lists so there is no accidental shared-state mutation later.
        extra_examples = []
        for e in bit_examples[:n_extra]:
            extra_examples.append(
                {
                    "problem_id": e["problem_id"],
                    "tokens": list(e["tokens"]),
                    "targets": list(e["targets"]),
                    "weights": list(e["weights"]),
                }
            )

        before = len(examples)
        examples.extend(extra_examples)

        print(
            f"BIT_REWEIGHT=True: duplicated {len(extra_examples)} bit examples; "
            f"examples {before} → {len(examples)}"
        )
```

## Expected log after patch

Expected diagnostics:

```text
Loaded 7830 problem_ids in training order from .../huikang/.../index.jsonl
BIT_REWEIGHT=True: found 1602 bit ids in train.csv; XXXX matching examples in loaded corpus
BIT_REWEIGHT=True: duplicated 200 bit examples; examples 7830 → 8030
Loaded 8030 examples, ...
SHUFFLE_DATASET=False: keeping corpus order (8030 examples)
WARNING: NUM_STEPS=1000 exceeds max_steps=250 (8030 examples // 32 batch). Clamping to 250.
Training: 250 steps, batch_size=32, micro_batch_size=4, lr=0.0002
```

Any nonzero matching bit examples is acceptable. Zero means the ID join failed.

## What must not change

Do not change:

```text
RESET_WEIGHTS=True
LEARNING_RATE=2e-4
BATCH_SIZE=32
MICRO_BATCH_SIZE=4
SHUFFLE_DATASET=False
LORA_RANK=32
LORA_ALPHA=32
LORA_DROPOUT=0.0
MOE_TIE_WEIGHTS=True
```

Do not replace Hui Kang data paths with Taha data paths.

Rejected data source for this clean experiment:

```text
/kaggle/input/datasets/tahaalam2009/nemotron-data/...
```

Required clean data source:

```text
/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/...
```

## Evaluation plan

After the patched notebook run completes:

1. Verify output includes:

```text
submission.zip
adapter_config.json
adapter_model.safetensors
```

2. Submit the produced `submission.zip` from the competition submissions page, not by a notebook-page button that reruns the notebook.

3. Keep old 0.82 selected unless the new score beats it.

Decision rule:

| Public score | Action |
|---:|---|
| >0.82 | Select new run; inspect validation next |
| 0.80–0.82 | Neutral; keep old 0.82 selected |
| <0.80 | Reject; bit duplication hurt balance |
| >=0.85 | Strong signal; next experiment can refine bit + equation/symbolic traces |

## Current state at session close

User reported the notebook edits are done.

Compute remaining:

```text
~4 hours GPU compute remaining
refresh in ~13 hours
```

Recommended next action:

- If not already running, consider waiting for refresh before a full run unless there is enough confidence the patched notebook will finish within the remaining quota.
- If the run is already underway, monitor only for the expected bit reweight log lines and final `submission.zip`.

## Bottom line

June 5 shifted from provenance clarification to a clean first data-only experiment. The correct base is the Hui Kang 0.82 V1 notebook. The first patch duplicates 200 existing bit_manipulation tokenized traces while leaving all model settings unchanged. This tests whether the observed bit failure cluster can be improved without relying on Taha's private copied dataset.
