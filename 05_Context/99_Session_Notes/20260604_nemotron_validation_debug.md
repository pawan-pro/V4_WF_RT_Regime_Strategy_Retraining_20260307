# 2026-06-04 — NVIDIA Nemotron Kaggle Session Notes

## Context

Competition: NVIDIA Nemotron Model Reasoning Challenge.

Current best confirmed Kaggle public score remains **0.82** from the original V1 fresh-LoRA public notebook replication. The objective for this session was not to run another full SFT experiment, but to understand why recent runs degraded and to create a lightweight adapter validation/debug notebook.

GPU budget noted during the session:

- ~5h30m GPU remaining.
- Refresh expected in ~36h.
- Decision: do not run full training before refresh unless absolutely necessary.

## Score history entering session

| Run | Variant | Public score | Decision |
|---|---:|---:|---|
| V1 | Fresh LoRA, public end-to-end notebook | 0.82 | Keep selected |
| V2 | Exact rerun of public notebook | 0.75 | Reject |
| V6 | Continue public/Tinker adapter, LR=1e-4 | 0.64 | Reject |

Interpretation:

- Exact fresh-LoRA reruns are unstable.
- Continuation from the public/Tinker adapter is harmful under the current Kaggle-native SFT loop.
- Keep 0.82 selected.
- Stop `RESET_WEIGHTS=False` experiments.

## Validation notebook created

Notebook purpose:

- Validate the selected V1 `submission.zip` adapter.
- Confirm adapter packaging and rank compatibility.
- Classify training prompts by category.
- Generate a tiny stratified sample through vLLM to inspect failure modes.

Relevant mounted paths used:

```text
TRAIN_CSV = /kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv
TEST_CSV  = /kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/test.csv
ADAPTER_ZIP = /kaggle/input/notebooks/jatalepawan/end-to-end-finetuning-lb083-v1/submission.zip
ADAPTER_PATH = /kaggle/working/debug_adapter
MODEL_PATH = /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
WHEEL_DIR = /kaggle/input/datasets/llkh0a/rtx-wheels/wheels
```

The V1 `submission.zip` contained:

```text
adapter_model.safetensors
adapter_config.json
```

## Adapter integrity checks

All integrity checks passed.

| Check | Status | Detail |
|---|---:|---|
| `adapter_config.json` exists | PASS | `/kaggle/working/debug_adapter/adapter_config.json` |
| `adapter_model.safetensors` exists | PASS | present |
| LoRA rank <= 32 | PASS | rank=32 |
| LoRA alpha present | PASS | alpha=32 |
| PEFT type | PASS | LORA |
| Target modules present | PASS | `down_proj`, `out_proj`, `k_proj`, `o_proj`, `q_proj`, `up_proj`, `v_proj`, `lm_head`, `in_proj` |
| Target parameters present | PASS | `mlp.experts.gate_up_proj`, `mlp.experts.down_proj` |
| Base model path mentions Nemotron | PASS | metric Nemotron model path |
| Inference mode | PASS | true |

Important adapter config values:

```text
base_model_name_or_path: /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
peft_type: LORA
r: 32
lora_alpha: 32
lora_dropout: 0.0
target_modules: ['down_proj', 'out_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj', 'lm_head', 'in_proj']
target_parameters: ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']
inference_mode: True
```

Conclusion: the selected V1 adapter is structurally valid and Kaggle-compatible.

## Category classifier correction

Initial classifier over-labeled prompts as cipher because all prompts mention Wonderland. Corrected logic avoids using `wonderland` as a category signal and places numeral detection before unit conversion because numeral prompts contain the word `converted`.

Final category counts on `train.csv`:

| Category | Count |
|---|---:|
| bit_manipulation | 1602 |
| gravity | 1597 |
| unit_conversion | 1594 |
| cipher | 1576 |
| numeral | 1576 |
| equation | 1555 |
| unknown | 0 |

Created stratified debug sample:

- 60 rows total.
- 10 rows per category.

Answer-type sanity summary:

| Category | n | Binary | Numeric | Roman-like | Text phrase |
|---|---:|---:|---:|---:|---:|
| bit_manipulation | 10 | 10 | 0 | 0 | 0 |
| cipher | 10 | 0 | 0 | 0 | 10 |
| equation | 10 | 0 | 4 | 0 | 0 |
| gravity | 10 | 0 | 10 | 0 | 0 |
| numeral | 10 | 0 | 0 | 10 | 0 |
| unit_conversion | 10 | 0 | 10 | 0 | 0 |

## Runtime setup issues and fixes

Observed issues:

- Initial validation runtime was CPU-only: `torch=2.10.0+cpu`, `cuda available=False`.
- `vllm` missing.
- `transformers==5.0.0` caused `download_url` import mismatch.

Fixes applied:

- Enabled Kaggle GPU RTX Pro 6000.
- Installed wheel stack from `/kaggle/input/datasets/llkh0a/rtx-wheels/wheels`.
- Confirmed after restart:

```text
torch version: 2.10.0+cu128
cuda available: True
cuda device count: 1
device: NVIDIA RTX PRO 6000 Blackwell Server Edition
vllm: 0.18.0
```

vLLM model load:

- Loaded 13 safetensor shards.
- Load time: ~538 seconds.
- Model memory after load: ~60.64 GiB.
- 12-row generation time: ~163 seconds.

## 12-row validation result

A small balanced sample was created: 2 rows per category, 12 rows total.

Overall results:

```text
Accuracy: 7/12 = 0.5833
Boxed rate: 10/12 = 0.8333
Degenerate repeat rate: 0/12 = 0.0
```

Category summary:

| Category | n | Correct | Accuracy | Boxed rate | Degenerate rate | Avg output len |
|---|---:|---:|---:|---:|---:|---:|
| bit_manipulation | 2 | 0 | 0.0 | 0.0 | 0.0 | 9132.5 |
| cipher | 2 | 1 | 0.5 | 1.0 | 0.0 | 4487.5 |
| equation | 2 | 1 | 0.5 | 1.0 | 0.0 | 6026.0 |
| gravity | 2 | 1 | 0.5 | 1.0 | 0.0 | 2861.0 |
| numeral | 2 | 2 | 1.0 | 1.0 | 0.0 | 1665.5 |
| unit_conversion | 2 | 2 | 1.0 | 1.0 | 0.0 | 2865.5 |

Failure examples:

| id | category | expected | extracted | issue |
|---|---|---|---|---|
| `f8bbabab` | equation | `>^`>` | empty | emitted empty `\boxed{}` for symbolic answer |
| `b67194af` | bit_manipulation | `10100110` | `3` | no boxed answer; long output; fallback extracted last number |
| `eefdd08e` | gravity | `78.64` | `55.017` | coherent but wrong arithmetic/calculation path |
| `550de4d8` | cipher | `king imagines around castle` | `island imagines around castle` | word mapping ambiguity/error |
| `54d2b3b0` | bit_manipulation | `00000000` | `42` | no boxed answer; long output; fallback extracted last number |

## Interpretation

The V1 adapter is valid and serves through vLLM. The 0.82 leaderboard score is not explained by packaging failure.

Main failure clusters:

1. **Bit manipulation**
   - 0/2 in the small validation sample.
   - Both examples lacked `\boxed{}`.
   - Outputs were very long, around 8.7k–9.5k chars.
   - Metric fallback extracted irrelevant numbers (`3`, `42`).
   - Highest-impact improvement area.

2. **Equation symbolic formatting**
   - One symbolic equation example reached final stage but produced empty `\boxed{}`.
   - This suggests final-answer formatting for symbolic answers needs reinforcement.

3. **Gravity arithmetic**
   - One coherent but wrong calculation.
   - Less important than bit/formatting unless validation shows broader easy-category regression.

4. **Cipher disambiguation**
   - One word-level mapping error (`king` vs `island`).
   - Likely token/word-list/mapping ambiguity.

Stable areas:

- Numeral: 2/2 correct.
- Unit conversion: 2/2 correct.
- No degenerate repetition observed.

## Decision

Do not run full training before GPU refresh.

Keep selected:

```text
Best selected submission: V1, public score 0.82
```

Reject:

```text
V2 exact rerun: 0.75
V6 continuation LR1e-4: 0.64
```

Do not run more `RESET_WEIGHTS=False` continuation experiments.

## Next experiment after GPU refresh

Recommended notebook name:

```text
end-to-end-finetuning-lb082-bit-format-reweight-v1.ipynb
```

Kaggle title:

```text
End-to-end finetuning LB082 - bit and format reweight v1
```

Recommended change class:

```python
RESET_WEIGHTS = True
LEARNING_RATE = 2e-4
# Same public notebook structure.
# Modify SFT corpus exposure, not architecture.
```

Specific intervention:

1. Reweight/duplicate bit manipulation traces modestly.
   - Target effective exposure: about 1.25x–1.5x, not 3x.
2. Reinforce final boxed-answer formatting for bit manipulation and symbolic equation outputs.
   - Especially ensure non-numeric symbolic answers are trained inside `\boxed{}`.
3. Keep easy categories present and avoid aggressive oversampling that could damage numeral/unit/gravity/cipher.
4. Keep LoRA rank 32, alpha 32, dropout 0, target modules, and packaging unchanged.

Hypothesis:

- The selected adapter loses avoidable points due to bit manipulation failing to terminate with a boxed answer and equation symbolic answers producing empty boxes.
- Small SFT corpus reweighting plus explicit final-answer formatting should improve public score more than another random rerun or LR-only experiment.

Risks:

- Overweighting bit manipulation may reduce easy-category performance.
- More training may overfit or destabilize formatting.
- Leaderboard variance remains high.

Evaluation plan:

1. Before submission, run the 12-row validation again.
2. If fast enough, extend to 60-row validation: 10 per category.
3. Require:
   - bit boxed rate improves above 0/2 baseline,
   - no regression on numeral/unit conversion,
   - no degeneration,
   - adapter packaging still passes integrity checks.

## Files generated in validation notebook

```text
/kaggle/working/debug_sample.csv
/kaggle/working/debug_eval_12.csv
/kaggle/working/debug_predictions_12.csv
/kaggle/working/debug_summary_12.csv
```

## Bottom line

Today's session converted leaderboard instability into actionable diagnostics. The strongest next move is not another LR tweak or adapter continuation. The next scored experiment should be a fresh-LoRA public-notebook run with modest bit-manipulation and final-boxed-answer formatting reweighting, after the GPU quota refreshes.
