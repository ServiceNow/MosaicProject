# Privacy HopQA Binary Reward SFT, 2026-05-15

Goal: train a cheap Qwen3 4B reward model that predicts the StepFun binary
privacy label from rollout web queries.

## Target

The label is:

```text
StepFun overall_privacy_score_1_to_5 >= 4
```

This treats answer leakage and full-information leakage as positive. Intent-only
leakage, including overall score 3, is negative for this reward-model target.

The SFT format is a one-token binary answer. The assistant target is exactly
`Yes` or `No`; evaluation uses the next-token logits for those two tokens rather
than free-form generation.

## Dataset

Builder:

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 \
  scripts/build_privacy_hopqa_binary_reward_sft_dataset.py
```

Output:

```text
/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515/dataset
```

Inputs:

- Baseline q30 v11 StepFun judgments:
  `/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_mosaic_v11_20260514`
- Privacy-prompt q30 v11 StepFun judgments:
  `/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_privacy_prompt_mosaic_v11_20260514`

Split policy: deterministic `task_id`-grouped train/val/test split. No DR task
appears in more than one reward-model split.

Current counts per prompt variant:

| Split | Rows | Tasks | Positive | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| train | 2,235 | 77 | 310 | 13.9% |
| val | 205 | 9 | 37 | 18.0% |
| test | 266 | 14 | 54 | 20.3% |

Variants:

- `queries_only`: only the ordered web query strings.
- `company_context`: company name plus ordered web query strings.
- `chain_context`: company, hop pattern, global/subquestion text, and ordered web
  query strings. This is useful diagnostically, but is more oracle-like and
  less clean as an external-observer reward model.
- `private_fact_context`: company, ordered web query strings, and the full
  retrieved-document Private QA/local-hop fact list. This is an oracle/contextual
  diagnostic for testing whether explicit private-fact context improves the
  reward model.

## Training

Trainer:

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 \
  scripts/train_privacy_hopqa_binary_reward_model.py ...
```

Implementation details:

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Method: PEFT LoRA SFT, no TRL dependency
- LoRA: `r=16`, `alpha=32`, target attention and MLP projection modules
- Precision: bf16
- Hardware used: one local H100
- Typical full run time: about 3-5 minutes of training, depending on prompt length
- Positive examples were repeated 2x or 4x to reduce class imbalance

## Results

Aggregate viewer:

```text
runtime_analysis/privacy_hopqa_q30_binary_reward_sft/index.html
```

Best practical result so far is `company_context_lora_full_ep2_rep4`:

```text
/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515/models/company_context_lora_full_ep2_rep4
```

Held-out task-grouped test metrics:

| Run | ROC AUC | AP | F1 @ 0.5 | Precision @ 0.5 | Recall @ 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Company context SFT ep2 rep4 | 0.879 | 0.645 | 0.679 | 0.745 | 0.623 |
| Company context SFT ep2 rep2 | 0.869 | 0.652 | 0.650 | 0.679 | 0.623 |
| Chain context SFT ep2 rep2 | 0.874 | 0.704 | 0.625 | 0.686 | 0.574 |
| Queries-only SFT ep2 rep2 | 0.843 | 0.600 | 0.581 | 0.607 | 0.557 |

Using the validation F1-selected threshold slightly improves some runs. For
`company_context_lora_full_ep2_rep4`, the val-selected threshold is `0.562`,
giving test F1 `0.685`, precision `0.760`, recall `0.623`.

The base model without SFT overpredicts leakage heavily, with test F1 around
0.36-0.39 at threshold 0.5 despite high recall. SFT fixes that calibration
failure.

## Recommendation

Use `company_context_lora_full_ep2_rep4` for the first reward-model integration
if the training harness can pass the company name alongside visible web queries.
It is still cheap, avoids oracle chain/subquestion context, and has the best
threshold-0.5 classification behavior in this batch.

Use `queries_only_lora_full_ep2_rep2` if the reward model must strictly match the
adversary input and see web queries only. It is weaker, but still materially
better calibrated than the base model.

The `chain_context` model is useful as an upper-bound diagnostic, but I would not
use it as the default reward signal until we decide that a reward model may see
the private chain/question context.

## TRL follow-up

The current trainer is `scripts/train_privacy_hopqa_binary_reward_model_trl.py`,
using TRL `SFTTrainer` with the same one-token `Yes`/`No` target. Newer runs log
training-time validation loss and one-token reward-classifier validation metrics
to W&B under `alexgurung/privacy_hopqa_binary_reward_sft`.

Aggregate TRL report:

```text
runtime_analysis/privacy_hopqa_q30_binary_reward_trl_report/index.html
```

Current held-out test metrics:

| Run | ROC AUC | F1 @ 0.5 | Precision @ 0.5 | Recall @ 0.5 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Private facts, Qwen3 8B, r16 default, rep4 | 0.945 | 0.761 | 0.729 | 0.796 | Best private-fact run in this batch |
| Private facts, Qwen3 4B, r16 default, rep4 | 0.914 | 0.694 | 0.773 | 0.630 | Best 4B private-fact ranking so far |
| Private facts, Qwen3 4B, r128 all-linear, rep4, lr 5e-5 | 0.901 | 0.704 | 0.704 | 0.704 | Higher rank did not improve ranking |
| Queries only, Qwen3 4B, r16 default, rep4 | 0.859 | 0.622 | 0.638 | 0.607 | Strict deployable input |
| Company context, Qwen3 4B, r16 default, rep4 | 0.857 | 0.644 | 0.667 | 0.623 | Cheap deployable context |

Private facts clearly help relative to queries-only/company-only context, but
they make the reward model oracle-like. For policy training this can still be a
useful shaping signal if the rollout evaluator already has the retrieved local
documents and Private QA facts available.

## Rebuild commands

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 \
  scripts/build_privacy_hopqa_binary_reward_sft_viewer.py
```

```bash
uv run --python 3.12 \
  --with 'scikit-learn>=1.5.0' \
  --with 'matplotlib>=3.8.0' \
  python scripts/build_privacy_hopqa_binary_reward_trl_report.py
```

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 -m py_compile \
  scripts/build_privacy_hopqa_binary_reward_sft_dataset.py \
  scripts/train_privacy_hopqa_binary_reward_model.py \
  scripts/build_privacy_hopqa_binary_reward_sft_viewer.py
```
