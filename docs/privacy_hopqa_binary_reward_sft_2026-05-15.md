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
| train | 2,158 | 77 | 286 | 13.3% |
| val | 202 | 9 | 35 | 17.3% |
| test | 288 | 14 | 61 | 21.2% |

Variants:

- `queries_only`: only the ordered web query strings.
- `company_context`: company name plus ordered web query strings.
- `chain_context`: company, hop pattern, global/subquestion text, and ordered web
  query strings. This is useful diagnostically, but is more oracle-like and
  less clean as an external-observer reward model.

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

## Rebuild commands

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 \
  scripts/build_privacy_hopqa_binary_reward_sft_viewer.py
```

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 -m py_compile \
  scripts/build_privacy_hopqa_binary_reward_sft_dataset.py \
  scripts/train_privacy_hopqa_binary_reward_model.py \
  scripts/build_privacy_hopqa_binary_reward_sft_viewer.py
```
