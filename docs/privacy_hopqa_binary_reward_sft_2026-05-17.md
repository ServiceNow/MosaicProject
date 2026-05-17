# Privacy HopQA Binary Reward SFT, 2026-05-17

This is the refreshed reward-model training batch after the StepFun privacy
evaluation prompt/runner update.

## Dataset

Builder:

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python \
  scripts/build_privacy_hopqa_binary_reward_sft_dataset.py \
  --baseline-artifact-roots /mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_test3_traininglike_20260515,/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_all1_traininglike_20260515 \
  --privacy-prompt-artifact-roots /mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_test3_traininglike_20260515_privacy_prompt,/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_all1_traininglike_20260515_privacy_prompt \
  --runner-version 19 \
  --prompt-version 14 \
  --output-dir /mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260517/dataset
```

Target: `primary_privacy_binary`, with the old score>=4 fallback only for legacy
rows. The current dataset is filtered to `runner_version=19` and
`prompt_version=14`, so it does not mix the old v11 privacy labels.

Current per-variant counts after including the late StepFun rows:

| Split | Rows | Positive | Positive Rate |
| --- | ---: | ---: | ---: |
| train | 19,390 | 5,952 | 30.7% |
| val | 2,385 | 824 | 34.5% |
| test | 2,747 | 908 | 33.1% |

Task split hygiene check: zero task IDs appear in more than one reward split.

Maximum tokenized full-text lengths with Qwen3-4B tokenizer:

| Variant | Max Tokens | p99 Tokens |
| --- | ---: | ---: |
| queries_only | 915 | 484 |
| company_context | 921 | 491 |
| chain_context | 1,286 | 806 |
| private_fact_context | 4,474 | 2,632 |

## Training Runs

Launcher:

```bash
tmux new-session -d -s reward_20260517 \
  "cd /home/toolkit/my_code/MosaicProject && bash scripts/launch_privacy_hopqa_binary_reward_trl_20260517.sh 2>&1 | tee /mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_trl_20260517/launch.log"
```

Runs can be filtered with `RUN_ONLY`, a comma-separated list of run
names. The current launch splits local/EAI work to avoid duplicate outputs.

Completed runs:

| Run | Variant | Test F1 | Precision | Recall | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `queries_only_4b_r128_rep2_lr5e5_20260517` | queries only | 0.633 | 0.619 | 0.648 | 0.830 |
| `queries_only_8b_r128_rep2_lr3e5_20260517` | queries only | 0.647 | 0.643 | 0.650 | 0.828 |
| `company_context_4b_r128_rep2_lr5e5_20260517` | company context | 0.650 | 0.630 | 0.672 | 0.839 |
| `private_fact_context_4b_r128_rep2_lr5e5_20260517` | private facts | 0.685 | 0.669 | 0.702 | 0.858 |
| `private_fact_context_8b_r128_rep2_lr3e5_20260517` | private facts | 0.684 | 0.651 | 0.720 | 0.858 |

All runs use TRL SFT with one-token `Yes`/`No` labels, positive repeat 2,
LoRA `r=128`, `alpha=32`, all-linear targets, no LoRA dropout, repeated
validation evaluations, and online W&B logging under
`alexgurung/privacy_hopqa_binary_reward_sft`. The launcher passes
`--resume-from-checkpoint auto` so a restarted preemptable job can continue from
the latest trainer checkpoint.

The private-fact context improves both models over query-only input, but the
new v14 labels remain harder than the earlier v11 private-fact training set.
The 4B private-fact run slightly edges the 8B private-fact run on F1 and
precision in this batch.

Report:

```text
runtime_analysis/privacy_hopqa_q30_binary_reward_trl_report_20260517/index.html
runtime_analysis/latest_privacy_hopqa_q30_binary_reward_trl_report.html
```
