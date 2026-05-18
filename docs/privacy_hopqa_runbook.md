# Privacy HopQA runbook

Canonical command reference for the current q30 Privacy HopQA dataset, helper jobs, and training jobs.

## Current dataset and code

```text
dataset_version=2026-05-13_q30_final_residual_review
full/train/val/test=1001/559/98/344 chains
test_holdout_company=MediConn Solutions
```

Use these branches before launching:

```bash
cd /home/toolkit/my_code/MosaicProject
git checkout main
git pull

cd /home/toolkit/my_code/PipelineRL
git checkout privacy_agent
git pull
```

Preflight:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
python3 scripts/validate_privacy_hopqa_setup.py
```

Shared retrieval artifacts:

```text
/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513
```

## Helper jobs

Use one helper per simultaneous trainer unless deliberately stress-testing helper capacity.

Default helper, with RRF ranking for web and local retrieval:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH

TS=$(date +%Y%m%d%H%M%S)
SHORT=$(date +%H%M%S)
HELPER_JOB="privacy_helper_q30_${TS}"
HELPER_DNS="phq30${SHORT}"

ACCOUNT="$PRIVACY_HOPQA_ACCOUNT" \
JOB_SPEC="$PRIVACY_HOPQA_HELPER_JOB_SPEC" \
JOB_NAME="$HELPER_JOB" \
DNS_NAME="$HELPER_DNS" \
./run_privacy_helper.sh
```

The launcher prints an internal URL like:

```bash
export PRIVACY_AGENT_HELPER_URL="http://<helper-internal-dns>:8012"
```

## Standard trainer template

This template uses the current full q30 train/val split and the 4B instruct model.

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS=4 \
LEARNING_RATE=1e-6 \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES privacy_hopqa.reward_mode=hop_accuracy" \
./run_privacy_hopqa.sh
```

Important defaults in `configs/privacy_hopqa_q30_current.env`:

```text
answer_match_mode=exact_or_f1
helper_timeout_s=180
max_parallel_doc_reads=10
reader_window_chars=10000
actor.llm_max_rollouts=64
```

Only increase `actor.llm_max_rollouts` after helper health is stable.

## Hop-step LR sweep, 50 steps

This is the current non-privacy-reward training sweep.

Reward semantics:

```text
hop_plan reward = (current hop correct + 0.25 * searched expected source type) / hop iterations
doc_choose and hop_resolve reward = current hop correct / hop iterations
```

Expected source type means the dataset hop type: local hops expect `local_document_search`, and web hops expect `web_search`.
The rollout-level reward is still hop accuracy; the per-call RL advantages use hop/stage buckets from the hop-step reward.

Common trainer settings:

```text
model=Qwen/Qwen3-4B-Instruct-2507
replicas=4
capture_mode=planning_only
training_reward_mode=hop_step
reward_mode=hop_accuracy
reader_window_chars=10000
actor.llm_max_rollouts=64
max_train_steps=50
interrupt_train_steps=50
save_checkpoint_steps=10
wandb_project=privacy_hopqa_q30_hopstep_lr_sweep_20260518
wandb_group=hopstep_lr_sweep_50s
```

Trainer command:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_hopstep_lr1e6_50s_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="privacy_hopqa_q30_hopstep_lr_sweep_20260518" \
WANDB_GROUP="hopstep_lr_sweep_50s" \
WANDB_NAME="$JOB" \
REPLICAS=4 \
LEARNING_RATE=1e-6 \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_HOP_STEP_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES privacy_hopqa.reader_window_chars=10000 finetune.max_train_steps=50 finetune.interrupt_train_steps=50 finetune.save_checkpoint_steps=10 actor.llm_max_rollouts=64" \
./run_privacy_hopqa.sh
```

For the higher-LR comparison, change:

```bash
LEARNING_RATE=5e-5
JOB="privacy_hopqa_q30_4b_hopstep_lr5e5_50s_${TS}"
```

## Live jobs: 2026-05-18 hop-step LR sweep

```text
lr=1e-6 helper=0ad9bdb6-872f-4a18-9c62-5421a7e805ba
lr=1e-6 trainer=035433ea-83fb-49c7-9d15-ee4be0bd3b4c
lr=1e-6 output_dir=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_hopstep_lr1e6_50s_20260518172653

lr=5e-5 helper=1b7ec1fb-e6b0-417b-8af6-ca4f3e48b73f
lr=5e-5 trainer=6da937d2-836b-4818-9409-eb63764d5a38
lr=5e-5 output_dir=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_hopstep_lr5e5_50s_20260518172653
```

Monitor EAI state:

```bash
eai job get \
  0ad9bdb6-872f-4a18-9c62-5421a7e805ba \
  1b7ec1fb-e6b0-417b-8af6-ca4f3e48b73f \
  035433ea-83fb-49c7-9d15-ee4be0bd3b4c \
  6da937d2-836b-4818-9409-eb63764d5a38 \
  --fields id,state,name,runs.state,runs.runTime,runs.exitCode \
  --format table
```

Check trainer logs:

```bash
OUT=/mnt/llmd/results/exps/alexg/reason/<JOB_NAME>
rg -n "Completed steps|Published weight update request|checkpoint|Saving" "$OUT/finetune/log/info_0.log"
rg -n "PrivacyHopQAHelperInfrastructureError|retry [2-9]|failed after|Traceback|Exception" \
  "$OUT/actor/info.log" "$OUT/actor/error.log" "$OUT/preprocess/error.log"
```

## Historical runbook

Older launch records and first-web notes remain in:

```text
docs/privacy_hopqa_runbook_2026-05-13.md
```
