# Privacy HopQA q30 runbook

Current canonical runbook: [Privacy HopQA runbook](./privacy_hopqa_runbook.md).

This file is the short command reference for launching Privacy HopQA q30 helper and training jobs.

Current dataset:

```text
2026-05-13_q30_final_residual_review
```

Current split:

```text
full/train/val/test = 1001/559/98/344 chains
```

## 1. Pull code and validate setup

```bash
cd /home/toolkit/my_code/MosaicProject
git checkout main
git pull

cd /home/toolkit/my_code/PipelineRL
git checkout privacy_agent
git pull

cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
python3 scripts/validate_privacy_hopqa_setup.py
```

## 2. Launch a helper

Use one helper per training run when comparing runs.

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH

TS=$(date +%Y%m%d%H%M%S)
HELPER_JOB="privacy_helper_q30_${TS}"
HELPER_DNS="phq30${TS}"

ACCOUNT="$PRIVACY_HOPQA_ACCOUNT" \
JOB_SPEC="$PRIVACY_HOPQA_HELPER_JOB_SPEC" \
JOB_NAME="$HELPER_JOB" \
DNS_NAME="$HELPER_DNS" \
./run_privacy_helper.sh
```

The helper launcher prints an internal URL. Use that URL for the trainer:

```bash
export PRIVACY_AGENT_HELPER_URL="http://<helper-internal-dns>:8012"
```

Optional health check:

```bash
curl "$PRIVACY_AGENT_HELPER_URL/health"
```

## 3. Launch 4B binary-reward training

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_binary_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_BINARY_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

## 4. Launch 4B hop-accuracy-reward training

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_hopacc_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_HOPACC_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

## 5. Launch 8B binary-reward training

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_8b_binary_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_8B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_BINARY_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

## 6. Launch 4B prefix-progress step-reward training

This is the experimental stepwise reward mode. It keeps outcome-based training as the default elsewhere.

Stepwise semantics:

```text
training reward at a captured call = prefix_correct_hops_so_far / total_hops
```

If a 3-hop chain has correctness `correct, correct, wrong`, captured calls on hop 1 get `1/3`, calls on hop 2 get `2/3`, and calls on hop 3 stay at `2/3`.
If the prefix breaks, later correct-looking hops do not increase progress. With `stop_after_incorrect_hop=true`, the training rollout stops after the first resolved wrong hop without injecting a gold answer.
For advantage calculation only, shorter rollouts are padded with their final progress reward so a rollout that reaches farther into the chain is compared fairly at later step indices.

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_prefix_step_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_PREFIX_STEP_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

### Live launch record: 2026-05-13 prefix-progress run

This is the exact paired helper/trainer launch used for the live 4B prefix-progress step-reward run.

Helper:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH

HELPER_JOB="privacy_helper_q30_prefix_step_20260513173046"
HELPER_DNS="phq30ps20260513173046"

ACCOUNT="$PRIVACY_HOPQA_ACCOUNT" \
JOB_SPEC="$PRIVACY_HOPQA_HELPER_JOB_SPEC" \
JOB_NAME="$HELPER_JOB" \
DNS_NAME="$HELPER_DNS" \
./run_privacy_helper.sh
```

Helper result:

```text
job_id=c445dee5-97de-48fe-a207-ca70a7f4afef
internal_url=http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30ps20260513173046:8012
```

Trainer:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30ps20260513173046:8012"

JOB="privacy_hopqa_q30_4b_prefix_step_20260513173122"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_PREFIX_STEP_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

Trainer result:

```text
job_id=3c154fdb-e4e0-401d-a219-5595854d58ec
wandb_project=privacy_hopqa_q30_runs
wandb_group=q30_full
output_dir=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122
```

## 7. Launch 4B hop-step LR sweep, 50 steps

Use this for a short non-privacy-reward comparison of the hop-local step reward.
Launch one helper per trainer.

Reward semantics:

```text
hop_plan reward = (current hop correct + 0.25 * searched expected source type) / hop iterations
doc_choose and hop_resolve reward = current hop correct / hop iterations
```

Expected source type means the dataset hop type: local hops expect `local_document_search`, and web hops expect `web_search`.
The run still logs hop accuracy as the rollout reward, while RL advantages are computed within hop/stage buckets from the hop-step rewards.

Helper for the `1e-6` run:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH

TS=$(date +%Y%m%d%H%M%S)
SHORT=$(date +%H%M%S)
HELPER_JOB="privacy_helper_q30_hopstep_lr1e6_${TS}"
HELPER_DNS="ph1e6${SHORT}"

ACCOUNT="$PRIVACY_HOPQA_ACCOUNT" \
JOB_SPEC="$PRIVACY_HOPQA_HELPER_JOB_SPEC" \
JOB_NAME="$HELPER_JOB" \
DNS_NAME="$HELPER_DNS" \
./run_privacy_helper.sh
```

Helper for the `5e-5` run:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH

TS=$(date +%Y%m%d%H%M%S)
SHORT=$(date +%H%M%S)
HELPER_JOB="privacy_helper_q30_hopstep_lr5e5_${TS}"
HELPER_DNS="ph5e5${SHORT}"

ACCOUNT="$PRIVACY_HOPQA_ACCOUNT" \
JOB_SPEC="$PRIVACY_HOPQA_HELPER_JOB_SPEC" \
JOB_NAME="$HELPER_JOB" \
DNS_NAME="$HELPER_DNS" \
./run_privacy_helper.sh
```

Trainer, `lr=1e-6`:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-lr1e6>:8012"

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

Trainer, `lr=5e-5`:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-lr5e5>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_hopstep_lr5e5_50s_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="privacy_hopqa_q30_hopstep_lr_sweep_20260518" \
WANDB_GROUP="hopstep_lr_sweep_50s" \
WANDB_NAME="$JOB" \
REPLICAS=4 \
LEARNING_RATE=5e-5 \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_HOP_STEP_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES privacy_hopqa.reader_window_chars=10000 finetune.max_train_steps=50 finetune.interrupt_train_steps=50 finetune.save_checkpoint_steps=10 actor.llm_max_rollouts=64" \
./run_privacy_hopqa.sh
```

Only change `actor.llm_max_rollouts=64` to `96` after the paired helper has shown stable health and low timeout/error rates.

### Live launch record: 2026-05-18 hop-step LR sweep

Both runs use:

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

`lr=1e-6` helper:

```text
job_id=0ad9bdb6-872f-4a18-9c62-5421a7e805ba
internal_url=http://dns-24e3447c-506e-4b21-92df-156e18db5087-ph1e6172456:8012
```

`lr=1e-6` trainer:

```text
job_id=035433ea-83fb-49c7-9d15-ee4be0bd3b4c
job_name=privacy_hopqa_q30_4b_hopstep_lr1e6_50s_20260518172653
output_dir=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_hopstep_lr1e6_50s_20260518172653
```

`lr=5e-5` helper:

```text
job_id=1b7ec1fb-e6b0-417b-8af6-ca4f3e48b73f
internal_url=http://dns-24e3447c-506e-4b21-92df-156e18db5087-ph5e5172456:8012
```

`lr=5e-5` trainer:

```text
job_id=6da937d2-836b-4818-9409-eb63764d5a38
job_name=privacy_hopqa_q30_4b_hopstep_lr5e5_50s_20260518172653
output_dir=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_hopstep_lr5e5_50s_20260518172653
```

## 8. Monitor jobs

```bash
eai job get <helper-job-id> <trainer-job-id> \
  --fields id,state,name,runs.state,runs.runTime,runs.exitCode \
  --format table
```

```bash
OUT=/mnt/llmd/results/exps/alexg/reason/<JOB_NAME>
rg -n "Completed steps|rl/reward|Published weight update request" "$OUT/finetune/log/info_0.log"
rg -n "PrivacyHopQAHelperInfrastructureError|retry [2-9]|failed after|Traceback|Exception" "$OUT/actor/info.log" "$OUT/actor/error.log"
```

## 9. Kill jobs

```bash
eai job kill <trainer-job-id>
eai job kill <helper-job-id>
```

Kill helpers after their paired trainers stop.
