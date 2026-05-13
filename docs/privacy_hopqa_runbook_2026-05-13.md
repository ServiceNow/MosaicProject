# Privacy HopQA q30 runbook

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

## 7. Monitor jobs

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

## 8. Kill jobs

```bash
eai job kill <trainer-job-id>
eai job kill <helper-job-id>
```

Kill helpers after their paired trainers stop.
