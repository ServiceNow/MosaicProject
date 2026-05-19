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

## Local-index coverage

Privacy HopQA dataset generation used every usable file under each task's
`DRxxxx/files/**` directory. Helper local indices must therefore be built from
the same all-files corpus, not only from `config/env.json`. The old
`local_indices_fixeddeps_20260415` artifact misses local gold documents for
some finalized q30 hops and should not be used for new training runs.

Build the all-files local index:

```bash
cd /home/toolkit/my_code/MosaicProject
eai job submit --account snow.research.adea -f privacy_hopqa_build_local_indices_allfiles.yaml
```

Validate local gold-document coverage before launching helpers:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/validate_privacy_hopqa_local_index_coverage.py \
  --local-index-root /mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/local_indices_allfiles_20260519
```

Expected result after the rebuild is `missing_local_hops: 0`.

## Helper jobs

Use one helper per simultaneous trainer unless deliberately stress-testing helper capacity.

Default helper, with RRF ranking for web and local retrieval. The current
high-concurrency helper uses 16 service workers and 32 threadpool tokens.
Do not raise `--workers` blindly: 32 process workers caused helper child
restarts during dense-retrieval startup on May 18, 2026.

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
LOCAL_INDEX_ROOT="$PRIVACY_AGENT_LOCAL_INDEX_ROOT" \
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
eval_every_n_versions=40960
```

Only increase `actor.llm_max_rollouts` after helper health is stable.

## Hop-step additive-cost LR sweep, 50 steps

This is the current non-privacy-reward training sweep.

Reward semantics:

```text
current_hop_correct = 1 if the current hop answer matches, else 0
searched_expected_source = 1 only for hop_plan calls that searched the hop's expected source type, else 0
efficiency_factor = min correct hop LLM calls in the comparison group / this rollout's hop LLM calls

hop_plan base reward = current_hop_correct + 0.25 * searched_expected_source
doc_choose and hop_resolve base reward = current_hop_correct
efficiency bonus = 0.25 * current_hop_correct * efficiency_factor
final hop-step reward = base reward + efficiency bonus
```

Expected source type means the dataset hop type: local hops expect `local_document_search`, and web hops expect `web_search`.
Hop LLM calls include `hop_plan`, `doc_choose`, `doc_read`, and `hop_resolve` calls for that hop. Search/retrieval calls are not counted.
The efficiency bonus is additive and only applies to correct hop attempts. This keeps any correct hop above any incorrect hop while still preferring correct rollouts that answered with fewer LLM calls.
The rollout-level reward is still hop accuracy; the per-call RL advantages use hop/stage buckets from the final hop-step reward.
When `finetune.rl.normalize_step_loss_by_advantage_group=true`, reward values
and leave-one-out baselines are unchanged, then each row's final advantage is
divided by the number of real trainable calls from the same rollout/hop/stage.
This prevents a rollout that used more planning/selection/resolve calls from
getting more policy-gradient signal solely because it made more trainable
decisions. GSPO's existing token weighting is left unchanged.

Common trainer settings:

```text
model=Qwen/Qwen3-4B-Instruct-2507
replicas=4
capture_mode=planning_only
training_reward_mode=hop_step
hop_step_efficiency_metric=llm_calls
reward_mode=hop_accuracy
reader_window_chars=10000
actor.llm_max_rollouts=64
eval_every_n_versions=40960
max_train_steps=50
interrupt_train_steps=50
save_checkpoint_steps=10
wandb_project=privacy_hopqa_q30_hopstep_addcost_20260519
wandb_group=hopstep_addcost10k_50s
```

Trainer command:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_hopstep_addcost10k_lr1e6_50s_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
LOCAL_INDEX_ROOT="$PRIVACY_AGENT_LOCAL_INDEX_ROOT" \
JOB_NAME="$JOB" \
WANDB_PROJECT="privacy_hopqa_q30_hopstep_addcost_20260519" \
WANDB_GROUP="hopstep_addcost10k_50s" \
WANDB_NAME="$JOB" \
REPLICAS=4 \
LEARNING_RATE=1e-6 \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_HOP_STEP_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES privacy_hopqa.reader_window_chars=10000 finetune.max_train_steps=50 finetune.interrupt_train_steps=50 finetune.save_checkpoint_steps=10 actor.llm_max_rollouts=64 eval_every_n_versions=40960" \
./run_privacy_hopqa.sh
```

New runs log helper retry health as:

```text
actor/rollout_retry_count_mean
actor/rollout_retry_any_mean
actor/rollout_retry_backoff_s_mean
```

For the higher-LR comparison, use `1e-5`. A `5e-5` trial on May 18, 2026
became unstable after two optimizer steps (`rl/kl` spiked and JSON/protocol
failures increased sharply), then failed after an actor-side vLLM disconnect.

```bash
LEARNING_RATE=1e-5
JOB="privacy_hopqa_q30_4b_hopstep_addcost10k_lr1e5_50s_${TS}"
```

## Hop-step call-normalized clipping experiment, 50 steps

This run keeps the additive hop-step reward above, but changes the optimizer
signal in two ways:

```text
finetune.rl.normalize_step_loss_by_advantage_group=true
finetune.rl.epsilon_low=0.03
finetune.rl.epsilon_high=0.04
```

Use this when testing whether previous stagnation was caused by overly tight
GSPO clipping and long multi-step traces contributing too many repeated
stage-local updates.

Planned trainer command after the new index is built and the matching helper is running:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-new-index>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_hopstep_callnorm_clip3e2_lr5e7_50s_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
LOCAL_INDEX_ROOT="$PRIVACY_AGENT_LOCAL_INDEX_ROOT" \
JOB_NAME="$JOB" \
WANDB_PROJECT="privacy_hopqa_q30_hopstep_callnorm_20260519" \
WANDB_GROUP="hopstep_callnorm_clip3e2_50s" \
WANDB_NAME="$JOB" \
REPLICAS=4 \
LEARNING_RATE=5e-7 \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_HOP_STEP_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES privacy_hopqa.reader_window_chars=10000 finetune.max_train_steps=50 finetune.interrupt_train_steps=50 finetune.save_checkpoint_steps=10 actor.llm_max_rollouts=64 eval_every_n_versions=40960 finetune.rl.epsilon_low=0.03 finetune.rl.epsilon_high=0.04 finetune.rl.normalize_step_loss_by_advantage_group=true" \
./run_privacy_hopqa.sh
```

## Live jobs: 2026-05-19 hop-step additive-cost LR sweep

```text
wandb_project=privacy_hopqa_q30_hopstep_addcost_20260519
wandb_group=hopstep_addcost10k_50s

lr=1e-6 helper=98da75be-a6bc-4b8b-b5d4-786fe884769d
lr=1e-6 helper_url=http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30ac10k1e6012339:8012
lr=1e-6 trainer=274441ef-dc29-45f6-9801-657453eee3ab
lr=1e-6 output_dir=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_hopstep_addcost10k_lr1e6_50s_20260519012927

lr=1e-5 helper=96dd8088-5adc-406d-90ff-9f1d53e7e097
lr=1e-5 helper_url=http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30ac10k1e5012352:8012
lr=1e-5 trainer=fa9ddbf2-5375-49c5-a302-aca0cee003f9
lr=1e-5 output_dir=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_hopstep_addcost10k_lr1e5_50s_20260519012959

Previous 2026-05-18 multiplicative LLM-call-efficiency jobs were cancelled before launching this sweep.
The first 2026-05-19 additive-cost launch used helper YAMLs that still had 16k web windows; those jobs were cancelled and replaced by the 10k jobs above.
```

Monitor EAI state:

```bash
eai job get \
  98da75be-a6bc-4b8b-b5d4-786fe884769d \
  96dd8088-5adc-406d-90ff-9f1d53e7e097 \
  274441ef-dc29-45f6-9801-657453eee3ab \
  fa9ddbf2-5375-49c5-a302-aca0cee003f9 \
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
