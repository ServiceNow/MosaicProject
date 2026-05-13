# Privacy HopQA q30 planning-only docread10 runs

Date: 2026-05-13

Purpose: full-dataset q30 training with planning-only RL targets, separate helpers for 4B and 8B, and `max_parallel_doc_reads=10`.

## Dataset

Full dataset:

`/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.jsonl`

Splits:

- train: `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl`
- val: `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_val.jsonl`
- test: `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_test.jsonl`

Split counts: train 559, val 98, test 344. Test is the held-out MediConn Solutions company split.

## Helper jobs

Both helpers use `privacy_helper_2gpu.yaml`, 16 workers, fixed local indices, BrowseComp live filtering, and separate internal DNS names.

4B helper command:

```bash
cd /home/toolkit/my_code/MosaicProject
ACCOUNT=snow.research.adea \
JOB_SPEC=privacy_helper_2gpu.yaml \
JOB_NAME=privacy_helper_q30_4b_20260513031042 \
DNS_NAME=q4 \
./run_privacy_helper.sh
```

4B helper status:

- job id: `e54f044f-bab3-4bd1-b0c2-6192eedfaa1b`
- state at launch check: RUNNING
- internal URL: `http://dns-24e3447c-506e-4b21-92df-156e18db5087-q4:8012`

8B helper command:

```bash
cd /home/toolkit/my_code/MosaicProject
ACCOUNT=snow.research.adea \
JOB_SPEC=privacy_helper_2gpu.yaml \
JOB_NAME=privacy_helper_q30_8b_20260513031042 \
DNS_NAME=q8 \
./run_privacy_helper.sh
```

8B helper status:

- job id: `fdf4d8e9-8b78-43ea-a337-d3dd73a9af21`
- state at launch check: RUNNING
- internal URL: `http://dns-24e3447c-506e-4b21-92df-156e18db5087-q8:8012`

## Final active trainer jobs

Earlier all-call and planning-only/docread6 trainer submissions were cancelled. The active intended runs are the `planning_docread10` runs below.

4B command:

```bash
cd /home/toolkit/my_code/MosaicProject
TS=20260513033026
JOB=privacy_hopqa_q30_full_planning_docread10_qwen3_4b_instruct_2507_${TS}

PRIVACY_AGENT_HELPER_URL=http://dns-24e3447c-506e-4b21-92df-156e18db5087-q4:8012 \
PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME=$JOB \
WANDB_PROJECT=privacy_hopqa_q30_full_planning_docread10_4b_20260513 \
WANDB_GROUP=q30_full_planning_docread10_4b \
WANDB_NAME=$JOB \
REPLICAS=2 \
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=10 actor.llm_max_rollouts=64" \
bash ./run_privacy_hopqa.sh
```

4B trainer status:

- job id: `3f865b0b-e729-4bd5-bf49-25ac970ad843`
- state at launch check: RUNNING
- output dir: `/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_full_planning_docread10_qwen3_4b_instruct_2507_20260513033026`
- W&B project: `privacy_hopqa_q30_full_planning_docread10_4b_20260513`

8B command:

```bash
cd /home/toolkit/my_code/MosaicProject
TS=20260513033026
JOB=privacy_hopqa_q30_full_planning_docread10_qwen3_8b_${TS}

PRIVACY_AGENT_HELPER_URL=http://dns-24e3447c-506e-4b21-92df-156e18db5087-q8:8012 \
PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME=$JOB \
WANDB_PROJECT=privacy_hopqa_q30_full_planning_docread10_8b_20260513 \
WANDB_GROUP=q30_full_planning_docread10_8b \
WANDB_NAME=$JOB \
REPLICAS=2 \
MODEL_PATH=/mnt/llmd/base_models/Qwen3-8B \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=10 actor.llm_max_rollouts=64" \
bash ./run_privacy_hopqa.sh
```

8B trainer status:

- job id: `8b22077b-17eb-4950-83f8-788e3d618221`
- state at launch check: RUNNING
- output dir: `/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_full_planning_docread10_qwen3_8b_20260513033026`
- W&B project: `privacy_hopqa_q30_full_planning_docread10_8b_20260513`

## Monitor

Background monitor:

- PID at creation: `352642`
- log: `/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_training_docread10_monitor_20260513.log`
- interval: 5 minutes

The monitor records job state plus recent notable helper/trainer log lines for:

- 4B helper: `e54f044f-bab3-4bd1-b0c2-6192eedfaa1b`
- 8B helper: `fdf4d8e9-8b78-43ea-a337-d3dd73a9af21`
- 4B trainer: `3f865b0b-e729-4bd5-bf49-25ac970ad843`
- 8B trainer: `8b22077b-17eb-4950-83f8-788e3d618221`

## Important args

- `privacy_hopqa.capture_mode=planning_only`
- `privacy_hopqa.max_parallel_doc_reads=10`
- `actor.llm_max_rollouts=64`
- `privacy_hopqa.helper_timeout_s=180`
- `privacy_hopqa.choose_top_k=3`
- `privacy_hopqa.retrieval_top_k=5`
- `privacy_hopqa.max_candidate_cards=20`
- `privacy_hopqa.read_all_windows_max_count=10`
- `finetune.rl.filter_zero_advantage_groups=true`

## Notes

- The helper jobs intentionally remain running and are shared by the final active trainer jobs.
- The initial helper submissions with long DNS names failed because EAI prefixes internal DNS names and the generated name exceeded 63 characters. The working helpers use short `DNS_NAME=q4` and `DNS_NAME=q8`.
- An initial 8B trainer launch used `Qwen/Qwen3-8B`; it was cancelled and relaunched with `/mnt/llmd/base_models/Qwen3-8B`.
- The all-call trainer runs were cancelled after confirming the intended setup should be `planning_only`.
- The first planning-only trainer submissions with docread default 6 were cancelled and replaced by the active docread10 runs.
