# MosaicProject

Repo for our privacy-aware dataset creation (and eventually agent training)

## Current Privacy HopQA training launch commands

Run these from the repo root:

```bash
cd /home/toolkit/my_code/MosaicProject
export PRIVACY_AGENT_HELPER_URL="http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-ph-fw4b-20260507201033:8012"
```

First-web-only 4B validation run, matching the low-concurrency validation setup:

```bash
TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_firstweb_qwen3_4b2507_shortanswer_promptfix_${TS}"

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_firstweb_4b_validation \
WANDB_GROUP=firstweb_shortanswer_promptfix \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
DATASET_SPLIT_ROOT=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset \
FINAL_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-07.jsonl \
TRAIN_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-07_train.jsonl \
VAL_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-07_val.jsonl \
TEST_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-07_test.jsonl \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.reward_mode=all_hops_correct privacy_hopqa.iteration_budget_per_hop=3 privacy_hopqa.helper_timeout_s=180 actor.llm_max_rollouts=16" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

Normal full-dataset 4B run:

```bash
TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_final_qwen3_4b_instruct_2507_shortanswer_${TS}"

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_final_qwen3_4b_instruct_2507 \
WANDB_GROUP=shortanswer_promptfix \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
DATASET_SPLIT_ROOT=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset \
FINAL_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-07_answer_variants.jsonl \
TRAIN_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/train.jsonl \
VAL_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/val.jsonl \
TEST_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/test.jsonl \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 actor.llm_max_rollouts=64" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

Normal full-dataset 8B run:

```bash
TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_final_qwen3_8b_shortanswer_${TS}"

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_final_qwen3_8b \
WANDB_GROUP=shortanswer_promptfix \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="Qwen/Qwen3-8B" \
DATASET_SPLIT_ROOT=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset \
FINAL_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-07_answer_variants.jsonl \
TRAIN_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/train.jsonl \
VAL_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/val.jsonl \
TEST_DATASET_PATH=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/test.jsonl \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 actor.llm_max_rollouts=64" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

The first-web command uses `actor.llm_max_rollouts=16` to match the earlier
low-concurrency validation run. The normal 4B and 8B jobs use `64`, matching the
previous full-dataset training runs.

[`trajectory_generation`](./trajectory_generation) shows how we generate multi-hop research questions that rely on both private (enterprise) documents and external web documents (from browsecomp-plus).

## Quick Start

```bash
pip install -e .
```

Place or symlink the required inputs into `trajectory_generation/data/`, or pass explicit paths on the command line:

- `chunks_local.jsonl`
- `chunks_web.jsonl`
- `secret_inventory.jsonl`
- optional `chunks_curated_web.jsonl`
- optional `docs_curated_web.jsonl`

Historical rerun defaults are captured in the repo-level scripts:

```bash
./scripts/run_generation_local_qwen.sh \
  --output-dir ~/trajectory_generation_runs/example_run
```

```bash
./scripts/run_generation_step.sh \
  --base-url http://your-step-endpoint:8000/v1 \
  --output-dir ~/trajectory_generation_runs/example_run
```

To rebuild the local inputs on this machine:

```bash
./scripts/build_local_chunks.sh
./scripts/extract_local_secrets_step.sh
./scripts/filter_local_secrets_step.sh
```

Start the reusable privacy-agent helper stack:

```bash
python -m trajectory_generation.src.retrieval.privacy_helper_stack \
  --host 0.0.0.0 \
  --port 8012 \
  --local-index-root /mnt/llmd/results/exps/alexg/privacy_agent/local_indices_fixeddeps_20260415
```

This service is designed to sit behind an internal DNS name for PipelineRL
actor jobs. The stack starts a colocated vLLM embedding server and a retrieval
API that serves:

- `/embed` for local/overlay embeddings
- `/local/search` for task-local private document retrieval
- `/browsecomp/search` for task-aware BrowseComp-Plus hybrid retrieval, blended
  with optional curated DRBench task URLs when `docs_curated_web.jsonl` and
  `chunks_curated_web.jsonl` are available
- `/health` for startup checks

Note on `local_indices_fixeddeps_20260415`:
- this dated root is intentional, not a random scratch path
- it was rebuilt on 2026-04-15 after installing the missing document-extraction
  dependencies in the `pipeline-rl` env (`python-docx`, `openpyxl`,
  `python-pptx`, `PyPDF2`, `pdfplumber`, `xlrd`)
- older local indices could contain stale placeholder strings like
  `python-docx library not available` instead of real extracted document text
- if you rebuild the local indices again, update the helper spec and this README
  deliberately rather than silently reverting to the older root

Render a generated JSONL file to HTML:

```bash
python -m trajectory_generation.scripts.view_generation \
  ~/trajectory_generation_runs/example_run/chains_filtered.jsonl \
  --open
```

## Docs

Use this as the main index for repo notes:

- [Architecture](./docs/architecture.md): high-level generation and retrieval design.
- [Current System](./docs/system.md): current end-to-end system notes.
- [Examples](./docs/examples.md): generated question examples.
- [Privacy HopQA final dataset](./docs/privacy_hopqa_final_dataset.md): finalized dataset, schema, viewer, answer variants, and split artifacts.
- [Privacy HopQA training](./docs/privacy_hopqa_training.md): PipelineRL launch path, helper setup, dataset split, EAI notes, and monitoring.
- [Privacy HopQA helper fix notes](./docs/privacy_hopqa_helper_fix_2026-05-06.md): helper environment, vLLM, FAISS, and benchmark details.
- [Privacy HopQA initial-state eval](./docs/privacy_hopqa_initial_state_eval.md): baseline evaluation plan and privacy-evaluator workflow.

## Repository Layout

- [`trajectory_generation/`](./trajectory_generation) main package
- [`trajectory_generation/data/`](./trajectory_generation/data) default location for local inputs
- [`trajectory_generation/src/retrieval/`](./trajectory_generation/src/retrieval) local and web retrieval code
- [`scripts/`](./scripts) explicit run wrappers for chain generation, local input prep, and the reusable web retriever
- [`trajectory_generation/src/retrieval/privacy_helper_stack.py`](./trajectory_generation/src/retrieval/privacy_helper_stack.py) colocated vLLM + helper stack for privacy-agent retrieval workloads
- [`docs/`](./docs) advisor-facing overview notes
