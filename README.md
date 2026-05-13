# MosaicProject

Repo for our privacy-aware dataset creation (and eventually agent training)

## Privacy HopQA q30 quick start for Privacy HopQA q30

Use the runbook as the canonical launch path:

[Privacy HopQA runbook, q30, 2026-05-13](./docs/privacy_hopqa_runbook_2026-05-13.md)

Minimal preflight:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
python3 scripts/validate_privacy_hopqa_setup.py
```

The helper-facing retrieval artifacts are staged under:

```bash
/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513
```

## Current Privacy HopQA training launch commands

Current canonical dataset: [Privacy HopQA q30 residual-review dataset](./docs/privacy_hopqa_q30_training_defaults_2026-05-13.md).

The repo-level helper specs now read from the shared q30 artifact bundle,
blacklist long BrowseComp parents, live-filter
BrowseComp hits to at most ten 16k/1k-overlap reader windows, and serve the
fixed local indices from `local_indices_fixeddeps_20260415`:

```bash
cd /home/toolkit/my_code/MosaicProject
ACCOUNT=snow.research.adea JOB_SPEC=privacy_helper_2gpu.yaml ./run_privacy_helper.sh
```

`run_privacy_hopqa.sh` defaults to the q30 full-dataset train/val/test files.
The train/val/test split is 559/98/344 chains, with MediConn Solutions held out
entirely as test.

Current first-web debugging runbook: [Privacy HopQA first-web q12 2026-05-12](./docs/privacy_hopqa_firstweb_q12_2026-05-12.md).
For first-web-only runs, pass first-web dataset paths explicitly instead of using the defaults below.

Previous 2026-05-11 notes are in [Privacy HopQA first-web-anywhere 2026-05-11](./docs/privacy_hopqa_firstweb_anywhere_2026-05-11.md).

Run these from the repo root:

```bash
cd /home/toolkit/my_code/MosaicProject
export PRIVACY_AGENT_HELPER_URL="http://<active-helper-internal-dns>:8012"
```

Normal full-dataset 4B run:

```bash
TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_qwen3_4b_instruct_2507_${TS}"

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_q30_qwen3_4b_instruct_2507 \
WANDB_GROUP=q30_full \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=10 actor.llm_max_rollouts=64" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

Normal full-dataset 8B run:

```bash
TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_qwen3_8b_${TS}"

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_q30_qwen3_8b \
WANDB_GROUP=q30_full \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH=/mnt/llmd/base_models/Qwen3-8B \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=10 actor.llm_max_rollouts=64" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

The normal 4B and 8B jobs train planning calls only (`CAPTURE_MODE=planning_only`), use `actor.llm_max_rollouts=64`, and allow up to 10 concurrent document-reader calls per rollout. Reward/success
metrics are logged by gold pattern and chain length, for example
`reward_by_pattern/LLW_mean` and `reward_by_chain_length/3_mean`.

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
- [Privacy HopQA first-web-anywhere 2026-05-11](./docs/privacy_hopqa_firstweb_anywhere_2026-05-11.md): current first-web-anywhere dataset, helper commands, 4B/8B launch commands, JSON parse errors, infrastructure errors, local-search bootstrap, and reward-signal audit notes.
- [Privacy HopQA spanfix 2026-05-10](./docs/privacy_hopqa_spanfix_2026-05-10.md): current spanfixed dataset version, validation, and first-web rerun settings.
- [Privacy HopQA training](./docs/privacy_hopqa_training.md): PipelineRL launch path, helper setup, dataset split, EAI notes, and monitoring.
- [Privacy HopQA helper fix notes](./docs/privacy_hopqa_helper_fix_2026-05-06.md): helper environment, vLLM, FAISS, and benchmark details.
- [Privacy HopQA initial-state eval](./docs/privacy_hopqa_initial_state_eval.md): baseline evaluation plan and privacy-evaluator workflow.

## Repository Layout

- [`trajectory_generation/`](./trajectory_generation) main package
- [`trajectory_generation/data/`](./trajectory_generation/data) default location for local inputs
- [`trajectory_generation/src/retrieval/`](./trajectory_generation/src/retrieval) local and web retrieval code
- [`scripts/`](./scripts) explicit run wrappers for chain generation, local input prep, and the reusable web retriever
- [`trajectory_generation/src/retrieval/privacy_helper_stack.py`](./trajectory_generation/src/retrieval/privacy_helper_stack.py) colocated vLLM + helper stack for privacy-agent retrieval workloads
- [`docs/`](./docs) overview notes
