# MosaicProject

Repo for our privacy-aware dataset creation (and eventually agent training)

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
