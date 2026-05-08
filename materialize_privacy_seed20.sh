#!/usr/bin/env bash

set -euo pipefail

PIPELINERL_ROOT=${PIPELINERL_ROOT:-/home/toolkit/my_code/PipelineRL}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}

ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-/home/toolkit/nice_code/drbench/chain_review/annotations_2026-04-06.json}
CURATED_PATH=${CURATED_PATH:-/home/toolkit/nice_code/drbench/chain_review/curated_chains.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/llmd/results/exps/alexg/privacy_agent/datasets/seed20.jsonl}
SAMPLE_SIZE=${SAMPLE_SIZE:-20}

"${CONDA_EXE}" run -n "${CONDA_ENV}" --no-capture-output \
    python -m pipelinerl.domains.privacy_agent.materialize_dataset \
    --dataset-name seed20 \
    --annotations-path "${ANNOTATIONS_PATH}" \
    --curated-path "${CURATED_PATH}" \
    --output "${OUTPUT_PATH}" \
    --sample-size "${SAMPLE_SIZE}"
