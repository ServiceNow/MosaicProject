#!/usr/bin/env bash

TIMESTAMP=$(date +%s)
JOB_NAME=${JOB_NAME:-privacy_helper}
LOCAL_INDEX_ROOT=${LOCAL_INDEX_ROOT:-/mnt/llmd/results/exps/alexg/privacy_agent/local_indices}

CONDA_ENV=${CONDA_ENV:-pipeline-rl}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
SNAPSHOT=${SNAPSHOT:-0}
BID=${BID:-7}

HOME_DATA_NAME=${HOME_DATA_NAME:-snow.home.alex_gurung}
TRANSFORMERS_CACHE_DATA=${TRANSFORMERS_CACHE_DATA:-snow.home.alex_gurung}
DATA_OBJ=${DATA_OBJ:-snow.research.tapes.data}
RESULTS_OBJ=${RESULTS_OBJ:-snow.research.rlar.results}
BASE_MODELS_OBJ=${BASE_MODELS_OBJ:-snow.research.tapes.base_models}

COMMAND="python -m trajectory_generation.src.retrieval.privacy_helper_stack \
--host 0.0.0.0 \
--port 8012 \
--local-index-root ${LOCAL_INDEX_ROOT} \
--browsecomp-index-glob /home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl \
--browsecomp-corpus Tevatron/browsecomp-plus-corpus \
--browsecomp-index-device cpu \
--embedding-model Qwen/Qwen3-Embedding-4B"

make job \
    ENV="${CONDA_ENV}" \
    CONDA_EXE="${CONDA_EXE}" \
    SNAPSHOT="${SNAPSHOT}" \
    BID="${BID}" \
    CPU=32 \
    CPU_MEM=512 \
    GPU=1 \
    GPU_MEM=80 \
    JOB_NAME="${JOB_NAME}_${TIMESTAMP}" \
    HOME_DATA_NAME="${HOME_DATA_NAME}" \
    TRANSFORMERS_CACHE_DATA="${TRANSFORMERS_CACHE_DATA}" \
    DATA_OBJ="${DATA_OBJ}" \
    RESULTS_OBJ="${RESULTS_OBJ}" \
    BASE_MODELS_OBJ="${BASE_MODELS_OBJ}" \
    COMMAND="${COMMAND}"
