#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=$(date +%s)

PIPELINERL_ROOT=${PIPELINERL_ROOT:-/home/toolkit/my_code/PipelineRL}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-/mnt/llmd/results/exps/alexg/reason}
JOB_NAME=${JOB_NAME:-privacy_agent_qwen3_8b}
WANDB_PROJECT=${WANDB_PROJECT:-privacy_agent_qwen3_8b}
LOCAL_INDEX_ROOT=${LOCAL_INDEX_ROOT:-/mnt/llmd/results/exps/alexg/privacy_agent/local_indices}
MODEL_PATH=${MODEL_PATH:-/mnt/llmd/base_models/Qwen3-8B}

PRIVACY_AGENT_HELPER_URL=${PRIVACY_AGENT_HELPER_URL:-}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}

if [[ -z "${PRIVACY_AGENT_HELPER_URL}" ]]; then
    echo "Set PRIVACY_AGENT_HELPER_URL before launching, for example:" >&2
    echo "  export PRIVACY_AGENT_HELPER_URL=http://<helper-host>:8012" >&2
    exit 1
fi

COMMAND="export PYTHONPATH=${PIPELINERL_ROOT}:\$PYTHONPATH && \
python -m pipelinerl.launch \
output_dir=${OUTPUT_DIR_BASE}/${JOB_NAME} \
model_path=${MODEL_PATH} \
wandb.wandb_workspace_root=/mnt/llmd/results/exps/alexg \
wandb.wandb_project_name=${WANDB_PROJECT} \
privacy_agent.local_index_root=${LOCAL_INDEX_ROOT} \
privacy_agent.helper_service_url=${PRIVACY_AGENT_HELPER_URL} \
privacy_agent.embedding_provider=huggingface \
privacy_agent.embedding_model=Qwen/Qwen3-Embedding-4B \
finetune.learning_rate=1e-6 \
finetune.rl.policy_loss=gspo \
finetune.rl.epsilon_low=3e-4 \
finetune.rl.epsilon_high=4e-4 \
finetune.seq_parallel=8 \
finetune.seq_length=16000 \
vllm_config.vllm_kwargs.max_model_len=16000 \
llm.parameters.max_tokens=12000 \
test_llm.parameters.max_tokens=12000 \
world.actor_fraction=4 \
world.preprocessor_fraction=0 \
world.finetune_fraction=4 \
streams=redis \
--config-name privacy_agent \
--config-dir ${PIPELINERL_ROOT}/conf"

make -C "${SCRIPT_DIR}" multi-replica-job \
    REPLICAS=2 \
    ENV="${CONDA_ENV}" \
    CONDA_EXE="${CONDA_EXE}" \
    SNAPSHOT=0 \
    NPROC=8 \
    BID=7 \
    JOB_NAME="${JOB_NAME}_${TIMESTAMP}" \
    HOME_DATA_NAME="snow.home.alex_gurung" \
    TRANSFORMERS_CACHE_DATA="snow.home.alex_gurung" \
    DATA_OBJ="snow.research.tapes.data" \
    RESULTS_OBJ="snow.research.rlar.results" \
    BASE_MODELS_OBJ="snow.research.tapes.base_models" \
    COMMAND="${COMMAND}"
