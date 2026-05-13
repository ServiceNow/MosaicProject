#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=$(date +%s)

PIPELINERL_ROOT=${PIPELINERL_ROOT:-/home/toolkit/my_code/PipelineRL}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-/mnt/llmd/results/exps/alexg/reason}
JOB_NAME=${JOB_NAME:-privacy_hopqa_qwen3_8b}
WANDB_PROJECT=${WANDB_PROJECT:-privacy_hopqa_qwen3_8b}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_NAME=${WANDB_NAME:-}
STREAMS_BACKEND=${STREAMS_BACKEND:-files}
FILTER_ZERO_ADVANTAGE_GROUPS=${FILTER_ZERO_ADVANTAGE_GROUPS:-true}
REPLICAS=${REPLICAS:-4}
LOCAL_INDEX_ROOT=${LOCAL_INDEX_ROOT:-/mnt/llmd/results/exps/alexg/privacy_agent/local_indices_fixeddeps_20260415}
MODEL_PATH=${MODEL_PATH:-/mnt/llmd/base_models/Qwen3-8B}
ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-/home/toolkit/nice_code/drbench/chain_review/annotations_2026-04-16.json}
CURATED_PATH=${CURATED_PATH:-/home/toolkit/nice_code/drbench/chain_review/curated_chains.jsonl}
TASK_DATA_ROOT=${TASK_DATA_ROOT:-/home/toolkit/nice_code/drbench/drbench/data/tasks}
DATASET_SPLIT_ROOT=${DATASET_SPLIT_ROOT:-/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset}
FINAL_DATASET_PATH=${FINAL_DATASET_PATH:-${DATASET_SPLIT_ROOT}/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.jsonl}
DATASET_PATH=${DATASET_PATH:-}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-${DATASET_SPLIT_ROOT}/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-${DATASET_SPLIT_ROOT}/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_val.jsonl}
TEST_DATASET_PATH=${TEST_DATASET_PATH:-${DATASET_SPLIT_ROOT}/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_test.jsonl}
LONG_HOLDOUT_DATASET_PATH=${LONG_HOLDOUT_DATASET_PATH:-${DATASET_SPLIT_ROOT}/long_holdout.jsonl}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
CAPTURE_MODE=${CAPTURE_MODE:-planning_only}

PRIVACY_AGENT_HELPER_URL=${PRIVACY_AGENT_HELPER_URL:-}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
EAI_REPLICAS_IN_FILE=${EAI_REPLICAS_IN_FILE:-1}
MULTI_REPLICA_TEMPLATE=${MULTI_REPLICA_TEMPLATE:-${SCRIPT_DIR}/multinode_rl.yaml}
MULTI_REPLICA_FILE=${MULTI_REPLICA_FILE:-}
TMP_MULTI_REPLICA_FILE=""

cleanup() {
    if [[ -n "${TMP_MULTI_REPLICA_FILE}" && -f "${TMP_MULTI_REPLICA_FILE}" ]]; then
        rm -f "${TMP_MULTI_REPLICA_FILE}"
    fi
}

trap cleanup EXIT

if [[ -z "${PRIVACY_AGENT_HELPER_URL}" ]]; then
    echo "Set PRIVACY_AGENT_HELPER_URL before launching, for example:" >&2
    echo "  export PRIVACY_AGENT_HELPER_URL=http://<helper-host>:8012" >&2
    exit 1
fi

if [[ "${EAI_REPLICAS_IN_FILE}" == "1" && -z "${MULTI_REPLICA_FILE}" ]]; then
    if [[ ! -f "${MULTI_REPLICA_TEMPLATE}" ]]; then
        echo "Multi-replica template not found: ${MULTI_REPLICA_TEMPLATE}" >&2
        exit 1
    fi

    DNS_NAME=${DNS_NAME:-$(
        python3 - <<'PY'
import uuid

print(f"phrl{uuid.uuid4().hex[:8]}")
PY
    )}
    TMP_MULTI_REPLICA_FILE=$(mktemp /tmp/privacy_hopqa_multinode.XXXXXX.yaml)

    python3 - "${MULTI_REPLICA_TEMPLATE}" "${TMP_MULTI_REPLICA_FILE}" "${REPLICAS}" "${DNS_NAME}" <<'PY'
import sys

src_path, dst_path, replicas, dns_name = sys.argv[1:5]

with open(src_path) as f:
    lines = f.readlines()

rewritten = []
in_resources = False
saw_replicas = False
in_internal_dns = False
replaced_dns_name = False

for line in lines:
    if line.startswith("resources:"):
        in_resources = True
        saw_replicas = False
        rewritten.append(line)
        continue

    if in_resources and line.startswith("  replicas:"):
        rewritten.append(f"  replicas: {replicas}\n")
        saw_replicas = True
        continue

    if in_resources and line.startswith("options:"):
        if not saw_replicas:
            rewritten.append(f"  replicas: {replicas}\n")
        in_resources = False

    if line.startswith("  internal-dns:"):
        in_internal_dns = True
        rewritten.append(line)
        continue

    if in_internal_dns and line.lstrip().startswith("name:"):
        indent = line[: len(line) - len(line.lstrip())]
        rewritten.append(f'{indent}name: "{dns_name}"\n')
        replaced_dns_name = True
        in_internal_dns = False
        continue

    rewritten.append(line)

if in_resources and not saw_replicas:
    rewritten.append(f"  replicas: {replicas}\n")

if not replaced_dns_name:
    raise SystemExit("Could not find options.internal-dns.name in multi-replica spec")

with open(dst_path, "w") as f:
    f.writelines(rewritten)
PY

    MULTI_REPLICA_FILE="${TMP_MULTI_REPLICA_FILE}"
fi

if [[ -n "${TRAIN_DATASET_PATH}" || -n "${VAL_DATASET_PATH}" ]]; then
    EFFECTIVE_TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-${DATASET_PATH}}
    EFFECTIVE_VAL_DATASET_PATH=${VAL_DATASET_PATH:-${DATASET_PATH}}
    DATASET_OVERRIDES="train_dataset_names=${EFFECTIVE_TRAIN_DATASET_PATH} test_dataset_names=${EFFECTIVE_VAL_DATASET_PATH}"
elif [[ -n "${DATASET_PATH}" ]]; then
    DATASET_OVERRIDES="train_dataset_names=${DATASET_PATH} test_dataset_names=${DATASET_PATH}"
else
    DATASET_OVERRIDES=""
fi

WANDB_OVERRIDES=""
if [[ -n "${WANDB_GROUP}" ]]; then
    WANDB_OVERRIDES="${WANDB_OVERRIDES} wandb.wandb_group=${WANDB_GROUP}"
fi
if [[ -n "${WANDB_NAME}" ]]; then
    WANDB_OVERRIDES="${WANDB_OVERRIDES} wandb.wandb_name=${WANDB_NAME}"
fi

COMMAND="PYTHONPATH=${PIPELINERL_ROOT} ${CONDA_EXE} run -n ${CONDA_ENV} --no-capture-output python -m pipelinerl.launch \
output_dir=${OUTPUT_DIR_BASE}/${JOB_NAME} \
model_path=${MODEL_PATH} \
wandb.wandb_workspace_root=/mnt/llmd/results/exps/alexg \
wandb.wandb_project_name=${WANDB_PROJECT} \
${WANDB_OVERRIDES} \
privacy_hopqa.annotations_path=${ANNOTATIONS_PATH} \
privacy_hopqa.curated_path=${CURATED_PATH} \
privacy_hopqa.final_dataset_path=${FINAL_DATASET_PATH} \
privacy_hopqa.final_train_dataset_path=${TRAIN_DATASET_PATH} \
privacy_hopqa.final_val_dataset_path=${VAL_DATASET_PATH} \
privacy_hopqa.final_test_dataset_path=${TEST_DATASET_PATH} \
privacy_hopqa.task_data_root=${TASK_DATA_ROOT} \
privacy_hopqa.local_index_root=${LOCAL_INDEX_ROOT} \
privacy_hopqa.helper_service_url=${PRIVACY_AGENT_HELPER_URL} \
privacy_hopqa.embedding_provider=huggingface \
privacy_hopqa.embedding_model=Qwen/Qwen3-Embedding-4B \
privacy_hopqa.capture_mode=${CAPTURE_MODE} \
+finetune.rl.filter_zero_advantage_groups=${FILTER_ZERO_ADVANTAGE_GROUPS} \
finetune.learning_rate=1e-6 \
finetune.rl.policy_loss=gspo \
finetune.rl.epsilon_low=3e-3 \
finetune.rl.epsilon_high=4e-3 \
actor.llm_max_rollouts=64 \
actor.shared_memory_entry_size=1000000000 \
preprocess.shared_memory_entry_size=1000000000 \
finetune.seq_parallel=8 \
finetune.seq_length=16384 \
vllm_config.vllm_kwargs.max_model_len=16384 \
llm.parameters.max_tokens=4096 \
test_llm.parameters.max_tokens=4096 \
world.actor_fraction=4 \
world.preprocessor_fraction=0 \
world.finetune_fraction=4 \
streams=${STREAMS_BACKEND} \
${DATASET_OVERRIDES} \
${EXTRA_OVERRIDES} \
--config-name privacy_hopqa \
--config-dir ${PIPELINERL_ROOT}/conf"

make -C "${SCRIPT_DIR}" multi-replica-job \
    REPLICAS="${REPLICAS}" \
    EAI_REPLICAS_IN_FILE="${EAI_REPLICAS_IN_FILE}" \
    MULTI_REPLICA_FILE="${MULTI_REPLICA_FILE}" \
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
