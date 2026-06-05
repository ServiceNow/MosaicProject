#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_BUNDLE=${MOSAICLEAKS_DATASET_BUNDLE:-${DATASET_BUNDLE:-}}
TRAIN_ROOT=${TRAIN_ROOT:-}
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-pipelinerl.launch}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-privacy_hopqa}
TRAIN_CONFIG_DIR=${TRAIN_CONFIG_DIR:-}
PIPELINERL_DOMAIN=${PIPELINERL_DOMAIN:-privacy_hopqa}
RUN_NAME=${RUN_NAME:-mosaicleaks_train_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-outputs/training}
MODEL_PATH=${MODEL_PATH:-}
RETRIEVAL_HELPER_URL=${MOSAICLEAKS_RETRIEVAL_HELPER_URL:-${RETRIEVAL_HELPER_URL:-}}
REQUIRE_RETRIEVAL_HELPER=${REQUIRE_RETRIEVAL_HELPER:-1}
CAPTURE_MODE=${CAPTURE_MODE:-planning_only}
TRAIN_REWARD_MODE=${TRAIN_REWARD_MODE:-all_hops_correct}
TRAINING_REWARD_MODE=${TRAINING_REWARD_MODE:-outcome}
ANSWER_MATCH_MODE=${ANSWER_MATCH_MODE:-exact_or_f1}
FILTER_ZERO_ADVANTAGE_GROUPS=${FILTER_ZERO_ADVANTAGE_GROUPS:-true}
HELPER_TIMEOUT_S=${HELPER_TIMEOUT_S:-180}
MAX_PARALLEL_DOC_READS=${MAX_PARALLEL_DOC_READS:-5}
TRAIN_LEARNING_RATE=${TRAIN_LEARNING_RATE:-1e-6}
TRAIN_POLICY_LOSS=${TRAIN_POLICY_LOSS:-gspo}
TRAIN_SEQ_LENGTH=${TRAIN_SEQ_LENGTH:-16384}
TRAIN_MAX_TOKENS=${TRAIN_MAX_TOKENS:-4096}
TRAIN_MAX_ROLLOUTS=${TRAIN_MAX_ROLLOUTS:-32}
PRIVACY_REWARD_SERVICE_URL=${PRIVACY_REWARD_SERVICE_URL:-}
PRIVACY_REWARD_MODEL=${PRIVACY_REWARD_MODEL:-privacy_reward}
TRAIN_EXTRA_OVERRIDES_FILE=${TRAIN_EXTRA_OVERRIDES_FILE:-}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "$DATASET_BUNDLE" ]]; then
    echo "Set MOSAICLEAKS_DATASET_BUNDLE or DATASET_BUNDLE to a published dataset bundle directory." \
        >&2
    exit 1
fi
if [[ -z "$MODEL_PATH" ]]; then
    echo "Set MODEL_PATH to the base model or checkpoint to train." >&2
    exit 1
fi
if [[ "$REQUIRE_RETRIEVAL_HELPER" == "1" && -z "$RETRIEVAL_HELPER_URL" ]]; then
    echo "Set MOSAICLEAKS_RETRIEVAL_HELPER_URL, or set REQUIRE_RETRIEVAL_HELPER=0 for offline trainers." \
        >&2
    exit 1
fi

DATASET_BUNDLE="$(
    python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
        "$DATASET_BUNDLE"
)"
TRAIN_PATH="$DATASET_BUNDLE/train.jsonl"
VAL_PATH="$DATASET_BUNDLE/val.jsonl"
TEST_PATH="$DATASET_BUNDLE/test.jsonl"
MANIFEST_PATH="$DATASET_BUNDLE/manifest.json"

for path in "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$MANIFEST_PATH"; do
    if [[ "$DRY_RUN" != "1" && ! -f "$path" ]]; then
        echo "Dataset bundle is missing required file: $path" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR_BASE"

if [[ -n "$TRAIN_ROOT" ]]; then
    export PYTHONPATH="$TRAIN_ROOT:${PYTHONPATH:-}"
fi

cmd=("${PYTHON:-python3}" -m "$TRAIN_ENTRYPOINT")
cmd+=("output_dir=${OUTPUT_DIR_BASE}/${RUN_NAME}")
cmd+=("model_path=${MODEL_PATH}")
cmd+=("train_dataset_names=${TRAIN_PATH}")

# PipelineRL uses test_dataset_names for routine evaluation. We point it at val;
# the published test split stays held out for final reporting.
cmd+=("test_dataset_names=${VAL_PATH}")

# The PipelineRL domain is still named privacy_hopqa in the training configs
# used for the paper. Keep the prefix configurable for downstream adapters.
cmd+=("${PIPELINERL_DOMAIN}.final_dataset_path=${DATASET_BUNDLE}/dataset.jsonl")
cmd+=("${PIPELINERL_DOMAIN}.final_train_dataset_path=${TRAIN_PATH}")
cmd+=("${PIPELINERL_DOMAIN}.final_val_dataset_path=${VAL_PATH}")
cmd+=("${PIPELINERL_DOMAIN}.final_test_dataset_path=${TEST_PATH}")
cmd+=("${PIPELINERL_DOMAIN}.capture_mode=${CAPTURE_MODE}")
cmd+=("${PIPELINERL_DOMAIN}.reward_mode=${TRAIN_REWARD_MODE}")
cmd+=("${PIPELINERL_DOMAIN}.training_reward_mode=${TRAINING_REWARD_MODE}")
cmd+=("${PIPELINERL_DOMAIN}.answer_match_mode=${ANSWER_MATCH_MODE}")
cmd+=("${PIPELINERL_DOMAIN}.helper_timeout_s=${HELPER_TIMEOUT_S}")
cmd+=("${PIPELINERL_DOMAIN}.max_parallel_doc_reads=${MAX_PARALLEL_DOC_READS}")
cmd+=("${PIPELINERL_DOMAIN}.hop_plan_max_tokens=${TRAIN_MAX_TOKENS}")
cmd+=("${PIPELINERL_DOMAIN}.doc_choose_max_tokens=${TRAIN_MAX_TOKENS}")
cmd+=("${PIPELINERL_DOMAIN}.doc_read_max_tokens=${TRAIN_MAX_TOKENS}")
cmd+=("${PIPELINERL_DOMAIN}.hop_resolve_max_tokens=${TRAIN_MAX_TOKENS}")
cmd+=("+finetune.rl.filter_zero_advantage_groups=${FILTER_ZERO_ADVANTAGE_GROUPS}")
cmd+=("finetune.learning_rate=${TRAIN_LEARNING_RATE}")
cmd+=("finetune.rl.policy_loss=${TRAIN_POLICY_LOSS}")
cmd+=("actor.llm_max_rollouts=${TRAIN_MAX_ROLLOUTS}")
cmd+=("finetune.seq_length=${TRAIN_SEQ_LENGTH}")
cmd+=("vllm_config.vllm_kwargs.max_model_len=${TRAIN_SEQ_LENGTH}")
cmd+=("llm.parameters.max_tokens=${TRAIN_MAX_TOKENS}")
cmd+=("test_llm.parameters.max_tokens=${TRAIN_MAX_TOKENS}")

if [[ -n "$RETRIEVAL_HELPER_URL" ]]; then
    cmd+=("${PIPELINERL_DOMAIN}.helper_service_url=${RETRIEVAL_HELPER_URL}")
fi
if [[ -n "$PRIVACY_REWARD_SERVICE_URL" ]]; then
    cmd+=("${PIPELINERL_DOMAIN}.privacy_reward_service_url=${PRIVACY_REWARD_SERVICE_URL}")
    cmd+=("${PIPELINERL_DOMAIN}.privacy_reward_model=${PRIVACY_REWARD_MODEL}")
fi
if [[ -n "$TRAIN_CONFIG_NAME" ]]; then
    cmd+=("--config-name" "$TRAIN_CONFIG_NAME")
fi
if [[ -n "$TRAIN_CONFIG_DIR" ]]; then
    cmd+=("--config-dir" "$TRAIN_CONFIG_DIR")
fi
if [[ -n "$TRAIN_EXTRA_OVERRIDES_FILE" ]]; then
    while IFS= read -r override; do
        [[ -z "$override" || "$override" =~ ^[[:space:]]*# ]] && continue
        cmd+=("$override")
    done < "$TRAIN_EXTRA_OVERRIDES_FILE"
fi

printf 'Running:'
for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
done
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

exec "${cmd[@]}"
