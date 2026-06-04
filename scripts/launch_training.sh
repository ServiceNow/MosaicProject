#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_BUNDLE=${MOSAICLEAKS_DATASET_BUNDLE:-${DATASET_BUNDLE:-}}
TRAIN_ROOT=${TRAIN_ROOT:-}
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-pipelinerl.launch}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-mosaicleaks}
TRAIN_CONFIG_DIR=${TRAIN_CONFIG_DIR:-}
RUN_NAME=${RUN_NAME:-mosaicleaks_train_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-outputs/training}
MODEL_PATH=${MODEL_PATH:-}
RETRIEVAL_HELPER_URL=${MOSAICLEAKS_RETRIEVAL_HELPER_URL:-${RETRIEVAL_HELPER_URL:-}}
REQUIRE_RETRIEVAL_HELPER=${REQUIRE_RETRIEVAL_HELPER:-1}
CAPTURE_MODE=${CAPTURE_MODE:-planning_only}
TRAIN_LEARNING_RATE=${TRAIN_LEARNING_RATE:-1e-6}
TRAIN_POLICY_LOSS=${TRAIN_POLICY_LOSS:-gspo}
TRAIN_SEQ_LENGTH=${TRAIN_SEQ_LENGTH:-16384}
TRAIN_MAX_TOKENS=${TRAIN_MAX_TOKENS:-4096}
TRAIN_MAX_ROLLOUTS=${TRAIN_MAX_ROLLOUTS:-32}
TRAIN_EXTRA_OVERRIDES_FILE=${TRAIN_EXTRA_OVERRIDES_FILE:-}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "$DATASET_BUNDLE" ]]; then
    echo "Set MOSAICLEAKS_DATASET_BUNDLE or DATASET_BUNDLE to a published dataset bundle directory." >&2
    exit 1
fi
if [[ -z "$MODEL_PATH" ]]; then
    echo "Set MODEL_PATH to the base model or checkpoint to train." >&2
    exit 1
fi
if [[ "$REQUIRE_RETRIEVAL_HELPER" == "1" && -z "$RETRIEVAL_HELPER_URL" ]]; then
    echo "Set MOSAICLEAKS_RETRIEVAL_HELPER_URL, or set REQUIRE_RETRIEVAL_HELPER=0 for offline trainers." >&2
    exit 1
fi

DATASET_BUNDLE="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$DATASET_BUNDLE")"
TRAIN_PATH="$DATASET_BUNDLE/train.jsonl"
VAL_PATH="$DATASET_BUNDLE/val.jsonl"
TEST_PATH="$DATASET_BUNDLE/test.jsonl"
MANIFEST_PATH="$DATASET_BUNDLE/manifest.json"

for path in "$TRAIN_PATH" "$VAL_PATH" "$TEST_PATH" "$MANIFEST_PATH"; do
    if [[ ! -f "$path" ]]; then
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
cmd+=("test_dataset_names=${VAL_PATH}")
cmd+=("mosaicleaks.dataset_bundle=${DATASET_BUNDLE}")
cmd+=("mosaicleaks.final_dataset_path=${DATASET_BUNDLE}/dataset.jsonl")
cmd+=("mosaicleaks.final_train_dataset_path=${TRAIN_PATH}")
cmd+=("mosaicleaks.final_val_dataset_path=${VAL_PATH}")
cmd+=("mosaicleaks.final_test_dataset_path=${TEST_PATH}")
cmd+=("mosaicleaks.capture_mode=${CAPTURE_MODE}")
cmd+=("mosaicleaks.learning_rate=${TRAIN_LEARNING_RATE}")
cmd+=("mosaicleaks.policy_loss=${TRAIN_POLICY_LOSS}")
cmd+=("actor.llm_max_rollouts=${TRAIN_MAX_ROLLOUTS}")
cmd+=("finetune.seq_length=${TRAIN_SEQ_LENGTH}")
cmd+=("vllm_config.vllm_kwargs.max_model_len=${TRAIN_SEQ_LENGTH}")
cmd+=("llm.parameters.max_tokens=${TRAIN_MAX_TOKENS}")
cmd+=("test_llm.parameters.max_tokens=${TRAIN_MAX_TOKENS}")

if [[ -n "$RETRIEVAL_HELPER_URL" ]]; then
    cmd+=("mosaicleaks.retrieval_helper_url=${RETRIEVAL_HELPER_URL}")
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
