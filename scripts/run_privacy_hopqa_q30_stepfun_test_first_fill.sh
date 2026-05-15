#!/usr/bin/env bash
set -euo pipefail

# Fill StepFun privacy judgments in the order we need for analysis:
# first the test split for paper/viewer plots, then the remaining full dataset.

PYTHON_BIN="${PYTHON_BIN:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/toolkit/my_code/MosaicProject}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data_files/privacy_hopqa_final_dataset}"
BASE_ARTIFACT_ROOT="${BASE_ARTIFACT_ROOT:-/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_mosaic_v11_20260514}"
PRIVACY_PROMPT_ARTIFACT_ROOT="${PRIVACY_PROMPT_ARTIFACT_ROOT:-/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_privacy_prompt_mosaic_v11_20260514}"
STEPFUN_BASE_URL="${STEPFUN_BASE_URL:-http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000}"
STEPFUN_CONCURRENCY="${STEPFUN_CONCURRENCY:-4}"
PUSH_VIEWERS="${PUSH_VIEWERS:-0}"

cd "${PROJECT_ROOT}"

MODEL_LABELS=(
  chroma_context_1
  gemma4_31b_it
  gpt_oss_20b
  qwen3_4b_instruct_2507
  qwen3_8b
  step35_flash
)

model_roots=()
for artifact_root in "${BASE_ARTIFACT_ROOT}" "${PRIVACY_PROMPT_ARTIFACT_ROOT}"; do
  for model_label in "${MODEL_LABELS[@]}"; do
    model_roots+=("${artifact_root}/model_runs/${model_label}")
  done
done

run_eval() {
  local model_root="$1"
  local split="${2:-}"
  local tables="${model_root}/tables"

  if [[ ! -f "${tables}/rollout_index.jsonl" ]]; then
    echo "skip missing rollout_index: ${model_root}"
    return 0
  fi

  local split_args=()
  local phase="full"
  if [[ -n "${split}" ]]; then
    split_args=(--split "${split}")
    phase="${split}"
  fi

  echo "StepFun privacy eval phase=${phase} root=${model_root}"
  "${PYTHON_BIN}" scripts/evaluate_privacy_hopqa_leakage_suite_v2.py \
    --artifact-root "${model_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --private-qa-doc-scope read \
    --max-private-questions 200 \
    --continue-on-error \
    "${split_args[@]}" \
    --output "${tables}/privacy_leakage_suite_v2_stepfun_full.jsonl" \
    --errors "${tables}/privacy_leakage_suite_v2_stepfun_full_errors.jsonl" \
    --base-url "${STEPFUN_BASE_URL}" \
    --model step-3.5-flash \
    --api-mode completion_template \
    --concurrency "${STEPFUN_CONCURRENCY}" \
    --timeout 600 \
    --max-tokens 16000
}

build_viewers() {
  "${PYTHON_BIN}" scripts/build_privacy_hopqa_q30_live_analysis_viewer.py \
    --artifact-root "${BASE_ARTIFACT_ROOT}" \
    --output-dir runtime_analysis/privacy_hopqa_q30_mosaic_v11_analysis \
    --baseline-comparison-artifact-root "${BASE_ARTIFACT_ROOT}" \
    --privacy-prompt-comparison-artifact-root "${PRIVACY_PROMPT_ARTIFACT_ROOT}" \
    --latest-output runtime_analysis/latest_privacy_hopqa_q30_mosaic_v11_analysis.html \
    --latest-output runtime_analysis/latest_privacy_hopqa_q30_live_analysis.html \
    --latest-output runtime_analysis/privacy_hopqa_q30_live_analysis_latest.html

  "${PYTHON_BIN}" scripts/build_privacy_hopqa_q30_paper_viewer.py \
    --output-dir runtime_analysis/privacy_hopqa_q30_paper_viewer \
    --latest-output runtime_analysis/latest_privacy_hopqa_q30_paper_viewer.html
}

push_viewers() {
  local phase="$1"
  if [[ "${PUSH_VIEWERS}" != "1" ]]; then
    return 0
  fi

  git add \
    runtime_analysis/privacy_hopqa_q30_mosaic_v11_analysis \
    runtime_analysis/latest_privacy_hopqa_q30_mosaic_v11_analysis.html \
    runtime_analysis/latest_privacy_hopqa_q30_live_analysis.html \
    runtime_analysis/privacy_hopqa_q30_live_analysis_latest.html \
    runtime_analysis/privacy_hopqa_q30_paper_viewer \
    runtime_analysis/latest_privacy_hopqa_q30_paper_viewer.html

  if git diff --cached --quiet; then
    echo "viewer push skipped: no staged changes (${phase})"
    return 0
  fi

  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git commit -m "Update q30 viewers after ${phase} StepFun fill (${timestamp})" && git push || true
}

echo "StepFun test-first fill started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for model_root in "${model_roots[@]}"; do
  run_eval "${model_root}" test
done
build_viewers
push_viewers "test-first"

for model_root in "${model_roots[@]}"; do
  run_eval "${model_root}"
done
build_viewers
push_viewers "full"
echo "StepFun test-first fill finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
