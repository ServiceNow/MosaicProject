#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/toolkit/my_code/MosaicProject}
PYTHON_BIN=${PYTHON_BIN:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}
DATASET_DIR=${DATASET_DIR:-/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260517/dataset}
MANUAL_ROOT=${MANUAL_ROOT:-/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260517}
TRL_ROOT=${TRL_ROOT:-/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_trl_20260517}
REPORT_DIR=${REPORT_DIR:-${REPO_ROOT}/runtime_analysis/privacy_hopqa_q30_binary_reward_trl_report_20260517}
LATEST_REPORT=${LATEST_REPORT:-${REPO_ROOT}/runtime_analysis/latest_privacy_hopqa_q30_binary_reward_trl_report.html}

export PYTHONPATH="${REPO_ROOT}/scripts:${PYTHONPATH:-}"
export WANDB_PROJECT=${WANDB_PROJECT:-privacy_hopqa_binary_reward_sft}
export WANDB_MODE=${WANDB_MODE:-online}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
RUN_ONLY=${RUN_ONLY:-}

mkdir -p "${TRL_ROOT}/models" "${TRL_ROOT}/logs" "${REPORT_DIR}"

should_run() {
  local run_name=$1
  if [[ -z "${RUN_ONLY}" ]]; then
    return 0
  fi
  case ",${RUN_ONLY}," in
    *",${run_name},"*) return 0 ;;
    *) return 1 ;;
  esac
}

rebuild_report() {
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_privacy_hopqa_binary_reward_trl_report.py" \
    --manual-root "${MANUAL_ROOT}" \
    --trl-root "${TRL_ROOT}" \
    --output-dir "${REPORT_DIR}" \
    --latest-output "${LATEST_REPORT}"
}

run_reward_model() {
  local run_name=$1
  local variant=$2
  local base_model=$3
  local max_length=$4
  local batch_size=$5
  local grad_accum=$6
  local eval_batch_size=$7
  local trainer_eval_batch_size=$8
  local learning_rate=$9
  shift 9

  if ! should_run "${run_name}"; then
    echo "[$(date -Is)] skipping ${run_name}"
    return 0
  fi

  echo "[$(date -Is)] starting ${run_name}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_privacy_hopqa_binary_reward_model_trl.py" \
    --dataset-dir "${DATASET_DIR}" \
    --variant "${variant}" \
    --base-model "${base_model}" \
    --output-root "${TRL_ROOT}/models" \
    --output-dir "${TRL_ROOT}/models/${run_name}" \
    --run-name "${run_name}" \
    --epochs 2 \
    --positive-repeat 2 \
    --learning-rate "${learning_rate}" \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --lora-r 128 \
    --lora-alpha 32 \
    --lora-dropout 0 \
    --lora-target-modules all-linear \
    --batch-size "${batch_size}" \
    --grad-accum "${grad_accum}" \
    --eval-batch-size "${eval_batch_size}" \
    --trainer-eval-batch-size "${trainer_eval_batch_size}" \
    --max-length "${max_length}" \
    --trainer-eval-steps 100 \
    --trainer-save-steps 100 \
    --save-total-limit 2 \
    --logging-steps 20 \
    --eval-on-start \
    --classifier-eval-max-examples 1024 \
    --resume-from-checkpoint auto \
    --report-to wandb \
    "$@"
  echo "[$(date -Is)] finished ${run_name}; rebuilding report"
  rebuild_report
}

run_reward_model \
  queries_only_4b_r128_rep2_lr5e5_20260517 \
  queries_only \
  Qwen/Qwen3-4B-Instruct-2507 \
  2048 8 4 32 8 5e-5

run_reward_model \
  company_context_4b_r128_rep2_lr5e5_20260517 \
  company_context \
  Qwen/Qwen3-4B-Instruct-2507 \
  2048 8 4 32 8 5e-5

run_reward_model \
  private_fact_context_4b_r128_rep2_lr5e5_20260517 \
  private_fact_context \
  Qwen/Qwen3-4B-Instruct-2507 \
  5120 2 16 8 2 5e-5 \
  --attn-implementation sdpa \
  --no-liger-kernel

run_reward_model \
  queries_only_8b_r128_rep2_lr3e5_20260517 \
  queries_only \
  /mnt/llmd/base_models/Qwen3-8B \
  2048 4 8 16 4 3e-5

run_reward_model \
  private_fact_context_8b_r128_rep2_lr3e5_20260517 \
  private_fact_context \
  /mnt/llmd/base_models/Qwen3-8B \
  5120 1 32 4 1 3e-5 \
  --attn-implementation sdpa \
  --no-liger-kernel

rebuild_report
echo "[$(date -Is)] all reward-model runs complete"
