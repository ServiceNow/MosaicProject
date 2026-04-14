#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=${JOB_NAME:-privacy_agent_qwen3_8b_no_plan_companyctx_v1}
export WANDB_PROJECT=${WANDB_PROJECT:-privacy_agent_qwen3_8b_variants}
export WANDB_GROUP=${WANDB_GROUP:-concise_qa_variants}
export WANDB_NAME=${WANDB_NAME:-no_plan_companyctx_v1}

# Keep the key behavioral differences explicit in the launcher, even if some
# may currently match config defaults.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-privacy_agent.use_research_plan=false}"

exec "${SCRIPT_DIR}/run_privacy_agent.sh"
