#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=${JOB_NAME:-privacy_agent_qwen3_8b_no_plan_companyctx_planningonly_v1}
export WANDB_PROJECT=${WANDB_PROJECT:-privacy_agent_qwen3_8b_variants}
export WANDB_GROUP=${WANDB_GROUP:-concise_qa_variants}
export WANDB_NAME=${WANDB_NAME:-no_plan_companyctx_planningonly_v1}
export LOCAL_DOCUMENT_SEARCH_MODE=${LOCAL_DOCUMENT_SEARCH_MODE:-retrieve_only}

# Keep the key behavioral differences explicit in the launcher, even if some
# may currently match config defaults. User-provided EXTRA_OVERRIDES come last
# so they can intentionally override these defaults.
BASE_EXTRA_OVERRIDES="privacy_agent.use_research_plan=false privacy_agent.capture_mode=planning_only privacy_agent.local_document_search_mode=${LOCAL_DOCUMENT_SEARCH_MODE}"
export EXTRA_OVERRIDES="${BASE_EXTRA_OVERRIDES}${EXTRA_OVERRIDES:+ ${EXTRA_OVERRIDES}}"

exec "${SCRIPT_DIR}/run_privacy_agent.sh"
