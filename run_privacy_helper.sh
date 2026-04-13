#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JOB_SPEC=${JOB_SPEC:-"${SCRIPT_DIR}/privacy_helper_1gpu.yaml"}

if [[ ! -f "${JOB_SPEC}" ]]; then
    echo "Job spec not found: ${JOB_SPEC}" >&2
    exit 1
fi

# The checked-in YAML is the source of truth for the helper job. This wrapper
# only gives us one stable submit command plus a readable dry-run path.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "eai job submit -f ${JOB_SPEC}"
    echo
    cat "${JOB_SPEC}"
    exit 0
fi

exec eai job submit -f "${JOB_SPEC}"
