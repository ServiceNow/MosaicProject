#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JOB_SPEC=${JOB_SPEC:-"${SCRIPT_DIR}/privacy_helper_probe.yaml"}
ACCOUNT=${ACCOUNT:-"snow.research.rlar"}

if [[ ! -f "${JOB_SPEC}" ]]; then
    echo "Job spec not found: ${JOB_SPEC}" >&2
    exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "eai job submit --account ${ACCOUNT} -f ${JOB_SPEC}"
    echo
    cat "${JOB_SPEC}"
    exit 0
fi

exec eai job submit --account "${ACCOUNT}" --format json -f "${JOB_SPEC}"
