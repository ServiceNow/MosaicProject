#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JOB_SPEC=${JOB_SPEC:-"${SCRIPT_DIR}/privacy_helper_1gpu.yaml"}
DEFAULT_SUFFIX=$(date +%s)
JOB_NAME=${JOB_NAME:-"privacy_helper_1gpu_${DEFAULT_SUFFIX}"}
DNS_NAME=${DNS_NAME:-"dns-privacy-helper"}
TMP_JOB_SPEC=""

if [[ ! -f "${JOB_SPEC}" ]]; then
    echo "Job spec not found: ${JOB_SPEC}" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${TMP_JOB_SPEC}" && -f "${TMP_JOB_SPEC}" ]]; then
        rm -f "${TMP_JOB_SPEC}"
    fi
}

trap cleanup EXIT

TMP_JOB_SPEC=$(mktemp)

python - "${JOB_SPEC}" "${TMP_JOB_SPEC}" "${JOB_NAME}" "${DNS_NAME}" <<'PY'
import sys

import yaml

src_path, dst_path, job_name, dns_name = sys.argv[1:5]
with open(src_path) as f:
    spec = yaml.safe_load(f)

spec["name"] = job_name
spec.setdefault("options", {}).setdefault("internal-dns", {})["name"] = dns_name

with open(dst_path, "w") as f:
    yaml.safe_dump(spec, f, sort_keys=False)
PY

helper_url="http://${DNS_NAME}:8012"

# The checked-in YAML is the source of truth for the helper job. This wrapper
# only fills in the runtime name so repeated submits do not collide.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "eai job submit -f ${TMP_JOB_SPEC}"
    echo "helper_url=${helper_url}"
    echo
    cat "${TMP_JOB_SPEC}"
    exit 0
fi

eai job submit -f "${TMP_JOB_SPEC}"
echo
echo "helper_url=${helper_url}"
